"""Resolve narrowly scoped local MCP bearer credentials.

This module deliberately supports one configuration shape only:

``local_bearer_token_file: ~/.hearth-secrets/<approved-bridge>.bearer``

The configuration stores a non-secret filename, never the bearer value. At
connection time the value is read from a regular, owner-only file and is added
only to its fixed loopback HTTP MCP configuration. This is intentionally
narrower than generic HTTP headers: it is for the three approved local bridge
admission controls, not a replacement for Hermes' existing remote-MCP
authentication options.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, NoReturn
from urllib.parse import urlsplit


LOCAL_BEARER_TOKEN_FILE_KEY = "local_bearer_token_file"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
_MAX_TOKEN_FILE_BYTES = 1024
_LOCAL_BRIDGE_POLICIES = {
    "digitail": ("digitail-mcp.bearer", 8765),
    "covet": ("covet-mcp.bearer", 8766),
    "idexx": ("idexx-mcp.bearer", 8767),
}


class LocalBearerConfigurationError(ValueError):
    """Raised without sensitive filesystem or credential detail."""

    def __init__(self) -> None:
        super().__init__("local MCP bearer configuration rejected")


def _default_secrets_root() -> Path:
    """Return the only permitted credential directory for this feature."""
    return Path.home() / ".hearth-secrets"


def _reject() -> NoReturn:
    raise LocalBearerConfigurationError()


def _is_expected_bridge_url(value: Any, *, port: int) -> bool:
    """Return whether *value* is the fixed streamable-MCP bridge endpoint."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        parsed_port = parsed.port
    except (TypeError, ValueError):
        return False
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        return False
    # A bridge token must not be released to a different local listener or a
    # URL credential channel.  The static name/file/port tuple is intentional.
    return (
        parsed_port == port
        and parsed.path == "/mcp"
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )


def validate_local_bearer_reference(
    server_name: str | None,
    config: Mapping[str, Any],
    *,
    secrets_root: Path | None = None,
) -> bool:
    """Validate the non-secret portion of a local bearer configuration.

    This intentionally does not read the credential file. Save-time and
    static configuration validation can use it without touching a secret; the
    resolver below performs the filesystem checks immediately before use.
    """
    if LOCAL_BEARER_TOKEN_FILE_KEY not in config:
        return True

    policy = _LOCAL_BRIDGE_POLICIES.get(server_name or "")
    if policy is None:
        return False
    expected_filename, expected_port = policy
    token_file = config.get(LOCAL_BEARER_TOKEN_FILE_KEY)
    if not isinstance(token_file, str) or not token_file.strip():
        return False
    root = secrets_root if secrets_root is not None else _default_secrets_root()
    try:
        candidate = Path(token_file).expanduser()
    except RuntimeError:
        return False
    if (
        not candidate.is_absolute()
        or candidate.parent != root
        or candidate.name != expected_filename
    ):
        return False
    if not _is_expected_bridge_url(config.get("url"), port=expected_port):
        return False

    headers = config.get("headers")
    if headers is not None and not isinstance(headers, Mapping):
        return False
    if isinstance(headers, Mapping):
        # Avoid an ambiguous or accidentally overridden authentication path.
        if any(str(key).lower() == "authorization" for key in headers):
            return False
    return True


def _read_owner_only_token(token_file: str, *, secrets_root: Path) -> str:
    """Read one direct-child credential file without following a symlink."""
    try:
        root_stat = secrets_root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            _reject()

        candidate = Path(token_file).expanduser()
        # Restrict to a direct child. Besides being easy to audit, this rejects
        # ``..`` traversal and any intermediate symlink component.
        if not candidate.is_absolute() or candidate.parent != secrets_root:
            _reject()

        before = candidate.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            _reject()
        if before.st_mode & 0o077:
            _reject()
        if before.st_size > _MAX_TOKEN_FILE_BYTES:
            _reject()

        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            # The safety contract is stronger than a best-effort fallback.
            _reject()
        flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(str(candidate), flags)
    except (OSError, RuntimeError):
        _reject()

    try:
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_mode & 0o077:
                _reject()
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                _reject()
            if opened.st_size > _MAX_TOKEN_FILE_BYTES:
                _reject()
            with os.fdopen(descriptor, "r", encoding="ascii") as handle:
                descriptor = -1
                token = handle.read().strip()
        except (OSError, UnicodeError):
            _reject()
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if not _TOKEN_PATTERN.fullmatch(token):
        _reject()
    return token


def resolve_local_bearer_config(
    server_name: str | None,
    config: Mapping[str, Any],
    *,
    secrets_root: Path | None = None,
) -> dict[str, Any]:
    """Return a connection-ready copy with a local bearer header attached.

    The caller receives no ``local_bearer_token_file`` field, preventing a
    downstream transport implementation from treating it as a user-supplied
    arbitrary header. Existing generic-header configurations are unchanged.
    """
    resolved = dict(config)
    if LOCAL_BEARER_TOKEN_FILE_KEY not in resolved:
        return resolved
    root = secrets_root if secrets_root is not None else _default_secrets_root()
    if not validate_local_bearer_reference(
        server_name, resolved, secrets_root=root
    ):
        _reject()

    token_file = resolved.pop(LOCAL_BEARER_TOKEN_FILE_KEY)
    token = _read_owner_only_token(token_file, secrets_root=root)

    headers = dict(resolved.get("headers") or {})
    headers["Authorization"] = f"Bearer {token}"
    resolved["headers"] = headers
    return resolved
