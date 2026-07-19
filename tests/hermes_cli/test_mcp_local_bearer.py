"""Tests for the narrow local-MCP bearer-file configuration path.

All credentials below are synthetic format-only values. No network connection
or real bridge is started by these tests.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import pytest

from hermes_cli import mcp_local_bearer
from hermes_cli.mcp_local_bearer import (
    LocalBearerConfigurationError,
    resolve_local_bearer_config,
    validate_local_bearer_reference,
)
from hermes_cli.mcp_security import validate_mcp_server_entry


_SYNTHETIC_TOKEN = "A" * 32


def _make_secret_file(
    root: Path,
    *,
    token: str = _SYNTHETIC_TOKEN,
    filename: str = "digitail-mcp.bearer",
) -> Path:
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    path = root / filename
    path.write_text(token + "\n", encoding="ascii")
    path.chmod(0o600)
    return path


def _bridge_config(token_file: Path, **overrides) -> dict:
    config = {
        "url": "http://127.0.0.1:8765/mcp",
        "local_bearer_token_file": str(token_file),
        "headers": {"X-Bridge-Client": "hermes"},
    }
    config.update(overrides)
    return config


def _assert_rejected(config: dict, root: Path, *, server_name: str = "digitail") -> None:
    with pytest.raises(LocalBearerConfigurationError) as exc_info:
        resolve_local_bearer_config(server_name, config, secrets_root=root)
    assert _SYNTHETIC_TOKEN not in str(exc_info.value)
    token_file = config.get("local_bearer_token_file")
    if isinstance(token_file, str):
        assert token_file not in str(exc_info.value)


class TestLocalBearerResolver:
    def test_adds_fresh_header_and_leaves_config_nonsecret(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        config = _bridge_config(token_file)
        original = copy.deepcopy(config)

        resolved = resolve_local_bearer_config("digitail", config, secrets_root=root)

        assert resolved["headers"] == {
            "X-Bridge-Client": "hermes",
            "Authorization": f"Bearer {_SYNTHETIC_TOKEN}",
        }
        assert "local_bearer_token_file" not in resolved
        assert config == original

    def test_reads_the_file_again_for_each_connection_config(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        config = _bridge_config(token_file)

        first = resolve_local_bearer_config("digitail", config, secrets_root=root)
        token_file.write_text("B" * 32, encoding="ascii")
        token_file.chmod(0o600)
        second = resolve_local_bearer_config("digitail", config, secrets_root=root)

        assert first["headers"]["Authorization"] == f"Bearer {_SYNTHETIC_TOKEN}"
        assert second["headers"]["Authorization"] == f"Bearer {'B' * 32}"

    def test_accepts_the_explicit_localhost_alias_for_the_bound_port(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)

        resolved = resolve_local_bearer_config(
            "digitail",
            _bridge_config(token_file, url="http://localhost:8765/mcp"),
            secrets_root=root,
        )

        assert resolved["headers"]["Authorization"] == f"Bearer {_SYNTHETIC_TOKEN}"

    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1:8765/mcp",
            "http://example.invalid/mcp",
            "http://[::1]:8765/mcp",
            "http://user:pass@127.0.0.1:8765/mcp",
            "http://127.0.0.1:8765/not-mcp",
        ],
    )
    def test_rejects_nonapproved_url_shapes(self, tmp_path, url):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)

        _assert_rejected(_bridge_config(token_file, url=url), root)

    def test_rejects_duplicate_authorization_header(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        config = _bridge_config(
            token_file,
            headers={"authorization": "already-configured"},
        )

        _assert_rejected(config, root)

    def test_rejects_world_readable_file_without_leaking_token(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        token_file.chmod(0o644)

        _assert_rejected(_bridge_config(token_file), root)

    def test_rejects_symlink_without_leaking_token(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        outside = tmp_path / "outside.bearer"
        outside.write_text(_SYNTHETIC_TOKEN, encoding="ascii")
        outside.chmod(0o600)
        token_file = root / "digitail-mcp.bearer"
        token_file.symlink_to(outside)

        _assert_rejected(_bridge_config(token_file), root)

    def test_rejects_file_outside_the_secret_root(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        outside = tmp_path / "outside.bearer"
        outside.write_text(_SYNTHETIC_TOKEN, encoding="ascii")
        outside.chmod(0o600)

        _assert_rejected(_bridge_config(outside), root)

    def test_rejects_invalid_token_format(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root, token="not a supported bearer token")

        _assert_rejected(_bridge_config(token_file), root)

    def test_existing_generic_header_configuration_is_unchanged(self):
        config = {
            "url": "https://remote.example/mcp",
            "headers": {"Authorization": "Bearer remote-config"},
        }

        assert resolve_local_bearer_config(None, config) == config

    def test_rejects_mismatched_bridge_name_file_or_port(self, tmp_path):
        root = tmp_path / ".hearth-secrets"
        covet_token_file = _make_secret_file(root, filename="covet-mcp.bearer")

        _assert_rejected(_bridge_config(covet_token_file), root)
        _assert_rejected(
            _bridge_config(covet_token_file),
            root,
            server_name="covet",
        )
        _assert_rejected(
            _bridge_config(_make_secret_file(root)),
            root,
            server_name="not-a-bridge",
        )


class TestLocalBearerIntegration:
    def test_static_validation_blocks_an_invalid_local_bearer_shape(self):
        config = {
            "url": "https://remote.example/mcp",
            "local_bearer_token_file": "~/.hearth-secrets/bridge.bearer",
        }

        assert validate_local_bearer_reference("digitail", config) is False
        issues = validate_mcp_server_entry("digitail", config)
        assert len(issues) == 1
        assert "local bearer" in issues[0]

    def test_static_validation_rejects_a_file_outside_the_secret_root(self):
        config = {
            "url": "http://127.0.0.1:8765/mcp",
            "local_bearer_token_file": "/tmp/not-a-bridge-token",
        }

        assert validate_local_bearer_reference("digitail", config) is False

    def test_runtime_loader_resolves_the_header_at_load_time(self, tmp_path, monkeypatch):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        config = _bridge_config(token_file)
        monkeypatch.setattr(
            mcp_local_bearer,
            "_default_secrets_root",
            lambda: root,
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"mcp_servers": {"digitail": config}},
        )

        from tools.mcp_tool import _load_mcp_config

        loaded = _load_mcp_config()

        assert loaded["digitail"]["headers"]["Authorization"] == f"Bearer {_SYNTHETIC_TOKEN}"
        assert "local_bearer_token_file" not in loaded["digitail"]
        assert config["local_bearer_token_file"] == str(token_file)

    def test_cli_probe_resolver_uses_the_same_fresh_file_path(self, tmp_path, monkeypatch):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        monkeypatch.setattr(
            mcp_local_bearer,
            "_default_secrets_root",
            lambda: root,
        )

        from hermes_cli.mcp_config import _resolve_mcp_server_config

        resolved = _resolve_mcp_server_config(
            _bridge_config(token_file), server_name="digitail"
        )

        assert resolved["headers"]["Authorization"] == f"Bearer {_SYNTHETIC_TOKEN}"
        assert "local_bearer_token_file" not in resolved

    def test_runtime_loader_skips_bad_file_without_logging_its_value(
        self, tmp_path, monkeypatch, caplog
    ):
        root = tmp_path / ".hearth-secrets"
        token_file = _make_secret_file(root)
        token_file.chmod(0o644)
        monkeypatch.setattr(
            mcp_local_bearer,
            "_default_secrets_root",
            lambda: root,
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"mcp_servers": {"digitail": _bridge_config(token_file)}},
        )

        from tools.mcp_tool import _load_mcp_config

        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            assert _load_mcp_config() == {}

        assert _SYNTHETIC_TOKEN not in caplog.text


def test_owner_only_file_is_not_rewritten_by_resolution(tmp_path):
    root = tmp_path / ".hearth-secrets"
    token_file = _make_secret_file(root)
    before = os.stat(token_file).st_mode & 0o777

    resolve_local_bearer_config("digitail", _bridge_config(token_file), secrets_root=root)

    assert os.stat(token_file).st_mode & 0o777 == before == 0o600
