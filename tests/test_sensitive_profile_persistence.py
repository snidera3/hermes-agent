"""Sensitive Hermes profiles must be able to run without durable content."""

import copy

import agent.credits_tracker
import cli
import hermes_logging
import hermes_state
from hermes_cli.config import DEFAULT_CONFIG


def test_file_logging_can_be_disabled(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("logging:\n  files_enabled: false\n", encoding="utf-8")
    monkeypatch.setattr(hermes_logging, "get_config_path", lambda: config_path)
    hermes_logging._reset_queued_handlers()
    hermes_logging._logging_initialized = False

    log_dir = hermes_logging.setup_logging(hermes_home=tmp_path)

    assert log_dir == tmp_path / "logs"
    assert hermes_logging.rotating_file_handlers() == []
    assert not (log_dir / "agent.log").exists()
    assert not (log_dir / "errors.log").exists()


def test_cli_skips_session_db_when_persistence_is_disabled(monkeypatch):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["sessions"]["persist"] = False
    config["model"] = {
        "default": "nemotron-3-nano",
        "provider": "lmstudio",
        "base_url": "http://127.0.0.1:1234/v1",
    }
    monkeypatch.setattr(cli, "CLI_CONFIG", config)

    def _unexpected_session_db(*args, **kwargs):
        raise AssertionError("SessionDB must not be constructed")

    monkeypatch.setattr(hermes_state, "SessionDB", _unexpected_session_db)
    instance = cli.HermesCLI(
        model="nemotron-3-nano",
        provider="lmstudio",
        api_key="lm-studio",
        base_url="http://127.0.0.1:1234/v1",
    )

    assert instance._persist_sessions is False
    assert instance._session_db is None

    class _Agent:
        _persist_disabled = False

    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(cli, "AIAgent", lambda **kwargs: _Agent())
    monkeypatch.setattr(instance, "_install_tool_callbacks", lambda: None)
    monkeypatch.setattr(instance, "_ensure_tirith_security", lambda: None)
    monkeypatch.setattr(instance, "_ensure_runtime_credentials", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.wait_for_mcp_discovery", lambda: None
    )
    monkeypatch.setattr(
        agent.credits_tracker, "seed_credits_at_session_start", lambda _agent: None
    )

    assert instance._init_agent() is True
    assert instance._session_db is None
    assert instance.agent._persist_disabled is True


def test_process_registry_skips_durable_restore_when_disabled(monkeypatch):
    import tools.async_delegation as async_delegation
    import tools.process_registry as process_registry

    monkeypatch.setenv("HERMES_NO_DURABLE_STATE", "1")

    def _unexpected_restore(*args, **kwargs):
        raise AssertionError("durable delegation state must not be opened")

    monkeypatch.setattr(
        async_delegation, "restore_undelivered_completions", _unexpected_restore
    )

    registry = process_registry.ProcessRegistry()
    assert registry.completion_queue.empty()
