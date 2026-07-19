"""Tests for the LM Studio wrong-model guard.

Regression context (live incident, 2026-07-14): LM Studio 0.4.19 answers
``POST /v1/chat/completions`` naming an UNLOADED model identifier with
whichever model IS loaded — only ``GET /v1/models/{id}`` 404s. A Hermes
session configured for a pinned identifier that a concurrent batch run had
swapped out was silently served by a 16K-context A/B instance and died with
a baffling "context length exceeded" at chat start. Empirically verified in
the same incident: the response body's ``model`` field reports the SERVED
instance id (e.g. ``ab-qwen3-32b``), not an echo of the requested id, and
the OpenAI-compat ``/v1/models`` listing merges downloaded catalog keys with
loaded-instance ids (so only the native ``/api/v1/models`` "loaded_instances"
data distinguishes what a POST will actually route to).

Guard layers under test:
  1. ``check_lmstudio_model_served`` — startup preflight verdicts.
  2. ``WrongModelServedError`` classification — terminal: no retry, no
     compression, no credential rotation, no model fallback hint.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import agent.model_metadata as mm
from agent.error_classifier import FailoverReason, classify_api_error
from agent.errors import WrongModelServedError


def _mock_state(loaded, keys):
    return patch.object(
        mm, "fetch_lmstudio_serving_state", lambda *a, **k: (loaded, keys)
    )


# Serving states replicated from the live estate:
#   CHAMPION — pinned identifier loaded (healthy daily-driver state).
#   INCIDENT — champion unloaded by a battery run; only an A/B instance
#              is resident (the 2026-07-14 failure state, byte-for-byte).
CHAMPION = (["nemotron-3-nano"], ["nvidia/nemotron-3-nano", "qwen3-32b"])
INCIDENT = (["ab-qwen3-32b"], ["nvidia/nemotron-3-nano", "qwen3-32b"])
BASE = "http://127.0.0.1:1234/v1"


def test_loaded_identifier_is_ok():
    with _mock_state(*CHAMPION):
        status, loaded = mm.check_lmstudio_model_served("nemotron-3-nano", BASE)
    assert status == "ok"
    assert loaded == ["nemotron-3-nano"]


def test_incident_state_is_missing():
    """The exact 2026-07-14 state must BLOCK: the requested id basename-matches
    a *downloaded* key, but LM Studio provably routed the request to the
    unrelated loaded model, so a downloaded-only match is a misroute."""
    with _mock_state(*INCIDENT):
        status, loaded = mm.check_lmstudio_model_served("nemotron-3-nano", BASE)
    assert status == "missing"
    assert loaded == ["ab-qwen3-32b"]


def test_exact_downloaded_key_is_jit():
    with _mock_state(*INCIDENT):
        status, _ = mm.check_lmstudio_model_served("nvidia/nemotron-3-nano", BASE)
    assert status == "jit"


def test_provider_prefix_is_stripped():
    with _mock_state(*CHAMPION):
        status, _ = mm.check_lmstudio_model_served("local:nemotron-3-nano", BASE)
    assert status == "ok"


def test_basename_match_on_loaded_instance_is_fuzzy():
    with _mock_state(["nvidia/nemotron-3-nano"], []):
        status, _ = mm.check_lmstudio_model_served("nemotron-3-nano", BASE)
    assert status == "fuzzy"


def test_unreachable_endpoint_is_unknown():
    """Fail OPEN when serving state can't be read — the ordinary connection
    error that follows is already a clear symptom."""
    with patch.object(mm, "fetch_lmstudio_serving_state", lambda *a, **k: None):
        status, loaded = mm.check_lmstudio_model_served("nemotron-3-nano", BASE)
    assert status == "unknown"
    assert loaded == []


def test_wrong_model_served_error_is_terminal():
    """Deterministic misroute: retrying reproduces it, compression can't fix
    it, and a model fallback would substitute yet another unrequested model."""
    classified = classify_api_error(
        WrongModelServedError("LM Studio served 'ab-qwen3-32b' instead of 'nemotron-3-nano'"),
        provider="lmstudio",
        model="nemotron-3-nano",
    )
    assert classified.reason == FailoverReason.model_not_found
    assert classified.retryable is False
    assert classified.should_compress is False
    assert classified.should_rotate_credential is False
    assert classified.should_fallback is False
    assert "ab-qwen3-32b" in classified.message


def test_response_guard_accepts_aliases_for_ordinary_profiles():
    mm.assert_lmstudio_response_model(
        "nemotron-3-nano", "nvidia/nemotron-3-nano"
    )
    mm.assert_lmstudio_response_model("nemotron-3-nano", "")


def test_response_guard_requires_present_exact_identity_when_requested():
    mm.assert_lmstudio_response_model(
        "nemotron-3-nano",
        "nemotron-3-nano",
        require_identity=True,
        exact=True,
    )

    for served in ("", "nvidia/nemotron-3-nano", "ab-qwen3-32b"):
        with pytest.raises(WrongModelServedError):
            mm.assert_lmstudio_response_model(
                "nemotron-3-nano",
                served,
                require_identity=True,
                exact=True,
            )
