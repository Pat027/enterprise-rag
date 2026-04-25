"""Unit tests for the multi-layer safety pipeline."""

from __future__ import annotations

import enterprise_rag.config as config_mod
from enterprise_rag.safety import pipeline
from enterprise_rag.safety.types import SafetyVerdict


class _ToggleSettings:
    def __init__(
        self,
        moderation: bool = False,
        llamaguard: bool = False,
        constitutional: bool = False,
    ):
        self.safety_openai_moderation = moderation
        self.safety_llamaguard = llamaguard
        self.safety_constitutional = constitutional


def _patch_settings(monkeypatch, **flags):
    monkeypatch.setattr(pipeline, "get_settings", lambda: _ToggleSettings(**flags))
    # Ensure the cached singleton in config does not leak across tests
    monkeypatch.setattr(config_mod, "_settings", None, raising=False)


def test_check_input_all_layers_disabled_returns_none(monkeypatch):
    _patch_settings(monkeypatch)  # all False
    assert pipeline.check_input("hello") is None


def test_check_input_returns_blocking_verdict_from_llamaguard(monkeypatch):
    _patch_settings(monkeypatch, llamaguard=True)
    blocked = SafetyVerdict(layer="llamaguard", allowed=False, reason="unsafe")
    monkeypatch.setattr(
        pipeline.llamaguard,
        "check_user_input",
        lambda _t: blocked,
    )
    out = pipeline.check_input("bad query")
    assert out is blocked
    assert out.allowed is False
    assert out.layer == "llamaguard"


def test_check_input_passes_when_layers_allow(monkeypatch):
    _patch_settings(monkeypatch, moderation=True, llamaguard=True)
    monkeypatch.setattr(
        pipeline.openai_moderation,
        "moderate",
        lambda _t: SafetyVerdict(layer="openai_moderation", allowed=True),
    )
    monkeypatch.setattr(
        pipeline.llamaguard,
        "check_user_input",
        lambda _t: SafetyVerdict(layer="llamaguard", allowed=True),
    )
    assert pipeline.check_input("safe query") is None


def test_check_input_fails_open_on_layer_exception(monkeypatch):
    """When a layer raises, the pipeline should swallow it and continue."""
    _patch_settings(monkeypatch, moderation=True, llamaguard=True)

    def _boom(_t):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(pipeline.openai_moderation, "moderate", _boom)
    monkeypatch.setattr(
        pipeline.llamaguard,
        "check_user_input",
        lambda _t: SafetyVerdict(layer="llamaguard", allowed=True),
    )
    # Even though moderation raised, llamaguard allows -> overall allow.
    assert pipeline.check_input("query") is None


def test_check_output_constitutional_blocks(monkeypatch):
    _patch_settings(monkeypatch, llamaguard=True, constitutional=True)
    monkeypatch.setattr(
        pipeline.llamaguard,
        "check_assistant_output",
        lambda _q, _r: SafetyVerdict(layer="llamaguard", allowed=True),
    )
    blocked = SafetyVerdict(layer="constitutional", allowed=False, reason="violates principle")
    monkeypatch.setattr(
        pipeline.constitutional,
        "critique",
        lambda _q, _c, _r: blocked,
    )
    out = pipeline.check_output("q", "ctx", "resp")
    assert out is blocked
