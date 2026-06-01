"""Drift escalation wiring + dependency injection."""
from __future__ import annotations

from agent.state import AgentDeps
from contracts.models import DriftResult


class _StubMonitor:
    def __init__(self, result):
        self._result = result

    def score(self, image):
        return self._result


def test_get_drift_monitor_returns_injected():
    stub = _StubMonitor(DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD: x up 9.0σ"))
    deps = AgentDeps(_drift_monitor=stub)
    assert deps.get_drift_monitor() is stub


def test_get_drift_monitor_none_when_artifact_absent(tmp_path, monkeypatch):
    # No reference artifact in a temp dir => get_drift_monitor must degrade to None, not raise.
    from config import settings

    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    deps = AgentDeps()
    assert deps.get_drift_monitor() is None
