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


import pytest
from PIL import Image

from agent.graph import build_graph, run_inspection
from agent.llm import StubProvider
from contracts.models import DetectResult
from memory import mes, seed as seed_module


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "mes.db"
    seed_module.seed(p, verbose=False)
    return p


def _png(tmp_path):
    p = tmp_path / "part.png"
    Image.new("RGB", (16, 16), (120, 120, 120)).save(p)
    return str(p)


def _clean_detect():
    # Confident GOOD: would NOT escalate on its own.
    return DetectResult(is_defective=False, confidence=0.95, anomaly_score=0.1, threshold=0.5, defect_area=0.0)


def test_ood_image_escalates_confident_good(db, tmp_path):
    ood = DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD: brightness down 3.0σ")
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(ood))
    out = run_inspection(build_graph(deps), "SCN-GOOD-1", image_path=_png(tmp_path), detect_result=_clean_detect())
    assert out.drift is not None and out.drift.is_ood is True
    assert out.escalated is True                                  # drift forced the hold
    assert out.actions.ncr is False and out.actions.capa is False  # actions held
    # The drift score is persisted on the audit row.
    conn = mes.connect(db)
    try:
        row = conn.execute(
            "SELECT drift_score FROM inspections WHERE part_id='SCN-GOOD-1' ORDER BY inspection_id DESC LIMIT 1"
        ).fetchone()
        assert row["drift_score"] == 9.0
    finally:
        conn.close()


def test_in_distribution_image_does_not_escalate(db, tmp_path):
    ok = DriftResult(is_ood=False, drift_score=0.2, threshold=1.0, note="In-distribution")
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(ok))
    out = run_inspection(build_graph(deps), "SCN-GOOD-1", image_path=_png(tmp_path), detect_result=_clean_detect())
    assert out.drift is not None and out.drift.is_ood is False
    assert out.escalated is False


def test_no_image_skips_drift_cleanly(db):
    # Scenario path: detect_result injected, no image => drift not assessed, behavior unchanged.
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(
        DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD")
    ))
    out = run_inspection(build_graph(deps), "SCN-GOOD-1", detect_result=_clean_detect())
    assert out.drift is None
    assert out.escalated is False


def test_defective_part_not_escalated_by_drift(db, tmp_path):
    # A defect sits far from the training-good manifold, so the drift monitor flags it OOD — but drift
    # must NOT suppress a defective part's corrective actions. The defect disposition governs.
    ood = DriftResult(is_ood=True, drift_score=9.0, threshold=1.0, note="OOD: noise up 9.0σ")
    deps = AgentDeps(db_path=db, provider=StubProvider(), _drift_monitor=_StubMonitor(ood))
    dr = DetectResult(
        is_defective=True, confidence=0.95, anomaly_score=0.9, threshold=0.5,
        defect_area=0.08, location="upper-left",
    )
    out = run_inspection(build_graph(deps), "SCN-SYSMACH-1", image_path=_png(tmp_path), detect_result=dr)
    assert out.drift is not None and out.drift.is_ood is True   # drift is still assessed + annotated
    assert out.escalated is False                                # ...but it did not force a hold
    assert out.actions.ncr is True                               # corrective actions proceeded
