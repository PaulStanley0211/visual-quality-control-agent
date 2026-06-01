"""Escalation behaviour at the confidence boundary (strict less-than routing)."""
from __future__ import annotations

import pytest

from agent.graph import build_graph, run_inspection
from agent.llm import StubProvider
from agent.state import AgentDeps
from config import settings
from contracts.models import DetectResult
from memory import seed as seed_module

pytestmark = pytest.mark.escalation


@pytest.fixture
def app(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)
    return build_graph(AgentDeps(db_path=db, provider=StubProvider()))


def _run(app, conf, defective=True, area=0.02, part="SCN-RANDOM-1"):
    dr = DetectResult(
        is_defective=defective, confidence=conf, anomaly_score=0.9 if defective else 0.4,
        threshold=0.5, defect_area=area, location="upper-left" if defective else None,
    )
    return run_inspection(app, part, detect_result=dr)


def test_confidence_just_above_threshold_not_escalated(app):
    assert _run(app, settings.confidence_threshold + 0.01).escalated is False


def test_confidence_exactly_at_threshold_not_escalated(app):
    # should_escalate uses strict <, so exactly at the threshold is acceptable.
    assert _run(app, settings.confidence_threshold).escalated is False


def test_confidence_just_below_threshold_escalated(app):
    assert _run(app, settings.confidence_threshold - 0.01).escalated is True


def test_uncertain_good_is_escalated(app):
    # A low-confidence "good" call is a false-accept risk and must escalate.
    assert _run(app, 0.45, defective=False, area=0.0, part="SCN-GOOD-1").escalated is True


def test_confident_good_not_escalated(app):
    assert _run(app, 0.95, defective=False, area=0.0, part="SCN-GOOD-1").escalated is False


def test_escalated_case_holds_automated_actions(app):
    out = _run(app, settings.confidence_threshold - 0.05)
    assert out.escalated is True
    assert out.actions.ncr is False and out.actions.capa is False and out.actions.machine_flag is False
