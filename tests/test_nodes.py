"""Unit tests for individual agent nodes against a freshly-seeded temp MES."""
from __future__ import annotations

import pytest

from agent import nodes
from agent.graph import build_graph, run_inspection
from agent.llm import StubProvider
from agent.nodes import decide, investigate
from agent.state import AgentDeps
from contracts.models import DetectResult, Disposition
from memory import seed as seed_module


@pytest.fixture
def deps(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)
    return AgentDeps(db_path=db, provider=StubProvider())


def _dr(defective=True, area=0.02, conf=0.95):
    return DetectResult(
        is_defective=defective, confidence=conf, anomaly_score=0.9 if defective else 0.4,
        threshold=0.5, defect_area=area, location="upper-left" if defective else None,
    )


def test_detect_node_passes_through_injected_result(deps):
    dr = _dr()
    out = nodes.make_detect_node(deps)({"part_id": "X", "detect_result": dr})
    assert out["detect_result"] is dr
    assert out["reasoning_trace"]


def test_gather_context_reads_high_rate_for_overdue_machine(deps):
    out = nodes.make_gather_context_node(deps)({"part_id": "SCN-SYSMACH-1"})
    assert out["context"]["machine"]["rate"] >= 0.30  # M2 is the systematic machine


def test_gather_context_reads_low_rate_for_clean_part(deps):
    out = nodes.make_gather_context_node(deps)({"part_id": "SCN-RANDOM-1"})
    assert out["context"]["machine"]["rate"] < 0.30
    assert out["context"]["batch"]["rate"] < 0.30


def test_investigate_flags_systematic_for_high_machine_rate():
    out = investigate({"context": {"machine": {"rate": 0.6}, "batch": {"rate": 0.0}}})
    assert out["investigation"]["fault_pattern"] == "systematic"
    assert "machine" in out["investigation"]["drivers"]


def test_investigate_flags_random_for_low_rates():
    out = investigate({"context": {"machine": {"rate": 0.10}, "batch": {"rate": 0.05}}})
    assert out["investigation"]["fault_pattern"] == "random"
    assert out["investigation"]["drivers"] == []


def test_decide_rejects_systematic_defect():
    out = decide({"detect_result": _dr(area=0.02), "investigation": {"fault_pattern": "systematic", "diagnosis_confidence": 0.95}})
    assert out["decision"].disposition is Disposition.REJECT
    assert out["escalated"] is False


def test_decide_reworks_minor_random_defect():
    out = decide({"detect_result": _dr(area=0.02), "investigation": {"fault_pattern": "random", "diagnosis_confidence": 0.9}})
    assert out["decision"].disposition is Disposition.REWORK


def test_decide_passes_good_part():
    out = decide({"detect_result": _dr(defective=False), "investigation": {"fault_pattern": "random", "diagnosis_confidence": 0.99}})
    assert out["decision"].disposition is Disposition.PASS


def test_repeated_inspection_does_not_self_reinforce(deps):
    # Regression for the feedback loop: the agent's own decision rows (source='agent') must not
    # feed back into the defect-rate, so re-inspecting a clean part as defective can't flip it
    # to systematic / trigger a CAPA + machine flag.
    app = build_graph(deps)
    dr = _dr(defective=True, area=0.02)
    patterns = [run_inspection(app, "SCN-RANDOM-1", detect_result=dr).diagnosis.fault_pattern.value for _ in range(10)]
    assert all(p == "random" for p in patterns), patterns
