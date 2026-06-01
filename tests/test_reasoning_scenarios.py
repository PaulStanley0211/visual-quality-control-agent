"""Agent-reasoning regression: disposition + random/systematic + escalation accuracy
must clear the target on the labeled scenario set (offline stub provider)."""
from __future__ import annotations

import pytest

from config import settings
from eval.agent_eval import evaluate

pytestmark = pytest.mark.reasoning


def test_agent_meets_accuracy_targets():
    metrics = evaluate()
    target = settings.agent_accuracy_target
    assert metrics["disposition_accuracy"] >= target, metrics
    assert metrics["fault_pattern_accuracy"] >= target, metrics
    assert metrics["escalation_accuracy"] >= target, metrics
    assert metrics["actions_accuracy"] >= target, metrics
