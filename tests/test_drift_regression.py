"""Drift-monitor regression: validates drift_metrics.json against budgets.

Skips cleanly if the calibration artifact hasn't been produced yet (run
`uv run python -m drift.reference` then `uv run python -m eval.drift_eval`)."""
from __future__ import annotations

import json

import pytest

from config import settings

pytestmark = pytest.mark.drift

requires_artifact = pytest.mark.skipif(
    not settings.drift_metrics_path.exists(),
    reason="Drift artifact missing; run drift.reference + eval.drift_eval first.",
)


@requires_artifact
def test_separability_auroc_meets_budget():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    assert metrics["category"] == settings.category
    assert metrics["separability_auroc"] >= 0.90, f"AUROC {metrics['separability_auroc']} below 0.90"


@requires_artifact
def test_clean_false_alarm_within_budget():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    far = metrics["holdout"]["false_alarm_rate"]
    assert far <= settings.drift_far_alarm_target + 1e-9, (
        f"Clean false-alarm rate {far} exceeds budget {settings.drift_far_alarm_target}"
    )


@requires_artifact
def test_every_perturbation_detected_above_chance():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    for name, rate in metrics["detection_rate_by_perturbation"].items():
        assert rate > 0.5, f"perturbation '{name}' detected at only {rate:.0%} (<= chance)"
