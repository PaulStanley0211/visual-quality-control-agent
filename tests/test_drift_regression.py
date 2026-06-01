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
    # The threshold is calibrated on a large leakage-free leave-one-out sample; that budget must hold.
    cal_far = metrics["calibration"]["false_alarm_rate"]
    cal_n = metrics["calibration"]["n"]
    assert cal_far <= settings.drift_far_alarm_target + 1.0 / cal_n + 1e-9, (
        f"Calibration false-alarm {cal_far} exceeds budget {settings.drift_far_alarm_target}"
    )
    # The disjoint holdout (small n) is reported honestly; require it within 1/n granularity of target.
    assert metrics["holdout"]["within_granularity"] is True, (
        f"Holdout false-alarm {metrics['holdout']['false_alarm_rate']} exceeds target by more than "
        f"its granularity {metrics['holdout']['far_alarm_granularity']}"
    )


@requires_artifact
def test_every_perturbation_detected_above_chance():
    metrics = json.loads(settings.drift_metrics_path.read_text())
    for name, rate in metrics["detection_rate_by_perturbation"].items():
        assert rate > 0.5, f"perturbation '{name}' detected at only {rate:.0%} (<= chance)"
