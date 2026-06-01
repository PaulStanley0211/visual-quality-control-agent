"""Perception regression suite.

Validates the trained PatchCore artifact: known-good images must read normal,
known-defect images must read defective, and image-level AUROC must clear the
budget. Skips cleanly if the model/metrics haven't been produced yet (run
`uv run python -m perception.train` then `uv run python -m eval.perception_eval`).
"""
from __future__ import annotations

import json

import pytest

from config import settings

pytestmark = pytest.mark.regression


def _artifacts_ready() -> bool:
    try:
        from perception.detector import find_exported_model

        find_exported_model()
    except FileNotFoundError:
        return False
    return settings.metrics_path.exists()


requires_artifacts = pytest.mark.skipif(
    not _artifacts_ready(),
    reason="Perception artifacts missing; run perception.train + eval.perception_eval first.",
)


@pytest.fixture(scope="module")
def detector():
    from perception.detector import Detector

    return Detector()


def _test_dir(defect: str):
    return settings.data_root / "MVTecAD" / settings.category / "test" / defect


@requires_artifacts
@pytest.mark.parametrize("defect", ["good"])
def test_known_good_reads_normal(detector, defect):
    images = sorted(_test_dir(defect).glob("*.png"))[:3]
    assert images, "no good test images found"
    for img in images:
        result = detector.detect(img, part_id=f"good-{img.stem}", save_heatmap=False)
        assert result.is_defective is False, f"{img.name} flagged defective (score {result.anomaly_score})"


@requires_artifacts
@pytest.mark.parametrize("defect", ["broken_large", "broken_small", "contamination"])
def test_known_defect_reads_defective(detector, defect):
    images = sorted(_test_dir(defect).glob("*.png"))[:2]
    assert images, f"no {defect} test images found"
    for img in images:
        result = detector.detect(img, part_id=f"{defect}-{img.stem}", save_heatmap=False)
        assert result.is_defective is True, f"{img.name} read normal (score {result.anomaly_score})"


@requires_artifacts
def test_auroc_and_far_meet_budget():
    metrics = json.loads(settings.metrics_path.read_text())
    # AUROC is threshold-free and the robust headline metric.
    assert metrics["image_auroc"] >= 0.95, f"AUROC {metrics['image_auroc']} below 0.95"
    # Held-out FAR must meet the budget to within the dataset's achievable resolution (1/n_defect):
    # with few defect samples a single miss can exceed 2% even for a near-perfect model.
    assert metrics["far_budget_met_within_granularity"] is True, (
        f"Holdout FAR {metrics['holdout']['far']} exceeds target {settings.far_target} "
        f"by more than the granularity {metrics['holdout']['far_granularity']}"
    )
    assert metrics["degenerate_separation"] is False
