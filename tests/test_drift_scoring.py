"""Unit tests for the drift subsystem's pure logic (config, scoring math, image stats)."""
from __future__ import annotations

from config import settings


def test_drift_config_defaults_and_paths():
    assert settings.drift_enabled is True
    assert settings.drift_k == 5
    assert settings.drift_far_alarm_target == 0.05
    assert settings.drift_window == 50
    assert settings.drift_psi_significant == 0.25
    # Per-category artifact layout mirrors perception_dir.
    assert settings.drift_dir == settings.artifacts_dir / "drift" / settings.category
    assert settings.drift_reference_path == settings.drift_dir / "reference.npz"
    assert settings.drift_metrics_path == settings.drift_dir / "drift_metrics.json"


from contracts.models import DriftResult, InspectionOutput, Decision, Disposition


def test_drift_result_contract():
    dr = DriftResult(is_ood=True, drift_score=1.23, threshold=0.8, note="OOD: brightness down 2.4σ")
    assert dr.is_ood is True
    assert dr.brightness_delta is None  # optional context defaults to None
    assert dr.note.startswith("OOD")


def test_inspection_output_drift_defaults_none():
    out = InspectionOutput(
        part_id="P1",
        decision=Decision(disposition=Disposition.PASS, confidence=0.9),
        summary="ok",
    )
    assert out.drift is None


import numpy as np

from drift.scoring import knn_distance, population_stability_index


def test_knn_distance_zero_on_exact_match():
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Query equals a reference point; with k=1 the nearest distance is 0.
    assert knn_distance(np.array([0.0, 0.0]), ref, k=1) == 0.0


def test_knn_distance_mean_of_k_nearest():
    ref = np.array([[0.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    # Distances from [0,0]: 0, 3, 4. Mean of 2 nearest = (0+3)/2 = 1.5.
    assert knn_distance(np.array([0.0, 0.0]), ref, k=2) == 1.5


def test_knn_distance_k_clamped_to_reference_size():
    ref = np.array([[0.0, 0.0], [2.0, 0.0]])
    # k larger than n uses all points: mean(0, 2) = 1.0.
    assert knn_distance(np.array([0.0, 0.0]), ref, k=10) == 1.0


def test_psi_zero_for_identical_distributions():
    p = [0.25, 0.25, 0.25, 0.25]
    assert population_stability_index(p, p) == 0.0


def test_psi_positive_and_grows_with_shift():
    expected = [0.4, 0.3, 0.2, 0.1]
    mild = [0.35, 0.3, 0.2, 0.15]
    severe = [0.1, 0.2, 0.3, 0.4]
    psi_mild = population_stability_index(expected, mild)
    psi_severe = population_stability_index(expected, severe)
    assert psi_mild > 0
    assert psi_severe > psi_mild


from PIL import Image

from drift.stats import image_stats


def test_image_stats_keys_and_brightness_ordering():
    dark = Image.new("RGB", (32, 32), (20, 20, 20))
    bright = Image.new("RGB", (32, 32), (200, 200, 200))
    sd, sb = image_stats(dark), image_stats(bright)
    assert set(sd) == {"brightness", "contrast", "sharpness"}
    assert sb["brightness"] > sd["brightness"]


def test_image_stats_contrast_and_sharpness_flat_image_is_zero():
    flat = Image.new("RGB", (32, 32), (128, 128, 128))
    s = image_stats(flat)
    assert s["contrast"] == 0.0      # uniform image has no spread
    assert s["sharpness"] == 0.0     # uniform image has no edges


from eval.drift_eval import calibrate_threshold, psi_reference_bins


def test_calibrate_threshold_bounds_clean_false_alarm():
    # 100 clean scores 0..0.99; with a 5% alarm budget the threshold should sit near the 95th pct.
    clean = np.linspace(0.0, 0.99, 100)
    t = calibrate_threshold(clean, far_alarm_target=0.05)
    # At most ~5% of clean scores may exceed the threshold.
    assert np.mean(clean >= t) <= 0.05 + 1e-9


def test_psi_reference_bins_sum_to_one():
    clean = np.linspace(0.0, 1.0, 50)
    ref = psi_reference_bins(clean, n_bins=8)
    assert len(ref["bin_edges"]) == 9
    assert abs(sum(ref["expected_props"]) - 1.0) < 1e-9
