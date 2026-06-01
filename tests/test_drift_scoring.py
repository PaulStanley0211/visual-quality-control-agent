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
