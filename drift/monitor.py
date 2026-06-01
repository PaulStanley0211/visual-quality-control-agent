"""Per-image drift scoring: DriftMonitor.score(image) -> DriftResult.

Loads the per-category training-good reference set and the calibrated OOD threshold, then scores
each image by mean kNN distance to the reference (PatchCore's idea at the image level). The raw
disposition is unaffected — an OOD result is consumed by the agent's escalation logic. Stateful
wrapper, load once and reuse, exactly like ``perception.detector.Detector``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import settings
from contracts.models import DriftResult
from drift.reference import Reference, load_reference
from drift.scoring import knn_distance
from drift.stats import image_stats

def _load_threshold() -> float:
    """Read the calibrated OOD threshold from drift_metrics.json (written by eval.drift_eval)."""
    path = settings.drift_metrics_path
    if not path.exists():
        raise FileNotFoundError(
            f"Drift calibration not found at {path}. Run `uv run python -m eval.drift_eval` after building the reference."
        )
    data = json.loads(path.read_text())
    if data.get("category") != settings.category:
        raise ValueError(
            f"Drift calibration is for '{data.get('category')}' but settings.category is '{settings.category}'. "
            "Re-run eval.drift_eval for the active category."
        )
    t = float(data["operating_threshold"])
    if not math.isfinite(t):
        raise ValueError(f"Drift calibration has a non-finite threshold ({t}). Re-run eval.drift_eval.")
    return t


def _describe(is_ood: bool, deltas: dict[str, float]) -> str:
    """Plain-language read; for OOD, name the most extreme image-stat delta as a hint."""
    if not is_ood:
        return "In-distribution"
    key = max(deltas, key=lambda k: abs(deltas[k]))
    direction = "up" if deltas[key] >= 0 else "down"
    return f"OOD: {key} {direction} {abs(deltas[key]):.1f}σ"


class DriftMonitor:
    """Load the reference + threshold once; score each image. Inject deps for testing."""

    def __init__(self, reference: Reference | None = None, threshold: float | None = None, extractor=None):
        self.reference = reference or load_reference(settings.drift_reference_path)
        if self.reference.category != settings.category:
            raise ValueError(
                f"Drift reference is for '{self.reference.category}' but settings.category is "
                f"'{settings.category}'. Re-build the reference for the active category."
            )
        self.threshold = _load_threshold() if threshold is None else float(threshold)
        self.k = settings.drift_k
        self._extractor = extractor  # lazily built if not injected

    @property
    def extractor(self):
        if self._extractor is None:
            from drift.extractor import EmbeddingExtractor

            self._extractor = EmbeddingExtractor()
        return self._extractor

    def score(self, image: str | Path | Image.Image) -> DriftResult:
        try:
            img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Could not read image for drift scoring ({image!r}): {e}") from e

        emb = self.extractor.embed(img)
        dist = knn_distance(emb, self.reference.embeddings, self.k)
        is_ood = bool(dist >= self.threshold)

        stats = image_stats(img)
        deltas = {
            k: (stats[k] - self.reference.stat_mean[k]) / max(self.reference.stat_std[k], 1e-6)
            for k in stats
        }
        return DriftResult(
            is_ood=is_ood,
            drift_score=round(dist, 6),
            threshold=round(self.threshold, 6),
            brightness_delta=round(deltas["brightness"], 3),
            contrast_delta=round(deltas["contrast"], 3),
            sharpness_delta=round(deltas["sharpness"], 3),
            note=_describe(is_ood, deltas),
        )
