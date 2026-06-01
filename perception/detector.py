"""Perception entry point.

Loads the exported PatchCore Torch model via anomalib's lightweight
``TorchInferencer`` (no training Engine) and exposes ``detect()`` returning a
schema-validated :class:`DetectResult`. This is the single perception interface
the agent and the FastAPI service both call.

The operating threshold and confidence width are calibrated by
``eval/perception_eval.py`` (persisted to ``metrics.json``); ``detect()`` reads
them so behaviour matches the validated error budget.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from anomalib.deploy import TorchInferencer

from config import settings
from contracts.models import DetectResult

logger = logging.getLogger(__name__)


def find_exported_model(perception_dir: Path | None = None) -> Path:
    """Locate the exported PatchCore ``model.pt`` under the perception artifacts dir.

    Training writes the deployable model to ``<perception_dir>/weights/torch/model.pt``;
    we prefer that exact path so a stale checkpoint elsewhere in the tree can't be picked.
    """
    root = perception_dir or settings.perception_dir
    preferred = root / "weights" / "torch" / "model.pt"
    if preferred.is_file():
        return preferred
    for pattern in ("**/weights/torch/model.pt", "**/model.pt"):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No exported PatchCore model found under {root}. "
        "Run `uv run python -m perception.train` first."
    )


def load_inferencer(model_path: Path | None = None, device: str = "cpu") -> TorchInferencer:
    """Load the exported PatchCore model for single-image inference.

    The anomalib TORCH export is a pickled model object, which ``torch.load``
    refuses by default. We opt in via ``TRUST_REMOTE_CODE`` because this artifact
    is produced by our own ``perception.train`` — it is not a third-party file.
    Set unconditionally so a pre-existing ``TRUST_REMOTE_CODE=0`` in the env can't
    silently break loading of our own model.
    """
    os.environ["TRUST_REMOTE_CODE"] = "1"
    path = model_path or find_exported_model()
    return TorchInferencer(path=str(path), device=device)


def _load_calibration() -> tuple[float, float]:
    """Return (operating_threshold, confidence_width) from the eval metrics file, validated."""
    if not settings.metrics_path.exists():
        raise FileNotFoundError(
            f"Calibration not found at {settings.metrics_path}. "
            "Run `uv run python -m eval.perception_eval` after training to calibrate the threshold."
        )
    try:
        data = json.loads(settings.metrics_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Calibration file {settings.metrics_path} is unreadable: {e}. Re-run eval.perception_eval.") from e

    for key in ("operating_threshold", "confidence_width", "category"):
        if key not in data:
            raise ValueError(f"Calibration file is missing '{key}'. Re-run eval.perception_eval.")
    if data["category"] != settings.category:
        raise ValueError(
            f"Calibration is for category '{data['category']}' but settings.category is '{settings.category}'. "
            "Re-run eval.perception_eval for the active category."
        )
    threshold = float(data["operating_threshold"])
    width = float(data["confidence_width"])
    if not math.isfinite(threshold) or not math.isfinite(width) or width <= 0:
        raise ValueError(
            f"Calibration has invalid threshold={threshold} / confidence_width={width}. Re-run eval.perception_eval."
        )
    return threshold, width


def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


class Detector:
    """Stateful perception model wrapper; load once, call ``detect()`` per image."""

    def __init__(self, threshold: float | None = None, confidence_width: float | None = None):
        self.model_path = find_exported_model()
        self.inferencer = load_inferencer(self.model_path, device="cpu")
        if threshold is None or confidence_width is None:
            cal_threshold, cal_width = _load_calibration()
            self.threshold = cal_threshold if threshold is None else threshold
            self.confidence_width = cal_width if confidence_width is None else confidence_width
        else:
            self.threshold = threshold
            self.confidence_width = confidence_width

    def _confidence(self, score: float, is_defective: bool) -> float:
        """Logistic confidence in the binary call, calibrated by ``confidence_width``.

        At the threshold the call is maximally ambiguous (0.5); confidence rises
        toward 1.0 as the score moves away from the threshold. Uses the tanh form
        of the logistic, which is overflow-safe for any score scale.
        """
        z = (score - self.threshold) / max(self.confidence_width, 1e-9)
        p_defective = 0.5 * (1.0 + math.tanh(z / 2.0))
        return p_defective if is_defective else (1.0 - p_defective)

    @staticmethod
    def _normalize_map(anomaly_map) -> np.ndarray:
        """Return the anomaly map as a 2D float array normalized to [0, 1]."""
        am = anomaly_map.detach().cpu().numpy() if isinstance(anomaly_map, torch.Tensor) else np.asarray(anomaly_map)
        am = np.squeeze(am).astype(np.float32)
        while am.ndim > 2:  # collapse any remaining leading dims deterministically
            am = am[0]
        lo, hi = float(am.min()), float(am.max())
        return (am - lo) / (hi - lo) if hi > lo else np.zeros_like(am)

    @staticmethod
    def _describe_anomaly(norm: np.ndarray, hot: float = 0.5) -> tuple[float, str | None]:
        """From the normalized map, return (anomalous area fraction, coarse location)."""
        mask = norm >= hot
        area = float(mask.mean())
        if not mask.any():
            return area, None
        rows, cols = np.nonzero(mask)
        cy, cx = rows.mean() / norm.shape[0], cols.mean() / norm.shape[1]
        vert = "upper" if cy < 0.34 else "lower" if cy > 0.66 else "middle"
        horiz = "left" if cx < 0.34 else "right" if cx > 0.66 else "center"
        return area, "center" if (vert, horiz) == ("middle", "center") else f"{vert}-{horiz}"

    def _save_heatmap(self, norm: np.ndarray, source_image: Image.Image, part_id: str) -> str:
        from matplotlib import cm

        colored = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
        heat = Image.fromarray(colored).resize(source_image.size, Image.BILINEAR)
        overlay = Image.blend(source_image.convert("RGB"), heat, alpha=0.5)

        settings.heatmaps_dir.mkdir(parents=True, exist_ok=True)
        out_path = settings.heatmaps_dir / f"{part_id}_heatmap.png"
        overlay.save(out_path)
        return str(out_path)

    def detect(self, image: str | Path | Image.Image, part_id: str = "part", save_heatmap: bool = True) -> DetectResult:
        """Run anomaly detection on a single image and return a ``DetectResult``."""
        try:
            source = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Could not read image for part '{part_id}' ({image!r}): {e}") from e

        result = self.inferencer.predict(image=source)
        score = _to_scalar(result.pred_score)
        # A non-finite score must never be silently dispositioned: NaN >= threshold is False,
        # which would read as 'good' — the worst-case false accept for a QC system.
        if not math.isfinite(score):
            raise ValueError(
                f"Perception produced a non-finite anomaly score ({score}) for part '{part_id}'; refusing to disposition."
            )

        is_defective = bool(score >= self.threshold)
        confidence = self._confidence(score, is_defective)

        defect_area: float | None = None
        location: str | None = None
        heatmap_path: str | None = None
        if result.anomaly_map is not None:
            norm = self._normalize_map(result.anomaly_map)
            defect_area, location = self._describe_anomaly(norm)
            if save_heatmap:
                try:
                    heatmap_path = self._save_heatmap(norm, source, part_id)
                except Exception as e:  # an optional overlay must never fail a valid detection
                    logger.warning("Heatmap generation failed for part '%s': %s", part_id, e)

        return DetectResult(
            is_defective=is_defective,
            confidence=round(confidence, 4),
            anomaly_score=round(score, 6),
            threshold=round(self.threshold, 6),
            heatmap_path=heatmap_path,
            defect_area=round(defect_area, 6) if defect_area is not None else None,
            location=location,
        )
