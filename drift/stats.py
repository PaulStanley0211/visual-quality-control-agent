"""Cheap, interpretable image statistics for drift *context* (never a decision input).

brightness = mean luminance, contrast = luminance std (RMS contrast), sharpness = variance of
the Laplacian (low => blurry/out-of-focus). Reported as σ-deltas vs the training baseline so a
human reviewer sees, e.g., 'brightness down 2.4σ'. Pure NumPy/PIL (no OpenCV).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian via edge-padded shifts (no SciPy/OpenCV dependency)."""
    p = np.pad(gray, 1, mode="edge")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * gray


def image_stats(img: Image.Image) -> dict[str, float]:
    """Return {brightness, contrast, sharpness} for a PIL image (luminance, 0–255 scale)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(_laplacian(gray).var()),
    }
