"""Build and load the per-category training-good reference embedding set (the drift "fit" step).

``build_reference()`` runs the extractor over every training-good image for the active category and
persists the embedding stack plus the brightness/contrast/sharpness baselines. Analogous to
``perception.train`` (a one-pass fit), idempotent, CPU-fast.

Run:  uv run python -m drift.reference
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from drift.stats import image_stats

_STAT_KEYS = ("brightness", "contrast", "sharpness")


@dataclass
class Reference:
    """The drift reference: training-good embeddings + image-stat baselines for one category."""

    embeddings: np.ndarray          # (n, d) float32, L2-normalized
    stat_mean: dict[str, float]
    stat_std: dict[str, float]
    category: str


def save_reference(ref: Reference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=ref.embeddings.astype(np.float32),
        stat_keys=np.array(_STAT_KEYS),
        stat_mean=np.array([ref.stat_mean[k] for k in _STAT_KEYS], dtype=np.float64),
        stat_std=np.array([ref.stat_std[k] for k in _STAT_KEYS], dtype=np.float64),
        category=np.array(ref.category),
    )


def load_reference(path: Path | None = None) -> Reference:
    path = path or settings.drift_reference_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Drift reference not found at {path}. Run `uv run python -m drift.reference` "
            "(after `perception.train`) to build it for the active category."
        )
    data = np.load(path, allow_pickle=False)
    keys = [str(k) for k in data["stat_keys"]]
    return Reference(
        embeddings=data["embeddings"].astype(np.float32),
        stat_mean=dict(zip(keys, (float(v) for v in data["stat_mean"]))),
        stat_std=dict(zip(keys, (float(v) for v in data["stat_std"]))),
        category=str(data["category"]),
    )


def _train_good_dir() -> Path:
    return settings.data_root / "MVTecAD" / settings.category / "train" / "good"


def build_reference() -> Reference:
    """Fit the reference set from the category's training-good images and persist it."""
    from drift.extractor import EmbeddingExtractor

    good_dir = _train_good_dir()
    images = sorted(good_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(
            f"No training-good images at {good_dir}. Run `perception.train` (which fetches the category) first."
        )

    extractor = EmbeddingExtractor()
    embeddings = np.empty((len(images), 384), dtype=np.float32)
    stats_accum = {k: [] for k in _STAT_KEYS}
    print(f"[drift.reference] Embedding {len(images)} training-good images for '{settings.category}' (CPU)...")
    for i, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        embeddings[i] = extractor.embed(img)
        s = image_stats(img)
        for k in _STAT_KEYS:
            stats_accum[k].append(s[k])

    stat_mean = {k: float(np.mean(v)) for k, v in stats_accum.items()}
    stat_std = {k: float(np.std(v)) for k, v in stats_accum.items()}
    ref = Reference(embeddings=embeddings, stat_mean=stat_mean, stat_std=stat_std, category=settings.category)
    save_reference(ref, settings.drift_reference_path)
    print(f"[drift.reference] Saved reference ({len(images)} embeddings) to {settings.drift_reference_path}")
    return ref


if __name__ == "__main__":
    build_reference()
