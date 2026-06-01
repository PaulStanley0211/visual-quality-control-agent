"""Pure scoring math for the drift monitor — no torch, fully unit-testable.

- ``knn_distance``: the per-image drift score (mean distance to the k nearest training-good
  embeddings). This is PatchCore's kNN-to-coreset idea at the image level.
- ``population_stability_index``: the windowed drift metric for the population monitor.
"""
from __future__ import annotations

import numpy as np


def knn_distance(embedding: np.ndarray, reference: np.ndarray, k: int) -> float:
    """Mean Euclidean distance from ``embedding`` to its ``k`` nearest rows of ``reference``.

    ``reference`` is the (n, d) stack of training-good embeddings. ``k`` is clamped to n so a
    tiny reference set never errors. Small distance => the image looks like the training-good
    cloud; large distance => out-of-distribution.
    """
    if reference.ndim != 2:
        raise ValueError(f"reference must be 2-D (n, d); got shape {reference.shape}")
    dists = np.linalg.norm(reference - embedding, axis=1)
    k = max(1, min(k, dists.shape[0]))
    nearest = np.partition(dists, k - 1)[:k]
    return float(np.mean(nearest))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two binned proportion vectors over the same bins.

    PSI = Σ (actual − expected) · ln(actual / expected). Both inputs are proportions (each
    summing to ~1) over identical bins. A small epsilon floor avoids div-by-zero / log(0) for
    empty bins. Standard interpretation: <0.1 stable, 0.1–0.25 moderate, >0.25 significant.
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    if e.shape != a.shape:
        raise ValueError(f"expected and actual must share shape; got {e.shape} vs {a.shape}")
    return float(np.sum((a - e) * np.log(a / e)))
