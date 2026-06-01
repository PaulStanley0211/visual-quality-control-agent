"""Reference save/load round-trip (no backbone needed — we hand-build a Reference)."""
from __future__ import annotations

import numpy as np

from config import settings
from drift.reference import Reference, load_reference, save_reference


def test_reference_round_trip(tmp_path):
    ref = Reference(
        embeddings=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        stat_mean={"brightness": 100.0, "contrast": 30.0, "sharpness": 500.0},
        stat_std={"brightness": 10.0, "contrast": 5.0, "sharpness": 50.0},
        category=settings.category,
    )
    path = tmp_path / "reference.npz"
    save_reference(ref, path)
    loaded = load_reference(path)

    np.testing.assert_allclose(loaded.embeddings, ref.embeddings)
    assert loaded.stat_mean == ref.stat_mean
    assert loaded.stat_std == ref.stat_std
    assert loaded.category == ref.category
