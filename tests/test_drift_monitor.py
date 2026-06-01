"""DriftMonitor.score with an injected reference + fake extractor (no backbone needed)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from config import settings
from drift.monitor import DriftMonitor
from drift.reference import Reference


class _FakeExtractor:
    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed(self, img):
        return self._vec


def _reference():
    # Five identical reference points at the origin in 4-D.
    return Reference(
        embeddings=np.zeros((5, 4), dtype=np.float32),
        stat_mean={"brightness": 128.0, "contrast": 30.0, "sharpness": 500.0},
        stat_std={"brightness": 10.0, "contrast": 5.0, "sharpness": 50.0},
        category=settings.category,
    )


def _img():
    return Image.new("RGB", (32, 32), (128, 128, 128))


def test_far_embedding_is_ood():
    m = DriftMonitor(reference=_reference(), threshold=0.5, extractor=_FakeExtractor([1.0, 0.0, 0.0, 0.0]))
    res = m.score(_img())
    assert res.is_ood is True              # distance 1.0 >= threshold 0.5
    assert res.drift_score == 1.0
    assert res.note.startswith("OOD")


def test_near_embedding_in_distribution():
    m = DriftMonitor(reference=_reference(), threshold=2.0, extractor=_FakeExtractor([1.0, 0.0, 0.0, 0.0]))
    res = m.score(_img())
    assert res.is_ood is False             # distance 1.0 < threshold 2.0
    assert res.note == "In-distribution"


def test_stat_deltas_reported_in_sigma_units():
    m = DriftMonitor(reference=_reference(), threshold=2.0, extractor=_FakeExtractor([0.0, 0.0, 0.0, 0.0]))
    res = m.score(_img())  # flat 128 image: brightness 128 == baseline mean => 0σ
    assert res.brightness_delta == 0.0
    assert res.contrast_delta is not None and res.sharpness_delta is not None


def test_category_mismatch_rejected():
    ref = _reference()
    ref.category = "not-the-active-category"
    try:
        DriftMonitor(reference=ref, threshold=1.0, extractor=_FakeExtractor([0, 0, 0, 0]))
        assert False, "expected category mismatch to raise"
    except ValueError:
        pass
