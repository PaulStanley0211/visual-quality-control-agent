"""EmbeddingExtractor shape + determinism. Skips if the backbone weights can't be constructed
(e.g. offline clean clone with no cached timm weights)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def _extractor_or_skip():
    try:
        from drift.extractor import EmbeddingExtractor

        return EmbeddingExtractor()
    except Exception as e:  # noqa: BLE001 - any construction failure => environment can't run this test
        pytest.skip(f"backbone unavailable: {e}")


def test_embedding_shape_and_l2_normalized():
    ext = _extractor_or_skip()
    img = Image.new("RGB", (256, 256), (123, 116, 100))
    emb = ext.embed(img)
    assert emb.shape == (384,)                       # layer2 (128) + layer3 (256), GAP'd
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)  # L2-normalized


def test_embedding_is_deterministic():
    ext = _extractor_or_skip()
    img = Image.new("RGB", (256, 256), (90, 90, 90))
    np.testing.assert_allclose(ext.embed(img), ext.embed(img), atol=1e-6)
