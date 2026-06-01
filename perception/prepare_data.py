"""Fetch one MVTec AD category and lay it out in anomalib's expected structure.

Why this exists: anomalib 2.5's built-in MVTec downloader points at a mydrive.ch
mirror that now returns HTTP 404. This module fetches only the configured
category (default ``bottle``, ~320 MB vs the full 4.9 GB archive) from a public
Hugging Face mirror that hosts the original images + masks, and reorganizes it
into the canonical MVTec layout::

    datasets/MVTecAD/<category>/
        train/good/*.png
        test/{good,<defect>...}/*.png
        ground_truth/<defect>/*_mask.png

Once that directory exists, anomalib's ``MVTecAD.prepare_data`` finds the
category and skips the (dead) download. Idempotent: re-running is a no-op once
the data is in place.

Run:  uv run python -m perception.prepare_data
"""
from __future__ import annotations

import shutil
from pathlib import Path

from config import settings

# Public mirror of MVTec AD in the original folder structure (transposed:
# images/<split>/<category>/... and masks/test/<category>/...).
HF_REPO_ID = "TheoM55/mvtec_anomaly_detection"

# Expected official train/good counts per category — a sanity check that the fetch is complete.
EXPECTED_TRAIN_GOOD = {"bottle": 209, "hazelnut": 391}


def _category_root() -> Path:
    return settings.data_root / "MVTecAD" / settings.category


def _is_already_prepared() -> bool:
    """True only if the category is structurally complete: train/good present (at least the
    expected count) and test/ has both 'good' and at least one defect class with images."""
    cat = _category_root()
    train_good = cat / "train" / "good"
    test_root = cat / "test"
    if not train_good.is_dir() or not test_root.is_dir():
        return False
    n_train = len(list(train_good.glob("*.png")))
    expected = EXPECTED_TRAIN_GOOD.get(settings.category, 1)
    if n_train < expected:  # lower-bound: partial download is treated as not-ready
        return False
    has_good = (test_root / "good").is_dir() and any((test_root / "good").glob("*.png"))
    has_defect = any(
        sub.is_dir() and sub.name != "good" and any(sub.glob("*.png")) for sub in test_root.iterdir()
    )
    return has_good and has_defect


def prepare() -> Path:
    """Download + reorganize the configured category; return its root directory."""
    cat_root = _category_root()
    if _is_already_prepared():
        print(f"[prepare_data] '{settings.category}' already present at {cat_root}; skipping fetch.")
        return cat_root

    from huggingface_hub import snapshot_download

    category = settings.category
    cache_dir = settings.data_root / "_hf_mvtec_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] Downloading '{category}' from HF mirror '{HF_REPO_ID}'...")
    snapshot = Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(cache_dir),
            allow_patterns=[
                f"images/train/{category}/**",
                f"images/test/{category}/**",
                f"masks/test/{category}/**",
            ],
        )
    )

    src_train = snapshot / "images" / "train" / category
    src_test = snapshot / "images" / "test" / category
    src_masks = snapshot / "masks" / "test" / category
    if not src_train.is_dir() or not src_test.is_dir():
        raise FileNotFoundError(
            f"HF mirror did not yield expected folders for '{category}' under {snapshot}."
        )

    print(f"[prepare_data] Reorganizing into anomalib layout at {cat_root}...")
    cat_root.mkdir(parents=True, exist_ok=True)
    # train/good and the full test/ tree map across directly.
    shutil.copytree(src_train, cat_root / "train", dirs_exist_ok=True)
    shutil.copytree(src_test, cat_root / "test", dirs_exist_ok=True)
    # masks/test/<cat>/<defect>/<n>_mask.png -> ground_truth/<defect>/<n>_mask.png
    if src_masks.is_dir():
        shutil.copytree(src_masks, cat_root / "ground_truth", dirs_exist_ok=True)

    n_train = len(list((cat_root / "train" / "good").glob("*.png")))
    n_test = len(list((cat_root / "test").glob("*/*.png")))
    if n_train == 0 or n_test == 0:
        raise RuntimeError(
            f"Data layout incomplete after fetch: {n_train} train/good, {n_test} test images at {cat_root}."
        )
    print(f"[prepare_data] Done: {n_train} train/good, {n_test} test images at {cat_root}.")
    return cat_root


if __name__ == "__main__":
    prepare()
