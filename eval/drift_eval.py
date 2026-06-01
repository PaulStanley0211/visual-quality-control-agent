"""Drift-monitor validation + threshold calibration (mirrors eval/perception_eval.py).

Methodology (honest, seeded, small-sample-aware):
  * Clean (in-distribution) set = the category's ``test/good`` images — DISJOINT from the reference
    (built on ``train/good``), so distances aren't artificially deflated.
  * Synthesize drift by perturbing copies of the clean images: brightness, contrast, gaussian blur,
    gaussian noise, JPEG compression (each at seeded severities).
  * Calibrate the OOD threshold on a seeded clean calibration split to bound the clean false-alarm
    rate at ``settings.drift_far_alarm_target``; report the alarm rate on the disjoint clean holdout
    with a Wilson 95% upper bound.
  * Report separability AUROC (clean vs drifted), per-perturbation detection rate, and the PSI
    reference bins consumed by drift/report.py.

Outputs:
  - artifacts/drift/<category>/drift_metrics.json
  - artifacts/drift/<category>/drift_separation.png

Run:  uv run python -m eval.drift_eval
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

from config import settings
from drift.reference import load_reference
from drift.scoring import knn_distance

CALIBRATION_FRACTION = 0.5


def _perturbations(rng: np.random.Generator) -> dict:
    """Name -> function(img: PIL.Image) -> PIL.Image. Deterministic given the seeded rng."""
    def brightness_down(im):
        return ImageEnhance.Brightness(im).enhance(0.55)

    def brightness_up(im):
        return ImageEnhance.Brightness(im).enhance(1.6)

    def contrast_down(im):
        return ImageEnhance.Contrast(im).enhance(0.5)

    def blur(im):
        return im.filter(ImageFilter.GaussianBlur(radius=2.5))

    def noise(im):
        arr = np.asarray(im, dtype=np.float32)
        arr = arr + rng.normal(0.0, 25.0, size=arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def jpeg(im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=20)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return {
        "brightness_down": brightness_down,
        "brightness_up": brightness_up,
        "contrast_down": contrast_down,
        "blur": blur,
        "noise": noise,
        "jpeg": jpeg,
    }


def calibrate_threshold(clean_scores: np.ndarray, far_alarm_target: float) -> float:
    """Smallest threshold whose clean false-alarm rate (fraction of clean scores >= t) <= target.

    Implemented as the (1 - target) quantile of the clean scores: at most ``target`` fraction of
    clean images exceed it.
    """
    return float(np.quantile(clean_scores, 1.0 - far_alarm_target))


def psi_reference_bins(clean_scores: np.ndarray, n_bins: int = 10) -> dict:
    """Histogram the clean (in-distribution) scores into n_bins; return edges + expected props.

    Edges span [min, max] of the clean scores with the outer edges pushed to ±inf so live scores
    beyond the observed clean range still fall into the end bins.
    """
    lo, hi = float(clean_scores.min()), float(clean_scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    inner = np.linspace(lo, hi, n_bins + 1)
    edges = inner.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(clean_scores, bins=edges)
    props = counts / counts.sum()
    # Store finite edges (replace ±inf with the observed bounds) for JSON + np.histogram reuse.
    finite_edges = inner.tolist()
    finite_edges[0], finite_edges[-1] = -1e9, 1e9
    return {"bin_edges": finite_edges, "expected_props": props.tolist()}


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(min(1.0, center + half))


def _clean_image_paths() -> list[Path]:
    good = settings.data_root / "MVTecAD" / settings.category / "test" / "good"
    if not good.is_dir():
        raise FileNotFoundError(f"Clean set not found at {good}. Run perception.train first.")
    return sorted(good.glob("*.png"))


def _save_plot(clean: np.ndarray, drifted: np.ndarray, t: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(clean, bins=20, alpha=0.6, label="clean (in-distribution)", color="#2980b9", density=True)
    ax.hist(drifted, bins=20, alpha=0.6, label="drifted (synthetic)", color="#c0392b", density=True)
    ax.axvline(t, ls=":", color="black", label=f"OOD threshold = {t:.3f}")
    ax.set_xlabel("Drift score (mean kNN distance to training-good)")
    ax.set_ylabel("Density")
    ax.set_title(f"Drift-score separation — MVTec '{settings.category}'")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _loo_clean_scores(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out kNN distances over the training-good reference: each good image's distance to
    the rest of the good manifold (self excluded). A large, leakage-free in-distribution clean-score
    sample for calibrating the OOD threshold (far more stable than a tiny held-out test split)."""
    n = embeddings.shape[0]
    scores = np.empty(n, dtype=float)
    for i in range(n):
        others = np.delete(embeddings, i, axis=0)
        scores[i] = knn_distance(embeddings[i], others, k)
    return scores


def evaluate() -> dict:
    rng = np.random.default_rng(settings.seed)
    reference = load_reference(settings.drift_reference_path)
    if reference.category != settings.category:
        raise ValueError(f"Reference is for '{reference.category}', not active '{settings.category}'.")

    from drift.extractor import EmbeddingExtractor

    extractor = EmbeddingExtractor()

    def score(img: Image.Image) -> float:
        return knn_distance(extractor.embed(img), reference.embeddings, settings.drift_k)

    # Calibrate the threshold on a leakage-free leave-one-out sample of the training-good reference
    # (large, in-distribution). The (1 - target) quantile bounds the clean false-alarm rate at the
    # target on a well-sampled set — far more stable than calibrating on a handful of test images.
    cal_scores = _loo_clean_scores(reference.embeddings, settings.drift_k)
    threshold = calibrate_threshold(cal_scores, settings.drift_far_alarm_target)
    cal_false_alarm = float(np.mean(cal_scores >= threshold))

    # Disjoint clean holdout: test/good (never used to build the reference) — the honest
    # generalization check. Synthesize drift by perturbing copies of these clean images.
    paths = _clean_image_paths()
    print(f"[drift.eval] Scoring {len(paths)} clean images + perturbations for '{settings.category}' (CPU)...")
    clean_scores = np.array([score(Image.open(p).convert("RGB")) for p in paths], dtype=float)

    perts = _perturbations(rng)
    drifted_by_type: dict[str, np.ndarray] = {}
    for name, fn in perts.items():
        drifted_by_type[name] = np.array(
            [score(fn(Image.open(p).convert("RGB"))) for p in paths], dtype=float
        )
    drifted_all = np.concatenate(list(drifted_by_type.values()))

    # Separability AUROC (threshold-free headline): clean=0, drifted=1.
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(drifted_all))])
    s = np.concatenate([clean_scores, drifted_all])
    auroc = float(roc_auc_score(y, s))

    n_hold = len(clean_scores)
    hold_alarms = int(np.sum(clean_scores >= threshold))
    holdout_far = hold_alarms / n_hold if n_hold else 0.0
    far_granularity = 1.0 / n_hold if n_hold else 1.0
    holdout_wilson = _wilson_upper(hold_alarms, n_hold)

    detection_rate = {name: float(np.mean(sc >= threshold)) for name, sc in drifted_by_type.items()}
    psi_ref = psi_reference_bins(clean_scores, n_bins=10)

    cal_granularity = 1.0 / len(cal_scores) if len(cal_scores) else 1.0
    alarm_ok = bool(cal_false_alarm <= settings.drift_far_alarm_target + cal_granularity + 1e-9)
    holdout_within_granularity = bool(holdout_far <= settings.drift_far_alarm_target + far_granularity + 1e-9)
    auroc_ok = bool(auroc >= 0.90)

    metrics = {
        "category": settings.category,
        "seed": settings.seed,
        "drift_k": settings.drift_k,
        "n_reference": int(reference.embeddings.shape[0]),
        "n_clean_holdout": int(n_hold),
        "n_drifted": int(len(drifted_all)),
        "separability_auroc": round(auroc, 4),
        "operating_threshold": round(threshold, 6),
        "far_alarm_target": settings.drift_far_alarm_target,
        "calibration": {
            "method": "train-good leave-one-out (leakage-free)",
            "n": int(len(cal_scores)),
            "false_alarm_rate": round(cal_false_alarm, 4),
        },
        "holdout": {
            "set": "test/good (disjoint from reference)",
            "n_clean": int(n_hold),
            "false_alarm_rate": round(holdout_far, 4),
            "far_alarm_granularity": round(far_granularity, 4),
            "false_alarm_wilson_upper95": round(holdout_wilson, 4),
            "within_granularity": holdout_within_granularity,
        },
        "detection_rate_by_perturbation": {k: round(v, 4) for k, v in detection_rate.items()},
        "psi_reference": psi_ref,
        "auroc_ok": auroc_ok,
        "alarm_ok": alarm_ok,
        "methodology_note": (
            "Threshold calibrated to bound the clean false-alarm rate at the target on a leakage-free "
            "leave-one-out sample of the training-good reference (large, in-distribution). The disjoint "
            "test/good holdout reports the generalization false-alarm rate with 1/n granularity and a "
            "Wilson 95% upper bound; with only n_clean_holdout held-out clean images it cannot certify "
            "the target, so it is read with its sampling uncertainty (mirrors the perception FAR "
            "methodology). Drift is synthesized via brightness/contrast/blur/noise/jpeg perturbations; "
            "AUROC is threshold-free."
        ),
    }

    settings.drift_dir.mkdir(parents=True, exist_ok=True)
    settings.drift_metrics_path.write_text(json.dumps(metrics, indent=2))
    _save_plot(clean_scores, drifted_all, threshold, settings.drift_dir / "drift_separation.png")

    print("[drift.eval] Drift metrics:")
    print(json.dumps(metrics, indent=2))
    print(
        f"[drift.eval] separability AUROC {auroc:.4f} (>=0.90: {auroc_ok}); "
        f"calibration false-alarm {cal_false_alarm:.1%} (budget {settings.drift_far_alarm_target:.0%}: {alarm_ok}); "
        f"disjoint holdout false-alarm {holdout_far:.1%} "
        f"(±granularity {far_granularity:.1%}, Wilson-upper {holdout_wilson:.1%}, n={n_hold})."
    )
    return metrics


if __name__ == "__main__":
    evaluate()
