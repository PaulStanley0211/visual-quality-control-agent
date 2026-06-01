"""Perception validation against the error budget.

Runs the trained PatchCore model over the pinned MVTec test split (good +
defective) and reports:

  * image-level **AUROC** (threshold-free separability) vs the published baseline;
  * the false-accept / false-reject tradeoff curve;
  * an operating threshold calibrated on a **disjoint calibration split** and the
    resulting FAR/FRR measured on a **held-out split** — so the error-budget claim
    is an honest generalization estimate, not in-sample (threshold fit and scored
    on the same images).

Because MVTec 'bottle' has few defect samples, the achievable FAR resolution is
coarse (1 / n_defect); we report the FAR granularity and a Wilson 95% upper bound
so the budget margin is read with its sampling uncertainty.

Outputs:
  - artifacts/perception/metrics.json
  - artifacts/perception/threshold_tradeoff.png

Run:  uv run python -m eval.perception_eval
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lightning.pytorch import seed_everything

from config import settings
from perception.detector import _to_scalar, load_inferencer

# PatchCore image-level AUROC from the original paper (Roth et al., 2022, "Towards Total Recall
# in Industrial Anomaly Detection"). Both bottle and hazelnut are reported at 100.0%.
PUBLISHED_AUROC_BASELINE = {"bottle": 1.000, "hazelnut": 1.000}

# Fraction of each class used to CALIBRATE the threshold; the rest is held out for reporting.
CALIBRATION_FRACTION = 0.5


def _enumerate_test_set() -> tuple[list[Path], np.ndarray]:
    """Return (image_paths, labels) for the category's test split. 0 = good, 1 = defective."""
    test_root = settings.data_root / "MVTecAD" / settings.category / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test split not found at {test_root}. Run perception.train first.")
    paths: list[Path] = []
    labels: list[int] = []
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for img in sorted(sub.glob("*.png")):
            paths.append(img)
            labels.append(label)
    labels_arr = np.array(labels, dtype=int)
    n_good, n_defect = int(np.sum(labels_arr == 0)), int(np.sum(labels_arr == 1))
    if n_good == 0 or n_defect == 0:
        raise ValueError(
            f"Test split at {test_root} has {n_good} good / {n_defect} defective images; both classes are required."
        )
    return paths, labels_arr


def _stratified_split(labels: np.ndarray, seed: int, cal_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible per-class split into (calibration_mask, holdout_mask)."""
    rng = np.random.default_rng(seed)
    cal = np.zeros(len(labels), dtype=bool)
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_cal = int(round(len(idx) * cal_fraction))
        n_cal = min(max(n_cal, 1), len(idx) - 1)  # keep at least 1 in each of cal and holdout
        cal[idx[:n_cal]] = True
    return cal, ~cal


def _select_operating_threshold(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """Highest threshold whose false-accept rate (defect predicted good) is <= far_target.

    Decision rule: predict defective iff score >= threshold. FAR is monotonically
    non-decreasing in the threshold, so the satisfying thresholds form a prefix and
    the largest one minimizes the false-reject rate.
    """
    defect_scores = scores[labels == 1]
    candidates = np.unique(scores)
    best = float(candidates.min())
    for t in candidates:  # ascending
        far = float(np.mean(defect_scores < t)) if defect_scores.size else 0.0
        if far <= far_target:
            best = float(t)
        else:
            break
    return best


def _rates_at(scores: np.ndarray, labels: np.ndarray, t: float) -> dict:
    pred = scores >= t
    pos = labels == 1
    neg = labels == 0
    far = float(np.mean(~pred[pos])) if pos.any() else 0.0  # defect predicted good
    frr = float(np.mean(pred[neg])) if neg.any() else 0.0   # good predicted defective
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    fn = int(np.sum(~pred & pos))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"far": far, "frr": frr, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score upper bound for a proportion k/n (honest small-sample uncertainty)."""
    if n == 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(min(1.0, center + half))


def _save_tradeoff_plot(scores: np.ndarray, labels: np.ndarray, t_star: float, far_target: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(float(scores.min()), float(scores.max()), 200)
    fars = [_rates_at(scores, labels, t)["far"] for t in grid]
    frrs = [_rates_at(scores, labels, t)["frr"] for t in grid]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, fars, label="False-accept rate (defect passed)", color="#c0392b")
    ax.plot(grid, frrs, label="False-reject rate (good rejected)", color="#2980b9")
    ax.axhline(far_target, ls="--", color="#c0392b", alpha=0.5, label=f"FAR target = {far_target:.0%}")
    ax.axvline(t_star, ls=":", color="black", alpha=0.7, label=f"Operating threshold = {t_star:.3f}")
    ax.set_xlabel("Anomaly-score threshold")
    ax.set_ylabel("Rate (full test set)")
    ax.set_title(f"Perception threshold tradeoff — MVTec '{settings.category}'")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def evaluate() -> dict:
    seed_everything(settings.seed, workers=True)
    paths, labels = _enumerate_test_set()

    inferencer = load_inferencer(device="cpu")
    print(f"[eval] Scoring {len(paths)} test images for '{settings.category}' (CPU)...")
    scores = np.array([_to_scalar(inferencer.predict(image=str(p)).pred_score) for p in paths], dtype=float)
    if not np.all(np.isfinite(scores)):
        bad = [str(paths[i]) for i in np.where(~np.isfinite(scores))[0]]
        raise ValueError(f"Non-finite anomaly scores for {len(bad)} image(s), e.g. {bad[:3]}. Cannot evaluate.")

    # Threshold-free separability on the full set (headline metric, unaffected by calibration).
    auroc_full = float(roc_auc_score(labels, scores))

    # Calibrate the threshold on a disjoint split; report the budget on the held-out split.
    cal_mask, hold_mask = _stratified_split(labels, settings.seed, CALIBRATION_FRACTION)
    t_star = _select_operating_threshold(scores[cal_mask], labels[cal_mask], settings.far_target)

    hold = _rates_at(scores[hold_mask], labels[hold_mask], t_star)
    in_sample = _rates_at(scores, labels, t_star)
    auroc_hold = float(roc_auc_score(labels[hold_mask], scores[hold_mask]))

    n_defect_hold = int(np.sum(labels[hold_mask] == 1))
    far_granularity = 1.0 / n_defect_hold if n_defect_hold else 1.0
    far_wilson_upper = _wilson_upper(hold["fn"], n_defect_hold)

    # Confidence width from full-set class separation; a degenerate model must fail loudly,
    # not silently clamp to a tiny width that makes every call ~100% confident.
    normal_med = float(np.median(scores[labels == 0]))
    defect_med = float(np.median(scores[labels == 1]))
    separation = defect_med - normal_med
    if separation <= 0:
        raise ValueError(
            f"Degenerate model: defect median score ({defect_med:.4f}) <= normal median ({normal_med:.4f}). "
            "Perception has not separated the classes; refusing to write a misleading calibration."
        )
    confidence_width = max(separation / 6.0, 1e-6)

    budget_met = bool(hold["far"] <= settings.far_target)
    budget_met_within_granularity = bool(hold["far"] <= settings.far_target + far_granularity)

    metrics = {
        "category": settings.category,
        "seed": settings.seed,
        "n_images": int(len(paths)),
        "n_normal": int(np.sum(labels == 0)),
        "n_defect": int(np.sum(labels == 1)),
        "image_auroc": round(auroc_full, 4),
        "auroc_holdout": round(auroc_hold, 4),
        "published_auroc_baseline": PUBLISHED_AUROC_BASELINE.get(settings.category),
        "operating_threshold": round(t_star, 6),
        "confidence_width": round(confidence_width, 6),
        "far_target": settings.far_target,
        "calibration_split": {
            "n_normal": int(np.sum(labels[cal_mask] == 0)),
            "n_defect": int(np.sum(labels[cal_mask] == 1)),
        },
        "holdout": {
            "n_normal": int(np.sum(labels[hold_mask] == 0)),
            "n_defect": n_defect_hold,
            "far": round(hold["far"], 4),
            "frr": round(hold["frr"], 4),
            "precision": round(hold["precision"], 4),
            "recall": round(hold["recall"], 4),
            "missed_defects": hold["fn"],
            "far_granularity": round(far_granularity, 4),
            "far_wilson_upper95": round(far_wilson_upper, 4),
        },
        "in_sample": {  # threshold fit AND scored on the full set — optimistic, for reference only
            "far": round(in_sample["far"], 4),
            "frr": round(in_sample["frr"], 4),
        },
        "far_budget_met": budget_met,
        "far_budget_met_within_granularity": budget_met_within_granularity,
        "degenerate_separation": False,
        "methodology_note": (
            "Operating threshold calibrated on a seeded, stratified calibration split; FAR/FRR reported on the "
            "disjoint holdout. AUROC is threshold-free. With few defect samples the FAR resolution is 1/n_defect, "
            "so read 'far' alongside 'far_granularity' and 'far_wilson_upper95'."
        ),
    }

    settings.perception_dir.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.write_text(json.dumps(metrics, indent=2))
    _save_tradeoff_plot(scores, labels, t_star, settings.far_target, settings.perception_dir / "threshold_tradeoff.png")

    print("[eval] Perception metrics:")
    print(json.dumps(metrics, indent=2))
    print(
        f"[eval] AUROC {metrics['image_auroc']} (baseline {metrics['published_auroc_baseline']}); "
        f"holdout FAR {hold['far']:.1%} (±granularity {far_granularity:.1%}, Wilson-upper {far_wilson_upper:.1%}); "
        f"budget<=2%: {budget_met} (within granularity: {budget_met_within_granularity}); holdout FRR {hold['frr']:.1%}."
    )
    return metrics


if __name__ == "__main__":
    evaluate()
