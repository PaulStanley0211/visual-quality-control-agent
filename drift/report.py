"""Population-level drift monitor: windowed PSI + %OOD over the MES.

Reads the most recent non-defective inspections that carry a drift score (the good-part stream;
synthetic 'qc' seed rows have NULL drift_score and are excluded). Defective parts sit far from
the training-good manifold by construction, so including their scores would let a spike of
genuine defects masquerade as input drift. The PSI reference is built from clean good images,
so the live good stream is the apples-to-apples comparison.

Run:  uv run python -m drift.report
"""
from __future__ import annotations

import json

import numpy as np

from config import settings
from drift.scoring import population_stability_index


def _recent_scores(conn, window: int) -> list[float]:
    rows = conn.execute(
        """
        SELECT drift_score FROM inspections
        WHERE drift_score IS NOT NULL AND is_defective = 0
        ORDER BY ts DESC, inspection_id DESC
        LIMIT ?
        """,
        (window,),
    ).fetchall()
    return [float(r["drift_score"]) for r in rows]


def _band(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < settings.drift_psi_significant:
        return "moderate"
    return "significant"


def population_report(conn, window: int | None = None, metrics: dict | None = None) -> dict:
    """Compute the windowed drift report. ``metrics`` defaults to the on-disk drift_metrics.json."""
    window = window or settings.drift_window
    if metrics is None:
        metrics = json.loads(settings.drift_metrics_path.read_text())
    ref = metrics["psi_reference"]
    bin_edges = np.asarray(ref["bin_edges"], dtype=float)
    expected = np.asarray(ref["expected_props"], dtype=float)
    threshold = float(metrics["operating_threshold"])

    scores = _recent_scores(conn, window)
    if not scores:
        return {"n": 0, "frac_ood": 0.0, "psi": 0.0, "band": "no-data",
                "message": "No drift-scored inspections yet — process some images first."}

    arr = np.asarray(scores, dtype=float)
    counts, _ = np.histogram(arr, bins=bin_edges)
    actual = counts / counts.sum()
    psi = population_stability_index(expected, actual)
    frac_ood = float(np.mean(arr >= threshold))
    band = _band(psi)
    message = (
        f"Line drift: PSI {psi:.2f} — {band.upper()}; {int(np.sum(arr >= threshold))}/{len(scores)} parts OOD."
    )
    return {"n": len(scores), "frac_ood": round(frac_ood, 4), "psi": round(psi, 4), "band": band, "message": message}


def main() -> None:
    from memory import mes
    from memory import seed as seed_module

    seed_module.ensure_seeded()
    conn = mes.connect()
    try:
        report = population_report(conn)
    finally:
        conn.close()
    print(json.dumps(report, indent=2))
    print(f"[drift.report] {report['message'] if report['n'] else report.get('message', 'no data')}")


if __name__ == "__main__":
    main()
