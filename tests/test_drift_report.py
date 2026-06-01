"""MES drift_score write path + (Task 11) population report."""
from __future__ import annotations

from memory import mes
from memory import seed as seed_module


def test_record_inspection_persists_drift_score(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)
    conn = mes.connect(db)
    try:
        mes.record_inspection(
            conn,
            part_id="SCN-GOOD-1",
            is_defective=False,
            confidence=0.9,
            anomaly_score=0.1,
            defect_type=None,
            disposition="pass",
            fault_pattern=None,
            escalated=False,
            reasoning="ok",
            actions={},
            drift_score=1.234,
            source="agent",
        )
        row = conn.execute(
            "SELECT drift_score FROM inspections WHERE part_id='SCN-GOOD-1' ORDER BY inspection_id DESC LIMIT 1"
        ).fetchone()
        assert row["drift_score"] == 1.234
    finally:
        conn.close()


def test_migrate_adds_drift_score_to_legacy_table(tmp_path):
    db = tmp_path / "legacy.db"
    conn = mes.connect(db)
    try:
        # Simulate a pre-drift inspections table (no drift_score column).
        conn.execute(
            "CREATE TABLE inspections (inspection_id INTEGER PRIMARY KEY AUTOINCREMENT, part_id TEXT, source TEXT)"
        )
        conn.commit()
        mes._migrate(conn)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(inspections)").fetchall()}
        assert "drift_score" in cols
    finally:
        conn.close()


from drift.report import population_report

_METRICS = {
    "category": "bottle",
    "operating_threshold": 1.0,
    "psi_reference": {
        # Two bins split at the threshold; reference is mostly in-distribution (low scores).
        "bin_edges": [0.0, 1.0, 100.0],
        "expected_props": [0.9, 0.1],
    },
}


def _insert(conn, part_id, score):
    conn.execute(
        """INSERT INTO inspections (part_id, ts, is_defective, drift_score, source)
           VALUES (?, '2026-06-01T00:00:00+00:00', 0, ?, 'agent')""",
        (part_id, score),
    )


def test_population_report_flags_significant_drift(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)
    conn = mes.connect(db)
    try:
        # All recent processed images are high-score (drifted) — far from the reference's 90/10 split.
        for i in range(20):
            _insert(conn, "SCN-GOOD-1", 5.0)
        conn.commit()
        report = population_report(conn, window=20, metrics=_METRICS)
        assert report["n"] == 20
        assert report["frac_ood"] == 1.0           # every score >= threshold 1.0
        assert report["psi"] > 0.25                 # large divergence
        assert report["band"] == "significant"
    finally:
        conn.close()


def test_population_report_empty_when_no_drift_scores(tmp_path):
    db = tmp_path / "mes.db"
    seed_module.seed(db, verbose=False)   # seed rows are 'qc' with NULL drift_score
    conn = mes.connect(db)
    try:
        report = population_report(conn, window=50, metrics=_METRICS)
        assert report["n"] == 0
        assert report["band"] == "no-data"
    finally:
        conn.close()
