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
