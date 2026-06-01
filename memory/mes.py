"""Long-term memory: query + write interface over the SQLite MES.

This is the seam designed to mirror a real MES API. The agent reads
``get_part_context`` (long-term history → systematic-vs-random signal) and writes
back the inspection audit trail plus any NCR / CAPA / machine-flag actions.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import settings

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_DIMENSIONS = {"machine_id", "batch_id", "operator_id"}


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or settings.mes_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # concurrent readers + a writer
    conn.execute("PRAGMA busy_timeout = 5000")  # wait, don't immediately error, on lock contention
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _next_id(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    # MAX(rowid)+1 is monotonic for these append-only audit tables (unlike COUNT(*)+1,
    # which collides after a delete). Generated inside the caller's transaction.
    n = conn.execute(f"SELECT COALESCE(MAX(rowid), 0) + 1 AS c FROM {table}").fetchone()["c"]  # noqa: S608 - literal table
    return f"{prefix}-{n:05d}"


def recent_defect_rate(conn: sqlite3.Connection, column: str, value: str, window: int) -> dict:
    """Defect rate over the most recent ``window`` PRODUCTION-QC inspections for parts where
    parts.<column> = value.

    Only ``source = 'qc'`` rows count: the established production history. The agent's own
    decision rows (``source = 'agent'``) are excluded so its dispositions can never feed back
    into its own random-vs-systematic classification. Seed writes one 'qc' row per part, so the
    window is effectively over recent distinct parts.
    """
    if column not in _DIMENSIONS:
        raise ValueError(f"Unsupported dimension column: {column!r}")
    rows = conn.execute(
        f"""
        SELECT i.is_defective AS d
        FROM inspections i JOIN parts p ON p.part_id = i.part_id
        WHERE p.{column} = ? AND i.source = 'qc'
        ORDER BY i.ts DESC, i.inspection_id DESC
        LIMIT ?
        """,  # noqa: S608 - column validated against _DIMENSIONS above
        (value, window),
    ).fetchall()
    n = len(rows)
    n_def = sum(r["d"] for r in rows)
    return {"rate": (n_def / n) if n else 0.0, "n_defect": n_def, "n_total": n}


def get_part_context(part_id: str, conn: sqlite3.Connection | None = None, window: int | None = None) -> dict:
    """Resolve a part's machine / batch / operator and their recent defect-rate history.

    The operator defect rate is surfaced for narrative context only — by design it does NOT
    drive the systematic-vs-random decision (which keys off machine and batch). Operator-driven
    defects are rarer and harder to attribute, so they are reported, not acted on automatically.
    """
    own = conn is None
    conn = conn or connect()
    window = window or settings.history_window
    try:
        part = conn.execute("SELECT * FROM parts WHERE part_id = ?", (part_id,)).fetchone()
        if part is None:
            raise KeyError(f"Part '{part_id}' not found in MES. Seed the database with `python -m memory.seed`.")
        machine = conn.execute("SELECT * FROM machines WHERE machine_id = ?", (part["machine_id"],)).fetchone()
        batch = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (part["batch_id"],)).fetchone()
        operator = conn.execute("SELECT * FROM operators WHERE operator_id = ?", (part["operator_id"],)).fetchone()
        for label, row, ref in (("machine", machine, part["machine_id"]), ("batch", batch, part["batch_id"]),
                                 ("operator", operator, part["operator_id"])):
            if row is None:
                raise KeyError(f"Part '{part_id}' references {label} '{ref}' which is missing from the MES.")
        return {
            "part_id": part_id,
            "part_type": part["part_type"],
            "machine": {
                "id": machine["machine_id"],
                "name": machine["name"],
                "status": machine["status"],
                "last_maintenance": machine["last_maintenance"],
                **recent_defect_rate(conn, "machine_id", part["machine_id"], window),
            },
            "batch": {
                "id": batch["batch_id"],
                "material_lot": batch["material_lot"],
                **recent_defect_rate(conn, "batch_id", part["batch_id"], window),
            },
            "operator": {
                "id": operator["operator_id"],
                "name": operator["name"],
                "shift": operator["shift"],
                **recent_defect_rate(conn, "operator_id", part["operator_id"], window),
            },
        }
    finally:
        if own:
            conn.close()


def record_inspection(
    conn: sqlite3.Connection,
    *,
    part_id: str,
    is_defective: bool,
    confidence: float | None,
    anomaly_score: float | None,
    defect_type: str | None,
    disposition: str | None,
    fault_pattern: str | None,
    escalated: bool,
    reasoning: str | None,
    actions: dict,
    source: str = "agent",
    commit: bool = True,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO inspections
          (part_id, ts, is_defective, confidence, anomaly_score, defect_type,
           disposition, fault_pattern, escalated, reasoning, actions_json, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            part_id, _now_iso(), int(is_defective), confidence, anomaly_score, defect_type,
            disposition, fault_pattern, int(escalated), reasoning, json.dumps(actions), source,
        ),
    )
    if commit:
        conn.commit()
    return int(cur.lastrowid)


def write_ncr(conn: sqlite3.Connection, part_id: str, defect_type: str | None, detail: str, commit: bool = True) -> str:
    ncr_id = _next_id(conn, "ncr", "NCR")
    conn.execute(
        "INSERT INTO ncr (ncr_id, part_id, ts, defect_type, detail) VALUES (?, ?, ?, ?, ?)",
        (ncr_id, part_id, _now_iso(), defect_type, detail),
    )
    if commit:
        conn.commit()
    return ncr_id


def create_capa(conn: sqlite3.Connection, machine_id: str, reason: str, detail: str, commit: bool = True) -> str:
    capa_id = _next_id(conn, "capa", "CAPA")
    conn.execute(
        "INSERT INTO capa (capa_id, machine_id, ts, reason, detail) VALUES (?, ?, ?, ?, ?)",
        (capa_id, machine_id, _now_iso(), reason, detail),
    )
    if commit:
        conn.commit()
    return capa_id


def flag_machine(conn: sqlite3.Connection, machine_id: str, commit: bool = True) -> None:
    conn.execute("UPDATE machines SET status = 'flagged' WHERE machine_id = ?", (machine_id,))
    if commit:
        conn.commit()
