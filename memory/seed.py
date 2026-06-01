"""Seed the synthetic MES with a realistic history.

The history is engineered so systematic-vs-random has a clear ground truth:
  * Machine M2 ("Filler-B") is overdue for maintenance and runs a high defect rate
    -> a part on M2 is a *systematic (machine)* case.
  * Batch B2 ("lot-CONTAM") is a bad material lot with a high defect rate, run on an
    otherwise-good machine -> a part in B2 is a *systematic (batch)* case, while that
    machine's own rate stays below the threshold (the batch is the isolated cause).
  * Good machines + good batches run a low baseline defect rate -> *random* cases.

A handful of named SCN-* parts (no prior inspection) are added for the scenario eval.

Run:  uv run python -m memory.seed
"""
from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta

from config import settings
from memory import mes

BASE_TIME = datetime(2026, 5, 1, 8, 0, 0)
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]

MACHINES = [
    ("M1", "Filler-A", "2026-05-20", "ok"),   # good, recently maintained
    ("M2", "Filler-B", "2026-01-10", "ok"),   # OVERDUE maintenance -> systematic machine
    ("M3", "Capper-A", "2026-05-18", "ok"),   # good
    ("M4", "Filler-C", "2026-05-15", "ok"),   # near-threshold: exactly 30% (ambiguous boundary)
]
BATCHES = [
    ("B1", "lot-OK-1", "2026-04-28"),
    ("B2", "lot-CONTAM", "2026-04-29"),        # bad material lot -> systematic batch
    ("B3", "lot-OK-2", "2026-04-30"),
    ("B4", "lot-OK-3", "2026-05-02"),          # carries M4's near-threshold parts
]
OPERATORS = [
    ("O1", "Alice", "day"),
    ("O2", "Bob", "night"),
    ("O3", "Carol", "day"),
]

# (machine, batch, count, defect_probability) — see module docstring for the rationale.
BACKGROUND_TRACKS = [
    ("M1", "B1", 40, 0.05),   # baseline
    ("M3", "B1", 30, 0.05),   # baseline
    ("M2", "B3", 24, 0.55),   # systematic machine (M2 overdue), isolated on its own batch B3
    ("M1", "B2", 16, 0.55),   # systematic batch (B2 bad lot) on good machine M1 (dilutes M1's rate)
]

# Deterministic near-threshold track: exactly DEFECTS of COUNT defective so the recent rate
# equals the systematic threshold exactly (ambiguous boundary -> low diagnosis confidence).
NEAR_TRACK = {"machine": "M4", "batch": "B4", "count": 20, "defects": 6}  # 6/20 = 0.30 == threshold

# Named parts for the scenario eval (machine, batch) -> expected ground truth in eval/scenarios.yaml.
SCENARIO_PARTS = [
    ("SCN-RANDOM-1", "M3", "B1"),    # good machine + good batch -> random
    ("SCN-RANDOM-2", "M1", "B1"),    # random
    ("SCN-SYSMACH-1", "M2", "B1"),   # overdue machine -> systematic (machine)
    ("SCN-SYSMACH-2", "M2", "B1"),   # systematic (machine)
    ("SCN-SYSBATCH-1", "M3", "B2"),  # bad lot on good machine -> systematic (batch)
    ("SCN-SYSBATCH-2", "M3", "B2"),  # systematic (batch)
    ("SCN-BOTH-1", "M2", "B2"),      # overdue machine AND bad lot -> systematic (both drivers)
    ("SCN-NEAR-1", "M4", "B1"),      # machine rate == threshold -> systematic but ambiguous -> escalate
    ("SCN-GOOD-1", "M3", "B1"),      # used for a pass (no-defect) scenario
    ("SCN-GOOD-2", "M1", "B1"),      # pass
]


def ensure_seeded(db_path=None) -> None:
    """Seed only if the MES is missing or has no parts yet (idempotent, safe to call on every startup)."""
    path = db_path or settings.mes_db_path
    if not path.exists():
        seed(db_path, verbose=False)
        return
    conn = mes.connect(path)
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM parts").fetchone()["c"]
    except sqlite3.OperationalError:  # 'no such table: parts' — empty/fresh DB file
        n = 0
    finally:
        conn.close()
    if n == 0:
        seed(db_path, verbose=False)
    else:
        # Already seeded: ensure the schema is current (additive migrations only).
        conn = mes.connect(path)
        try:
            with conn:
                mes._migrate(conn)
        finally:
            conn.close()


def _insert_history(conn, rng, op_ids, idx: int, machine: str, batch: str, is_defective: bool) -> None:
    """Insert one production part + its 'qc' (ground-truth) inspection row."""
    produced = BASE_TIME + timedelta(minutes=5 * idx)
    part_id = f"P{idx + 1:04d}"
    conn.execute(
        "INSERT INTO parts VALUES (?,?,?,?,?,?)",
        (part_id, "bottle", machine, batch, op_ids[idx % len(op_ids)], produced.isoformat(timespec="seconds")),
    )
    defect_type = rng.choice(DEFECT_TYPES) if is_defective else None
    conn.execute(
        """INSERT INTO inspections
           (part_id, ts, is_defective, confidence, anomaly_score, defect_type,
            disposition, fault_pattern, escalated, reasoning, actions_json, source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            part_id,
            (produced + timedelta(minutes=1)).isoformat(timespec="seconds"),
            int(is_defective),
            round(rng.uniform(0.75, 0.99) if is_defective else rng.uniform(0.55, 0.85), 3),
            None,
            defect_type,
            ("reject" if is_defective else "pass"),
            None,
            0,
            None,
            "{}",
            "qc",  # production ground-truth history (drives the defect-rate signal)
        ),
    )


def seed(db_path=None, verbose: bool = True) -> None:
    rng = random.Random(settings.seed)
    conn = mes.connect(db_path)
    try:
        # Fresh start every seed for reproducibility.
        for table in ("inspections", "ncr", "capa", "parts", "machines", "batches", "operators"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        mes.init_db(conn)

        conn.executemany("INSERT INTO machines VALUES (?,?,?,?)", MACHINES)
        conn.executemany("INSERT INTO batches VALUES (?,?,?)", BATCHES)
        conn.executemany("INSERT INTO operators VALUES (?,?,?)", OPERATORS)

        op_ids = [o[0] for o in OPERATORS]

        # Build background part specs, then shuffle so each dimension is interleaved in time
        # (keeps the recent-window defect rate representative rather than clustered).
        specs: list[tuple[str, str, float]] = []
        for machine, batch, count, p in BACKGROUND_TRACKS:
            specs.extend([(machine, batch, p)] * count)
        rng.shuffle(specs)

        idx = 0
        for machine, batch, p in specs:
            _insert_history(conn, rng, op_ids, idx, machine, batch, rng.random() < p)
            idx += 1

        # Deterministic near-threshold track: exactly NEAR_TRACK['defects'] of 'count' defective,
        # so the recent rate equals the systematic threshold exactly (ambiguous boundary).
        flags = [j < NEAR_TRACK["defects"] for j in range(NEAR_TRACK["count"])]
        rng.shuffle(flags)
        for flag in flags:
            _insert_history(conn, rng, op_ids, idx, NEAR_TRACK["machine"], NEAR_TRACK["batch"], flag)
            idx += 1

        # Named scenario parts (no prior inspection): the part being inspected "now".
        now = BASE_TIME + timedelta(minutes=5 * (idx + 10))
        for j, (part_id, machine, batch) in enumerate(SCENARIO_PARTS):
            conn.execute(
                "INSERT INTO parts VALUES (?,?,?,?,?,?)",
                (part_id, "bottle", machine, batch, op_ids[j % len(op_ids)],
                 (now + timedelta(minutes=j)).isoformat(timespec="seconds")),
            )
        conn.commit()

        if verbose:
            _print_summary(conn)
    finally:
        conn.close()


def _print_summary(conn) -> None:
    n_parts = conn.execute("SELECT COUNT(*) c FROM parts").fetchone()["c"]
    n_insp = conn.execute("SELECT COUNT(*) c FROM inspections").fetchone()["c"]
    print(f"[seed] {n_parts} parts, {n_insp} inspections at {settings.mes_db_path}")
    print("[seed] recent defect rates (window={}):".format(settings.history_window))
    for col, ids in (("machine_id", [m[0] for m in MACHINES]), ("batch_id", [b[0] for b in BATCHES])):
        for v in ids:
            r = mes.recent_defect_rate(conn, col, v, settings.history_window)
            flag = " <- systematic" if r["rate"] >= settings.systematic_defect_rate else ""
            print(f"  {col[:-3]:8} {v}: {r['rate']:.0%}  ({r['n_defect']}/{r['n_total']}){flag}")


if __name__ == "__main__":
    seed()
