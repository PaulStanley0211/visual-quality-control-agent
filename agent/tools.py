"""Agent tools: corrective-action side effects, all writing to the MES audit trail.

Thin wrappers over ``memory.mes`` that also return a human-readable trace line, so the
agent's reasoning trace records exactly which actions fired and the record ids created.
"""
from __future__ import annotations

import sqlite3

from memory import mes


def raise_ncr(conn: sqlite3.Connection, part_id: str, defect_type: str | None, detail: str, commit: bool = True) -> tuple[str, str]:
    ncr_id = mes.write_ncr(conn, part_id, defect_type, detail, commit=commit)
    return ncr_id, f"Raised non-conformance report {ncr_id} for part {part_id}."


def open_capa(conn: sqlite3.Connection, machine_id: str, reason: str, detail: str, commit: bool = True) -> tuple[str, str]:
    capa_id = mes.create_capa(conn, machine_id, reason, detail, commit=commit)
    return capa_id, f"Opened corrective-action ticket {capa_id} for machine {machine_id}."


def flag_machine(conn: sqlite3.Connection, machine_id: str, commit: bool = True) -> str:
    mes.flag_machine(conn, machine_id, commit=commit)
    return f"Flagged machine {machine_id} for attention."
