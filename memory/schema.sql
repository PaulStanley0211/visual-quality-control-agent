-- Long-term memory: a synthetic-but-realistic Manufacturing Execution System (MES).
-- The agent reads part/machine/batch/operator history to judge systematic-vs-random,
-- and writes every inspection + any corrective records here as the audit trail.

CREATE TABLE IF NOT EXISTS machines (
    machine_id       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    last_maintenance TEXT,                       -- ISO date; stale => maintenance overdue
    status           TEXT NOT NULL DEFAULT 'ok'  -- 'ok' | 'flagged'
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id     TEXT PRIMARY KEY,
    material_lot TEXT NOT NULL,
    started_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operators (
    operator_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    shift       TEXT NOT NULL                     -- 'day' | 'night'
);

CREATE TABLE IF NOT EXISTS parts (
    part_id     TEXT PRIMARY KEY,
    part_type   TEXT NOT NULL,
    machine_id  TEXT NOT NULL REFERENCES machines(machine_id),
    batch_id    TEXT NOT NULL REFERENCES batches(batch_id),
    operator_id TEXT NOT NULL REFERENCES operators(operator_id),
    produced_at TEXT NOT NULL                     -- ISO timestamp
);

-- Full audit trail: one row per inspection, including the agent's reasoning + actions.
-- `source` separates established production-QC history ('qc') from the agent's own
-- decision audit rows ('agent'). Only 'qc' rows inform the systematic-vs-random defect
-- rate, so the agent's own dispositions can never feed back into its own classification.
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id       TEXT NOT NULL REFERENCES parts(part_id),
    ts            TEXT NOT NULL,
    is_defective  INTEGER NOT NULL,               -- 0 | 1
    confidence    REAL,
    anomaly_score REAL,
    drift_score   REAL,                           -- input-distribution drift score (NULL if not assessed)
    defect_type   TEXT,
    disposition   TEXT,                            -- 'pass' | 'rework' | 'reject'
    fault_pattern TEXT,                            -- 'random' | 'systematic'
    escalated     INTEGER NOT NULL DEFAULT 0,
    reasoning     TEXT,
    actions_json  TEXT,
    source        TEXT NOT NULL DEFAULT 'agent'    -- 'qc' (production history) | 'agent' (decision audit)
);

CREATE TABLE IF NOT EXISTS ncr (
    ncr_id      TEXT PRIMARY KEY,
    part_id     TEXT NOT NULL REFERENCES parts(part_id),
    ts          TEXT NOT NULL,
    defect_type TEXT,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS capa (
    capa_id    TEXT PRIMARY KEY,
    machine_id TEXT NOT NULL REFERENCES machines(machine_id),
    ts         TEXT NOT NULL,
    reason     TEXT,
    detail     TEXT
);

CREATE INDEX IF NOT EXISTS idx_parts_machine     ON parts(machine_id);
CREATE INDEX IF NOT EXISTS idx_parts_batch       ON parts(batch_id);
CREATE INDEX IF NOT EXISTS idx_parts_operator    ON parts(operator_id);
CREATE INDEX IF NOT EXISTS idx_inspections_part  ON inspections(part_id);
CREATE INDEX IF NOT EXISTS idx_inspections_ts    ON inspections(ts);
