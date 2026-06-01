"""FastAPI service exposing the inspection agent.

POST /inspect  (multipart: part_id + image)  -> InspectionOutput JSON
GET  /health                                 -> readiness + active config

The compiled LangGraph (real PatchCore detector + MES) is built once at startup;
the MES is seeded on first run if empty. The confidence threshold and LLM provider
come from `config` (env-overridable), so the service is tunable without code changes.

Run:  uv run uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from agent.graph import build_graph, run_inspection
from agent.state import AgentDeps
from config import settings
from contracts.models import InspectionOutput
from memory import mes
from memory import seed as seed_module

logger = logging.getLogger(__name__)

_ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_module.ensure_seeded()
    app.state.graph = build_graph(AgentDeps())  # detector is lazy; startup stays fast
    yield


app = FastAPI(
    title="Visual Quality Control Agent",
    version="0.1.0",
    description="Autonomous single-station visual QC: PatchCore perception + LangGraph reasoning over an MES.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "category": settings.category,
        "llm_provider": settings.llm_provider,
        "confidence_threshold": settings.confidence_threshold,
        "drift_enabled": settings.drift_enabled,
        "drift_reference_present": settings.drift_reference_path.exists() and settings.drift_metrics_path.exists(),
    }


@app.get("/drift")
def drift() -> dict:
    """Windowed population drift report (PSI + %OOD over recent drift-scored inspections)."""
    if not settings.drift_metrics_path.exists():
        raise HTTPException(status_code=503, detail="Drift monitor not calibrated; run drift.reference + eval.drift_eval.")
    from drift.report import population_report

    conn = mes.connect()
    try:
        return population_report(conn)
    finally:
        conn.close()


# Plain `def` (not async): Starlette runs it in the threadpool, so the blocking CPU inference
# and synchronous SQLite work never stall the event loop / other requests.
@app.post("/inspect", response_model=InspectionOutput)
def inspect(part_id: str = Form(...), image: UploadFile = File(...)) -> InspectionOutput:
    """Inspect one part: validate input, run perception + the agent loop, return the result."""
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type {image.content_type!r}; expected PNG or JPEG.")
    try:
        mes.get_part_context(part_id)  # cheap existence check before doing expensive work
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    suffix = os.path.splitext(image.filename or "")[1].lower()
    if suffix not in _ALLOWED_SUFFIXES:
        suffix = ".png"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := image.file.read(1 << 20):  # bounded 1 MiB chunks, capped total
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_upload_mb} MB limit.")
                out.write(chunk)
        return run_inspection(app.state.graph, part_id, image_path=tmp_path)
    except HTTPException:
        raise
    except FileNotFoundError as e:  # model/calibration not present
        raise HTTPException(status_code=503, detail=f"Perception model not ready: {e}") from e
    except ValueError as e:  # unreadable image / non-finite score
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - sanitize unexpected errors; full detail to server logs
        logger.exception("Inspection failed for part %s", part_id)
        raise HTTPException(status_code=500, detail="Internal inspection error.") from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Could not remove temp file %s", tmp_path)
