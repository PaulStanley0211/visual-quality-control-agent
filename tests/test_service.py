"""FastAPI service tests (skipped unless the trained model + seeded MES are present)."""
from __future__ import annotations

import pytest

from config import settings

pytestmark = pytest.mark.service


def _ready() -> bool:
    try:
        from perception.detector import find_exported_model

        find_exported_model()
    except Exception:
        return False
    return settings.metrics_path.exists() and settings.mes_db_path.exists()


requires = pytest.mark.skipif(not _ready(), reason="needs trained model + seeded MES (perception.train + memory.seed)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from service.app import app

    with TestClient(app) as c:
        yield c


def _defect_image():
    base = settings.data_root / "MVTecAD" / settings.category / "test"
    return sorted((base / "broken_large").glob("*.png"))[0]


@requires
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@requires
def test_inspect_systematic_defect(client):
    with open(_defect_image(), "rb") as f:
        r = client.post("/inspect", data={"part_id": "SCN-SYSMACH-1"}, files={"image": ("p.png", f, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"]["disposition"] == "reject"
    assert body["diagnosis"]["fault_pattern"] == "systematic"
    assert body["actions"]["ncr"] and body["actions"]["capa"] and body["actions"]["machine_flag"]


@requires
def test_inspect_unknown_part_returns_404(client):
    with open(_defect_image(), "rb") as f:
        r = client.post("/inspect", data={"part_id": "NOPE"}, files={"image": ("p.png", f, "image/png")})
    assert r.status_code == 404


@requires
def test_health_exposes_drift_fields(client):
    body = client.get("/health").json()
    assert "drift_enabled" in body
    assert "drift_reference_present" in body


@requires
def test_drift_report_endpoint(client):
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "band" in body and "n" in body
