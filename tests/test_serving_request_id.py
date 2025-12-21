import sys
from pathlib import Path
import uuid


# ensure repo root on sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from serving.app import create_app


def test_request_id_auto_generated_and_unique():
    app = create_app()
    client = app.test_client()
    qs = {
        "artifact_type": "grid_30m",
        "spatial_id": "mumbai",
        "timestamp": "2025-12-20",
        "version": "vtest",
    }
    r1 = client.get("/artifact", query_string=qs)
    r2 = client.get("/artifact", query_string=qs)
    rid1 = r1.headers.get("X-Request-ID")
    rid2 = r2.headers.get("X-Request-ID")
    assert rid1 is not None and rid2 is not None
    assert rid1 != rid2


def test_client_request_id_echoed():
    app = create_app()
    client = app.test_client()
    client_id = "client-12345"
    qs = {
        "artifact_type": "grid_30m",
        "spatial_id": "mumbai",
        "timestamp": "2025-12-20",
        "version": "vtest",
    }
    r = client.get("/artifact", query_string=qs, headers={"X-Client-Request-ID": client_id})
    assert r.headers.get("X-Request-ID") == client_id
