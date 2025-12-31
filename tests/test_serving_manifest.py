import json
import pytest
import sys
from pathlib import Path
import pprint


def _make_key():
    return {"row_idx": 1, "timestamp": "2025-12-20", "artifact_type": "grid_30m", "version": "vtest"}


def test_serving_manifest_states(monkeypatch, tmp_path):
    # Run entire test with tmp_path as CWD so Manifest uses manifest/manifest.db there
    monkeypatch.chdir(tmp_path)

    # Ensure repo root is on sys.path so imports like `manifest.db` resolve
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    # Import here so modules pick up monkeypatched CWD
    from manifest.db import Manifest
    from scripts import fake_producer, fake_consumer
    from scripts.serve_mock import app as flask_app

    client = flask_app.test_client()
    key = _make_key()

    # 1) Missing => 404
    r = client.get(
        "/artifact",
        query_string={
            "row_idx": key["row_idx"],
            "timestamp": key["timestamp"],
            "artifact_type": key["artifact_type"],
            "version": key["version"],
        },
    )
    print("STEP 1: Missing")
    print("  response:", r.status_code)
    print("  body:\n", pprint.pformat(r.get_json(), indent=4))
    assert r.status_code == 404
    body = r.get_json()
    assert body["status"] == "missing"

    # 2) Producing => 202
    m = Manifest()
    m.mark_producing(key, "req-1")
    row = m.get(key)
    print("Marked producing in manifest:")
    print(pprint.pformat(row, indent=4))
    r = client.get(
        "/artifact",
        query_string={
            "row_idx": key["row_idx"],
            "timestamp": key["timestamp"],
            "artifact_type": key["artifact_type"],
            "version": key["version"],
        },
    )
    print("STEP 2: Producing")
    print("  response:", r.status_code)
    print("  body:\n", pprint.pformat(r.get_json(), indent=4))
    assert r.status_code == 202
    body = r.get_json()
    assert body["status"] == "producing"

    # 3) Error state => 503
    # mark_error requires existing producing row; reuse manifest
    m.mark_error(key, "simulated failure")
    row = m.get(key)
    print("Marked error in manifest:")
    print(pprint.pformat(row, indent=4))
    r = client.get(
        "/artifact",
        query_string={
            "row_idx": key["row_idx"],
            "timestamp": key["timestamp"],
            "artifact_type": key["artifact_type"],
            "version": key["version"],
        },
    )
    print("STEP 3: Error")
    print("  response:", r.status_code)
    print("  body:\n", pprint.pformat(r.get_json(), indent=4))
    assert r.status_code == 503
    body = r.get_json()
    assert body["status"] == "error"

    # 4) OK state via fake_producer => 200 and data present
    # For a clean run, create a fresh key with different row_idx
    key2 = {"row_idx": 2, "timestamp": "2025-12-20", "artifact_type": "grid_30m", "version": "vtest"}
    res = fake_producer.produce(key2)
    print("Producer result:")
    print(pprint.pformat(res, indent=4))
    assert res["status"] == "ok"

    row2 = Manifest().get(key2)
    print("Manifest row for produced artifact:")
    print(pprint.pformat(row2, indent=4))

    r = client.get(
        "/artifact",
        query_string={
            "row_idx": key2["row_idx"],
            "timestamp": key2["timestamp"],
            "artifact_type": key2["artifact_type"],
            "version": key2["version"],
        },
    )
    print("STEP 4: OK")
    print("  response:", r.status_code)
    assert r.status_code == 200
    body = r.get_json()
    print("Response body:\n", pprint.pformat(body, indent=4))
    assert body["status"] == "ok"
    data = body.get("data")
    assert isinstance(data, list)
    # fake_producer writes a single-row with value 42
    assert any(row.get("value") == 42 for row in data)
