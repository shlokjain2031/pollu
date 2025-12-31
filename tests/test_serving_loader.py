"""
tests/test_serving_loader.py

This test file verifies the Stage 2 serving/data boundary wiring.

What we test here:
- Seed the in-memory manifest with a `ManifestRow` whose `state` is "ok".
- Issue an HTTP GET to the `/artifact` endpoint for that identity.
- Assert that the serving decision allows serving (HTTP 200) and that the
  `ArtifactLoader` (the placeholder implementation) is invoked and its raw
  bytes are returned as the response body.

Notes:
- This is intentionally lightweight: the loader returns static bytes and the
  test does not touch DuckDB, parquet, or any real artifact formats.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure repo root is on sys.path so imports like `serving` resolve
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from serving.types import CanonicalIdentity, ManifestRow
from serving.stores import InMemoryManifestStore, InMemoryIntentStore
from serving.app import create_app


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_manifest_row(identity: CanonicalIdentity, state: str, path: str | None = None) -> ManifestRow:
    mid = f"{identity.artifact_type}::{identity.spatial_id}::{identity.timestamp}::{identity.version}"
    return ManifestRow(
        manifest_id=mid,
        state=state,
        artifact_path=path,
        request_id=None,
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )


def test_loader_returns_placeholder_bytes(tmp_path):
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    # create a dummy artifact file (decision logic checks existence)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fake")

    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(artifact))
    print(f"Seeding manifest with manifest_id={mr.manifest_id} state={mr.state} artifact_path={mr.artifact_path}")
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    app = create_app(manifest_store=manifest, intent_store=intents)
    client = app.test_client()

    print("Issuing GET /artifact for identity:", identity.__dict__)
    resp = client.get(
        "/artifact",
        query_string={
            "artifact_type": identity.artifact_type,
            "spatial_id": identity.spatial_id,
            "timestamp": identity.timestamp,
            "version": identity.version,
        },
    )

    print(f"Response status: {resp.status_code}")
    print(f"Response headers: {dict(resp.headers)}")
    print(f"Response body (len): {len(resp.data)} bytes")

    assert resp.status_code == 200
    # Stage 3: serving should return the raw bytes from the file on disk
    assert resp.data == b"fake"
    assert resp.headers.get("X-Manifest-ID") == mr.manifest_id
