import sys
from pathlib import Path
from datetime import datetime, timezone

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import duckdb
import pytest

from serving.app import create_app
from serving.stores import InMemoryManifestStore, InMemoryIntentStore
from serving.types import CanonicalIdentity, ManifestRow
from serving.data_loader import DuckDBQueryLoader
from serving.query_descriptors import DESCRIPTORS, QueryKind


FIXTURES_DIR = Path("tests/fixtures")
OBS_PQ = FIXTURES_DIR / "observations.parquet"


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


def _build_duckdb_from_parquet(tmp_path: Path) -> Path:
    # create a duckdb file that reads the parquet as table `observations`
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(database=str(db_path), read_only=False)
    # Install and load spatial extension so spatial functions like ST_Within are available
    try:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
    except Exception:
        # If installation/loading fails, proceed; loader failure will surface in tests
        pass
    # Create observations table converting WKT geom strings into GEOMETRY type
    con.execute(f"CREATE TABLE observations AS SELECT spatial_id, timestamp, value, ST_GeomFromText(geom) AS geom FROM read_parquet('{str(OBS_PQ)}')")
    con.close()
    return db_path


@pytest.mark.parametrize("kind", list(DESCRIPTORS.keys()))
def test_query_kinds_return_expected_results(kind, tmp_path):
    # Build a small duckdb file from the checked-in parquet fixture
    db_path = _build_duckdb_from_parquet(tmp_path)

    identity = CanonicalIdentity("observations", "pt1", "2025-01-02T00:00:00", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(db_path))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = DuckDBQueryLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    # Build a minimal query for each descriptor kind. These are deterministic
    qs = {"artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version, "query_kind": kind.value}
    desc = DESCRIPTORS[kind]
    for p in desc.params:
        if p.name not in qs:
            # supply simple values
            if p.name in ("start", "end", "as_of"):
                qs[p.name] = "2025-01-02T00:00:00"
            elif p.name == "wkt":
                qs[p.name] = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
            else:
                qs[p.name] = "pt1"

    resp = client.get("/query", query_string=qs)

    assert resp.status_code == 200, f"query_kind {kind} failed: {resp.data}\n{resp.status_code}"
    assert resp.headers.get("X-Manifest-ID") == mr.manifest_id
    # Basic assertion: non-empty bytes
    assert isinstance(resp.data, (bytes, bytearray)) and len(resp.data) > 0


def test_time_window_exceed_rejected_before_duckdb(tmp_path):
    db_path = _build_duckdb_from_parquet(tmp_path)
    identity = CanonicalIdentity("observations", "pt1", "2025-01-02T00:00:00", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(db_path))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = DuckDBQueryLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    # Send a request with a time window > 90 days
    qs = {"query_kind": QueryKind.POINT_TIME_RANGE.value, "artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version, "start": "2000-01-01T00:00:00", "end": "2025-01-01T00:00:00"}
    resp = client.get("/query", query_string=qs)
    assert resp.status_code == 422


def test_manifest_ok_but_file_missing_returns_500(tmp_path):
    # manifest points to a non-existent file
    identity = CanonicalIdentity("observations", "pt1", "2025-01-02T00:00:00", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(tmp_path / "nonexistent.duckdb"))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = DuckDBQueryLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    qs = {"query_kind": QueryKind.POINT_LATEST.value, "artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version, "as_of": "2025-01-02T00:00:00"}
    resp = client.get("/query", query_string=qs)
    assert resp.status_code == 500
