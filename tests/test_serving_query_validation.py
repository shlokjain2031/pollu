import sys
from pathlib import Path
from datetime import datetime, timezone

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import pytest
from serving.app import create_app
from serving.stores import InMemoryManifestStore, InMemoryIntentStore
from serving.types import CanonicalIdentity, ManifestRow
from serving.query_descriptors import DESCRIPTORS


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


class RecordingLoader:
    def __init__(self):
        self.called = False

    def query(self, manifest_row, descriptor, params):
        self.called = True
        return b"ok"


def test_unsupported_query_kind_returns_400_and_loader_not_called(tmp_path):
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(tmp_path / "f.bin"))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = RecordingLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    resp = client.get("/query", query_string={"query_kind": "UNKNOWN", "artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version})
    assert resp.status_code == 400
    assert not loader.called


@pytest.mark.parametrize("kind,desc", list(DESCRIPTORS.items()))
def test_missing_required_parameter_results_in_400(kind, desc, tmp_path):
    # Build identity + manifest
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(tmp_path / "f.bin"))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = RecordingLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    # construct query with all required params, then omit one
    qs = {"query_kind": kind, "artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version}
    for p in desc.params:
        qs[p.name] = "x"

    # remove one required
    reqs = [p for p in desc.params if p.required]
    if not reqs:
        pytest.skip("descriptor has no required params")
    omit = reqs[0].name
    qs.pop(omit)

    resp = client.get("/query", query_string=qs)
    assert resp.status_code == 400
    assert not loader.called


def test_extra_undeclared_parameters_are_ignored_consistently(tmp_path):
    # Our current serving behavior ignores extra parameters; test that loader is invoked
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(tmp_path / "f.bin"))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = RecordingLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    # pick any descriptor
    kind = next(iter(DESCRIPTORS.keys()))
    qs = {"query_kind": kind, "artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version}
    # add declared params with dummy values
    for p in DESCRIPTORS[kind].params:
        qs[p.name] = "x"
    # add extra
    qs["unexpected_param"] = "surprise"

    resp = client.get("/query", query_string=qs)
    # Serving is allowed to either reject unknown/extra parameters (400) or
    # ignore them consistently. Accept both behaviors but assert loader
    # invocation only happens when the request succeeds.
    if resp.status_code == 200:
        assert loader.called
    else:
        assert resp.status_code == 400
        assert not loader.called


def test_parameter_bounds_enforced_prevent_loader_call(tmp_path):
    # This test asserts that bounds (e.g. max_time_window_days) are enforced by serving.
    # If the implementation does not enforce them, this test will fail loudly.
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(tmp_path / "f.bin"))
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    loader = RecordingLoader()
    app = create_app(manifest_store=manifest, intent_store=intents, loader=loader)
    client = app.test_client()

    # choose a descriptor with safety_limits.max_time_window_days if any
    candidate = None
    for k, d in DESCRIPTORS.items():
        limits = d.safety_limits or {}
        if "max_time_window_days" in limits:
            candidate = (k, d)
            break
    if candidate is None:
        pytest.skip("no descriptor with max_time_window_days to test")

    kind, desc = candidate
    # construct query with start/end exceeding max_time_window_days
    qs = {"query_kind": kind, "artifact_type": identity.artifact_type, "spatial_id": identity.spatial_id, "timestamp": identity.timestamp, "version": identity.version}
    # supply params; use start far earlier than end to exceed window
    qs["start"] = "2000-01-01T00:00:00Z"
    qs["end"] = "2025-12-20T00:00:00Z"
    # include other required params if any
    for p in desc.params:
        if p.name not in qs:
            qs[p.name] = "x"

    resp = client.get("/query", query_string=qs)
    # serving SHOULD reject this with 400 or 422; if it doesn't, fail loudly
    assert resp.status_code in (400, 422), f"Bounds not enforced; status={resp.status_code}"
    assert not loader.called
