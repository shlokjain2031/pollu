import os
import tempfile
from datetime import datetime, timezone
import sys
from pathlib import Path
import pprint


# Ensure repo root is on sys.path so imports like `manifest.db` resolve
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from serving.types import CanonicalIdentity, ManifestRow
from serving.stores import InMemoryManifestStore, InMemoryIntentStore
from serving.decision import decide_request


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_manifest_row(identity: CanonicalIdentity, state: str, path: str | None = None, request_id: str | None = None, error_text: str | None = None) -> ManifestRow:
    mid = f"{identity.artifact_type}::{identity.spatial_id}::{identity.timestamp}::{identity.version}"
    return ManifestRow(
        manifest_id=mid,
        state=state,
        artifact_path=path,
        request_id=request_id,
        created_at=_now_iso(),
        updated_at=_now_iso(),
        error_text=error_text,
    )


def test_decision_ok_path_exists(tmp_path):
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    # create fake artifact file
    artifact = tmp_path / "artifact.parquet"
    artifact.write_text("fake")

    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(artifact), request_id="r1")
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    decision = decide_request(identity, manifest.get(identity), False, intents)
    print("TEST: OK_PATH_EXISTS")
    print("  identity:", identity)
    print("  manifest:\n", pprint.pformat(mr.__dict__, indent=4))
    print("  decision:\n", pprint.pformat({"http_status": decision.http_status, "headers": decision.headers, "body": decision.body}, indent=4))
    assert decision.http_status == 200
    assert decision.body["status"] == "ok"


def test_decision_ok_file_missing(tmp_path):
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "ok", path=str(tmp_path / "nope.parquet"), request_id="r1")
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    decision = decide_request(identity, manifest.get(identity), False, intents)
    print("TEST: OK_FILE_MISSING")
    print("  identity:", identity)
    print("  manifest:\n", pprint.pformat(mr.__dict__, indent=4))
    print("  decision:\n", pprint.pformat({"http_status": decision.http_status, "headers": decision.headers, "body": decision.body}, indent=4))
    assert decision.http_status == 500
    assert decision.body["status"] == "inconsistent"


def test_producing_returns_202():
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "producing", path=None, request_id="r2")
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    decision = decide_request(identity, manifest.get(identity), False, intents)
    print("TEST: PRODUCING")
    print("  identity:", identity)
    print("  manifest:\n", pprint.pformat(mr.__dict__, indent=4))
    print("  decision:\n", pprint.pformat({"http_status": decision.http_status, "headers": decision.headers, "body": decision.body}, indent=4))
    assert decision.http_status == 202
    assert decision.body["status"] == "producing"


def test_error_with_trigger_creates_intent():
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    mr = make_manifest_row(identity, "error", path=None, request_id="r3", error_text="bad")
    manifest.seed(mr)

    intents = InMemoryIntentStore()
    decision = decide_request(identity, manifest.get(identity), True, intents)
    print("TEST: ERROR_WITH_TRIGGER (first)")
    print("  identity:", identity)
    print("  manifest:\n", pprint.pformat(mr.__dict__, indent=4))
    print("  decision:\n", pprint.pformat({"http_status": decision.http_status, "headers": decision.headers, "body": decision.body}, indent=4))
    assert decision.http_status == 202
    assert "intent" in decision.body
    intent_id = decision.body["intent"]["intent_id"]
    # repeated call returns same intent (dedupe)
    decision2 = decide_request(identity, manifest.get(identity), True, intents)
    print("TEST: ERROR_WITH_TRIGGER (second)")
    print("  decision2:\n", pprint.pformat({"http_status": decision2.http_status, "headers": decision2.headers, "body": decision2.body}, indent=4))
    assert decision2.body["intent"]["intent_id"] == intent_id


def test_missing_with_trigger_creates_intent_and_dedupes():
    identity = CanonicalIdentity("landsat_signals", "mumbai", "2025-12-20", "v1")
    manifest = InMemoryManifestStore()
    intents = InMemoryIntentStore()
    decision = decide_request(identity, manifest.get(identity), True, intents)
    print("TEST: MISSING_WITH_TRIGGER (first)")
    print("  identity:", identity)
    print("  decision:\n", pprint.pformat({"http_status": decision.http_status, "headers": decision.headers, "body": decision.body}, indent=4))
    assert decision.http_status == 202
    intent_id = decision.body["intent"]["intent_id"]
    # second call returns same intent
    decision2 = decide_request(identity, manifest.get(identity), True, intents)
    print("TEST: MISSING_WITH_TRIGGER (second)")
    print("  decision2:\n", pprint.pformat({"http_status": decision2.http_status, "headers": decision2.headers, "body": decision2.body}, indent=4))
    assert decision2.body["intent"]["intent_id"] == intent_id
