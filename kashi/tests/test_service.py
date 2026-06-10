"""Kashi service tests: feature determinism, validation, endpoint contracts.

Requires kashi/data/delhi_features.parquet (run kashi.build_features once).
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kashi.config import FEATURES_PARQUET
from kashi.features import FeatureStore
from kashi.service import app

pytestmark = pytest.mark.skipif(
    not FEATURES_PARQUET.exists(), reason="features parquet not built (run kashi.build_features)"
)

# A coordinate inside the Delhi coverage (Connaught Place) and one far outside.
CP = (28.6315, 77.2167)
MUMBAI = (19.0, 72.8)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager triggers startup
        yield c


@pytest.fixture(scope="module")
def store():
    return FeatureStore()


class TestFeatureStore:
    def test_snap_inside(self, store):
        idx, dist = store.snap(np.array([CP[0]]), np.array([CP[1]]))
        assert 0 <= idx[0] < store.point_count and dist[0] < 500

    def test_snap_outside(self, store):
        _, dist = store.snap(np.array([MUMBAI[0]]), np.array([MUMBAI[1]]))
        assert dist[0] > 100_000

    def test_assemble_deterministic(self, store):
        ts = datetime(2026, 6, 9, 14, tzinfo=timezone.utc)
        lats, lons = np.array([CP[0], 28.5]), np.array([CP[1], 77.1])
        assert np.array_equal(store.assemble(lats, lons, ts), store.assemble(lats, lons, ts))

    def test_assemble_varies_with_hour(self, store):
        lats, lons = np.array([CP[0]]), np.array([CP[1]])
        a = store.assemble(lats, lons, datetime(2026, 6, 9, 4, tzinfo=timezone.utc))
        b = store.assemble(lats, lons, datetime(2026, 6, 9, 16, tzinfo=timezone.utc))
        assert not np.array_equal(a, b)

    def test_assemble_no_nan(self, store):
        ts = datetime(2026, 1, 1, 0, tzinfo=timezone.utc)
        idx = np.arange(0, store.point_count, 977)
        X = store.assemble(store.lat[idx], store.lon[idx], ts)
        assert np.isfinite(X).all()


class TestEndpoints:
    def test_inference_success(self, client):
        r = client.post("/inference", json={
            "coords": [list(CP), [28.5, 77.1]],
            "timestamp_hour": "2026-06-09T14:00:00Z",
        })
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 2
        assert all(0 <= v < 1500 for v in preds)

    def test_inference_idempotent(self, client):
        body = {"coords": [list(CP)], "timestamp_hour": "2026-03-15T09:00:00Z"}
        assert client.post("/inference", json=body).json() == client.post("/inference", json=body).json()

    def test_inference_timestamp_normalized(self, client):
        a = client.post("/inference", json={"coords": [list(CP)], "timestamp_hour": "2026-06-09T14:37:00Z"})
        b = client.post("/inference", json={"coords": [list(CP)], "timestamp_hour": "2026-06-09T19:30:00+05:30"})
        assert a.json() == b.json()

    def test_inference_outside_coverage_404(self, client):
        r = client.post("/inference", json={"coords": [list(MUMBAI)], "timestamp_hour": "2026-06-09T14:00:00Z"})
        assert r.status_code == 404

    def test_inference_bad_timestamp_400(self, client):
        r = client.post("/inference", json={"coords": [list(CP)], "timestamp_hour": "banana"})
        assert r.status_code == 400

    def test_inference_out_of_window_400(self, client):
        r = client.post("/inference", json={"coords": [list(CP)], "timestamp_hour": "1999-01-01T00:00:00Z"})
        assert r.status_code == 400

    def test_inference_out_of_range_400(self, client):
        r = client.post("/inference", json={"coords": [[91.0, 200.0]], "timestamp_hour": "2026-06-09T14:00:00Z"})
        assert r.status_code == 400

    def test_inference_empty_batch_422(self, client):
        r = client.post("/inference", json={"coords": [], "timestamp_hour": "2026-06-09T14:00:00Z"})
        assert r.status_code == 422

    def test_inference_oversized_batch_400(self, client):
        from kashi.config import MAX_BATCH
        r = client.post("/inference", json={
            "coords": [list(CP)] * (MAX_BATCH + 1),
            "timestamp_hour": "2026-06-09T14:00:00Z",
        })
        assert r.status_code == 400

    def test_meta_and_health(self, client):
        meta = client.get("/meta").json()
        assert meta["point_count"] == 65736 and meta["feature_count"] == 130
        assert client.get("/health").json()["status"] == "healthy"

    def test_seasonal_signal(self, client):
        """Winter Delhi must predict materially worse air than monsoon."""
        winter = client.post("/inference", json={
            "coords": [list(CP)], "timestamp_hour": "2026-01-15T08:00:00Z"}).json()["predictions"][0]
        monsoon = client.post("/inference", json={
            "coords": [list(CP)], "timestamp_hour": "2026-08-10T08:00:00Z"}).json()["predictions"][0]
        assert winter > monsoon
