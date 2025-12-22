import sys
from types import ModuleType
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import pytest

from serving.data_loader import DuckDBQueryLoader
from serving.query_descriptors import DESCRIPTORS
from serving.types import ManifestRow


def _make_manifest(path: str, state: str = "ok") -> ManifestRow:
    return ManifestRow(manifest_id="m::1", state=state, artifact_path=path, request_id=None, created_at="t", updated_at="t")


def test_duckdb_import_enforced(monkeypatch):
    # If duckdb is available in the environment, DuckDBQueryLoader() should
    # succeed; otherwise it should raise ImportError. This test asserts that
    # the loader enforces the dependency but remains tolerant when the
    # environment provides it.
    import importlib.util

    has_duckdb = importlib.util.find_spec("duckdb") is not None
    if not has_duckdb:
        with pytest.raises(ImportError):
            DuckDBQueryLoader()
    else:
        # In environments with duckdb installed, initialization succeeds.
        loader = DuckDBQueryLoader()
        assert loader is not None


class FakeCursor:
    def __init__(self):
        self.last_sql = None
        self.last_params = None

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params

    def fetchdf(self):
        raise Exception("no pandas")

    def fetchall(self):
        return [(1, 2, 3)]


class FakeConn:
    def __init__(self, cursor):
        self._cur = cursor

    def cursor(self):
        return self._cur

    def close(self):
        pass


class FakeDuckDB(ModuleType):
    def __init__(self):
        super().__init__("duckdb")
        self._last_conn = None

    def connect(self, database=None, read_only=False):
        cur = FakeCursor()
        self._last_conn = FakeConn(cur)
        return self._last_conn


def test_query_dispatch_and_parameter_binding(monkeypatch, tmp_path):
    fake = FakeDuckDB()
    monkeypatch.setitem(sys.modules, "duckdb", fake)

    loader = DuckDBQueryLoader()
    # pick a descriptor to test dispatch
    kind, desc = next(iter(DESCRIPTORS.items()))
    manifest = _make_manifest(path=str(tmp_path / "db.duckdb"))
    # Build params expected by descriptor
    params = {}
    for p in desc.params:
        params[p.name] = "VAL"

    # Call query; since fake cursor fetchdf raises, loader will use fetchall->CSV
    b = loader.query(manifest, desc, params)
    assert isinstance(b, (bytes, bytearray))
    # verify that fake received SQL and params
    conn = fake._last_conn
    assert conn is not None
    cur = conn.cursor()
    assert cur.last_sql is not None
    assert cur.last_params is not None


def test_loader_state_enforcement(monkeypatch, tmp_path):
    fake = FakeDuckDB()
    monkeypatch.setitem(sys.modules, "duckdb", fake)
    loader = DuckDBQueryLoader()
    manifest = _make_manifest(path=str(tmp_path / "db.duckdb"), state="producing")
    kind, desc = next(iter(DESCRIPTORS.items()))
    params = {p.name: "x" for p in desc.params}
    with pytest.raises(ValueError):
        loader.query(manifest, desc, params)
