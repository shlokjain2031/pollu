from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Mapping, Any

from .types import ManifestRow


class ArtifactLoader(ABC):
    """Abstract boundary responsible for loading artifact bytes given a manifest row.

    Implementations MUST assume the manifest_row.state == "ok" and should
    immediately raise a ValueError (or subclass) if that's not true.
    """

    @abstractmethod
    def load(self, manifest_row: ManifestRow) -> bytes:
        raise NotImplementedError()

    def query(self, manifest_row: ManifestRow, descriptor: "QueryDescriptor", params: Mapping[str, Any]) -> bytes:
        """Optional query interface. Default implementations should raise.

        Serving will only call this for Stage 4 query dispatch. Implementations
        that do not support queries should raise NotImplementedError.
        """
        raise NotImplementedError()


class LocalPlaceholderLoader(ArtifactLoader):
    """Trivial loader used as a placeholder for Stage 2.

    This loader ignores artifact_path and returns static bytes. It enforces
    the invariant that the provided manifest row must be in the "ok" state.
    """

    def load(self, manifest_row: ManifestRow) -> bytes:
        if manifest_row.state != "ok":
            raise ValueError("ArtifactLoader.load() requires manifest_row.state == 'ok'")
        return b"artifact data placeholder"


class LocalFileLoader(ArtifactLoader):
    """Loader that reads raw bytes from the filesystem using the manifest's artifact_path.

    This implementation is deliberately minimal and defensive:
    - It requires `manifest_row.state == 'ok'`.
    - It opens the file in binary read-only mode and returns the full contents.
    - Any I/O errors (missing file, permissions, etc.) are propagated as exceptions
      so the HTTP adapter can translate them to 500 responses.
    """

    def load(self, manifest_row: ManifestRow) -> bytes:
        if manifest_row.state != "ok":
            raise ValueError("ArtifactLoader.load() requires manifest_row.state == 'ok'")

        path = manifest_row.artifact_path
        if not path:
            raise ValueError("ManifestRow.artifact_path is required for LocalFileLoader")

        # Open and read the file in binary mode. Let any IOError propagate.
        with open(path, "rb") as fh:
            return fh.read()

    def query(self, manifest_row: ManifestRow, descriptor: "QueryDescriptor", params: Mapping[str, Any]) -> bytes:
        # LocalFileLoader does not support analytical queries via DuckDB.
        raise NotImplementedError("LocalFileLoader does not support query()")


class DuckDBQueryLoader(ArtifactLoader):
    """Loader that executes a small set of pre-defined DuckDB queries.

    This implementation assumes `manifest_row.artifact_path` points to a
    DuckDB database file (or a location DuckDB can operate against). It opens
    DuckDB in read-only mode, maps a descriptor to a single fixed query
    template, binds parameters, executes the query, and returns CSV bytes.

    This implementation is intentionally conservative and bounded. It does
    not accept raw SQL from callers.
    """

    def __init__(self):
        try:
            import duckdb  # type: ignore
        except Exception as e:
            raise ImportError("duckdb is required for DuckDBQueryLoader") from e
        self._duckdb = duckdb

    def load(self, manifest_row: ManifestRow) -> bytes:
        # Fallback: not intended for raw file serving here.
        raise NotImplementedError("DuckDBQueryLoader.load() not implemented; use query()")

    def query(self, manifest_row: ManifestRow, descriptor: "QueryDescriptor", params: Mapping[str, Any]) -> bytes:
        if manifest_row.state != "ok":
            raise ValueError("ArtifactLoader.query() requires manifest_row.state == 'ok'")

        db_path = manifest_row.artifact_path
        if not db_path:
            raise ValueError("ManifestRow.artifact_path is required for DuckDBQueryLoader")

        conn = self._duckdb.connect(database=db_path, read_only=True)
        try:
            # Map descriptor -> hardcoded query template. Templates assume an
            # internal table `observations` exists with columns: spatial_id, timestamp, value, geom
            kind = descriptor.query_kind
            if kind.name == "POINT_LATEST":
                sql = "SELECT * FROM observations WHERE spatial_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1"
                params_tuple = (params["spatial_id"], params["as_of"])  # type: ignore
            elif kind.name == "POINT_TIME_RANGE":
                sql = "SELECT * FROM observations WHERE spatial_id = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp ASC"
                params_tuple = (params["spatial_id"], params["start"], params["end"])  # type: ignore
            elif kind.name == "POLYGON_LATEST_AVG":
                sql = "SELECT AVG(value) as avg_value FROM observations WHERE ST_Within(geom, ST_GeomFromText(?)) AND timestamp <= ?"
                params_tuple = (params["wkt"], params["as_of"])  # type: ignore
            elif kind.name == "POLYGON_TIME_RANGE_AVG":
                sql = "SELECT AVG(value) as avg_value FROM observations WHERE ST_Within(geom, ST_GeomFromText(?)) AND timestamp BETWEEN ? AND ?"
                params_tuple = (params["wkt"], params["start"], params["end"])  # type: ignore
            elif kind.name == "POLYGON_TIME_RANGE_RAW":
                sql = "SELECT * FROM observations WHERE ST_Within(geom, ST_GeomFromText(?)) AND timestamp BETWEEN ? AND ? ORDER BY timestamp ASC LIMIT ?"
                max_rows = descriptor.safety_limits.get("max_rows") if descriptor.safety_limits else None
                limit = int(max_rows) if max_rows is not None else 10000
                params_tuple = (params["wkt"], params["start"], params["end"], limit)  # type: ignore
            else:
                raise ValueError(f"Unsupported query kind: {kind}")

            # Execute with parameter binding. duckdb supports parameterized execution via execute(sql, params)
            cur = conn.cursor()
            cur.execute(sql, params_tuple)
            # Fetch into pandas DataFrame for serialization
            try:
                df = cur.fetchdf()
            except Exception:
                # If fetchdf is not available, fetchall and return CSV minimal
                rows = cur.fetchall()
                # Convert to CSV-like bytes naively
                import csv
                import io

                buf = io.StringIO()
                writer = csv.writer(buf)
                for r in rows:
                    writer.writerow(r)
                return buf.getvalue().encode("utf-8")

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            return csv_bytes
        finally:
            try:
                conn.close()
            except Exception:
                pass

