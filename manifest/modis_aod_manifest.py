"""MODIS AOD-specific manifest for tracking MODIS MAIAC AOD signal artifacts.

This manifest is separate from the central manifest.db. Each city gets its own
MODIS AOD manifest stored alongside the artifacts:
  artifacts/v1/modis_aod/{city}/manifest/modis_aod_manifest.db

Schema
------
- id: str (composite key: city::date::version)
- city: str
- date: str (YYYY-MM-DD)
- version: str (e.g., "v1")
- state: str (producing, ok, error)
- request_id: str
- artifact_path: str (relative or absolute path to parquet)
- error: str (NULL if no error)
- created_at: str (ISO 8601 timestamp)
- updated_at: str (ISO 8601 timestamp)

Lifecycle
---------
missing (no row) -> producing -> ok | error
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("pollu.modis_aod_manifest")


class ModisAodManifest:
    """SQLite-backed manifest for MODIS AOD artifact lifecycle tracking."""

    def __init__(self, city: str, artifacts_root: Path, version: str = "v1"):
        """
        Initialize MODIS AOD manifest for a specific city.

        Parameters
        ----------
        city : str
            City name (e.g., "mumbai")
        artifacts_root : Path
            Root artifacts directory (e.g., Path("artifacts"))
        version : str
            Artifact version (default: "v1")
        """
        self.city = city
        self.version = version
        self.artifacts_root = Path(artifacts_root).expanduser().resolve()
        
        # Store manifest alongside artifacts: artifacts/v1/modis_aod/{city}/manifest/
        manifest_dir = self.artifacts_root / version / "modis_aod" / city / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = manifest_dir / "modis_aod_manifest.db"
        
        self._conn = sqlite3.connect(
            str(self.db_path),
            isolation_level="EXCLUSIVE",
            check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS modis_aod_artifacts (
                id TEXT PRIMARY KEY,
                city TEXT NOT NULL,
                date TEXT NOT NULL,
                version TEXT NOT NULL,
                state TEXT NOT NULL,
                request_id TEXT,
                artifact_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_modis_aod_state ON modis_aod_artifacts(state);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_modis_aod_date ON modis_aod_artifacts(date);
            """
        )
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _make_id(self, city: str, date: str, version: str) -> str:
        """Generate deterministic ID from city, date, version."""
        return f"{city}::{date}::{version}"

    def get(self, date: str) -> Optional[dict]:
        """Get artifact entry for a specific date."""
        artifact_id = self._make_id(self.city, date, self.version)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM modis_aod_artifacts WHERE id = ?",
            (artifact_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def mark_producing(self, date: str, request_id: str) -> None:
        """Mark artifact as being produced."""
        artifact_id = self._make_id(self.city, date, self.version)
        now = self._now()
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO modis_aod_artifacts
                (id, city, date, version, state, request_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, self.city, date, self.version, "producing", request_id, now, now),
            )
            self._conn.commit()
            logger.info("Marked producing: %s (request=%s)", artifact_id, request_id)
        except sqlite3.IntegrityError:
            # Already exists - update to producing state
            cur.execute(
                """
                UPDATE modis_aod_artifacts
                SET state = ?, request_id = ?, updated_at = ?
                WHERE id = ?
                """,
                ("producing", request_id, now, artifact_id),
            )
            self._conn.commit()
            logger.info("Re-marked producing: %s (request=%s)", artifact_id, request_id)

    def mark_ok(self, date: str, artifact_path: str | Path) -> None:
        """Mark artifact as successfully produced."""
        artifact_id = self._make_id(self.city, date, self.version)
        cur = self._conn.cursor()
        cur.execute("SELECT state FROM modis_aod_artifacts WHERE id = ?", (artifact_id,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Cannot mark ok: artifact not marked producing")
        state = row[0]
        if state != "producing":
            raise RuntimeError(f"Invalid transition: {state} -> ok")

        p = Path(artifact_path)
        if not p.exists():
            raise FileNotFoundError(f"Artifact file not found: {p}")

        now = self._now()
        cur.execute(
            """
            UPDATE modis_aod_artifacts
            SET state = ?, artifact_path = ?, error = NULL, updated_at = ?
            WHERE id = ?
            """,
            ("ok", str(p), now, artifact_id),
        )
        self._conn.commit()
        logger.info("Marked ok: %s -> %s", artifact_id, p)

    def mark_error(self, date: str, error: str) -> None:
        """Mark artifact production as failed."""
        artifact_id = self._make_id(self.city, date, self.version)
        cur = self._conn.cursor()
        cur.execute("SELECT state FROM modis_aod_artifacts WHERE id = ?", (artifact_id,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Cannot mark error: artifact not marked producing")
        state = row[0]
        if state != "producing":
            raise RuntimeError(f"Invalid transition: {state} -> error")
        now = self._now()
        cur.execute(
            """
            UPDATE modis_aod_artifacts
            SET state = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            ("error", error, now, artifact_id),
        )
        self._conn.commit()
        logger.info("Marked error for %s: %s", artifact_id, error)

    def list_ok(self) -> list[dict]:
        """List all successfully produced artifacts."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT * FROM modis_aod_artifacts
            WHERE city = ? AND version = ? AND state = 'ok'
            ORDER BY date
            """,
            (self.city, self.version),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def list_errors(self) -> list[dict]:
        """List all failed artifacts."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT * FROM modis_aod_artifacts
            WHERE city = ? AND version = ? AND state = 'error'
            ORDER BY date
            """,
            (self.city, self.version),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def close(self) -> None:
        """Close database connection."""
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:
            pass
