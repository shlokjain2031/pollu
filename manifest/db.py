"""Minimal manifest DB for Pollu artifacts.

This module provides a tiny, authoritative manifest that tracks the
lifecycle of immutable Parquet artifacts. It is intentionally small and
dependency-free (uses stdlib `sqlite3`).

API (instance methods on `Manifest`):
 - get(key) -> Optional[dict]
 - mark_producing(key, request_id) -> None
 - mark_ok(key, artifact_path) -> None
 - mark_error(key, error) -> None

Key contract
------------
Keys are dictionaries (or any mapping) with the required fields:
  - row_idx: int
  - timestamp: str (ISO date)
  - artifact_type: str
  - version: str

Artifact path contract
----------------------
Deterministic path for a key is computed as:
  <artifacts_root>/<artifact_type>/<timestamp>/<version>/<row_idx>.parquet

Lifecycle
---------
missing (no manifest row) -> producing -> ok | error

The manifest enforces transitions and is the source of truth for existence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json
import logging

logger = logging.getLogger("pollu.manifest")

# Allowed artifact types are kept in sync with docs/artifact_contracts.yml
ALLOWED_ARTIFACT_TYPES = {
    "grid_30m",
    "landsat_signals",
    "sentinel_evi_signals",
    "s5p_no2_signals",
    "openmeteo_meteo",
    "openaq_daily",
}


def _validate_artifact_type(artifact_type: str) -> None:
    if artifact_type not in ALLOWED_ARTIFACT_TYPES:
        raise ValueError(
            f"Unsupported artifact_type '{artifact_type}'. Allowed: {sorted(ALLOWED_ARTIFACT_TYPES)}"
        )


@dataclass
class ArtifactKey:
    row_idx: int
    timestamp: str
    artifact_type: str
    version: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ArtifactKey":
        return cls(
            row_idx=int(mapping["row_idx"]),
            timestamp=str(mapping["timestamp"]),
            artifact_type=str(mapping["artifact_type"]),
            version=str(mapping["version"]),
        )

    def id(self) -> str:
        # Deterministic, stable string id for sqlite primary key
        safe = lambda s: str(s).replace("/", "_")
        return f"{safe(self.artifact_type)}::{safe(self.timestamp)}::{safe(self.row_idx)}::{safe(self.version)}"


class Manifest:
    """SQLite-backed manifest for artifact lifecycle tracking."""

    def __init__(self, db_path: Optional[Path] = None, artifacts_root: Optional[Path] = None):
        default_db = Path("manifest/manifest.db")
        self.db_path = Path(db_path or default_db).expanduser().resolve()
        # ensure parent directory exists so sqlite can create the file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root = Path(artifacts_root or Path("artifacts")).expanduser().resolve()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level="EXCLUSIVE", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                row_idx INTEGER,
                timestamp TEXT,
                artifact_type TEXT,
                version TEXT,
                state TEXT,
                request_id TEXT,
                artifact_path TEXT,
                error TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_artifacts_state ON artifacts(state);
            """
        )
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def artifact_path_for_key(self, key: ArtifactKey) -> Path:
        _validate_artifact_type(key.artifact_type)
        path = self.artifacts_root / key.artifact_type / key.timestamp / key.version
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{key.row_idx}.parquet"

    def get(self, key_mapping: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        key = ArtifactKey.from_mapping(key_mapping)
        _validate_artifact_type(key.artifact_type)
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM artifacts WHERE id = ?", (key.id(),))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def mark_producing(self, key_mapping: Mapping[str, Any], request_id: str) -> None:
        key = ArtifactKey.from_mapping(key_mapping)
        _validate_artifact_type(key.artifact_type)
        now = self._now()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO artifacts (id, row_idx, timestamp, artifact_type, version, state, request_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key.id(), key.row_idx, key.timestamp, key.artifact_type, key.version, "producing", request_id, now, now),
            )
            self._conn.commit()
            logger.info("Marked producing: %s (request=%s)", key.id(), request_id)
        except sqlite3.IntegrityError:
            # Already exists - fetch current state and raise to enforce lifecycle invariant
            cur.execute("SELECT state FROM artifacts WHERE id = ?", (key.id(),))
            row = cur.fetchone()
            state = row[0] if row else "unknown"
            raise RuntimeError(f"Cannot mark producing; artifact already exists with state={state}")

    def mark_ok(self, key_mapping: Mapping[str, Any], artifact_path: str | Path) -> None:
        key = ArtifactKey.from_mapping(key_mapping)
        _validate_artifact_type(key.artifact_type)
        cur = self._conn.cursor()
        cur.execute("SELECT state FROM artifacts WHERE id = ?", (key.id(),))
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
            "UPDATE artifacts SET state = ?, artifact_path = ?, updated_at = ? WHERE id = ?",
            ("ok", str(p), now, key.id()),
        )
        self._conn.commit()
        logger.info("Marked ok: %s -> %s", key.id(), p)

    def mark_error(self, key_mapping: Mapping[str, Any], error: str) -> None:
        key = ArtifactKey.from_mapping(key_mapping)
        _validate_artifact_type(key.artifact_type)
        cur = self._conn.cursor()
        cur.execute("SELECT state FROM artifacts WHERE id = ?", (key.id(),))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Cannot mark error: artifact not marked producing")
        state = row[0]
        if state != "producing":
            raise RuntimeError(f"Invalid transition: {state} -> error")
        now = self._now()
        cur.execute(
            "UPDATE artifacts SET state = ?, error = ?, updated_at = ? WHERE id = ?",
            ("error", error, now, key.id()),
        )
        self._conn.commit()
        logger.info("Marked error for %s: %s", key.id(), error)

    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:
            pass


__all__ = ["Manifest", "ArtifactKey"]
