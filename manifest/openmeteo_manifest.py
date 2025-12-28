"""Per-city Open-Meteo artifact manifest.

Tracks meteorology artifacts at the per-city level.
Each artifact is identified by (city, date, version) and stored at:
    artifacts/v1/openmeteo/{city}/{date}.parquet

Manifest database stored at:
    artifacts/v1/openmeteo/{city}/manifest/openmeteo_manifest.db

Schema:
    - city: str
    - date: str (YYYY-MM-DD)
    - version: str (e.g., "v1")
    - status: "producing" | "ok" | "error"
    - artifact_path: str (optional, set when status="ok")
    - error_message: str (optional, set when status="error")
    - request_id: str (optional, for tracking requests)
    - created_at: timestamp
    - updated_at: timestamp
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class OpenMeteoEntry:
    """Open-Meteo manifest entry."""
    city: str
    date: str
    version: str
    status: str
    artifact_path: Optional[str] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class OpenMeteoManifest:
    """Per-city Open-Meteo artifact manifest.
    
    Parameters
    ----------
    city : str
        City name (e.g., "mumbai")
    artifacts_root : Path
        Root artifacts directory (default: "artifacts")
    version : str
        Artifact version (default: "v1")
    """

    def __init__(
        self,
        city: str,
        artifacts_root: Path | str = Path("artifacts"),
        version: str = "v1",
    ):
        self.city = city
        self.artifacts_root = Path(artifacts_root)
        self.version = version
        
        # Manifest stored alongside artifacts: artifacts/v1/openmeteo/{city}/manifest/
        manifest_dir = self.artifacts_root / version / "openmeteo" / city / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = manifest_dir / "openmeteo_manifest.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize manifest database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS openmeteo_artifacts (
                city TEXT NOT NULL,
                date TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_path TEXT,
                error_message TEXT,
                request_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (city, date, version)
            )
        """)
        
        # Index for fast status queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON openmeteo_artifacts(status)
        """)
        
        conn.commit()
        conn.close()

    def get(self, date: str) -> Optional[OpenMeteoEntry]:
        """Get manifest entry for a date.
        
        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format
            
        Returns
        -------
        Optional[OpenMeteoEntry]
            Entry if exists, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT city, date, version, status, artifact_path, error_message, 
                   request_id, created_at, updated_at
            FROM openmeteo_artifacts
            WHERE city = ? AND date = ? AND version = ?
            """,
            (self.city, date, self.version),
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return OpenMeteoEntry(
            city=row[0],
            date=row[1],
            version=row[2],
            status=row[3],
            artifact_path=row[4],
            error_message=row[5],
            request_id=row[6],
            created_at=row[7],
            updated_at=row[8],
        )

    def mark_producing(self, date: str, request_id: str) -> None:
        """Mark date as producing (start of processing).
        
        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format
        request_id : str
            Request identifier for tracking
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute(
            """
            INSERT INTO openmeteo_artifacts 
            (city, date, version, status, request_id, created_at, updated_at)
            VALUES (?, ?, ?, 'producing', ?, ?, ?)
            ON CONFLICT(city, date, version) DO UPDATE SET
                status = 'producing',
                request_id = excluded.request_id,
                updated_at = excluded.updated_at
            """,
            (self.city, date, self.version, request_id, now, now),
        )
        
        conn.commit()
        conn.close()

    def mark_ok(self, date: str, artifact_path: Path | str) -> None:
        """Mark date as successfully completed.
        
        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format
        artifact_path : Path | str
            Path to the artifact file
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute(
            """
            UPDATE openmeteo_artifacts
            SET status = 'ok',
                artifact_path = ?,
                error_message = NULL,
                updated_at = ?
            WHERE city = ? AND date = ? AND version = ?
            """,
            (str(artifact_path), now, self.city, date, self.version),
        )
        
        conn.commit()
        conn.close()

    def mark_error(self, date: str, error_message: str) -> None:
        """Mark date as failed.
        
        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format
        error_message : str
            Error description
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute(
            """
            UPDATE openmeteo_artifacts
            SET status = 'error',
                error_message = ?,
                updated_at = ?
            WHERE city = ? AND date = ? AND version = ?
            """,
            (error_message, now, self.city, date, self.version),
        )
        
        conn.commit()
        conn.close()

    def list_ok(self) -> list[OpenMeteoEntry]:
        """List all successfully completed entries.
        
        Returns
        -------
        list[OpenMeteoEntry]
            All entries with status='ok'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT city, date, version, status, artifact_path, error_message,
                   request_id, created_at, updated_at
            FROM openmeteo_artifacts
            WHERE city = ? AND version = ? AND status = 'ok'
            ORDER BY date
            """,
            (self.city, self.version),
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            OpenMeteoEntry(
                city=row[0],
                date=row[1],
                version=row[2],
                status=row[3],
                artifact_path=row[4],
                error_message=row[5],
                request_id=row[6],
                created_at=row[7],
                updated_at=row[8],
            )
            for row in rows
        ]
