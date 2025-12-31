"""CPCB AQI artifact manifest.

Tracks hourly AQI artifacts at the per-date level.
Each artifact is identified by (date, version) and stored at:
    artifacts/v1/aqi/{date}.parquet

Manifest database stored at:
    artifacts/v1/aqi/manifest/cpcb_aqi_manifest.db

Schema:
    - date: str (YYYY-MM-DD)
    - version: str (e.g., "v1")
    - status: "producing" | "ok" | "error"
    - artifact_path: str (optional, set when status="ok")
    - num_records: int (optional, number of records in artifact)
    - num_sensors: int (optional, number of unique sensors)
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
class CPCBAqiEntry:
    """CPCB AQI manifest entry."""
    date: str
    version: str
    status: str
    artifact_path: Optional[str] = None
    num_records: Optional[int] = None
    num_sensors: Optional[int] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CPCBAqiManifest:
    """CPCB AQI artifact manifest.
    
    Parameters
    ----------
    artifacts_root : Path
        Root artifacts directory (default: "artifacts")
    version : str
        Artifact version (default: "v1")
    """

    def __init__(
        self,
        artifacts_root: Path | str = Path("artifacts"),
        version: str = "v1",
    ):
        self.artifacts_root = Path(artifacts_root)
        self.version = version
        
        # Manifest stored alongside artifacts: artifacts/v1/aqi/manifest/
        manifest_dir = self.artifacts_root / version / "aqi" / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = manifest_dir / "cpcb_aqi_manifest.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize manifest database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cpcb_aqi_artifacts (
                date TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_path TEXT,
                num_records INTEGER,
                num_sensors INTEGER,
                error_message TEXT,
                request_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (date, version)
            )
        """)
        
        # Index for fast status queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cpcb_aqi_status 
            ON cpcb_aqi_artifacts(status)
        """)
        
        conn.commit()
        conn.close()

    def get(self, date: str) -> Optional[CPCBAqiEntry]:
        """Get manifest entry for a date.
        
        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format
            
        Returns
        -------
        Optional[CPCBAqiEntry]
            Entry if exists, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT date, version, status, artifact_path, num_records, num_sensors,
                   error_message, request_id, created_at, updated_at
            FROM cpcb_aqi_artifacts
            WHERE date = ? AND version = ?
            """,
            (date, self.version),
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return CPCBAqiEntry(
            date=row[0],
            version=row[1],
            status=row[2],
            artifact_path=row[3],
            num_records=row[4],
            num_sensors=row[5],
            error_message=row[6],
            request_id=row[7],
            created_at=row[8],
            updated_at=row[9],
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
            INSERT INTO cpcb_aqi_artifacts 
            (date, version, status, request_id, created_at, updated_at)
            VALUES (?, ?, 'producing', ?, ?, ?)
            ON CONFLICT(date, version) DO UPDATE SET
                status = 'producing',
                request_id = excluded.request_id,
                updated_at = excluded.updated_at
            """,
            (date, self.version, request_id, now, now),
        )
        
        conn.commit()
        conn.close()

    def mark_ok(
        self,
        date: str,
        artifact_path: Path | str,
        num_records: int,
        num_sensors: int,
    ) -> None:
        """Mark date as successfully completed.
        
        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format
        artifact_path : Path | str
            Path to the artifact file
        num_records : int
            Number of records in the artifact
        num_sensors : int
            Number of unique sensors in the artifact
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute(
            """
            UPDATE cpcb_aqi_artifacts
            SET status = 'ok',
                artifact_path = ?,
                num_records = ?,
                num_sensors = ?,
                error_message = NULL,
                updated_at = ?
            WHERE date = ? AND version = ?
            """,
            (str(artifact_path), num_records, num_sensors, now, date, self.version),
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
            UPDATE cpcb_aqi_artifacts
            SET status = 'error',
                error_message = ?,
                updated_at = ?
            WHERE date = ? AND version = ?
            """,
            (error_message, now, date, self.version),
        )
        
        conn.commit()
        conn.close()

    def list_ok(self) -> list[CPCBAqiEntry]:
        """List all successfully completed entries.
        
        Returns
        -------
        list[CPCBAqiEntry]
            All entries with status='ok'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT date, version, status, artifact_path, num_records, num_sensors,
                   error_message, request_id, created_at, updated_at
            FROM cpcb_aqi_artifacts
            WHERE version = ? AND status = 'ok'
            ORDER BY date
            """,
            (self.version,),
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            CPCBAqiEntry(
                date=row[0],
                version=row[1],
                status=row[2],
                artifact_path=row[3],
                num_records=row[4],
                num_sensors=row[5],
                error_message=row[6],
                request_id=row[7],
                created_at=row[8],
                updated_at=row[9],
            )
            for row in rows
        ]

    def list_all(self) -> list[CPCBAqiEntry]:
        """List all entries.
        
        Returns
        -------
        list[CPCBAqiEntry]
            All entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT date, version, status, artifact_path, num_records, num_sensors,
                   error_message, request_id, created_at, updated_at
            FROM cpcb_aqi_artifacts
            WHERE version = ?
            ORDER BY date
            """,
            (self.version,),
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            CPCBAqiEntry(
                date=row[0],
                version=row[1],
                status=row[2],
                artifact_path=row[3],
                num_records=row[4],
                num_sensors=row[5],
                error_message=row[6],
                request_id=row[7],
                created_at=row[8],
                updated_at=row[9],
            )
            for row in rows
        ]
