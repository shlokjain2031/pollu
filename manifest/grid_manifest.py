"""Grid manifest for tracking canonical grids.

This manifest is separate from artifact manifest because grids:
- Do not have lifecycle states (no producing/ok/error)
- Do not have timestamps
- Are immutable once created
- Serve as the foundation for all other artifacts

Schema
------
- grid_id: str (city identifier, e.g., "mumbai")
- grid_version: str (e.g., "v1")
- crs: str (e.g., "EPSG:32643")
- origin_x: float
- origin_y: float
- resolution_m: float
- nx: int (full grid columns before clipping)
- ny: int (full grid rows before clipping)
- grid_hash: str (sha256 of grid spec for deterministic verification)
- created_at: str (ISO 8601 timestamp)

Grid hash computation
---------------------
grid_hash = sha256(crs + resolution_m + origin_x + origin_y + nx + ny)
This enables downstream verification that grid hasn't changed.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("pollu.grid_manifest")


def compute_grid_hash(crs: str, resolution_m: float, origin_x: float, origin_y: float, nx: int, ny: int) -> str:
    """Compute deterministic SHA256 hash of grid specification.
    
    Parameters
    ----------
    crs : str
        EPSG code (e.g., "EPSG:32643")
    resolution_m : float
        Grid resolution in meters
    origin_x : float
        Origin X coordinate
    origin_y : float
        Origin Y coordinate
    nx : int
        Number of columns in full grid (before clipping)
    ny : int
        Number of rows in full grid (before clipping)
    
    Returns
    -------
    str
        Hex-encoded SHA256 hash
    """
    # Canonical string representation for hashing
    spec = f"{crs}|{resolution_m:.10f}|{origin_x:.6f}|{origin_y:.6f}|{nx}|{ny}"
    return hashlib.sha256(spec.encode('utf-8')).hexdigest()


class GridManifest:
    """SQLite-backed manifest for grid registry."""
    
    def __init__(self, db_path: Optional[Path] = None):
        # Place grid manifest DB under manifest/grid/ to keep grid-specific
        # metadata files separated from other manifest DBs.
        default_db = Path("manifest/grid/grid_manifest.db")
        self.db_path = Path(db_path or default_db).expanduser().resolve()
        # Ensure the directory exists (manifest/grid/)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level="EXCLUSIVE", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS grids (
                grid_id TEXT PRIMARY KEY,
                grid_version TEXT NOT NULL,
                crs TEXT NOT NULL,
                origin_x REAL NOT NULL,
                origin_y REAL NOT NULL,
                resolution_m REAL NOT NULL,
                nx INTEGER NOT NULL,
                ny INTEGER NOT NULL,
                grid_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_grids_version ON grids(grid_version);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_grids_hash ON grids(grid_hash);
            """
        )
        self._conn.commit()
    
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def register(
        self,
        grid_id: str,
        grid_version: str,
        crs: str,
        origin_x: float,
        origin_y: float,
        resolution_m: float,
        nx: int,
        ny: int,
    ) -> str:
        """Register a grid in the manifest.
        
        Parameters
        ----------
        grid_id : str
            City/grid identifier (e.g., "mumbai")
        grid_version : str
            Grid version (e.g., "v1")
        crs : str
            Coordinate reference system (e.g., "EPSG:32643")
        origin_x : float
            Grid origin X coordinate
        origin_y : float
            Grid origin Y coordinate
        resolution_m : float
            Grid resolution in meters
        nx : int
            Number of columns (before clipping)
        ny : int
            Number of rows (before clipping)
        
        Returns
        -------
        str
            Grid hash (SHA256)
        
        Raises
        ------
        RuntimeError
            If grid_id already exists with different spec
        """
        grid_hash = compute_grid_hash(crs, resolution_m, origin_x, origin_y, nx, ny)
        now = self._now()
        
        cur = self._conn.cursor()
        
        # Check if grid already exists
        cur.execute("SELECT grid_hash FROM grids WHERE grid_id = ?", (grid_id,))
        row = cur.fetchone()
        
        if row:
            existing_hash = row[0]
            if existing_hash != grid_hash:
                raise RuntimeError(
                    f"Grid '{grid_id}' already exists with different spec. "
                    f"Existing hash: {existing_hash}, new hash: {grid_hash}. "
                    f"Cannot overwrite grid without explicit version change."
                )
            logger.info("Grid '%s' already registered with matching hash %s", grid_id, grid_hash[:8])
            return grid_hash
        
        # Insert new grid
        try:
            cur.execute(
                """
                INSERT INTO grids (grid_id, grid_version, crs, origin_x, origin_y, resolution_m, nx, ny, grid_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (grid_id, grid_version, crs, origin_x, origin_y, resolution_m, nx, ny, grid_hash, now),
            )
            self._conn.commit()
            logger.info("Registered grid '%s' (version=%s, hash=%s)", grid_id, grid_version, grid_hash[:8])
            return grid_hash
        except sqlite3.IntegrityError as e:
            raise RuntimeError(f"Failed to register grid '{grid_id}': {e}")
    
    def get(self, grid_id: str) -> Optional[dict]:
        """Retrieve grid metadata by grid_id.
        
        Parameters
        ----------
        grid_id : str
            Grid identifier (e.g., "mumbai")
        
        Returns
        -------
        dict or None
            Grid metadata if exists, None otherwise
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM grids WHERE grid_id = ?", (grid_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    
    def verify_hash(self, grid_id: str, expected_hash: str) -> bool:
        """Verify that grid_id has expected hash.
        
        Parameters
        ----------
        grid_id : str
            Grid identifier
        expected_hash : str
            Expected SHA256 hash
        
        Returns
        -------
        bool
            True if hash matches, False otherwise
        """
        grid = self.get(grid_id)
        if grid is None:
            return False
        return grid["grid_hash"] == expected_hash
    
    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
