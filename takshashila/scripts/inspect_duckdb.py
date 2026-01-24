#!/usr/bin/env python3
"""Inspect all separate DuckDB files in takshashila/duckdb/ directory.

Checks schema, row counts, and sample data for:
- grid.duckdb
- pm25_observations.duckdb
- meteo.duckdb
- landsat8.duckdb
- sentinel2.duckdb
- predictions.duckdb
"""
import sys
from pathlib import Path
import duckdb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from takshashila.core.config import (
    DUCKDB_DIR, DUCKDB_GRID_PATH, DUCKDB_PM25_PATH, DUCKDB_METEO_PATH,
    DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, DUCKDB_PREDICTIONS_PATH
)

# Map table names to their file paths
DB_FILES = {
    "grid": DUCKDB_GRID_PATH,
    "pm25_observation": DUCKDB_PM25_PATH,
    "meteo": DUCKDB_METEO_PATH,
    "landsat8": DUCKDB_LANDSAT8_PATH,
    "sentinel2": DUCKDB_SENTINEL2_PATH,
    "predictions": DUCKDB_PREDICTIONS_PATH,
}


def inspect_database(db_path: Path, table_name: str):
    """Inspect a single DuckDB file."""
    if not db_path.exists():
        print(f"❌ File not found: {db_path}")
        return False
    
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        
        print(f"\n{'='*80}")
        print(f"FILE: {db_path.name}")
        print(f"{'='*80}")
        
        # Check if table exists
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if not table_names:
            print("❌ No tables found in this database file")
            con.close()
            return False
        
        print(f"✓ Tables: {', '.join(table_names)}")
        print()
        
        # For each table in this file
        for (table,) in tables:
            # Schema
            print(f"--- SCHEMA: {table} ---")
            schema = con.execute(f"DESCRIBE {table}").fetchdf()
            print(schema)
            print()
            
            # Row count
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Row count: {count:,}")
            print()
            
            # Sample rows
            if count > 0:
                print(f"Sample rows (limit 5):")
                sample = con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf()
                print(sample)
            else:
                print("(No rows in table)")
            print()
        
        con.close()
        return True
    
    except Exception as e:
        print(f"❌ Error reading {db_path}: {e}")
        return False


def main():
    """Inspect all DuckDB files."""
    print("\n" + "="*80)
    print("TAKSHASHILA DUCKDB INSPECTION")
    print("="*80)
    print(f"Directory: {DUCKDB_DIR}")
    print()
    
    # Check if directory exists
    if not DUCKDB_DIR.exists():
        print(f"❌ DuckDB directory not found: {DUCKDB_DIR}")
        print("Run: python3 takshashila/scripts/init_duckdb.py")
        return 1
    
    # Inspect each database file
    results = {}
    for table_name, db_path in DB_FILES.items():
        results[table_name] = inspect_database(db_path, table_name)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    exists_count = sum(1 for p in DB_FILES.values() if p.exists())
    print(f"Files found: {exists_count}/{len(DB_FILES)}")
    print()
    
    for table_name, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {table_name:25} ({DB_FILES[table_name].name})")
    
    print()
    
    # File sizes
    print("File sizes:")
    total_size = 0
    for table_name, db_path in DB_FILES.items():
        if db_path.exists():
            size_mb = db_path.stat().st_size / 1024 / 1024
            total_size += db_path.stat().st_size
            print(f"  {db_path.name:30} {size_mb:8.2f} MB")
    
    total_size_mb = total_size / 1024 / 1024
    print(f"  {'Total':30} {total_size_mb:8.2f} MB")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
