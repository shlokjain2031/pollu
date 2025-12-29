#!/usr/bin/env python3
"""
CPCB AQI ETL Pipeline

Processes CPCB hourly air quality CSV files and produces daily parquet artifacts.

Input:
    cpcb_aqi_data/Raw_data_1Hr_{year}_site_{id}_{station_name}_1Hr.csv

Output:
    artifacts/v1/aqi/{date}.parquet
    
    Schema:
        - sensor_id: str (e.g., "site_5811")
        - timestamp: datetime
        - pm25: float

Usage:
    python -m patliputra.cpcb_aqi
    python -m patliputra.cpcb_aqi --input-dir cpcb_aqi_data --force
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from manifest.cpcb_aqi_manifest import CPCBAqiManifest


# Filename pattern: Raw_data_1Hr_{year}_site_{id}_{station_name}_1Hr.csv
FILENAME_PATTERN = re.compile(r"Raw_data_1Hr_(\d{4})_site_(\d+)_(.+)_1Hr\.csv")

# PM2.5 column name in CSV (with unicode micro sign)
PM25_COLUMN = "PM2.5 (µg/m³)"
TIMESTAMP_COLUMN = "Timestamp"


def parse_filename(filename: str) -> tuple[str, str, str] | None:
    """
    Parse CPCB filename to extract year, site_id, and station_name.
    
    Parameters
    ----------
    filename : str
        Filename like "Raw_data_1Hr_2020_site_5397_Khindipada-Bhandup_West_Mumbai_IITM_1Hr.csv"
    
    Returns
    -------
    tuple[str, str, str] | None
        (year, site_id, station_name) or None if pattern doesn't match
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        year, site_num, station_name = match.groups()
        site_id = f"site_{site_num}"
        return year, site_id, station_name
    return None


def read_cpcb_csv(filepath: Path, site_id: str) -> Iterator[dict]:
    """
    Read CPCB CSV file and yield records with sensor_id, timestamp, pm25.
    
    Skips rows where PM2.5 is NA or invalid.
    
    Parameters
    ----------
    filepath : Path
        Path to CSV file
    site_id : str
        Sensor ID (e.g., "site_5811")
    
    Yields
    ------
    dict
        Record with keys: sensor_id, timestamp, pm25
    """
    try:
        # Read CSV with NA handling
        df = pd.read_csv(
            filepath,
            usecols=[TIMESTAMP_COLUMN, PM25_COLUMN],
            na_values=["NA", "None", "", " "],
            parse_dates=[TIMESTAMP_COLUMN],
        )
    except Exception as e:
        print(f"  ⚠ Error reading {filepath.name}: {e}")
        return
    
    # Drop rows with missing PM2.5
    df = df.dropna(subset=[PM25_COLUMN])
    
    # Convert PM2.5 to numeric (coerce errors to NaN, then drop)
    df[PM25_COLUMN] = pd.to_numeric(df[PM25_COLUMN], errors="coerce")
    df = df.dropna(subset=[PM25_COLUMN])
    
    for _, row in df.iterrows():
        yield {
            "sensor_id": site_id,
            "timestamp": row[TIMESTAMP_COLUMN],
            "pm25": row[PM25_COLUMN],
        }


def process_cpcb_files(
    input_dir: Path,
    output_dir: Path,
    manifest: CPCBAqiManifest,
    force: bool = False,
) -> dict:
    """
    Process all CPCB CSV files and produce daily parquet artifacts.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing CPCB CSV files
    output_dir : Path
        Output directory for parquet files (e.g., artifacts/v1/aqi)
    manifest : CPCBAqiManifest
        Manifest for tracking artifacts
    force : bool
        If True, reprocess dates that already have status='ok'
    
    Returns
    -------
    dict
        Statistics: {dates_processed, dates_skipped, total_records}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all records grouped by date
    records_by_date: dict[str, list[dict]] = defaultdict(list)
    
    print(f"Scanning CSV files in {input_dir}...")
    csv_files = list(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    valid_files = 0
    for csv_path in sorted(csv_files):
        parsed = parse_filename(csv_path.name)
        if parsed is None:
            print(f"  ⊙ Skipping (pattern mismatch): {csv_path.name}")
            continue
        
        year, site_id, station_name = parsed
        valid_files += 1
        
        # Read and collect records
        for record in read_cpcb_csv(csv_path, site_id):
            # Extract date from timestamp
            ts = record["timestamp"]
            if pd.isna(ts):
                continue
            date_str = ts.strftime("%Y-%m-%d")
            records_by_date[date_str].append(record)
    
    print(f"Processed {valid_files} valid files")
    print(f"Found data for {len(records_by_date)} unique dates")
    
    # Write parquet files by date
    stats = {"dates_processed": 0, "dates_skipped": 0, "total_records": 0}
    
    for date_str in sorted(records_by_date.keys()):
        records = records_by_date[date_str]
        
        # Check manifest
        existing = manifest.get(date_str)
        if existing and existing.status == "ok" and not force:
            stats["dates_skipped"] += 1
            continue
        
        # Mark as producing
        request_id = str(uuid.uuid4())[:8]
        manifest.mark_producing(date_str, request_id)
        
        try:
            # Create DataFrame
            df = pd.DataFrame(records)
            df = df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
            
            # Write parquet
            artifact_path = output_dir / f"{date_str}.parquet"
            df.to_parquet(artifact_path, index=False)
            
            # Mark as ok
            num_records = len(df)
            num_sensors = df["sensor_id"].nunique()
            manifest.mark_ok(date_str, artifact_path, num_records, num_sensors)
            
            stats["dates_processed"] += 1
            stats["total_records"] += num_records
            
            print(f"  ✓ {date_str}: {num_records} records, {num_sensors} sensors")
            
        except Exception as e:
            manifest.mark_error(date_str, str(e))
            print(f"  ✗ {date_str}: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process CPCB AQI CSV files into daily parquet artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("cpcb_aqi_data"),
        help="Directory containing CPCB CSV files (default: cpcb_aqi_data)",
    )
    
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root artifacts directory (default: artifacts)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess dates that already have status='ok'",
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = args.artifacts_root / "v1" / "aqi"
    manifest = CPCBAqiManifest(artifacts_root=args.artifacts_root)
    
    print("=" * 70)
    print("CPCB AQI ETL Pipeline")
    print("=" * 70)
    print(f"Input: {args.input_dir}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest.db_path}")
    print("=" * 70)
    
    if not args.input_dir.exists():
        print(f"✗ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Process
    stats = process_cpcb_files(
        input_dir=args.input_dir,
        output_dir=output_dir,
        manifest=manifest,
        force=args.force,
    )
    
    # Summary
    print("=" * 70)
    print("Summary")
    print(f"  Dates processed: {stats['dates_processed']}")
    print(f"  Dates skipped: {stats['dates_skipped']}")
    print(f"  Total records: {stats['total_records']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
