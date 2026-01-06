"""CLI for target data assembly.

Usage:
    python -m nalanda.target_cli --date 2021-06-07
    python -m nalanda.target_cli --start-date 2021-06-01 --end-date 2021-06-30
    python -m nalanda.target_cli --date 2021-06-07 --force --radius 300
    python -m nalanda.target_cli --start-date 2021-06-01 --end-date 2021-06-30 --workers 4

Output:
    artifacts/v1/training/{date}/target.parquet
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nalanda.target import TargetAssembler, DEFAULT_RADIUS_M, DEFAULT_TRUST_KERNEL


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GRID_PARQUET = PROJECT_ROOT / "artifacts" / "v1" / "grids" / "mumbai" / "grid.parquet"
DEFAULT_AQI_DIR = PROJECT_ROOT / "artifacts" / "v1" / "aqi"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "v1" / "training"
DEFAULT_BBOX = "72.7763,18.8939,72.9797,19.2701"  # Mumbai


def get_aqi_artifact_path(date: str, aqi_dir: Path = DEFAULT_AQI_DIR) -> Path:
    """Get path to AQI artifact for a date."""
    return aqi_dir / f"{date}.parquet"


def get_output_path(date: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Get output path for target parquet."""
    return output_dir / date / "target.parquet"


def write_target_parquet(
    target_df: pd.DataFrame,
    output_path: Path,
    metadata: dict,
) -> None:
    """Write target DataFrame to parquet with metadata.
    
    Args:
        target_df: Target DataFrame with columns:
            row_idx, timestamp, pm25_t, target_trust, n_sensors,
            pm25_t_minus_1h, pm25_t_minus_2h, ..., pm25_t_minus_24h
        output_path: Path to write parquet file.
        metadata: Dict of assembly parameters to store in parquet metadata.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optimize dtypes for storage
    df = target_df.copy()
    
    # Sort by row_idx and timestamp for deterministic ordering
    df = df.sort_values(['row_idx', 'timestamp']).reset_index(drop=True)
    
    df['row_idx'] = df['row_idx'].astype('int64')
    df['pm25_t'] = df['pm25_t'].astype('float32')
    df['target_trust'] = df['target_trust'].astype('float32')
    df['n_sensors'] = df['n_sensors'].astype('int8')
    
    # Ensure lag columns are float32
    lag_cols = [col for col in df.columns if col.startswith('pm25_t_minus_')]
    for col in lag_cols:
        df[col] = df[col].astype('float32')
    
    # Convert to PyArrow table
    table = pa.Table.from_pandas(df, preserve_index=False)
    
    # Add custom metadata
    existing_metadata = table.schema.metadata or {}
    custom_metadata = {k.encode(): v.encode() for k, v in metadata.items()}
    combined_metadata = {**existing_metadata, **custom_metadata}
    
    table = table.replace_schema_metadata(combined_metadata)
    
    # Write with compression
    pq.write_table(
        table,
        output_path,
        compression='zstd',
        compression_level=3,
    )


def process_date(
    date: str,
    assembler: TargetAssembler,
    aqi_dir: Path,
    output_dir: Path,
    force: bool = False,
    prev_day_df: pd.DataFrame = None,
) -> dict:
    """Process a single date and write target parquet.
    
    Args:
        date: Date string in YYYY-MM-DD format.
        assembler: Initialized TargetAssembler.
        aqi_dir: Directory containing AQI artifacts.
        output_dir: Directory for output files.
        force: Overwrite existing output if True.
        prev_day_df: Optional target DataFrame from previous day for lag computation.
        
    Returns:
        Dict with processing statistics.
    """
    result = {
        'date': date,
        'status': 'unknown',
        'input_rows': 0,
        'output_rows': 0,
        'unique_cells': 0,
        'error': None,
    }
    
    # Check input exists
    aqi_path = get_aqi_artifact_path(date, aqi_dir)
    if not aqi_path.exists():
        result['status'] = 'skipped'
        result['error'] = 'AQI artifact not found'
        return result
    
    # Check output exists
    output_path = get_output_path(date, output_dir)
    if output_path.exists() and not force:
        result['status'] = 'skipped'
        result['error'] = 'Output exists (use --force to overwrite)'
        return result
    
    try:
        # Load AQI data
        aqi_df = pd.read_parquet(aqi_path)
        result['input_rows'] = len(aqi_df)
        
        if len(aqi_df) == 0:
            result['status'] = 'skipped'
            result['error'] = 'Empty AQI artifact'
            return result
        
        # Assemble target with lags
        target_df = assembler.assemble_day(aqi_df, prev_day_df=prev_day_df)
        result['output_rows'] = len(target_df)
        result['unique_cells'] = target_df['row_idx'].nunique()
        
        if len(target_df) == 0:
            result['status'] = 'skipped'
            result['error'] = 'No target rows generated'
            return result
        
        # Write output
        metadata = assembler.get_metadata()
        metadata['date'] = date
        metadata['input_rows'] = str(result['input_rows'])
        metadata['output_rows'] = str(result['output_rows'])
        metadata['created_at'] = datetime.now().isoformat()
        
        write_target_parquet(target_df, output_path, metadata)
        
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def process_date_worker(args: tuple) -> dict:
    """Worker function for parallel processing.
    
    This initializes its own TargetAssembler to avoid pickling issues.
    Note: Parallel workers process dates independently, so cross-day lags
    (pm25_t_minus_24h for early hours) may be NaN unless previous day
    was already processed.
    
    Args:
        args: Tuple of (date, grid_parquet, openaq_token, bbox, radius_m, aqi_dir, output_dir, force)
        
    Returns:
        Processing result dict.
    """
    date, grid_parquet, openaq_token, bbox, radius_m, aqi_dir, output_dir, force = args
    
    # Initialize assembler in worker process
    assembler = TargetAssembler(
        grid_parquet_path=grid_parquet,
        openaq_token=openaq_token,
        bbox=bbox,
        radius_m=radius_m,
    )
    
    # Try to load previous day's target for lag computation
    prev_day = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    prev_day_path = get_output_path(prev_day, output_dir)
    prev_day_df = None
    if prev_day_path.exists():
        try:
            prev_day_df = pd.read_parquet(prev_day_path)
        except Exception:
            pass  # Use None if loading fails
    
    # Process the date
    return process_date(date, assembler, aqi_dir, output_dir, force, prev_day_df=prev_day_df)


def date_range(start_date: str, end_date: str) -> list[str]:
    """Generate list of date strings between start and end (inclusive)."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates


def write_summary_report(
    results: list[dict],
    output_path: Path,
) -> None:
    """Write processing summary to CSV.
    
    Args:
        results: List of processing result dicts.
        output_path: Path to write CSV summary.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Assemble target data for air quality modeling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single date
    python -m nalanda.target_cli --date 2021-06-07
    
    # Process date range
    python -m nalanda.target_cli --start-date 2021-06-01 --end-date 2021-06-30
    
    # Force overwrite with custom radius
    python -m nalanda.target_cli --date 2021-06-07 --force --radius 300
    
    # Parallel processing with 4 workers
    python -m nalanda.target_cli --start-date 2021-06-01 --end-date 2021-06-30 --workers 4
        """,
    )
    
    # Date selection (mutually exclusive groups)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        '--date',
        type=str,
        help='Single date to process (YYYY-MM-DD)',
    )
    date_group.add_argument(
        '--start-date',
        type=str,
        help='Start date for range processing (YYYY-MM-DD)',
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for range processing (YYYY-MM-DD), required with --start-date',
    )
    
    # Processing options
    parser.add_argument(
        '--radius',
        type=float,
        default=DEFAULT_RADIUS_M,
        help=f'Influence radius in meters (default: {DEFAULT_RADIUS_M})',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output files',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)',
    )
    
    # OpenAQ and spatial options (required)
    parser.add_argument(
        '--openaq-token',
        type=str,
        required=True,
        help='OpenAQ API token for fetching sensor locations',
    )
    parser.add_argument(
        '--bbox',
        type=str,
        default=DEFAULT_BBOX,
        help=f'Bounding box as "west,south,east,north" (default: {DEFAULT_BBOX})',
    )
    parser.add_argument(
        '--grid-parquet',
        type=Path,
        default=DEFAULT_GRID_PARQUET,
        help='Path to grid parquet file',
    )
    parser.add_argument(
        '--aqi-dir',
        type=Path,
        default=DEFAULT_AQI_DIR,
        help='Directory containing AQI artifacts',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for target files',
    )
    parser.add_argument(
        '--summary',
        type=Path,
        help='Write processing summary to CSV file',
    )
    
    args = parser.parse_args()
    
    # Validate date range arguments
    if args.start_date and not args.end_date:
        parser.error('--end-date is required when using --start-date')
    
    # Validate workers
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    
    # Determine dates to process
    if args.date:
        dates = [args.date]
    else:
        dates = date_range(args.start_date, args.end_date)
    
    print(f"Target Assembly Pipeline")
    print(f"=" * 60)
    print(f"Dates to process: {len(dates)}")
    print(f"Radius: {args.radius}m")
    print(f"Force overwrite: {args.force}")
    print(f"Workers: {args.workers}")
    print(f"=" * 60)
    
    # Process dates
    results = []
    
    if args.workers == 1:
        # Serial processing with single assembler
        print(f"\nInitializing assembler...")
        assembler = TargetAssembler(
            grid_parquet_path=args.grid_parquet,
            openaq_token=args.openaq_token,
            bbox=args.bbox,
            radius_m=args.radius,
        )
        print(f"  {assembler}")
        print()
        
        # Track previous day's target for lag computation
        prev_day_df = None
        
        for date in tqdm(dates, desc="Processing dates", unit="date"):
            # Try to load previous day if not in memory
            if prev_day_df is None:
                prev_day = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                prev_day_path = get_output_path(prev_day, args.output_dir)
                if prev_day_path.exists():
                    try:
                        prev_day_df = pd.read_parquet(prev_day_path)
                    except Exception:
                        pass
            
            result = process_date(
                date=date,
                assembler=assembler,
                aqi_dir=args.aqi_dir,
                output_dir=args.output_dir,
                force=args.force,
                prev_day_df=prev_day_df,
            )
            results.append(result)
            
            # Load current day's target for next iteration (if successful)
            if result['status'] == 'success':
                output_path = get_output_path(date, args.output_dir)
                try:
                    prev_day_df = pd.read_parquet(output_path)
                except Exception:
                    prev_day_df = None
            else:
                # Try to load from disk if exists
                output_path = get_output_path(date, args.output_dir)
                if output_path.exists():
                    try:
                        prev_day_df = pd.read_parquet(output_path)
                    except Exception:
                        prev_day_df = None
                else:
                    prev_day_df = None
            
            # Print status
            if result['status'] == 'success':
                tqdm.write(f"  ✓ {date}: {result['output_rows']:,} rows, {result['unique_cells']:,} cells")
            elif result['status'] == 'skipped':
                tqdm.write(f"  ⊘ {date}: {result['error']}")
            else:
                tqdm.write(f"  ✗ {date}: {result['error']}")
    else:
        # Parallel processing with worker pool
        print(f"\nStarting {args.workers} worker processes...")
        print()
        
        # Prepare worker arguments
        worker_args = [
            (
                date,
                args.grid_parquet,
                args.openaq_token,
                args.bbox,
                args.radius,
                args.aqi_dir,
                args.output_dir,
                args.force,
            )
            for date in dates
        ]
        
        # Process in parallel with progress bar
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_date_worker, arg): arg[0] for arg in worker_args}
            
            # Collect results with progress bar
            with tqdm(total=len(dates), desc="Processing dates", unit="date") as pbar:
                for future in as_completed(futures):
                    date = futures[future]
                    result = future.result()
                    results.append(result)
                    
                    # Print status
                    if result['status'] == 'success':
                        tqdm.write(f"  ✓ {date}: {result['output_rows']:,} rows, {result['unique_cells']:,} cells")
                    elif result['status'] == 'skipped':
                        tqdm.write(f"  ⊘ {date}: {result['error']}")
                    else:
                        tqdm.write(f"  ✗ {date}: {result['error']}")
                    
                    pbar.update(1)
    
    # Sort results by date for consistent output
    results.sort(key=lambda x: x['date'])
    
    # Count statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    skip_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    # Summary
    print()
    print(f"=" * 60)
    print(f"Summary")
    print(f"  Processed: {success_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {error_count}")
    print(f"=" * 60)
    
    # Write summary report if requested
    if args.summary:
        write_summary_report(results, args.summary)
        print(f"\nSummary report written to: {args.summary}")
    
    # Exit code
    if error_count > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
