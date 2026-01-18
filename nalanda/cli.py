"""Command-line interface for training data pipeline.

This module provides CLI tools for:
- Batch processing multiple dates
- Validating assembled training data
- Generating pipeline reports

Usage:
    # Process single date
    python -m nalanda.cli assemble --date 2021-06-12
    
    # Process date range
    python -m nalanda.cli assemble --start-date 2021-06-07 --end-date 2021-06-18
    
    # Validate existing training data
    python -m nalanda.cli validate --date 2021-06-12
    
    # Validate all available dates
    python -m nalanda.cli validate --all
    
    # Generate report
    python -m nalanda.cli report --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from nalanda.training import (
    assemble_training_data,
    save_training_data,
    load_target,
    get_training_feature_columns,
    DEFAULT_TARGET_DIR,
    DEFAULT_OUTPUT_DIR,
)
from nalanda.training_splits_logged import materialize_splits_logged
from nalanda.splits import (
    validate_splits,
    assert_no_leakage,
    get_split_mask,
    print_split_summary,
    SPLIT_RANGES,
)
from nalanda.feature_freeze import (
    freeze_feature_schema,
    freeze_from_dataframe,
    validate_splits_schema,
    FrozenFeatureSchema,
)
from nalanda.features import get_feature_list


# ============================================================================
# Data classes for results
# ============================================================================

@dataclass
class AssemblyResult:
    """Result from assembling training data for one date."""
    date: str
    success: bool
    rows: int = 0
    columns: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    output_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result from validating training data for one date."""
    date: str
    valid: bool
    rows: int = 0
    columns: int = 0
    null_pct: Dict[str, float] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.null_pct is None:
            self.null_pct = {}
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Date utilities
# ============================================================================

def get_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate list of dates between start and end (inclusive).
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        List of date strings.
    """
    start = dt.strptime(start_date, "%Y-%m-%d")
    end = dt.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates


def get_available_target_dates(target_dir: Path = DEFAULT_TARGET_DIR) -> List[str]:
    """Get list of dates that have target data available.
    
    Args:
        target_dir: Base directory for target artifacts.
        
    Returns:
        Sorted list of date strings.
    """
    if not target_dir.exists():
        return []
    
    dates = []
    for d in target_dir.iterdir():
        if d.is_dir() and (d / "target.parquet").exists():
            try:
                dt.strptime(d.name, "%Y-%m-%d")
                dates.append(d.name)
            except ValueError:
                continue
    
    return sorted(dates)


def get_available_training_dates(output_dir: Path = DEFAULT_OUTPUT_DIR) -> List[str]:
    """Get list of dates that have assembled training data.
    
    Args:
        output_dir: Base directory for training outputs.
        
    Returns:
        Sorted list of date strings.
    """
    if not output_dir.exists():
        return []
    
    dates = []
    for d in output_dir.iterdir():
        if d.is_dir() and (d / "training.parquet").exists():
            try:
                dt.strptime(d.name, "%Y-%m-%d")
                dates.append(d.name)
            except ValueError:
                continue
    
    return sorted(dates)


# ============================================================================
# Assembly command
# ============================================================================

def assemble_single(
    date: str,
    city: str = 'mumbai',
    include_bands: bool = False,
    no_spatial: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> AssemblyResult:
    """Assemble training data for a single date.
    
    Args:
        date: Target date (YYYY-MM-DD)
        city: City name
        include_bands: Include raw Landsat bands
        no_spatial: Skip Sentinel spatial features
        output_dir: Output directory
        dry_run: If True, compute but don't save
        
    Returns:
        AssemblyResult with status and metrics.
    """
    start_time = time.time()
    
    try:
        df = assemble_training_data(
            date=date,
            city=city,
            include_landsat_bands=include_bands,
            include_sentinel_spatial=not no_spatial,
            skip_missing_satellite=True,
        )
        
        elapsed = time.time() - start_time
        
        if dry_run:
            return AssemblyResult(
                date=date,
                success=True,
                rows=len(df),
                columns=len(df.columns),
                elapsed_seconds=elapsed,
                output_path=None,
            )
        
        output_path = save_training_data(df, date, output_dir)
        
        return AssemblyResult(
            date=date,
            success=True,
            rows=len(df),
            columns=len(df.columns),
            elapsed_seconds=elapsed,
            output_path=str(output_path),
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        return AssemblyResult(
            date=date,
            success=False,
            elapsed_seconds=elapsed,
            error=str(e),
        )


def assemble_batch(
    dates: List[str],
    city: str = 'mumbai',
    include_bands: bool = False,
    no_spatial: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    verbose: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> List[AssemblyResult]:
    """Assemble training data for multiple dates.
    
    Args:
        dates: List of dates to process
        city: City name
        include_bands: Include raw Landsat bands
        no_spatial: Skip Sentinel spatial features
        output_dir: Output directory
        dry_run: If True, compute but don't save
        verbose: Print progress
        parallel: If True, process dates in parallel
        max_workers: Max parallel workers (default: CPU count)
        
    Returns:
        List of AssemblyResults.
    """
    total = len(dates)
    
    if parallel:
        # Parallel processing with ProcessPoolExecutor
        if verbose:
            print(f"\nProcessing {total} dates in parallel (workers: {max_workers or 'auto'})...")
        
        results = []
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(
                    assemble_single,
                    date=date,
                    city=city,
                    include_bands=include_bands,
                    no_spatial=no_spatial,
                    output_dir=output_dir,
                    dry_run=dry_run,
                ): date
                for date in dates
            }
            
            # Process as they complete
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if verbose:
                        if result.success:
                            print(f"  [{completed_count}/{total}] ✓ {date}: {result.rows:,} rows in {result.elapsed_seconds:.1f}s")
                        else:
                            print(f"  [{completed_count}/{total}] ✗ {date}: {result.error}")
                except Exception as e:
                    # Handle unexpected errors during future execution
                    error_result = AssemblyResult(
                        date=date,
                        success=False,
                        error=f"Parallel execution error: {e}",
                    )
                    results.append(error_result)
                    if verbose:
                        print(f"  [{completed_count}/{total}] ✗ {date}: {e}")
        
        # Sort results by date to maintain consistent order
        results.sort(key=lambda r: r.date)
        
    else:
        # Sequential processing (original behavior)
        results = []
        
        for i, date in enumerate(dates, 1):
            if verbose:
                print(f"\n[{i}/{total}] Processing {date}...")
            
            result = assemble_single(
                date=date,
                city=city,
                include_bands=include_bands,
                no_spatial=no_spatial,
                output_dir=output_dir,
                dry_run=dry_run,
            )
            results.append(result)
            
            if verbose:
                if result.success:
                    print(f"  ✓ {result.rows:,} rows, {result.columns} cols in {result.elapsed_seconds:.1f}s")
                else:
                    print(f"  ✗ Error: {result.error}")
    
    return results


# ============================================================================
# Validation command
# ============================================================================

def validate_training_data(
    date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ValidationResult:
    """Validate assembled training data for a single date.
    
    Checks:
    - File exists and is readable
    - Required columns present
    - No all-null columns
    - Reasonable null percentages
    - Value ranges for key features
    
    Args:
        date: Date to validate (YYYY-MM-DD)
        output_dir: Directory containing training.parquet files
        
    Returns:
        ValidationResult with status and diagnostics.
    """
    warnings = []
    errors = []
    
    # Check file exists
    training_path = output_dir / date / "training.parquet"
    if not training_path.exists():
        return ValidationResult(
            date=date,
            valid=False,
            errors=[f"File not found: {training_path}"],
        )
    
    # Try to load
    try:
        df = pd.read_parquet(training_path)
    except Exception as e:
        return ValidationResult(
            date=date,
            valid=False,
            errors=[f"Failed to read parquet: {e}"],
        )
    
    # Check required columns
    required_cols = ['row_idx', 'timestamp', 'pm25_t', 'target_trust', 'n_sensors']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for all-null columns
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        errors.append(f"All-null columns: {all_null_cols}")
    
    # Compute null percentages
    null_pct = {}
    high_null_threshold = 0.5  # 50%
    
    for col in df.columns:
        pct = df[col].isna().mean()
        null_pct[col] = round(pct * 100, 1)
        
        if pct > high_null_threshold and col not in ['evi', 'ui_light', 'si', 'local_ndvi_std', 'ndvi_s2']:
            warnings.append(f"High null rate ({pct*100:.1f}%) in column: {col}")
    
    # Check value ranges
    if 'pm25_t' in df.columns:
        pm25 = df['pm25_t'].dropna()
        if pm25.min() < 0:
            errors.append(f"Negative pm25_t values found (min: {pm25.min()})")
        if pm25.max() > 1000:
            warnings.append(f"Unusually high pm25_t values (max: {pm25.max()})")
    
    if 'temperature_2m' in df.columns:
        temp = df['temperature_2m'].dropna()
        if temp.min() < -60 or temp.max() > 60:
            warnings.append(f"Temperature out of expected range: [{temp.min():.1f}, {temp.max():.1f}]")
    
    if 'ndvi' in df.columns:
        ndvi = df['ndvi'].dropna()
        if ndvi.min() < -1 or ndvi.max() > 1:
            warnings.append(f"NDVI out of [-1, 1] range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    
    # Check row count
    if len(df) == 0:
        errors.append("DataFrame is empty")
    elif len(df) < 1000:
        warnings.append(f"Low row count: {len(df)}")
    
    # Check for duplicate keys
    if 'row_idx' in df.columns and 'timestamp' in df.columns:
        dups = df.duplicated(subset=['row_idx', 'timestamp']).sum()
        if dups > 0:
            errors.append(f"Found {dups} duplicate (row_idx, timestamp) pairs")
    
    return ValidationResult(
        date=date,
        valid=len(errors) == 0,
        rows=len(df),
        columns=len(df.columns),
        null_pct=null_pct,
        warnings=warnings,
        errors=errors,
    )


def validate_batch(
    dates: List[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> List[ValidationResult]:
    """Validate training data for multiple dates.
    
    Args:
        dates: List of dates to validate
        output_dir: Directory containing training.parquet files
        verbose: Print progress
        
    Returns:
        List of ValidationResults.
    """
    results = []
    
    for date in dates:
        result = validate_training_data(date, output_dir)
        results.append(result)
        
        if verbose:
            status = "✓" if result.valid else "✗"
            if result.valid:
                warn_count = len(result.warnings)
                warn_str = f" ({warn_count} warnings)" if warn_count > 0 else ""
                print(f"  {status} {date}: {result.rows:,} rows, {result.columns} cols{warn_str}")
            else:
                print(f"  {status} {date}: {', '.join(result.errors)}")
    
    return results


# ============================================================================
# Report command
# ============================================================================

def generate_report(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_dir: Path = DEFAULT_TARGET_DIR,
) -> Dict[str, Any]:
    """Generate a summary report of the training pipeline.
    
    Args:
        output_dir: Directory containing training.parquet files
        target_dir: Directory containing target.parquet files
        
    Returns:
        Dict with report data.
    """
    target_dates = get_available_target_dates(target_dir)
    training_dates = get_available_training_dates(output_dir)
    
    # Missing dates (have target but no training)
    missing_dates = sorted(set(target_dates) - set(training_dates))
    
    # Validate all training dates
    validation_results = []
    for date in training_dates:
        result = validate_training_data(date, output_dir)
        validation_results.append(result.to_dict())
    
    # Compute summary statistics
    valid_count = sum(1 for r in validation_results if r['valid'])
    invalid_count = len(validation_results) - valid_count
    
    total_rows = sum(r['rows'] for r in validation_results)
    
    # Get expected columns
    expected_cols = get_training_feature_columns()
    
    report = {
        'generated_at': dt.now().isoformat(),
        'summary': {
            'target_dates_available': len(target_dates),
            'training_dates_assembled': len(training_dates),
            'missing_dates': len(missing_dates),
            'valid_training_files': valid_count,
            'invalid_training_files': invalid_count,
            'total_rows': total_rows,
            'expected_feature_columns': len(expected_cols),
        },
        'date_coverage': {
            'first_target_date': target_dates[0] if target_dates else None,
            'last_target_date': target_dates[-1] if target_dates else None,
            'first_training_date': training_dates[0] if training_dates else None,
            'last_training_date': training_dates[-1] if training_dates else None,
            'missing_dates': missing_dates,
        },
        'validation_results': validation_results,
    }
    
    return report


# ============================================================================
# CLI entry points
# ============================================================================

def cmd_assemble(args):
    """Handle 'assemble' command."""
    # Determine dates to process
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        dates = get_date_range(args.start_date, args.end_date)
    elif args.all:
        dates = get_available_target_dates()
        if not dates:
            print("No target dates found.")
            return 1
    else:
        print("Error: Must specify --date, --start-date/--end-date, or --all")
        return 1
    
    print(f"Assembling training data for {len(dates)} date(s)...")
    print("=" * 60)
    
    results = assemble_batch(
        dates=dates,
        city=args.city,
        include_bands=args.include_bands,
        no_spatial=args.no_spatial,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        verbose=True,
        parallel=args.parallel,
        max_workers=args.workers,
    )
    
    # Summary
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count
    total_rows = sum(r.rows for r in results if r.success)
    total_time = sum(r.elapsed_seconds for r in results)
    
    print("\n" + "=" * 60)
    print(f"Summary: {success_count}/{len(results)} succeeded")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total time: {total_time:.1f}s")
    
    if fail_count > 0:
        print(f"\nFailed dates:")
        for r in results:
            if not r.success:
                print(f"  - {r.date}: {r.error}")
    
    return 0 if fail_count == 0 else 1


def cmd_validate(args):
    """Handle 'validate' command."""
    # Determine dates to validate
    if args.date:
        dates = [args.date]
    elif args.all:
        dates = get_available_training_dates(Path(args.output_dir))
        if not dates:
            print("No training files found.")
            return 1
    else:
        print("Error: Must specify --date or --all")
        return 1
    
    print(f"Validating training data for {len(dates)} date(s)...")
    print("=" * 60)
    
    results = validate_batch(
        dates=dates,
        output_dir=Path(args.output_dir),
        verbose=True,
    )
    
    # Summary
    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count
    warning_count = sum(len(r.warnings) for r in results)
    
    print("\n" + "=" * 60)
    print(f"Summary: {valid_count}/{len(results)} valid")
    if warning_count > 0:
        print(f"  Total warnings: {warning_count}")
    
    if invalid_count > 0:
        print(f"\nInvalid dates:")
        for r in results:
            if not r.valid:
                for err in r.errors:
                    print(f"  - {r.date}: {err}")
    
    # Show detailed warnings if requested
    if args.verbose and warning_count > 0:
        print(f"\nWarnings:")
        for r in results:
            for w in r.warnings:
                print(f"  - {r.date}: {w}")
    
    return 0 if invalid_count == 0 else 1


def cmd_report(args):
    """Handle 'report' command."""
    print("Generating pipeline report...")
    
    report = generate_report(
        output_dir=Path(args.output_dir),
        target_dir=Path(args.target_dir),
    )
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")
    else:
        # Print summary to stdout
        s = report['summary']
        print("\n" + "=" * 60)
        print("Pipeline Report")
        print("=" * 60)
        print(f"\nDate Coverage:")
        print(f"  Target dates available: {s['target_dates_available']}")
        print(f"  Training dates assembled: {s['training_dates_assembled']}")
        print(f"  Missing dates: {s['missing_dates']}")
        
        print(f"\nValidation:")
        print(f"  Valid files: {s['valid_training_files']}")
        print(f"  Invalid files: {s['invalid_training_files']}")
        
        print(f"\nData Volume:")
        print(f"  Total rows: {s['total_rows']:,}")
        print(f"  Expected features: {s['expected_feature_columns']}")
        
        if report['date_coverage']['missing_dates']:
            print(f"\nMissing dates: {', '.join(report['date_coverage']['missing_dates'])}")
    
    return 0


def cmd_validate_splits(args):
    """Handle 'validate-splits' command - verify temporal split boundaries."""
    
    # If --summary flag, just print split configuration
    if args.summary:
        print_split_summary()
        return 0
    
    # Load training data to validate
    output_dir = Path(args.output_dir)
    
    if args.file:
        # Validate a specific parquet file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return 1
        
        print(f"Validating split boundaries in: {file_path.name}")
        df = pd.read_parquet(file_path)
        
        if args.timestamp_col not in df.columns:
            print(f"Error: Timestamp column '{args.timestamp_col}' not found", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
            return 1
        
        try:
            results = validate_splits(df, timestamp_col=args.timestamp_col, strict=not args.no_strict)
            
            # Print results
            print("\n" + "=" * 60)
            print("Split Validation Results")
            print("=" * 60)
            print(f"Valid: {results['valid']}")
            print(f"Total rows: {results['total_rows']:,}")
            print(f"Unassigned rows: {results['unassigned_rows']:,}")
            print(f"\nDate range: {results['date_range']['min']} to {results['date_range']['max']}")
            print(f"\nSplit distribution:")
            for split, count in results['split_counts'].items():
                pct = 100 * count / results['total_rows'] if results['total_rows'] > 0 else 0
                print(f"  {split:12} {count:8,} rows ({pct:5.1f}%)")
            
            if results['issues']:
                print(f"\nWARNING: Issues found ({len(results['issues'])}):")
                for issue in results['issues']:
                    print(f"  - {issue}")
                return 1
            else:
                print("\n✓ All split invariants satisfied")
                return 0
                
        except AssertionError as e:
            print(f"\nERROR: Split validation failed:", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            return 1
    
    elif args.splits:
        # Validate that train/val/test DataFrames have no leakage
        # Expects three files: train.parquet, val.parquet, test.parquet
        train_file = output_dir / "train.parquet"
        val_file = output_dir / "val.parquet"
        test_file = output_dir / "test.parquet"
        
        missing = []
        if not train_file.exists():
            missing.append("train.parquet")
        if not val_file.exists():
            missing.append("val.parquet")
        if not test_file.exists():
            missing.append("test.parquet")
        
        if missing:
            print(f"Error: Missing split files: {', '.join(missing)}", file=sys.stderr)
            print(f"Expected in: {output_dir}", file=sys.stderr)
            return 1
        
        print(f"Loading train/val/test splits from: {output_dir}")
        train_df = pd.read_parquet(train_file)
        val_df = pd.read_parquet(val_file)
        test_df = pd.read_parquet(test_file)
        
        print(f"  Train: {len(train_df):,} rows")
        print(f"  Val:   {len(val_df):,} rows")
        print(f"  Test:  {len(test_df):,} rows")
        
        try:
            assert_no_leakage(train_df, val_df, test_df, timestamp_col=args.timestamp_col)
            print("\n✓ No temporal leakage detected")
            print("✓ 48h gaps enforced between splits")
            print("✓ Temporal ordering verified (train < val < test)")
            return 0
        except AssertionError as e:
            print(f"\nERROR: Leakage detected:", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            return 1
    
    else:
        # No specific validation mode selected
        print("Error: Must specify --file, --splits, or --summary", file=sys.stderr)
        print("Run with --help for usage", file=sys.stderr)
        return 1


def _freeze_features_after_splits(
    output_base_dir: Path,
    splits_metadata: Dict[str, Any],
) -> bool:
    """Freeze feature schema after splits are materialized.
    
    This function:
    1. Freezes schema from canonical definition
    2. Validates against actual split dataframes
    3. Persists to artifacts/v1/features/
    
    Args:
        output_base_dir: Base directory where splits were written.
        splits_metadata: Metadata returned from materialize_splits_logged.
    
    Returns:
        True if successful, False if warnings occurred.
    """
    import pandas as pd
    from datetime import datetime
    
    # 1. Freeze from actual split data (not canonical definition)
    print("\n1. Freezing schema from actual split data...")
    
    # Load a representative file to get actual columns
    train_path = output_base_dir / "train"
    if not train_path.exists():
        raise ValueError(f"Train split directory not found: {train_path}")
    
    parquet_files = sorted(train_path.glob("*_training.parquet"))
    if not parquet_files:
        raise ValueError(f"No training.parquet files found in {train_path}")
    
    # Load middle file to get representative schema
    sample_file = parquet_files[len(parquet_files) // 2]
    print(f"   Sampling schema from: {sample_file.name}")
    sample_df = pd.read_parquet(sample_file)
    
    # Build metadata
    freeze_metadata = {
        "version": "v1",
        "model": "lgbm_baseline",
        "frozen_at": datetime.utcnow().isoformat() + "Z",
        "frozen_after_splits": True,
        "include_landsat": True,
        "include_sentinel": True,
        "sample_file": str(sample_file),
        "split_artifacts": {
            split_name: {
                "path": str(meta["path"]),
                "rows": meta["rows"],
                "date_range": f"{meta.get('min_timestamp', 'N/A')} to {meta.get('max_timestamp', 'N/A')}",
            }
            for split_name, meta in splits_metadata.items()
            if split_name in ["train", "val", "test"]  # Only include actual split dictionaries
        },
    }
    
    # Freeze from actual dataframe
    from nalanda.feature_freeze import freeze_from_dataframe
    schema = freeze_from_dataframe(sample_df, metadata=freeze_metadata)
    print(f"   ✓ Frozen {schema.n_features} features from actual data")
    print(f"   ✓ Schema hash: {schema.schema_hash}")
    
    # 2. Validate against actual split dataframes
    print("\n2. Validating against materialized splits...")
    has_warnings = False
    
    split_dfs = {}
    # Use correct split names: train, val, test (not validation)
    for split_name in ["train", "val", "test"]:
        split_path = output_base_dir / split_name
        
        if not split_path.exists():
            print(f"   ⚠ Skipping {split_name} (directory not found: {split_path})")
            has_warnings = True
            continue
        
        # Load a sample to validate schema (first few files from directory)
        if split_path.is_dir():
            parquet_files = sorted(split_path.glob("*_training.parquet"))
            if not parquet_files:
                print(f"   ⚠ No training.parquet files found in {split_path}")
                has_warnings = True
                continue
            # Sample middle file to get representative data
            sample_path = parquet_files[len(parquet_files) // 2]
        else:
            sample_path = split_path
        
        print(f"   Loading {split_name} sample from {sample_path.name}...")
        df = pd.read_parquet(sample_path)
        split_dfs[split_name] = df
        
        # Validate this split
        try:
            schema.validate(df, strict=False)
            print(f"   ✓ {split_name}: schema validated ({len(df)} rows, {len(df.columns)} cols)")
        except ValueError as e:
            print(f"   ERROR: {split_name}: validation failed")
            print(f"      {e}")
            raise
    
    # Cross-validate splits have consistent schemas
    if len(split_dfs) >= 2:
        print("\n3. Cross-validating split consistency...")
        try:
            # Use correct split name: 'val' not 'validation'
            validate_splits_schema(
                train_df=split_dfs.get("train", split_dfs[list(split_dfs.keys())[0]]),
                val_df=split_dfs.get("val", split_dfs[list(split_dfs.keys())[0]]),
                test_df=split_dfs.get("test", split_dfs[list(split_dfs.keys())[0]]),
                schema=schema,
            )
            print("   ✓ All splits have consistent schemas")
        except ValueError as e:
            print(f"   ERROR: Schema mismatch across splits:")
            print(f"      {e}")
            raise
    else:
        print("\n3. Skipping cross-validation (need at least 2 splits)")
        has_warnings = True
    
    # 4. Persist frozen schema
    print("\n4. Persisting frozen schema...")
    features_dir = output_base_dir / "v1" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    schema_path = features_dir / "feature_schema.json"
    schema.save(schema_path)
    print(f"   ✓ Saved to: {schema_path}")
    
    # Also save a human-readable feature list
    feature_list_path = features_dir / "feature_list.txt"
    with open(feature_list_path, "w") as f:
        f.write(f"# Frozen Feature List\n")
        f.write(f"# Schema hash: {schema.schema_hash}\n")
        f.write(f"# Created: {schema.created_at}\n")
        f.write(f"# Total features: {schema.n_features}\n\n")
        for i, feature in enumerate(schema.feature_list):
            f.write(f"{i:3d}  {feature}\n")
    print(f"   ✓ Feature list: {feature_list_path}")
    
    # Save metadata separately for easy inspection
    metadata_path = features_dir / "freeze_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(freeze_metadata, f, indent=2)
    print(f"   ✓ Metadata: {metadata_path}")
    
    return not has_warnings


def cmd_create_splits(args):
    """Handle 'create-splits' command - materialize train/val/test splits."""
    
    print("Materializing train/validation/test splits...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()
    
    try:
        splits_metadata = materialize_splits_logged(
            input_dir=Path(args.input_dir),
            output_base_dir=Path(args.output_dir),
            timestamp_col=args.timestamp_col,
        )
        
        print("\n" + "="*60)
        print("SPLIT SUMMARY")
        print("="*60)
        
        total_rows = sum(s["rows"] for s in splits_metadata.values())
        total_size = sum(s["size_mb"] for s in splits_metadata.values())
        
        for split_name in ["train", "validation", "test"]:
            if split_name in splits_metadata:
                s = splits_metadata[split_name]
                pct = 100 * s["rows"] / total_rows if total_rows > 0 else 0
                print(f"\n{split_name.upper()}:")
                print(f"  Rows: {s['rows']:,} ({pct:.1f}%)")
                print(f"  Date range: {s['min_timestamp']} to {s['max_timestamp']}")
                print(f"  Path: {s['path']}")
        
        print(f"\nTotal: {total_rows:,} rows, {total_size:.2f} MB")
        print("\n✓ Splits materialized successfully")
        
        # Freeze feature schema
        print("\n" + "="*60)
        print("FREEZING FEATURE SCHEMA")
        print("="*60)
        
        try:
            freeze_success = _freeze_features_after_splits(
                output_base_dir=Path(args.output_dir),
                splits_metadata=splits_metadata,
            )
            if freeze_success:
                print("\n✓ Feature schema frozen successfully")
            else:
                print("\n⚠ Feature schema freeze completed with warnings", file=sys.stderr)
        except Exception as freeze_err:
            print(f"\nERROR: Failed to freeze feature schema:", file=sys.stderr)
            print(f"  {freeze_err}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            # Don't fail the entire command, just warn
            print("\n⚠ Continuing despite freeze failure...", file=sys.stderr)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Failed to create splits:", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_stratified_split(args):
    """Handle 'stratified-split' command - create stratified random splits mixing all years."""
    import shutil
    import numpy as np
    from collections import defaultdict
    
    input_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("CREATING STRATIFIED RANDOM SPLITS")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_base_dir}")
    print(f"Split ratios: Train {args.train_pct}%, Val {args.val_pct}%, Test {args.test_pct}%")
    print(f"Random seed: {args.seed}")
    print()
    
    # Validate split ratios
    total_pct = args.train_pct + args.val_pct + args.test_pct
    if abs(total_pct - 100.0) > 0.01:
        print(f"ERROR: Split ratios must sum to 100%, got {total_pct}%", file=sys.stderr)
        return 1
    
    # Get all training files
    training_files = sorted(input_dir.glob("*_training.parquet"))
    target_files = sorted(input_dir.glob("*_target.parquet"))
    
    if not training_files:
        print(f"ERROR: No training files found in {input_dir}", file=sys.stderr)
        return 1
    
    print(f"Found {len(training_files)} training files")
    print(f"Found {len(target_files)} target files")
    print()
    
    # Load all data to get grid_id distribution for stratification
    print("Loading all data to analyze grid_id distribution (for stratification)...")
    all_data = []
    for f in training_files[:200]:  # Sample first 200 files to get distribution
        try:
            df = pd.read_parquet(f, columns=['grid_id', 'pm25_t'])
            all_data.append(df)
        except Exception as e:
            pass
    
    if all_data:
        sample_df = pd.concat(all_data, ignore_index=True)
        unique_grids = sample_df['grid_id'].nunique()
        print(f"Found ~{unique_grids} unique grid_ids in sample")
        print()
    
    # Strategy: Stratify by grid_id to ensure spatial balance
    # Group files by their grid_id distribution
    print("Stratifying by grid_id to ensure spatial balance across splits...")
    
    # Load grid_id from all files and assign to splits
    file_grid_assignments = {}
    grid_file_map = defaultdict(list)
    
    print("Mapping grid_ids to files...")
    for i, f in enumerate(training_files):
        try:
            df = pd.read_parquet(f, columns=['grid_id'])
            if len(df) > 0:
                # Get dominant grid_id(s) for this file
                grid_id = df['grid_id'].mode()[0] if not df['grid_id'].mode().empty else df['grid_id'].iloc[0]
                grid_file_map[grid_id].append(f.stem.split('_')[0])
                file_grid_assignments[f.stem.split('_')[0]] = grid_id
        except:
            pass
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(training_files)} files")
    
    print(f"Processed {len(file_grid_assignments)} files with grid assignments")
    print()
    
    # Now stratify: for each grid_id, split its files proportionally
    np.random.seed(args.seed)
    
    train_dates = []
    val_dates = []
    test_dates = []
    
    print("Splitting files by grid_id...")
    for grid_id in sorted(grid_file_map.keys()):
        dates_for_grid = grid_file_map[grid_id]
        dates_shuffled = np.random.permutation(dates_for_grid).tolist()
        
        n = len(dates_shuffled)
        n_train = int(n * args.train_pct / 100)
        n_val = int(n * args.val_pct / 100)
        
        train_dates.extend(dates_shuffled[:n_train])
        val_dates.extend(dates_shuffled[n_train:n_train + n_val])
        test_dates.extend(dates_shuffled[n_train + n_val:])
    
    print(f"✓ Stratification complete")
    print()
    
    print()
    print(f"Total splits: Train {len(train_dates)}, Val {len(val_dates)}, Test {len(test_dates)}")
    print()
    
    # Create output directories
    train_dir = output_base_dir / "train"
    val_dir = output_base_dir / "val"
    test_dir = output_base_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to splits
    def copy_files(dates_list, target_dir, split_name):
        print(f"Copying {split_name} files...")
        copied = 0
        total_rows = 0
        total_size = 0
        min_ts = None
        max_ts = None
        
        for date in dates_list:
            # Copy training file
            training_file = input_dir / f"{date}_training.parquet"
            target_file = input_dir / f"{date}_target.parquet"
            
            if training_file.exists():
                shutil.copy2(training_file, target_dir / training_file.name)
                copied += 1
                
                # Get metadata
                df = pd.read_parquet(training_file)
                total_rows += len(df)
                total_size += training_file.stat().st_size / (1024 * 1024)
                
                if 'timestamp' in df.columns:
                    ts_min = df['timestamp'].min()
                    ts_max = df['timestamp'].max()
                    min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
                    max_ts = ts_max if max_ts is None else max(max_ts, ts_max)
            
            # Copy target file if exists
            if target_file.exists():
                shutil.copy2(target_file, target_dir / target_file.name)
        
        print(f"  ✓ Copied {copied} files")
        return {
            "files": copied,
            "rows": total_rows,
            "size_mb": total_size,
            "min_timestamp": str(min_ts) if min_ts else None,
            "max_timestamp": str(max_ts) if max_ts else None,
            "path": str(target_dir),
        }
    
    train_meta = copy_files(train_dates, train_dir, "train")
    val_meta = copy_files(val_dates, val_dir, "val")
    test_meta = copy_files(test_dates, test_dir, "test")
    
    # Save split metadata
    metadata = {
        "created_at": dt.now().isoformat(),
        "split_method": "stratified_random",
        "stratify_by": "grid_id (spatial location) to ensure balanced PM2.5 distributions",
        "random_seed": args.seed,
        "train_pct": args.train_pct,
        "val_pct": args.val_pct,
        "test_pct": args.test_pct,
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
    }
    
    metadata_path = output_base_dir / "splits_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    
    total_rows = train_meta["rows"] + val_meta["rows"] + test_meta["rows"]
    
    for split_name, meta in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        pct = 100 * meta["rows"] / total_rows if total_rows > 0 else 0
        print(f"\n{split_name.upper()}:")
        print(f"  Files: {meta['files']:,}")
        print(f"  Rows: {meta['rows']:,} ({pct:.1f}%)")
        print(f"  Size: {meta['size_mb']:.2f} MB")
        print(f"  Date range: {meta['min_timestamp']} to {meta['max_timestamp']}")
        print(f"  Path: {meta['path']}")
    
    print(f"\nMetadata: {metadata_path}")
    print("\n✓ Stratified splits created successfully")
    
    # Optionally freeze schema
    if not args.no_freeze:
        print()
        print("=" * 80)
        print("FREEZING FEATURE SCHEMA")
        print("=" * 80)
        
        try:
            freeze_success = _freeze_features_after_splits(
                output_base_dir=output_base_dir,
                splits_metadata=metadata,
            )
            if freeze_success:
                print("\n✓ Feature schema frozen successfully")
            else:
                print("\n⚠ Feature schema freeze completed with warnings", file=sys.stderr)
        except Exception as freeze_err:
            print(f"\nERROR: Failed to freeze feature schema:", file=sys.stderr)
            print(f"  {freeze_err}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            print("\n⚠ Continuing despite freeze failure...", file=sys.stderr)
    
    return 0


def cmd_status(args):
    """Handle 'status' command - quick overview."""
    target_dates = get_available_target_dates()
    training_dates = get_available_training_dates()
    
    print("Pipeline Status")
    print("=" * 60)
    print(f"\nTarget data: {len(target_dates)} dates")
    if target_dates:
        print(f"  Range: {target_dates[0]} to {target_dates[-1]}")
    
    print(f"\nTraining data: {len(training_dates)} dates")
    if training_dates:
        print(f"  Range: {training_dates[0]} to {training_dates[-1]}")
    
    missing = sorted(set(target_dates) - set(training_dates))
    if missing:
        print(f"\nMissing ({len(missing)}): {', '.join(missing[:5])}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    else:
        print("\n✓ All target dates have training data")
    
    return 0


def cmd_freeze_schema(args):
    """Handle 'freeze-schema' command - freeze feature schema from splits."""
    import json
    from datetime import datetime
    
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir) if args.output_dir else splits_dir / "v1" / "features"
    
    print("=" * 80)
    print("FREEZING FEATURE SCHEMA FROM SPLITS")
    print("=" * 80)
    print(f"Splits dir: {splits_dir}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Check that splits exist
    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"
    test_dir = splits_dir / "test"
    
    if not train_dir.exists():
        print(f"ERROR: Train split directory not found: {train_dir}", file=sys.stderr)
        return 1
    
    # Load metadata if available
    metadata_file = splits_dir / "splits_metadata.json"
    splits_metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata_content = json.load(f)
            # Extract the splits dictionary
            if isinstance(metadata_content, dict) and "train" in metadata_content:
                splits_metadata = metadata_content
            else:
                print(f"⚠ Could not parse splits_metadata.json", file=sys.stderr)
    
    # Call the freeze function
    try:
        freeze_success = _freeze_features_after_splits(
            output_base_dir=splits_dir,
            splits_metadata=splits_metadata,
        )
        
        if freeze_success:
            print("\n✓ Feature schema frozen successfully")
            return 0
        else:
            print("\n⚠ Feature schema freeze completed with warnings", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"\n❌ Failed to freeze feature schema:", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Training data pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check pipeline status
  python -m nalanda.cli status
  
  # Assemble single date
  python -m nalanda.cli assemble --date 2021-06-12
  
  # Assemble date range
  python -m nalanda.cli assemble --start-date 2021-06-07 --end-date 2021-06-18
  
  # Assemble all available target dates
  python -m nalanda.cli assemble --all
  
  # Validate all training data
  python -m nalanda.cli validate --all
  
  # Generate JSON report
  python -m nalanda.cli report --output report.json
""",
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.set_defaults(func=cmd_status)
    
    # Assemble command
    assemble_parser = subparsers.add_parser('assemble', help='Assemble training data')
    assemble_parser.add_argument('--date', '-d', help='Single date (YYYY-MM-DD)')
    assemble_parser.add_argument('--start-date', help='Start date for range')
    assemble_parser.add_argument('--end-date', help='End date for range')
    assemble_parser.add_argument('--all', action='store_true', help='Process all available dates')
    assemble_parser.add_argument('--city', '-c', default='mumbai', help='City name')
    assemble_parser.add_argument('--include-bands', action='store_true', help='Include raw Landsat bands')
    assemble_parser.add_argument('--no-spatial', action='store_true', help='Skip Sentinel spatial features')
    assemble_parser.add_argument('--output-dir', '-o', default=str(DEFAULT_OUTPUT_DIR), help='Output directory')
    assemble_parser.add_argument('--dry-run', action='store_true', help='Compute but do not save')
    assemble_parser.add_argument('--parallel', '-p', action='store_true', help='Process dates in parallel')
    assemble_parser.add_argument('--workers', '-w', type=int, help='Max parallel workers (default: CPU count)')
    assemble_parser.set_defaults(func=cmd_assemble)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate training data')
    validate_parser.add_argument('--date', '-d', help='Single date (YYYY-MM-DD)')
    validate_parser.add_argument('--all', action='store_true', help='Validate all available dates')
    validate_parser.add_argument('--output-dir', '-o', default=str(DEFAULT_OUTPUT_DIR), help='Training data directory')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed warnings')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate pipeline report')
    report_parser.add_argument('--output', '-o', help='Output JSON file (omit for stdout)')
    report_parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Training data directory')
    report_parser.add_argument('--target-dir', default=str(DEFAULT_TARGET_DIR), help='Target data directory')
    report_parser.set_defaults(func=cmd_report)
    
    # Validate-splits command
    validate_splits_parser = subparsers.add_parser(
        'validate-splits',
        help='Validate temporal split boundaries and check for leakage'
    )
    validate_splits_parser.add_argument(
        '--file', '-f',
        help='Validate split boundaries in a single parquet file'
    )
    validate_splits_parser.add_argument(
        '--splits', '-s',
        action='store_true',
        help='Validate train/val/test split files for leakage (expects train.parquet, val.parquet, test.parquet)'
    )
    validate_splits_parser.add_argument(
        '--summary',
        action='store_true',
        help='Print split configuration summary and exit'
    )
    validate_splits_parser.add_argument(
        '--output-dir', '-o',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory containing split files (for --splits mode)'
    )
    validate_splits_parser.add_argument(
        '--timestamp-col',
        default='timestamp',
        help='Name of the timestamp column (default: timestamp)'
    )
    validate_splits_parser.add_argument(
        '--no-strict',
        action='store_true',
        help='Report issues as warnings instead of failing'
    )
    validate_splits_parser.set_defaults(func=cmd_validate_splits)
    
    # Create-splits command
    create_splits_parser = subparsers.add_parser(
        'create-splits',
        help='Materialize train/validation/test splits from assembled training data'
    )
    create_splits_parser.add_argument(
        '--input-dir', '-i',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Input directory containing date-partitioned training parquet files'
    )
    create_splits_parser.add_argument(
        '--output-dir', '-o',
        default=str(Path(__file__).parent.parent / "artifacts"),
        help='Output base directory (creates train/, val/, test/ subdirs)'
    )
    create_splits_parser.add_argument(
        '--timestamp-col',
        default='timestamp',
        help='Name of the timestamp column (default: timestamp)'
    )
    create_splits_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed error traces'
    )
    create_splits_parser.set_defaults(func=cmd_create_splits)
    
    # Stratified-split command
    stratified_split_parser = subparsers.add_parser(
        'stratified-split',
        help='Create stratified random splits mixing all years (for non-temporal models)'
    )
    stratified_split_parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Input directory containing all training parquet files (e.g., artifacts/v1/traininig/)'
    )
    stratified_split_parser.add_argument(
        '--output-dir', '-o',
        default=str(Path(__file__).parent.parent / "artifacts"),
        help='Output base directory (creates train/, val/, test/ subdirs)'
    )
    stratified_split_parser.add_argument(
        '--train-pct',
        type=float,
        default=70.0,
        help='Train split percentage (default: 70.0)'
    )
    stratified_split_parser.add_argument(
        '--val-pct',
        type=float,
        default=15.0,
        help='Validation split percentage (default: 15.0)'
    )
    stratified_split_parser.add_argument(
        '--test-pct',
        type=float,
        default=15.0,
        help='Test split percentage (default: 15.0)'
    )
    stratified_split_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    stratified_split_parser.add_argument(
        '--no-freeze',
        action='store_true',
        help='Skip feature schema freezing after split'
    )
    stratified_split_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed error traces'
    )
    stratified_split_parser.set_defaults(func=cmd_stratified_split)
    
    # Freeze-schema command
    freeze_schema_parser = subparsers.add_parser(
        'freeze-schema',
        help='Freeze feature schema from materialized splits'
    )
    freeze_schema_parser.add_argument(
        '--splits-dir',
        required=True,
        help='Directory containing materialized splits (train/, val/, test/)'
    )
    freeze_schema_parser.add_argument(
        '--output-dir',
        help='Output directory for schema (defaults to splits-dir/v1/features/)'
    )
    freeze_schema_parser.set_defaults(func=cmd_freeze_schema)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
