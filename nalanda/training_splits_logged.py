"""Logged version of materialize_splits with extensive logging and crash handling."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from nalanda.training import (
    PROJECT_ROOT,
    DEFAULT_OUTPUT_DIR,
    _determine_split_for_file,
    _copy_file_to_split,
)


def materialize_splits_logged(
    input_dir: Path = DEFAULT_OUTPUT_DIR,
    output_base_dir: Optional[Path] = None,
    timestamp_col: str = "timestamp",
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> dict:
    """Materialize train/val/test splits with extensive logging.
    
    All activity is logged to logs/materialize_splits_TIMESTAMP.log
    Crashes are captured and saved to the log file.
    
    Args:
        input_dir: Directory containing date-partitioned training parquet files
        output_base_dir: Base directory for output (creates train/, val/, test/ subdirs)
        timestamp_col: Name of timestamp column
        parallel: If True, process files in parallel
        max_workers: Max parallel workers (default: CPU count)
        
    Returns:
        Dictionary with metadata about each split (row counts, date ranges, file counts)
    """
    # Setup logging
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"materialize_splits_{timestamp}.log"
    
    # Configure logging to both file and console
    logger = logging.getLogger(f"materialize_splits_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler - captures everything including DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info("SPLIT MATERIALIZATION STARTED")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Timestamp column: {timestamp_col}")
    logger.info(f"Parallel processing: {parallel}")
    logger.info(f"Max workers: {max_workers or 'auto (CPU count)'}")
    logger.info("="*80)
    
    try:
        if output_base_dir is None:
            output_base_dir = PROJECT_ROOT / "artifacts"
        
        output_base_dir = Path(output_base_dir)
        logger.info(f"Output base directory: {output_base_dir}")
        
        # Create output directories
        train_dir = output_base_dir / "train"
        val_dir = output_base_dir / "val"
        test_dir = output_base_dir / "test"
        
        output_dirs = {
            "train": train_dir,
            "validation": val_dir,
            "test": test_dir,
        }
        
        logger.info("\nCreating output directories...")
        for split_name, d in output_dirs.items():
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"  Created: {d}")
        
        logger.info(f"\nScanning training data from: {input_dir}")
        
        # Find all parquet files
        logger.debug("Searching for date directories...")
        date_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        logger.debug(f"  Found {len(date_dirs)} date directories")
        
        if not date_dirs:
            raise ValueError(f"No date directories found in {input_dir}")
        
        all_parquet_files = []
        for date_dir in date_dirs:
            parquet_files = list(date_dir.glob("*.parquet"))
            logger.debug(f"  {date_dir.name}: {len(parquet_files)} parquet files")
            all_parquet_files.extend(parquet_files)
        
        if not all_parquet_files:
            raise ValueError(f"No parquet files found in {input_dir}")
        
        logger.info(f"  Found {len(all_parquet_files)} parquet files from {len(date_dirs)} dates")
        
        # Count file types
        target_count = sum(1 for f in all_parquet_files if 'target' in f.name.lower())
        training_count = sum(1 for f in all_parquet_files if 'training' in f.name.lower())
        other_count = len(all_parquet_files) - target_count - training_count
        logger.info(f"  File breakdown: {target_count} target, {training_count} training, {other_count} other")
        
        # Process files
        logger.info("\n" + "="*80)
        logger.info("DETERMINING SPLIT ASSIGNMENTS")
        logger.info("="*80)
        
        file_assignments = {}
        
        if parallel and len(all_parquet_files) > 1:
            logger.info(f"  Processing in parallel (workers: {max_workers or 'auto'})...")
            logger.debug(f"  Total files to process: {len(all_parquet_files)}")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(_determine_split_for_file, pq_file, timestamp_col): pq_file
                    for pq_file in all_parquet_files
                }
                
                for idx, future in enumerate(as_completed(future_to_file), 1):
                    pq_file = future_to_file[future]
                    try:
                        split_name = future.result()
                        if split_name:
                            file_assignments[pq_file] = split_name
                            logger.debug(f"  [{idx}/{len(all_parquet_files)}] {pq_file.name} → {split_name}")
                        else:
                            logger.debug(f"  [{idx}/{len(all_parquet_files)}] {pq_file.name} → SKIPPED (gap/excluded)")
                    except Exception as e:
                        logger.error(f"  ⚠️  Error processing {pq_file.name}: {e}")
                        logger.debug(traceback.format_exc())
        else:
            logger.info("  Processing sequentially...")
            for idx, pq_file in enumerate(all_parquet_files, 1):
                logger.debug(f"  [{idx}/{len(all_parquet_files)}] Processing {pq_file.name}")
                split_name = _determine_split_for_file(pq_file, timestamp_col)
                if split_name:
                    file_assignments[pq_file] = split_name
                    logger.debug(f"    → Assigned to {split_name}")
                else:
                    logger.debug(f"    → SKIPPED (gap/excluded)")
        
        # Summary of assignments
        logger.info("\nAssignment summary:")
        assignment_counts = {}
        for split_name in file_assignments.values():
            assignment_counts[split_name] = assignment_counts.get(split_name, 0) + 1
        
        for split_name, count in sorted(assignment_counts.items()):
            logger.info(f"  {split_name}: {count} files")
        
        skipped = len(all_parquet_files) - len(file_assignments)
        if skipped > 0:
            logger.info(f"  SKIPPED: {skipped} files (gap/excluded data)")
        
        # Copy files to splits
        logger.info("\n" + "="*80)
        logger.info("COPYING FILES TO SPLIT DIRECTORIES")
        logger.info("="*80)

        copy_results = []

        # Group files so we run target files first, then training files, then others.
        def _group_files_assignments(assignments: dict):
            target_group = []
            training_group = []
            other_group = []
            for sf, sn in assignments.items():
                name = sf.name.lower()
                if "target" in name:
                    target_group.append((sf, sn))
                elif "training" in name or ("train" in name and "training" not in name):
                    training_group.append((sf, sn))
                else:
                    other_group.append((sf, sn))
            return [target_group, training_group, other_group]

        groups = _group_files_assignments(file_assignments)
        logger.info(f"Grouped files: {len(groups[0])} target, {len(groups[1])} training, {len(groups[2])} other")

        # Process each group sequentially (target -> training -> others). Within each group
        # we may use parallel workers if requested.
        for group_idx, group in enumerate(groups, 1):
            if not group:
                logger.debug(f"  Group {group_idx}: EMPTY, skipping")
                continue

            group_name = "target" if group_idx == 1 else ("training" if group_idx == 2 else "other")
            logger.info(f"\n  Processing group: {group_name.upper()} ({len(group)} files)")
            logger.info("  " + "-"*60)

            if parallel and len(group) > 1:
                logger.info(f"  Using parallel processing ({max_workers or 'auto'} workers)")
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            _copy_file_to_split,
                            source_file,
                            split_name,
                            output_dirs,
                            timestamp_col
                        ): source_file
                        for source_file, split_name in group
                    }

                    completed = 0
                    for future in as_completed(future_to_file):
                        completed += 1
                        source_file = future_to_file[future]
                        try:
                            result = future.result()
                            copy_results.append(result)
                            logger.debug(f"    [{completed}/{len(group)}] Copied {source_file.name}")
                            if completed % 10 == 0:
                                logger.info(f"    Progress: {completed}/{len(group)} files completed")
                        except Exception as e:
                            logger.error(f"  ⚠️  Error copying {source_file.name}: {e}")
                            logger.debug(traceback.format_exc())
            else:
                logger.info(f"  Using sequential processing")
                for idx, (source_file, split_name) in enumerate(group, 1):
                    logger.debug(f"    [{idx}/{len(group)}] Copying {source_file.name} → {split_name}")
                    try:
                        result = _copy_file_to_split(source_file, split_name, output_dirs, timestamp_col)
                        copy_results.append(result)
                    except Exception as e:
                        logger.error(f"  ⚠️  Error copying {source_file.name}: {e}")
                        logger.debug(traceback.format_exc())
        
        # Aggregate results
        logger.info("\n" + "="*80)
        logger.info("AGGREGATING SPLIT STATISTICS")
        logger.info("="*80)
        
        split_stats = {
            "train": {"files": [], "rows": 0, "min_ts": None, "max_ts": None},
            "validation": {"files": [], "rows": 0, "min_ts": None, "max_ts": None},
            "test": {"files": [], "rows": 0, "min_ts": None, "max_ts": None},
        }
        
        logger.info(f"Processing {len(copy_results)} copy results...")
        
        for result in copy_results:
            if result["type"] == "mixed":
                for sub_result in result["results"]:
                    split = sub_result["split"]
                    split_stats[split]["files"].append(sub_result["target"])
                    split_stats[split]["rows"] += sub_result["rows"]
                    
                    min_ts = pd.Timestamp(sub_result["min_timestamp"])
                    max_ts = pd.Timestamp(sub_result["max_timestamp"])
                    
                    if split_stats[split]["min_ts"] is None or min_ts < split_stats[split]["min_ts"]:
                        split_stats[split]["min_ts"] = min_ts
                    if split_stats[split]["max_ts"] is None or max_ts > split_stats[split]["max_ts"]:
                        split_stats[split]["max_ts"] = max_ts
            
            elif result["type"] == "copy":
                split = result["split"]
                split_stats[split]["files"].append(result["target"])
                split_stats[split]["rows"] += result["rows"]
                
                min_ts = pd.Timestamp(result["min_timestamp"])
                max_ts = pd.Timestamp(result["max_timestamp"])
                
                if split_stats[split]["min_ts"] is None or min_ts < split_stats[split]["min_ts"]:
                    split_stats[split]["min_ts"] = min_ts
                if split_stats[split]["max_ts"] is None or max_ts > split_stats[split]["max_ts"]:
                    split_stats[split]["max_ts"] = max_ts
        
        # Compute directory-level checksums and sizes
        logger.info("\n" + "="*80)
        logger.info("COMPUTING CHECKSUMS AND SIZES")
        logger.info("="*80)
        
        splits_metadata = {}
        
        for split_name in ["train", "validation", "test"]:
            stats = split_stats[split_name]
            output_dir = output_dirs[split_name]
            
            if stats["rows"] == 0:
                logger.warning(f"  ⚠️  {split_name.upper()}: No data")
                continue
            
            logger.info(f"\n  Computing for {split_name.upper()}...")
            
            # Compute total size
            logger.debug(f"    Computing file sizes for {len(stats['files'])} files...")
            total_size = sum(Path(f).stat().st_size for f in stats["files"]) / (1024 * 1024)
            logger.debug(f"    Total size: {total_size:.2f} MB")
            
            # Compute combined checksum of all files
            logger.debug(f"    Computing combined SHA256 checksum...")
            hash_obj = hashlib.sha256()
            for idx, file_path in enumerate(sorted(stats["files"]), 1):
                logger.debug(f"      Hashing file {idx}/{len(stats['files'])}: {Path(file_path).name}")
                with open(file_path, "rb") as f:
                    hash_obj.update(f.read())
            combined_hash = hash_obj.hexdigest()
            logger.debug(f"    Checksum: {combined_hash}")
            
            splits_metadata[split_name] = {
                "path": str(output_dir),
                "file_count": len(stats["files"]),
                "rows": stats["rows"],
                "min_timestamp": str(stats["min_ts"]),
                "max_timestamp": str(stats["max_ts"]),
                "size_mb": round(total_size, 2),
                "sha256": combined_hash,
            }
            
            logger.info(f"  {split_name.upper()}:")
            logger.info(f"    Files: {len(stats['files'])}")
            logger.info(f"    Rows: {stats['rows']:,}")
            logger.info(f"    Date range: {stats['min_ts']} to {stats['max_ts']}")
            logger.info(f"    Size: {total_size:.2f} MB")
            logger.info(f"    SHA256: {combined_hash[:16]}...")
        
        # Save metadata
        logger.info("\n" + "="*80)
        logger.info("SAVING METADATA")
        logger.info("="*80)
        metadata_path = output_base_dir / "splits_metadata.json"
        logger.info(f"Writing to: {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(splits_metadata, f, indent=2)
        logger.info("✓ Metadata saved successfully")
        
        logger.info("\n" + "="*80)
        logger.info("SPLIT MATERIALIZATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total splits created: {len(splits_metadata)}")
        logger.info(f"Log file saved to: {log_file}")
        logger.info("="*80)
        
        return splits_metadata
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("FATAL ERROR - SPLIT MATERIALIZATION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*80)
        logger.error(f"Crash log saved to: {log_file}")
        logger.error("="*80)
        raise
