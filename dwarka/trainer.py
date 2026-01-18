"""LightGBM trainer with frozen feature schema enforcement.

This module implements the training pipeline for PM2.5 prediction:
- Loads frozen feature schema (enforces immutable feature contract)
- Chunked loading of training data (memory efficient)
- Trust-weighted training with sqrt(target_trust) sample weights
- Persists model + schema + metadata as a versioned bundle

Usage (as script):
    python -m dwarka.trainer \\
        --schema artifacts/v1/features/feature_schema.json \\
        --train-dir artifacts/train \\
        --val-dir artifacts/val \\
        --output-dir dwarka/models/baseline_v1

Usage (as module):
    from dwarka.trainer import train_baseline
    
    results = train_baseline(
        schema_path=Path("artifacts/v1/features/feature_schema.json"),
        train_dir=Path("artifacts/train"),
        val_dir=Path("artifacts/val"),
        output_dir=Path("dwarka/models/baseline_v1"),
    )
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("WARNING: lightgbm not installed. Training will fail.", file=sys.stderr)

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nalanda.feature_freeze import FrozenFeatureSchema
from nalanda.splits import SPLIT_RANGES, get_split
from dwarka.config import LGB_PARAMS, TRAINING_CONFIG, MODEL_OUTPUT_CONFIG


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress LightGBM's verbose leaf split warnings
logging.getLogger("lightgbm").setLevel(logging.WARNING)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingMetadata:
    """Complete metadata for a trained model."""
    
    # Model identification
    model_version: str
    model_hash: str
    created_at: str
    
    # Schema info
    schema_hash: str
    n_features: int
    feature_list: List[str]
    
    # Training params
    lgb_params: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Dataset info
    n_train_samples: int
    n_val_samples: int
    n_train_chunks: int
    train_date_range: str
    val_date_range: str
    
    # Training results
    best_iteration: int
    training_duration_seconds: float
    
    # Metrics at best iteration
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    
    # Additional info
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelManifest:
    """Manifest contract for model artifacts."""
    
    # Identification
    manifest_version: str = "1.0"
    model_version: str = ""
    model_hash: str = ""
    
    # Artifact paths (relative to model directory)
    model_file: str = "model.txt"
    schema_file: str = "feature_schema.json"
    metadata_file: str = "training_metadata.json"
    
    # Contract
    schema_hash: str = ""
    n_features: int = 0
    target_col: str = "pm25_t"
    weight_col: str = "target_trust"
    
    # Timestamps
    created_at: str = ""
    
    # Checksums for integrity
    checksums: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Checkpointing Functions
# =============================================================================

def generate_session_id(model_version: str = "baseline") -> str:
    """Generate unique training session ID.
    
    Format: {model_version}_{YYYYMMDD}_{HHMMSS}
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{model_version}_{timestamp}"


def save_chunk_checkpoint(
    model: "lgb.Booster",
    chunk_idx: int,
    chunk_date_range: Tuple[str, str],
    output_dir: Path,
    training_info: Dict[str, Any],
    data_info: Dict[str, Any],
) -> Path:
    """Save model and metadata after each chunk completes (only latest kept).
    
    Args:
        model: Trained LightGBM model
        chunk_idx: Chunk number (0-based)
        chunk_date_range: (start_date, end_date) strings
        output_dir: Base output directory (e.g., artifacts/models/baseline_v1/)
        training_info: {best_iteration, best_score, ...}
        data_info: {n_rows, n_features, dropped_cols, ...}
    
    Returns:
        Path to latest_model directory
    """
    checkpoints_dir = output_dir / "checkpoints"
    latest_model_dir = checkpoints_dir / "latest_model"
    latest_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model (overwrites previous)
    model_path = latest_model_dir / "model.txt"
    temp_model_path = latest_model_dir / "model.txt.tmp"
    model.save_model(str(temp_model_path))
    temp_model_path.rename(model_path)
    
    # Build chunk metadata
    chunk_metadata = {
        "chunk_idx": chunk_idx,
        "date_range": {
            "start": chunk_date_range[0],
            "end": chunk_date_range[1],
        },
        "data_info": {
            "n_rows": data_info.get("n_rows", 0),
            "n_features_before_drop": data_info.get("n_features_before_drop", 0),
            "n_features_after_drop": data_info.get("n_features_after_drop", 0),
            "dropped_non_numeric": data_info.get("dropped_non_numeric", []),
            "dropped_all_nan": data_info.get("dropped_all_nan", []),
        },
        "training_info": {
            "best_iteration": training_info.get("best_iteration", 0),
            "early_stopped": training_info.get("best_iteration", 0) < training_info.get("total_iterations", 5000),
        },
        "metrics": {
            "train": training_info.get("best_score", {}).get("train", {}),
            "val": training_info.get("best_score", {}).get("val", {}),
        },
        "timing": {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "duration_seconds": data_info.get("duration_seconds", 0),
        },
    }
    
    # Save metadata to latest_model/ (overwrites previous)
    metadata_path = latest_model_dir / "metadata.json"
    temp_metadata_path = latest_model_dir / "metadata.json.tmp"
    with open(temp_metadata_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    temp_metadata_path.rename(metadata_path)
    
    logger.info(f"  Checkpoint saved: {latest_model_dir}")
    
    return latest_model_dir


def update_progress(
    output_dir: Path,
    chunk_idx: int,
    total_chunks: int,
    chunk_date_range: Tuple[str, str],
    training_info: Dict[str, Any],
    session_id: str,
    status: str = "completed",
) -> None:
    """Update progress.json after each chunk.
    
    Args:
        output_dir: Base output directory
        chunk_idx: Just completed chunk (0-based)
        total_chunks: Total chunks to train
        chunk_date_range: (start_date, end_date) strings
        training_info: Metrics, timing, etc.
        session_id: Training session ID
        status: "completed", "failed", "in_progress"
    """
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    progress_file = checkpoints_dir / "progress.json"
    
    # Load existing progress or create new
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {
            "training_session_id": session_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_chunks": total_chunks,
            "completed_chunks": [],
            "last_completed_chunk": None,
            "chunks": {},
        }
    
    # Extract metrics safely
    best_score = training_info.get("best_score", {})
    train_metrics = best_score.get("train", {})
    val_metrics = best_score.get("val", {})
    
    # Add/update chunk info
    progress["chunks"][str(chunk_idx)] = {
        "date_range": f"{chunk_date_range[0]} to {chunk_date_range[1]}",
        "status": status,
        "best_iteration": training_info.get("best_iteration", 0),
        "train_l1": train_metrics.get("l1", 0),
        "val_l1": val_metrics.get("l1", 0),
        "val_rmse": val_metrics.get("rmse", 0),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "duration_seconds": training_info.get("duration_seconds", 0),
    }
    
    # Update completed chunks list
    if status == "completed":
        if chunk_idx not in progress["completed_chunks"]:
            progress["completed_chunks"].append(chunk_idx)
            progress["completed_chunks"].sort()
        progress["last_completed_chunk"] = chunk_idx
    
    progress["last_updated"] = datetime.utcnow().isoformat() + "Z"
    
    # Update resume info
    progress["resume_info"] = {
        "can_resume": status == "completed",
        "next_chunk": chunk_idx + 1 if status == "completed" else chunk_idx,
        "init_model_path": "checkpoints/latest_model/model.txt" if status == "completed" else None,
    }
    
    # Atomic write
    temp_progress = checkpoints_dir / "progress.json.tmp"
    with open(temp_progress, "w") as f:
        json.dump(progress, f, indent=2)
    temp_progress.rename(progress_file)
    
    # Log progress
    pct = (len(progress["completed_chunks"]) / total_chunks) * 100
    logger.info(f"  Progress: {len(progress['completed_chunks'])}/{total_chunks} chunks ({pct:.1f}%)")


def verify_checkpoint(latest_model_dir: Path) -> bool:
    """Verify checkpoint integrity.
    
    Args:
        latest_model_dir: Latest model directory to verify
        
    Returns:
        True if checkpoint is valid
    """
    if not latest_model_dir.exists():
        return False
    
    model_path = latest_model_dir / "model.txt"
    metadata_path = latest_model_dir / "metadata.json"
    
    # Check required files exist
    if not model_path.exists() or not metadata_path.exists():
        logger.warning(f"Checkpoint incomplete: {latest_model_dir}")
        return False
    
    # Check for temporary files (indicates interrupted save)
    temp_model = latest_model_dir / "model.txt.tmp"
    temp_metadata = latest_model_dir / "metadata.json.tmp"
    if temp_model.exists() or temp_metadata.exists():
        logger.warning(f"Checkpoint has temporary files: {latest_model_dir}")
        return False
    
    # Verify metadata can be loaded
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check required fields
        required_fields = ["chunk_idx", "date_range", "data_info", "training_info"]
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Checkpoint missing field '{field}': {latest_model_dir}")
                return False
        
        return True
    except Exception as e:
        logger.warning(f"Checkpoint metadata corrupt: {latest_model_dir} - {e}")
        return False


def load_checkpoint_for_resume(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint information for resuming training.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dict with resume info, or None if no valid checkpoint found
        {
            "init_model_path": Path to model file,
            "next_chunk_idx": Index of next chunk to train,
            "session_id": Original session ID,
            "completed_chunks": List of completed chunk indices,
            "total_chunks": Total chunks in training,
        }
    """
    checkpoints_dir = output_dir / "checkpoints"
    progress_file = checkpoints_dir / "progress.json"
    
    if not progress_file.exists():
        logger.info("No previous training session found")
        return None
    
    # Load progress
    try:
        with open(progress_file) as f:
            progress = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load progress.json: {e}")
        return None
    
    # Check if resumable
    if not progress.get("resume_info", {}).get("can_resume", False):
        logger.warning("Previous training session not resumable")
        return None
    
    last_chunk = progress.get("last_completed_chunk")
    if last_chunk is None:
        logger.info("No completed chunks found")
        return None
    
    # Verify latest checkpoint
    latest_model_dir = checkpoints_dir / "latest_model"
    if not verify_checkpoint(latest_model_dir):
        logger.error(f"Latest checkpoint verification failed")
        return None
    
    model_path = latest_model_dir / "model.txt"
    next_chunk = last_chunk + 1
    
    resume_info = {
        "init_model_path": model_path,
        "next_chunk_idx": next_chunk,
        "session_id": progress.get("training_session_id"),
        "completed_chunks": progress.get("completed_chunks", []),
        "total_chunks": progress.get("total_chunks", 0),
        "last_chunk_date_range": progress.get("chunks", {}).get(str(last_chunk), {}).get("date_range"),
    }
    
    logger.info(f"Found resumable checkpoint: chunk_{last_chunk}")
    logger.info(f"   Session: {resume_info['session_id']}")
    logger.info(f"   Completed: {len(resume_info['completed_chunks'])}/{resume_info['total_chunks']} chunks")
    logger.info(f"   Will resume from chunk {next_chunk}")
    
    return resume_info


def cleanup_failed_checkpoint(output_dir: Path, chunk_idx: int = None) -> None:
    """Clean up a failed/incomplete checkpoint.
    
    Args:
        output_dir: Base output directory
        chunk_idx: Ignored (kept for API compatibility)
    """
    latest_model_dir = output_dir / "checkpoints" / "latest_model"
    
    if not latest_model_dir.exists():
        return
    
    logger.info(f"Cleaning up failed checkpoint")
    
    # Remove temporary files
    temp_files = [
        latest_model_dir / "model.txt.tmp",
        latest_model_dir / "metadata.json.tmp",
    ]
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()
            logger.info(f"  Removed temp file: {temp_file.name}")
    
    # Remove checkpoint directory if incomplete
    if not verify_checkpoint(latest_model_dir):
        import shutil
        shutil.rmtree(latest_model_dir, ignore_errors=True)
        logger.info(f"  Removed incomplete checkpoint directory")


# =============================================================================
# Schema Loading
# =============================================================================

def load_frozen_schema(schema_path: Path) -> FrozenFeatureSchema:
    """Load and validate frozen feature schema.
    
    Args:
        schema_path: Path to feature_schema.json
        
    Returns:
        FrozenFeatureSchema instance
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema hash verification fails
    """
    logger.info(f"Loading frozen schema from: {schema_path}")
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    schema = FrozenFeatureSchema.load(schema_path)
    
    logger.info(f"  Schema hash: {schema.schema_hash}")
    logger.info(f"  N features: {schema.n_features}")
    logger.info(f"  Created: {schema.created_at}")
    
    return schema


# =============================================================================
# Data Loading (Chunked)
# =============================================================================

def get_date_dirs_for_split(
    data_dir: Path,
    split_name: str,
) -> List[Path]:
    """Get sorted list of training parquet files for a given split.
    
    Args:
        data_dir: Base directory containing training parquet files
        split_name: One of 'train', 'validation', 'test' (unused, kept for compatibility)
        
    Returns:
        Sorted list of training parquet file paths
    """
    # Data is now pre-split into train/val/test folders
    # Files are named like YYYY-MM-DD_training.parquet
    training_files = []
    for f in sorted(data_dir.iterdir()):
        if f.suffix == '.parquet' and '_training.parquet' in f.name:
            training_files.append(f)
    
    return training_files


def _load_parquet_file(path: Path) -> Optional[pd.DataFrame]:
    """Load a single parquet file (helper for parallel loading)."""
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        pass
    return None


def load_date_chunk(
    date_files: List[Path],
    schema: FrozenFeatureSchema,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
    n_workers: int = 4,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load a chunk of training files and extract X, y, weights.
    
    Args:
        date_files: List of training parquet files to load
        schema: Frozen feature schema
        target_col: Name of target column
        weight_col: Name of weight column
        n_workers: Number of parallel workers for file loading
        
    Returns:
        Tuple of (X, y, weights)
    """
    # Parallel file loading for better I/O performance
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_load_parquet_file, date_files))
    
    dfs = [df for df in results if df is not None]
    found_count = len(dfs)
    
    if not dfs:
        logger.warning(f"WARNING: No training files found in {len(date_files)} files")
        if date_files:
            logger.warning(f"   File range: {date_files[0].name} to {date_files[-1].name}")
        logger.warning(f"   This chunk will be skipped")
        return None, None, None  # Return None to indicate empty chunk
    
    if found_count < len(date_files):
        logger.warning(f"WARNING: Found {found_count}/{len(date_files)} training files in this chunk (skipped {len(date_files) - found_count} missing files)")
    
    # Filter out empty/all-NA DataFrames before concat to avoid FutureWarning
    dfs = [df for df in dfs if not df.empty]
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort for reproducibility
    if "row_idx" in combined.columns and "timestamp" in combined.columns:
        combined = combined.sort_values(["row_idx", "timestamp"]).reset_index(drop=True)
    
    # Extract features using frozen schema
    X = schema.extract_X(combined)
    
    # Validate and convert feature dtypes to numeric (vectorized for performance)
    non_numeric_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    all_nan_cols = X.columns[X.isna().all()].tolist()
    
    # Exclude grid_id from features (for spatial generalization testing)
    # grid_id leaks information about location, breaking spatial holdout test
    # PM2.5 lags are kept - they're IDW-interpolated from neighbors, providing legitimate temporal signal
    cols_to_drop = ['grid_id']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
    logger.info(f"Dropped columns for spatial generalization: {cols_to_drop}")
    
    # Convert non-numeric columns (except grid_id which is already dropped) to float
    non_numeric_cols = [col for col in non_numeric_cols if col not in cols_to_drop]
    if non_numeric_cols:
        # Make a copy to avoid SettingWithCopyWarning
        X = X.copy()
        for col in non_numeric_cols:
            # Try to convert to numeric (pd.to_numeric handles strings, errors='coerce' converts non-convertible to NaN)
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        if non_numeric_cols:
            logger.info(f"Converted {len(non_numeric_cols)} non-numeric columns to float: {non_numeric_cols}")
    
    # Check for all-NaN columns but DON'T drop them - LightGBM can handle NaN
    # (Schema features must be present even if all-NaN to maintain feature count consistency)
    all_nan_cols_present = [col for col in all_nan_cols if col in X.columns]
    if all_nan_cols_present:
        logger.warning(f"Found {len(all_nan_cols_present)} all-NaN feature columns: {all_nan_cols_present}")
        logger.warning("Keeping all-NaN columns (LightGBM handles NaN natively)")
    
    logger.info(f"Using {len(X.columns)} features after validation")
    
    # Extract target
    if target_col not in combined.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    y = combined[target_col]
    
    # Extract and transform weights
    if weight_col not in combined.columns:
        logger.warning(f"Weight column '{weight_col}' not found, using uniform weights")
        weights = np.ones(len(combined))
    else:
        weights = np.sqrt(combined[weight_col].values)
        # Handle any NaN weights
        weights = np.nan_to_num(weights, nan=0.0)
    
    return X, y, weights


def iter_training_chunks(
    data_dir: Path,
    schema: FrozenFeatureSchema,
    chunk_size_days: int = 30,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
) -> Iterator[Tuple[pd.DataFrame, pd.Series, np.ndarray, int, Tuple[str, str], int]]:
    """Iterate over training data in chunks.
    
    Args:
        data_dir: Base directory containing date-partitioned data
        schema: Frozen feature schema
        chunk_size_days: Number of days per chunk
        target_col: Target column name
        weight_col: Weight column name
        
    Yields:
        Tuple of (X, y, weights, chunk_idx, date_range, total_chunks)
    """
    date_files = get_date_dirs_for_split(data_dir, "train")
    
    if not date_files:
        raise ValueError(f"No training files found in {data_dir}")
    
    logger.info(f"Found {len(date_files)} training files")
    logger.info(f"File range: {date_files[0].stem.split('_')[0]} to {date_files[-1].stem.split('_')[0]}")
    
    # Calculate total chunks
    total_chunks = (len(date_files) + chunk_size_days - 1) // chunk_size_days
    logger.info(f"Total chunks: {total_chunks}")
    
    # Chunk the training files
    for chunk_idx, start_idx in enumerate(range(0, len(date_files), chunk_size_days)):
        chunk_files = date_files[start_idx : start_idx + chunk_size_days]
        date_range = (chunk_files[0].stem.split('_')[0], chunk_files[-1].stem.split('_')[0])
        
        logger.info(f"Loading chunk {chunk_idx + 1}/{total_chunks}: {date_range[0]} to {date_range[1]} ({len(chunk_files)} days)")
        
        X, y, weights = load_date_chunk(
            chunk_files, schema, target_col, weight_col
        )
        
        # Skip empty chunks
        if X is None:
            logger.info(f"  Skipping empty chunk {chunk_idx}")
            continue
        
        logger.info(f"  Chunk size: {len(X):,} rows")
        
        yield X, y, weights, chunk_idx, date_range, total_chunks


def load_validation_data(
    data_dir: Path,
    schema: FrozenFeatureSchema,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load full validation set into memory.
    
    Args:
        data_dir: Base directory containing date-partitioned data
        schema: Frozen feature schema
        target_col: Target column name
        weight_col: Weight column name
        
    Returns:
        Tuple of (X, y, weights)
    """
    date_files = get_date_dirs_for_split(data_dir, "validation")
    
    if not date_files:
        raise ValueError(f"No validation files found in {data_dir}")
    
    logger.info(f"Loading validation data: {len(date_files)} files")
    logger.info(f"File range: {date_files[0].stem.split('_')[0]} to {date_files[-1].stem.split('_')[0]}")
    
    X, y, weights = load_date_chunk(date_files, schema, target_col, weight_col)
    
    logger.info(f"Validation set size: {len(X):,} rows")
    
    return X, y, weights


def iter_validation_chunks(
    data_dir: Path,
    schema: FrozenFeatureSchema,
    chunk_size_days: int = 30,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
) -> Iterator[Tuple[pd.DataFrame, pd.Series, np.ndarray, int]]:
    """Iterate over validation data in chunks (memory-efficient).
    
    Args:
        data_dir: Base directory containing date-partitioned data
        schema: Frozen feature schema
        chunk_size_days: Number of days per chunk
        target_col: Target column name
        weight_col: Weight column name
        
    Yields:
        Tuple of (X, y, weights, chunk_idx)
    """
    date_files = get_date_dirs_for_split(data_dir, "validation")
    
    if not date_files:
        raise ValueError(f"No validation files found in {data_dir}")
    
    logger.info(f"Found {len(date_files)} validation files")
    logger.info(f"File range: {date_files[0].stem.split('_')[0]} to {date_files[-1].stem.split('_')[0]}")
    
    # Chunk the training files
    for chunk_idx, start_idx in enumerate(range(0, len(date_files), chunk_size_days)):
        chunk_files = date_files[start_idx : start_idx + chunk_size_days]
        
        logger.info(f"Loading validation chunk {chunk_idx + 1}: {chunk_files[0].stem.split('_')[0]} to {chunk_files[-1].stem.split('_')[0]} ({len(chunk_files)} files)")
        
        X, y, weights = load_date_chunk(
            chunk_files, schema, target_col, weight_col
        )
        
        logger.info(f"  Validation chunk size: {len(X):,} rows")
        
        yield X, y, weights, chunk_idx


# =============================================================================
# LightGBM Dataset Construction
# =============================================================================

def create_lgb_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    feature_names: List[str],
    reference: Optional[lgb.Dataset] = None,
) -> lgb.Dataset:
    """Create LightGBM Dataset with explicit feature names.
    
    Args:
        X: Feature matrix
        y: Target vector
        weights: Sample weights
        feature_names: Ordered feature names (from frozen schema)
        reference: Reference dataset (for validation set)
        
    Returns:
        LightGBM Dataset
    """
    return lgb.Dataset(
        data=X,
        label=y,
        weight=weights,
        feature_name=feature_names,
        reference=reference,
        free_raw_data=False,  # Keep raw data for continued training
    )


# =============================================================================
# Training
# =============================================================================

def train_single_chunk(
    train_ds: lgb.Dataset,
    val_ds: lgb.Dataset,
    params: Dict[str, Any],
    init_model: Optional[lgb.Booster] = None,
    num_iterations: int = 5000,
    early_stopping_rounds: int = 100,
    log_period: int = 50,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """Train LightGBM on a single chunk.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        params: LightGBM parameters
        init_model: Model to continue training from (for chunked training)
        num_iterations: Maximum iterations
        early_stopping_rounds: Early stopping patience
        log_period: Log metrics every N iterations
        
    Returns:
        Tuple of (trained model, training info)
    """
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=log_period),
    ]
    
    model = lgb.train(
        params=params,
        train_set=train_ds,
        num_boost_round=num_iterations,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
        init_model=init_model,
    )
    
    # Extract training info
    info = {
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
    }
    
    return model, info


def train_chunked(
    data_dir: Path,
    schema: FrozenFeatureSchema,
    val_X: Optional[pd.DataFrame],
    val_y: Optional[pd.Series],
    val_weights: Optional[np.ndarray],
    params: Dict[str, Any],
    output_dir: Path,
    chunk_size_days: int = 30,
    num_iterations_per_chunk: int = 500,
    early_stopping_rounds: int = 100,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
    val_dir: Optional[Path] = None,
    use_chunked_val: bool = False,
    val_chunk_size: int = 30,
    session_id: Optional[str] = None,
    resume_info: Optional[Dict[str, Any]] = None,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """Train LightGBM with chunked data loading and checkpointing.
    
    Args:
        data_dir: Base directory with date-partitioned training data
        schema: Frozen feature schema
        val_X, val_y, val_weights: Validation data (if use_chunked_val=False)
        params: LightGBM parameters
        output_dir: Output directory for model and checkpoints
        chunk_size_days: Days per training chunk
        num_iterations_per_chunk: Max iterations per chunk
        early_stopping_rounds: Early stopping patience
        target_col: Target column name
        weight_col: Weight column name
        val_dir: Validation directory (if use_chunked_val=True)
        use_chunked_val: Whether to use chunked validation loading
        val_chunk_size: Days per validation chunk (if use_chunked_val=True)
        session_id: Training session ID (auto-generated if None)
        resume_info: Resume info from load_checkpoint_for_resume() (if resuming)
        
    Returns:
        Tuple of (trained model, training info)
    """
    # Handle resuming from checkpoint
    if resume_info is not None:
        session_id = resume_info["session_id"]
        start_chunk_idx = resume_info["next_chunk_idx"]
        init_model_path = resume_info["init_model_path"]
        logger.info("=" * 60)
        logger.info("🔄 RESUMING FROM CHECKPOINT")
        logger.info("=" * 60)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Completed chunks: {len(resume_info['completed_chunks'])}/{resume_info['total_chunks']}")
        logger.info(f"Resuming from chunk {start_chunk_idx}")
        logger.info(f"Loading model from: {init_model_path}")
        
        # Load the checkpoint model
        if lgb is None:
            raise ImportError("lightgbm required")
        model = lgb.Booster(model_file=str(init_model_path))
        logger.info(f"Model loaded successfully (best_iteration: {model.best_iteration})")
    else:
        # Fresh training
        start_chunk_idx = 0
        model = None
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = generate_session_id()
        
        logger.info(f"Training session ID: {session_id}")
        logger.info(f"Checkpoints will be saved to: {output_dir / 'checkpoints'}")
    
    # Create validation dataset (either in-memory or prepare for chunked)
    if use_chunked_val:
        if val_dir is None:
            raise ValueError("val_dir required when use_chunked_val=True")
        val_ds = None  # Will create per-chunk
        logger.info("Using chunked validation evaluation")
    else:
        if val_X is None or val_y is None or val_weights is None:
            raise ValueError("val_X, val_y, val_weights required when use_chunked_val=False")
        val_ds = create_lgb_dataset(
            val_X, val_y, val_weights,
            feature_names=schema.feature_list,
        )
        logger.info("Using in-memory validation evaluation")
    
    total_samples = 0
    n_chunks = 0
    total_chunks = 0
    
    for X, y, weights, chunk_idx, date_range, total_chunks in iter_training_chunks(
        data_dir, schema, chunk_size_days, target_col, weight_col
    ):
        # Skip already-completed chunks when resuming
        if chunk_idx < start_chunk_idx:
            logger.info(f"Skipping chunk {chunk_idx + 1}/{total_chunks} (already completed)")
            continue
        
        chunk_start_time = time.time()
        n_features_before = len(schema.feature_list)
        n_features_after = len(X.columns)
        
        total_samples += len(X)
        n_chunks += 1
        
        # Create training dataset for this chunk
        train_ds = create_lgb_dataset(
            X, y, weights,
            feature_names=list(X.columns),  # Use actual columns after dropping
            reference=val_ds if (model is None and val_ds is not None) else None,
        )
        
        # If using chunked validation and val_ds not yet created, load first chunk ONCE
        # (This block only executes on the first training chunk iteration)
        if use_chunked_val and val_ds is None:
            # Load first validation chunk for early stopping (single iteration, not full list)
            for val_X_chunk, val_y_chunk, val_weights_chunk, _ in iter_validation_chunks(
                val_dir, schema, val_chunk_size, target_col, weight_col
            ):
                val_ds = create_lgb_dataset(
                    val_X_chunk, val_y_chunk, val_weights_chunk,
                    feature_names=schema.feature_list,
                )
                logger.info(f"Using validation chunk with {len(val_X_chunk):,} samples for early stopping")
                break  # Only need first chunk
        
        # Train (continue from previous model if exists)
        model, info = train_single_chunk(
            train_ds=train_ds,
            val_ds=val_ds,
            params=params,
            init_model=model,
            num_iterations=num_iterations_per_chunk,
            early_stopping_rounds=early_stopping_rounds,
        )
        
        chunk_duration = time.time() - chunk_start_time
        info["duration_seconds"] = chunk_duration
        
        logger.info(f"Chunk {chunk_idx + 1}/{total_chunks} complete. Best iteration: {model.best_iteration}")
        logger.info(f"  Duration: {chunk_duration:.1f}s")
        
        # === CHECKPOINTING ===
        data_info = {
            "n_rows": len(y),
            "n_features_before_drop": n_features_before,
            "n_features_after_drop": n_features_after,
            "dropped_non_numeric": [],  # TODO: track these in load_date_chunk
            "dropped_all_nan": [],
            "duration_seconds": chunk_duration,
        }
        
        # Save checkpoint
        save_chunk_checkpoint(
            model=model,
            chunk_idx=chunk_idx,
            chunk_date_range=date_range,
            output_dir=output_dir,
            training_info=info,
            data_info=data_info,
        )
        
        # Update progress
        update_progress(
            output_dir=output_dir,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            chunk_date_range=date_range,
            training_info=info,
            session_id=session_id,
            status="completed",
        )
        
        # Free memory explicitly
        del train_ds, X, y, weights
        gc.collect()  # Force garbage collection to prevent memory fragmentation
    
    training_info = {
        "n_train_samples": total_samples,
        "n_chunks": n_chunks,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
    }
    
    return model, training_info


def train_full(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    train_weights: np.ndarray,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    val_weights: np.ndarray,
    schema: FrozenFeatureSchema,
    params: Dict[str, Any],
    num_iterations: int = 5000,
    early_stopping_rounds: int = 100,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """Train LightGBM with full data in memory.
    
    Args:
        train_X, train_y, train_weights: Training data
        val_X, val_y, val_weights: Validation data
        schema: Frozen feature schema
        params: LightGBM parameters
        num_iterations: Maximum iterations
        early_stopping_rounds: Early stopping patience
        
    Returns:
        Tuple of (trained model, training info)
    """
    train_ds = create_lgb_dataset(
        train_X, train_y, train_weights,
        feature_names=schema.feature_list,
    )
    
    val_ds = create_lgb_dataset(
        val_X, val_y, val_weights,
        feature_names=schema.feature_list,
        reference=train_ds,
    )
    
    model, info = train_single_chunk(
        train_ds=train_ds,
        val_ds=val_ds,
        params=params,
        num_iterations=num_iterations,
        early_stopping_rounds=early_stopping_rounds,
    )
    
    training_info = {
        "n_train_samples": len(train_X),
        "n_chunks": 1,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
    }
    
    return model, training_info


# =============================================================================
# Model Persistence
# =============================================================================

def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def save_model_bundle(
    model: lgb.Booster,
    schema: FrozenFeatureSchema,
    output_dir: Path,
    training_info: Dict[str, Any],
    params: Dict[str, Any],
    train_date_range: str,
    val_date_range: str,
    n_val_samples: int,
    model_version: str = "v1",
    notes: str = "",
) -> Path:
    """Save model, schema, and metadata as a versioned bundle.
    
    Args:
        model: Trained LightGBM model
        schema: Frozen feature schema
        output_dir: Output directory
        training_info: Training info dict
        params: LightGBM params used
        train_date_range: Training date range string
        val_date_range: Validation date range string
        n_val_samples: Number of validation samples
        model_version: Version string
        notes: Optional notes
        
    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model bundle to: {output_dir}")
    
    # 1. Save model
    model_path = output_dir / MODEL_OUTPUT_CONFIG["model_filename"]
    model.save_model(str(model_path))
    logger.info(f"  Saved model: {model_path}")
    
    # 2. Save schema
    schema_path = output_dir / MODEL_OUTPUT_CONFIG["schema_filename"]
    schema.save(schema_path)
    logger.info(f"  Saved schema: {schema_path}")
    
    # 3. Create and save metadata
    metadata = TrainingMetadata(
        model_version=model_version,
        model_hash=compute_file_hash(model_path),
        created_at=datetime.utcnow().isoformat() + "Z",
        schema_hash=schema.schema_hash,
        n_features=schema.n_features,
        feature_list=schema.feature_list,
        lgb_params=params,
        training_config=TRAINING_CONFIG,
        n_train_samples=training_info["n_train_samples"],
        n_val_samples=n_val_samples,
        n_train_chunks=training_info["n_chunks"],
        train_date_range=train_date_range,
        val_date_range=val_date_range,
        best_iteration=training_info["best_iteration"],
        training_duration_seconds=training_info.get("duration_seconds", 0),
        train_metrics=training_info["best_score"].get("train", {}),
        val_metrics=training_info["best_score"].get("val", {}),
        notes=notes,
    )
    
    metadata_path = output_dir / MODEL_OUTPUT_CONFIG["metadata_filename"]
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    logger.info(f"  Saved metadata: {metadata_path}")
    
    # 4. Create and save manifest
    manifest = ModelManifest(
        manifest_version="1.0",
        model_version=model_version,
        model_hash=metadata.model_hash,
        model_file=MODEL_OUTPUT_CONFIG["model_filename"],
        schema_file=MODEL_OUTPUT_CONFIG["schema_filename"],
        metadata_file=MODEL_OUTPUT_CONFIG["metadata_filename"],
        schema_hash=schema.schema_hash,
        n_features=schema.n_features,
        target_col=TRAINING_CONFIG["target_col"],
        weight_col=TRAINING_CONFIG["weight_col"],
        created_at=metadata.created_at,
        checksums={
            "model": compute_file_hash(model_path),
            "schema": compute_file_hash(schema_path),
            "metadata": compute_file_hash(metadata_path),
        },
    )
    
    manifest_path = output_dir / MODEL_OUTPUT_CONFIG["manifest_filename"]
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    logger.info(f"  Saved manifest: {manifest_path}")
    
    return output_dir


# =============================================================================
# Main Training Orchestrator
# =============================================================================

def train_baseline(
    schema_path: Path,
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    model_version: str = "baseline_v1",
    use_chunked: bool = True,
    chunk_size_days: int = 30,
    use_chunked_val: bool = False,
    val_chunk_size: int = 30,
    notes: str = "",
    force_restart: bool = False,
) -> Dict[str, Any]:
    """End-to-end training pipeline with automatic checkpoint resume.
    
    Args:
        schema_path: Path to frozen feature schema
        train_dir: Directory with training data
        val_dir: Directory with validation data
        output_dir: Output directory for model bundle
        model_version: Version string
        use_chunked: Whether to use chunked training
        chunk_size_days: Days per chunk (if chunked)
        use_chunked_val: Whether to use chunked validation loading
        val_chunk_size: Days per validation chunk (if use_chunked_val=True)
        notes: Optional notes
        force_restart: If True, ignore existing checkpoints and restart training
        
    Returns:
        Summary dict with results
    """
    if lgb is None:
        raise ImportError("lightgbm is required for training. Install with: pip install lightgbm")
    
    start_time = time.time()
    
    # 0. Check for resumable checkpoint (unless force_restart)
    resume_info = None
    if use_chunked and not force_restart:
        logger.info("=" * 60)
        logger.info("STEP 0: Checking for resumable checkpoint")
        logger.info("=" * 60)
        resume_info = load_checkpoint_for_resume(output_dir)
        if resume_info:
            logger.info("Will resume from previous training session")
        else:
            logger.info("Starting fresh training session")
    elif force_restart:
        logger.info("🔄 Force restart requested - ignoring any existing checkpoints")
    
    # 1. Load frozen schema
    logger.info("=" * 60)
    logger.info("STEP 1: Loading frozen feature schema")
    logger.info("=" * 60)
    schema = load_frozen_schema(schema_path)
    
    # 2. Load validation data (chunked or full)
    logger.info("=" * 60)
    logger.info("STEP 2: Loading validation data")
    logger.info("=" * 60)
    
    if use_chunked_val:
        logger.info("Using chunked validation (memory-efficient)")
        val_X, val_y, val_weights = None, None, None
        # Count validation samples efficiently from parquet metadata
        val_files = get_date_dirs_for_split(val_dir, "validation")
        n_val_samples = sum(
            pd.read_parquet(f, columns=[]).shape[0]
            for f in val_files if f.exists() and '_training.parquet' in f.name
        )
        logger.info(f"Total validation samples (from metadata): {n_val_samples:,}")
    else:
        logger.info("Loading full validation set into memory")
        val_X, val_y, val_weights = load_validation_data(
            val_dir,
            schema,
            target_col=TRAINING_CONFIG["target_col"],
            weight_col=TRAINING_CONFIG["weight_col"],
        )
        n_val_samples = len(val_X)
    
    # Get validation date range
    val_files = get_date_dirs_for_split(val_dir, "validation")
    val_date_range = f"{val_files[0].stem.split('_')[0]} to {val_files[-1].stem.split('_')[0]}" if val_files else "unknown"
    
    # 3. Train model
    logger.info("=" * 60)
    logger.info("STEP 3: Training LightGBM model")
    logger.info("=" * 60)
    
    params = LGB_PARAMS.copy()
    
    # Remove any LightGBM-invalid parameters
    invalid_params = ["huber_delta"]  # Not a standard LightGBM param
    for param in invalid_params:
        if param in params:
            logger.warning(f"Removing invalid LightGBM parameter: {param}")
            params.pop(param)
    
    logger.info(f"Using {len(params)} LightGBM parameters")
    
    if use_chunked:
        model, training_info = train_chunked(
            data_dir=train_dir,
            schema=schema,
            val_X=val_X,
            val_y=val_y,
            val_weights=val_weights,
            params=params,
            output_dir=output_dir,
            chunk_size_days=chunk_size_days,
            num_iterations_per_chunk=TRAINING_CONFIG.get("iterations_per_chunk", 500),
            early_stopping_rounds=TRAINING_CONFIG["early_stopping_rounds"],
            target_col=TRAINING_CONFIG["target_col"],
            weight_col=TRAINING_CONFIG["weight_col"],
            val_dir=val_dir if use_chunked_val else None,
            use_chunked_val=use_chunked_val,
            val_chunk_size=val_chunk_size,
            resume_info=resume_info,
        )
        
        train_files = get_date_dirs_for_split(train_dir, "train")
        train_date_range = f"{train_files[0].stem.split('_')[0]} to {train_files[-1].stem.split('_')[0]}" if train_files else "unknown"
    else:
        # Load full training data
        train_files = get_date_dirs_for_split(train_dir, "train")
        train_date_range = f"{train_files[0].stem.split('_')[0]} to {train_files[-1].stem.split('_')[0]}" if train_files else "unknown"
        
        train_X, train_y, train_weights = load_date_chunk(
            train_files,
            schema,
            target_col=TRAINING_CONFIG["target_col"],
            weight_col=TRAINING_CONFIG["weight_col"],
        )
        
        model, training_info = train_full(
            train_X, train_y, train_weights,
            val_X, val_y, val_weights,
            schema=schema,
            params=params,
            num_iterations=LGB_PARAMS.get("num_iterations", 5000),
            early_stopping_rounds=TRAINING_CONFIG["early_stopping_rounds"],
        )
    
    training_info["duration_seconds"] = time.time() - start_time
    
    # 4. Save model bundle
    logger.info("=" * 60)
    logger.info("STEP 4: Saving model bundle")
    logger.info("=" * 60)
    
    save_model_bundle(
        model=model,
        schema=schema,
        output_dir=output_dir,
        training_info=training_info,
        params=params,
        train_date_range=train_date_range,
        val_date_range=val_date_range,
        n_val_samples=n_val_samples,
        model_version=model_version,
        notes=notes,
    )
    
    # 5. Summary
    duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Best iteration: {training_info['best_iteration']}")
    logger.info(f"Train samples: {training_info['n_train_samples']:,}")
    logger.info(f"Val samples: {n_val_samples:,}")
    
    val_metrics = training_info["best_score"].get("val", {})
    for metric, value in val_metrics.items():
        logger.info(f"Val {metric}: {value:.4f}")
    
    logger.info(f"Model saved to: {output_dir}")
    
    return {
        "success": True,
        "output_dir": str(output_dir),
        "best_iteration": training_info["best_iteration"],
        "duration_seconds": duration,
        "n_train_samples": training_info["n_train_samples"],
        "n_val_samples": n_val_samples,
        "val_metrics": val_metrics,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LightGBM PM2.5 model with frozen feature schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with chunked loading (default)
    python -m dwarka.trainer \\
        --schema artifacts/v1/features/feature_schema.json \\
        --train-dir artifacts/train \\
        --val-dir artifacts/val \\
        --output-dir dwarka/models/baseline_v1

    # Train with full data in memory
    python -m dwarka.trainer \\
        --schema artifacts/v1/features/feature_schema.json \\
        --train-dir artifacts/train \\
        --val-dir artifacts/val \\
        --output-dir dwarka/models/baseline_v1 \\
        --no-chunked
""",
    )
    
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("artifacts/v1/features/feature_schema.json"),
        help="Path to frozen feature schema JSON",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("artifacts/train"),
        help="Directory with training data",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("artifacts/val"),
        help="Directory with validation data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dwarka/models/baseline_v1"),
        help="Output directory for model bundle",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="baseline_v1",
        help="Model version string",
    )
    parser.add_argument(
        "--no-chunked",
        action="store_true",
        help="Load full training data into memory (not recommended for large datasets)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Days per training chunk (default: 30)",
    )
    parser.add_argument(
        "--no-chunked-val",
        action="store_true",
        help="Load full validation data into memory (default: chunked for memory efficiency)",
    )
    parser.add_argument(
        "--val-chunk-size",
        type=int,
        default=30,
        help="Days per validation chunk (default: 30)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes to include in metadata",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart training from scratch, ignoring any existing checkpoints",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        results = train_baseline(
            schema_path=args.schema,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            model_version=args.model_version,
            use_chunked=not args.no_chunked,
            chunk_size_days=args.chunk_size,
            use_chunked_val=not args.no_chunked_val,
            val_chunk_size=args.val_chunk_size,
            notes=args.notes,
            force_restart=args.force_restart,
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS")
        print("=" * 60)
        print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
