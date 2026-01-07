"""LightGBM trainer with frozen feature schema enforcement.

This module implements the training pipeline for PM2.5 prediction:
- Loads frozen feature schema (enforces immutable feature contract)
- Chunked loading of training data (memory efficient)
- Trust-weighted training with sqrt(target_trust) sample weights
- Persists model + schema + metadata as a versioned bundle

Usage (as script):
    python -m dwarka.trainer \\
        --schema artifacts/v1/features/feature_schema.json \\
        --train-dir artifacts/v1/training \\
        --val-dir artifacts/v1/training \\
        --output-dir dwarka/models/baseline_v1

Usage (as module):
    from dwarka.trainer import train_baseline
    
    results = train_baseline(
        schema_path=Path("artifacts/v1/features/feature_schema.json"),
        train_dir=Path("artifacts/v1/training"),
        val_dir=Path("artifacts/v1/training"),
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
    """Get sorted list of date directories for a given split.
    
    Args:
        data_dir: Base directory containing date-partitioned data
        split_name: One of 'train', 'validation', 'test'
        
    Returns:
        Sorted list of date directories belonging to the split
    """
    split_start, split_end = SPLIT_RANGES[split_name]
    
    date_dirs = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        
        try:
            date = pd.Timestamp(d.name)
            if split_start <= date <= split_end:
                date_dirs.append(d)
        except (ValueError, TypeError):
            # Skip non-date directories
            continue
    
    return date_dirs


def load_date_chunk(
    date_dirs: List[Path],
    schema: FrozenFeatureSchema,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load a chunk of dates and extract X, y, weights.
    
    Args:
        date_dirs: List of date directories to load
        schema: Frozen feature schema
        target_col: Name of target column
        weight_col: Name of weight column
        
    Returns:
        Tuple of (X, y, weights)
    """
    dfs = []
    
    for date_dir in date_dirs:
        training_path = date_dir / "training.parquet"
        if training_path.exists():
            df = pd.read_parquet(training_path)
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No training.parquet files found in {len(date_dirs)} directories")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort for reproducibility
    if "row_idx" in combined.columns and "timestamp" in combined.columns:
        combined = combined.sort_values(["row_idx", "timestamp"]).reset_index(drop=True)
    
    # Extract features using frozen schema
    X = schema.extract_X(combined)
    
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
) -> Iterator[Tuple[pd.DataFrame, pd.Series, np.ndarray, int]]:
    """Iterate over training data in chunks.
    
    Args:
        data_dir: Base directory containing date-partitioned data
        schema: Frozen feature schema
        chunk_size_days: Number of days per chunk
        target_col: Target column name
        weight_col: Weight column name
        
    Yields:
        Tuple of (X, y, weights, chunk_idx)
    """
    date_dirs = get_date_dirs_for_split(data_dir, "train")
    
    if not date_dirs:
        raise ValueError(f"No training date directories found in {data_dir}")
    
    logger.info(f"Found {len(date_dirs)} training date directories")
    logger.info(f"Date range: {date_dirs[0].name} to {date_dirs[-1].name}")
    
    # Chunk the date directories
    for chunk_idx, start_idx in enumerate(range(0, len(date_dirs), chunk_size_days)):
        chunk_dirs = date_dirs[start_idx : start_idx + chunk_size_days]
        
        logger.info(f"Loading chunk {chunk_idx + 1}: {chunk_dirs[0].name} to {chunk_dirs[-1].name} ({len(chunk_dirs)} days)")
        
        X, y, weights = load_date_chunk(
            chunk_dirs, schema, target_col, weight_col
        )
        
        logger.info(f"  Chunk size: {len(X):,} rows")
        
        yield X, y, weights, chunk_idx


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
    date_dirs = get_date_dirs_for_split(data_dir, "validation")
    
    if not date_dirs:
        raise ValueError(f"No validation date directories found in {data_dir}")
    
    logger.info(f"Loading validation data: {len(date_dirs)} days")
    logger.info(f"Date range: {date_dirs[0].name} to {date_dirs[-1].name}")
    
    X, y, weights = load_date_chunk(date_dirs, schema, target_col, weight_col)
    
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
    date_dirs = get_date_dirs_for_split(data_dir, "validation")
    
    if not date_dirs:
        raise ValueError(f"No validation date directories found in {data_dir}")
    
    logger.info(f"Found {len(date_dirs)} validation date directories")
    logger.info(f"Date range: {date_dirs[0].name} to {date_dirs[-1].name}")
    
    # Chunk the date directories
    for chunk_idx, start_idx in enumerate(range(0, len(date_dirs), chunk_size_days)):
        chunk_dirs = date_dirs[start_idx : start_idx + chunk_size_days]
        
        logger.info(f"Loading validation chunk {chunk_idx + 1}: {chunk_dirs[0].name} to {chunk_dirs[-1].name} ({len(chunk_dirs)} days)")
        
        X, y, weights = load_date_chunk(
            chunk_dirs, schema, target_col, weight_col
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
    chunk_size_days: int = 30,
    num_iterations_per_chunk: int = 500,
    early_stopping_rounds: int = 100,
    target_col: str = "pm25_t",
    weight_col: str = "target_trust",
    val_dir: Optional[Path] = None,
    use_chunked_val: bool = False,
    val_chunk_size: int = 30,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """Train LightGBM with chunked data loading.
    
    Args:
        data_dir: Base directory with date-partitioned training data
        schema: Frozen feature schema
        val_X, val_y, val_weights: Validation data (if use_chunked_val=False)
        params: LightGBM parameters
        chunk_size_days: Days per training chunk
        num_iterations_per_chunk: Max iterations per chunk
        early_stopping_rounds: Early stopping patience
        target_col: Target column name
        weight_col: Weight column name
        val_dir: Validation directory (if use_chunked_val=True)
        use_chunked_val: Whether to use chunked validation loading
        val_chunk_size: Days per validation chunk (if use_chunked_val=True)
        
    Returns:
        Tuple of (trained model, training info)
    """
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
    
    model = None
    total_samples = 0
    n_chunks = 0
    
    for X, y, weights, chunk_idx in iter_training_chunks(
        data_dir, schema, chunk_size_days, target_col, weight_col
    ):
        total_samples += len(X)
        n_chunks += 1
        
        # Create training dataset for this chunk
        train_ds = create_lgb_dataset(
            X, y, weights,
            feature_names=schema.feature_list,
            reference=val_ds if (model is None and val_ds is not None) else None,
        )
        
        # If using chunked validation, load first validation chunk for this iteration
        if use_chunked_val and val_ds is None:
            # Load first validation chunk for early stopping
            val_chunks = list(iter_validation_chunks(
                val_dir, schema, val_chunk_size, target_col, weight_col
            ))
            if val_chunks:
                val_X_chunk, val_y_chunk, val_weights_chunk, _ = val_chunks[0]
                val_ds = create_lgb_dataset(
                    val_X_chunk, val_y_chunk, val_weights_chunk,
                    feature_names=schema.feature_list,
                )
                logger.info(f"Using validation chunk with {len(val_X_chunk):,} samples for early stopping")
        
        # Train (continue from previous model if exists)
        model, info = train_single_chunk(
            train_ds=train_ds,
            val_ds=val_ds,
            params=params,
            init_model=model,
            num_iterations=num_iterations_per_chunk,
            early_stopping_rounds=early_stopping_rounds,
        )
        
        logger.info(f"Chunk {chunk_idx + 1} complete. Best iteration: {model.best_iteration}")
        
        # Free memory
        del train_ds, X, y, weights
    
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
) -> Dict[str, Any]:
    """End-to-end training pipeline.
    
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
        
    Returns:
        Summary dict with results
    """
    if lgb is None:
        raise ImportError("lightgbm is required for training. Install with: pip install lightgbm")
    
    start_time = time.time()
    
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
        n_val_samples = 0
        # Count validation samples
        for _, val_y_chunk, _, _ in iter_validation_chunks(
            val_dir, schema, val_chunk_size,
            TRAINING_CONFIG["target_col"],
            TRAINING_CONFIG["weight_col"],
        ):
            n_val_samples += len(val_y_chunk)
        logger.info(f"Total validation samples (across chunks): {n_val_samples:,}")
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
    val_dirs = get_date_dirs_for_split(val_dir, "validation")
    val_date_range = f"{val_dirs[0].name} to {val_dirs[-1].name}" if val_dirs else "unknown"
    
    # 3. Train model
    logger.info("=" * 60)
    logger.info("STEP 3: Training LightGBM model")
    logger.info("=" * 60)
    
    params = LGB_PARAMS.copy()
    
    if use_chunked:
        model, training_info = train_chunked(
            data_dir=train_dir,
            schema=schema,
            val_X=val_X,
            val_y=val_y,
            val_weights=val_weights,
            params=params,
            chunk_size_days=chunk_size_days,
            num_iterations_per_chunk=TRAINING_CONFIG.get("iterations_per_chunk", 500),
            early_stopping_rounds=TRAINING_CONFIG["early_stopping_rounds"],
            target_col=TRAINING_CONFIG["target_col"],
            weight_col=TRAINING_CONFIG["weight_col"],
            val_dir=val_dir if use_chunked_val else None,
            use_chunked_val=use_chunked_val,
            val_chunk_size=val_chunk_size,
        )
        
        train_dirs = get_date_dirs_for_split(train_dir, "train")
        train_date_range = f"{train_dirs[0].name} to {train_dirs[-1].name}" if train_dirs else "unknown"
    else:
        # Load full training data
        train_dirs = get_date_dirs_for_split(train_dir, "train")
        train_date_range = f"{train_dirs[0].name} to {train_dirs[-1].name}" if train_dirs else "unknown"
        
        train_X, train_y, train_weights = load_date_chunk(
            train_dirs,
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
        --train-dir artifacts/v1/training \\
        --val-dir artifacts/v1/training \\
        --output-dir dwarka/models/baseline_v1

    # Train with full data in memory
    python -m dwarka.trainer \\
        --schema artifacts/v1/features/feature_schema.json \\
        --train-dir artifacts/v1/training \\
        --val-dir artifacts/v1/training \\
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
        default=Path("artifacts/v1/training"),
        help="Directory with training data",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("artifacts/v1/training"),
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
