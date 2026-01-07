"""LightGBM hyperparameter configuration.

This module defines the canonical LightGBM configuration for PM2.5 prediction.
All parameters are justified in dwarka/model_spec.md.

Usage:
    from dwarka.config import LGB_PARAMS, TRAINING_CONFIG
"""

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================================
# LightGBM Hyperparameters
# =============================================================================

LGB_PARAMS: Dict[str, Any] = {
    # ─────────────────────────────────────────────────────────────
    # OBJECTIVE & LOSS
    # ─────────────────────────────────────────────────────────────
    "objective": "huber",           # Robust to outliers in high-PM regime
    "huber_delta": 10.0,            # δ=10 µg/m³: quadratic for |e|<10, linear beyond
    "metric": ["mae", "rmse", "huber"],  # Track all three
    
    # ─────────────────────────────────────────────────────────────
    # TREE STRUCTURE (prevent grid_id memorization)
    # ─────────────────────────────────────────────────────────────
    "num_leaves": 127,              # 2^7 - 1: moderate complexity
    "max_depth": 10,                # Hard cap on tree depth
    "min_data_in_leaf": 100,        # Require 100+ samples per leaf
    "min_sum_hessian_in_leaf": 10.0,  # Hessian regularization
    
    # ─────────────────────────────────────────────────────────────
    # LEARNING RATE & ITERATIONS
    # ─────────────────────────────────────────────────────────────
    "learning_rate": 0.05,          # Conservative: allows early stopping
    "num_iterations": 5000,         # Upper bound; early stopping will terminate earlier
    
    # ─────────────────────────────────────────────────────────────
    # SAMPLING (reduce overfitting, speed up training)
    # ─────────────────────────────────────────────────────────────
    "feature_fraction": 0.7,        # Use 70% of features per tree
    "bagging_fraction": 0.8,        # Use 80% of rows per tree
    "bagging_freq": 1,              # Resample every iteration
    
    # ─────────────────────────────────────────────────────────────
    # REGULARIZATION
    # ─────────────────────────────────────────────────────────────
    "lambda_l1": 0.0,               # No L1 initially
    "lambda_l2": 5.0,               # L2 on leaf weights
    "min_gain_to_split": 0.01,      # Minimum gain to split
    "path_smooth": 1.0,             # Smoothing on leaf values
    
    # ─────────────────────────────────────────────────────────────
    # HISTOGRAM BINNING
    # ─────────────────────────────────────────────────────────────
    "max_bin": 255,                 # Default; sufficient for continuous features
    "min_data_in_bin": 10,          # Minimum samples per histogram bin
    
    # ─────────────────────────────────────────────────────────────
    # PERFORMANCE
    # ─────────────────────────────────────────────────────────────
    "num_threads": -1,              # Use all cores
    "device_type": "cpu",           # CPU training
    "verbosity": 1,                 # Show progress
    "seed": 42,                     # Reproducibility
    "deterministic": True,          # Reproducible splits
    "force_row_wise": True,         # Better for wide datasets with sampling
}


# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_CONFIG: Dict[str, Any] = {
    # Early stopping
    "early_stopping_rounds": 100,   # Stop if no improvement for 100 rounds
    "log_evaluation_period": 50,    # Log metrics every 50 iterations
    
    # Chunked loading
    "chunk_size_days": 30,          # Load 30 days at a time for training
    
    # Target and weight columns
    "target_col": "pm25_t",
    "weight_col": "target_trust",
    "timestamp_col": "timestamp",
    
    # Weight transformation
    "weight_transform": "sqrt",     # sqrt(target_trust) as sample weights
}


# =============================================================================
# Split Directories (relative to artifacts root)
# =============================================================================

SPLIT_DIRS: Dict[str, str] = {
    "train": "v1/training",         # Date-partitioned training data
    "validation": "v1/training",    # Same dir, filtered by timestamp
    "test": "v1/training",          # Same dir, filtered by timestamp
}


# =============================================================================
# Model Output Configuration
# =============================================================================

MODEL_OUTPUT_CONFIG: Dict[str, Any] = {
    "models_dir": "dwarka/models",
    "model_filename": "model.txt",
    "schema_filename": "feature_schema.json",
    "metadata_filename": "training_metadata.json",
    "manifest_filename": "manifest.json",
}
