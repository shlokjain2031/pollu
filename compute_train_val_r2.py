#!/usr/bin/env python3
"""Compute R² for training and validation data on prod_v1 model."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
import lightgbm as lgb

def get_last_n_files(directory, n=1, suffix='_training.parquet'):
    """Get last n files matching suffix, sorted by date."""
    path = Path(directory)
    files = sorted(path.glob(f"*{suffix}"))
    return files[-n:] if len(files) >= n else files

def load_features_and_target(training_files, target_files, features):
    """Load features and target from parquet files."""
    dfs_features = []
    dfs_target = []
    
    for train_file, tgt_file in zip(training_files, target_files):
        df_train = pd.read_parquet(train_file)[features]
        df_target = pd.read_parquet(tgt_file)[['pm25_t']]  # target column
        
        dfs_features.append(df_train)
        dfs_target.append(df_target)
    
    X = pd.concat(dfs_features, ignore_index=True)
    y = pd.concat(dfs_target, ignore_index=True)['pm25_t'].values
    
    return X, y

def main():
    # Paths
    model_dir = Path("dwarka/models/prod_v1")
    train_dir = Path("artifacts/stratified_splits/train")
    val_dir = Path("artifacts/stratified_splits/val")
    metadata_path = model_dir / "training_metadata.json"
    model_path = model_dir / "checkpoints/latest_model/model.txt"
    
    # Load metadata to get feature list
    print("Loading model metadata...")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    features = metadata["feature_list"]
    n_features = len(features)
    print(f"✓ Loaded {n_features} features")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    booster = lgb.Booster(model_file=str(model_path))
    print(f"✓ Model loaded with {booster.num_feature()} features")
    
    # Get last training file
    print(f"\nLoading last training chunk...")
    train_files = get_last_n_files(train_dir, n=1, suffix='_training.parquet')
    train_target_files = [f.parent / f.name.replace('_training.parquet', '_target.parquet') 
                          for f in train_files]
    
    print(f"  Files: {[f.name for f in train_files]}")
    X_train, y_train = load_features_and_target(train_files, train_target_files, features)
    print(f"  ✓ Loaded {len(X_train)} training samples")
    
    # Get last validation file
    print(f"\nLoading last validation chunk...")
    val_files = get_last_n_files(val_dir, n=1, suffix='_training.parquet')
    val_target_files = [f.parent / f.name.replace('_training.parquet', '_target.parquet') 
                        for f in val_files]
    
    print(f"  Files: {[f.name for f in val_files]}")
    X_val, y_val = load_features_and_target(val_files, val_target_files, features)
    print(f"  ✓ Loaded {len(X_val)} validation samples")
    
    # Make predictions
    print(f"\nMaking predictions...")
    y_train_pred = booster.predict(X_train)
    y_val_pred = booster.predict(X_val)
    print(f"  ✓ Predictions complete")
    
    # Compute R²
    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)
    
    print(f"\n{'='*60}")
    print(f"R² SCORES (last chunk only)")
    print(f"{'='*60}")
    print(f"Training R²:   {r2_train:.6f}")
    print(f"Validation R²: {r2_val:.6f}")
    print(f"{'='*60}\n")
    
    # Update metadata
    print("Updating training_metadata.json...")
    metadata["train_metrics"]["r2"] = float(r2_train)
    metadata["val_metrics"]["r2"] = float(r2_val)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved results to {metadata_path}")

if __name__ == "__main__":
    main()
