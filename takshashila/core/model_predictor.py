"""LightGBM model loading and prediction.

Loads trained model and runs inference on feature DataFrames.
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from .errors import ModelInferenceError
from .config import N_FEATURES

logger = logging.getLogger(__name__)


class ModelPredictor:
    """LightGBM model predictor with lazy loading.
    
    Validates:
    - Model file exists and loads successfully
    - Input DataFrame has correct shape (1 row, 64 features)
    - Prediction output is valid (non-NaN, positive)
    """
    
    def __init__(self, model_path: str):
        """Initialize model predictor.
        
        Args:
            model_path: Path to LightGBM model.txt file
        """
        self.model_path = Path(model_path)
        self.model: Optional[lgb.Booster] = None
        
        if not self.model_path.exists():
            raise ModelInferenceError(
                grid_id="N/A",
                timestamp_hour=None,
                reason=f"Model file not found: {model_path}"
            )
    
    def load_model(self) -> None:
        """Load LightGBM model from disk (lazy initialization).
        
        Raises:
            ModelInferenceError: If model loading fails
        """
        if self.model is not None:
            logger.debug("Model already loaded, skipping")
            return
        
        try:
            logger.info(f"Loading LightGBM model from {self.model_path}")
            self.model = lgb.Booster(model_file=str(self.model_path))
            logger.info(f"Model loaded successfully ({self.model.num_feature()} features)")
        
        except Exception as e:
            raise ModelInferenceError(
                grid_id="N/A",
                timestamp_hour=None,
                reason=f"Failed to load model: {e}"
            )
    
    def predict(self, features_df: pd.DataFrame, grid_id: str = "N/A", timestamp_hour=None) -> float:
        """Run inference on single-row feature DataFrame.
        
        Args:
            features_df: Single-row DataFrame with 64 features (ordered correctly)
            grid_id: Grid cell identifier (for error context)
            timestamp_hour: Prediction timestamp (for error context)
        
        Returns:
            PM2.5 prediction value (float)
        
        Raises:
            ModelInferenceError: If prediction fails or output is invalid
        """
        # Lazy load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Validate input shape
        if features_df.shape[0] != 1:
            raise ModelInferenceError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason=f"Expected 1 row, got {features_df.shape[0]} rows"
            )
        
        if features_df.shape[1] != N_FEATURES:
            raise ModelInferenceError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason=f"Expected {N_FEATURES} features, got {features_df.shape[1]} features"
            )
        
        # Check for NaN/inf in input
        if features_df.isnull().any().any():
            null_cols = features_df.columns[features_df.isnull().any()].tolist()
            raise ModelInferenceError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason=f"Input contains NaN values in columns: {null_cols}"
            )
        
        if np.isinf(features_df.values).any():
            raise ModelInferenceError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason="Input contains infinite values"
            )
        
        # Run prediction
        try:
            prediction = self.model.predict(features_df, num_iteration=self.model.best_iteration)
            
            # LightGBM returns array of predictions
            if len(prediction) != 1:
                raise ModelInferenceError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    reason=f"Expected 1 prediction, got {len(prediction)}"
                )
            
            pm25_value = float(prediction[0])
            
            # Validate output
            if np.isnan(pm25_value):
                raise ModelInferenceError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    reason="Model returned NaN prediction"
                )
            
            if np.isinf(pm25_value):
                raise ModelInferenceError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    reason="Model returned infinite prediction"
                )
            
            if pm25_value < 0:
                raise ModelInferenceError(
                    grid_id=grid_id,
                    timestamp_hour=timestamp_hour,
                    reason=f"Model returned negative prediction: {pm25_value}"
                )
            
            logger.debug(f"Model prediction: {pm25_value:.2f} for {grid_id} @ {timestamp_hour}")
            
            return pm25_value
        
        except ModelInferenceError:
            # Re-raise our own errors
            raise
        
        except Exception as e:
            raise ModelInferenceError(
                grid_id=grid_id,
                timestamp_hour=timestamp_hour,
                reason=f"Model prediction failed: {e}"
            )
    
    def predict_batch(self, features_dict: dict, timestamp_hour=None) -> dict:
        """Run inference on multiple feature DataFrames at once (batch mode).
        
        Optimized for throughput by running LightGBM inference on multiple
        rows in a single call, which is more efficient than sequential predictions.
        
        Args:
            features_dict: Dict mapping grid_id → single-row DataFrame with 64 features
                (as returned by FeatureAssembler.fetch_features_batch())
            timestamp_hour: Prediction timestamp (for error context, optional)
        
        Returns:
            Dict mapping grid_id → PM2.5 prediction value (float)
            Only includes grids for which prediction succeeded.
            Grids with invalid features are logged as warnings and excluded.
        
        Performance:
            - LightGBM.predict() on 100 rows: ~50ms vs 500ms sequential
            - Scales to 1000s of grids efficiently
        
        Examples:
            >>> predictions = engine.predict_batch(
            ...     {"g00000001": df1, "g00000002": df2, "g00000003": df3},
            ...     datetime(2026, 1, 19, 16, 47)
            ... )
            >>> print(predictions)
            {'g00000001': 45.2, 'g00000002': 48.1, 'g00000003': 42.5}
        """
        # Lazy load model if not already loaded
        if self.model is None:
            self.load_model()
        
        if not features_dict:
            logger.warning("predict_batch called with empty features_dict")
            return {}
        
        logger.debug(f"Batch predict: {len(features_dict)} grids")
        
        result = {}
        
        # Concatenate all DataFrames into single batch
        grid_ids = list(features_dict.keys())
        features_batch = pd.concat([features_dict[gid] for gid in grid_ids], ignore_index=True)
        
        # Validate input shape
        if features_batch.shape[0] != len(grid_ids):
            raise ModelInferenceError(
                grid_id="batch",
                timestamp_hour=timestamp_hour,
                reason=f"Expected {len(grid_ids)} rows, got {features_batch.shape[0]}"
            )
        
        if features_batch.shape[1] != N_FEATURES:
            raise ModelInferenceError(
                grid_id="batch",
                timestamp_hour=timestamp_hour,
                reason=f"Expected {N_FEATURES} features, got {features_batch.shape[1]}"
            )
        
        # Check for NaN/inf in input
        null_mask = features_batch.isnull().any(axis=1)
        if null_mask.any():
            bad_grids = [grid_ids[i] for i in range(len(grid_ids)) if null_mask.iloc[i]]
            logger.warning(f"Skipping {len(bad_grids)} grids with NaN features: {bad_grids}")
            # Remove rows with NaN
            features_batch = features_batch[~null_mask]
            grid_ids = [grid_ids[i] for i in range(len(grid_ids)) if not null_mask.iloc[i]]
        
        if np.isinf(features_batch.values).any():
            logger.warning("Input contains infinite values, some predictions may fail")
        
        if not grid_ids:
            logger.warning("All grids had NaN features, returning empty dict")
            return {}
        
        # Run batch prediction
        try:
            predictions = self.model.predict(features_batch, num_iteration=self.model.best_iteration)
            
            # LightGBM returns array of predictions
            if len(predictions) != len(grid_ids):
                raise ModelInferenceError(
                    grid_id="batch",
                    timestamp_hour=timestamp_hour,
                    reason=f"Expected {len(grid_ids)} predictions, got {len(predictions)}"
                )
            
            # Validate each prediction and build result dict
            for grid_id, pm25_value in zip(grid_ids, predictions):
                pm25_value = float(pm25_value)
                
                # Validate individual prediction
                if np.isnan(pm25_value):
                    logger.warning(f"Skipping {grid_id}: Model returned NaN")
                    continue
                
                if np.isinf(pm25_value):
                    logger.warning(f"Skipping {grid_id}: Model returned infinite value")
                    continue
                
                if pm25_value < 0:
                    logger.warning(f"Skipping {grid_id}: Model returned negative prediction {pm25_value}")
                    continue
                
                result[grid_id] = pm25_value
            
            logger.debug(f"Batch predict completed: {len(result)}/{len(grid_ids)} predictions valid")
            
            return result
        
        except ModelInferenceError:
            # Re-raise our own errors
            raise
        
        except Exception as e:
            raise ModelInferenceError(
                grid_id="batch",
                timestamp_hour=timestamp_hour,
                reason=f"Batch model prediction failed: {e}"
            )

