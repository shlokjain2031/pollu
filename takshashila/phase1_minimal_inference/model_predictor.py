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
