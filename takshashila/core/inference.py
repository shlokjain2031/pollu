"""Main inference engine orchestrating the prediction pipeline.

Single entry point for computing PM2.5 predictions.
Implements contracts from SYSTEM_CONTRACTS.md.
"""
from datetime import datetime
from pathlib import Path
import json
import logging

from .persistence import PersistenceLayer
from .feature_assembler import FeatureAssembler
from .model_predictor import ModelPredictor
from .utils import round_to_hour, setup_logging
from .config import (
    MODEL_VERSION, SCHEMA_HASH, TABLE_GRID, TABLE_PM25_OBSERVATION,
    TABLE_METEO, TABLE_LANDSAT8, TABLE_SENTINEL2, TABLE_PREDICTIONS,
    DUCKDB_GRID_PATH, DUCKDB_PM25_PATH, DUCKDB_METEO_PATH,
    DUCKDB_LANDSAT8_PATH, DUCKDB_SENTINEL2_PATH, DUCKDB_PREDICTIONS_PATH
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Orchestrates the full inference pipeline.
    
    Pipeline stages:
    1. Normalize timestamp to hour boundary
    2. Check if prediction already exists (idempotency)
    3. Assemble 64 features from DuckDB
    4. Run model inference
    5. Write prediction to DuckDB
    6. Validate write
    7. Return prediction
    
    Implements:
    - Unit of truth: (grid_id, timestamp_hour) → pm25_prediction (§1)
    - Time semantics: timestamp floored to hour (§2)
    - Authority chain: Python writes to DuckDB (§3)
    - Fail loudly: All errors explicit (§5)
    - Idempotency: Same input → same output (§1.2)
    """
    
    def __init__(self, db_paths: dict = None, model_path: str = None, feature_schema_path: str = None):
        """Initialize inference engine.
        
        Args:
            db_paths: Dict mapping table names to file paths. If None, uses default paths from config.
                {
                    "grid": "/path/to/grid.duckdb",
                    "pm25_observation": "/path/to/pm25_observations.duckdb",
                    "meteo": "/path/to/meteo.duckdb",
                    "landsat8": "/path/to/landsat8.duckdb",
                    "sentinel2": "/path/to/sentinel2.duckdb",
                    "predictions": "/path/to/predictions.duckdb"
                }
            model_path: Path to LightGBM model.txt
            feature_schema_path: Path to feature_schema.json
        """
        # Use default paths if not provided
        if db_paths is None:
            db_paths = {
                TABLE_GRID: str(DUCKDB_GRID_PATH),
                TABLE_PM25_OBSERVATION: str(DUCKDB_PM25_PATH),
                TABLE_METEO: str(DUCKDB_METEO_PATH),
                TABLE_LANDSAT8: str(DUCKDB_LANDSAT8_PATH),
                TABLE_SENTINEL2: str(DUCKDB_SENTINEL2_PATH),
                TABLE_PREDICTIONS: str(DUCKDB_PREDICTIONS_PATH)
            }
        
        # Load feature schema
        with open(feature_schema_path, 'r') as f:
            feature_schema = json.load(f)
        
        # Initialize components
        self.persistence = PersistenceLayer(db_paths[TABLE_PREDICTIONS])
        self.model_predictor = ModelPredictor(model_path)
        self.feature_assembler = FeatureAssembler(db_paths, feature_schema)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("InferenceEngine initialized with separate DuckDB files")
    
    def predict(self, grid_id: str, timestamp_hour: datetime) -> float:
        """Compute PM2.5 prediction for (grid_id, timestamp_hour).
        
        This is the single public API for Phase 1.
        
        Args:
            grid_id: Grid cell identifier (e.g., "g00000001")
            timestamp_hour: Prediction timestamp (will be floored to hour)
        
        Returns:
            PM2.5 prediction value (float, in ppm)
        
        Raises:
            InvalidRequestError: If grid_id invalid or not in grids table
            FeatureAssemblyError: If required feature missing
            ModelInferenceError: If model prediction fails
            PersistenceError: If DuckDB read/write fails
            TimeoutError: If computation exceeds timeout (Phase 2+)
        
        Examples:
            >>> engine = InferenceEngine(db_path, model_path, schema_path)
            >>> pm25 = engine.predict("g00000001", datetime(2026, 1, 19, 16, 47))
            >>> print(f"PM2.5: {pm25:.2f} ppm")
        """
        # Stage 1: Normalize timestamp
        ts_rounded = round_to_hour(timestamp_hour)
        
        self.logger.info(f"Predict request: grid_id={grid_id}, timestamp={ts_rounded}")
        
        # Stage 2: Check if prediction already exists (idempotency)
        existing_prediction = self.persistence.read_prediction(grid_id, ts_rounded)
        
        if existing_prediction is not None:
            self.logger.info(
                f"Prediction exists for grid_id={grid_id}, ts={ts_rounded}: "
                f"{existing_prediction:.2f} ppm (returning cached)"
            )
            return existing_prediction
        
        # Stage 3: Assemble features
        self.logger.debug(f"Assembling features for {grid_id} @ {ts_rounded}")
        features_df = self.feature_assembler.fetch_features(grid_id, ts_rounded)
        
        # Stage 4: Run model inference
        self.logger.debug(f"Running model inference for {grid_id} @ {ts_rounded}")
        prediction = self.model_predictor.predict(features_df, grid_id, ts_rounded)
        
        # Stage 5: Write to DuckDB
        self.logger.debug(f"Writing prediction to DuckDB: {prediction:.2f} ppm")
        self.persistence.write_prediction(
            grid_id=grid_id,
            timestamp_hour=ts_rounded,
            pm25_value=prediction,
            model_version=MODEL_VERSION,
            schema_hash=SCHEMA_HASH
        )
        
        # Stage 6: Success (validation happened in write_prediction)
        self.logger.info(
            f"Prediction computed for grid_id={grid_id}, ts={ts_rounded}: "
            f"{prediction:.2f} ppm"
        )
        
        # Stage 7: Return
        return prediction
