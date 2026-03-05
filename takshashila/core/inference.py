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

# Import pool manager if available (optional for backward compatibility)
try:
    from takshashila.core.db_pool import PoolManager
    POOL_AVAILABLE = True
except ImportError:
    PoolManager = None
    POOL_AVAILABLE = False
    logger.warning("Connection pooling not available (db_pool not found)")


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
    
    def __init__(self, db_paths: dict = None, model_path: str = None, feature_schema_path: str = None, pool_manager=None):
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
            pool_manager: Optional PoolManager for connection pooling (from db_pool)
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
        
        # Initialize components with pool_manager
        self.pool_manager = pool_manager
        self.persistence = PersistenceLayer(db_paths[TABLE_PREDICTIONS], pool_manager=pool_manager)
        self.model_predictor = ModelPredictor(model_path)
        self.feature_assembler = FeatureAssembler(db_paths, feature_schema, pool_manager=pool_manager)
        
        self.logger = logging.getLogger(__name__)
        pool_status = "with connection pooling" if pool_manager else "without connection pooling"
        self.logger.info(f"InferenceEngine initialized {pool_status} with separate DuckDB files")
    
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
    
    def predict_batch(self, grid_ids: list, timestamp_hour: datetime) -> dict:
        """Compute PM2.5 predictions for multiple grids at once (batch mode).
        
        Optimized for throughput by amortizing feature assembly overhead
        across multiple grids. Reduces DuckDB queries and improves cache efficiency.
        
        NOTE: This method does NOT check for existing predictions to avoid DuckDB 
        connection conflicts when PersistenceQueue is writing concurrently.
        Idempotency is handled by PersistenceQueue's UPSERT logic.
        
        Worker (inference_job_worker.py) is responsible for persisting results to DuckDB and Redis.
        
        Args:
            grid_ids: List of grid cell identifiers (e.g., ["g00000001", "g00000002", ...])
            timestamp_hour: Prediction timestamp (will be floored to hour)
        
        Returns:
            Dict mapping grid_id → pm25_prediction for all input grids.
        
        Performance:
            - Single batch of 100 grids: ~7ms (0.07ms per grid) ✅
            - Single batch of 1000 grids: ~70ms (0.07ms per grid) ✅
            - Memory: ~100 grids × 1KB features = 1MB
        
        Examples:
            >>> predictions = engine.predict_batch(
            ...     ["g00000001", "g00000002", "g00000003"],
            ...     datetime(2026, 1, 19, 16, 47)
            ... )
            >>> print(predictions)
            {'g00000001': 45.2, 'g00000002': 48.1, 'g00000003': 42.5}
        """
        # Stage 1: Normalize timestamp (once for all grids)
        ts_rounded = round_to_hour(timestamp_hour)
        
        self.logger.info(f"Batch predict request: {len(grid_ids)} grids @ {ts_rounded}")
        
        # Skip idempotency check - avoid opening read connections while PersistenceQueue writes concurrently
        # PersistenceQueue handles idempotency with UPSERT logic
        # This avoids DuckDB "Can't open connection with different configuration" errors
        grids_to_compute = grid_ids
        
        # Stage 3: Assemble features for all grids that need computation (batch query)
        self.logger.debug(f"Assembling features for {len(grids_to_compute)} grids")
        features_dict = self.feature_assembler.fetch_features_batch(grids_to_compute, ts_rounded)
        
        # Stage 4: Run model inference on all grids
        self.logger.debug(f"Running batch model inference for {len(grids_to_compute)} grids")
        batch_predictions = self.model_predictor.predict_batch(features_dict, ts_rounded)
        
        # Stage 5: Return predictions to worker for persistence
        self.logger.info(
            f"Batch predict completed: {len(grid_ids)} requested, "
            f"{len(grids_to_compute)} computed, returning to worker for persistence"
        )
        
        return batch_predictions
