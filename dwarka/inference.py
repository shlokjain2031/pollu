"""Production inference engine for PM2.5 predictions.

This module provides the complete pipeline from trained LightGBM model to
production predictions at 30m resolution. It handles model loading, real-time
feature assembly, batch prediction, and output formatting.

Key components:
- InferenceModel: Model loading with frozen feature schema validation
- assemble_inference_features: Real-time feature computation
- predict_batch: Efficient batch prediction for all grid cells
- predict_point: Single location prediction
- Output formatters: GeoJSON, CSV, Parquet

Performance targets:
- Model loading: <1s
- Feature assembly (59k cells): <8s
- Prediction (59k cells): <2s
- Total latency: <10s

Author: Dwarka Inference Team
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Import nalanda feature computation modules
from nalanda.features_meteo import compute_meteo_features
from nalanda.features_satellite import (
    compute_landsat_features,
    compute_sentinel_features,
    compute_satellite_staleness,
)
from nalanda.features_temporal import compute_temporal_features

logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading
# =============================================================================


class InferenceModel:
    """Production inference model with frozen feature schema.
    
    Encapsulates:
    - Trained LightGBM model
    - Frozen feature schema (64 features)
    - Model metadata (version, training date, metrics)
    
    Thread-safe for concurrent predictions.
    
    Example:
        >>> model = InferenceModel('artifacts/models/v1/model.txt')
        >>> print(f"Model version: {model.version}")
        >>> print(f"Features: {len(model.feature_names)}")
        Model version: v1.0.0
        Features: 64
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        schema_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        """Load model and validate feature schema.
        
        Args:
            model_path: Path to LightGBM model file (.txt)
            schema_path: Path to frozen feature schema (.json)
                        If None, looks for schema.json in same dir
            metadata_path: Path to model metadata (.json)
                          If None, looks for metadata.json in same dir
        
        Raises:
            FileNotFoundError: If model or schema not found
            ValueError: If schema validation fails
        """
        self.model_path = Path(model_path)
        
        # Auto-detect schema and metadata paths
        if schema_path is None:
            schema_path = self.model_path.parent / "schema.json"
        if metadata_path is None:
            metadata_path = self.model_path.parent / "metadata.json"
        
        self.schema_path = Path(schema_path)
        self.metadata_path = Path(metadata_path)
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading LightGBM model from {self.model_path}")
        self.model = lgb.Booster(model_file=str(self.model_path))
        
        # Load frozen feature schema
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            schema_data = json.load(f)
        
        self._feature_names = schema_data.get('features', [])
        self._feature_groups = schema_data.get('feature_groups', {})
        
        if not self._feature_names:
            raise ValueError(f"No features found in schema: {self.schema_path}")
        
        # Load metadata (optional)
        self._metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
        
        # Validate model feature names match schema
        model_features = self.model.feature_name()
        if set(model_features) != set(self._feature_names):
            missing_in_model = set(self._feature_names) - set(model_features)
            extra_in_model = set(model_features) - set(self._feature_names)
            msg = f"Feature mismatch between model and schema:\n"
            if missing_in_model:
                msg += f"  Missing in model: {missing_in_model}\n"
            if extra_in_model:
                msg += f"  Extra in model: {extra_in_model}"
            raise ValueError(msg)
        
        logger.info(
            f"Model loaded successfully: {len(self._feature_names)} features, "
            f"version {self.version}"
        )
    
    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names from frozen schema."""
        return self._feature_names.copy()
    
    @property
    def feature_groups(self) -> Dict[str, List[str]]:
        """Feature groups (pm25_lags, meteo, satellite, temporal)."""
        return self._feature_groups.copy()
    
    @property
    def version(self) -> str:
        """Model version identifier."""
        return self._metadata.get('version', 'unknown')
    
    @property
    def training_date(self) -> str:
        """Date model was trained."""
        return self._metadata.get('training_date', 'unknown')
    
    @property
    def metrics(self) -> Dict[str, float]:
        """Training/validation metrics."""
        return self._metadata.get('metrics', {})
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict PM2.5 for feature DataFrame.
        
        Args:
            X: DataFrame with features matching frozen schema
            
        Returns:
            Array of PM2.5 predictions (µg/m³)
            
        Raises:
            ValueError: If features don't match schema
        """
        # Validate features
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing features in input DataFrame: {missing}\n"
                f"Expected {len(self._feature_names)} features, "
                f"got {len(X.columns)}"
            )
        
        # Reorder columns to match frozen schema
        X_ordered = X[self._feature_names].copy()
        
        # Predict
        predictions = self.model.predict(X_ordered)
        
        # Apply bounds (PM2.5 cannot be negative or > 500 µg/m³)
        predictions = np.clip(predictions, 0, 500)
        
        return predictions


# =============================================================================
# Feature Assembly for Inference
# =============================================================================


def assemble_inference_features(
    target_datetime: Union[str, datetime],
    city: str = 'mumbai',
    grid_ids: Optional[Set[str]] = None,
    historical_pm25_path: Optional[str] = None,
    meteo_dir: Optional[str] = None,
    landsat_dir: Optional[str] = None,
    sentinel_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Assemble features for inference at a specific datetime.
    
    This is the inference equivalent of nalanda.training.assemble_training_data().
    Uses the same feature computation functions but optimized for real-time use.
    
    Args:
        target_datetime: Target prediction time (ISO format or datetime)
        city: City name (default: 'mumbai')
        grid_ids: Optional subset of grid cells (None = all cells)
        historical_pm25_path: Path to historical PM2.5 data for lags
        meteo_dir: Override meteorological data directory
        landsat_dir: Override Landsat data directory
        sentinel_dir: Override Sentinel data directory
        cache_dir: Cache directory for computed features
        
    Returns:
        DataFrame with columns:
            - grid_id: str
            - timestamp: datetime (= target_datetime)
            - [64 feature columns matching frozen schema]
        
    Raises:
        ValueError: If target_datetime is invalid
        
    Notes:
        - Uses same functions as training: compute_meteo_features(),
          compute_landsat_features(), compute_sentinel_features()
        - Handles missing data gracefully (fills with NaN)
        - Logs warnings for missing data sources
        - Caches intermediate results if cache_dir provided
        
    Example:
        >>> features = assemble_inference_features(
        ...     target_datetime='2024-07-15T14:00:00',
        ...     city='mumbai',
        ...     cache_dir='cache/inference'
        ... )
        >>> print(features.shape)
        (59832, 66)  # 59832 grid cells, 66 columns
    """
    # Parse target datetime
    if isinstance(target_datetime, str):
        target_dt = pd.to_datetime(target_datetime)
    else:
        target_dt = target_datetime
    
    target_date_str = target_dt.date().isoformat()
    
    logger.info(
        f"Assembling inference features for {target_dt} in {city}"
    )
    
    # Load grid if grid_ids not specified
    if grid_ids is None:
        grid_path = Path(f"resources/{city}/{city}_grid.parquet")
        if grid_path.exists():
            grid = pd.read_parquet(grid_path)
            grid_ids = set(grid['grid_id'].unique())
            logger.info(f"Loaded {len(grid_ids)} grid cells from {grid_path}")
        else:
            raise ValueError(f"Grid file not found and grid_ids not provided: {grid_path}")
    
    # Initialize feature DataFrame
    features = pd.DataFrame({
        'grid_id': list(grid_ids),
        'timestamp': target_dt
    })
    
    logger.info(f"Computing features for {len(grid_ids)} grid cells")
    
    # 1. PM2.5 lag features
    logger.info("Computing PM2.5 lag features...")
    pm25_lags = _compute_pm25_lags(
        target_dt, 
        grid_ids, 
        historical_pm25_path,
        city
    )
    if pm25_lags is not None:
        features = features.merge(pm25_lags, on='grid_id', how='left')
        logger.info(f"Added {len(pm25_lags.columns) - 1} PM2.5 lag features")
    else:
        logger.warning("PM2.5 lags not available, will use NaN")
    
    # 2. Temporal encoding (cheap, always compute)
    logger.info("Computing temporal features...")
    features = compute_temporal_features(features)
    logger.info("Added 4 temporal features (hour_sin, hour_cos, dow_sin, dow_cos)")
    
    # 3. Meteorological features
    logger.info("Computing meteorological features...")
    try:
        meteo_df = compute_meteo_features(
            date=target_date_str,
            city=city,
            grid_ids=grid_ids
        )
        features = features.merge(
            meteo_df, 
            on=['grid_id', 'timestamp'], 
            how='left'
        )
        logger.info(f"Added {len(meteo_df.columns) - 2} meteorological features")
    except Exception as e:
        logger.warning(f"Meteorological features failed: {e}")
    
    # 4. Landsat features
    logger.info("Computing Landsat features...")
    try:
        landsat_df = _compute_satellite_features(
            target_date_str,
            city,
            grid_ids,
            source='landsat',
            data_dir=landsat_dir
        )
        if landsat_df is not None:
            features = features.merge(landsat_df, on='grid_id', how='left')
            logger.info(f"Added {len(landsat_df.columns) - 1} Landsat features")
    except Exception as e:
        logger.warning(f"Landsat features failed: {e}")
    
    # 5. Sentinel features
    logger.info("Computing Sentinel features...")
    try:
        sentinel_df = _compute_satellite_features(
            target_date_str,
            city,
            grid_ids,
            source='sentinel',
            data_dir=sentinel_dir
        )
        if sentinel_df is not None:
            features = features.merge(sentinel_df, on='grid_id', how='left')
            logger.info(f"Added {len(sentinel_df.columns) - 1} Sentinel features")
    except Exception as e:
        logger.warning(f"Sentinel features failed: {e}")
    
    # 6. Satellite staleness features
    logger.info("Computing satellite staleness features...")
    try:
        staleness_df = compute_satellite_staleness(
            features,
            target_date_col='timestamp'
        )
        features = staleness_df
        logger.info("Added 2 staleness features (days_since_landsat, days_since_sentinel)")
    except Exception as e:
        logger.warning(f"Staleness features failed: {e}")
    
    logger.info(
        f"Feature assembly complete: {features.shape[0]} rows, "
        f"{features.shape[1]} columns"
    )
    
    return features


def _compute_pm25_lags(
    target_dt: datetime,
    grid_ids: Set[str],
    historical_path: Optional[str],
    city: str
) -> Optional[pd.DataFrame]:
    """Compute PM2.5 lag features from historical data.
    
    Args:
        target_dt: Target datetime
        grid_ids: Grid cell IDs
        historical_path: Path to historical PM2.5 data (parquet/csv)
        city: City name
        
    Returns:
        DataFrame with columns: grid_id, pm25_t_minus_1h, ..., pm25_t_minus_24h
        Or None if historical data not available
    """
    # Define lag hours
    lag_hours = [1, 2, 3, 6, 12, 24]
    
    # If no historical data provided, return None
    if historical_path is None:
        logger.warning("No historical PM2.5 path provided, lags will be NaN")
        return None
    
    hist_path = Path(historical_path)
    if not hist_path.exists():
        logger.warning(f"Historical PM2.5 file not found: {hist_path}")
        return None
    
    try:
        # Load historical data
        if hist_path.suffix == '.parquet':
            hist_df = pd.read_parquet(hist_path)
        elif hist_path.suffix == '.csv':
            hist_df = pd.read_csv(hist_path, parse_dates=['timestamp'])
        else:
            logger.warning(f"Unsupported file format: {hist_path.suffix}")
            return None
        
        # Filter to relevant grid cells
        hist_df = hist_df[hist_df['grid_id'].isin(grid_ids)].copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(hist_df['timestamp']):
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        
        # Initialize result DataFrame
        result = pd.DataFrame({'grid_id': list(grid_ids)})
        
        # Compute each lag
        for lag_h in lag_hours:
            lag_time = target_dt - timedelta(hours=lag_h)
            
            # Find closest timestamp within 30 minutes
            hist_df['time_diff'] = (hist_df['timestamp'] - lag_time).abs()
            closest = hist_df[hist_df['time_diff'] <= timedelta(minutes=30)].copy()
            
            if len(closest) == 0:
                logger.warning(f"No data found for lag {lag_h}h at {lag_time}")
                result[f'pm25_t_minus_{lag_h}h'] = np.nan
                continue
            
            # Take closest value per grid cell
            closest = closest.sort_values('time_diff').groupby('grid_id').first().reset_index()
            
            # Merge into result
            lag_col = f'pm25_t_minus_{lag_h}h'
            closest = closest[['grid_id', 'pm25']].rename(columns={'pm25': lag_col})
            result = result.merge(closest, on='grid_id', how='left')
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to compute PM2.5 lags: {e}")
        return None


def _compute_satellite_features(
    target_date: str,
    city: str,
    grid_ids: Set[str],
    source: str,
    data_dir: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Compute satellite features for inference.
    
    Args:
        target_date: Target date (ISO format)
        city: City name
        grid_ids: Grid cell IDs
        source: 'landsat' or 'sentinel'
        data_dir: Override data directory
        
    Returns:
        DataFrame with satellite features or None if not available
    """
    try:
        # Find most recent satellite data before target date
        if data_dir:
            data_path = Path(data_dir)
        else:
            data_path = Path(f"artifacts/{source}/{city}")
        
        if not data_path.exists():
            logger.warning(f"{source} data directory not found: {data_path}")
            return None
        
        # Get available dates
        available_files = sorted(data_path.glob("*.parquet"))
        if not available_files:
            logger.warning(f"No {source} data files found in {data_path}")
            return None
        
        # Extract dates from filenames (assuming format: city_YYYY-MM-DD.parquet)
        available_dates = []
        for f in available_files:
            try:
                # Extract date from filename
                date_str = f.stem.split('_')[-1]
                available_dates.append((date_str, f))
            except:
                continue
        
        if not available_dates:
            logger.warning(f"No valid date files found for {source}")
            return None
        
        # Find most recent date before or equal to target
        target_dt = pd.to_datetime(target_date)
        valid_dates = [
            (d, f) for d, f in available_dates 
            if pd.to_datetime(d) <= target_dt
        ]
        
        if not valid_dates:
            logger.warning(f"No {source} data before {target_date}")
            return None
        
        # Use most recent
        most_recent_date, most_recent_file = max(valid_dates, key=lambda x: x[0])
        days_old = (target_dt - pd.to_datetime(most_recent_date)).days
        
        logger.info(
            f"Using {source} data from {most_recent_date} "
            f"({days_old} days before target)"
        )
        
        # Compute features using nalanda functions
        if source == 'landsat':
            features_df = compute_landsat_features(
                date=most_recent_date,
                city=city,
                grid_ids=grid_ids
            )
        elif source == 'sentinel':
            features_df = compute_sentinel_features(
                date=most_recent_date,
                city=city,
                grid_ids=grid_ids
            )
        else:
            raise ValueError(f"Unknown source: {source}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Failed to compute {source} features: {e}")
        return None


# =============================================================================
# Batch Prediction
# =============================================================================


def predict_batch(
    model: InferenceModel,
    target_datetime: Union[str, datetime],
    city: str = 'mumbai',
    grid_ids: Optional[Set[str]] = None,
    return_features: bool = False,
    return_metadata: bool = True,
    cache_dir: Optional[str] = None,
    historical_pm25_path: Optional[str] = None,
) -> pd.DataFrame:
    """Predict PM2.5 for all grid cells at target datetime.
    
    Complete end-to-end inference pipeline:
    1. Assemble features
    2. Align with model schema
    3. Predict
    4. Add metadata
    
    Args:
        model: Loaded InferenceModel
        target_datetime: Prediction target time
        city: City name
        grid_ids: Optional grid cell subset
        return_features: If True, include features in output
        return_metadata: If True, include prediction metadata
        cache_dir: Feature cache directory
        historical_pm25_path: Path to historical PM2.5 for lags
        
    Returns:
        DataFrame with columns:
            - grid_id: str
            - timestamp: datetime
            - pm25_pred: float (predicted PM2.5 in µg/m³)
            
            If return_metadata=True, also includes:
            - prediction_time: datetime (when prediction was made)
            - model_version: str
            - feature_completeness: float (0-1, % of features non-null)
            - has_landsat: bool
            - has_sentinel: bool
            - days_since_landsat: int
            - days_since_sentinel: int
            
            If return_features=True, also includes:
            - [all 64 feature columns]
            
    Example:
        >>> model = InferenceModel('artifacts/models/v1/model.txt')
        >>> predictions = predict_batch(
        ...     model,
        ...     target_datetime='2024-07-15T14:00:00',
        ...     return_metadata=True
        ... )
        >>> print(predictions[['grid_id', 'pm25_pred', 'feature_completeness']].head())
    """
    logger.info(f"Starting batch prediction for {city} at {target_datetime}")
    prediction_start = datetime.now()
    
    # 1. Assemble features
    features = assemble_inference_features(
        target_datetime=target_datetime,
        city=city,
        grid_ids=grid_ids,
        cache_dir=cache_dir,
        historical_pm25_path=historical_pm25_path,
    )
    
    # 2. Align features with model schema
    feature_cols = model.feature_names
    missing_features = set(feature_cols) - set(features.columns)
    
    if missing_features:
        logger.warning(
            f"Missing {len(missing_features)} features, filling with NaN: "
            f"{missing_features}"
        )
        for feat in missing_features:
            features[feat] = np.nan
    
    # 3. Predict
    X = features[feature_cols]
    predictions = model.predict(X)
    
    # 4. Build output DataFrame
    result = pd.DataFrame({
        'grid_id': features['grid_id'],
        'timestamp': features['timestamp'],
        'pm25_pred': predictions
    })
    
    # 5. Add metadata if requested
    if return_metadata:
        # Feature completeness (% non-null)
        feature_completeness = X.notna().mean(axis=1).values
        
        # Satellite availability
        has_landsat = features['landsat_date'].notna().values if 'landsat_date' in features else np.zeros(len(result), dtype=bool)
        has_sentinel = features['sentinel_date'].notna().values if 'sentinel_date' in features else np.zeros(len(result), dtype=bool)
        
        days_since_landsat = features['days_since_landsat'].values if 'days_since_landsat' in features else np.full(len(result), np.nan)
        days_since_sentinel = features['days_since_sentinel'].values if 'days_since_sentinel' in features else np.full(len(result), np.nan)
        
        result['prediction_time'] = prediction_start
        result['model_version'] = model.version
        result['feature_completeness'] = feature_completeness
        result['has_landsat'] = has_landsat
        result['has_sentinel'] = has_sentinel
        result['days_since_landsat'] = days_since_landsat
        result['days_since_sentinel'] = days_since_sentinel
    
    # 6. Add features if requested
    if return_features:
        for col in feature_cols:
            result[col] = features[col]
    
    prediction_time = (datetime.now() - prediction_start).total_seconds()
    logger.info(
        f"Batch prediction complete: {len(result)} predictions in {prediction_time:.2f}s"
    )
    
    return result


# =============================================================================
# Point Prediction
# =============================================================================


def predict_point(
    model: InferenceModel,
    lat: float,
    lon: float,
    target_datetime: Union[str, datetime],
    city: str = 'mumbai',
    return_metadata: bool = True,
    historical_pm25_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict PM2.5 for a single lat/lon point.
    
    Args:
        model: Loaded InferenceModel
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        target_datetime: Prediction target time
        city: City name
        return_metadata: Include prediction metadata
        historical_pm25_path: Path to historical PM2.5 for lags
        
    Returns:
        Dict with:
            - lat: float
            - lon: float
            - timestamp: str (ISO format)
            - pm25_pred: float
            - grid_id: str (which grid cell contains this point)
            
            If return_metadata=True:
            - prediction_time: str (ISO)
            - model_version: str
            - feature_completeness: float
            - aqi_category: str ('Good', 'Moderate', 'Unhealthy', etc.)
            
    Example:
        >>> result = predict_point(
        ...     model,
        ...     lat=19.0760,
        ...     lon=72.8777,
        ...     target_datetime='2024-07-15T14:00:00'
        ... )
        >>> print(f"PM2.5: {result['pm25_pred']:.1f} µg/m³")
        >>> print(f"AQI: {result['aqi_category']}")
    """
    logger.info(f"Point prediction for ({lat}, {lon}) at {target_datetime}")
    
    # 1. Find which grid cell contains this point
    grid_path = Path(f"resources/{city}/{city}_grid.parquet")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    
    grid = gpd.read_parquet(grid_path)
    point = Point(lon, lat)
    
    # Find containing grid cell
    containing = grid[grid.geometry.contains(point)]
    
    if len(containing) == 0:
        raise ValueError(
            f"Point ({lat}, {lon}) not within {city} grid bounds"
        )
    
    grid_id = containing.iloc[0]['grid_id']
    logger.info(f"Point maps to grid cell: {grid_id}")
    
    # 2. Predict for this grid cell
    predictions = predict_batch(
        model,
        target_datetime=target_datetime,
        city=city,
        grid_ids={grid_id},
        return_metadata=return_metadata,
        historical_pm25_path=historical_pm25_path,
    )
    
    # 3. Extract result
    row = predictions.iloc[0]
    
    result = {
        'lat': lat,
        'lon': lon,
        'timestamp': row['timestamp'].isoformat(),
        'pm25_pred': float(row['pm25_pred']),
        'grid_id': grid_id,
    }
    
    if return_metadata:
        result.update({
            'prediction_time': row['prediction_time'].isoformat(),
            'model_version': row['model_version'],
            'feature_completeness': float(row['feature_completeness']),
            'aqi_category': _pm25_to_aqi_category(row['pm25_pred']),
        })
    
    return result


def _pm25_to_aqi_category(pm25: float) -> str:
    """Convert PM2.5 to AQI category.
    
    Based on US EPA AQI breakpoints for PM2.5.
    """
    if pm25 <= 12.0:
        return 'Good'
    elif pm25 <= 35.4:
        return 'Moderate'
    elif pm25 <= 55.4:
        return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4:
        return 'Unhealthy'
    elif pm25 <= 250.4:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


# =============================================================================
# Output Formatting
# =============================================================================


def format_predictions_geojson(
    predictions: pd.DataFrame,
    grid_path: str = None,
    city: str = 'mumbai',
    include_geometry: bool = True,
) -> Dict[str, Any]:
    """Format predictions as GeoJSON for mapping.
    
    Args:
        predictions: Output from predict_batch()
        grid_path: Path to grid with geometries (default: resources/{city}/{city}_grid.parquet)
        city: City name (used if grid_path not provided)
        include_geometry: Include polygon geometries (vs just centroids)
        
    Returns:
        GeoJSON FeatureCollection with:
            - type: 'FeatureCollection'
            - features: List[Feature]
                - geometry: Point or Polygon
                - properties: {pm25_pred, timestamp, grid_id, ...}
    """
    if grid_path is None:
        grid_path = f"resources/{city}/{city}_grid.parquet"
    
    grid = gpd.read_parquet(grid_path)
    
    # Merge predictions with grid
    gdf = grid.merge(predictions, on='grid_id', how='inner')
    
    # Convert to centroids if not including full geometry
    if not include_geometry:
        gdf['geometry'] = gdf.geometry.centroid
    
    # Convert timestamp to string
    if 'timestamp' in gdf.columns:
        gdf['timestamp'] = gdf['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    if 'prediction_time' in gdf.columns:
        gdf['prediction_time'] = gdf['prediction_time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Convert to GeoJSON
    geojson = json.loads(gdf.to_json())
    
    return geojson


def format_predictions_csv(
    predictions: pd.DataFrame,
    output_path: str,
    include_features: bool = False,
) -> None:
    """Save predictions to CSV.
    
    Args:
        predictions: Output from predict_batch()
        output_path: CSV output path
        include_features: Include feature columns
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select columns
    if include_features:
        cols = predictions.columns.tolist()
    else:
        # Only metadata and prediction
        cols = [c for c in predictions.columns if not c.startswith('pm25_t_') 
                and not c.startswith('temp_') and not c.startswith('precip_')
                and not c.startswith('wind_') and not c.startswith('pressure_')
                and not c.startswith('humidity_') and not c.startswith('cloud_')
                and not c.startswith('ndvi') and not c.startswith('ndbi')
                and not c.startswith('mndwi') and not c.startswith('bsi')
                and not c.startswith('ibi') and not c.startswith('evi')
                and not c.startswith('local_') and not c.startswith('hour_')
                and not c.startswith('dow_')]
    
    predictions[cols].to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def format_predictions_parquet(
    predictions: pd.DataFrame,
    output_path: str,
    partition_by: Optional[str] = None,
) -> None:
    """Save predictions to Parquet with optional partitioning.
    
    Args:
        predictions: Output from predict_batch()
        output_path: Parquet output path
        partition_by: Column to partition by (e.g., 'timestamp', None)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if partition_by:
        # Convert to Pandas if needed
        if hasattr(predictions, 'to_pandas'):
            predictions = predictions.to_pandas()
        
        # Pyarrow partitioning
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pandas(predictions)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=[partition_by]
        )
    else:
        predictions.to_parquet(output_path, index=False)
    
    logger.info(f"Saved predictions to {output_path}")


# =============================================================================
# Validation
# =============================================================================


def validate_features(
    df: pd.DataFrame,
    expected_features: List[str]
) -> Dict[str, Any]:
    """Validate features against expected schema.
    
    Args:
        df: DataFrame with features
        expected_features: List of expected feature names
        
    Returns:
        Dict with validation results:
            - valid: bool
            - missing_features: List[str]
            - extra_features: List[str]
            - null_features: List[str] (features with >50% nulls)
            - warnings: List[str]
    """
    result = {
        'valid': True,
        'missing_features': [],
        'extra_features': [],
        'null_features': [],
        'warnings': []
    }
    
    # Check missing/extra features
    expected_set = set(expected_features)
    actual_set = set(df.columns)
    
    result['missing_features'] = list(expected_set - actual_set)
    result['extra_features'] = list(actual_set - expected_set)
    
    if result['missing_features']:
        result['valid'] = False
        result['warnings'].append(
            f"Missing {len(result['missing_features'])} features: "
            f"{result['missing_features'][:5]}"
        )
    
    # Check null rates
    for feat in expected_features:
        if feat in df.columns:
            null_rate = df[feat].isna().mean()
            if null_rate > 0.5:
                result['null_features'].append(feat)
                result['warnings'].append(
                    f"Feature '{feat}' has {null_rate:.1%} null values"
                )
    
    return result
