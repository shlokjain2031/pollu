# Inference Module Specification

**Module**: `dwarka/inference.py`  
**Purpose**: Production inference engine for PM2.5 predictions at 30m resolution  
**Dependencies**: `lightgbm`, `pandas`, `numpy`, `nalanda.features`, `nalanda.training`

---

## 1. Overview

The inference module provides a complete pipeline from trained LightGBM model to production predictions. It handles:

1. **Model Loading**: Load frozen model + feature schema with version validation
2. **Feature Assembly**: Real-time feature computation for any date/time/location
3. **Batch Prediction**: Efficient prediction across all grid cells (59k+ cells)
4. **Output Formatting**: API-ready responses with metadata and confidence
5. **Caching**: Feature computation caching to avoid redundant work

### Design Principles

- **Reproducibility**: Use same feature computation as training (nalanda modules)
- **Performance**: <5 seconds for full city prediction (59k cells)
- **Safety**: Validate feature schema matches frozen schema from training
- **Observability**: Log feature computation times, cache hits, prediction stats
- **Graceful Degradation**: Return predictions even with missing satellite data

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MODEL LOADING                                               │
│     ├─ Load LightGBM model (.txt)                               │
│     ├─ Load frozen feature schema (.json)                       │
│     └─ Validate compatibility                                   │
│                                                                 │
│  2. FEATURE ASSEMBLY (for target datetime)                      │
│     ├─ Target computation (PM2.5 lags from historical data)     │
│     ├─ Meteorological features (OpenMeteo + lags)               │
│     ├─ Satellite features (Landsat + Sentinel indices)          │
│     ├─ Temporal encoding (hour_sin, hour_cos, dow_sin, dow_cos) │
│     └─ Staleness features (days_since_landsat, days_since_sent) │
│                                                                 │
│  3. FEATURE ALIGNMENT                                           │
│     ├─ Reorder columns to match frozen schema                   │
│     ├─ Handle missing features (fill with NaN or default)       │
│     └─ Validate dtypes and ranges                               │
│                                                                 │
│  4. PREDICTION                                                  │
│     ├─ Batch predict for all grid cells                         │
│     ├─ Apply bounds (0 ≤ PM2.5 ≤ 500 µg/m³)                     │
│     └─ Compute prediction metadata (feature availability)       │
│                                                                 │
│  5. OUTPUT FORMATTING                                           │
│     ├─ Add spatial metadata (lat/lon, grid_id)                  │
│     ├─ Add temporal metadata (prediction_time, source_dates)    │
│     ├─ Add confidence indicators (feature completeness)         │
│     └─ Format for API (GeoJSON, CSV, Parquet)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Core API Specifications

### 3.1 Model Loading

```python
class InferenceModel:
    """Production inference model with frozen feature schema.
    
    Encapsulates:
    - Trained LightGBM model
    - Frozen feature schema (64 features)
    - Model metadata (version, training date, metrics)
    
    Thread-safe for concurrent predictions.
    """
    
    def __init__(
        self,
        model_path: str | Path,
        schema_path: str | Path = None,
        metadata_path: str | Path = None,
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
        
    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names from frozen schema."""
        
    @property
    def version(self) -> str:
        """Model version identifier."""
        
    @property
    def training_date(self) -> str:
        """Date model was trained."""
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict PM2.5 for feature DataFrame.
        
        Args:
            X: DataFrame with features matching frozen schema
            
        Returns:
            Array of PM2.5 predictions (µg/m³)
            
        Raises:
            ValueError: If features don't match schema
        """
```

**Example usage:**
```python
model = InferenceModel(
    model_path='artifacts/models/v1/model.txt',
    schema_path='artifacts/models/v1/schema.json'
)

print(f"Model version: {model.version}")
print(f"Features: {len(model.feature_names)}")
# Model version: v1.0.0
# Features: 64
```

---

### 3.2 Feature Assembly for Inference

```python
def assemble_inference_features(
    target_datetime: str | datetime,
    city: str = 'mumbai',
    grid_ids: Optional[Set[str]] = None,
    historical_pm25_path: str = None,
    meteo_dir: str = None,
    landsat_dir: str = None,
    sentinel_dir: str = None,
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
        FileNotFoundError: If required data sources missing
        
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
        (59832, 66)  # 59832 grid cells, 66 columns (grid_id + timestamp + 64 features)
    """
```

**Key implementation details:**
- **PM2.5 Lags**: Load historical predictions/observations to compute lags (1h, 2h, 3h, 6h, 12h, 24h)
- **Meteorological**: Query OpenMeteo for target datetime + historical for lags
- **Satellite**: Use most recent Landsat/Sentinel before target datetime
- **Temporal**: Compute from target_datetime directly (no data needed)
- **Staleness**: Compute days since satellite acquisition

---

### 3.3 Batch Prediction

```python
def predict_batch(
    model: InferenceModel,
    target_datetime: str | datetime,
    city: str = 'mumbai',
    grid_ids: Optional[Set[str]] = None,
    return_features: bool = False,
    return_metadata: bool = True,
    cache_dir: Optional[str] = None,
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
        
              grid_id  pm25_pred  feature_completeness
        0  89284a53...      85.3                 0.984
        1  89284a57...      92.1                 0.969
        2  89284a5b...      78.5                 1.000
    """
```

---

### 3.4 Point Prediction (Single Location)

```python
def predict_point(
    model: InferenceModel,
    lat: float,
    lon: float,
    target_datetime: str | datetime,
    city: str = 'mumbai',
    return_metadata: bool = True,
) -> Dict[str, Any]:
    """Predict PM2.5 for a single lat/lon point.
    
    Args:
        model: Loaded InferenceModel
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        target_datetime: Prediction target time
        city: City name
        return_metadata: Include prediction metadata
        
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
        PM2.5: 85.3 µg/m³
        AQI: Moderate
    """
```

---

### 3.5 Output Formatting

```python
def format_predictions_geojson(
    predictions: pd.DataFrame,
    grid_path: str = 'resources/mumbai/mumbai_grid.parquet',
    include_geometry: bool = True,
) -> Dict[str, Any]:
    """Format predictions as GeoJSON for mapping.
    
    Args:
        predictions: Output from predict_batch()
        grid_path: Path to grid with geometries
        include_geometry: Include polygon geometries (vs just centroids)
        
    Returns:
        GeoJSON FeatureCollection with:
            - type: 'FeatureCollection'
            - features: List[Feature]
                - geometry: Point or Polygon
                - properties: {pm25_pred, timestamp, grid_id, ...}
    """

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

def format_predictions_parquet(
    predictions: pd.DataFrame,
    output_path: str,
    partition_by: Optional[str] = 'timestamp',
) -> None:
    """Save predictions to Parquet with optional partitioning.
    
    Args:
        predictions: Output from predict_batch()
        output_path: Parquet output path
        partition_by: Column to partition by (e.g., 'timestamp', None)
    """
```

---

## 4. Feature Computation Details

### 4.1 PM2.5 Lag Features

**Source**: Historical predictions or observations

For inference at time T:
- `pm25_t_minus_1h`: PM2.5 at T-1h
- `pm25_t_minus_2h`: PM2.5 at T-2h
- ... up to `pm25_t_minus_24h`

**Implementation options:**

**Option A: Use previous predictions** (cold start)
```python
def get_pm25_lags(target_datetime, prediction_history_db):
    """Load lags from previous predictions."""
    # Query prediction database for T-1h, T-2h, ..., T-24h
    # If missing, use observation-based interpolation
```

**Option B: Use observations** (warm start with sensors)
```python
def get_pm25_lags(target_datetime, sensor_data):
    """Interpolate lags from sensor observations."""
    # Use kriging/IDW to interpolate sensor readings to grid
    # Same as training target computation
```

**Recommendation**: Start with Option B (observations), transition to Option A after accumulating predictions.

---

### 4.2 Meteorological Features

**Source**: OpenMeteo API (real-time) or cached historical data

```python
from nalanda.features_meteo import compute_meteo_features

# Same function as training
meteo_df = compute_meteo_features(
    date=target_datetime.date().isoformat(),
    city='mumbai',
    grid_ids=grid_ids
)
```

**Optimization**: Cache API responses for recent dates to avoid redundant queries.

---

### 4.3 Satellite Features

**Source**: Most recent Landsat/Sentinel before target datetime

```python
from nalanda.features_satellite import (
    compute_landsat_features,
    compute_sentinel_features,
    find_most_recent_date
)

# Find most recent acquisition
landsat_date = find_most_recent_date(
    target_datetime.date().isoformat(),
    get_available_dates(landsat_dir, city)
)

# Compute indices (same as training)
landsat_df = compute_landsat_features(
    landsat_date, city=city, grid_ids=grid_ids
)
```

**Note**: Satellite features can be cached for hours/days since they change slowly.

---

### 4.4 Temporal Encoding

**Source**: Computed directly from target_datetime

```python
from nalanda.features_temporal import compute_temporal_features

# Create dummy DataFrame with timestamp
df = pd.DataFrame({
    'timestamp': [target_datetime] * len(grid_ids),
    'grid_id': list(grid_ids)
})

# Compute temporal features
df = compute_temporal_features(df)
# Adds: hour_sin, hour_cos, dow_sin, dow_cos
```

**Performance**: Negligible (<1ms for 60k cells).

---

## 5. Performance Requirements

| Operation | Target | Acceptable | Critical |
|-----------|--------|------------|----------|
| Model loading | <1s | <3s | <5s |
| Feature assembly (full city) | <3s | <8s | <15s |
| Prediction (59k cells) | <1s | <2s | <5s |
| **Total latency** | **<5s** | **<10s** | **<20s** |
| Memory usage | <2GB | <4GB | <8GB |
| Cache size | <100MB/day | <500MB/day | <2GB/day |

**Optimization strategies:**
1. **Lazy loading**: Load model once, reuse for multiple predictions
2. **Feature caching**: Cache meteo/satellite features with TTL
3. **Batch processing**: Vectorized operations for all grid cells
4. **Parallel feature computation**: Compute meteo/satellite/temporal in parallel

---

## 6. Error Handling & Graceful Degradation

### 6.1 Missing Data Scenarios

| Scenario | Behavior | Impact on Prediction |
|----------|----------|---------------------|
| **No PM2.5 history** | Use city-wide average lags | High uncertainty, log warning |
| **No meteorology** | Fill with seasonal normals | Moderate uncertainty |
| **No Landsat data** | Use last available (up to 16 days old) | Low impact if recent |
| **No Sentinel data** | Use last available (up to 5 days old) | Low impact if recent |
| **No satellite at all** | Fill with NaN, model handles | Degraded performance (~10% worse) |

### 6.2 Validation Checks

```python
def validate_features(df: pd.DataFrame, schema: FrozenFeatureSchema) -> ValidationResult:
    """Validate features against frozen schema.
    
    Checks:
    1. All required columns present
    2. Correct dtypes
    3. Value ranges reasonable
    4. No all-null columns
    
    Returns:
        ValidationResult with warnings/errors
    """
```

---

## 7. Caching Strategy

### 7.1 Cache Structure

```
cache/inference/
├── meteo/
│   ├── 2024-07-15/
│   │   └── mumbai_meteo.parquet
│   └── 2024-07-16/
│       └── mumbai_meteo.parquet
├── landsat/
│   └── 2024-07-10/
│       └── mumbai_landsat.parquet
├── sentinel/
│   └── 2024-07-14/
│       └── mumbai_sentinel.parquet
└── predictions/
    └── 2024-07-15T14:00:00/
        └── mumbai_predictions.parquet
```

### 7.2 Cache Invalidation

- **Meteorology**: TTL = 1 hour (API updates hourly)
- **Landsat**: TTL = 16 days (revisit period)
- **Sentinel**: TTL = 5 days (revisit period)
- **Predictions**: TTL = None (permanent archive)

---

## 8. API Integration (serving/app.py)

### 8.1 Endpoints

```python
# In serving/app.py

from dwarka.inference import InferenceModel, predict_batch, predict_point

# Global model instance
MODEL = None

@app.before_first_request
def load_model():
    global MODEL
    MODEL = InferenceModel(
        model_path='artifacts/models/v1/model.txt',
        schema_path='artifacts/models/v1/schema.json'
    )

@app.route('/predict/batch', methods=['POST'])
def predict_batch_endpoint():
    """Predict for all grid cells at target datetime.
    
    Request:
        {
            "datetime": "2024-07-15T14:00:00",
            "city": "mumbai",
            "format": "geojson"  # or "csv", "parquet"
        }
        
    Response:
        GeoJSON FeatureCollection or CSV/Parquet download
    """
    data = request.json
    predictions = predict_batch(
        MODEL,
        target_datetime=data['datetime'],
        city=data.get('city', 'mumbai')
    )
    
    if data.get('format') == 'geojson':
        return jsonify(format_predictions_geojson(predictions))
    # ... handle other formats

@app.route('/predict/point', methods=['GET'])
def predict_point_endpoint():
    """Predict for single lat/lon point.
    
    Query params:
        ?lat=19.0760&lon=72.8777&datetime=2024-07-15T14:00:00
        
    Response:
        {
            "lat": 19.0760,
            "lon": 72.8777,
            "pm25_pred": 85.3,
            "timestamp": "2024-07-15T14:00:00",
            "aqi_category": "Moderate"
        }
    """
    lat = float(request.args['lat'])
    lon = float(request.args['lon'])
    datetime = request.args['datetime']
    
    result = predict_point(MODEL, lat, lon, datetime)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_version': MODEL.version if MODEL else None
    })
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
def test_model_loading():
    """Test model loads correctly."""
    model = InferenceModel('path/to/model.txt')
    assert len(model.feature_names) == 64
    assert model.version is not None

def test_feature_assembly():
    """Test features match training format."""
    features = assemble_inference_features(
        target_datetime='2024-07-15T14:00:00',
        city='mumbai'
    )
    assert len(features.columns) == 66  # grid_id + timestamp + 64 features
    assert 'pm25_t_minus_1h' in features.columns

def test_prediction_bounds():
    """Test predictions are in valid range."""
    model = InferenceModel('path/to/model.txt')
    predictions = predict_batch(model, '2024-07-15T14:00:00')
    assert (predictions['pm25_pred'] >= 0).all()
    assert (predictions['pm25_pred'] <= 500).all()
```

### 9.2 Integration Tests

```python
def test_end_to_end_inference():
    """Test complete inference pipeline."""
    model = InferenceModel('artifacts/models/v1/model.txt')
    predictions = predict_batch(
        model,
        target_datetime='2024-07-15T14:00:00',
        city='mumbai',
        return_metadata=True
    )
    
    assert len(predictions) > 0
    assert 'pm25_pred' in predictions.columns
    assert 'feature_completeness' in predictions.columns
    assert predictions['feature_completeness'].mean() > 0.9
```

---

## 10. Deployment Considerations

### 10.1 Docker Container

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY nalanda/ /app/nalanda/
COPY dwarka/ /app/dwarka/
COPY serving/ /app/serving/
COPY artifacts/models/v1/ /app/models/

# Set environment variables
ENV MODEL_PATH=/app/models/model.txt
ENV SCHEMA_PATH=/app/models/schema.json

# Run server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "serving.app:app"]
```

### 10.2 Model Versioning

```
artifacts/models/
├── v1/
│   ├── model.txt
│   ├── schema.json
│   └── metadata.json
├── v2/
│   ├── model.txt
│   ├── schema.json
│   └── metadata.json
└── latest -> v2  # Symlink to current production model
```

---

## 11. Success Criteria

The inference module is complete when:

1. ✅ Can load model + schema in <1s
2. ✅ Can predict for all 59k grid cells in <10s total
3. ✅ Features match training format exactly (validated against frozen schema)
4. ✅ Handles missing data gracefully (no crashes)
5. ✅ Returns predictions with metadata (version, completeness, timestamps)
6. ✅ Supports multiple output formats (GeoJSON, CSV, Parquet)
7. ✅ Integrates with Flask/FastAPI serving layer
8. ✅ Has comprehensive unit and integration tests
9. ✅ Documented with usage examples

---

## 12. Implementation Checklist

- [ ] `InferenceModel` class with model loading and validation
- [ ] `assemble_inference_features()` using nalanda modules
- [ ] `predict_batch()` for batch predictions
- [ ] `predict_point()` for single location
- [ ] Output formatters (GeoJSON, CSV, Parquet)
- [ ] Caching layer for features
- [ ] Integration with serving/app.py
- [ ] Unit tests for all core functions
- [ ] Integration test for end-to-end pipeline
- [ ] Performance benchmarking
- [ ] Documentation and examples
