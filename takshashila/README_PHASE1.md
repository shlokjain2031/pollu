# Takshashila Phase 1: Minimal Inference Engine

**Status**: ✅ Complete  
**Date**: January 19, 2026  
**Version**: 1.0.0

---

## Overview

Phase 1 implements a **minimal, correct, testable** inference engine for PM2.5 prediction. This is a single-threaded, standalone system that computes predictions using LightGBM and persists results to DuckDB.

**What Phase 1 delivers**:
- ✅ End-to-end prediction pipeline: features → model → persistence
- ✅ DuckDB-backed storage with 509k grid cells
- ✅ 64-feature assembly from multiple tables
- ✅ LightGBM model inference
- ✅ Single-writer persistence with validation
- ✅ Fail-loud error semantics
- ✅ Idempotent predictions
- ✅ Full test coverage (unit + integration)

**What Phase 1 does NOT include** (future phases):
- ❌ Single-flight deduplication (Phase 2)
- ❌ Go integration / HTTP API (Phase 2)
- ❌ Freshness checking (Phase 3)
- ❌ Caching (Phase 5)
- ❌ Scheduler (Phase 7+)

---

## Architecture

```
┌─────────────────────────────────────────┐
│       InferenceEngine (inference.py)    │
│  predict(grid_id, timestamp_hour) → pm25│
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┬─────────────┬──────────────┐
    │             │             │              │
    ▼             ▼             ▼              ▼
┌────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│Feature │  │  Model   │  │Persist  │  │  Utils   │
│Assembler│  │Predictor│  │ Layer   │  │          │
└────────┘  └──────────┘  └─────────┘  └──────────┘
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
          ┌──────────────┐
          │   DuckDB     │
          │ (predictions.│
          │    duckdb)   │
          └──────────────┘
```

---

## Quick Start

### 1. Initialize Database

```bash
cd /Users/shlokjain/pollu

# Initialize schema + load 509k grid cells
python3 Takshashila/scripts/init_duckdb.py

# Populate mock feature data for testing
python3 Takshashila/scripts/populate_mock_features.py
```

### 2. Run Tests

```bash
# Run all tests
pytest Takshashila/tests/ -v

# Run with coverage
pytest Takshashila/tests/ --cov=Takshashila/phase1_minimal_inference --cov-report=term

# Run specific test file
pytest Takshashila/tests/test_inference.py -v
```

### 3. Use Inference Engine

```python
from datetime import datetime, timezone
from Takshashila.phase1_minimal_inference.inference import InferenceEngine
from Takshashila.phase1_minimal_inference.config import (
    DUCKDB_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH
)

# Initialize engine
engine = InferenceEngine(
    db_path=str(DUCKDB_PATH),
    model_path=str(MODEL_PATH),
    feature_schema_path=str(FEATURE_SCHEMA_PATH)
)

# Run prediction
grid_id = "g00000001"
timestamp = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)

pm25 = engine.predict(grid_id, timestamp)
print(f"PM2.5 prediction: {pm25:.2f} ppm")
```

---

## File Structure

```
Takshashila/
├── SYSTEM_CONTRACTS.md          # Phase 0 design document
├── phase1_minimal_inference/
│   ├── __init__.py
│   ├── inference.py              # Main API (InferenceEngine)
│   ├── feature_assembler.py     # Feature fetching from DuckDB
│   ├── model_predictor.py       # LightGBM inference
│   ├── persistence.py           # DuckDB read/write
│   ├── errors.py                # 5 error classes
│   ├── schema.py                # DuckDB table definitions
│   ├── config.py                # Constants & paths
│   └── utils.py                 # Timestamp rounding, logging
│
├── scripts/
│   ├── init_duckdb.py           # Initialize schema
│   └── populate_mock_features.py # Load test data
│
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   ├── test_inference.py        # Integration tests (10 tests)
│   ├── test_feature_assembler.py # Unit tests (6 tests)
│   ├── test_model_predictor.py  # Unit tests (7 tests)
│   ├── test_persistence.py      # Unit tests (5 tests)
│   ├── test_errors.py           # Unit tests (7 tests)
│   ├── test_utils.py            # Unit tests (10 tests)
│   └── fixtures/
│       └── mock_grids.json      # Test grid definitions
│
└── duckdb/
    └── predictions.duckdb        # DuckDB file (gitignored)
```

---

## API Reference

### `InferenceEngine.predict(grid_id, timestamp_hour)`

Compute PM2.5 prediction for a specific grid cell and time.

**Parameters**:
- `grid_id` (str): Grid cell identifier (e.g., `"g00000001"`)
- `timestamp_hour` (datetime): Prediction timestamp (will be floored to hour)

**Returns**:
- `float`: PM2.5 prediction in ppm

**Raises**:
- `InvalidRequestError`: If `grid_id` not in grids table
- `FeatureAssemblyError`: If required feature missing
- `ModelInferenceError`: If model prediction fails
- `PersistenceError`: If DuckDB read/write fails

**Examples**:

```python
# Basic usage
pm25 = engine.predict("g00000001", datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc))

# Timestamp rounding (16:47:30 → 16:00:00)
pm25 = engine.predict("g00000001", datetime(2026, 1, 19, 16, 47, 30, tzinfo=timezone.utc))

# Idempotency (second call returns cached result)
pm25_1 = engine.predict("g00000001", datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc))
pm25_2 = engine.predict("g00000001", datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc))
assert pm25_1 == pm25_2
```

---

## Database Schema

### Tables

1. **grids** (509,209 rows)
   - Grid reference data from Mumbai parquet
   - Columns: `row_idx`, `grid_id`, `x_m`, `y_m`, `centroid_lon`, `centroid_lat`, etc.

2. **features_pm25**
   - PM2.5 lag features (6 features per grid/timestamp)

3. **features_meteo**
   - Meteorological features (43 features per grid/timestamp)

4. **features_satellite**
   - Satellite imagery features (13 features per grid/timestamp)

5. **features_spatial**
   - Static spatial features (2 features per grid)

6. **predictions**
   - Computed PM2.5 predictions
   - Primary key: `(grid_id, timestamp_hour)`
   - Includes: `pm25_prediction`, `model_version`, `schema_hash`, `created_at`

---

## Error Handling

All errors are **explicit and structured**:

```python
from Takshashila.phase1_minimal_inference.errors import (
    InvalidRequestError,        # Bad input (invalid grid_id)
    FeatureAssemblyError,       # Missing/incomplete features
    ModelInferenceError,        # Model prediction failure
    PersistenceError,           # DuckDB read/write failure
    TimeoutError                # Computation timeout (not enforced yet)
)

try:
    pm25 = engine.predict("invalid_grid", timestamp)
except InvalidRequestError as e:
    print(f"Invalid grid: {e.reason}")
except FeatureAssemblyError as e:
    print(f"Missing feature: {e.missing_feature} for {e.grid_id}")
except ModelInferenceError as e:
    print(f"Model failed: {e.reason}")
```

---

## Testing

### Test Categories

1. **Unit Tests** (35 tests)
   - `test_errors.py`: Error class attributes
   - `test_utils.py`: Timestamp rounding logic
   - `test_persistence.py`: DuckDB read/write
   - `test_feature_assembler.py`: Feature fetching
   - `test_model_predictor.py`: Model inference

2. **Integration Tests** (10 tests)
   - `test_inference.py`: End-to-end pipeline
   - Tests idempotency, error handling, edge cases

### Test Data

Mock data includes **30 adversarial grid cells**:
- 15 valid grids (complete features)
- 3 missing PM2.5 features
- 3 missing meteorological features
- 3 missing satellite features
- 3 edge timestamp cases
- 3 invalid grid_ids

### Running Tests

```bash
# All tests
pytest Takshashila/tests/ -v

# Specific category
pytest Takshashila/tests/test_inference.py -v

# With coverage
pytest Takshashila/tests/ --cov=Takshashila/phase1_minimal_inference

# Verbose output
pytest Takshashila/tests/ -vv -s
```

---

## Contract Compliance

Phase 1 implements the following contracts from [SYSTEM_CONTRACTS.md](SYSTEM_CONTRACTS.md):

| Contract | Status | Implementation |
|----------|--------|----------------|
| §1: Unit of Truth | ✅ | `(grid_id, timestamp_hour)` is atomic and idempotent |
| §2: Time Semantics | ✅ | Timestamp floored to hour via `round_to_hour()` |
| §3: Authority Chain | ✅ | Python writes to DuckDB; DuckDB is source of truth |
| §4: Freshness State Machine | ⏸️ | Deferred to Phase 3 (all requests assumed fresh) |
| §5: Fail Loudly | ✅ | All errors explicit; no silent fallbacks |
| §6: Deduplication | ⏸️ | Deferred to Phase 2 (no single-flight yet) |
| §7: Scheduler Contract | ⏸️ | Defined but not implemented |
| §8: Partial Results | ⏸️ | Deferred to Phase 6+ (heatmap API) |
| §9: Bounded Timeouts | ⏸️ | Defined in config but not enforced yet |
| §10: Non-Goals | ✅ | Explicitly excluded features documented |

---

## Performance Notes

**Phase 1 is NOT optimized for performance.** Focus is on correctness.

Typical latencies (single prediction):
- Feature assembly: ~10-50ms
- Model inference: ~5-20ms
- DuckDB write: ~5-10ms
- **Total**: ~20-80ms per prediction

Performance optimizations come in later phases:
- Phase 2: Single-flight deduplication
- Phase 4: Batch processing
- Phase 5: Caching
- Phase 8: Profiling & optimization

---

## Known Limitations

1. **Single-threaded**: No concurrent request handling yet
2. **No caching**: Every request fetches from DuckDB
3. **No freshness**: Assumes all features are fresh
4. **Mock test data**: Only 30 grids have features
5. **No timeout enforcement**: Timeouts defined but not enforced

These are **intentional** design choices for Phase 1. They will be addressed in subsequent phases.

---

## Next Steps (Phase 2)

1. Go serving layer integration
2. HTTP/gRPC API
3. Single-flight deduplication
4. Concurrent request handling
5. Proper timeout enforcement

---

## Troubleshooting

### DuckDB file not found

```bash
# Solution: Initialize database
python3 Takshashila/scripts/init_duckdb.py
```

### No mock feature data

```bash
# Solution: Populate features
python3 Takshashila/scripts/populate_mock_features.py
```

### Tests fail with "grid_id not found"

Make sure you're using grid_ids from the valid set:
- Valid: `g00000001`, `g00000042`, `g00001337`, etc.
- Invalid: `g99999999`, `invalid_grid`

### Model file not found

Check that `dwarka/models/prod_v1/model.txt` exists in the project root.

---

## Contributors

- System design: Phase 0 (SYSTEM_CONTRACTS.md)
- Phase 1 implementation: Completed January 19, 2026

---

**End of Phase 1 README**
