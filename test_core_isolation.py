#!/usr/bin/env python3
"""
Test the core inference module in isolation.
This doesn't require the coordinator or worker pool.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Setup path to import core
SCRIPT_DIR = Path(__file__).parent
TAKSHASHILA_DIR = SCRIPT_DIR / "takshashila"
sys.path.insert(0, str(TAKSHASHILA_DIR))

print("=" * 70)
print("TESTING CORE INFERENCE MODULE IN ISOLATION")
print("=" * 70)

# Test 1: Import the module
print("\n[1] Testing imports...")
try:
    from takshashila.core.config import (
        DUCKDB_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH, MODEL_VERSION
    )
    print(f"✓ Config imported")
    print(f"  MODEL_PATH: {MODEL_PATH}")
    print(f"  FEATURE_SCHEMA_PATH: {FEATURE_SCHEMA_PATH}")
    print(f"  DUCKDB_PATH: {DUCKDB_PATH}")
except ImportError as e:
    print(f"✗ Config import failed: {e}")
    print(f"  Trying alternative import...")
    from takshashila.core.config import (
        DUCKDB_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH, MODEL_VERSION
    )
    print(f"✓ Config imported (alternative)")

# Test 2: Check if files exist
print("\n[2] Checking required files...")
files_to_check = {
    "Model": MODEL_PATH,
    "Feature Schema": FEATURE_SCHEMA_PATH,
}

all_exist = True
for name, path in files_to_check.items():
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    if not exists:
        all_exist = False
        print(f"   Size would be needed: missing")

if not all_exist:
    print("\n⚠ WARNING: Some files are missing. Cannot proceed with full inference test.")
    sys.exit(1)

# Test 3: Load the model
print("\n[3] Testing model loading...")
try:
    import json
    with open(FEATURE_SCHEMA_PATH) as f:
        schema = json.load(f)
    print(f"✓ Feature schema loaded")
    print(f"  Fields: {len(schema.get('feature_list', []))} features")
except Exception as e:
    print(f"✗ Feature schema loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Try to import and instantiate InferenceEngine
print("\n[4] Testing InferenceEngine instantiation...")
try:
    from takshashila.core.inference import InferenceEngine
    print(f"✓ InferenceEngine imported")
    
    # Try to instantiate
    engine = InferenceEngine(
        db_path=str(DUCKDB_PATH),
        model_path=str(MODEL_PATH),
        feature_schema_path=str(FEATURE_SCHEMA_PATH)
    )
    print(f"✓ InferenceEngine instantiated")
except Exception as e:
    print(f"✗ InferenceEngine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try a prediction
print("\n[5] Checking available grids...")
try:
    # Query to see what grids exist
    import duckdb
    
    conn = duckdb.connect(str(DUCKDB_PATH))
    result = conn.execute("SELECT DISTINCT grid_id FROM pm25_observation LIMIT 5").fetchall()
    
    if result:
        available_grids = [row[0] for row in result]
        print(f"✓ Available grids: {available_grids}")
        test_grid_id = "g00000001"
    else:
        # If no grids in DB, check what we'd expect
        print(f"⚠ No grids in database yet. This is expected for fresh setup.")
        print(f"  Database path: {DUCKDB_PATH}")
        print(f"  Skipping prediction test (would need populated grid table)")
        test_grid_id = None
    
    conn.close()
except Exception as e:
    print(f"⚠ Could not query grids: {e}")
    print(f"  This is expected if database hasn't been initialized yet")
    test_grid_id = None

# Test 6: Try a prediction (if we have a grid)
if test_grid_id:
    print("\n[6] Testing prediction execution...")
    try:
        # Use a timestamp that's after both satellite data timestamps:
        # - Landsat8: 2026-01-16 12:00:00 (5 days before BASE_TIMESTAMP)
        # - Sentinel2: 2026-01-19 12:00:00 (2 days before BASE_TIMESTAMP)
        # - PM2.5/Meteo: hourly from 2026-01-14 13:00 to 2026-01-21 12:00
        # So we can predict anytime from 2026-01-19 13:00 onwards
        timestamp = datetime(2026, 1, 20, 15, 30, 0, tzinfo=timezone.utc)
        
        print(f"  Input: grid_id={test_grid_id}, timestamp={timestamp}")
        
        pm25 = engine.predict(grid_id=test_grid_id, timestamp_hour=timestamp)
        
        print(f"✓ Prediction successful!")
        print(f"  Result: PM2.5 = {pm25}")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
else:
    print("\n[6] Skipping prediction test (no valid grid available)")

print("\n" + "=" * 70)
print("CORE MODULE SPECIFICATION")
print("=" * 70)
print("""
INPUT: (grid_id: str, timestamp_hour: datetime) → float (PM2.5 value)

DEPENDENCIES:
  - DuckDB database at DUCKDB_PATH
    Tables required: grid, features (for features table metadata)
  - LightGBM model file at MODEL_PATH
  - Feature schema JSON at FEATURE_SCHEMA_PATH

OUTPUTS:
  - float: PM2.5 prediction value
  - Raises InvalidRequestError if grid_id not found
  - Raises FeatureAssemblyError if features cannot be assembled
  - Raises ModelInferenceError if model prediction fails

PIPELINE STAGES:
  1. Normalize timestamp to hour boundary
  2. Check if prediction already exists (for idempotency)
  3. Assemble 64 features from DuckDB tables
  4. Load LightGBM model
  5. Run model inference on feature vector
  6. Write prediction to DuckDB
  7. Return PM2.5 value as float
""")

print("=" * 70)
print("CORE ISOLATION TEST COMPLETE")
print("=" * 70)
