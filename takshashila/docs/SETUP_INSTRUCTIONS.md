# Phase 1 Setup Instructions

**Date**: January 20, 2026

---

## Prerequisites

- Python 3.12
- Virtual environment activated (`.venv`)
- Working directory: `/Users/shlokjain/pollu`

---

## Step 1: Install Dependencies

```bash
# From project root
pip install -r requirements.txt

# Verify lightgbm, pytest, pytest-cov are installed
pip list | grep -E 'lightgbm|pytest|duckdb'
```

Expected output:
```
duckdb                        0.9.0
lightgbm                      4.0.0 (or higher)
pytest                        7.4.0 (or higher)
pytest-cov                    4.1.0 (or higher)
```

---

## Step 2: Initialize DuckDB

```bash
# Navigate to project root
cd /Users/shlokjain/pollu

# Run initialization script
# Method 1 (from project root):
python3 Takshashila/scripts/init_duckdb.py

# Method 2 (from Takshashila directory):
cd Takshashila
python3 scripts/init_duckdb.py
cd ..
```

This will:
- Create DuckDB file: `Takshashila/duckdb/predictions.duckdb`
- Create 6 tables (grids, features_*, predictions)
- Load 509,209 grid cells from `artifacts/v1/grids/mumbai/grid.parquet`
- Populate `features_spatial` table

**Expected output:**
```
================================================================================
Initializing DuckDB schema for Takshashila Phase 1
================================================================================
DuckDB path: /Users/shlokjain/pollu/Takshashila/duckdb/predictions.duckdb
Grid parquet: /Users/shlokjain/pollu/artifacts/v1/grids/mumbai/grid.parquet
================================================================================
[INFO] Creating tables...
[INFO] Loaded 509209 grid cells
[INFO] Populated 509209 spatial features
================================================================================
Schema initialization complete!
DuckDB file created: /Users/shlokjain/pollu/Takshashila/duckdb/predictions.duckdb
File size: XX.XX MB
================================================================================
```

---

## Step 3: Populate Mock Features

```bash
# From project root:
python3 Takshashila/scripts/populate_mock_features.py

# Or from Takshashila directory:
cd Takshashila
python3 scripts/populate_mock_features.py
cd ..
```

This will:
- Create mock features for 30 adversarial test grids
- Populate: features_pm25, features_meteo, features_satellite tables
- Create test data for error condition validation

**Expected output:**
```
================================================================================
Populating mock features for Phase 1 testing
================================================================================
[INFO] Clearing existing feature data...
[INFO] Populating 15 valid grids...
[INFO] Populating 3 grids with missing PM2.5...
[INFO] Populating 3 grids with missing meteo...
[INFO] Populating 3 grids with missing satellite...
[INFO] Populating 3 edge timestamp grids...
================================================================================
Mock feature data populated successfully!
  features_pm25: 24 rows
  features_meteo: 24 rows
  features_satellite: 24 rows
================================================================================

Next step: Run tests with pytest Takshashila/tests/
```

---

## Step 4: Run Tests

```bash
# From project root, run all tests:
pytest Takshashila/tests/ -v

# Run specific test file:
pytest Takshashila/tests/test_inference.py -v

# Run with coverage report:
pytest Takshashila/tests/ -v --cov=Takshashila/phase1_minimal_inference --cov-report=term-missing

# Run specific test:
pytest Takshashila/tests/test_inference.py::TestInferenceEngine::test_predict_valid_grid_success -v

# Run with detailed output:
pytest Takshashila/tests/ -vv -s
```

**Expected: All 45 tests pass**

---

## Step 5: Test the API

```bash
# Quick test from Python REPL:
python3 << 'EOF'
from datetime import datetime, timezone
from Takshashila.phase1_minimal_inference.inference import InferenceEngine
from Takshashila.phase1_minimal_inference.config import DUCKDB_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH

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

print(f"✓ Prediction successful!")
print(f"  Grid: {grid_id}")
print(f"  Timestamp: {timestamp}")
print(f"  PM2.5: {pm25:.2f} ppm")
EOF
```

**Expected output:**
```
✓ Prediction successful!
  Grid: g00000001
  Timestamp: 2026-01-19 16:00:00+00:00
  PM2.5: XX.XX ppm
```

---

## Complete Setup Script

If you want to do everything in one go, create a file `setup_phase1.sh`:

```bash
#!/bin/bash
set -e

PROJECT_ROOT="/Users/shlokjain/pollu"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Phase 1 Complete Setup"
echo "=========================================="

echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Initializing DuckDB..."
python3 Takshashila/scripts/init_duckdb.py

echo ""
echo "Step 3: Populating mock features..."
python3 Takshashila/scripts/populate_mock_features.py

echo ""
echo "Step 4: Running tests..."
pytest Takshashila/tests/ -v

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To test the API:"
echo "  python3 -c 'from Takshashila.phase1_minimal_inference.inference import InferenceEngine; ...'"
```

Then run:
```bash
chmod +x setup_phase1.sh
./setup_phase1.sh
```

---

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'Takshashila'

**Fix**: Make sure you're running from the project root (`/Users/shlokjain/pollu`), not from a subdirectory.

```bash
# Correct:
cd /Users/shlokjain/pollu
python3 Takshashila/scripts/init_duckdb.py

# Wrong (don't do this):
cd /Users/shlokjain/pollu/Takshashila
python3 scripts/init_duckdb.py  # Will fail
```

### Error: FileNotFoundError: artifacts/v1/grids/mumbai/grid.parquet not found

**Fix**: Verify the file exists and you're running from the correct directory.

```bash
# Check if file exists:
ls -lh /Users/shlokjain/pollu/artifacts/v1/grids/mumbai/grid.parquet

# Should be ~18 MB and exist
```

### Error: lightgbm module not found

**Fix**: Install the missing dependency.

```bash
pip install lightgbm>=4.0.0

# Verify installation:
python3 -c "import lightgbm; print(lightgbm.__version__)"
```

### Tests fail with "grid_id not found"

**Fix**: Make sure mock features were populated.

```bash
# Check DuckDB contents:
python3 << 'EOF'
import duckdb
conn = duckdb.connect('Takshashila/duckdb/predictions.duckdb')
grids_count = conn.execute("SELECT COUNT(*) FROM grids").fetchone()[0]
features_count = conn.execute("SELECT COUNT(*) FROM features_pm25").fetchone()[0]
print(f"Grids: {grids_count}")
print(f"Features: {features_count}")
conn.close()
EOF

# If features count is 0, run populate_mock_features.py again
python3 Takshashila/scripts/populate_mock_features.py
```

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| DuckDB | `Takshashila/duckdb/predictions.duckdb` | Database (created by init) |
| Model | `dwarka/models/prod_v1/model.txt` | LightGBM model |
| Grid data | `artifacts/v1/grids/mumbai/grid.parquet` | 509k grid cells |
| Feature schema | `dwarka/models/prod_v1/feature_schema.json` | 64-feature list |
| Tests | `Takshashila/tests/` | Test suite |
| Config | `Takshashila/phase1_minimal_inference/config.py` | All paths |

---

## Verification Checklist

- [ ] Dependencies installed: `pip list | grep -E 'lightgbm|pytest|duckdb'`
- [ ] DuckDB created: `ls -lh Takshashila/duckdb/predictions.duckdb`
- [ ] 509k grids loaded: `python3 -c "import duckdb; print(duckdb.connect('Takshashila/duckdb/predictions.duckdb').execute('SELECT COUNT(*) FROM grids').fetchone())"`
- [ ] 24 feature rows populated: `python3 -c "import duckdb; print(duckdb.connect('Takshashila/duckdb/predictions.duckdb').execute('SELECT COUNT(*) FROM features_pm25').fetchone())"`
- [ ] All 45 tests pass: `pytest Takshashila/tests/ -q`
- [ ] API works: Run Step 5 test

---

**Ready to start Phase 1!**

Next steps:
1. Run the setup commands above
2. Read [README_PHASE1.md](README_PHASE1.md) for API documentation
3. Explore [SYSTEM_CONTRACTS.md](SYSTEM_CONTRACTS.md) for design rules
