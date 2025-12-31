#!/bin/bash
# Example: Run Patliputra pipeline for a small date range (testing)

set -e  # Exit on error

CITY="mumbai"
START_DATE="2021-06-01"
END_DATE="2021-06-10"
EE_PROJECT="your-earth-engine-project-id"  # CHANGE THIS

echo "================================"
echo "Patliputra Pipeline Test Run"
echo "================================"
echo "City: $CITY"
echo "Date range: $START_DATE to $END_DATE"
echo ""

# Activate virtual environment if needed
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Step 1: Check if grid exists
GRID_PATH="artifacts/v1/grids/$CITY/grid.parquet"
if [ ! -f "$GRID_PATH" ]; then
    echo "❌ Grid not found at $GRID_PATH"
    echo "Run create_grid.py first:"
    echo "  python patliputra/create_grid.py --grid-id $CITY"
    exit 1
else
    echo "✓ Grid exists: $GRID_PATH"
fi

# Step 2: Run pipeline
echo ""
echo "Running pipeline..."
echo ""

python -m patliputra.pipeline run \
    --city "$CITY" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --ee-project "$EE_PROJECT" \
    --stages meteo,sentinel2 \
    --verbose \
    2>&1 | tee "logs/pipeline_test_$(date +%Y%m%d_%H%M%S).log"

# Step 3: Check status
echo ""
echo "Checking artifact status..."
python -m patliputra.pipeline status --city "$CITY"

echo ""
echo "================================"
echo "Test run complete!"
echo "================================"
