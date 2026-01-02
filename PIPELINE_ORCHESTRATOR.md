# Pipeline Orchestrator Guide

## Overview

`run_pipeline.py` is a master orchestration script that runs the **patliputra** and **nalanda** data pipelines sequentially for a given date range. It handles workflow management, logging, error handling, and provides a unified interface for both pipelines.

## Features

- **Sequential Execution**: Runs patliputra → nalanda (nalanda only runs if patliputra succeeds)
- **Error Handling**: Stops gracefully on pipeline failure; detailed error reporting
- **Flexible Skipping**: Can skip either pipeline independently with `--skip-patliputra` or `--skip-nalanda`
- **Progress Logging**: Real-time output from both pipelines
- **Summary Reports**: Optional JSON summary of execution results
- **Timeout Safety**: No hard timeouts; pipelines run to completion or explicit failure
- **Target Row IDX Support**: Optionally filter to specific grid cells (from `compute_target_row_idxs.py`)

## Pipeline Flow

```
Input (date range, city) 
    ↓
[1] PATLIPUTRA PIPELINE
    - Fetches satellite data (Landsat-8, Sentinel-2)
    - Fetches meteorological data (OpenMeteo)
    - Produces artifacts in artifacts/v1/{landsat8,sentinel2,openmeteo,aqi}
    ↓
[2] NALANDA PIPELINE (runs if [1] succeeds)
    - Assembles training data from patliputra outputs
    - Joins satellite + meteo + target data
    - Produces training.parquet in artifacts/v1/training/
    ↓
Output: Combined artifacts + training dataset ready for model training
```

## Usage

### Basic Usage: Full Pipeline (2024 data)

```bash
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --city mumbai
```

### With Target Row IDX Filtering

If you want to limit pipeline output to specific grid cells (e.g., after running `compute_target_row_idxs.py`):

```bash
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --city mumbai \
  --target-row-idx /tmp/target_row_idx.parquet
```

### Run Only Patliputra (Skip Nalanda)

Useful for just fetching satellite/meteo data:

```bash
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --city mumbai \
  --skip-nalanda
```

### Run Only Nalanda (Skip Patliputra)

Useful if patliputra data already exists and you just need to assemble training data:

```bash
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --skip-patliputra
```

### Verbose Mode

For detailed output from both pipelines:

```bash
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --verbose
```

### Save Execution Summary

Save a JSON report of the pipeline execution:

```bash
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-summary artifacts/pipeline_report_2024.json
```

## Arguments Reference

| Argument | Required | Type | Description |
|----------|----------|------|-------------|
| `--start-date` | Yes | string | Start date (YYYY-MM-DD) |
| `--end-date` | Yes | string | End date (YYYY-MM-DD) |
| `--city` | No | string | City name for patliputra (default: `mumbai`) |
| `--target-row-idx` | No | path | Path to target row_idx parquet file |
| `--skip-patliputra` | No | flag | Skip patliputra pipeline |
| `--skip-nalanda` | No | flag | Skip nalanda pipeline |
| `--verbose` | No | flag | Enable verbose output |
| `--output-summary` | No | path | Save JSON summary to file |

## Output Summary (JSON)

When using `--output-summary`, you get a JSON file like:

```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "city": "mumbai",
  "total_duration_seconds": 1234.5,
  "start_time": "2024-01-15T10:00:00.123456",
  "end_time": "2024-01-15T10:20:34.567890",
  "patliputra_result": {
    "pipeline": "patliputra",
    "status": "success",
    "start_time": "2024-01-15T10:00:00.123456",
    "end_time": "2024-01-15T10:15:00.567890",
    "duration_seconds": 900.4,
    "returncode": 0,
    "stdout_lines": 256,
    "stderr_lines": 0,
    "error_message": null
  },
  "nalanda_result": {
    "pipeline": "nalanda",
    "status": "success",
    "start_time": "2024-01-15T10:15:00.567890",
    "end_time": "2024-01-15T10:20:34.567890",
    "duration_seconds": 334.0,
    "returncode": 0,
    "stdout_lines": 128,
    "stderr_lines": 0,
    "error_message": null
  },
  "overall_status": "success"
}
```

## Exit Codes

- **0**: Success (both pipelines completed successfully)
- **1**: Failure or partial failure (one or both pipelines failed)

## Common Workflows

### Scenario 1: Process Multiple Months

```bash
# January 2024
python run_pipeline.py --start-date 2024-01-01 --end-date 2024-01-31 --verbose

# February 2024
python run_pipeline.py --start-date 2024-02-01 --end-date 2024-02-29 --verbose

# ... repeat for each month
```

### Scenario 2: Reprocess with Updated Target Cells

After running `compute_target_row_idxs.py` to select sensor-relevant grid cells:

```bash
# First, run compute_target_row_idxs to get the selection
python patliputra/compute_target_row_idxs.py \
  --city mumbai \
  --target-row-idx /tmp/target_row_idx.parquet

# Then run full pipeline with target filtering
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --target-row-idx /tmp/target_row_idx.parquet \
  --output-summary artifacts/pipeline_filtered_2024.json
```

### Scenario 3: Quick Test (Single Day)

```bash
python run_pipeline.py \
  --start-date 2024-01-15 \
  --end-date 2024-01-15 \
  --verbose
```

### Scenario 4: Backfill Year, Then Train

```bash
# Step 1: Fetch all 2023 data
python run_pipeline.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --verbose

# Step 2: Check results
cat artifacts/pipeline_report_2023.json

# Step 3: Train model (after all data is assembled)
python train.py \
  --data artifacts/v1/training/training.parquet \
  --output artifacts/v1/models/lightgbm_2023
```

## Troubleshooting

### Patliputra Times Out or Fails

**Check:**
- Do you have Earth Engine credentials configured? (`gcloud auth application-default login`)
- Is the date range too large? Try smaller ranges (e.g., one month at a time)
- Check disk space in `artifacts/` (requires ~2-4× your date range's data size)

**Fix:**
```bash
# Retry with verbose output to see where it fails
python run_pipeline.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --verbose \
  --skip-nalanda  # Skip nalanda if you just want to debug patliputra
```

### Nalanda Fails (Patliputra Succeeded)

**Check:**
- Does patliputra output exist? Check `artifacts/v1/landsat8/mumbai/`, `artifacts/v1/sentinel2/mumbai/`, etc.
- Are there any null/invalid values in the satellite or meteo data?
- Is the grid artifact loaded correctly?

**Fix:**
```bash
# Re-run nalanda separately to debug
python -m nalanda.cli assemble \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --verbose
```

### Permission Denied (Artifact Writes)

**Fix:**
```bash
# Ensure you have write access to artifacts directory
chmod -R u+w artifacts/
```

## Performance Notes

- **Patliputra**: ~5–30 minutes per week (depends on Earth Engine API quotas, data volume, satellite coverage)
- **Nalanda**: ~1–5 minutes per week (local join/assembly, depends on data size and disk I/O)
- **Total**: Plan for 10–60 minutes per month on first run; faster on retry (some data cached)

## Advanced: Chaining Multiple Runs

Run multiple date ranges in a shell loop:

```bash
#!/bin/bash
for month in {01..12}; do
  echo "Processing 2024-$month..."
  python run_pipeline.py \
    --start-date 2024-${month}-01 \
    --end-date 2024-${month}-31 \
    --output-summary artifacts/pipeline_2024_${month}.json
  if [ $? -ne 0 ]; then
    echo "Failed on 2024-$month, stopping."
    exit 1
  fi
done
echo "All months processed successfully!"
```

## Next Steps

After a successful run:
1. Check artifacts in `artifacts/v1/training/` for assembled data
2. Run your model training pipeline
3. Evaluate on test set (2024–2025 if you prepared splits)
