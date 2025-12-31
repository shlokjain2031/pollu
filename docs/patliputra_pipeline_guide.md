# Patliputra Pipeline Quick Reference

## Installation

Created a job runner at [patliputra/pipeline.py](patliputra/pipeline.py) that orchestrates all data fetching scripts.

## Should You Run Concurrently or Sequentially?

### ❌ Run SEQUENTIALLY

**Why?**
- Most stages share Earth Engine quota (concurrent = faster exhaustion)
- Open-Meteo has API rate limits (concurrent = IP ban risk)  
- Scripts already have internal parallelism (`--workers` flags)
- Concurrent saves <1% time but adds complexity and failure risk

**Exception**: AQI processing (local CSV) can run in background.

See detailed analysis: [docs/patliputra_concurrency_analysis.md](docs/patliputra_concurrency_analysis.md)

## Pipeline Stages

| Stage | Type | Rate Limited | Can Parallelize? |
|-------|------|--------------|------------------|
| Grid | One-time local | No | ✅ (but run once) |
| Meteo | Open-Meteo API | Yes (0.5-2s/req) | ❌ Sequential only |
| Landsat-8 | Earth Engine | Yes (quota) | ❌ Sequential (has internal workers) |
| Sentinel-2 | Earth Engine | Yes (quota) | ❌ Sequential (has internal workers) |
| SVF | Earth Engine | Yes (quota) | ❌ Sequential (expensive) |
| AQI | Local CSV | No | ✅ Can run anytime |

## Quick Start

### Full Pipeline (All Stages)

```bash
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --ee-project your-ee-project-id \
  --hourly \
  --verbose
```

### Specific Stages Only

```bash
# Just meteorological data
python -m patliputra.pipeline run \
  --stages meteo \
  --city mumbai \
  --start-date 2021-06-01 \
  --end-date 2021-06-30 \
  --hourly

# Meteo + Satellite
python -m patliputra.pipeline run \
  --stages meteo,sentinel2 \
  --city mumbai \
  --start-date 2021-06-01 \
  --end-date 2021-06-30 \
  --ee-project your-project
```

### Check Artifact Status

```bash
python -m patliputra.pipeline status --city mumbai
```

Output:
```json
{
  "grid": {
    "exists": true,
    "path": "artifacts/v1/grids/mumbai/grid.parquet"
  },
  "meteo": {
    "exists": true,
    "file_count": 365,
    "path": "artifacts/v1/openmeteo/mumbai"
  },
  ...
}
```

## Running Individual Scripts (Advanced)

If you need fine-grained control, run scripts directly:

### Grid Creation (One-time)
```bash
python patliputra/create_grid.py --grid-id mumbai
```

### Meteorological Data
```bash
python patliputra/openmeteo_signals.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --hourly \
  --rate-limit 1.0
```

### Landsat-8
```bash
python patliputra/landsat8_signals.py \
  --dates-file patliputra/landsat8_image_dates.txt \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --project your-ee-project-id \
  --workers 2
```

### Sentinel-2
```bash
python patliputra/sentinel2_signals.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --project your-ee-project-id \
  --workers 2
```

### Sky View Factor
```bash
python patliputra/svf_earth_engine.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --project your-ee-project-id \
  --year 2019 \
  --workers 2
```

### CPCB AQI (Ground Truth)
```bash
python patliputra/cpcb_aqi.py \
  --input-dir cpcb_aqi_data \
  --force
```

## Time Estimates (7 Years: 2014-2020)

| Stage | Sequential Time | Notes |
|-------|----------------|-------|
| Grid | 5 min | One-time setup |
| Meteo | 16-20 hours | API rate limited |
| Landsat-8 | 2-4 hours | ~200-300 dates |
| Sentinel-2 | 8-12 hours | ~1,000-1,500 dates |
| SVF | 4-8 hours | Most expensive |
| AQI | 30 min | Local processing |
| **Total** | **31-45 hours** | |

## Production Run (7 Years)

### Option 1: Single Command (Recommended)

```bash
# Run in tmux/screen for resilience
tmux new -s patliputra

# Execute full pipeline
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --ee-project your-project-id \
  --hourly \
  --verbose \
  2>&1 | tee logs/pipeline_7years_$(date +%Y%m%d_%H%M%S).log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t patliputra
```

### Option 2: Staged Batching (With Checkpoints)

```bash
# Stage 1: Grid (prerequisite)
python patliputra/create_grid.py --grid-id mumbai

# Stage 2: Meteo (longest running)
python patliputra/openmeteo_signals.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --hourly \
  --rate-limit 1.0

# Stage 3: Satellite (yearly batches for checkpoints)
for year in {2014..2020}; do
  echo "Processing year: $year"
  
  # Sentinel-2
  python patliputra/sentinel2_signals.py \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --city mumbai \
    --start-date ${year}-01-01 \
    --end-date ${year}-12-31 \
    --project your-project-id \
    --workers 2
done

# Stage 4: SVF (one-time)
python patliputra/svf_earth_engine.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --project your-project-id \
  --year 2019

# Stage 5: AQI (independent, can run anytime)
python patliputra/cpcb_aqi.py
```

## Monitoring

### Watch Progress
```bash
# File counts
watch -n 60 'ls -1 artifacts/v1/openmeteo/mumbai/*.parquet | wc -l'

# Disk usage
watch -n 60 'du -h artifacts/v1/*/mumbai/'

# Logs
tail -f logs/pipeline_*.log
```

### Earth Engine Quota
Visit: https://code.earthengine.google.com/

Check quota dashboard for EECU consumption.

## Common Issues

### Earth Engine quota exhausted
```
Error: Earth Engine quota exceeded
```

**Solution**: Wait for quota reset (monthly) or reduce `--workers` count.

### Open-Meteo rate limit
```
Error: 429 Too Many Requests
```

**Solution**: Increase `--rate-limit` parameter:
```bash
--rate-limit 2.0  # 2 seconds between requests
```

### Grid not found
```
Error: Grid artifact not found
```

**Solution**: Run grid creation first:
```bash
python patliputra/create_grid.py --grid-id mumbai
```

## Test Script

Quick test with small date range:
```bash
./scripts/run_pipeline_test.sh
```

Edit the script to customize dates and stages.

## See Also

- Full concurrency analysis: [docs/patliputra_concurrency_analysis.md](docs/patliputra_concurrency_analysis.md)
- Pipeline implementation: [patliputra/pipeline.py](patliputra/pipeline.py)
- Individual script documentation: Check each script's `--help`
