# Sentinel-2 Signal Pipeline

Refactored pipeline for fetching Sentinel-2 band values (B2, B4, B8, AOT, QA10) at 30m grid resolution.

## Overview

- **No intermediate TIFFs**: Data flows directly from Earth Engine → Parquet
- **Downsampling**: Sentinel-2 native 10m pixels aggregated to 30m grid (mean of 3×3 pixel window)
- **Batched + concurrent**: Uses `ThreadPoolExecutor` to parallelize Earth Engine requests
- **Deterministic ordering**: Output rows match canonical grid order
- **Atomic writes**: Temp file + move pattern with metadata sidecars
- **Per-city manifest**: Self-contained artifact tracking at `artifacts/v1/sentinel2/{city}/manifest/`

## Architecture

```
patliputra/copernicus_signals.py     # Producer: EE → Parquet
manifest/sentinel2_manifest.py       # Per-city manifest (SQLite)
tests/test_sentinel2_consumer.py     # Consumer validation test
```

### Artifact Layout

```
artifacts/v1/sentinel2/{city}/
├── manifest/
│   └── sentinel2_manifest.db        # Per-city manifest
├── {date}.parquet                   # Band data: grid_id, B2, B4, B8, AOT, QA10
└── {date}.parquet.meta.json         # Metadata sidecar
```

## Bands

All bands native 10m resolution, sampled at `scale=30` to aggregate to grid:

- **B2**: Blue (459-525 nm)
- **B4**: Red (650-680 nm)  
- **B8**: NIR (785-900 nm)
- **AOT**: Aerosol Optical Thickness
- **QA10**: Quality band (bitmask)

## Usage

### 1. Fetch All Sentinel-2 Dates

Enumerate and produce artifacts for all available Sentinel-2 dates in a date range:

```bash
PYTHONPATH=. python patliputra/copernicus_signals.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --cloud-cover-max 50 \
  --batch-size 5000 \
  --max-workers 8 \
  --project your-ee-project-id
```

**Parameters:**
- `--start-date`, `--end-date`: Date range to query Sentinel-2 collection
- `--cloud-cover-max`: Filter by `CLOUDY_PIXEL_PERCENTAGE < N` (default: 50%)
- `--batch-size`: Grid cells per EE request (default: 1000; increase for faster large-area fetches)
- `--max-workers`: Concurrent batch workers (default: 10; tune based on EE quota)

### 2. Test Artifacts

Validate produced artifacts using the consumer test:

```bash
PYTHONPATH=. python tests/test_sentinel2_consumer.py \
  --city mumbai \
  --artifacts-root artifacts \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet
```

**Checks:**
- Manifest entries marked 'ok'
- Parquet files exist at registered paths
- Schema: `grid_id`, `B2`, `B4`, `B8`, `AOT`, `QA10`
- Grid coverage: all canonical grid_ids present
- Deterministic row ordering
- Numeric range sanity (reflectance 0-20k, AOT 0-2)

## Implementation Details

### Downsampling Strategy (10m → 30m)

Sentinel-2 bands are native 10m resolution. We sample at `scale=30` in Earth Engine's `reduceRegions`:

```python
reduced = image.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.mean(),  # Mean of 3×3 10m pixels → 30m cell
    scale=30,
    tileScale=1,
)
```

Each 30m grid cell aggregates **9 pixels** (3×3) from the 10m native data using **mean** reduction.

### Key Refactor Changes (from old EVI pipeline)

1. **No TIFFs**: Old pipeline downloaded GeoTIFFs → raster_to_grid. New pipeline uses `reduceRegions` directly.
2. **Batching + concurrency**: `fetch_band_values_batched()` splits grid into batches and uses `ThreadPoolExecutor` for parallelism.
3. **Retry/backoff**: EE `getInfo()` wrapped with exponential backoff (1s, 2s, 4s) to handle transient failures.
4. **Deterministic ordering**: Results merged with canonical grid order to ensure stable row indices.
5. **Band sorting**: Custom sort key ensures `B2, B4, B8, AOT, QA10` column order.
6. **Atomic writes**: Write to `.parquet.tmp` → rename to avoid partial files.
7. **Per-city manifest**: SQLite manifest at `artifacts/v1/sentinel2/{city}/manifest/sentinel2_manifest.db`.
8. **Canonical paths**: Artifacts written to `artifacts/v1/sentinel2/{city}/{date}.parquet`.

### Manifest Lifecycle

```python
manifest.mark_producing(date, request_id)   # Start
try:
    # ... fetch and write ...
    manifest.mark_ok(date, artifact_path)   # Success
except Exception as e:
    manifest.mark_error(date, str(e))       # Failure
```

**States**: `producing` → `ok` | `error`

## Performance

**Mumbai grid (509,209 cells):**
- Batch size: 5000 cells
- Workers: 8 concurrent
- ~102 batches per date
- ~2-5 min per date (varies with EE quota and network)

**Output size:**
- Parquet: ~11 MB per date
- Metadata JSON: ~500 bytes per date

## Comparison to Landsat Pipeline

| Feature | Landsat | Sentinel-2 |
|---------|---------|------------|
| Native resolution | 30m | 10m |
| Sampling scale | 30m (point) | 30m (mean of 3×3) |
| Bands | B1-B11 (11 bands) | B2, B4, B8, AOT, QA10 (5 bands) |
| Collection | `LANDSAT/LC08/C02/T1_TOA` | `COPERNICUS/S2_SR_HARMONIZED` |
| Temporal interval | 16 days | 5 days |
| Manifest | `landsat_manifest.py` | `sentinel2_manifest.py` |
| Artifact path | `artifacts/v1/landsat8/{city}/` | `artifacts/v1/sentinel2/{city}/` |

## Future Enhancements

- [ ] Add `stdDev` reducer for sub-30m variability (Option 2 multi-scale)
- [ ] Cloud masking via `QA60` or `SCL` band
- [ ] Compute derived indices (NDVI, EVI) at assembly time
- [ ] Date-based partitioning for large-scale fetches
- [ ] Shared `SatelliteProducer` base class for Landsat/Sentinel-2/S5P

## Troubleshooting

**"No Sentinel-2 image found for {date}"**
- Date may not have coverage or exceeds cloud threshold
- Try increasing `--cloud-cover-max`
- Check EE collection availability for your bbox

**"EE getInfo failed after 3 attempts"**
- Transient EE quota/network issue
- Reduce `--max-workers` or `--batch-size`
- Wait and retry (manifest tracks 'producing' state)

**"Row ordering does not match canonical grid"**
- Bug in reindexing logic (should not occur after refactor)
- Report issue with date and grid artifact details

## References

- [Sentinel-2 SR Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Artifact manifest spec](../manifest/README.md)
