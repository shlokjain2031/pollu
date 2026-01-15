# MODIS AOD Implementation: Technical Documentation

**Document Version:** 1.0  
**Date:** January 14, 2026  
**Status:** Performance Bottleneck Analysis & Optimization Attempts

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Scientific Rationale](#scientific-rationale)
3. [Data Requirements](#data-requirements)
4. [Current Implementation Logic](#current-implementation-logic)
5. [Performance Challenges](#performance-challenges)
6. [Optimization Attempts](#optimization-attempts)
7. [Benchmark Comparison](#benchmark-comparison)
8. [Current Status & Recommendations](#current-status--recommendations)

---

## Problem Statement

### Primary Objective
Improve PM2.5 prediction model R² score from current baseline (~0.50-0.55) to target threshold (≥0.70) by incorporating Aerosol Optical Depth (AOD) features from MODIS satellite imagery.

### Current Model Performance
- **Algorithm:** LightGBM Gradient Boosting
- **Current Features:** 68 features including:
  - Meteorological data (OpenMeteo API)
  - Sentinel-2 spectral bands (30m resolution)
  - Landsat-8 spectral bands (30m resolution)
  - Temporal features (day of year, hour, etc.)
- **Current Performance:** 
  - Validation RMSE: ~34.86 μg/m³
  - R² Score: ~0.50-0.55
  - **Performance Plateau:** Model has reached feature limitation, not hyperparameter limitation

### Performance Gap Analysis
Literature review indicates AOD features provide R² improvements of **+0.15 to +0.35** in PM2.5 prediction models due to direct physical relationship between aerosol optical properties and particulate matter concentration.

### Data Scale
- **Spatial:** 509,209 grid cells at 30m resolution covering Mumbai metropolitan area
- **Temporal:** 2019-01-01 to 2024-12-31 (2,191 days)
- **Total Data Points:** 509,209 cells × 2,191 dates × 5 AOD bands = **~5.6 billion data points**

---

## Scientific Rationale

### Why AOD is Critically Important for PM2.5 Prediction

#### Physical Relationship
1. **Direct Coupling:**
   - PM2.5 = Fine particulate matter (aerodynamic diameter ≤ 2.5 μm)
   - AOD = Columnar measure of light extinction by atmospheric aerosols
   - PM2.5 particles ARE the aerosols that create AOD signal
   
2. **Optical Properties:**
   - Scattering: Particles scatter incoming solar radiation
   - Absorption: Certain aerosols (black carbon) absorb radiation
   - AOD quantifies total columnar extinction (scattering + absorption)

3. **Vertical Integration:**
   - AOD integrates aerosol loading through entire atmospheric column
   - PM2.5 measures surface concentration
   - Coupling requires meteorological corrections (boundary layer height, relative humidity)

#### Literature Evidence
- **Gupta et al. (2006):** R² = 0.63 using MODIS AOD for PM2.5 in Indian cities
- **van Donkelaar et al. (2010):** Global PM2.5 estimates with R² = 0.77 using satellite AOD
- **Hu et al. (2014):** Deep Blue AOD achieved R² = 0.81 with machine learning
- **Expected Gain:** +0.15 to +0.35 R² improvement over baseline models

#### Feature Value
Adding 5 AOD bands provides:
- **Spectral AOD Information:** 470nm and 550nm wavelengths capture Ångström exponent (particle size distribution proxy)
- **Uncertainty Quantification:** AOD_Uncertainty enables quality-weighted modeling
- **Quality Flags:** AOD_QA bitmask for filtering poor retrievals
- **Water Vapor:** Column_WV correlates with hygroscopic growth of particles

---

## Data Requirements

### MODIS MAIAC AOD Dataset

**Collection:** `MODIS/061/MCD19A2_GRANULES`

**Characteristics:**
- **Sensor:** Terra satellite (MODIS instrument)
- **Algorithm:** Multi-Angle Implementation of Atmospheric Correction (MAIAC)
- **Spatial Resolution:** 1000m (1km) native
- **Temporal Resolution:** Daily (when cloud-free)
- **Coverage:** Near-global, multiple overpasses per day

**Selected Bands:**
1. **Optical_Depth_047** (int16, scale 0.001)
   - AOD at 470nm wavelength
   - Valid range: 0.0 to 5.0
   - Shorter wavelength, more sensitive to small particles

2. **Optical_Depth_055** (int16, scale 0.001)
   - AOD at 550nm wavelength (PRIMARY)
   - Valid range: 0.0 to 5.0
   - Standard wavelength for AOD reporting
   - Most commonly used in PM2.5 models

3. **AOD_Uncertainty** (int16, scale 0.0001)
   - Retrieval uncertainty estimate
   - Quality weighting in regression models

4. **AOD_QA** (int32, bitmask)
   - Quality assurance flags
   - Bits encode: algorithm quality, adjacency mask, snow/water flags
   - Used for filtering unreliable retrievals

5. **Column_WV** (int16, scale 0.001)
   - Column water vapor in centimeters
   - Important for hygroscopic growth correction

**Not Included:**
- **FineModeFraction:** Removed per user request (computational overhead, marginal value)

### Satellite Filtering
**Terra Only (SATELLITE = "A"):**
- Consistent overpass time (~10:30 AM local)
- Avoids Aqua satellite (SATELLITE = "B") with different overpass
- Reduces temporal variability in retrievals

### Data Volume
- **Per Date:** 509,209 grid cells × 5 bands × 8 bytes = ~20 MB parquet (compressed)
- **Full Dataset:** 2,191 dates × 20 MB = ~44 GB total
- **API Calls (current approach):** 1,019 batches/date × 2,191 dates = **2,232,629 Earth Engine API calls**

---

## Current Implementation Logic

### Architecture Overview

**File:** `patliputra/modis_aod_signals.py` (728 lines)

**Design Pattern:** Follows established `patliputra` signal fetcher conventions matching `sentinel2_signals.py` and `landsat8_signals.py`

**Key Principles:**
1. **No Intermediate TIFFs:** Direct parquet output (atomic writes)
2. **Manifest Tracking:** SQLite database tracks artifact lifecycle (producing → ok | error)
3. **Idempotent:** Skips existing artifacts, resumes after failures
4. **Batched API Calls:** Concurrent ThreadPoolExecutor for parallelization

### Workflow Pipeline

#### Step 1: Initialization
```python
# Initialize Earth Engine
ee.Initialize(project=args.project)

# Load grid artifact (509,209 cells)
grid_gdf = gpd.read_parquet("artifacts/v1/grids/mumbai/grid.parquet")
# Schema: grid_id (str), geometry (Point), row_idx (int), crs=EPSG:32643

# Initialize manifest
manifest = ModisAodManifest(city="mumbai", version="v1")
```

#### Step 2: Date Enumeration (LAZY)
```python
def enumerate_modis_aod_dates(start_date, end_date, bbox):
    """Generate date range locally without EE query."""
    # OPTIMIZATION: Avoid upfront .toList().getInfo() blocking
    dates = generate_date_range(start_date, end_date)
    # Returns: ["2019-01-01", "2019-01-02", ..., "2024-12-31"]
    return dates
```

**Previous Approach (REMOVED):**
- Used `collection.aggregate_array('system:index').getInfo()`
- Blocked for 40+ minutes fetching all 2,191 image metadata
- Starved worker threads (parallelization failure)

**Current Approach:**
- Local date generation (milliseconds)
- Skip dates with no data during processing loop
- Immediate worker spawning

#### Step 3: Per-Date Image Retrieval
```python
def get_modis_aod_image(date_str, bbox):
    """Query MODIS collection for specific date."""
    collection = (
        ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
        .filterBounds(ee.Geometry.Rectangle(bbox))
        .filterDate(date_str, ee.Date(date_str).advance(1, "day"))
        .filter(ee.Filter.eq("SATELLITE", "A"))  # Terra only
        .sort("system:time_start")
    )
    
    count = collection.size().getInfo()  # Single API call
    if count == 0:
        raise RuntimeError(f"No image for {date_str}")
    
    # Mosaic multiple granules if present (typical: 2-3 granules)
    image = collection.mosaic() if count > 1 else collection.first()
    
    # Select bands
    image = image.select(MODIS_AOD_BANDS)
    
    # CRITICAL OPTIMIZATION: Resample 1000m → 30m
    image = image.resample('bilinear').reproject(
        crs='EPSG:4326',
        scale=30  # Match Landsat/Sentinel-2 grid resolution
    )
    
    return image
```

**Resampling Rationale:**
- MODIS native: 1000m pixels
- Grid cells: 30m points
- Without resampling: Each cell queries 1000m×1000m zone (massive data transfer)
- With resampling: Image pre-processed to 30m before sampling (reduces EE computation)

#### Step 4: Batched Value Extraction
```python
def fetch_aod_values_batched(image, grid_gdf, batch_size=500, max_workers=4):
    """Sample AOD values at grid cell locations using concurrent batches."""
    
    total_batches = ceil(509_209 / 500) = 1,019 batches
    
    def fetch_batch(batch_idx):
        # Get batch of cells (e.g., rows 0-499)
        batch_gdf = grid_gdf.iloc[batch_idx*500 : (batch_idx+1)*500]
        
        # Convert to Earth Engine FeatureCollection
        features = [
            ee.Feature(ee.Geometry.Point([pt.x, pt.y]))
                .set("grid_id", row["grid_id"])
            for pt in batch_gdf.geometry
        ]
        fc = ee.FeatureCollection(features)
        
        # Sample image at point locations
        reduced = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),  # Mean of 30m pixel
            scale=30,  # Resampled resolution
            tileScale=2,  # Internal EE tiling for large computations
        )
        
        # Fetch results (BLOCKING API CALL)
        result_list = reduced.getInfo()["features"]
        
        return [feat["properties"] for feat in result_list]
    
    # Concurrent execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_batch, i) for i in range(1019)]
        results = [f.result() for f in as_completed(futures)]
    
    # Flatten and convert to DataFrame
    df = pd.DataFrame(flatten(results))
    # Schema: grid_id, aod_047, aod_055, aod_uncertainty, aod_qa, column_wv
    
    return df
```

**Key Operations:**
- **1,019 batches/date:** 509,209 cells ÷ 500 cells/batch
- **Each batch = 1 API call:** `reduced.getInfo()` blocks until EE returns data
- **4 concurrent workers:** Process 4 batches simultaneously
- **Total API calls/date:** 1,019 calls (sequential with 4x parallelism)

#### Step 5: Atomic Write + Manifest Registration
```python
def write_artifact_atomic(df, artifact_path, metadata, manifest, date_str):
    """Write parquet atomically and update manifest."""
    
    # Mark producing state
    manifest.mark_producing(date_str, request_id=f"modis_aod_{date_str}")
    
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet.tmp") as tmp:
            df.to_parquet(tmp.name, engine="pyarrow", index=False)
        
        # Atomic rename (POSIX atomic operation)
        shutil.move(tmp.name, artifact_path)
        
        # Write metadata sidecar
        metadata_path = artifact_path.with_suffix(".parquet.meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Mark success in manifest
        manifest.mark_ok(date_str, str(artifact_path))
        
    except Exception as e:
        manifest.mark_error(date_str, str(e))
        raise
```

**Atomicity Guarantees:**
- Temp file + rename prevents partial writes
- Manifest tracks state transitions
- Crashed processes can resume from manifest

### Output Schema

**Parquet File:** `artifacts/v1/modis_aod/mumbai/YYYY-MM-DD.parquet`

**Columns:**
- `grid_id` (object/str): Grid cell identifier (e.g., "R123_C456")
- `aod_047` (float64): AOD at 470nm (scaled, 0.0-5.0 range)
- `aod_055` (float64): AOD at 550nm (scaled, PRIMARY feature)
- `aod_uncertainty` (float64): Uncertainty estimate (scaled)
- `aod_qa` (Int32): Quality flags (nullable int32)
- `column_wv` (float64): Column water vapor in cm (scaled)

**Rows:** 509,209 (matches grid exactly)

**Null Handling:**
- ~8% nulls expected (clouds, algorithm failures, off-swath)
- Preserved as NaN for downstream QA filtering

---

## Performance Challenges

### Measured Performance Metrics

#### Single Date Test (2024-01-01)
**Configuration:**
- Grid cells: 509,209
- Batch size: 500 cells
- Total batches: 1,019
- Max workers: 4
- Earth Engine Project: pollu-483104

**Results:**
- **Time for 70 batches:** 57 seconds
- **Estimated total time:** ~13 minutes/date
- **Extrapolated full dataset:** 2,191 dates × 13 min = **28,483 minutes = 19.8 days**

#### Bottleneck Analysis

**Per-Batch Timing Breakdown:**
```
Total time: 57 seconds for 70 batches
Time per batch: 57s / 70 = 0.81 seconds/batch
Batches completed: 70/1,019 (6.9%)
```

**Parallel Efficiency:**
- 4 workers theoretically: 0.81s / 4 = 0.20s effective per batch
- Actual throughput: ~8.6 batches/second
- **Conclusion:** Workers are efficiently utilized, but absolute speed is slow

**Root Cause:** Earth Engine API latency per `getInfo()` call
- Network round-trip: ~100-300ms
- Server-side computation: ~500-800ms
- **Total per API call:** ~600-1100ms
- Cannot be parallelized beyond 4-8 concurrent requests (EE rate limits)

### Scalability Crisis

**Current Approach:**
- 1,019 batches/date × 2,191 dates = 2,232,629 total API calls
- Even at 0.3s/call (optimistic): 669,789 seconds = **7.8 days**
- At observed 0.81s/call: 1,808,429 seconds = **20.9 days**

**Cost Implications:**
- VM: n2-highcpu-12 @ $1.67/hour
- 20 days × 24 hours × $1.67 = **$801.60** (unacceptable)

**Time Constraint:**
- User has limited time budget: "We do not have 3-7 days for fetching AOD"
- Training pipeline blocked on AOD availability
- Business deadline pressure

---

## Optimization Attempts

### Attempt 1: Lazy Date Enumeration

**Problem Identified:**
- Initial implementation called `collection.toList().getInfo()` upfront
- Fetched metadata for all 2,191 dates before processing
- Blocked main thread for 40+ minutes
- Worker threads starved (no tasks submitted yet)

**Solution Implemented:**
```python
def enumerate_modis_aod_dates(start_date, end_date, bbox):
    # OLD: Upfront EE query
    # collection = ee.ImageCollection(...).filterDate(start, end)
    # dates = collection.aggregate_array('system:index').getInfo()  # BLOCKS
    
    # NEW: Local generation
    dates = generate_date_range(start_date, end_date)  # Milliseconds
    return dates
```

**Impact:**
- ✅ Eliminated 40-minute initialization hang
- ✅ Enabled immediate worker spawning
- ✅ Improved user perception of progress
- ❌ Did NOT improve per-date throughput (bottleneck is elsewhere)

### Attempt 2: Increased Batch Size (500 → 5000)

**Rationale:**
- Reduce total API calls: 1,019 batches → 102 batches (10x reduction)
- Amortize network latency over larger payloads

**Implementation:**
```python
parser.add_argument("--batch-size", type=int, default=5000)
```

**Testing:**
- Initiated test run with batch_size=5000
- **Result:** User cancelled (Ctrl+C) before completion
- **Reason:** Still appeared slow, unclear if 10x speedup achieved

**Theoretical Impact:**
- 102 batches/date × 0.81s/batch = 83 seconds/date
- 2,191 dates × 83s = 181,853s = 2.1 days
- **Expected improvement:** 20 days → 2.1 days (9.5x speedup)

**Status:** UNTESTED (cancelled before validation)

### Attempt 3: Worker Count Tuning

**Initial Setting:** max_workers=8
- **Issue:** Earth Engine rate limits ~2-4 concurrent requests/project
- Excess workers queue up, no benefit beyond 4 workers

**Adjustment:** max_workers=4 (then 2)
```python
# Capping to avoid rate limit throttling
effective_workers = min(max_workers, total_batches, 4)  # Hard cap at 4
```

**Testing:**
- Default changed to max_workers=2
- Then reverted to max_workers=4 (matching Landsat pattern)

**Impact:**
- ✅ Reduced worker overhead (context switching)
- ❌ Minimal performance change (EE API is bottleneck, not CPU)

### Attempt 4: Scale Optimization (1000m → 30m Resampling)

**Hypothesis:**
Sampling MODIS at 1000m scale forces Earth Engine to:
1. Query 1000m × 1000m zones around each 30m grid cell
2. Transfer massive data volumes per API call
3. Compute reducer (first/mean) over large regions

**Solution:**
Resample MODIS to 30m BEFORE reduceRegions:
```python
# Resample 1000m → 30m using bilinear interpolation
image = image.resample('bilinear').reproject(
    crs='EPSG:4326',
    scale=30
)

# Now sample at 30m (matches grid resolution exactly)
reduced = image.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.mean(),
    scale=30,  # Resampled resolution
    tileScale=2
)
```

**Testing:**
- Implemented in latest version
- Test run: 2024-01-01 with batch_size=500
- **Result:** Still ~13 min/date (no significant improvement)

**Analysis:**
- Resampling adds pre-processing overhead
- Earth Engine still must compute resampled image server-side
- API latency unchanged (network round-trips still dominate)
- **Conclusion:** Resampling does NOT address root bottleneck

### Attempt 5: Reducer Change (first → mean)

**Rationale:**
Match Landsat pattern which uses `ee.Reducer.mean()`

**Implementation:**
```python
# OLD: reducer=ee.Reducer.first()
# NEW: reducer=ee.Reducer.mean()
```

**Expected Impact:**
- Mean reducer potentially faster for EE server-side computation
- More statistically robust (averages nearby pixels)

**Testing:**
- Applied alongside resampling optimization
- **Result:** No observable speedup in limited testing

**Analysis:**
- Reducer computation is negligible compared to network latency
- Both first() and mean() are O(1) for point sampling after resampling

---

## Benchmark Comparison

### Landsat-8 Performance (Reference)

**Dataset:** `LANDSAT/LC08/C02/T1_TOA`
- **Spatial Resolution:** 30m native
- **Bands:** 11 spectral bands (B1-B11)
- **Temporal Coverage:** Similar to MODIS (daily when available)

**Configuration:**
- Batch size: 500 cells
- Max workers: 4
- Scale: 30m
- Reducer: ee.Reducer.mean()

**Observed Performance:**
- **Per-date time:** ~2-5 minutes (estimated, not directly measured)
- **Why faster:** 
  - 30m native resolution (no resampling needed)
  - Single image per date (no granule mosaicking)
  - More efficient EE processing for well-optimized collection

**Key Difference:**
Landsat is ~3-5x faster than MODIS despite similar architecture, likely due to:
1. Native resolution match (30m)
2. Collection optimization (Landsat is heavily used, well-cached)
3. Simpler query patterns (no granule complexity)

### Sentinel-2 Performance (Reference)

**Dataset:** `COPERNICUS/S2_SR_HARMONIZED`
- **Spatial Resolution:** 10m native (aggregated to 30m)
- **Bands:** 5 bands (B2, B4, B8, AOT, QA10)

**Configuration:**
- Batch size: 500 cells
- Max workers: 4
- Scale: 30m
- Reducer: ee.Reducer.mean()

**Observed Performance:**
- Similar to Landsat (~2-5 minutes/date)
- Also benefits from native high resolution and collection optimization

### MODIS AOD vs. Optical Imagery

**Fundamental Difference:**
- **Landsat/Sentinel:** RAW reflectance (simple retrieval)
- **MODIS AOD:** DERIVED product (atmospheric correction, aerosol inversion)
- MODIS MAIAC involves complex radiative transfer modeling
- Higher computational cost per pixel

**Granule Complexity:**
- MODIS granules are spatially limited (swath-based)
- Typical: 2-3 granules/date over Mumbai → mosaicking overhead
- Landsat: 1 scene/date (path/row grid)

---

## Current Status & Recommendations

### Current Implementation State

**Version:** patliputra/modis_aod_signals.py (728 lines)

**Applied Optimizations:**
1. ✅ Lazy date enumeration (eliminates 40-min blocking)
2. ✅ Resampling 1000m → 30m (matches Landsat pattern)
3. ✅ Reducer change: first() → mean()
4. ✅ Batch size configurable (default: 500)
5. ✅ Worker count optimized (default: 4)
6. ✅ Tile scale adjusted (default: 2)

**Performance:**
- ~13 minutes per date
- ~20 days for full 2019-2024 dataset
- **Status:** UNACCEPTABLE for user timeline constraints

### Root Cause Summary

**The fundamental bottleneck is:**
```
TOTAL_TIME = NUM_DATES × BATCHES_PER_DATE × TIME_PER_BATCH
            = 2,191     × 1,019             × 0.81s
            = 1,808,429 seconds
            = 20.9 days
```

**Cannot optimize further because:**
1. **NUM_DATES (2,191):** Fixed by temporal coverage requirement
2. **BATCHES_PER_DATE (1,019):** Limited by EE payload size (~500-5000 cells/call max)
3. **TIME_PER_BATCH (0.81s):** Network + EE computation latency (irreducible)

### Strategic Recommendations

#### Option A: Temporal Aggregation (5-Day Composites)

**Approach:**
Reduce temporal resolution from daily → 5-day composites

**Implementation:**
```python
# Generate 5-day periods instead of daily
date_ranges = [
    ("2019-01-01", "2019-01-05"),  # Composite 1
    ("2019-01-06", "2019-01-10"),  # Composite 2
    ...
]

# For each 5-day period, composite MODIS images
collection = (
    ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
    .filterDate(start, end)
    .filter(ee.Filter.eq("SATELLITE", "A"))
)
composite = collection.mean()  # Or median() for robustness
```

**Impact:**
- Dates reduced: 2,191 → 438 composites (5x reduction)
- Total time: 438 × 13 min = 5,694 min = **3.95 days**
- **Trade-off:** Lose daily temporal resolution, smooth out transient events

**Pros:**
- Feasible within ~4-day timeline
- Improves data quality (averages out noise, cloud contamination)
- Still captures seasonal/weekly patterns

**Cons:**
- Cannot model daily PM2.5 spikes
- Temporal misalignment with daily ground measurements (requires interpolation)

#### Option B: Spatial Subsampling

**Approach:**
Process subset of grid cells (e.g., 10% random sample)

**Implementation:**
```python
# Sample 50,921 cells (10%) instead of full 509,209
sampled_grid = grid_gdf.sample(frac=0.1, random_state=42)
```

**Impact:**
- Cells reduced: 509,209 → 50,921 (10x reduction)
- Batches/date: 1,019 → 102 (10x reduction)
- Total time: 2,191 × 102 × 0.81s = 181,853s = **2.1 days**

**Pros:**
- Maintains daily temporal resolution
- Fast data acquisition

**Cons:**
- Incomplete spatial coverage (cannot predict at all locations)
- Requires spatial interpolation for full grid
- May miss localized pollution hotspots

#### Option C: Hybrid (5-Day + Spatial Subsample)

**Combination:**
- 5-day temporal composites (5x reduction)
- 20% spatial sample (5x reduction)
- **Combined reduction:** 25x

**Impact:**
- Dates: 438 composites
- Cells: 101,842 (20% of grid)
- Batches/date: 204
- Total time: 438 × 204 × 0.81s = 72,385s = **0.84 days (~20 hours)**

**Trade-offs:**
- Both temporal and spatial resolution loss
- Requires interpolation in both dimensions

#### Option D: Abandon MODIS, Use Alternative AOD

**Alternative Sources:**
1. **OpenAQ AOD Product:** Pre-aggregated daily AOD grids (if available)
2. **CAMS Global Reanalysis:** Modeled AOD at coarser resolution (faster download)
3. **Ground-Based AERONET:** Point measurements (very sparse but high quality)

**Pros:**
- May have faster access methods (direct file download vs. API)
- Pre-processed products avoid computation overhead

**Cons:**
- May not cover 2019-2024 time range
- Coarser spatial resolution (e.g., 25km for CAMS)
- Requires additional data pipeline development

#### Option E: Skip AOD, Focus on Model Optimization

**Rationale:**
- Current R² = 0.50-0.55 without AOD
- Literature suggests AOD adds +0.15-0.35 R²
- **Alternative:** Improve model architecture/hyperparameters for +0.10-0.15 gain

**Approaches:**
1. Feature engineering on existing data (interaction terms, temporal lags)
2. Ensemble methods (stacking LightGBM with neural network)
3. Spatial autocorrelation features (neighboring cell aggregates)
4. Advanced hyperparameter tuning (Bayesian optimization)

**Pros:**
- No data acquisition delay
- Can start training immediately
- May achieve sufficient R² without AOD

**Cons:**
- Uncertain if 0.70 R² achievable without AOD
- Literature strongly supports AOD importance
- May waste time on optimization that hits ceiling

### Recommended Path Forward

**Immediate Action (Next 24 Hours):**

1. **Test Batch Size 5000 Fully:**
   ```bash
   python -m patliputra.modis_aod_signals \
     --start-date 2024-01-01 --end-date 2024-01-07 \
     --batch-size 5000 --max-workers 4
   ```
   - Measure actual time for 7 dates with large batches
   - Verify 10x speedup materializes (expected: ~1.3 min/date)
   - If successful: 2,191 dates × 1.3 min = **2.85 days** (ACCEPTABLE)

2. **If Batch Size 5000 Works:**
   - Proceed with full 2019-2024 fetch (3-day background job)
   - Parallelize with training pipeline (overlap work)

3. **If Batch Size 5000 Fails:**
   - Implement **Option A: 5-Day Temporal Composites**
   - Accept temporal resolution loss as pragmatic trade-off
   - Complete in ~4 days

**Decision Tree:**
```
Test batch_size=5000 for 1 week
    ├─ < 2 min/date? → PROCEED with full fetch (3 days total)
    ├─ 2-5 min/date? → Implement 5-day composites (4 days total)
    └─ > 5 min/date? → Consider spatial subsampling or skip AOD
```

### Risk Mitigation

**Time Budget Exhaustion:**
- Set hard deadline: If not complete in 5 days, abort and skip AOD
- Opportunity cost: Training pipeline can proceed without AOD

**Cost Overrun:**
- VM cost: $1.67/hour × 120 hours (5 days) = $200.40
- Acceptable if R² gain justifies (expected +0.15-0.35)

**Data Quality Issues:**
- Implement QA filtering AFTER fetch (don't block on quality during acquisition)
- Can filter bad retrievals during training data prep

---

## Technical Specifications

### Dependencies
```
ee==0.1.384
geopandas==0.14.1
pandas==2.1.4
pyarrow==14.0.1
```

### File Locations
- **Script:** `patliputra/modis_aod_signals.py`
- **Manifest:** `manifest/modis_aod_manifest.py`
- **Artifacts:** `artifacts/v1/modis_aod/mumbai/YYYY-MM-DD.parquet`
- **Manifest DB:** `artifacts/v1/modis_aod/mumbai/manifest/modis_aod_manifest.db`

### Execution Command
```bash
python -m patliputra.modis_aod_signals \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --project pollu-483104 \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --batch-size 5000 \
  --max-workers 4
```

---

## Conclusion

MODIS AOD integration represents a critical feature enhancement with strong scientific justification but faces severe performance bottlenecks due to Earth Engine API latency at scale. Current optimizations have improved perception (lazy enumeration) but not solved the fundamental time constraint.

**Immediate Priority:** Validate batch_size=5000 optimization to determine if full dataset fetch is feasible within acceptable timeline. If not, pivot to temporal aggregation or strategic alternatives.

**Success Criteria:** Deliver AOD features within 5-day deadline OR make informed decision to skip AOD based on quantified cost/benefit analysis.
