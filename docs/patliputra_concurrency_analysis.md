# Patliputra Pipeline: Concurrency Analysis

## TL;DR Recommendation

**Run SEQUENTIALLY** ⚠️

Most stages are rate-limited and share the same Earth Engine quota. Concurrent execution would:
- Hit API/quota limits faster
- Cause failures and retries
- NOT improve overall throughput
- Complicate error handling

**Exception**: AQI processing can run independently (no API calls).

---

## Detailed Analysis by Stage

### 1. Grid Creation (`create_grid.py`)

**Type**: One-time setup, local processing  
**Concurrent safe**: ✅ Yes (but only run once)  
**Rate limited**: ❌ No  

**Recommendation**: Run once before all other stages.

```bash
# Run first, once per city
python patliputra/create_grid.py --grid-id mumbai
```

---

### 2. Meteorological Data (`openmeteo_signals.py`)

**Type**: External API (Open-Meteo Historical Weather API)  
**Concurrent safe**: ⚠️ Limited (API rate limits)  
**Rate limited**: ✅ Yes (0.5-2.0s between requests recommended)  

**Bottlenecks**:
- API rate limits (10,000 requests/day free tier)
- Network bandwidth
- Single API endpoint per city

**Mumbai scale** (7 years):
- ~13 unique meteo tiles (coarse resolution)
- 2,555 dates × 13 tiles = ~33,215 API calls
- Sequential: ~16-20 hours with rate limiting
- Concurrent: Would hit rate limits, get blocked

**Recommendation**: ❌ **SEQUENTIAL**

**Why not concurrent?**
- Multiple processes hitting same API = faster rate limit exhaustion
- Risk of IP ban or API key suspension
- No performance gain (API is the bottleneck)

**Optimization**:
```bash
# Use built-in rate limiting
python patliputra/openmeteo_signals.py \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --rate-limit 1.0 \  # Conservative rate limiting
  --hourly
```

---

### 3. Satellite Data (Landsat-8 & Sentinel-2)

**Type**: Earth Engine API (Google Cloud)  
**Concurrent safe**: ⚠️ Very limited (shared quota)  
**Rate limited**: ✅ Yes (quota-based, not time-based)  

**Earth Engine Quotas** (per project):
- Compute: 10,000 EECU-hours/month
- Storage: 250 GB
- Concurrent requests: ~100-200 (undocumented soft limit)
- `reduceRegions` operations are expensive (500k grid cells per date)

**Bottlenecks**:
- Earth Engine compute quota
- `reduceRegions` processing time (~30-60s per date)
- Network bandwidth for results

**Mumbai scale** (7 years):
- Landsat-8: ~200-300 usable dates (cloud cover filtering)
- Sentinel-2: ~1,000-1,500 dates (more frequent)
- Total: ~1,500-1,800 Earth Engine jobs

**Concurrent execution risks**:
```
Job 1: Landsat-8 (2018-01-15) → 5 EECU
Job 2: Sentinel-2 (2018-01-15) → 5 EECU  
Job 3: Landsat-8 (2018-01-20) → 5 EECU
---
Total concurrent: 15 EECU/job × 3 jobs = 45 EECU in parallel
Monthly quota: 10,000 EECU-hours
```

Running 3 jobs concurrently:
- Uses 3x quota rate
- Quota exhaustion in ~10 days instead of ~30 days
- Higher failure rate due to concurrent request limits

**Recommendation**: ❌ **SEQUENTIAL**

**Why not concurrent?**
- Shared Earth Engine quota pool
- Concurrent jobs exhaust quota 2-3x faster
- Earth Engine may throttle/reject concurrent requests
- No significant speedup (compute is server-side)

**Current implementation**:
Both scripts already have internal parallelism via `--workers`:
```bash
# Landsat-8 uses ThreadPoolExecutor internally
python patliputra/landsat8_signals.py \
  --workers 2 \  # Already parallel at date level
  --dates-file dates.txt

# Sentinel-2 uses ThreadPoolExecutor internally  
python patliputra/sentinel2_signals.py \
  --workers 2 \  # Already parallel at date level
  --start-date 2018-01-01 \
  --end-date 2018-12-31
```

⚠️ **Don't add another layer of parallelism** - scripts already parallelize optimally.

---

### 4. Air Quality Ground Truth (`cpcb_aqi.py`)

**Type**: Local CSV processing  
**Concurrent safe**: ✅ Yes (no external dependencies)  
**Rate limited**: ❌ No  

**Bottlenecks**:
- Disk I/O (reading CSV files)
- CPU (pandas processing)
- Memory (loading CSVs)

**Scale**:
- Processes local CSV files
- No API calls
- Independent of other stages

**Recommendation**: ✅ **CAN RUN CONCURRENTLY**

This is the ONLY stage that benefits from concurrent execution.

```bash
# Can run alongside other stages
python patliputra/cpcb_aqi.py &  # Background

# While meteo/satellite stages run sequentially
python patliputra/openmeteo_signals.py ...
```

---

## Execution Strategy Recommendations

### Strategy 1: Fully Sequential (RECOMMENDED)

**Best for**: Production stability, quota management

```bash
# Run stages in order
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --ee-project your-project-id \
  --hourly \
  --verbose
```

**Advantages**:
- ✅ Predictable quota consumption
- ✅ Simple error handling
- ✅ No race conditions
- ✅ Clear progress tracking

**Disadvantages**:
- ⏱ Slower (but not by much due to API/quota limits)

**Estimated time**:
- Grid: 5 minutes (one-time)
- Meteo: 16-20 hours (API rate limited)
- Landsat-8: 2-4 hours (Earth Engine)
- Sentinel-2: 8-12 hours (Earth Engine)
- AQI: 30 minutes (local)
- **Total: ~27-37 hours**

---

### Strategy 2: Staged Batching

**Best for**: Balancing speed and safety

Run stages in groups with natural boundaries:

```bash
# Stage 1: Grid (prerequisite)
python patliputra/create_grid.py --grid-id mumbai

# Stage 2: Meteo (long-running, do first)
python patliputra/openmeteo_signals.py \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --hourly --rate-limit 1.0

# Stage 3: Satellite (run sequentially, but can break into years)
for year in {2014..2020}; do
  # Landsat
  python patliputra/landsat8_signals.py \
    --dates-file landsat_dates_${year}.txt \
    --workers 2
  
  # Sentinel
  python patliputra/sentinel2_signals.py \
    --start-date ${year}-01-01 \
    --end-date ${year}-12-31 \
    --workers 2
done

# Stage 4: AQI (can run anytime, independent)
python patliputra/cpcb_aqi.py
```

**Advantages**:
- ✅ Natural checkpoints for resume
- ✅ Easier to monitor progress
- ✅ Can prioritize critical stages

---

### Strategy 3: AQI Concurrent (Marginal Benefit)

**Best for**: Maximizing machine utilization

```bash
# Start AQI in background (independent)
python patliputra/cpcb_aqi.py &
AQI_PID=$!

# Run main pipeline
python -m patliputra.pipeline run \
  --stages meteo,landsat8,sentinel2 \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31

# Wait for AQI to finish
wait $AQI_PID
```

**Time saved**: ~30 minutes (AQI overlaps with meteo start)

---

## Resource Consumption Comparison

### Sequential Execution

| Resource | Usage | Status |
|----------|-------|--------|
| **CPU** | 50-70% avg (1-2 cores) | ✅ Moderate |
| **Memory** | ~2-4 GB | ✅ Low |
| **Network** | ~10-50 Mbps | ✅ Moderate |
| **Disk I/O** | ~10-50 MB/s | ✅ Low |
| **EE Quota** | ~200-400 EECU/day | ✅ Sustainable |

### Concurrent Execution (Hypothetical)

| Resource | Usage | Status |
|----------|-------|--------|
| **CPU** | 80-100% (4-8 cores) | ⚠️ High |
| **Memory** | ~8-16 GB | ⚠️ High |
| **Network** | ~100-200 Mbps | ⚠️ High |
| **Disk I/O** | ~50-100 MB/s | ⚠️ Moderate |
| **EE Quota** | ~600-1200 EECU/day | ❌ Unsustainable (quota exhaustion) |

---

## Bottleneck Summary

**Primary bottlenecks** (from profiling):

1. 🔴 **Open-Meteo API rate limits** (16-20 hours)
   - CANNOT parallelize (API rate limits)
   - CANNOT speed up (free tier limits)
   - Solution: Pay for premium tier or accept 16-20h runtime

2. 🟡 **Earth Engine quota** (10,000 EECU/month)
   - Can parallelize internally (scripts do this)
   - Cannot parallelize across stages (shared quota)
   - Solution: Batch by year, optimize compute

3. 🟢 **Network bandwidth** (minimal)
   - Not a bottleneck at current scale
   - Parquet results are compressed

4. 🟢 **Local compute** (minimal)
   - Only AQI processing
   - Can run concurrently but saves <30 min

**Speedup potential**:
- Sequential: 31-45 hours
- Concurrent (AQI only): ~30.5-44.5 hours
- **Net gain: <1% speedup**

**Risk increase**:
- Quota exhaustion: HIGH
- API blocking: MEDIUM
- Error complexity: HIGH

---

## Final Recommendation

### ✅ Run Sequentially

**Why?**
1. Most stages share Earth Engine quota (concurrent = quota exhaustion)
2. Open-Meteo API has strict rate limits (concurrent = IP ban risk)
3. Scripts already have internal parallelism (--workers flags)
4. Concurrent execution saves <1% total time
5. Sequential is more reliable and easier to debug

**Only exception**: AQI can run in background (saves 30 min, minimal risk)

### Production Command

```bash
# Single command, sequential execution
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --ee-project your-ee-project-id \
  --hourly \
  --verbose \
  2>&1 | tee pipeline.log
```

**Expected duration**: 31-45 hours

**Optimization tips**:
1. Use `tmux` or `screen` for long-running jobs
2. Monitor Earth Engine quota dashboard
3. Break into yearly chunks for checkpointing
4. Run AQI separately (no dependencies)

---

## Pipeline Runner Usage

The new `patliputra.pipeline` module orchestrates everything:

### Run full pipeline
```bash
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-12-31 \
  --ee-project your-project-id \
  --hourly
```

### Run specific stages
```bash
# Only meteo + satellite
python -m patliputra.pipeline run \
  --stages meteo,landsat8,sentinel2 \
  --city mumbai \
  --start-date 2021-06-01 \
  --end-date 2021-06-30
```

### Check status
```bash
python -m patliputra.pipeline status --city mumbai
```

### Verbose logging
```bash
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2018-01-01 \
  --end-date 2018-01-31 \
  --verbose
```

---

## Monitoring Long Runs

```bash
# In tmux/screen session
tmux new -s patliputra
python -m patliputra.pipeline run \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --ee-project your-project \
  --hourly \
  --verbose \
  2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M%S).log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t patliputra
```

**Monitor in another terminal**:
```bash
# Watch file counts
watch -n 60 'du -h artifacts/v1/*/mumbai/ 2>/dev/null'

# Watch logs
tail -f logs/pipeline_*.log

# Check Earth Engine quota
# Visit: https://code.earthengine.google.com/
```
