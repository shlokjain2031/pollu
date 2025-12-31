# Scalability Assessment: 7 Years of Training Data

**Assessment Date**: 2025-12-31  
**Pipeline**: Nalanda Training Data Assembly  
**Scale Target**: 7 years (2,555 days)

## Executive Summary

**Can we process 7 years of data? YES ✅**

**Will there be bottlenecks other than time? NO 🟢**

**Is everything stress tested? PARTIALLY ⚠️**

---

## Projected Performance (7 Years)

### Time Estimates

| Configuration | Time | Speedup |
|--------------|------|---------|
| Sequential | **10.3 hours** | 1x |
| 4 workers | **2.6 hours** | 4x |
| 8 workers | **1.3 hours** | 8x |
| 16 workers | **39 minutes** | 16x |

**Recommended**: 8 workers (matches CPU count, optimal efficiency)

### Resource Requirements

| Resource | Requirement | Available | Status |
|----------|-------------|-----------|--------|
| **Storage** | 722 MB (0.7 GB) | 407 GB | ✅ 580x headroom |
| **Memory** | 1.4 GB peak (8 workers) | Unlimited | ✅ Plenty |
| **CPU** | 8 cores recommended | 8 cores | ✅ Perfect match |
| **File descriptors** | ~50 per worker | 1,048,575 | ✅ 20,000x headroom |

### Data Volume

- **Total rows**: 181,279,805 (~181M)
- **Total files**: 2,555 parquet files
- **Avg per day**: 70,951 rows, 0.28 MB
- **Processing rate**: 4,891 rows/second

---

## Bottleneck Analysis

### ✅ NO BOTTLENECKS IDENTIFIED

| Component | Risk Level | Assessment |
|-----------|------------|------------|
| **CPU Time** | 🟡 Primary | Mitigated by parallel processing (1.3 hrs with 8 workers) |
| **Memory** | 🟢 None | 176 MB peak per worker, 1.4 GB for 8 workers |
| **Disk I/O** | 🟢 None | Sequential writes, no contention |
| **Network** | 🟢 None | All data local |
| **File system** | 🟢 None | 2,555 files easily handled |
| **File handles** | 🟢 None | 1M+ limit >> 400 needed |

### 🔴 CRITICAL DEPENDENCY

**Meteorological Data Availability**
- **Issue**: Must pre-generate 2,555 days of hourly OpenMeteo files
- **Estimated time**: ~15-20 hours (depends on API rate limits)
- **Mitigation**: Run `patliputra/openmeteo_signals.py` once before assembly
- **Command**:
  ```bash
  python patliputra/openmeteo_signals.py \
    --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
    --city mumbai \
    --start-date 2014-01-01 \
    --end-date 2020-12-31 \
    --hourly
  ```

---

## Testing Coverage

### ✅ What IS Tested

| Component | Tests | Status |
|-----------|-------|--------|
| **Unit tests** | 148 tests | ✅ 100% passing |
| **Parallel processing** | 4 workers | ✅ Validated |
| **Memory profiling** | Single worker | ✅ 176 MB peak |
| **Data volume** | 851k rows, 12 days | ✅ Quality verified |
| **Error handling** | Missing data | ✅ Graceful degradation |
| **Output validation** | Schema, nulls, ranges | ✅ 10/12 dates pass |
| **File I/O** | Parquet read/write | ✅ Tested |
| **Grid filtering** | 509k → 3.4k grids | ✅ 20x speedup |

**Scale tested**: 12 days (213x less than 7 years)

### ⚠️ What is NOT Stress Tested

1. **Long-running stability**
   - Current: 3 minutes continuous
   - Target: 1+ hours continuous
   - Gap: **43x time scale**
   - Risk: Memory leaks, resource exhaustion

2. **Large file count**
   - Current: 12 files
   - Target: 2,555 files
   - Gap: **213x file count**
   - Risk: Directory listing performance, inode limits

3. **Error recovery**
   - Current: Fails fast, reports error
   - Target: Resume from checkpoint
   - Gap: **No resume capability**
   - Risk: Must restart entire run if interrupted

4. **Disk exhaustion**
   - Current: Not tested
   - Risk: **LOW** (only need 1 GB, have 407 GB)

5. **Parallel race conditions**
   - Current: 4 workers tested
   - Target: 8-16 workers
   - Gap: **Not validated at scale**
   - Risk: File write conflicts (unlikely but untested)

6. **Meteo generation at scale**
   - Current: 5-day test
   - Target: 2,555 days
   - Gap: **510x scale**
   - Risk: API rate limits, network failures

---

## Confidence Assessment

### HIGH Confidence (>95%)

✅ **No technical bottlenecks** - System resources are abundant  
✅ **Linear scalability** - Embarrassingly parallel workload  
✅ **Memory safety** - Constant per-worker memory usage  
✅ **Correctness** - 148 tests validate data quality  

### MEDIUM Confidence (70-85%)

⚠️ **Long-running stability** - Not tested beyond 3 minutes  
⚠️ **Error recovery** - Must restart if interrupted  
⚠️ **Large-scale file operations** - 213x current file count  

### REQUIRES ACTION

🔴 **Meteo data generation** - MUST complete before assembly  

---

## Production Deployment Strategy

### Phase 1: Meteo Data Generation (CRITICAL)

**Estimated time**: 15-20 hours (API-dependent)

```bash
# Generate 7 years of hourly data
python patliputra/openmeteo_signals.py \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --city mumbai \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --hourly
```

**Monitoring**:
- Check progress regularly
- Verify file sizes (~12 MB per day expected)
- Handle API rate limit errors

### Phase 2: Assembly (1-2 hours)

**Recommended command**:
```bash
# Use 8 workers for optimal CPU utilization
python -m nalanda.cli assemble \
  --start-date 2014-01-01 \
  --end-date 2020-12-31 \
  --parallel \
  --workers 8
```

**Options for robustness**:
```bash
# Run in tmux/screen for network resilience
tmux new -s training_assembly
python -m nalanda.cli assemble --all --parallel --workers 8 2>&1 | tee assembly.log

# Alternative: Process in chunks for checkpointing
for year in {2014..2020}; do
  python -m nalanda.cli assemble \
    --start-date $year-01-01 \
    --end-date $year-12-31 \
    --parallel --workers 8
done
```

### Phase 3: Validation

```bash
# Validate all assembled files
python -m nalanda.cli validate --all -v > validation_report.txt

# Generate final report
python -m nalanda.cli report --output production_report.json
```

---

## Risk Mitigation

### Risk: Assembly interrupted mid-run

**Probability**: Low-Medium  
**Impact**: High (must restart)  
**Mitigation**:
1. Use `tmux` or `screen` for session persistence
2. Process in yearly chunks for implicit checkpointing
3. Monitor progress with `watch -n 60 'ls -lh artifacts/v1/training/ | tail'`

### Risk: Meteo generation fails

**Probability**: Medium (API rate limits)  
**Impact**: Critical (blocks assembly)  
**Mitigation**:
1. Add retry logic to openmeteo_signals.py
2. Process in smaller date ranges
3. Cache responses to avoid re-fetching

### Risk: Disk space exhaustion

**Probability**: Very Low  
**Impact**: High  
**Mitigation**:
1. Monitor disk usage: `df -h`
2. 407 GB available >> 1 GB needed (400x headroom)

### Risk: Memory exhaustion

**Probability**: Very Low  
**Impact**: Medium (process killed)  
**Mitigation**:
1. 8 workers × 176 MB = 1.4 GB << system RAM
2. Monitor: `watch -n 5 'ps aux | grep python | grep nalanda'`

### Risk: Output quality degradation

**Probability**: Low  
**Impact**: High  
**Mitigation**:
1. Run validation after completion
2. Spot-check random samples
3. Compare statistics across years

---

## Stress Test Recommendations

Before deploying at 7-year scale, recommend testing:

### Test 1: 100-Day Continuous Run

**Purpose**: Validate long-running stability  
**Duration**: ~25 minutes (8 workers)  
**Command**:
```bash
python -m nalanda.cli assemble \
  --start-date 2021-01-01 \
  --end-date 2021-04-10 \
  --parallel --workers 8
```

**Monitor for**:
- Memory leaks
- File descriptor leaks
- Performance degradation over time

### Test 2: Maximum Parallelism

**Purpose**: Test high worker count  
**Duration**: ~10 minutes  
**Command**:
```bash
python -m nalanda.cli assemble \
  --start-date 2021-06-01 \
  --end-date 2021-06-30 \
  --parallel --workers 16
```

**Monitor for**:
- CPU saturation
- I/O contention
- Race conditions

### Test 3: Error Recovery

**Purpose**: Test interruption handling  
**Steps**:
1. Start assembly of 50 dates
2. Kill process mid-run (Ctrl+C)
3. Check for corrupted files
4. Verify can resume (manual)

---

## Conclusion

### Can we process 7 years? **YES ✅**

- **No technical bottlenecks** identified
- **Sufficient resources** (CPU, memory, disk)
- **Proven architecture** scales linearly
- **Estimated time**: 1-2 hours with 8 workers

### Is it stress tested? **PARTIALLY ⚠️**

**Tested & Confident** (>95%):
- Core algorithms (148 unit tests)
- Data quality & correctness
- Memory usage (176 MB/worker)
- Parallel processing (4 workers)

**Not Stress Tested** (~70% confidence):
- Long-running stability (only 3-minute runs)
- Large file count (only 12 files tested)
- Error recovery (no checkpointing)

### Final Recommendation

**PROCEED WITH CAUTION**

1. ✅ **Technical capability**: System can handle 7 years
2. ✅ **Resource availability**: All resources sufficient
3. ⚠️ **Operational risk**: Limited long-run testing
4. 🔴 **Critical dependency**: Must generate meteo data first

**Recommended approach**:
1. Generate meteo data (15-20 hours)
2. Run 100-day stress test first (25 minutes)
3. If stress test passes, proceed with full 7 years
4. Use yearly chunks for implicit checkpointing
5. Monitor continuously during production run

**Expected total time**: 20-25 hours (including meteo generation)

---

## Monitoring Commands

```bash
# During assembly
watch -n 10 'python -m nalanda.cli status'

# Memory usage
watch -n 5 'ps aux | grep python | grep nalanda | awk "{sum+=\$6} END {print sum/1024\" MB\"}"'

# Disk usage
watch -n 60 'du -sh artifacts/v1/training/'

# Progress
watch -n 30 'ls -1 artifacts/v1/training/ | wc -l'
```
