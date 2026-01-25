# Kashi: Batch Inference Scheduler - Implementation Phases

**Document:** Implementation Roadmap for TECHNICAL_DESIGN.md  
**Status:** Planning  
**Target Date:** Sequential completion Jan 2026  
**Owner:** Engineering Team

---

## **Overview: 3 Sequential Phases**

```
Phase 1: WriterJob (Raw Data Fetching & DuckDB Writing)
    ↓ (Phase 1 must complete before Phase 2 starts)
Phase 2: InferenceJob (Feature Assembly, Inference, Caching)
    ↓ (Phase 2 must complete before Phase 3 starts)
Phase 3: Testing, Logging, Metrics, Documentation & Validation
```

Each phase is **self-contained** but builds on previous phases. Phases **cannot be parallelized** due to dependency ordering (Phase 2 requires working WriterJob, Phase 3 requires working InferenceJob).

---

# **PHASE 1: WriterJob Implementation**

**Duration:** 3-4 days  
**Objective:** Fetch raw data from 4 sources, write to DuckDB with minimal memory footprint  
**Success Criteria:** WriterJob completes in <40 seconds, writes all 4 data sources correctly

## **1.1 Phase 1 Scope**

### **Core Responsibilities**
- ✅ Fetch meteorological data from OpenMeteo API (1-2s)
- ✅ Fetch PM2.5 observations from CPCB API (2-5s)
- ✅ Fetch Landsat8 data from Google Earth Engine (10-15s when available)
- ✅ Fetch Sentinel2 data from Google Earth Engine (15-25s when available)
- ✅ Write all fetched data to DuckDB sequentially (<10s)
- ✅ Handle partial failures gracefully (skip unavailable sources)
- ✅ Structured logging with cycle_id for traceability

### **Out of Scope (Phase 2 or later)**
- ❌ Feature assembly
- ❌ LightGBM inference
- ❌ Cache population
- ❌ Prediction writing
- ❌ Comprehensive metrics/monitoring (basic logging only)

---

## **1.2 Files to Create/Modify**

### **New Files**

| File Path | Purpose | Size | Dependencies |
|-----------|---------|------|--------------|
| `kashi/__init__.py` | Package initialization | 50 lines | None |
| `kashi/__main__.py` | Entry point: `python -m kashi` | 30 lines | orchestrator |
| `kashi/config.py` | Configuration classes | 80 lines | pathlib, yaml |
| `kashi/config.yaml` | Configuration file | 50 lines | (manually created) |
| `kashi/orchestrator.py` | APScheduler wrapper, WriterJob orchestration | 120 lines | subprocess, logging |
| `kashi/writer_job.py` | WriterJob: fetch + write orchestration | 150 lines | pathlib, logging, threading |
| `kashi/fetchers/__init__.py` | Fetchers package | 20 lines | None |
| `kashi/fetchers/meteo_fetcher.py` | OpenMeteo API client | 100 lines | requests, pandas |
| `kashi/fetchers/pm25_fetcher.py` | CPCB/OpenAQ API client | 100 lines | requests, pandas |
| `kashi/fetchers/landsat_fetcher.py` | Google Earth Engine client | 150 lines | geemap, ee, pandas |
| `kashi/fetchers/sentinel_fetcher.py` | Google Earth Engine client | 150 lines | geemap, ee, pandas |
| `kashi/writers/__init__.py` | Writers package | 20 lines | None |
| `kashi/writers/duckdb_writer.py` | DuckDB write operations | 150 lines | duckdb, pandas |
| `kashi/tests/conftest.py` | Pytest fixtures & test helpers | 80 lines | pytest, duckdb |
| `WRITER_JOB_IMPLEMENTATION_LOG.md` | Phase 1 progress tracking | 100 lines | (auto-updated) |

**Total Files:** 14 new files  
**Estimated Code:** ~1,400 lines

### **Modified Files**

| File Path | Changes | Lines |
|-----------|---------|-------|
| `pyproject.toml` | Add kashi dependencies (requests, geemap, ee) | +5 |
| `requirements.txt` | Update dependencies | +10 |

---

## **1.3 Implementation Checklist**

### **Step 1.3.1: Configuration Foundation** (30 min)
- [ ] Create `kashi/config.py` with `KashiConfig` dataclass
  - Database paths
  - API endpoints
  - Timeouts
  - Scheduling parameters
- [ ] Create `kashi/config.yaml` with sensible defaults
- [ ] Create `kashi/__init__.py` package init

### **Step 1.3.2: Fetcher Implementation** (2 days)
- [ ] `kashi/fetchers/meteo_fetcher.py`
  - Calls OpenMeteo API for current meteorology
  - Returns list of dicts: `[{"grid_id": "...", "temperature_2m": ...}, ...]`
  - Handles 500K grid broadcast (same meteo tile = same value for all grids in tile)
  - Timeout: 15s
  - Test: Mock API response, verify 500K rows

- [ ] `kashi/fetchers/pm25_fetcher.py`
  - Calls data.gov.in API (CPCB Air Quality Index endpoint)
  - Returns list of dicts: `[{"latitude": ..., "longitude": ..., "avg_value": ...}, ...]` (use avg_value only, ignore min/max)
  - **Real data:** 26 PM2.5 monitoring stations in Mumbai with coordinates and average PM2.5 values
  - **API:** `https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`
  - **Filters:** `filters%5Bcity%5D=Mumbai&filters%5Bpollutant_id%5D=PM2.5`
  - **Extraction command:** 
    ```bash
    curl -s "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=<KEY>&format=json&filters%5Bcity%5D=Mumbai&filters%5Bpollutant_id%5D=PM2.5&limit=100" | jq '.records[] | {latitude: .latitude, longitude: .longitude, avg_value: .avg_value}'
    ```
  - **Data freshness:** Latest available measurements from CPCB stations (updated throughout day)
  - **Instruction:** Use `avg_value` field only for PM2.5 concentration. Discard min/max fields.
  - Timeout: 15s
  - Test: Verify 26+ stations returned, coordinates valid (latitude/longitude), avg_value present and numeric

- [ ] `kashi/fetchers/landsat_fetcher.py`
  - Calls Google Earth Engine for latest Landsat8 data
  - Available 1/16 days (~every 16 days)
  - If available: returns `[{"grid_id": "...", "B1": ..., "B11": ...}, ...]` for 500K grids
  - If unavailable: returns empty list
  - Batch processing: fetch 500 batches × 1000 cells concurrently via GEE
  - Timeout: 60s
  - Test: Mock GEE response, verify band count

- [ ] `kashi/fetchers/sentinel_fetcher.py`
  - Calls Google Earth Engine for latest Sentinel2 data
  - Available ~4/5 days
  - If available: returns `[{"grid_id": "...", "B2": ..., "QA10": ...}, ...]` for 500K grids
  - If unavailable: returns empty list
  - Batch processing: fetch 500 batches × 1000 cells concurrently via GEE
  - Timeout: 60s
  - Test: Mock GEE response, verify band count

### **Step 1.3.3: DuckDB Writer** (1 day)
- [ ] `kashi/writers/duckdb_writer.py`
  - `write_meteo(data, timestamp)`: Insert 500K meteo rows
  - `write_pm25(data, timestamp)`: Insert ~10 PM2.5 rows
  - `write_landsat8(data, timestamp)`: Insert 500K landsat rows (if available)
  - `write_sentinel2(data, timestamp)`: Insert 500K sentinel rows (if available)
  - Batch inserts for performance
  - Test: Verify row count, column values

### **Step 1.3.4: WriterJob Orchestration** (1 day)
- [ ] `kashi/writer_job.py`
  - `WriterJob` class: orchestrates fetch + write
  - `_compute_write_timestamp()`: Round to nearest hour
  - `run()`: Fetch all 4 sources in parallel (ThreadPoolExecutor), write sequentially
  - Error handling: Retry fetch once, skip on timeout, write errors cause job failure
  - Logging: Detailed logs with cycle_id
  - Test: Mock all fetchers, verify write calls

### **Step 1.3.5: Orchestrator (Scheduler)** (1 day)
- [ ] `kashi/orchestrator.py`
  - `InferenceOrchestrator` class: APScheduler wrapper
  - `run_daily_cycle()`: Execute WriterJob, log results
  - `_run_writer_job()`: Spawn subprocess, handle timeout/failure
  - Graceful error handling (WriterJob failure doesn't crash coordinator)
  - Logging: All events logged with cycle_id
  - Test: Mock subprocess, verify scheduling

- [ ] `kashi/__main__.py`
  - Entry point: `python -m kashi`
  - Initialize config, create orchestrator, start scheduler
  - Handle SIGTERM for graceful shutdown

### **Step 1.3.6: Testing Setup** (1 day)
- [ ] `kashi/tests/__init__.py`
- [ ] `kashi/tests/conftest.py`
  - Pytest fixtures for test DuckDB, mock API responses
  - Helper functions for data generation
- [ ] `kashi/tests/test_meteo_fetcher.py`
  - Test OpenMeteo API call structure
  - Test grid broadcast logic
  - Test timeout handling
- [ ] `kashi/tests/test_pm25_fetcher.py`
  - Test CPCB API integration
  - Test timestamp handling (t=-60m)
- [ ] `kashi/tests/test_landsat_fetcher.py`
  - Test availability detection
  - Test batch processing
- [ ] `kashi/tests/test_sentinel_fetcher.py`
  - Test availability detection
  - Test batch processing
- [ ] `kashi/tests/test_duckdb_writer.py`
  - Test insertion of 500K rows
  - Test batch handling
- [ ] `kashi/tests/test_writer_job.py`
  - End-to-end: fetch → write
  - Test partial failures (skip sources)
  - Test retry logic

---

## **1.4 Phase 1 Success Criteria**

| Criterion | Target | Method |
|-----------|--------|--------|
| **WriterJob Latency** | <40s | Time multiple runs with all 4 sources available |
| **Meteo Fetch** | <3s | Time OpenMeteo API call |
| **PM25 Fetch** | <5s | Time CPCB API call |
| **Landsat Fetch** | <15s (when available) | Time GEE call for 500K cells |
| **Sentinel Fetch** | <25s (when available) | Time GEE call for 500K cells |
| **DuckDB Write** | <10s | Time 4 sequential inserts |
| **Unit Test Pass Rate** | 100% | `pytest kashi/tests/test_*.py -v` |
| **Logging Quality** | Detailed | Every fetch/write logged with cycle_id |
| **Error Handling** | Graceful | Partial source failures don't crash job |
| **Configuration** | Complete | All API keys from env vars, paths correct |

---

## **1.5 Phase 1 Dependencies**

### **External Libraries (already available or need install)**
```
requests                    # HTTP calls to OpenMeteo, data.gov.in CPCB API
duckdb                     # Database writes
geemap                     # Google Earth Engine wrapper
ee                         # Google Earth Engine API
pandas                     # Data manipulation
APScheduler                # Job scheduling
python-dotenv              # Environment variables
```

### **Existing Project Code**
- None (Phase 1 is standalone)

### **Prerequisites**
- Google Earth Engine API credentials (`.env` file)
- data.gov.in API key (`.env` file) for CPCB PM2.5 data
- DuckDB database initialized with table schemas

---

## **1.6 Phase 1 Risk Mitigations**

| Risk | Mitigation |
|------|-----------|
| GEE API quota exhaustion | Batch requests carefully, cache availability checks |
| OpenMeteo/CPCB API downtime | Retry with exponential backoff, graceful failure |
| DuckDB write conflicts | Use `READ_WRITE` isolation, handle lock errors |
| Large memory for 500K rows | Stream inserts in batches (not all at once) |
| Network timeouts | Set conservative timeouts (15-60s), retry once |

---

---

# **PHASE 2: InferenceJob Implementation**

**Duration:** 3-4 days (after Phase 1 complete)  
**Objective:** Load features, run LightGBM, populate cache, write predictions  
**Success Criteria:** InferenceJob completes in <90 seconds total, all 500K grids cached

## **2.1 Phase 2 Scope**

### **Core Responsibilities**
- ✅ Load [500K × 64] feature matrix from DuckDB (40-45s)
  - Depends on WriterJob-populated meteo, PM25, landsat, sentinel tables
  - Assemble features for all grids in single batch
  - Handle missing features gracefully (NaN)
- ✅ Run LightGBM.predict() on vectorized features (10-15s)
- ✅ Populate in-memory cache in 3 stages:
  - Stage 1 (sync): Hot grids 0-5K (~5s)
  - Stage 2 (async): Warm grids 5K-50K (~30s background)
  - Stage 3 (async): Cold grids 50K-500K (~20s background)
- ✅ Write predictions to DuckDB asynchronously (non-blocking)
- ✅ Return cache readiness status (hot ready? warm ready? full ready?)
- ✅ Detailed logging and timing

### **Out of Scope (Phase 3)**
- ❌ Comprehensive metrics/monitoring (basic timing only)
- ❌ Full test suite (basic smoke tests only)
- ❌ Production logging configuration
- ❌ Graceful error recovery with retries

---

## **2.2 Files to Create/Modify**

### **New Files**

| File Path | Purpose | Size | Dependencies |
|-----------|---------|------|--------------|
| `kashi/inference_job.py` | InferenceJob: feature load → infer → cache | 200 lines | lightgbm, numpy, duckdb |
| `kashi/batch_feature_assembler.py` | Batch load features for all 500K grids | 200 lines | duckdb, pandas, numpy |
| `kashi/cache.py` | Cache interface (for Coordinator integration) | 50 lines | None |
| `kashi/tests/test_inference_job.py` | InferenceJob smoke tests | 100 lines | pytest, mock |
| `kashi/tests/test_batch_assembler.py` | Feature assembler tests | 100 lines | pytest |
| `INFERENCE_JOB_IMPLEMENTATION_LOG.md` | Phase 2 progress tracking | 100 lines | (auto-updated) |

**Total Files:** 6 new files  
**Estimated Code:** ~750 lines

### **Modified Files**

| File Path | Changes | Lines |
|-----------|---------|-------|
| `kashi/orchestrator.py` | Add `_run_inference_job()` method | +50 |
| `kashi/config.py` | Add inference-specific config (cache sizes, timeouts) | +20 |
| `kashi/config.yaml` | Add inference config section | +15 |

---

## **2.3 Implementation Checklist**

### **Step 2.3.1: Batch Feature Assembler** (1.5 days)
- [ ] Create `kashi/batch_feature_assembler.py`
  - `BatchFeatureAssembler` class
  - Constructor: Load grid IDs, feature schema
  - `assemble_all(timestamp_hour)`: Return [500K × 64] NumPy array
    - For each grid: fetch from 5 DuckDB tables + compute 64 features
    - Reuse existing `takshashila.core.feature_assembler.FeatureAssembler` for single-grid logic
    - Handle NaN gracefully (acceptable up to 1000 failed grids)
    - Memory management: Process in batches to avoid OOM
  - Timing: Should complete in 40-45s
  - Test: Verify shape [500K, 64], check for NaN handling

### **Step 2.3.2: LightGBM Model Loading** (0.5 day)
- [ ] Implement model loading in `kashi/inference_job.py`
  - Load pre-trained model from `dwarka/model.pkl`
  - Verify model input shape matches feature count (64)
  - Test model prediction on dummy features

### **Step 2.3.3: InferenceJob Orchestration** (2 days)
- [ ] Create `kashi/inference_job.py`
  - `InferenceJob` class
  - Constructor: config, cache provider, cycle_id
  - `run()`: orchestrate feature load → infer → cache
  - `_compute_timestamp_hour()`: Current hour (UTC)
  - `_load_model()`: Load LightGBM model
  - `_assemble_features_all()`: Use BatchFeatureAssembler
  - `_run_inference()`: Call model.predict()
  - `_populate_cache_staged()`: 3-stage cache population
    - Stage 1 (sync): Hot grids 0-5K
      - Direct inserts to cache provider
      - Blocks until complete (~5s)
    - Stage 2 (async): Warm grids 5K-50K
      - Spawn background thread
      - Non-blocking return
    - Stage 3 (async): Cold grids 50K-500K
      - Spawn background thread (low priority)
      - Non-blocking return
  - `_schedule_prediction_writes()`: Async DuckDB writes
    - Spawn background thread
    - Batch inserts (100 rows at a time)
    - Non-blocking return
  - Timing: Return from run() at t+60s (before all stages complete)
  - Test: Mock feature assembler, cache, verify thread spawning

### **Step 2.3.4: Cache Interface** (0.5 day)
- [ ] Create `kashi/cache.py`
  - `CacheProvider` abstract base class
  - Methods: `set(grid_id, prediction, timestamp)`
  - Methods: `get(grid_id) → Optional[Dict]`
  - Methods: `get_batch(grid_ids) → Dict`
  - (Concrete implementation in Coordinator, not in Kashi)

### **Step 2.3.5: Orchestrator Integration** (0.5 day)
- [ ] Modify `kashi/orchestrator.py`
  - Add `_run_inference_job()` method (similar to `_run_writer_job()`)
  - Call in `run_daily_cycle()` after WriterJob succeeds
  - Handle subprocess timeout (5 minutes max)
  - Log success/failure with cycle_id
  - Retry immediately (up to 3 times on failure)

### **Step 2.3.6: Configuration Updates** (0.5 day)
- [ ] Modify `kashi/config.py`
  - Add inference-specific config:
    - Model path
    - Feature schema path
    - Cache sizes (hot: 5000, warm: 45000, cold: 450000)
    - Model version, schema hash
- [ ] Modify `kashi/config.yaml`
  - Add inference config section

### **Step 2.3.7: Testing** (1 day)
- [ ] Create `kashi/tests/test_batch_assembler.py`
  - Test shape [500K, 64]
  - Test feature values (non-NaN where expected)
  - Test NaN handling
  - Test performance (should complete <60s)
- [ ] Create `kashi/tests/test_inference_job.py`
  - Mock feature assembler, cache, model
  - Test full InferenceJob.run()
  - Verify cache stages triggered
  - Verify async writes spawned

---

## **2.4 Phase 2 Success Criteria**

| Criterion | Target | Method |
|-----------|--------|--------|
| **Feature Assembly** | <45s | Time BatchFeatureAssembler.assemble_all() |
| **LightGBM Inference** | <15s | Time model.predict() |
| **Cache Stage 1 (Hot)** | <10s | Time synchronous population |
| **Cache Stage 2 (Warm)** | <30s | Time async thread completion |
| **Cache Stage 3 (Cold)** | <20s | Time async thread completion |
| **Total InferenceJob** | <90s | End-to-end run() duration |
| **Cache Readiness** | Hot at t+5s | /health endpoint reports hot cache ready |
| **Prediction Writes** | Async non-blocking | DuckDB writes don't block run() return |
| **Unit Test Pass Rate** | 100% | `pytest kashi/tests/test_inference_job.py -v` |
| **Feature Shape** | [500K, 64] | Verify ndarray dimensions |
| **NaN Handling** | <1000 NaN rows | Acceptable failure threshold |

---

## **2.5 Phase 2 Dependencies**

### **Internal Dependencies (from Kashi Phase 1)**
- `kashi.config.KashiConfig`
- `kashi.orchestrator.InferenceOrchestrator`
- Working WriterJob (must complete before InferenceJob starts)

### **External Dependencies (existing Pollu code)**
- `takshashila.core.feature_assembler.FeatureAssembler`
- `dwarka.model.py` (pre-trained LightGBM model)
- `takshashila.core.inference.InferenceEngine` (reference for feature patterns)

### **External Libraries**
- lightgbm
- numpy
- pandas
- duckdb

### **Prerequisites**
- Phase 1 (WriterJob) must be complete and tested
- DuckDB tables populated with test data (meteo, pm25, landsat, sentinel)
- Pre-trained LightGBM model saved at `dwarka/model.pkl`
- Grid reference table with all 500K grid IDs

---

## **2.6 Phase 2 Risk Mitigations**

| Risk | Mitigation |
|------|-----------|
| Feature assembly OOM | Process in batches, monitor memory usage |
| Missing grid/feature data | NaN handling + threshold (allow <1% failures) |
| LightGBM model mismatch | Verify feature count (64) before inference |
| Cache write failures | Log errors, continue (predictions still written to DB) |
| Async thread deaths | Retry with exponential backoff, notify owner |
| Timestamp mismatches | Centralize timestamp computation in orchestrator |

---

---

# **PHASE 3: Testing, Logging, Metrics, Documentation & Validation**

**Duration:** 2-3 days (after Phase 2 complete)  
**Objective:** Comprehensive testing, observability, error recovery, production readiness  
**Success Criteria:** 100% test pass rate, <40s WriterJob, <90s InferenceJob, full production logging

## **3.1 Phase 3 Scope**

### **Core Responsibilities**
- ✅ Comprehensive unit tests for all components
- ✅ End-to-end integration tests (WriterJob → InferenceJob)
- ✅ Load tests (simulate concurrent batch operations)
- ✅ Prometheus metrics & instrumentation
- ✅ Structured logging (JSON format, all events)
- ✅ Error recovery & retry logic
- ✅ Production deployment validation
- ✅ Final documentation & success criteria validation

### **Out of Scope**
- ❌ Kubernetes deployment (handled by Coordinator)
- ❌ Multi-region failover (single-region MVP)
- ❌ Model retraining pipeline (separate project)

---

## **3.2 Files to Create/Modify**

### **New Files**

| File Path | Purpose | Size | Dependencies |
|-----------|---------|------|--------------|
| `kashi/monitoring.py` | Prometheus metrics & instrumentation | 150 lines | prometheus_client |
| `kashi/tests/test_orchestrator.py` | Orchestrator tests | 100 lines | pytest, mock |
| `kashi/tests/test_writer_job_integration.py` | End-to-end WriterJob tests | 150 lines | pytest |
| `kashi/tests/test_inference_job_integration.py` | End-to-end InferenceJob tests | 150 lines | pytest |
| `kashi/tests/test_full_cycle.py` | Full daily cycle tests | 200 lines | pytest |
| `kashi/tests/test_error_recovery.py` | Retry logic & error handling tests | 150 lines | pytest, mock |
| `kashi/load_test.py` | Load testing suite | 300 lines | concurrent.futures, statistics |
| `LOAD_TEST_RESULTS.md` | Load test output & analysis | 200 lines | (auto-generated) |
| `KASHI_TESTING_REPORT.md` | Comprehensive test results | 200 lines | (auto-generated) |
| `KASHI_DEPLOYMENT_GUIDE.md` | Production deployment steps | 150 lines | (manually created) |
| `KASHI_OPERATIONS_GUIDE.md` | Operational runbook (troubleshooting) | 150 lines | (manually created) |

**Total Files:** 11 new files  
**Estimated Code:** ~1,800 lines

### **Modified Files**

| File Path | Changes | Lines |
|-----------|---------|-------|
| `kashi/orchestrator.py` | Add retry logic, exponential backoff, detailed logging | +100 |
| `kashi/writer_job.py` | Add retry logic, structured logging, error recovery | +80 |
| `kashi/inference_job.py` | Add retry logic, structured logging, graceful failure | +80 |
| `kashi/config.py` | Add logging config, metrics config, retry params | +40 |
| `kashi/config.yaml` | Add logging level, retry delays, metric export settings | +30 |
| `kashi/__main__.py` | Add signal handlers (SIGTERM), graceful shutdown | +30 |

---

## **3.3 Implementation Checklist**

### **Step 3.3.1: Monitoring & Metrics** (1 day)
- [ ] Create `kashi/monitoring.py`
  - Prometheus metrics:
    - `kashi_writer_job_duration_seconds` (histogram)
    - `kashi_inference_job_duration_seconds` (histogram)
    - `kashi_feature_assembly_duration_seconds` (histogram)
    - `kashi_predictions_cached_total` (counter)
    - `kashi_predictions_written_total` (counter)
    - `kashi_job_failures_total` (counter, label: job_type)
    - `kashi_cache_population_time_seconds` (histogram, label: stage)
    - `kashi_grid_failures_total` (counter)
  - Helper decorators for timing
  - Prometheus exporter endpoint for Coordinator to scrape
  - Test: Verify metric emission

### **Step 3.3.2: Structured Logging** (1 day)
- [ ] Enhance `kashi/orchestrator.py`
  - Add structured logging (JSON format)
  - Log every event with cycle_id, timestamp, status
  - Sample logs:
    ```
    {"timestamp": "2026-01-24T14:00:00Z", "cycle_id": "abc123", "event": "cycle_started", "job_type": "writer"}
    {"timestamp": "2026-01-24T14:02:00Z", "cycle_id": "abc123", "event": "fetch_complete", "source": "meteo", "row_count": 500000, "latency_ms": 1500}
    {"timestamp": "2026-01-24T14:04:00Z", "cycle_id": "abc123", "event": "write_complete", "total_rows": 1010000, "latency_ms": 8000}
    ```

- [ ] Enhance `kashi/writer_job.py`
  - Add cycle_id to all log messages
  - Log fetch start/end for each source
  - Log write start/end for each table
  - Log errors with full context

- [ ] Enhance `kashi/inference_job.py`
  - Log feature assembly progress (every 50K grids)
  - Log cache stage completion
  - Log async thread status
  - Log prediction write status

- [ ] Enhance `kashi/config.py`
  - Add logging configuration
  - JSON formatter option
  - File & console handlers

### **Step 3.3.3: Error Recovery & Retry Logic** (1 day)
- [ ] Modify `kashi/orchestrator.py`
  - Implement exponential backoff for WriterJob failures
  - Retry WriterJob up to 3 times (100ms, 500ms, 1s delays)
  - Implement immediate retry for InferenceJob failures (up to 3 times)
  - Log all retry attempts
  - Track failure reasons

- [ ] Modify `kashi/writer_job.py`
  - Per-fetcher retry logic (each source retries independently)
  - Exponential backoff: 500ms, 1s, 2s
  - Max 2 retries per fetcher
  - Skip source if all retries fail

- [ ] Modify `kashi/inference_job.py`
  - Feature assembly NaN threshold (allow up to 1000)
  - Cache write error handling (non-blocking)
  - Prediction write error handling (non-blocking)
  - Log all errors with context

### **Step 3.3.4: Signal Handling & Graceful Shutdown** (0.5 day)
- [ ] Modify `kashi/__main__.py`
  - Add SIGTERM signal handler
  - Graceful shutdown: finish current cycle before exit
  - Log shutdown event

### **Step 3.3.5: Unit Tests - Complete Suite** (1.5 days)
- [ ] Enhance `kashi/tests/test_meteo_fetcher.py`
  - Test successful fetch
  - Test timeout handling
  - Test retry logic
  - Test malformed responses
  - Test grid count verification

- [ ] Enhance `kashi/tests/test_pm25_fetcher.py`
  - Test API call structure
  - Test sensor data parsing
  - Test empty response handling

- [ ] Enhance `kashi/tests/test_landsat_fetcher.py`
  - Test availability detection
  - Test batch processing
  - Test timeout handling
  - Test empty response (unavailable data)

- [ ] Enhance `kashi/tests/test_sentinel_fetcher.py`
  - Test availability detection
  - Test batch processing
  - Test timeout handling

- [ ] Enhance `kashi/tests/test_duckdb_writer.py`
  - Test batch inserts
  - Test large dataset handling (500K rows)
  - Test write failures (disk full, permissions)
  - Test concurrent writes (should serialize)

- [ ] Enhance `kashi/tests/test_writer_job.py`
  - Test full fetch + write cycle
  - Test partial failures (skip some sources)
  - Test all sources available
  - Test no sources available (skip entire write)

- [ ] Create `kashi/tests/test_batch_assembler.py`
  - Test feature shape [500K, 64]
  - Test NaN handling (<1000 allowed)
  - Test missing data scenario
  - Test performance (<60s)

- [ ] Create `kashi/tests/test_inference_job.py`
  - Test full run() execution
  - Test cache stage triggering
  - Test async thread spawning
  - Test prediction write scheduling

- [ ] Create `kashi/tests/test_orchestrator.py`
  - Test scheduler initialization
  - Test daily cycle execution
  - Test subprocess management
  - Test error handling

- [ ] Create `kashi/tests/test_error_recovery.py`
  - Test WriterJob retry logic
  - Test InferenceJob retry logic
  - Test exponential backoff
  - Test max retries exceeded

### **Step 3.3.6: Integration Tests** (1 day)
- [ ] Create `kashi/tests/test_writer_job_integration.py`
  - Full end-to-end: Orchestrator → WriterJob → DuckDB
  - Verify data written to all 4 tables
  - Verify timestamps correct
  - Verify error recovery works

- [ ] Create `kashi/tests/test_inference_job_integration.py`
  - Full end-to-end: Orchestrator → InferenceJob → Cache
  - Verify features loaded correctly
  - Verify predictions made
  - Verify cache populated
  - Verify async writes scheduled

- [ ] Create `kashi/tests/test_full_cycle.py`
  - Full daily cycle: WriterJob → InferenceJob → Cache
  - Simulate 500 grids (not full 500K to keep test time reasonable)
  - Verify all data flows correctly
  - Verify timing expectations

### **Step 3.3.7: Load Tests** (1 day)
- [ ] Create `kashi/load_test.py`
  - Simulate realistic WriterJob loads:
    - Single daily run (baseline)
    - API concurrent requests (3 concurrent fetchers)
    - Large data handling (500K rows to DuckDB)
  - Simulate realistic InferenceJob loads:
    - Feature assembly for 500K grids
    - LightGBM batch inference
    - Cache population in 3 stages
  - Measure:
    - Total duration
    - Memory usage
    - CPU usage
    - Latency percentiles (P50, P95, P99)
  - Output: Summary statistics, pass/fail
  - Test: Load test should complete with <40s WriterJob, <90s InferenceJob

### **Step 3.3.8: Documentation** (1 day)
- [ ] Create `KASHI_DEPLOYMENT_GUIDE.md`
  - Prerequisites (GEE credentials, API keys, DuckDB setup)
  - Installation steps
  - Configuration walkthrough
  - Startup verification
  - Health checks

- [ ] Create `KASHI_OPERATIONS_GUIDE.md`
  - Daily operations checklist
  - Troubleshooting guide (common failures)
  - Log interpretation
  - Metric alerts & thresholds
  - Manual recovery procedures

- [ ] Create `KASHI_TESTING_REPORT.md`
  - Test suite summary
  - Pass/fail statistics
  - Coverage analysis
  - Known limitations

- [ ] Create `LOAD_TEST_RESULTS.md`
  - Load test execution summary
  - Latency percentiles
  - Memory/CPU usage
  - Bottleneck analysis
  - Recommendations

- [ ] Update `TECHNICAL_DESIGN.md`
  - Add references to implementation documents
  - Add final success criteria validation

---

## **3.4 Phase 3 Success Criteria**

### **Testing**
| Criterion | Target | Method |
|-----------|--------|--------|
| **Unit Test Pass Rate** | 100% | `pytest kashi/tests/ -v` |
| **Test Coverage** | >90% | Coverage report |
| **Integration Tests** | All pass | Full cycle tests |
| **Load Test Duration** | <40s (Writer), <90s (Inference) | Load test execution |

### **Observability**
| Criterion | Target | Method |
|-----------|--------|--------|
| **Logging Completeness** | All events logged | Grep log file for key events |
| **Metrics Emission** | All metrics available | Prometheus scrape endpoint |
| **Error Traceability** | Every error has cycle_id | Log analysis |

### **Performance**
| Criterion | Target | Method |
|-----------|--------|--------|
| **WriterJob Latency** | <40s | Time from start to exit |
| **InferenceJob Latency** | <90s | Time from start to exit |
| **Hot Cache Ready** | <10s after inference start | Check cache readiness status |
| **Memory Usage** | <2GB peak | Monitor during load test |

### **Resilience**
| Criterion | Target | Method |
|-----------|--------|--------|
| **Retry Logic** | Works as designed | Failure injection tests |
| **Graceful Degradation** | Partial failures don't crash | Skip unavailable sources |
| **Error Recovery** | 3 retries with backoff | Verify exponential delays |

### **Documentation**
| Criterion | Target | Method |
|-----------|--------|--------|
| **Deployment Guide** | Complete & accurate | Follow guide end-to-end |
| **Operations Guide** | Clear troubleshooting | All common failures covered |
| **Test Report** | Comprehensive | All test results documented |

---

## **3.5 Phase 3 Dependencies**

### **Internal Dependencies (from Phases 1 & 2)**
- `kashi.orchestrator.InferenceOrchestrator`
- `kashi.writer_job.WriterJob`
- `kashi.inference_job.InferenceJob`
- All Phase 1 & 2 code must be complete and working

### **External Libraries**
- prometheus_client (metrics)
- pytest (testing)
- pytest-cov (coverage)

### **Prerequisites**
- Phase 1 and Phase 2 complete and tested
- Full system integration tested
- All credentials/configs in place

---

## **3.6 Phase 3 Risk Mitigations**

| Risk | Mitigation |
|------|-----------|
| Test flakiness | Mock external APIs, use deterministic test data |
| Load test failures | Scale down to 100K grids for testing, extrapolate results |
| Metric cardinality explosion | Limit labels, aggregate carefully |
| Logging volume | Use log levels (DEBUG, INFO, WARNING) to control verbosity |
| Documentation drift | Auto-generate docs from code, version control documentation |

---

---

# **Cross-Phase Dependencies & Constraints**

## **Dependency Graph**

```
Phase 1: WriterJob
  ├─ Config foundation
  ├─ Orchestrator skeleton
  ├─ 4 Fetchers (parallel)
  ├─ DuckDB writer
  └─ WriterJob orchestration
         ↓ (Must complete)
Phase 2: InferenceJob
  ├─ Feature assembler
  ├─ LightGBM inference
  ├─ Cache population (3 stages)
  ├─ Async prediction writes
  └─ InferenceJob orchestration
         ↓ (Must complete)
Phase 3: Testing, Logging, Metrics
  ├─ Comprehensive test suite
  ├─ Error recovery & retries
  ├─ Prometheus metrics
  ├─ Structured logging
  └─ Production documentation
```

## **Cannot Parallelize**
- Phase 1 → Phase 2 (InferenceJob depends on WriterJob-populated tables)
- Phase 2 → Phase 3 (Testing requires complete implementation)

## **Can Parallelize Within Each Phase**
- **Phase 1:** All 4 fetchers can be implemented in parallel
- **Phase 2:** Feature assembler and LightGBM loading can be done in parallel
- **Phase 3:** Testing, metrics, logging can be done in parallel (in separate PRs)

---

# **Timeline Estimate**

| Phase | Duration | Start | End | Blockers |
|-------|----------|-------|-----|----------|
| **Phase 1** | 3-4 days | Day 1 | Day 4 | GEE credentials |
| **Phase 2** | 3-4 days | Day 5 | Day 8 | Phase 1 complete |
| **Phase 3** | 2-3 days | Day 9 | Day 11 | Phase 2 complete |
| **Total** | **8-11 days** | | | |

---

# **Success Validation Checklist**

## **Phase 1 Validation** ✓
- [ ] WriterJob runs end-to-end (WriterJob → DuckDB)
- [ ] All 4 data sources written correctly
- [ ] Timing <40 seconds
- [ ] Unit tests 100% pass
- [ ] Error handling works (skip unavailable sources)

## **Phase 2 Validation** ✓
- [ ] InferenceJob runs end-to-end (Features → Inference → Cache)
- [ ] 500K grids cached in 3 stages
- [ ] Timing <90 seconds
- [ ] Unit tests 100% pass
- [ ] Async writes non-blocking

## **Phase 3 Validation** ✓
- [ ] Full daily cycle runs (WriterJob → InferenceJob)
- [ ] All tests pass (unit + integration + load)
- [ ] Metrics available at Prometheus endpoint
- [ ] Logs are structured (JSON) and complete
- [ ] Error recovery works (retries, exponential backoff)
- [ ] Documentation complete & accurate
- [ ] Production deployment guide verified end-to-end

---

**Document Status:** Complete Implementation Phases  
**Last Updated:** 2026-01-24  
**Next Action:** Begin Phase 1 implementation (WriterJob)
