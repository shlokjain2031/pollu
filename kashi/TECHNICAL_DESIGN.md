# Kashi: Batch Inference Scheduler - Technical Design Document

**Status:** Ready for Implementation (Phase 2a)  
**Language:** Python 3.10+  
**Dependencies:** APScheduler, DuckDB, LightGBM, NumPy, Pandas  
**Integration:** Subprocess of Go Coordinator  

---

## **1. OVERVIEW**

Kashi is a Python-based batch inference scheduler that runs inference jobs hourly to populate a prediction cache for 500K grid cells. The system comprises two sequential subprocess jobs:

1. **WriterJob** (t=-2m): Fetch fresh raw data from 4 sources, write to DuckDB (**20-40 seconds typical**)
2. **InferenceJob** (t=0m): Assemble features, run LightGBM, populate cache, write predictions (**60 seconds total**)

**Design Principles:**
- Minimal memory footprint for raw data (reserved for cache only)
- Single batch inference call (vectorized, not streaming)
- Graceful degradation (serve stale cache on failures)
- Process-based isolation (WriterJob & InferenceJob are independent subprocesses)

---

## **2. ARCHITECTURE**

### **2.1 Process Hierarchy**

```
Coordinator (Go)
├── /predict handler (uses shared cache)
├── /health handler
└── Kashi subprocess (Python)
    ├── Orchestrator (APScheduler)
    ├── WriterJob subprocess
    │   ├── MeteoFetcher (OpenMeteo API)
    │   ├── PM25Fetcher (CPCB API)
    │   ├── LandsatFetcher (GEE API)
    │   ├── SentinelFetcher (GEE API)
    │   └── DuckDBWriter (sequential, minimal memory)
    │
    └── InferenceJob subprocess
        ├── FeatureAssembler (batch all 500K at once)
        ├── LightGBMInference (vectorized predict)
        ├── CachePopulator (staged: hot → warm → cold)
        └── PredictionWriter (async DuckDB writes)
```

### **2.2 Job Lifecycle**

```
[Kashi starts]
  ↓
[APScheduler initializes]
  ↓
[Every hour at fixed time]:
  t=-2m:00s (e.g., 13:58 UTC):
  1. WriterJob spawns (subprocess)
     - Fetches from 4 APIs in parallel (OpenMeteo, CPCB, GEE-Landsat, GEE-Sentinel)
     - Duration: 20-40 seconds typical
     - Components:
       * OpenMeteo: 1-2s (current meteorology, 3-6 tiles)
       * CPCB: 2-5s (hourly PM2.5 observations from 5-10 sensors)
       * Landsat-8: 0s or 10-15s (available 1/16 days, batch 500K cells via GEE)
       * Sentinel-2: 0s or 15-25s (available 4/5 days, batch 500K cells via GEE)
     - All run concurrently → bottleneck is slowest (usually Sentinel-2 when available)
  2. Writer writes to DuckDB (sequential)
     - Duration: 4-10 seconds
     - meteo.duckdb: 2-3s for 500K rows
     - pm25_observation.duckdb: <1s for ~10 rows
     - landsat8.duckdb: 2-3s if available
     - sentinel2.duckdb: 2-3s if available
  3. Writer subprocess exits
  ↓
[t=-1m:00s (e.g., 13:59 UTC): WriterJob completes, guaranteed before InferenceJob starts]
  ↓
[t=0m:00s (e.g., 14:00 UTC):
  4. InferenceJob spawns (subprocess)
  5. Loads [500K × 64] features from DuckDB (~40-45s)
  6. Runs LightGBM.predict([500K × 64]) (~10-15s)
  7. Populates cache (staged: hot 5K → warm 45K → cold 450K)
  8. Async writes predictions to DuckDB (background)
  9. InferenceJob subprocess exits
  ↓
[t=1m:05s: Hot cache (5K) queryable]
[t=1m:35s: Warm cache (50K) queryable]
[t=1m:55s: All 500K predictions cached]

[If InferenceJob fails]:
  10. Retry immediately (max 3 attempts)
  11. If retries exhausted, serve stale cache with metadata
  ↓
[Sleep until next scheduled time (t=2h:00s)]
```

**Key Timing Guarantees:**
- WriterJob starts at t=-2m, finishes by t=-1m (1 minute window)
- InferenceJob starts at t=0m (guaranteed WriterJob done)
- Hot cache ready by t=1m:05s
- Full cache ready by t=1m:55s

---

## **2.5 WriterJob Timing Analysis**

**See:** [WRITER_JOB_TIMING_ANALYSIS.md](WRITER_JOB_TIMING_ANALYSIS.md) for detailed breakdown.

**Summary:**

| Component | Typical | 95th %ile | Notes |
|-----------|---------|-----------|-------|
| OpenMeteo | 1-2s | 3s | 3-6 tiles, parallel fetch |
| CPCB PM25 | 2-5s | 5s | 5-10 sensors, hourly data |
| Landsat-8 | 0s (unavailable) | 15s | 1/16 days, batch 500K via GEE |
| Sentinel-2 | 0s or 15-25s | 30s | 4/5 days available, batch 500K |
| DuckDB Write | 4-10s | 10s | Sequential, 4-5 tables |
| **Total** | **20-35s** | **<40s** | Median ~20s, safe window 2 min |

**Scheduling Decision:**
- WriterJob starts at t=-2m (2 minutes before hour)
- 95% of runs finish by t=-1m:30s (30s margin)
- InferenceJob guaranteed to start clean at t=0m



### **3.1 Writer Job Data Flow**

```
Timestamp t=-2m:00s (e.g., 13:58 UTC)
↓
[Parallel API Fetches - All 4 run concurrently]
├─ MeteoFetcher (target: 1-2s)
│  └─ OpenMeteo API → Get hourly meteo for "now" (13:58)
│     (3-6 unique tiles for Mumbai, due to ~9km native resolution)
│     Response: 500K values (one value per tile, broadcast to all grids in tile)
│
├─ PM25Fetcher (target: 2-5s)
│  └─ CPCB API or CSV → Get PM2.5 from 5-10 sensors
│     Data is 1 hour old (t=-60m, i.e., 12:58)
│     Response: 5-10 observations (one per sensor)
│
├─ LandsatFetcher (target: 10-15s when available, 0s otherwise)
│  └─ Google Earth Engine API → Get latest Landsat8 for Mumbai
│     Frequency: New data every ~16 days
│     If available: Batch fetch 500 batches × 1000 cells concurrently
│     Response: 500K rows × 11 bands (B1-B11)
│     If not available: Skip (0 seconds)
│
└─ SentinelFetcher (target: 15-25s when available, 0s otherwise)
   └─ Google Earth Engine API → Get latest Sentinel2 for Mumbai
      Frequency: New data every ~5 days (more frequent than Landsat)
      If available: Batch fetch 500 batches × 1000 cells concurrently
      Response: 500K rows × 5 bands (B2, B4, B8, AOT, QA10)
      If not available: Skip (0 seconds)

[Parallel execution bottleneck]
Typical case (80% of days, no satellites): max(1-2s, 2-5s, 0s, 0s) = 5 seconds
With Sentinel2 (60% probability): max(1-2s, 2-5s, 0s, 15-25s) = 20 seconds
With both satellites (rare): max(1-2s, 2-5s, 10-15s, 15-25s) = 25 seconds

Weighted average fetch time: ~15-20 seconds
↓
[Sequential DuckDB Writes - All writes complete <10 seconds total]
INSERT INTO meteo.duckdb (grid_id, timestamp, temperature_2m, ...) → ~2-3s
INSERT INTO pm25_observation.duckdb (site_id, timestamp, pm25) → <1s
INSERT INTO landsat8.duckdb (grid_id, timestamp, B1-B11) → ~2-3s (if available)
INSERT INTO sentinel2.duckdb (grid_id, timestamp, B2, B4, B8, AOT, QA10) → ~2-3s (if available)
↓
[WriterJob exits with code 0 if successful, non-zero on error]

Total WriterJob Duration:
- No satellites (80% of days): 20-30 seconds
- With Sentinel2 (60% of days): 25-35 seconds
- Weighted average: 15-20 seconds median, <40 seconds 95th percentile
```

### **3.2 Inference Job Data Flow**

```
Timestamp t=0m (e.g., 14:00 UTC), after WriterJob succeeds
↓
[Feature Assembly - Batch All 500K]
BatchFeatureAssembler.assemble_all(timestamp_hour=t=0m)
  → Loads [500K] grid IDs from grid table (cached in memory)
  → For each grid: fetch features from 5 tables + compute 64 derived features
  → Sources: meteo.duckdb (updated this hour), pm25_observation.duckdb, 
             landsat8.duckdb (updated every 16 days), sentinel2.duckdb (updated every 5 days),
             grid.duckdb (static reference)
  → Returns [500K × 64] NumPy array (~256MB)
  → Failed grids get NaN features (acceptable up to 1000 NaN rows)
  → Duration: ~40-45 seconds
↓
[LightGBM Inference - Vectorized]
model.predict([500K × 64]) → [500K] predictions
  → Single batch inference call (not streaming)
  → Duration: ~10-15 seconds
↓
[Staged Cache Population]
PHASE 1 (Synchronous - blocks until complete)
  ├─ Load hot_grids = first 5K grids (from config)
  ├─ predictions_hot = [500K][hot_grids] subset
  └─ Cache.populate(hot_grids, predictions_hot) → ~5 seconds
    (At this point: t≈1m:05s, hot cache ready, /predict requests hit cache)
↓
PHASE 2 (Async - spawns thread, non-blocking)
  ├─ warm_grids = grids[5K:50K]
  ├─ predictions_warm = [500K][warm_grids] subset
  └─ Thread: Cache.populate(warm_grids, predictions_warm) → ~30 seconds
    (Target completion: t≈1m:35s)
↓
PHASE 3 (Async - spawns thread, lower priority)
  ├─ cold_grids = grids[50K:500K]
  ├─ predictions_cold = [500K][cold_grids] subset
  └─ Thread: Cache.populate(cold_grids, predictions_cold) → ~20 seconds
    (Target completion: t≈1m:55s)
↓
[Background Async DuckDB Write]
Thread: Write predictions to predictions table (separate DuckDB file)
  UPSERT INTO predictions (grid_id, timestamp_hour, pm25_prediction, model_version, schema_hash, created_at)
  VALUES (...) for each [grid_id, pm25_prediction] pair
  Batched: every 100 predictions or timeout (~5-10 seconds for all 500K)
  Non-blocking: InferenceJob subprocess can exit before writes complete
↓
[InferenceJob exits with code 0 if successful, non-zero on error]
```

### **3.3 Cache Architecture**

```
ResultCache (in Coordinator memory)
├─ Structure: Dict[grid_id] → (pm25_prediction, timestamp_created)
├─ Capacity: ~500K entries = ~4MB metadata + predictions
├─ TTL: Infinite (overwritten hourly by Kashi)
├─ Access: /predict requests do direct lookup
└─ Population Strategy:
    t=0m:05s: Hot grids (5K) queryable
    t=0m:35s: Hot + Warm grids (50K) queryable
    t=0m:55s: All grids (500K) queryable
    t=1m:00s: Ready for full cache hit rate

If Cache Miss (grid not yet cached):
  → Return stale prediction from previous hour
  → Include metadata: {pm25: value, age_seconds: X, from_cache: true, stale: true}

If Complete Cache Failure (InferenceJob never completes):
  → Serve predictions from predictions table (DuckDB)
  → These are from previous successful run
  → Include metadata about age
```

---

## **4. DETAILED COMPONENT SPECIFICATIONS**

### **4.1 Orchestrator (Main Scheduler)**

**File:** `kashi/orchestrator.py`

**Class:** `InferenceOrchestrator`

```python
class InferenceOrchestrator:
    """Orchestrates WriterJob and InferenceJob scheduling."""
    
    def __init__(self, config: KashiConfig):
        self.config = config
        self.scheduler = BackgroundScheduler()
        self.retry_metadata = {}  # In-memory retry state
        self.last_successful_completion = {}  # Per-grid tracking
        
    def start(self):
        """Start APScheduler background thread."""
        self.scheduler.add_job(
            self.run_daily_cycle,
            'cron',
            hour=self.config.schedule_hour,
            minute=self.config.schedule_minute,
            name='daily_inference_cycle'
        )
        self.scheduler.start()
    
    def run_daily_cycle(self):
        """Execute WriterJob → InferenceJob sequence."""
        cycle_id = str(uuid.uuid4())  # Unique ID for this cycle
        logger.info(f"[{cycle_id}] Starting daily cycle at {datetime.utcnow()}")
        
        try:
            # Step 1: Run WriterJob
            writer_result = self._run_writer_job(cycle_id)
            if not writer_result['success']:
                logger.error(f"[{cycle_id}] WriterJob failed: {writer_result['error']}")
                # Postpone inference, but next cycle still runs at fixed time
                return
            
            logger.info(f"[{cycle_id}] WriterJob completed successfully")
            
            # Step 2: Run InferenceJob
            inference_result = self._run_inference_job(cycle_id)
            if not inference_result['success']:
                logger.error(f"[{cycle_id}] InferenceJob failed: {inference_result['error']}")
                # Retry immediately
                max_retries = 3
                for attempt in range(max_retries):
                    logger.info(f"[{cycle_id}] Retry attempt {attempt+1}/{max_retries}")
                    inference_result = self._run_inference_job(cycle_id)
                    if inference_result['success']:
                        break
                
                if not inference_result['success']:
                    logger.error(f"[{cycle_id}] InferenceJob failed after {max_retries} retries")
                    # Serve stale cache, continue
            
            logger.info(f"[{cycle_id}] Daily cycle completed")
        
        except Exception as e:
            logger.exception(f"[{cycle_id}] Unexpected error in daily cycle: {e}")
    
    def _run_writer_job(self, cycle_id: str) -> Dict:
        """Spawn WriterJob subprocess, wait for completion."""
        try:
            process = subprocess.Popen(
                [sys.executable, '-m', 'kashi.writer_job', '--cycle-id', cycle_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.project_root
            )
            stdout, stderr = process.communicate(timeout=self.config.writer_timeout_seconds)
            
            success = process.returncode == 0
            return {
                'success': success,
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8') if stdout else '',
                'stderr': stderr.decode('utf-8') if stderr else '',
                'error': None if success else f"Exit code {process.returncode}"
            }
        except subprocess.TimeoutExpired:
            process.kill()
            return {'success': False, 'error': f'Timeout after {self.config.writer_timeout_seconds}s'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_inference_job(self, cycle_id: str) -> Dict:
        """Spawn InferenceJob subprocess, wait for completion."""
        try:
            process = subprocess.Popen(
                [sys.executable, '-m', 'kashi.inference_job', '--cycle-id', cycle_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.project_root
            )
            stdout, stderr = process.communicate(timeout=self.config.inference_timeout_seconds)
            
            success = process.returncode == 0
            return {
                'success': success,
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8') if stdout else '',
                'stderr': stderr.decode('utf-8') if stderr else '',
                'error': None if success else f"Exit code {process.returncode}"
            }
        except subprocess.TimeoutExpired:
            process.kill()
            return {'success': False, 'error': f'Timeout after {self.config.inference_timeout_seconds}s'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def shutdown(self):
        """Gracefully shutdown scheduler."""
        self.scheduler.shutdown()
```

---

### **4.2 WriterJob**

**File:** `kashi/writer_job.py`

**Purpose:** Fetch fresh raw data, write to DuckDB with minimal memory footprint

```python
class WriterJob:
    """Fetch and write raw observation data for next inference cycle."""
    
    def __init__(self, config: KashiConfig, cycle_id: str):
        self.config = config
        self.cycle_id = cycle_id
        self.timestamp_to_write = self._compute_write_timestamp()
        
    def _compute_write_timestamp(self) -> datetime:
        """Compute what timestamp to write data for.
        
        If WriterJob runs at t=-5m, we fetch "current" data and write it as
        the data for t=0m to t=60m.
        """
        now = datetime.utcnow()
        # Round to nearest hour
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    def run(self):
        """Execute writer job: fetch from 4 sources, write to DuckDB."""
        logger.info(f"[{self.cycle_id}] WriterJob starting")
        logger.info(f"[{self.cycle_id}] Writing data for timestamp: {self.timestamp_to_write}")
        
        try:
            # Step 1: Fetch from all 4 sources in parallel
            fetchers = {
                'meteo': MeteoFetcher(self.config),
                'pm25': PM25Fetcher(self.config),
                'landsat8': LandsatFetcher(self.config),
                'sentinel2': SentinelFetcher(self.config),
            }
            
            fetch_results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    source: executor.submit(fetcher.fetch)
                    for source, fetcher in fetchers.items()
                }
                for source, future in futures.items():
                    try:
                        fetch_results[source] = future.result(timeout=60)
                        logger.info(f"[{self.cycle_id}] Fetched {source}: {len(fetch_results[source])} rows")
                    except Exception as e:
                        logger.error(f"[{self.cycle_id}] Fetch failed for {source}: {e}")
                        fetch_results[source] = None
            
            # Step 2: Write to DuckDB (sequential, minimal memory)
            conn = duckdb.connect(str(self.config.duckdb_path), read_only=False)
            try:
                writer = DuckDBWriter(conn)
                
                # Write meteo
                if fetch_results['meteo']:
                    writer.write_meteo(fetch_results['meteo'], self.timestamp_to_write)
                
                # Write PM25
                if fetch_results['pm25']:
                    writer.write_pm25(fetch_results['pm25'], self.timestamp_to_write)
                
                # Write Landsat
                if fetch_results['landsat8']:
                    writer.write_landsat8(fetch_results['landsat8'], self.timestamp_to_write)
                
                # Write Sentinel
                if fetch_results['sentinel2']:
                    writer.write_sentinel2(fetch_results['sentinel2'], self.timestamp_to_write)
                
                conn.commit()
                logger.info(f"[{self.cycle_id}] WriterJob completed successfully")
                return 0  # Success
            
            finally:
                conn.close()
        
        except Exception as e:
            logger.exception(f"[{self.cycle_id}] WriterJob failed: {e}")
            return 1  # Failure
```

**Fetcher Classes:**

```python
class MeteoFetcher:
    """Fetch meteorological observations from OpenMeteo API."""
    def fetch(self) -> List[Dict]:
        # Calls OpenMeteo API
        # Returns list of dicts: [{"grid_id": "...", "temperature_2m": ..., ...}, ...]
        # Format: one row per grid for a single timestamp
        pass

class PM25Fetcher:
    """Fetch PM2.5 observations from CPCB API."""
    def fetch(self) -> List[Dict]:
        # Calls CPCB API
        # Returns list of dicts for one hour in the past (t=-60m)
        pass

class LandsatFetcher:
    """Fetch Landsat8 data from GEE."""
    def fetch(self) -> List[Dict]:
        # Calls GEE API
        # Returns list of dicts with B1-B11 for each grid (only if new data available)
        pass

class SentinelFetcher:
    """Fetch Sentinel2 data from GEE."""
    def fetch(self) -> List[Dict]:
        # Calls GEE API
        # Returns list of dicts with B2, B4, B8, AOT, QA10 for each grid
        pass
```

**DuckDB Writer:**

```python
class DuckDBWriter:
    """Write fetched data to DuckDB tables."""
    
    def write_meteo(self, data: List[Dict], timestamp: datetime):
        """Insert meteo observations."""
        for row in data:
            self.conn.execute("""
                INSERT INTO meteo (grid_id, timestamp, temperature_2m, ...) 
                VALUES (?, ?, ?, ...)
            """, [row['grid_id'], timestamp, row.get('temperature_2m'), ...])
    
    def write_pm25(self, data: List[Dict], timestamp: datetime):
        """Insert PM2.5 observations."""
        for row in data:
            self.conn.execute("""
                INSERT INTO pm25_observation (grid_id, timestamp, pm25) 
                VALUES (?, ?, ?)
            """, [row['grid_id'], timestamp, row['pm25']])
    
    # Similar for landsat8, sentinel2
```

---

### **4.3 InferenceJob**

**File:** `kashi/inference_job.py`

**Purpose:** Load features, run LightGBM, populate cache, write predictions

```python
class InferenceJob:
    """Run batch inference for 500K grids."""
    
    def __init__(self, config: KashiConfig, cycle_id: str, cache_provider):
        self.config = config
        self.cycle_id = cycle_id
        self.cache = cache_provider  # Injected from Coordinator
        self.timestamp_hour = self._compute_timestamp_hour()
        
    def _compute_timestamp_hour(self) -> datetime:
        """Compute timestamp for this inference cycle."""
        now = datetime.utcnow()
        return now.replace(minute=0, second=0, microsecond=0)
    
    def run(self):
        """Execute inference job: load features → infer → cache → write."""
        logger.info(f"[{self.cycle_id}] InferenceJob starting")
        
        try:
            # Step 1: Load [500K × 64] features
            logger.info(f"[{self.cycle_id}] Loading features for timestamp {self.timestamp_hour}")
            start_time = time.time()
            
            feature_assembler = BatchFeatureAssembler(self.config)
            features = feature_assembler.assemble_all(self.timestamp_hour)
            # Returns [500K × 64] NumPy array
            
            load_time = time.time() - start_time
            logger.info(f"[{self.cycle_id}] Features loaded in {load_time:.1f}s")
            
            # Step 2: Run LightGBM inference
            logger.info(f"[{self.cycle_id}] Running inference on {len(features)} grids")
            start_time = time.time()
            
            model = self._load_model()
            predictions = model.predict(features)
            # Returns [500K] array of float predictions
            
            infer_time = time.time() - start_time
            logger.info(f"[{self.cycle_id}] Inference completed in {infer_time:.1f}s")
            
            # Step 3: Populate cache (staged: hot → warm → cold)
            logger.info(f"[{self.cycle_id}] Populating cache (staged)")
            self._populate_cache_staged(predictions, features)
            
            # Step 4: Write predictions to DuckDB (async, non-blocking)
            logger.info(f"[{self.cycle_id}] Scheduling prediction writes")
            self._schedule_prediction_writes(predictions)
            
            logger.info(f"[{self.cycle_id}] InferenceJob completed successfully")
            return 0  # Success
        
        except Exception as e:
            logger.exception(f"[{self.cycle_id}] InferenceJob failed: {e}")
            return 1  # Failure
    
    def _load_model(self) -> lightgbm.Booster:
        """Load pre-trained LightGBM model."""
        model = lightgbm.Booster(model_file=str(self.config.model_path))
        return model
    
    def _populate_cache_staged(self, predictions: np.ndarray, features: np.ndarray):
        """Populate cache in stages: hot → warm → cold."""
        grid_ids = self._load_all_grid_ids()  # [500K]
        
        # Stage 1: Hot grids (synchronous)
        logger.info(f"[{self.cycle_id}] [Stage 1] Caching hot grids (0:5000)")
        hot_slice = slice(0, 5000)
        hot_grids = grid_ids[hot_slice]
        hot_preds = predictions[hot_slice]
        
        start = time.time()
        for grid_id, pred in zip(hot_grids, hot_preds):
            self.cache.set(grid_id, pred, self.timestamp_hour)
        hot_time = time.time() - start
        logger.info(f"[{self.cycle_id}] [Stage 1] Cached 5K grids in {hot_time:.1f}s")
        
        # Stage 2: Warm grids (async)
        logger.info(f"[{self.cycle_id}] [Stage 2] Caching warm grids (5000:50000) async")
        warm_slice = slice(5000, 50000)
        Thread(
            target=self._cache_batch,
            args=(grid_ids[warm_slice], predictions[warm_slice], "warm")
        ).start()
        
        # Stage 3: Cold grids (async, lower priority)
        logger.info(f"[{self.cycle_id}] [Stage 3] Caching cold grids (50000:) async")
        cold_slice = slice(50000, len(grid_ids))
        Thread(
            target=self._cache_batch,
            args=(grid_ids[cold_slice], predictions[cold_slice], "cold")
        ).start()
    
    def _cache_batch(self, grid_ids: np.ndarray, predictions: np.ndarray, stage_name: str):
        """Helper to cache a batch of predictions in background thread."""
        start = time.time()
        for grid_id, pred in zip(grid_ids, predictions):
            self.cache.set(grid_id, pred, self.timestamp_hour)
        elapsed = time.time() - start
        logger.info(f"[{self.cycle_id}] [{stage_name.upper()}] Cached {len(grid_ids)} grids in {elapsed:.1f}s")
    
    def _schedule_prediction_writes(self, predictions: np.ndarray):
        """Schedule async writes of predictions to DuckDB."""
        grid_ids = self._load_all_grid_ids()
        
        def write_predictions_batch():
            conn = duckdb.connect(str(self.config.duckdb_path), read_only=False)
            try:
                batch_size = 100
                for i in range(0, len(predictions), batch_size):
                    batch_grids = grid_ids[i:i+batch_size]
                    batch_preds = predictions[i:i+batch_size]
                    
                    for grid_id, pred in zip(batch_grids, batch_preds):
                        conn.execute("""
                            INSERT OR REPLACE INTO predictions 
                            (grid_id, timestamp_hour, pm25_prediction, model_version, schema_hash, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, [
                            grid_id,
                            self.timestamp_hour,
                            float(pred),
                            self.config.model_version,
                            self.config.schema_hash,
                            datetime.utcnow()
                        ])
                    
                    conn.commit()
                
                logger.info(f"[{self.cycle_id}] All predictions written to DuckDB")
            
            finally:
                conn.close()
        
        # Run in background thread
        Thread(target=write_predictions_batch, daemon=True).start()
    
    def _load_all_grid_ids(self) -> np.ndarray:
        """Load all 500K grid IDs from grid table."""
        conn = duckdb.connect(str(self.config.duckdb_path), read_only=True)
        result = conn.execute("SELECT grid_id FROM grid ORDER BY grid_id").fetchall()
        conn.close()
        return np.array([row[0] for row in result])
```

---

### **4.4 BatchFeatureAssembler**

**File:** `kashi/batch_feature_assembler.py`

**Purpose:** Load features for all 500K grids at once (minimal memory management)

```python
class BatchFeatureAssembler:
    """Assemble features for all 500K grids at once."""
    
    def __init__(self, config: KashiConfig):
        self.config = config
        self.feature_schema = self._load_feature_schema()
        self.grid_ids = self._load_grid_ids()
    
    def assemble_all(self, timestamp_hour: datetime) -> np.ndarray:
        """Assemble [500K × 64] feature matrix.
        
        Returns:
            np.ndarray of shape [500000, 64] with 64 features for each grid.
            Missing features filled with NaN.
        """
        logger.info(f"Assembling features for {len(self.grid_ids)} grids at {timestamp_hour}")
        
        features_list = []
        errors = {'count': 0, 'grids': []}
        
        conn = duckdb.connect(str(self.config.duckdb_path), read_only=True)
        
        try:
            for i, grid_id in enumerate(self.grid_ids):
                if i % 50000 == 0:
                    logger.info(f"  Assembled features for {i}/{len(self.grid_ids)} grids")
                
                try:
                    features_df = self._fetch_features_single(conn, grid_id, timestamp_hour)
                    features_list.append(features_df.values[0])
                except Exception as e:
                    logger.warning(f"Failed to assemble features for {grid_id}: {e}")
                    # Use NaN row for this grid
                    features_list.append([np.nan] * 64)
                    errors['count'] += 1
                    errors['grids'].append(grid_id)
        
        finally:
            conn.close()
        
        if errors['count'] > 0:
            logger.warning(f"Feature assembly had {errors['count']} failures (will use NaN)")
        
        features_array = np.array(features_list)  # [500K × 64]
        logger.info(f"Feature assembly complete: shape={features_array.shape}, dtype={features_array.dtype}")
        
        return features_array
    
    def _fetch_features_single(
        self,
        conn: duckdb.DuckDBPyConnection,
        grid_id: str,
        timestamp_hour: datetime
    ) -> pd.DataFrame:
        """Fetch and compute 64 features for single grid (reuses existing FeatureAssembler)."""
        assembler = FeatureAssembler(str(self.config.duckdb_path), self.feature_schema)
        return assembler.fetch_features(grid_id, timestamp_hour)
    
    def _load_feature_schema(self) -> Dict:
        """Load feature schema from config."""
        with open(self.config.feature_schema_path) as f:
            return json.load(f)
    
    def _load_grid_ids(self) -> List[str]:
        """Load all grid IDs from grid table."""
        conn = duckdb.connect(str(self.config.duckdb_path), read_only=True)
        result = conn.execute("SELECT grid_id FROM grid ORDER BY grid_id").fetchall()
        conn.close()
        return [row[0] for row in result]
```

---

### **4.5 Cache Provider Interface**

**File:** `kashi/cache.py`

```python
class CacheProvider:
    """Interface for cache operations from InferenceJob."""
    
    def set(self, grid_id: str, prediction: float, timestamp: datetime):
        """Store a prediction in cache."""
        raise NotImplementedError
    
    def get(self, grid_id: str) -> Optional[Dict]:
        """Retrieve prediction from cache. Returns {prediction, timestamp, age_seconds}."""
        raise NotImplementedError
    
    def get_batch(self, grid_ids: List[str]) -> Dict[str, Dict]:
        """Retrieve multiple predictions."""
        raise NotImplementedError
```

**Coordinator provides cache instance to InferenceJob via IPC (TBD: Redis, shared memory, or file).**

---

## **5. ERROR HANDLING & RESILIENCE**

### **5.1 WriterJob Failures**

| Scenario | Handling |
|----------|----------|
| API fetch timeout | Retry once, then skip that source |
| Partial API success | Write what succeeded, log warning |
| DuckDB write fails | Retry entire job, fail if 3 attempts exhausted |
| Timeout > 5 minutes | Kill process, retry next cycle |

### **5.2 InferenceJob Failures**

| Scenario | Handling |
|----------|----------|
| Feature assembly fails | Use NaN for missing features, continue |
| LightGBM crash | Retry immediately (up to 3 times) |
| Cache write fails | Log error, predictions still go to DuckDB |
| DuckDB write fails | Log error, retry async (non-blocking) |
| Timeout > 5 minutes | Kill process, retry immediately |
| All retries exhausted | Serve stale cache from previous hour |

### **5.3 State Persistence**

```python
class JobStateManager:
    """Persist minimal state for recovery."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load_state()
    
    def record_completion(self, cycle_id: str, job_type: str, grid_ids: List[str]):
        """Record that these grids were successfully processed."""
        self.state[f"{cycle_id}_{job_type}"] = {
            'grids': grid_ids,
            'timestamp': datetime.utcnow(),
            'job_id': cycle_id
        }
        self._save_state()
    
    def _load_state(self) -> Dict:
        """Load state from file (or empty if not exists)."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        """Persist state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
```

---

## **6. CONFIGURATION**

**File:** `kashi/config.py`

```python
class KashiConfig:
    """Kashi scheduler configuration."""
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    duckdb_path: Path = project_root / "duckdb" / "takshashila.duckdb"
    model_path: Path = project_root / "dwarka" / "model.pkl"
    feature_schema_path: Path = project_root / "artifacts" / "feature_schema.json"
    config_file: Path = project_root / "kashi" / "config.yaml"
    
    # Scheduling
    schedule_hour: int = 0  # Start at top of hour (UTC)
    schedule_minute: int = 0
    writer_start_offset_minutes: int = 2  # WriterJob starts 2 min before hour (was: TBD/5)
    writer_timeout_seconds: int = 300  # 5 minutes (generous buffer for satellite fetches)
    inference_start_delay_seconds: int = 120  # Always wait 2min after hour before starting InferenceJob
    
    # Versioning
    model_version: str = "1.0.0"  # Read from config file
    schema_hash: str = "abc123"  # Read from config file
    
    @classmethod
    def load_from_file(cls, config_file: Path) -> 'KashiConfig':
        """Load config from YAML file."""
        with open(config_file) as f:
            data = yaml.safe_load(f)
        config = cls()
        config.model_version = data['model_version']
        config.schema_hash = data['schema_hash']
        return config
```

**File:** `kashi/config.yaml`

```yaml
# Kashi Configuration

# Model versioning
model_version: "1.0.0"
schema_hash: "20260124_phase2a"

# Scheduling
schedule:
  hour: 0           # UTC hour (0-23)
  minute: 0         # UTC minute (0-59)
  
schedule_offsets:
  writer_start_minutes: 2        # WriterJob starts 2 min before hour
  inference_start_seconds: 120   # InferenceJob always starts at top of hour + 2 min

# Timeouts (seconds)
timeouts:
  writer_job: 300           # 5 minutes (generous for satellite fetches)
  inference_job: 300        # 5 minutes (for feature assembly + inference)

# Cache configuration
cache:
  hot_grid_count: 5000      # Synchronous cache population
  warm_grid_count: 45000    # Async thread (priority)
  cold_grid_count: 450000   # Async thread (background)

# Logging
logging:
  level: INFO
  format: JSON
  file: /var/log/kashi/kashi.log

# API Credentials (sourced from environment)
openmeteo:
  endpoint: https://archive-api.open-meteo.com/v1/archive
  timeout_seconds: 15
  rate_limit_delay_seconds: 0.5  # Built into Open-Meteo SDK

pm25:
  endpoint: https://api.openaqdata.org/v3/measurements  # or local CSV path
  timeout_seconds: 15
  api_key: ${PM25_API_KEY}

gee:
  service_account: ${GEE_SERVICE_ACCOUNT}
  project: ${GEE_PROJECT}
  timeout_seconds: 60  # GEE operations slower
  batch_size: 1000
  max_workers: 8
```

---

## **7. FILE STRUCTURE**

```
/Users/shlokjain/pollu/kashi/
├── __init__.py
├── __main__.py                 # Entry point: python -m kashi
├── orchestrator.py             # Main scheduler (WriterJob + InferenceJob coordination)
├── writer_job.py               # Fetch raw data, write to DuckDB
├── inference_job.py            # Load features, infer, cache, write predictions
├── batch_feature_assembler.py  # Batch feature loading for 500K grids
├── cache.py                    # Cache interface (provider injected from coordinator)
├── config.py                   # Configuration classes
├── config.yaml                 # Configuration file
├── TECHNICAL_DESIGN.md         # This document
│
├── fetchers/
│   ├── __init__.py
│   ├── meteo_fetcher.py        # OpenMeteo API client
│   ├── pm25_fetcher.py         # CPCB API client
│   ├── landsat_fetcher.py      # GEE Landsat8 client
│   └── sentinel_fetcher.py     # GEE Sentinel2 client
│
├── writers/
│   ├── __init__.py
│   └── duckdb_writer.py        # DuckDB write operations
│
├── tests/
│   ├── __init__.py
│   ├── test_orchestrator.py
│   ├── test_writer_job.py
│   ├── test_inference_job.py
│   ├── test_batch_assembler.py
│   └── conftest.py             # Pytest fixtures
│
└── logs/
    └── kashi.log               # Rotating log file
```

---

## **8. INTEGRATION WITH COORDINATOR**

### **8.1 Coordinator Startup**

```go
// coordinator/main.go
func main() {
    config := DefaultConfig()
    coordinator, err := NewCoordinator(config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Start Kashi subprocess
    kashiCmd := exec.Command(
        "python",
        "-m", "kashi",
    )
    kashiCmd.Stdout = os.Stdout
    kashiCmd.Stderr = os.Stderr
    err = kashiCmd.Start()
    if err != nil {
        log.Fatal(err)
    }
    
    // Start HTTP server
    http.HandleFunc("/predict", coordinator.handlePredict)
    http.HandleFunc("/health", coordinator.handleHealth)
    
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### **8.2 Cache Sharing (TBD)**

**Question:** How is in-memory cache shared between Kashi subprocess and Coordinator?

**Options:**
1. **Redis:** Both Kashi and Coordinator write/read from Redis
2. **Shared memory:** Python `multiprocessing.shared_memory`
3. **File-based:** Kashi writes predictions to file, Coordinator reads
4. **IPC socket:** Custom protocol between processes

**Recommendation for MVP:** Use option 1 (Redis) for simplicity.

```python
# In inference_job.py
def _populate_cache_staged(self, predictions, features):
    """Populate Redis cache."""
    redis_client = redis.Redis(host='localhost', port=6379)
    
    # Stage 1: Hot grids
    for grid_id, pred in zip(hot_grids, hot_preds):
        redis_client.set(
            f"cache:{grid_id}",
            json.dumps({'pm25': float(pred), 'timestamp': self.timestamp_hour.isoformat()}),
            ex=3600  # 1 hour TTL
        )
```

```go
// In coordinator
func (c *Coordinator) getFromCache(gridID string) (float64, bool) {
    redisClient := redis.NewClient(...)
    val, err := redisClient.Get(ctx, fmt.Sprintf("cache:%s", gridID)).Result()
    if err != nil {
        return 0, false
    }
    var cached map[string]interface{}
    json.Unmarshal([]byte(val), &cached)
    return cached["pm25"].(float64), true
}
```

---

## **9. MONITORING & OBSERVABILITY**

### **9.1 Metrics to Emit**

```python
# Prometheus metrics
kashi_writer_job_duration_seconds = Histogram(
    'kashi_writer_job_duration_seconds',
    'Duration of writer job'
)
kashi_inference_job_duration_seconds = Histogram(
    'kashi_inference_job_duration_seconds',
    'Duration of inference job'
)
kashi_feature_assembly_duration_seconds = Histogram(
    'kashi_feature_assembly_duration_seconds',
    'Time to assemble all 500K features'
)
kashi_predictions_cached_total = Counter(
    'kashi_predictions_cached_total',
    'Number of predictions written to cache'
)
kashi_predictions_written_total = Counter(
    'kashi_predictions_written_total',
    'Number of predictions written to DuckDB'
)
kashi_job_failures_total = Counter(
    'kashi_job_failures_total',
    'Number of failed jobs',
    ['job_type']
)
```

### **9.2 Logging**

All log messages include `cycle_id` for traceability:

```
[550e8400-e29b-41d4-a716-446655440000] WriterJob starting
[550e8400-e29b-41d4-a716-446655440000] Fetched meteo: 500000 rows
[550e8400-e29b-41d4-a716-446655440000] InferenceJob starting
[550e8400-e29b-41d4-a716-446655440000] Features loaded in 45.2s
[550e8400-e29b-41d4-a716-446655440000] Inference completed in 12.3s
[550e8400-e29b-41d4-a716-446655440000] Daily cycle completed
```

---

## **10. TESTING STRATEGY**

### **10.1 Unit Tests**

```python
# kashi/tests/test_batch_assembler.py
def test_assemble_all_returns_correct_shape():
    """Feature matrix should be [500K × 64]."""
    assembler = BatchFeatureAssembler(test_config)
    features = assembler.assemble_all(test_timestamp)
    assert features.shape == (500000, 64)

def test_assemble_handles_missing_features():
    """Missing features filled with NaN."""
    assembler = BatchFeatureAssembler(test_config)
    features = assembler.assemble_all(test_timestamp)
    # Some grids might have NaN
    assert np.isnan(features).sum() <= 1000  # Acceptable threshold
```

### **10.2 Integration Tests**

```python
# kashi/tests/test_inference_job.py
def test_inference_job_end_to_end():
    """Full inference job: load → infer → cache → write."""
    # Use mock DuckDB with 1000 test grids
    # Run InferenceJob
    # Verify: cache populated, predictions written, exit code 0
```

### **10.3 Load Tests**

```bash
# kashi/tests/load_test.py
# Simulate 500K grids with realistic feature loading latency
# Measure: total duration, cache population timing, memory usage
```

---

## **11. SUCCESS CRITERIA**

| Metric | Target | Acceptable | Notes |
|--------|--------|-----------|-------|
| **WriterJob Duration** | <30s | <40s | 95% of days complete in <40s |
| **Feature Assembly Time** | <45s | <60s | Batch load 500K grids |
| **LightGBM Inference Time** | <15s | <30s | Vectorized predict call |
| **Cache Population (Hot)** | <10s | <15s | 5K grids synchronous |
| **Total InferenceJob Duration** | <60s | <90s | From start to hot cache ready |
| **Cache Hit Rate** | 99% | >95% | All 500K predictions cached hourly |
| **Prediction Write Latency** | <5ms per grid | <10ms | Async, non-blocking |
| **Error Recovery** | 3 retries max | Works | Exponential backoff |
| **Stale Cache Serving** | <1 hour old | <2 hours | Fallback on InferenceJob failure |
| **System Availability** | 99%+ | >95% | Handle satellite unavailability gracefully |

---

## **12. FUTURE ENHANCEMENTS**

1. **Incremental Updates:** Instead of full 500K batch, update only changed grids
2. **Distributed Inference:** Shard predictions across multiple machines
3. **Dynamic Hot Grid Detection:** Learn which grids are frequently queried, prioritize
4. **Model Retraining:** Periodic model updates without stopping scheduler
5. **Feature Caching:** Cache features (not just predictions) to avoid recomputation

---

**Document Status:** Ready for Implementation  
**Last Updated:** 2026-01-24  
**Next Phase:** Begin Phase 2a implementation (BatchFeatureAssembler)
