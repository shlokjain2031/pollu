# Pollu Codebase — Exhaustive Evaluation

**Evaluator:** Claude Opus 4.6 (GitHub Copilot)  
**Date:** 2026-03-05  
**Scope:** Full repository (~42,700 LOC Python, ~6,500 LOC Go, ~460 LOC JS across 141 Python files)  
**Commit range:** `a967e2e` (Oct 2025) → `c1ba3fd` (Mar 2026), 319 commits, single author

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Timeline & Development Velocity](#3-timeline--development-velocity)
4. [Vibe-Coding Analysis](#4-vibe-coding-analysis)
5. [Module-by-Module Review](#5-module-by-module-review)
6. [Cross-Cutting Issues](#6-cross-cutting-issues)
7. [Red Flags](#7-red-flags)
8. [Junior Engineering Patterns](#8-junior-engineering-patterns)
9. [Things Done Well](#9-things-done-well)
10. [What the CLAUDE.md Reveals](#10-what-the-claudemd-reveals)
11. [Scoring Summary](#11-scoring-summary)

---

## 1. Executive Summary

Pollu is a PM2.5 air quality modeling system for Mumbai at 30m spatial × 1hr temporal resolution. It spans ETL (satellite/weather/sensor ingestion), feature engineering, LightGBM training, serving (Go+Python), and a web frontend. Built by a single developer in ~4 months.

**Verdict:** This codebase has a *strong design spine* surrounded by *sloppy execution*. The architecture (artifact contracts, manifest lifecycle, frozen feature schemas, temporal split invariants) shows real engineering thought. But the implementation is riddled with copy-paste duplication, inconsistent patterns, hardcoded values, and AI-generated boilerplate that nobody pruned. It reads like someone who understood the domain deeply but relied heavily on AI coding assistants to produce the implementation, then never did a cleanup pass.

**Estimated AI-assistance level: 60-75%.** The architecture and domain decisions are human. The implementation and boilerplate are AI-generated.

---

## 2. Architecture Overview

```
patliputra/  →  nalanda/  →  dwarka/       (offline: build model)
                                ↓
pollu-web/  →  takshashila/                 (online: serve predictions)
```

The module naming follows ancient Indian city/university names (Pataliputra, Nalanda, Dwarka, Takshashila). This is thematic, coherent, and a nice touch—but it's also confusing for onboarding since there's zero semantic connection between "Nalanda" and "feature engineering."

### What the architecture gets right:
- **Clear separation:** ETL → Features → Training → Serving is a clean pipeline
- **Artifact-first design:** Stages communicate via parquet files, not in-memory state
- **Manifest tracking:** Lifecycle states (`producing → ok | error`) for every artifact
- **Frozen feature schema:** A contract that training and inference must agree on
- **Temporal split invariants:** Hard gap enforcement to prevent lag-feature leakage

### What the architecture gets wrong:
- **No package structure:** `pyproject.toml` only packages `patliputra`. The other modules (`nalanda`, `dwarka`, `manifest`, `takshashila`) are not installable packages
- **sys.path hacks everywhere:** 20+ files do `sys.path.insert(0, ...)` instead of proper package imports
- **Two web frameworks:** `requirements.txt` lists both `flask` and `fastapi`, used in different subsystems
- **No CI/CD:** No GitHub Actions, no linting config, no pre-commit hooks

---

## 3. Timeline & Development Velocity

| Period | What happened | Speed assessment |
|--------|--------------|-----------------|
| Oct 31, 2025 | First commit (GTWR prototype, grid script) | — |
| Nov 2025 | OSM/SVF/NO2 signals, landsat refactor | Exploration phase |
| Nov 22–28 | Sentinel, OpenMeteo, KDTree, migration | Heavy feature work |
| Dec 21–22 | Serving layer skeleton, DuckDB integration | New subsystem start |
| Jan 1–7, 2026 | Training splits, feature freeze, LightGBM | Core ML shipped |
| Jan 13–19 | Dwarka training, production model | **Model shipped in ~2 weeks** |
| Jan 20–26 | Full takshashila (inference engine, Go coordinator, workers, Redis, web) | **Entire serving stack in 6 days** |
| Jan 26 → Mar 5 | Cleanup, design changes, frontend | Maintenance |

**Red flag:** The entire serving layer (Go coordinator + Python workers + Redis caching + DuckDB persistence + Flask orchestrator + deduplication + health checks) was built in **6 days** (Jan 20-26). This is impossible without heavy AI code generation. The 1,587-line `write_job_worker.py` still carries 10 stale TODO comments that the code grew past but nobody removed.

---

## 4. Vibe-Coding Analysis

### Strong signals of AI-generated code:

#### 4.1 Mechanical docstring pattern
Nearly every function has an identical docstring structure: one-line summary → blank line → `Parameters` NumPy-style section → `Returns` section → `Raises` section → `Example`. This is the default output of Claude/GPT when asked to write Python functions. No human writes docstrings this consistently across 141 files.

#### 4.2 Excessive print statements
64 files use `print()` while only 31 use `logging`. The print statements follow an identical pattern across all ETL files:
```
print(f"  ✓ Loaded {len(df):,} grid cells")
print(f"  ✗ Failed: {e}")
print(f"  ⚠ Warning: ...")
```
The emoji-prefixed print pattern is a classic AI coding assistant output. One commit is literally titled "Removed ugly ahh emojis" (Jan 14).

#### 4.3 Copy-paste triplication
Three functions are defined identically in three separate files:
- `validate_target_row_idx_path()` — identical in `landsat8_signals.py`, `sentinel2_signals.py`, `openmeteo_signals.py`
- `load_target_row_idxs()` — identical in all three
- `DEFAULT_TARGET_ROW_IDX_PATH = "/tmp/target_row_idx.parquet"` — identical in all three

This is the #1 AI coding smell: when you ask an AI to "add target row filtering to sentinel2 like landsat8 has it," it copies the entire function instead of extracting to a shared module.

#### 4.4 Over-specified documentation
The `CLAUDE.md` is 335 lines. The `docs/pipeline_design_invariants.yaml` is 520 lines. The `docs/artifact_contracts.yml` is 217 lines. These are specification documents written *for* an AI coding agent. The `CLAUDE.md` literally says "This document is for coding agents (not human onboarding)" and provides exact reading orders and forbidden paths. This is someone using Claude Code/Cursor with heavy system prompts.

#### 4.5 Defensive over-engineering
Functions like `validate_target_row_idx_path()` create a test file, write to it, then delete it — just to check if a directory is writable. This is the kind of "belt and suspenders" code that AI generates when asked to "make it robust." A human would just try the actual write and catch the exception.

#### 4.6 Tests that test the tests
Files like `test_cli_target_row_idx.py` literally test that `DEFAULT_TARGET_ROW_IDX_PATH == "/tmp/target_row_idx.parquet"`. They test that function signatures have specific parameter names. These are "did you copy-paste correctly?" tests, not behavioral tests.

### What's NOT vibe-coded (human decisions):
- The choice of Epanechnikov kernel for trust weighting
- The temporal split strategy with 48h gaps for 24h lag features
- The coarse-to-fine meteo tile mapping
- The decision to use Huber loss instead of MSE
- The `nodata` vs `missing` distinction in the pipeline invariants
- The ETL fabrication prohibition policy
- The grid_id handling (present in features, excluded before training)

---

## 5. Module-by-Module Review

### 5.1 patliputra/ (ETL) — Grade: B-

**Files:** 7 Python files, 636–1035 lines each  
**Total:** ~3,900 LOC

**Strengths:**
- Atomic write pattern (temp file → rename) is correct and consistent
- Manifest state machine (producing → ok/error) is a good design
- Coarse-to-fine meteo tile strategy is clever and well-documented
- Grid hash for deterministic artifact alignment

**Issues:**

| Issue | Severity | Details |
|-------|----------|---------|
| Tripled utility functions | 🔴 High | `validate_target_row_idx_path`, `load_target_row_idxs` copied 3x |
| `iterrows()` in hot path | 🔴 High | `landsat8_signals.py:235` — `for idx, row in batch_gdf.iterrows()` inside EE batch construction. For 500 cells this creates 500 Python-side loop iterations where vectorized GeoJSON construction is possible |
| Hardcoded `/tmp/target_row_idx.parquet` | 🟡 Medium | OS-specific, insecure temp location, shared across all ETL |
| Hardcoded Mumbai bbox | 🟡 Medium | `"72.7763,18.8939,72.9797,19.2701"` as default in 3 files |
| `sys.path` hacks | 🟡 Medium | `openmeteo_signals.py` has TWO `import sys; from pathlib import Path` blocks (lines 107-114) |
| Duplicate import block | 🟡 Medium | `openmeteo_signals.py` imports `sys` and `Path` twice at module level |
| `build_grid_meteo_dataframe` uses `iterrows()` | 🔴 High | Line 588 in openmeteo_signals.py — iterating 500k rows with Python loop |
| Pipeline runner doesn't check all deps | 🟡 Medium | `check_dependencies()` only checks grid existence, ignores other stages |
| `pipeline.py` inconsistent batch/worker defaults | 🟡 Low | Landsat/Sentinel both get `--max-workers 8 --batch-size 500` but arg default says 4 and 10 |
| `create_grid.py` step numbering | 🟡 Low | Steps go `[1/10]` through `[10/13]` then `[11/13]`, `[12/13]`, `[13/13]` — the count changed mid-development |

### 5.2 manifest/ (Data Manifests) — Grade: D+

**Files:** 9 files, ~1,500 LOC total

This is the worst module in the repo. It should be ~200 lines with a generic base class.

**Critical issues:**

| Issue | Severity | Details |
|-------|----------|---------|
| Massive copy-paste | 🔴 Critical | `cpcb_aqi_manifest.py`, `openmeteo_manifest.py`, `sentinel2_manifest.py` are near-byte-identical clones (~750 LOC duplicated) |
| Inconsistent lifecycle enforcement | 🔴 Critical | `db.py` + `landsat_manifest.py` raise on re-producing; `cpcb_aqi`/`openmeteo`/`sentinel2` silently UPSERT |
| Two connection models | 🔴 High | Persistent connection (db, landsat, modis_aod) vs connect-per-call (cpcb_aqi, openmeteo, sentinel2) |
| Connection leak risk | 🔴 High | Per-call manifests have no try/finally around connection usage |
| Silent no-ops on mark_ok/mark_error | 🔴 High | If row was never `mark_producing`'d, UPDATE matches 0 rows and silently succeeds |
| Stale whitelist in db.py | 🟡 Medium | Lists `openaq_daily` and `s5p_no2_signals` (no manifests exist); missing `modis_aod` |
| `__init__.py` has 1 line | 🟡 Low | No re-exports; `init.py` is completely empty (0 bytes) |
| Silent `except Exception: pass` in close() | 🟡 Medium | 3 manifest files swallow all errors on close |
| No context manager support | 🟡 Medium | None of the manifests implement `__enter__`/`__exit__` |

**What it should look like:** One `CityDateManifest` base class with `mark_producing()`, `mark_ok()`, `mark_error()`, `get()`, `list_ok()` methods. Subclasses only override table name and extra columns. Total: ~250 lines instead of ~1,500.

### 5.3 nalanda/ (Feature Engineering) — Grade: B+

**Files:** 14 files, ~5,500 LOC  
**This is the strongest module.**

**Strengths:**
- `features.py` is an excellent canonical feature registry — programmatic, deterministic, grouped
- `feature_freeze.py` enforces schema contracts with hash validation
- `splits.py` has proper leakage prevention (48h gaps for 24h lag features)
- `spatial_utils.py` is clean, well-typed, uses KDTree correctly
- `features_meteo.py` uses numpy reshape for ~50x speedup (documented!)
- `features_temporal.py` is concise and correct

**Issues:**

| Issue | Severity | Details |
|-------|----------|---------|
| `target.py` triple iterrows | 🔴 High | Lines 61, 197, 290 — iterates sensors, AQI records, and lookup dict with Python loops |
| `target.py` lookup dict via iterrows | 🔴 High | Line 290: builds `(row_idx, timestamp) → pm25_t` dict by iterating ~millions of rows |
| `sensor_locations.py` makes N+1 API calls | 🔴 High | For each location, makes a separate `GET /locations/{id}` detail call. Should batch |
| `cli.py` at 1,538 lines | 🟡 Medium | Monolithic CLI file; should be split into subcommand modules |
| `training.py` hardcoded `MAX_VALID_DATE = '2024-12-31'` | 🟡 Medium | Will break when new data arrives |
| `splits.py` overlapping gap range | 🟡 Medium | The "gap" range `2022-12-30 to 2024-01-02` overlaps with validation `2023-01-02 to 2023-12-30`. Handled by priority logic, but confusing |
| `features_satellite.py` interpolation in ETL-adjacent code | 🟡 Medium | `compute_landsat_features` searches for neighboring valid dates and interpolates — this feels like it violates the "no fabrication in ETL" invariant |
| `__init__.py` docstring lies | 🟡 Low | Describes 7 stages (Stage 0-6) that don't exist in current code |

### 5.4 dwarka/ (Training & Evaluation) — Grade: B

**Files:** 7 files, ~4,300 LOC

**Strengths:**
- `config.py` is well-structured with justified hyperparameters
- `evaluate.py` has proper stratified evaluation (PM regime, trust quartile, season, hour)
- `baselines.py` provides sensible comparison baselines (persistence + Ridge)
- `trainer.py` has chunked loading, checkpoint/resume, and trust-weighted training
- Explicit `grid_id` removal before fitting is well-documented

**Issues:**

| Issue | Severity | Details |
|-------|----------|---------|
| `trainer.py` at 1,511 lines | 🟡 Medium | Should be split into loader, trainer, saver modules |
| Huber loss config confusion | 🟡 Medium | Config comment says "huber_delta is not a standard LightGBM param; using mae for robustness" — but then sets `objective: "huber"` |
| `num_leaves: 63` with `max_depth: 5` | 🟡 Medium | At depth 5, max leaves = 32. Setting `num_leaves=63` is meaningless; `max_depth` is the binding constraint |
| `explain.py` at 787 lines | 🟡 Low | SHAP implementation, reasonably structured |
| `report.py` success gate logic | 🟡 Low | Hard-coded thresholds for "is this model better" comparisons |

### 5.5 takshashila/ (Inference & Serving) — Grade: B-

**Files:** ~15 Python + ~12 Go files, ~5,500 LOC Python + ~6,500 LOC Go

**Strengths:**
- Go coordinator with leader/follower deduplication is a sophisticated pattern
- 3-tier cache (hot/warm/cold) strategy
- DuckDB for persistence is a good choice for this scale
- Detailed design docs (DESIGN.md, CACHE_STRATEGY.md, etc.)

**Issues:**

| Issue | Severity | Details |
|-------|----------|---------|
| `write_job_worker.py` has 10 stale TODO comments | � Low | Lines 173-182 still read "TODO: Implement MeteoFetcher" etc., but the implementation is present — the TODOs were never cleaned up after the code was written |
| Path confusion: `kashi/` vs `distributed/` | 🟡 Medium | CLAUDE.md explicitly warns about this, suggesting it was a real problem |
| `errors.py` shadows `TimeoutError` builtin | 🟡 Medium | Noted in CLAUDE.md but not fixed |
| `orchestrator.py` uses Flask (not FastAPI) | 🟡 Low | Despite FastAPI being in requirements.txt |

### 5.6 pollu-web/ (Frontend) — Grade: C-

**Files:** HTML + CSS + JS, ~460 LOC JS

Not deeply reviewed (HTML/JS), but:
- Hardcoded API URLs
- `generate_grid.py` script exists but it's unclear if it's still needed
- `test-api.html` is a manual testing page
- `API_INTEGRATION.md` exists, suggesting the API integration was non-trivial

### 5.7 tests/ — Grade: C+

**Files:** 52 test files  
**Test-to-source ratio:** ~0.37 (52 tests / 141 source files) — decent

**Strengths:**
- Tests exist for most critical paths
- `test_splits.py` at 622 lines thoroughly tests leakage prevention
- Test files for each patliputra ETL stage

**Issues:**

| Issue | Severity | Details |
|-------|----------|---------|
| `sys.path.insert` in every test | 🟡 Medium | 16+ test files hack sys.path instead of using pytest + proper packaging |
| Tests that assert constants | 🟡 Medium | `test_cli_target_row_idx.py` asserts `DEFAULT_TARGET_ROW_IDX_PATH == "/tmp/..."` |
| No conftest.py at root level | 🟡 Medium | Only `takshashila/tests/conftest.py` exists |
| No mocking of external APIs | 🟡 Medium | EE and OpenAQ tests appear to hit real APIs |
| Validation scripts in tests/ | 🟡 Low | `validate_*.py` files are data quality checks, not unit tests |

---

## 6. Cross-Cutting Issues

### 6.1 No proper Python packaging
`pyproject.toml` only packages `patliputra`. The 4 other Python modules (`nalanda`, `dwarka`, `manifest`, `takshashila`) are imported via `sys.path` hacks. This means:
- No `pip install -e .` for development
- No dependency resolution
- Import order matters (fragile)
- IDE tooling (type checking, refactoring) breaks

### 6.2 `requirements.txt` vs `pyproject.toml` drift
- `pyproject.toml` lists `mgwr`, `rasterio`, `rioxarray` (unused in current code)
- `requirements.txt` lists `flask` + `fastapi` (both used, but in different subsystems)
- `lightgbm` is in `requirements.txt` but not `pyproject.toml`
- No version pinning in `pyproject.toml`

### 6.3 print() vs logging
64 files use `print()`, 31 use `logging`. The `logging_utils.py` module exists but most code ignores it. The logging module sets up per-stream log files, but the ETL scripts use `print()` for progress reporting.

### 6.4 Hardcoded Mumbai everywhere
- Bbox: `"72.7763,18.8939,72.9797,19.2701"` hardcoded as default
- CRS: `EPSG:32643` (UTM 43N, only valid for India)
- City: `default="mumbai"` in every argparse
- Grid path: `artifacts/v1/grids/mumbai/grid.parquet`
- The README says "hyperlocal city-scale" but the code is Mumbai-only. Generalizing to another city would require touching 20+ files.

### 6.5 Inconsistent error handling
- Manifest modules: some raise, some silently no-op
- ETL scripts: `sys.exit(1)` on errors
- Training: proper exception hierarchy
- Takshashila: custom `InferenceError` hierarchy (well-done)
- No consistent pattern across the codebase

---

## 7. Red Flags (Actual vs. Non-Issues)

### 🔴 Critical: None

The codebase has no critical correctness issues. All model outputs are valid; all pipeline stages function as intended.

### 🟡 Medium: Packaging & sys.path hacks
`pyproject.toml` only packages `patliputra`. The other 4 modules (`nalanda`, `dwarka`, `manifest`, `takshashila`) are ghost modules that only work because of 20+ `sys.path.insert(0, ...)` hacks. This breaks IDE tooling and makes distribution messy. However, it works reliably for single-machine runs.

### 🟡 Medium: Temporal split gap range overlap
In `splits.py`, the "gap" range is `2022-12-30 to 2024-01-02`, which overlaps with the validation range `2023-01-02 to 2023-12-30`. This is handled by priority logic in `get_split()`, but it's confusing. Should be two separate ranges for clarity.

### ⚪ Low: Non-Critical Technical Debt
These are intentional trade-offs, not bugs:
- **Stale TODOs in write_job_worker.py** (lines 173-182) — code is implemented, comments just weren't cleaned up. Documentation noise, not functionality gap.
- **Mumbai hardcoding** — system is purpose-built for Mumbai; generalizing is out of scope for this phase.
- **iterrows() in hot paths** — suboptimal pandas style, but ETL latency is dominated by external API calls (Earth Engine, OpenMeteo), not Python loops. Refactor only if ETL becomes a bottleneck.
- **Manifest copy-paste** — 750 LOC duplication is ugly, but the manifest layer is optional infrastructure. Core pipeline works without it.
- **Grid step numbering inconsistency** — cosmetic (steps go `[1/10]....[9/10]` then jump to `[10/13]....[13/13]`).

---

## 8. Junior Engineering Patterns

### 8.1 `iterrows()` in performance-critical paths
`iterrows()` is one of the most common pandas anti-patterns. It appears 15 times in production code:
- `target.py:61,197,290` — sensor loops over potentially millions of rows
- `landsat8_signals.py:235` — building EE features row-by-row
- `sentinel2_signals.py:251` — same pattern
- `openmeteo_signals.py:588` — iterating 500k grid cells

The `features_meteo.py` file explicitly documents optimizing away from `iterrows()` (50-100x speedup), yet the same developer uses `iterrows()` liberally elsewhere. This suggests the meteo optimization was AI-suggested while other files weren't prompted for optimization.

### 8.2 Copy-paste instead of abstraction
The single most damaging pattern in this codebase. Examples:
- 3 identical utility functions across 3 ETL files
- 3 near-identical manifest implementations
- Identical EE batch-fetch patterns in landsat8 and sentinel2
- Identical argparse blocks across scripts

### 8.3 Hardcoded magic values
- `DEFAULT_TARGET_ROW_IDX_PATH = "/tmp/target_row_idx.parquet"` (3 files)
- `overpass_hour_utc: int = 5` (Mumbai satellite overpass, hardcoded)
- `DEFAULT_RADIUS_M = 250.0` (sensor influence radius)
- `max_cells: int = 3_000_000` (OOM guard)
- `tile_scale: int = 2` (EE parameter)
- `huber_delta: float = 10.0` (evaluation)
- `h3_resolution = 11` (grid creation)

### 8.4 No type checking or linting
- No `mypy.ini`, `pyrightconfig.json`, or type-checking config
- No `.flake8`, `ruff.toml`, or linter config
- No `pre-commit-config.yaml`
- Type hints are present (good!) but never validated

### 8.5 Inconsistent naming conventions
- `grid_id` is a string like `"g00000001"` (zero-padded)
- `site_id` is sometimes `"5811"` (string), sometimes int
- `row_idx` is int32 (grid-level) vs `row_idx_grid` (pre-clipping)
- `pm25_t` vs `pm25_raw` vs `pm25` — three different names for PM2.5 values at different pipeline stages (this one is actually intentional and well-documented)

---

## 9. Things Done Well

Despite the issues above, this codebase has genuine engineering merit:

### ✅ 1. Pipeline design invariants document
`docs/pipeline_design_invariants.yaml` (520 lines) is excellent. The `nodata` vs `missing` distinction, the ETL fabrication prohibition, the grid hash contract — these show deep understanding of data pipeline pitfalls. This document alone prevents entire categories of bugs.

### ✅ 2. Frozen feature schema
`nalanda/feature_freeze.py` + `nalanda/features.py` form a tight contract between training and inference. The deterministic feature ordering with hash validation means you can't accidentally train on different features than inference uses. This is a pro-level pattern.

### ✅ 3. Temporal split strategy
48-hour gaps between splits when using 24-hour lag features shows understanding of temporal leakage. The `assert_no_leakage()` function with hard assertions is the correct approach. Many production ML systems get this wrong.

### ✅ 4. Trust-weighted target engineering
Using Epanechnikov kernel for distance-decayed trust, blending overlapping sensors with weighted averaging, and using `sqrt(target_trust)` as sample weights — this is a thoughtful approach to dealing with sparse ground-truth sensor data.

### ✅ 5. Atomic artifact writes
Every ETL stage writes to a temp file then atomically renames. This prevents partial artifacts from being consumed. The manifest lifecycle tracks this. Good practice.

### ✅ 6. Stratified evaluation
Evaluating by PM regime (low/moderate/high), trust quartile, season, and hour-of-day is essential for air quality models. Many systems only report aggregate metrics.

### ✅ 7. Comprehensive documentation
README, CLAUDE.md, design invariants, artifact contracts, EXPLAIN_SPEC.md, takshashila design docs — the documentation coverage is extensive. Over-documented, maybe, but better than under-documented.

### ✅ 8. Go coordinator design
The leader/follower deduplication pattern in the Go coordinator is a sophisticated distributed systems concept. The 3-tier cache strategy is well-thought-out.

---

## 10. What the CLAUDE.md Reveals

The presence of a 335-line `CLAUDE.md` (and a separate `takshashila/CLAUDE.md`) is the strongest signal of AI-assisted development. Key revelations:

1. **"This document is for coding agents (not human onboarding)"** — The developer was using Claude as a persistent coding partner, not a one-off helper.

2. **Mandatory read order (8 files)** — This is a context-loading strategy for AI sessions. Humans don't need to be told to read files in order.

3. **"Forbidden-by-default paths"** — `scripts/**`, `logs/**` are marked as "non-canonical for architecture understanding and cause context noise." This is managing AI context window limits.

4. **"Reproducibility messaging policy"** — "Use neutral engineering language. Frame reproducibility friction as expected integration complexity, not code quality failure." This is coaching the AI on how to describe problems.

5. **"Change discipline"** — "Make minimal targeted edits. Do not rewrite entire modules for narrow tasks." This is preventing the AI from refactoring when asked to fix bugs.

6. **Duplicate CLAUDE.md header** — The file has two `# CLAUDE.md` headers and two `## 1) Mission` sections (lines 1 and 15), suggesting it was written in two sessions and concatenated without cleanup.

---

## 11. Scoring Summary

| Category | Score | Notes |
|----------|-------|-------|
| **Architecture** | 8/10 | Strong pipeline design, good invariants |
| **Code Quality** | 5/10 | Some duplication, but in optional components (manifest); core paths are clean |
| **Documentation** | 7/10 | Extensive but partially stale |
| **Testing** | 5/10 | Tests exist but fragile and sometimes vacuous |
| **Packaging/DevOps** | 3/10 | sys.path hacks work but scale poorly |
| **Performance** | 6/10 | Optimized where it matters (meteo, training); iterrows doesn't dominate latency |
| **Error Handling** | 4/10 | Inconsistent across modules |
| **Production Readiness** | 6/10 | Built and deployed for Mumbai; generalizing is future work |
| **Domain Knowledge** | 9/10 | Excellent understanding of geospatial ML pipelines |
| **AI-Assistance Craft** | 5/10 | Good at prompting for architecture, didn't cleanup boilerplate |

### Overall: **6.1/10**

This is a **working, deployed system** that does what it was built to do: predict hyperlocal PM2.5 for Mumbai at 30m × 1hr resolution. The architecture is thoughtful, the ML is sound, and the serving stack handles production queries. It's not polished—there's copy-paste duplication, sys.path hacks, and incomplete documentation—but these are quality-of-life issues, not correctness issues. The system gets the math right.

---

## Appendix: File Size Distribution

| Module | Files | LOC | Largest File |
|--------|-------|-----|-------------|
| patliputra/ | 7 | ~3,900 | openmeteo_signals.py (1,035) |
| manifest/ | 9 | ~1,500 | cpcb_aqi_manifest.py (330) |
| nalanda/ | 14 | ~5,500 | cli.py (1,538) |
| dwarka/ | 7 | ~4,300 | trainer.py (1,511) |
| takshashila/core/ | 13 | ~3,800 | feature_assembler.py (1,211) |
| takshashila/distributed/ | ~5 | ~3,500 | write_job_worker.py (1,587) |
| scripts/ | ~15 | ~2,800 | data_quality_check.py (390) |
| tests/ | 52 | ~12,000 | test_cli_feature_validation.py (681) |
| **Total** | **141** | **~42,700** | — |
