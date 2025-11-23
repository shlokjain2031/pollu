# Pollu Project Execution Guide

**Date Started**: November 23, 2025  
**Purpose**: Step-by-step guide to run the Pollu pollution modeling pipeline from scratch

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Pipeline Execution](#pipeline-execution)
5. [Validation](#validation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8+ (we're using Python 3.12.10)
- ~10GB disk space for data cache
- Internet connection (for Earth Engine, OpenAQ, Open-Meteo APIs)

### Required Accounts
- Google Earth Engine service account (optional, for satellite data)
- OpenAQ API token (optional, for ground truth PM2.5)

---

## Environment Setup

### Python Dependencies

The following packages are required (from `pyproject.toml`):

**Core Packages:**
- `numpy` (2.3.5) - Numerical computing
- `pandas` (2.3.3) - Data manipulation
- `geopandas` (1.1.1) - Geospatial data handling

**Geospatial Libraries:**
- `shapely` (2.1.2) - Geometric operations
- `pyproj` (3.7.2) - Coordinate reference system transformations
- `rasterio` (1.4.3) - Raster data I/O
- `rioxarray` (0.20.0) - Raster operations with xarray
- `pyogrio` (0.11.1) - Fast vector I/O
- `pyarrow` - Parquet file format support (required for GeoParquet)

**Machine Learning:**
- `scikit-learn` (1.7.2) - ML algorithms
- `mgwr` (2.2.1) - Geographically Weighted Regression
- `scipy` (1.16.3) - Scientific computing

**API & Storage:**
- `fastapi` (0.121.3) - REST API framework
- `uvicorn` (0.38.0) - ASGI server
- `sqlalchemy` (2.0.44) - Database ORM
- `duckdb` (1.4.2) - Embedded analytics database
- `earthengine-api` - Google Earth Engine Python API

**Utilities:**
- `requests` (2.32.5) - HTTP library
- `python-dotenv` (1.2.1) - Environment variable management

### Installation

```powershell
# Navigate to project directory
cd "c:\Users\user\Documents\New folder\pollu"

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -e .

# Install pyarrow for Parquet file support
pip install pyarrow

# Install Earth Engine API for satellite data
pip install earthengine-api
```

**What this does:**
- Creates an isolated Python environment in `.venv/`
- Installs all packages listed in `pyproject.toml`
- Installs the `pollu` package in editable mode (changes take effect immediately)

### Earth Engine Authentication

Before using satellite data, authenticate with Google Earth Engine:

```powershell
python -c "import ee; ee.Authenticate()"
```

**What this does:**
- Opens a browser window for Google authentication
- Saves credentials locally for future use
- Required only once per machine

---

## Data Preparation

### Step 1: Create 30m Grid for Mumbai

**File:** `scripts/create_grid_mumbai.py`

**Command:**
```powershell
python scripts/create_grid_mumbai.py
```

**What it does:**
1. Reads Mumbai ward boundaries from `resources/mumbai_wards.geojson` (WGS84/EPSG:4326)
2. Reprojects to UTM Zone 43N (EPSG:32643) for metric calculations
3. Creates a regular 30m × 30m grid covering the bounding box
4. Generates centroid points at the center of each cell (offset by 15m from edges)
5. Clips grid to Mumbai boundary using spatial intersection
6. Assigns sequential `grid_id` starting from 0
7. Saves as GeoParquet to `data/mumbai/grid_30m.parquet`

**Key Parameters:**
- `--boundary`: Input boundary file (default: `resources/mumbai_wards.geojson`)
- `--out`: Output parquet file (default: `data/mumbai/grid_30m.parquet`)
- `--crs`: Target CRS (default: `EPSG:32643`)
- `--res`: Grid resolution in meters (default: `30.0`)

**Expected output:**
```
Saved grid (509209 points) to data/mumbai/grid_30m.parquet
```

**Why we need this:**
All feature extraction (Landsat, Sentinel, SVF, OSM) will be sampled at these exact grid points to ensure perfect spatial alignment across all datasets.

---

### Step 2: Fetch Landsat-8 Image Dates

**File:** `scripts/fetch_landsat56.py`

**Prerequisites:**
- Earth Engine authentication completed (see [Earth Engine Authentication](#earth-engine-authentication))
- Google Earth Engine project ID

**Command:**
```powershell
python scripts/fetch_landsat56.py --project "fast-archive-465917-m0"
```

**What it does:**
1. Connects to Google Earth Engine with your project ID
2. Queries Landsat-8 Collection 2 Level-1 TOA imagery
3. Filters by Mumbai bounding box (72.7763°, 18.8939° to 72.9797°, 19.2701°)
4. Filters by date range (2018-01-01 to 2025-11-01)
5. Filters by cloud cover (< 15%)
6. Extracts unique image dates
7. Saves dates to `patliputra/landsat8_image_dates.txt`

**Key Parameters:**
- `--project`: Google Earth Engine project ID (required)
- `--output`: Output file path (default: `patliputra/landsat8_image_dates.txt`)
- `--bbox`: Bounding box as west,south,east,north (default: Mumbai)
- `--start-date`: Start date YYYY-MM-DD (default: `2018-01-01`)
- `--end-date`: End date YYYY-MM-DD (default: `2025-11-01`)
- `--max-cloud-cover`: Maximum cloud cover % (default: `15.0`)

**Expected output:**
```
Initializing Earth Engine with project: fast-archive-465917-m0
Saved XXX image dates to patliputra/landsat8_image_dates.txt
```

**Why we need this:**
This generates a list of all cloud-free Landsat-8 images available for Mumbai. These dates will be used to download and process the actual satellite imagery in the next step.

---

## Pipeline Execution

_[To be filled as we execute each step]_

---
