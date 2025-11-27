
#!/usr/bin/env python3
"""Fetch OpenAQ PM2.5 readings for seeded sensors across Landsat image dates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

# Provenance and deterministic output helpers
PARQUET_ENGINE: str = "pyarrow"
MANIFEST_SUFFIX: str = ".manifest.json"

def get_git_sha() -> str:
    """Return current git commit SHA or 'unknown' if unavailable."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, encoding="utf-8").strip()
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha.lower()):
            return sha
        return "unknown"
    except Exception:
        return "unknown"

DEFAULT_DATES_PATH = Path("patliputra/landsat8_image_dates.txt")
DEFAULT_CACHE_ROOT = Path("cache")
DEFAULT_SENSOR_SEED = DEFAULT_CACHE_ROOT / "openaq_sensor_ids.parquet"
DEFAULT_OUTPUT = DEFAULT_CACHE_ROOT / "openaq_daily_pm25.parquet"
OPENAQ_BASE = "https://api.openaq.org/v3"
DEFAULT_TOKEN_ENV = "5881288ec6eee8a04c8029d8aa1ff991cf7a91c7d10537f8c3288ab5918700cc"
PM25_PARAMETER_ID = 2
DEFAULT_SLEEP_SECONDS = 1.0  # Increased to avoid rate limits
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 522, 524}
DEFAULT_MAX_SEARCH_DAYS = 7  # Bidirectional window search
DEFAULT_MIN_SAMPLE_COUNT = 3  # Minimum samples for valid PM2.5 reading
DEFAULT_SEASONAL_FALLBACK = True  # Try same date from previous/next year
DEFAULT_MAX_RETRIES = 5  # Max retry attempts for rate limit errors


@dataclass
class SensorMetadata:
    sensor_id: int
    sensor_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    first_date: dt.date
    last_date: dt.date


@dataclass
class FallbackResult:
    """Result of searching for PM2.5 data with fallback strategy."""

    value: Optional[float]
    sample_count: int
    actual_date: Optional[dt.date]  # Date where data was found
    fallback_type: str  # "exact", "window", "seasonal", or "none"
    days_offset: int  # Days from target date (0 for exact match)


def read_dates(path: Path = DEFAULT_DATES_PATH) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dates file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        dates = [line.strip() for line in fh if line.strip()]
    if not dates:
        raise ValueError(f"No dates found in {path}")
    return sorted(dates)


def _request_json(
    url: str,
    params: Optional[Dict],
    token: Optional[str],
    retry: int = 5,
    sleep_s: float = 2.0,
) -> Dict:
    headers = {"X-API-Key": token} if token else None
    attempt = 0
    while True:
        try:
            resp = requests.get(url, params=params or None, headers=headers, timeout=45)
        except requests.RequestException as exc:  # network hiccups
            if attempt >= retry:
                raise
            backoff = min(120.0, sleep_s * (2**attempt))  # Exponential backoff
            print(
                f"    Network error (attempt {attempt+1}/{retry}), retrying in {backoff:.1f}s..."
            )
            time.sleep(backoff)
            attempt += 1
            continue

        # Handle rate limiting and other retryable errors
        if resp.status_code in RETRYABLE_STATUS:
            if attempt >= retry:
                resp.raise_for_status()  # Raise on final attempt

            # Exponential backoff with jitter for rate limits
            if resp.status_code == 429:
                backoff = min(
                    120.0, sleep_s * (2**attempt) * 1.5
                )  # Extra delay for rate limits
                print(
                    f"    Rate limit hit (attempt {attempt+1}/{retry}), waiting {backoff:.1f}s..."
                )
            else:
                backoff = min(120.0, sleep_s * (2**attempt))
                print(
                    f"    HTTP {resp.status_code} (attempt {attempt+1}/{retry}), retrying in {backoff:.1f}s..."
                )

            time.sleep(backoff)
            attempt += 1
            continue

        resp.raise_for_status()
        payload = resp.json()
        if "results" not in payload:
            raise RuntimeError(f"Unexpected OpenAQ payload: {payload}")
        return payload


def load_sensor_ids_from_parquet(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Seed parquet not found: {path}")
    df = pd.read_parquet(path)
    ids = sorted(set(int(v) for v in df["sensor_id"].tolist()))
    if not ids:
        raise RuntimeError(f"No sensor_ids present in {path}")
    return ids


def _parse_iso_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_sensor_metadata(token: Optional[str], sensor_id: int) -> SensorMetadata:
    payload = _request_json(
        f"{OPENAQ_BASE}/sensors/{sensor_id}",
        params={},
        token=token,
    )
    results = payload.get("results", [])
    if not results:
        raise RuntimeError(f"No metadata returned for sensor {sensor_id}")
    entry = results[0]
    first = _parse_iso_datetime(entry.get("datetimeFirst", {}).get("utc"))
    last = _parse_iso_datetime(entry.get("datetimeLast", {}).get("utc"))
    if not first or not last:
        raise RuntimeError(f"Sensor {sensor_id} missing first/last timestamps")
    latest = entry.get("latest") or {}
    coordinates = latest.get("coordinates") or entry.get("coordinates") or {}
    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")
    return SensorMetadata(
        sensor_id=int(entry["id"]),
        sensor_name=str(entry.get("name") or "pm25_sensor"),
        latitude=float(lat) if lat is not None else None,
        longitude=float(lon) if lon is not None else None,
        first_date=first.date(),
        last_date=last.date(),
    )


def eligible_dates_for_sensor(
    dates: List[dt.date], first: dt.date, last: dt.date
) -> List[dt.date]:
    return [d for d in dates if first <= d <= last]


def fetch_daily_pm25_with_fallback(
    token: Optional[str],
    sensor_id: int,
    target_date: dt.date,
    max_search_days: int = DEFAULT_MAX_SEARCH_DAYS,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
    enable_seasonal: bool = DEFAULT_SEASONAL_FALLBACK,
) -> Tuple[FallbackResult, Optional[str], Optional[str]]:
    """
    Fetch PM2.5 with bidirectional window search and seasonal fallback.

    Returns:
        (FallbackResult, min_ts_iso_or_None, max_ts_iso_or_None)
        - FallbackResult: fallback metadata/result
        - min_ts_iso_or_None: earliest UTC ISO timestamp for chosen date, or None
        - max_ts_iso_or_None: latest UTC ISO timestamp for chosen date, or None
    """
    # Try exact date first
    value, sample_count, min_ts, max_ts = fetch_daily_pm25_average(token, sensor_id, target_date)
    # Map (value, sample_count, min_ts, max_ts) → FallbackResult and propagate timestamps
    if value is not None and sample_count >= min_sample_count:
        return (
            FallbackResult(
                value=value,
                sample_count=sample_count,
                actual_date=target_date,
                fallback_type="exact",
                days_offset=0,
            ),
            min_ts,
            max_ts,
        )

    # Bidirectional window search: ±1, ±2, ..., ±max_search_days
    for offset in range(1, max_search_days + 1):
        # Try earlier date first (prefer past data)
        for sign in [-1, 1]:
            candidate_date = target_date + dt.timedelta(days=sign * offset)
            value, sample_count, min_ts, max_ts = fetch_daily_pm25_average(
                token, sensor_id, candidate_date
            )
            # Map (value, sample_count, min_ts, max_ts) → FallbackResult and propagate timestamps
            if value is not None and sample_count >= min_sample_count:
                return (
                    FallbackResult(
                        value=value,
                        sample_count=sample_count,
                        actual_date=candidate_date,
                        fallback_type="window",
                        days_offset=sign * offset,
                    ),
                    min_ts,
                    max_ts,
                )

    # Seasonal fallback: try same date from previous/next year
    if enable_seasonal:
        for year_offset in [-1, 1]:
            try:
                seasonal_date = target_date.replace(year=target_date.year + year_offset)
                value, sample_count, min_ts, max_ts = fetch_daily_pm25_average(
                    token, sensor_id, seasonal_date
                )
                # Map (value, sample_count, min_ts, max_ts) → FallbackResult and propagate timestamps
                if value is not None and sample_count >= min_sample_count:
                    days_diff = (seasonal_date - target_date).days
                    return (
                        FallbackResult(
                            value=value,
                            sample_count=sample_count,
                            actual_date=seasonal_date,
                            fallback_type="seasonal",
                            days_offset=days_diff,
                        ),
                        min_ts,
                        max_ts,
                    )
            except ValueError:
                # Handle Feb 29 edge case
                continue

    # No valid data found
    return (
        FallbackResult(
            value=None,
            sample_count=0,
            actual_date=None,
            fallback_type="none",
            days_offset=0,
        ),
        None,
        None,
    )


def fetch_daily_pm25_average(
    token: Optional[str], sensor_id: int, date_obj: dt.date
) -> Tuple[Optional[float], int, Optional[str], Optional[str]]:
    """
    Fetch daily PM2.5 average for a sensor and date.
    Returns (average_value_or_None, sample_count, min_timestamp_iso_or_None, max_timestamp_iso_or_None)
    - average_value_or_None: float or None if no valid values
    - sample_count: int, number of valid PM2.5 samples
    - min_timestamp_iso_or_None: earliest UTC ISO timestamp (str) or None
    - max_timestamp_iso_or_None: latest UTC ISO timestamp (str) or None
    """
    dt_from = f"{date_obj.isoformat()}T00:00:00Z"
    next_day = date_obj + dt.timedelta(days=1)
    dt_to = f"{next_day.isoformat()}T00:00:00Z"
    limit = 200
    page = 1
    values: List[float] = []
    ts_list: List[str] = []
    while True:
        params = {
            "datetime_from": dt_from,
            "datetime_to": dt_to,
            "limit": limit,
            "page": page,
        }
        payload = _request_json(
            f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
            params,
            token,
        )
        results = payload.get("results", [])
        for entry in results:
            parameter_info = entry.get("parameter") or {}
            param_id = parameter_info.get("id")
            param_name = parameter_info.get("name")
            if param_id is not None and int(param_id) != PM25_PARAMETER_ID:
                continue
            if param_name and param_name.lower() not in {"pm25", "pm2.5"}:
                continue
            value = entry.get("value")
            if value is None:
                continue
            # Robust timestamp extraction
            ts: Optional[str] = None
            date_field = entry.get("date")
            if isinstance(date_field, dict):
                ts = date_field.get("utc") or date_field.get("local")
            elif isinstance(date_field, str):
                ts = date_field
            if ts:
                dt_obj = _parse_iso_datetime(ts)
                if dt_obj:
                    ts_list.append(dt_obj.isoformat())
            try:
                float_value = float(value)
                # Reject negative PM2.5 values (physically impossible)
                if float_value < 0:
                    continue
                values.append(float_value)
            except Exception:
                continue
        if len(results) < limit:
            break
        page += 1
    if not values:
        return None, 0, None, None
    avg = sum(values) / len(values)
    min_ts = min(ts_list) if ts_list else None
    max_ts = max(ts_list) if ts_list else None
    return avg, len(values), min_ts, max_ts


def process(
    token: Optional[str],
    dates_path: Path,
    sensor_seed_path: Path,
    output_path: Path,
    sleep_seconds: float,
    max_search_days: int = DEFAULT_MAX_SEARCH_DAYS,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
    enable_seasonal: bool = DEFAULT_SEASONAL_FALLBACK,
) -> None:
    date_strings = read_dates(dates_path)
    date_objs = [dt.date.fromisoformat(ds) for ds in date_strings]
    sensor_ids = load_sensor_ids_from_parquet(sensor_seed_path)
    print(f"Loaded {len(sensor_ids)} unique sensors from {sensor_seed_path}")

    sensor_metas: List[SensorMetadata] = []
    for idx, sensor_id in enumerate(sensor_ids, start=1):
        try:
            meta = fetch_sensor_metadata(token, sensor_id)
            sensor_metas.append(meta)
            print(
                f"[{idx}/{len(sensor_ids)}] Sensor {sensor_id}"
                f" window {meta.first_date}→{meta.last_date}"
            )
        except Exception as exc:  # log but continue so other sensors still process
            print(f"Failed to load metadata for sensor {sensor_id}: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    # Track fallback statistics
    stats = {
        "exact": 0,
        "window": 0,
        "seasonal": 0,
        "none": 0,
    }

    # Provenance and reproducibility fields
    GIT_SHA = get_git_sha()
    fetched_at_utc = dt.datetime.utcnow().isoformat() + "Z"

    for sensor in sensor_metas:
        valid_dates = eligible_dates_for_sensor(
            date_objs, sensor.first_date, sensor.last_date
        )
        if not valid_dates:
            continue
        print(
            f"Sensor {sensor.sensor_id} has {len(valid_dates)} matching Landsat dates"
        )
        for date_obj in valid_dates:
            result, min_ts, max_ts = fetch_daily_pm25_with_fallback(
                token,
                sensor.sensor_id,
                date_obj,
                max_search_days=max_search_days,
                min_sample_count=min_sample_count,
                enable_seasonal=enable_seasonal,
            )

            stats[result.fallback_type] += 1

            # Log fallback info
            if result.fallback_type == "exact":
                fallback_info = "exact match"
            elif result.fallback_type == "window":
                fallback_info = f"window fallback (±{abs(result.days_offset)}d, actual: {result.actual_date})"
            elif result.fallback_type == "seasonal":
                fallback_info = f"seasonal fallback (year ±{abs(result.days_offset)//365}, actual: {result.actual_date})"
            elif result.fallback_type == "none":
                fallback_info = "no data found (will need interpolation)"
            else:
                fallback_info = "unknown"

            rows.append(
                {
                    "date": date_obj.isoformat(),
                    "sensor_id": int(sensor.sensor_id),
                    "sensor_name": str(sensor.sensor_name),
                    "sensor_provider": "openaq",
                    "latitude": float(sensor.latitude) if sensor.latitude is not None else None,
                    "longitude": float(sensor.longitude) if sensor.longitude is not None else None,
                    "first_date": sensor.first_date.isoformat(),
                    "last_date": sensor.last_date.isoformat(),
                    "pm25": float(result.value) if result.value is not None else None,
                    "aggregation_method": "mean",
                    "sample_count": int(result.sample_count),
                    "measurement_ts_min_utc": min_ts,
                    "measurement_ts_max_utc": max_ts,
                    "actual_date": result.actual_date.isoformat() if result.actual_date else None,
                    "fallback_type": result.fallback_type,
                    "days_offset": int(result.days_offset),
                    "fetched_at_utc": fetched_at_utc,
                    "_provenance": json.dumps({"script": "scripts/fetch_openaq.py", "git_sha": GIT_SHA, "openaq_base": OPENAQ_BASE}),
                    "label_distance_m": None,
                    "sensor_quality_flag": "ok" if result.sample_count >= min_sample_count and result.value is not None else "low_coverage",
                    "notes": None,
                }
            )
            print(
                f"  {date_obj} sensor={sensor.sensor_id}"
                f" pm25={result.value if result.value is not None else 'None'}"
                f" samples={result.sample_count}"
                f" ({fallback_info})"
            )
            time.sleep(max(0.0, sleep_seconds))

    if not rows:
        raise RuntimeError("No sensor/date combinations produced any rows")


    df = pd.DataFrame(rows)
    # Enforce correct dtypes
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "actual_date" in df.columns:
        df["actual_date"] = pd.to_datetime(df["actual_date"]).dt.date
    df["pm25"] = df["pm25"].astype("float32")
    df["sample_count"] = df["sample_count"].astype("int16")
    df["days_offset"] = df["days_offset"].astype("int16")
    df["sensor_id"] = df["sensor_id"].astype("int32")
    df.sort_values(["sensor_id", "date"], inplace=True)
    df.to_parquet(output_path, engine=PARQUET_ENGINE, index=False)

    # Write manifest JSON
    manifest = {
        "created_at_utc": fetched_at_utc,
        "script": "scripts/fetch_openaq.py",
        "git_sha": GIT_SHA,
        "num_rows": len(df),
        "source_api": OPENAQ_BASE,
        "notes": None,
    }
    manifest_path = output_path.with_suffix(MANIFEST_SUFFIX)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # (Optional) Partitioning: To partition by 'date', use pyarrow or pandas to write partitioned parquet files.
    # This is useful for large datasets and enables efficient reads by date, but is not enabled by default here.
    # Example: df.to_parquet(output_path, partition_cols=["date"], engine=PARQUET_ENGINE)
    # Only enable if the caller passes a flag for partitioning.

    # Print fallback statistics
    total_attempts = sum(stats.values())
    print("\n=== Fallback Statistics ===")
    print(f"Total date attempts: {total_attempts}")
    print(
        f"  Exact matches: {stats['exact']} ({100*stats['exact']/total_attempts:.1f}%)"
    )
    print(
        f"  Window fallbacks: {stats['window']} ({100*stats['window']/total_attempts:.1f}%)"
    )
    print(
        f"  Seasonal fallbacks: {stats['seasonal']} ({100*stats['seasonal']/total_attempts:.1f}%)"
    )
    print(f"  No data found: {stats['none']} ({100*stats['none']/total_attempts:.1f}%)")
    print(
        f"\nSuccessful retrievals: {len(df)}/{total_attempts} ({100*len(df)/total_attempts:.1f}%)"
    )
    print(
        f"\nWrote {len(df)} sensor-date PM2.5 rows"
        f" ({len(sensor_metas)} sensors processed) to {output_path}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch OpenAQ PM2.5 measurements for seeded sensors across Landsat dates"
        ),
    )
    parser.add_argument(
        "--token",
        default=os.getenv(DEFAULT_TOKEN_ENV),
        help="OpenAQ API token (default: read from OPENAQ_TOKEN env)",
    )
    parser.add_argument(
        "--dates-path",
        type=Path,
        default=DEFAULT_DATES_PATH,
        help="Path to Landsat date list",
    )
    parser.add_argument(
        "--sensor-seed",
        type=Path,
        default=DEFAULT_SENSOR_SEED,
        help="Seed parquet containing available sensor IDs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Parquet file to write sensor-wide PM2.5 values",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause between sensor requests to avoid rate limits (default: 1.0s)",
    )
    parser.add_argument(
        "--max-search-days",
        type=int,
        default=DEFAULT_MAX_SEARCH_DAYS,
        help="Maximum days to search bidirectionally for PM2.5 data (default: 7)",
    )
    parser.add_argument(
        "--min-sample-count",
        type=int,
        default=DEFAULT_MIN_SAMPLE_COUNT,
        help="Minimum samples required for valid PM2.5 reading (default: 3)",
    )
    parser.add_argument(
        "--no-seasonal-fallback",
        action="store_true",
        help="Disable seasonal fallback (same date from previous/next year)",
    )
    args = parser.parse_args(argv)

    process(
        token=args.token,
        dates_path=args.dates_path,
        sensor_seed_path=args.sensor_seed,
        output_path=args.output,
        sleep_seconds=args.sleep_seconds,
        max_search_days=args.max_search_days,
        min_sample_count=args.min_sample_count,
        enable_seasonal=not args.no_seasonal_fallback,
    )


if __name__ == "__main__":
    main()
