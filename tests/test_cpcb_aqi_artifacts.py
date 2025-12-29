"""Consumer tests for CPCB AQI artifacts.

Ensures the generated parquet files follow the expected contract so downstream
consumers can rely on the schema and basic quality guarantees.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ARTIFACT_DIR = Path("artifacts") / "v1" / "aqi"


def _list_artifacts() -> list[Path]:
    return sorted(ARTIFACT_DIR.glob("*.parquet"))


def test_artifacts_exist():
    files = _list_artifacts()
    assert files, "No AQI artifacts found under artifacts/v1/aqi"


def test_artifact_schema_and_quality():
    files = _list_artifacts()
    assert files, "No AQI artifacts available for schema validation"

    # Sample up to 5 artifacts across the range
    sample_files = [files[0], files[len(files) // 2], files[-1]]
    if len(files) > 3:
        sample_files.extend(random.sample(files, k=min(2, len(files) - 3)))
    sample_files = sorted(set(sample_files))

    required_columns = {"sensor_id", "timestamp", "pm25"}

    for parquet_path in sample_files:
        df = pd.read_parquet(parquet_path)
        assert set(df.columns) == required_columns, (
            f"Unexpected columns in {parquet_path.name}: {df.columns}"
        )
        assert df["sensor_id"].dtype == "object", "sensor_id must be string"
        assert pd.api.types.is_datetime64_ns_dtype(df["timestamp"]), (
            "timestamp must be datetime64[ns]"
        )
        assert pd.api.types.is_float_dtype(df["pm25"]), "pm25 must be float"

        # Basic quality checks
        assert not df["sensor_id"].isna().any(), (
            f"sensor_id contains NA in {parquet_path.name}"
        )
        assert not df["timestamp"].isna().any(), (
            f"timestamp contains NA in {parquet_path.name}"
        )
        assert not df["pm25"].isna().any(), (
            f"pm25 contains NA in {parquet_path.name}"
        )
        assert (df["pm25"] >= 0).all(), (
            f"pm25 has negative values in {parquet_path.name}"
        )
        assert df["timestamp"].is_monotonic_increasing or df["timestamp"].is_monotonic_decreasing, (
            f"timestamp not sorted in {parquet_path.name}"
        )
        assert df["sensor_id"].nunique() > 0, (
            f"No sensor ids found in {parquet_path.name}"
        )

