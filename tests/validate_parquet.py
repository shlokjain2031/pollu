#!/usr/bin/env python3
"""
Simple validator for a GeoParquet / Parquet produced by the landsat8_signals module

Loads the parquet (via GeoPandas if available, else pandas) and prints
three random rows along with a short summary.

Usage:
    python3 scripts/validate_parquet.py /tmp/landsat8.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import traceback
from typing import Iterable, Sequence

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency
    gpd = None

import pandas as pd
import numpy as np

S5P_DEFAULT_PATH = Path("cache/landsat_2025-03-25.parquet")
S5P_REQUIRED_COLUMNS = ("s5p_no2", "s5p_no2_is_nodata", "s5p_no2_meta")

try:
    from tabulate import tabulate  # optional nice table formatter
except Exception:
    tabulate = None


def load_parquet(path: Path):
    """Try GeoPandas first, fall back to pandas."""
    if gpd is not None:
        try:
            gdf = gpd.read_parquet(path)
            return gdf
        except Exception:
            # fall through to pandas
            pass

    # pandas fallback
    return pd.read_parquet(path)


def has_columns(path: Path, columns: Sequence[str]) -> bool:
    """Return True if every column exists in the parquet schema."""
    df = pd.read_parquet(path, columns=list(columns))
    return all(col in df.columns for col in columns)


def report_s5p_status(target_path: Path = S5P_DEFAULT_PATH) -> None:
    if not target_path.exists():
        print(f"S5P check: {target_path} is missing")
        return
    try:
        present = has_columns(target_path, S5P_REQUIRED_COLUMNS)
    except Exception as exc:
        print(f"S5P check: failed to inspect {target_path}: {exc}")
        return
    if present:
        print(
            f"S5P check: {target_path.name} contains columns {', '.join(S5P_REQUIRED_COLUMNS)}"
        )
    else:
        print(
            f"S5P check: {target_path.name} is missing one of {', '.join(S5P_REQUIRED_COLUMNS)}"
        )


def main():
    p = argparse.ArgumentParser(description="Validate a landsat parquet file")
    p.add_argument("path", nargs="?", default="/tmp/landsat8.parquet")
    p.add_argument(
        "-n", "--nrows", type=int, default=3, help="Number of random rows to print"
    )
    p.add_argument(
        "--s5p-path",
        type=Path,
        default=S5P_DEFAULT_PATH,
        help="Path to landsat parquet that should include S5P NO2 columns",
    )
    p.add_argument(
        "--no-s5p-check",
        action="store_true",
        help="Disable the S5P NO2 column presence check",
    )

    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    if not args.no_s5p_check:
        report_s5p_status(args.s5p_path)

    try:
        obj = load_parquet(path)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        traceback.print_exc()
        sys.exit(3)

    # Basic summary
    try:
        n = len(obj)
    except Exception:
        n = None
    print("\n== Parquet summary ==")
    print(f"Path: {path}")
    print(f"Type: {type(obj).__name__}")
    print(f"Rows: {n}")

    # Columns and dtypes
    try:
        cols = list(obj.columns)
        print("Columns:")
        # Show name -> dtype
        try:
            dtypes = obj.dtypes.to_dict()
            for c in cols:
                print(f"  - {c}: {dtypes.get(c)}")
        except Exception:
            print("  ", cols)
    except Exception:
        print("Could not list columns")

    # Null counts for key columns
    try:
        print("\nNull counts (top 10 columns):")
        nulls = obj.isnull().sum()
        nulls = nulls.sort_values(ascending=False)
        for name, val in nulls.iloc[:10].items():
            print(f"  {name}: {int(val)}")
    except Exception:
        pass

    # Print random sample rows (pretty)
    try:
        sample_n = min(args.nrows, n if n is not None else args.nrows)
        if sample_n <= 0:
            print("No rows to sample")
            return

        # Prepare a printable DataFrame
        if gpd is not None and isinstance(obj, gpd.GeoDataFrame):
            sample_df = obj.sample(n=sample_n).copy()
            # Convert geometry to WKT and truncate for readability
            try:
                sample_df["geometry"] = sample_df.geometry.apply(
                    lambda g: (
                        (g.wkt[:120] + "...")
                        if g is not None and len(g.wkt) > 120
                        else (g.wkt if g is not None else None)
                    )
                )
            except Exception:
                pass
        else:
            sample_df = obj.sample(n=sample_n)

        # Format floats to 4 decimal places for display
        fmt_df = sample_df.copy()
        float_cols = fmt_df.select_dtypes(include=[np.floating]).columns.tolist()
        for c in float_cols:
            fmt_df[c] = fmt_df[c].map(
                lambda v: (f"{v:.4f}" if (v is not None and not pd.isna(v)) else "")
            )

        print("\n== Sample rows ==")
        if tabulate is not None:
            print(tabulate(fmt_df, headers="keys", tablefmt="github", showindex=False))
        else:
            print(fmt_df.to_string(index=False))
    except Exception as e:
        print("Failed to sample rows:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
