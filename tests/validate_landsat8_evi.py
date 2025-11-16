#!/usr/bin/env python3
"""Validate Landsat8 parquet files for Sentinel-2 EVI consistency."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVI_VALUE_COLUMN = "sentinel2_evi"
EVI_FLAG_COLUMN = "sentinel2_evi_is_nodata"
GRID_COLUMN = "grid_id"


class ValidationIssue(Exception):
    """Raised when a validation rule fails."""


def _parse_date_from_name(path: Path) -> dt.date:
    stem = path.stem  # e.g. landsat_2020-01-01
    if "_" not in stem:
        raise ValueError(f"Cannot parse date from filename: {path}")
    date_str = stem.split("_", 1)[1]
    try:
        return dt.date.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(f"Filename does not contain an ISO date: {path}") from exc


def _load_evi_columns(path: Path) -> Tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Landsat parquet missing: {path}")
    df = pd.read_parquet(path, columns=[GRID_COLUMN, EVI_VALUE_COLUMN, EVI_FLAG_COLUMN])
    missing = [
        c
        for c in (GRID_COLUMN, EVI_VALUE_COLUMN, EVI_FLAG_COLUMN)
        if c not in df.columns
    ]
    if missing:
        raise ValidationIssue(f"{path} missing columns: {missing}")
    df = df.sort_values(GRID_COLUMN)
    values = df.set_index(GRID_COLUMN)[EVI_VALUE_COLUMN].astype("float64")
    flags = df.set_index(GRID_COLUMN)[EVI_FLAG_COLUMN].astype("bool")
    return values, flags


def _ensure_data_present(values: pd.Series, flags: pd.Series, path: Path) -> None:
    valid_mask = (~flags) & values.notna()
    if not bool(valid_mask.any()):
        raise ValidationIssue(f"{path} has no valid Sentinel-2 EVI pixels (all nodata)")


def _compare_monthly_series(
    current_values: pd.Series,
    current_flags: pd.Series,
    reference_values: pd.Series,
    reference_flags: pd.Series,
    tolerance: float,
    month_label: str,
    current_path: Path,
    reference_path: Path,
) -> None:
    current_values = current_values.reindex(reference_values.index)
    current_flags = current_flags.reindex(reference_flags.index).fillna(True)

    # Any mismatch in nodata masks indicates inconsistent sampling.
    nodata_mismatch = reference_flags != current_flags
    if bool(nodata_mismatch.any()):
        count = int(nodata_mismatch.sum())
        raise ValidationIssue(
            f"{current_path} nodata mask diverges from {reference_path} for {count} pixels in {month_label}"
        )

    valid_mask = (~reference_flags) & (~current_flags)
    if not bool(valid_mask.any()):
        raise ValidationIssue(
            f"{current_path} and {reference_path} do not share any valid pixels for {month_label}"
        )
    diff = np.abs(reference_values[valid_mask] - current_values[valid_mask])
    max_diff = float(np.nanmax(diff)) if len(diff) else 0.0
    if max_diff > tolerance:
        raise ValidationIssue(
            f"{current_path} EVI differs from {reference_path} in {month_label} (max diff={max_diff:.3e})"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Landsat parquet files with Sentinel-2 EVI columns. "
            "Ensures monthly parity and data availability."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache"),
        help="Directory containing landsat_YYYY-MM-DD.parquet files",
    )
    parser.add_argument(
        "--pattern",
        default="landsat_*.parquet",
        help="Glob pattern relative to cache root to select Landsat parquets",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute EVI difference for same-month files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files to evaluate (useful for debugging)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cache_root = args.cache_root
    files = sorted(cache_root.glob(args.pattern))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print(
            f"No files matched pattern '{args.pattern}' under {cache_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    month_refs: Dict[Tuple[int, int], Tuple[pd.Series, pd.Series, Path]] = {}
    issues: List[str] = []

    for path in files:
        try:
            date_val = _parse_date_from_name(path)
        except Exception as exc:
            issues.append(f"{path}: {exc}")
            continue
        key = (date_val.year, date_val.month)
        month_label = f"{date_val.year:04d}-{date_val.month:02d}"
        try:
            values, flags = _load_evi_columns(path)
            _ensure_data_present(values, flags, path)
        except Exception as exc:
            issues.append(str(exc))
            continue

        if key not in month_refs:
            month_refs[key] = (values, flags, path)
            continue

        ref_values, ref_flags, ref_path = month_refs[key]
        try:
            _compare_monthly_series(
                current_values=values,
                current_flags=flags,
                reference_values=ref_values,
                reference_flags=ref_flags,
                tolerance=args.tolerance,
                month_label=month_label,
                current_path=path,
                reference_path=ref_path,
            )
        except ValidationIssue as exc:
            issues.append(str(exc))

    if issues:
        print("EVI validation failed:", file=sys.stderr)
        for issue in issues:
            print(f" - {issue}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Validated {len(files)} Landsat files across {len(month_refs)} months with tolerance {args.tolerance}."
    )


if __name__ == "__main__":
    main()
