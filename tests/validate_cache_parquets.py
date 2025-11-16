#!/usr/bin/env python3
"""Validate GeoParquet outputs produced under cache/copernicus_* folders."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import pyarrow.parquet as pq

LOGGER = logging.getLogger("validate_cache_parquets")
DEFAULT_CACHE_ROOT = Path("cache")
DEFAULT_REQUIRED_COLUMNS = ("grid_id", "toa_b2", "raster_meta")


def iter_cache_dirs(cache_root: Path) -> List[Path]:
    dirs: List[Path] = []
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")
    for child in sorted(cache_root.iterdir()):
        if child.is_dir() and (
            child.name.startswith("copernicus_") or child.name.startswith("copetnicus_")
        ):
            dirs.append(child)
    if not dirs:
        raise RuntimeError(f"No copernicus directories found in {cache_root}")
    return dirs


def iter_parquet_files(cache_dirs: Sequence[Path]) -> Iterable[Path]:
    for folder in cache_dirs:
        found = sorted(folder.glob("*.parquet"))
        if not found:
            LOGGER.warning("No parquet files found inside %s", folder)
        for parquet_path in found:
            yield parquet_path


def validate_parquet(parquet_path: Path, required_columns: Sequence[str]) -> None:
    pf = pq.ParquetFile(parquet_path)
    schema_names = set(pf.schema.names)
    missing = [col for col in required_columns if col not in schema_names]
    if missing:
        raise ValueError(f"Missing columns {missing} in {parquet_path}")
    row_count = pf.metadata.num_rows if pf.metadata else "unknown"
    column_count = len(pf.schema.names)
    LOGGER.info("VALID %s rows=%s cols=%s", parquet_path, row_count, column_count)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cache/copernicus_* GeoParquet outputs"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Root directory that contains copernicus_* folders",
    )
    parser.add_argument(
        "--require-columns",
        nargs="*",
        default=list(DEFAULT_REQUIRED_COLUMNS),
        help="List of column names that must exist in every parquet",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failure instead of continuing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    cache_dirs = iter_cache_dirs(args.cache_root)
    failures: List[str] = []
    total = 0

    for parquet_path in iter_parquet_files(cache_dirs):
        total += 1
        try:
            validate_parquet(parquet_path, args.require_columns)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - want to log any failure and keep going
            failures.append(f"{parquet_path}: {exc}")
            LOGGER.error("FAILED %s: %s", parquet_path, exc)
            if args.fail_fast:
                break

    if failures:
        LOGGER.error(
            "Validation finished with %d failure(s) out of %d files",
            len(failures),
            total,
        )
        for failure in failures:
            LOGGER.error("  %s", failure)
        return 1

    LOGGER.info("Validation finished successfully for %d parquet file(s)", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
