"""Patliputra ingestion orchestrator.

This module coordinates signal providers for a city and date range. It
is designed to be reusable (not a one-off script) and can be imported and
used in notebooks, servers, or CI.

High-level behavior:
 - load canonical grid (GeoParquet)
 - for each provider: try load_cache(); if not present, fetch_raw() -> preprocess()
 - sample provider artifact at grid centroids and join result
 - write out a feature-enhanced Parquet (atomic write)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def run_providers_on_grid(
    grid_path: Path,
    providers: Iterable,
    out_path: Path = None,
    start: str = None,
    end: str = None,
    overwrite: bool = False,
):
    """Run a list of provider instances on the given grid.

    providers: iterable of SignalProvider instances
    grid_path: path to canonical city grid GeoParquet
    out_path: where to write enhanced grid (if None, write back to grid_path)
    start/end: optional date strings passed to providers
    overwrite: if False, will not overwrite existing out_path
    """
    grid_path = Path(grid_path)
    if out_path is None:
        out_path = grid_path
    else:
        out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        raise RuntimeError(f"Out path {out_path} exists; set overwrite=True to replace")

    grid = pd.read_parquet(grid_path)

    # Ensure we have grid_id column
    if "grid_id" not in grid.columns:
        raise RuntimeError("grid must contain grid_id column")

    for provider in providers:
        print(f"Processing provider: {provider.__class__.__name__}")
        processed = provider.load_cache()
        if processed is None:
            raw = provider.fetch_raw(start=start, end=end)
            processed = provider.preprocess(raw)

        sampled = provider.sample_to_grid(processed, grid)
        # join on grid_id index
        grid = grid.merge(sampled, left_on="grid_id", right_index=True, how="left")

    # atomic write
    tmp = out_path.with_suffix(".tmp.parquet")
    grid.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    print(f"Wrote enhanced grid to {out_path}")
    return out_path


def run_ingestion():
    print("Running Patliputra ingestion...")


if __name__ == "__main__":
    run_ingestion()
