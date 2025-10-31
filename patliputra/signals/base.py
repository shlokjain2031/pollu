"""Base interface for a signal provider.

Providers implement a small, testable surface:
 - fetch_raw(): obtain raw data (download or read local) and return a path or object
 - preprocess(raw): convert raw to a processed artifact (raster path or array)
 - sample_to_grid(processed, grid_gdf): sample processed data at grid centroids and
   return a DataFrame/array of values aligned to the grid
 - save_cache(processed) / load_cache()

Providers should not hard-code city-specific paths; instead accept a `city_id`
and `cache_dir` and keep outputs namespaced under cache_dir/city_id/signal/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass
class SignalProvider:
    """Minimal provider contract for Patliputra signals.

    Subclasses must implement the abstract methods below. Keep implementations
    small and focused on a single signal (e.g., TOA_B2). The pipeline will call
    these methods in order.
    """

    city_id: str
    cache_dir: Path
    config: Dict[str, Any]

    def fetch_raw(self, start: Optional[str] = None, end: Optional[str] = None) -> Any:
        """Fetch raw data for the requested time range.

        Return a backend-specific raw object (path, dict, etc.).
        """
        raise NotImplementedError()

    def preprocess(self, raw: Any) -> Path:
        """Convert raw data into a processed artifact suitable for sampling.

        For raster signals this should return a Path to a single-band raster
        file (in a supported format) in a consistent CRS. Implementations
        should write into `self.cache_dir / city_id / signal_name / ...`.
        """
        raise NotImplementedError()

    def sample_to_grid(self, processed: Path, grid_gdf) -> pd.DataFrame:
        """Sample a processed artifact at grid centroids and return a table
        with the sampled values aligned to grid index.
        """
        raise NotImplementedError()

    def save_cache(self, processed: Path) -> None:
        """Optional: move processed artifact to a durable cache location."""
        pass

    def load_cache(self) -> Optional[Path]:
        """Optional: return a cached processed artifact if present."""
        return None
