"""Light Earth Engine client wrapper.

This module provides a minimal wrapper around the `ee` Python package.
It is intentionally small: we only initialize EE and provide helper
functions to build an Image (or collection) object and return either the
ee.Image or a download URL. The wrapper raises friendly errors when the
EE SDK is missing or not authenticated.

Note: actual downloading from EE (getDownloadURL + HTTP GET) is left to
the caller or provider; here we return either ee.Image objects or URLs so
implementations can decide how to persist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


def _ensure_ee_imported():
    try:
        import ee
    except Exception as e:
        raise ImportError(
            "Earth Engine Python API is not installed. Install with `pip install earthengine-api` and authenticate (`earthengine authenticate`) before using the Earth Engine backend.`"
        ) from e
    return ee


@dataclass
class EarthEngineClient:
    """Minimal EE client wrapper.

    Usage:
        ee_client = EarthEngineClient()
        img = ee_client.landsat_toa_image(collection='LANDSAT/LC08/C01/T1_TOA', bbox=bbox, start='2020-01-01', end='2020-01-10')
    """

    initialized: bool = False

    def initialize(self, force: bool = False) -> None:
        ee = _ensure_ee_imported()
        if not self.initialized or force:
            try:
                ee.Initialize()
                self.initialized = True
            except Exception as e:
                # Provide helpful message about authentication
                raise RuntimeError(
                    "Failed to initialize Earth Engine. Ensure you have run `earthengine authenticate` and have network connectivity. Original error: "
                    + str(e)
                ) from e

    def landsat_toa_image(
        self,
        collection: str,
        bbox: Optional[Dict[str, float]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        reducer: str = "median",
    ):
        """Return an ee.Image representing the TOA mosaic for the requested range.

        - collection: EE collection id (e.g., 'LANDSAT/LC08/C01/T1_TOA')
        - bbox: dict with keys minx,miny,maxx,maxy in lon/lat (EPSG:4326)
        - start,end: ISO date strings
        - reducer: 'median'|'mean' etc.

        Returns: ee.Image
        """
        ee = _ensure_ee_imported()
        self.initialize()

        col = ee.ImageCollection(collection)
        if start and end:
            col = col.filterDate(start, end)
        if bbox is not None:
            geom = ee.Geometry.Rectangle(
                [bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]]
            )
            col = col.filterBounds(geom)

        if reducer == "median":
            img = col.median()
        elif reducer == "mean":
            img = col.mean()
        else:
            img = col.median()

        return img

    def get_download_url(self, img, params: Optional[Dict[str, Any]] = None) -> str:
        """Return a download URL for an ee.Image using getDownloadURL parameters.

        Callers should follow the EE docs for allowed parameters. This method
        simply wraps ee.Image.getDownloadURL.
        """
        ee = _ensure_ee_imported()
        self.initialize()
        params = params or {}
        try:
            url = img.getDownloadURL(params)
            return url
        except Exception as e:
            raise RuntimeError(
                "Failed to obtain download URL from Earth Engine: " + str(e)
            ) from e
