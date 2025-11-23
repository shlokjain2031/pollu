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
import time
import logging


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
    project_id: str = "fast-archive-465917-m0"
    service_account: Optional[str] = (
        "pollu-earth-engine-client@fast-archive-465917-m0.iam.gserviceaccount.com"
    )
    key_path: Optional[str] = (
        "/Users/shlokjain/pollu/fast-archive-465917-m0-3e5edc46c27a.json"
    )

    def initialize(self, force: bool = False) -> None:
        ee = _ensure_ee_imported()
        if self.initialized and not force:
            return

        try:
            # Prefer explicit service account credentials if available
            # if self.service_account and os.path.exists(self.key_path):
            #     credentials = ee.ServiceAccountCredentials(self.service_account, self.key_path)
            #     ee.Initialize(credentials, project=self.project_id)
            # else:
            #     ee.Initialize(project=self.project_id)
            ee.Initialize(project=self.project_id)

            # --- Critical override to stop 517222506229 fallback ---
            if hasattr(ee.data, "setCloudProject"):
                ee.data.setCloudProject(self.project_id)
            else:
                ee.data._cloud_api_resource_prefix = f"projects/{self.project_id}"

            self.initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Earth Engine for project {self.project_id}: {e}"
            ) from e

    def landsat_toa_image(
        self,
        collection: str,
        bbox: Optional[Dict[str, float]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        reducer: str = "median",
        bands: Optional[list] = None,
    ):
        """Return an ee.Image representing the TOA mosaic for the requested range.

        - collection: EE collection id (e.g., 'LANDSAT/LC08/C01/T1_TOA')
        - bbox: dict with keys minx,miny,maxx,maxy in lon/lat (EPSG:4326) or an ee.Geometry
        - start,end: ISO date strings
        - reducer: 'median'|'mean' (applied to the collection)

        Returns: ee.Image
        """
        ee = _ensure_ee_imported()
        self.initialize()

        col = ee.ImageCollection(collection)
        if start and end:
            col = col.filterDate(start, end)

        geom = None
        if bbox is not None:
            # accept either a dict bbox or an ee.Geometry
            if isinstance(bbox, dict):
                geom = ee.Geometry.Rectangle(
                    [bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]]
                )
            else:
                geom = bbox
            col = col.filterBounds(geom)

        if bands:
            col = col.select(bands)

        # Ensure there is at least one image
        try:
            count = int(col.size().getInfo())
        except Exception:
            # If getInfo fails, fall back to first() check
            count = None

        if count == 0:
            raise RuntimeError(
                f"No images found in collection {collection} for the requested date range and region."
            )

        if reducer == "median":
            img = col.median()
        elif reducer == "mean":
            img = col.mean()
        else:
            img = col.median()

        if geom is not None:
            img = img.clip(geom)

        return img

    def get_download_url(self, img, params: Optional[Dict[str, Any]] = None) -> str:
        """Return a download URL for an ee.Image using getDownloadURL parameters.

        Callers should follow the EE docs for allowed parameters. This method
        simply wraps ee.Image.getDownloadURL.
        """
        self.initialize()
        params = params or {}
        try:
            url = img.getDownloadURL(params)
            return url
        except Exception as e:
            raise RuntimeError(
                "Failed to obtain download URL from Earth Engine: " + str(e)
            ) from e

    def export_to_drive(
        self,
        img,
        description: str,
        folder: Optional[str] = None,
        file_name_prefix: Optional[str] = None,
        region: Optional[Any] = None,
        crs: Optional[str] = None,
        scale: Optional[float] = None,
        max_pixels: float = 1e13,
        wait: bool = True,
        poll_interval: int = 10,
        timeout: int = 3600,
    ) -> Dict[str, Any]:
        """Export the EE image to Google Drive as a GeoTIFF.

        Parameters
        ----------
        img: ee.Image
            The Earth Engine image to export.
        description: str
            Task description.
        folder: str, optional
            Google Drive folder to export into.
        file_name_prefix: str, optional
            Prefix for exported filename.
        region: optional
            Region geometry (ee.Geometry or GeoJSON-like) to export.
        crs: str, optional
            CRS to export in (e.g. 'EPSG:32643').
        scale: float, optional
            Pixel size in metres.
        max_pixels: float
            maxPixels parameter for EE export.
        wait: bool
            If True, poll the task until completion (or failure/timeout).
        poll_interval: int
            Seconds between polls.
        timeout: int
            Maximum seconds to wait when wait=True.

        Returns
        -------
        dict
            The final task status dict (task.status()) on completion, or the
            initial task.status() dict if wait=False.
        """
        ee = _ensure_ee_imported()
        self.initialize()

        # Build export kwargs
        kwargs: Dict[str, Any] = {
            "image": img,
            "description": description,
            "maxPixels": int(max_pixels),
        }
        if folder:
            kwargs["folder"] = folder
        if file_name_prefix:
            kwargs["fileNamePrefix"] = file_name_prefix
        if region is not None:
            # accept either ee.Geometry or geojson/list; let EE coerce if needed
            kwargs["region"] = region
        if crs is not None:
            kwargs["crs"] = crs
        if scale is not None:
            kwargs["scale"] = float(scale)

        try:
            task = ee.batch.Export.image.toDrive(**kwargs)
            task.start()
        except Exception as e:
            raise RuntimeError(f"Failed to start Drive export task: {e}") from e

        status = task.status()
        # Optionally wait for completion
        if not wait:
            return status

        start_time = time.time()
        logger = logging.getLogger(__name__)
        logger.info(
            "Started Drive export task '%s' -> folder=%s prefix=%s",
            description,
            folder,
            file_name_prefix,
        )

        while True:
            status = task.status() or {}
            state = status.get("state")
            if state == "COMPLETED":
                logger.info("Drive export task completed: %s", description)
                return status
            if state == "FAILED":
                logger.error("Drive export task failed: %s", status)
                raise RuntimeError(f"Drive export failed: {status}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                # Optionally try to cancel
                try:
                    task.cancel()
                except Exception:
                    pass
                raise TimeoutError(
                    f"Drive export did not complete within {timeout} seconds; last status: {status}"
                )

            time.sleep(poll_interval)
