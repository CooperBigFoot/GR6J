"""DEM (Digital Elevation Model) analysis utilities.

This module provides functions for computing elevation statistics and hypsometric
curves from raster DEM files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DEMStatistics:
    """Statistics computed from a Digital Elevation Model.

    Attributes:
        min_elevation: Minimum elevation value in the DEM [m].
        max_elevation: Maximum elevation value in the DEM [m].
        mean_elevation: Mean elevation value in the DEM [m].
        median_elevation: Median elevation value in the DEM [m].
        hypsometric_curve: Array of shape (101,) containing elevation values
            at percentiles 0-100, representing the hypsometric curve.
    """

    min_elevation: float
    max_elevation: float
    mean_elevation: float
    median_elevation: float
    hypsometric_curve: np.ndarray

    def __repr__(self) -> str:
        """Return readable representation of key DEM statistics."""
        return (
            f"DEMStatistics(\n"
            f"  min_elevation={self.min_elevation:.2f},\n"
            f"  max_elevation={self.max_elevation:.2f},\n"
            f"  mean_elevation={self.mean_elevation:.2f},\n"
            f"  median_elevation={self.median_elevation:.2f},\n"
            f"  hypsometric_curve=<array shape={self.hypsometric_curve.shape}>\n"
            f")"
        )


def _mask_nodata(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Extract valid pixels from raster data, excluding NoData and invalid values.

    Excludes:
    - NaN and infinite values
    - The NoData value from raster metadata (if set)
    - Zero values (common fill value for clipped DEMs when NoData is not set)

    Args:
        data: Raw raster data array (can be any shape).
        nodata: NoData value from the raster metadata, or None if not defined.

    Returns:
        1D array containing only valid pixel values.
    """
    # Flatten to 1D for easier processing
    flat_data = data.ravel()

    # Create mask for valid data (exclude NaN, inf, and zero)
    # Zero is excluded because clipped DEMs commonly use 0 as a fill value
    # when the NoData metadata is not properly set
    valid_mask = np.isfinite(flat_data) & (flat_data != 0)

    # Also exclude NoData value if specified
    if nodata is not None:
        valid_mask &= flat_data != nodata

    # Log warning if zeros were excluded and nodata wasn't set
    if nodata is None:
        zero_count = np.sum(flat_data == 0)
        if zero_count > 0:
            logger.debug(
                "Excluded %d zero-valued pixels (likely NoData in clipped DEM)",
                zero_count,
            )

    return flat_data[valid_mask]


def _compute_hypsometric_curve(valid_data: np.ndarray) -> np.ndarray:
    """Compute 101-point hypsometric curve from elevation data.

    The hypsometric curve represents the cumulative distribution of elevations,
    computed as percentiles from 0 to 100.

    Args:
        valid_data: 1D array of valid elevation values.

    Returns:
        Array of shape (101,) containing elevation values at percentiles 0-100.
    """
    percentiles = np.arange(101)
    return np.percentile(valid_data, percentiles)


def analyze_dem(dem_path: str | Path) -> DEMStatistics:
    """Analyze a DEM raster file and compute elevation statistics.

    Opens the raster file, extracts valid pixels (excluding NoData, NaN, and
    infinite values), and computes summary statistics including a 101-point
    hypsometric curve.

    Args:
        dem_path: Path to the DEM raster file (e.g., GeoTIFF).

    Returns:
        DEMStatistics containing min, max, mean, median elevation and
        hypsometric curve.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the raster file cannot be read or contains no valid pixels.
    """
    dem_path = Path(dem_path)

    # Check file exists
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    # Try to open the raster file
    try:
        with rasterio.open(dem_path) as src:
            data = src.read(1)  # Read first band
            nodata = src.nodata
    except RasterioIOError as e:
        raise ValueError(f"Invalid raster file: {dem_path}") from e

    # Extract valid pixels
    valid_data = _mask_nodata(data, nodata)

    # Check we have valid data
    if valid_data.size == 0:
        raise ValueError("All pixels are NoData or invalid")

    logger.debug(
        "Analyzing DEM %s: %d valid pixels out of %d total",
        dem_path.name,
        valid_data.size,
        data.size,
    )

    # Compute statistics
    min_elevation = float(np.min(valid_data))
    max_elevation = float(np.max(valid_data))
    mean_elevation = float(np.mean(valid_data))
    median_elevation = float(np.median(valid_data))
    hypsometric_curve = _compute_hypsometric_curve(valid_data)

    return DEMStatistics(
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        mean_elevation=mean_elevation,
        median_elevation=median_elevation,
        hypsometric_curve=hypsometric_curve,
    )
