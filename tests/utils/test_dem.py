"""Tests for DEM (Digital Elevation Model) analysis utilities.

Tests verify the correctness of elevation statistics computation and
hypsometric curve generation from raster DEM files.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from gr6j.utils.dem import DEMStatistics, analyze_dem


def _create_dem_file(path: Path, data: np.ndarray, nodata: float = -9999.0) -> None:
    """Create synthetic GeoTIFF for testing.

    Args:
        path: Path where the GeoTIFF will be written.
        data: 2D numpy array of elevation values.
        nodata: NoData value to use in the raster metadata.
    """
    height, width = data.shape
    transform = from_bounds(0, 0, width, height, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def simple_dem(tmp_path: Path) -> Path:
    """Create a 10x10 DEM with elevations 100-1000m."""
    data = np.linspace(100, 1000, 100).reshape(10, 10).astype(np.float32)
    dem_path = tmp_path / "simple_dem.tif"
    _create_dem_file(dem_path, data)
    return dem_path


@pytest.fixture
def dem_with_nodata(tmp_path: Path) -> Path:
    """Create a DEM with some NoData values (-9999)."""
    data = np.linspace(100, 500, 100).reshape(10, 10).astype(np.float32)
    # Set corners to NoData
    data[0, 0] = -9999.0
    data[0, 9] = -9999.0
    data[9, 0] = -9999.0
    data[9, 9] = -9999.0
    dem_path = tmp_path / "dem_with_nodata.tif"
    _create_dem_file(dem_path, data)
    return dem_path


@pytest.fixture
def single_pixel_dem(tmp_path: Path) -> Path:
    """Create a 1x1 DEM with single value (500.0)."""
    data = np.array([[500.0]], dtype=np.float32)
    dem_path = tmp_path / "single_pixel_dem.tif"
    _create_dem_file(dem_path, data)
    return dem_path


@pytest.fixture
def flat_dem(tmp_path: Path) -> Path:
    """Create a DEM with all same value (1000.0)."""
    data = np.full((10, 10), 1000.0, dtype=np.float32)
    dem_path = tmp_path / "flat_dem.tif"
    _create_dem_file(dem_path, data)
    return dem_path


@pytest.fixture
def dem_with_nan(tmp_path: Path) -> Path:
    """Create a DEM with some NaN values."""
    data = np.linspace(200, 800, 100).reshape(10, 10).astype(np.float32)
    # Set some pixels to NaN
    data[2, 3] = np.nan
    data[5, 5] = np.nan
    data[7, 8] = np.nan
    dem_path = tmp_path / "dem_with_nan.tif"
    _create_dem_file(dem_path, data, nodata=-9999.0)
    return dem_path


class TestDEMStatistics:
    """Tests for DEMStatistics dataclass."""

    def test_dataclass_fields_exist(self) -> None:
        """Check all fields exist on DEMStatistics."""
        hypso = np.arange(101, dtype=np.float64)
        stats = DEMStatistics(
            min_elevation=100.0,
            max_elevation=1000.0,
            mean_elevation=550.0,
            median_elevation=550.0,
            hypsometric_curve=hypso,
        )

        assert hasattr(stats, "min_elevation")
        assert hasattr(stats, "max_elevation")
        assert hasattr(stats, "mean_elevation")
        assert hasattr(stats, "median_elevation")
        assert hasattr(stats, "hypsometric_curve")

    def test_frozen_immutable(self) -> None:
        """Verify cannot modify fields of frozen dataclass."""
        hypso = np.arange(101, dtype=np.float64)
        stats = DEMStatistics(
            min_elevation=100.0,
            max_elevation=1000.0,
            mean_elevation=550.0,
            median_elevation=550.0,
            hypsometric_curve=hypso,
        )

        with pytest.raises(FrozenInstanceError):
            stats.min_elevation = 200.0  # type: ignore[misc]

    def test_repr_format(self) -> None:
        """Check __repr__ contains key info."""
        hypso = np.arange(101, dtype=np.float64)
        stats = DEMStatistics(
            min_elevation=100.0,
            max_elevation=1000.0,
            mean_elevation=550.0,
            median_elevation=550.0,
            hypsometric_curve=hypso,
        )

        repr_str = repr(stats)

        assert "DEMStatistics" in repr_str
        assert "min_elevation=100.00" in repr_str
        assert "max_elevation=1000.00" in repr_str
        assert "mean_elevation=550.00" in repr_str
        assert "median_elevation=550.00" in repr_str
        assert "hypsometric_curve=<array shape=(101,)>" in repr_str


class TestAnalyzeDem:
    """Tests for analyze_dem function basic functionality."""

    def test_returns_dem_statistics(self, simple_dem: Path) -> None:
        """Returns correct type."""
        result = analyze_dem(simple_dem)

        assert isinstance(result, DEMStatistics)

    def test_min_max_correct(self, simple_dem: Path) -> None:
        """Min/max match expected values."""
        result = analyze_dem(simple_dem)

        # Data is linspace(100, 1000, 100)
        np.testing.assert_allclose(result.min_elevation, 100.0, rtol=1e-5)
        np.testing.assert_allclose(result.max_elevation, 1000.0, rtol=1e-5)

    def test_mean_median_correct(self, simple_dem: Path) -> None:
        """Mean/median computed correctly."""
        result = analyze_dem(simple_dem)

        # For linspace(100, 1000, 100), mean and median should be 550
        np.testing.assert_allclose(result.mean_elevation, 550.0, rtol=1e-5)
        np.testing.assert_allclose(result.median_elevation, 550.0, rtol=1e-5)

    def test_accepts_string_path(self, simple_dem: Path) -> None:
        """Works with str path."""
        result = analyze_dem(str(simple_dem))

        assert isinstance(result, DEMStatistics)
        np.testing.assert_allclose(result.min_elevation, 100.0, rtol=1e-5)

    def test_accepts_path_object(self, simple_dem: Path) -> None:
        """Works with Path object."""
        result = analyze_dem(simple_dem)

        assert isinstance(result, DEMStatistics)
        np.testing.assert_allclose(result.min_elevation, 100.0, rtol=1e-5)


class TestAnalyzeDemErrorHandling:
    """Tests for analyze_dem error handling."""

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError for missing file."""
        nonexistent_path = tmp_path / "nonexistent.tif"

        with pytest.raises(FileNotFoundError, match="DEM file not found"):
            analyze_dem(nonexistent_path)

    def test_raises_on_invalid_file(self, tmp_path: Path) -> None:
        """ValueError for non-raster file."""
        invalid_file = tmp_path / "not_a_raster.txt"
        invalid_file.write_text("This is not a raster file")

        with pytest.raises(ValueError, match="Invalid raster file"):
            analyze_dem(invalid_file)

    def test_raises_on_all_nodata(self, tmp_path: Path) -> None:
        """ValueError when all pixels are NoData."""
        data = np.full((5, 5), -9999.0, dtype=np.float32)
        dem_path = tmp_path / "all_nodata.tif"
        _create_dem_file(dem_path, data)

        with pytest.raises(ValueError, match="All pixels are NoData or invalid"):
            analyze_dem(dem_path)

    def test_raises_on_all_zeros(self, tmp_path: Path) -> None:
        """ValueError when all pixels are zero (treated as NoData)."""
        data = np.full((5, 5), 0.0, dtype=np.float32)
        dem_path = tmp_path / "all_zeros.tif"
        _create_dem_file(dem_path, data)

        with pytest.raises(ValueError, match="All pixels are NoData or invalid"):
            analyze_dem(dem_path)


class TestAnalyzeDemEdgeCases:
    """Tests for analyze_dem edge cases."""

    def test_single_pixel_dem(self, single_pixel_dem: Path) -> None:
        """Works with 1x1 DEM."""
        result = analyze_dem(single_pixel_dem)

        np.testing.assert_allclose(result.min_elevation, 500.0, rtol=1e-5)
        np.testing.assert_allclose(result.max_elevation, 500.0, rtol=1e-5)
        np.testing.assert_allclose(result.mean_elevation, 500.0, rtol=1e-5)
        np.testing.assert_allclose(result.median_elevation, 500.0, rtol=1e-5)

    def test_flat_terrain(self, flat_dem: Path) -> None:
        """All same elevation (min=max=mean=median)."""
        result = analyze_dem(flat_dem)

        np.testing.assert_allclose(result.min_elevation, 1000.0, rtol=1e-5)
        np.testing.assert_allclose(result.max_elevation, 1000.0, rtol=1e-5)
        np.testing.assert_allclose(result.mean_elevation, 1000.0, rtol=1e-5)
        np.testing.assert_allclose(result.median_elevation, 1000.0, rtol=1e-5)

    def test_handles_nodata_values(self, dem_with_nodata: Path) -> None:
        """Correctly excludes NoData pixels."""
        result = analyze_dem(dem_with_nodata)

        # Original data was linspace(100, 500, 100) with 4 corners set to -9999
        # The 4 NoData values should be excluded
        # Check that -9999 is not in the stats
        assert result.min_elevation > 0  # Should not be -9999
        assert result.min_elevation >= 100.0
        assert result.max_elevation <= 500.0

    def test_handles_nan_values(self, dem_with_nan: Path) -> None:
        """Correctly excludes NaN pixels."""
        result = analyze_dem(dem_with_nan)

        # NaN values should be excluded; stats should be valid
        assert not np.isnan(result.min_elevation)
        assert not np.isnan(result.max_elevation)
        assert not np.isnan(result.mean_elevation)
        assert not np.isnan(result.median_elevation)
        # Original data was linspace(200, 800, 100)
        np.testing.assert_allclose(result.min_elevation, 200.0, rtol=1e-5)
        np.testing.assert_allclose(result.max_elevation, 800.0, rtol=1e-5)

    def test_extreme_elevations(self, tmp_path: Path) -> None:
        """Very high (8848) and low (-400) values work."""
        # Create DEM with Everest and Dead Sea elevations
        # Note: 0.0 is excluded as potential NoData, so use -400 as min
        data = np.array(
            [
                [-400.0, 100.0, 1000.0],
                [2000.0, 4000.0, 6000.0],
                [7000.0, 8000.0, 8848.0],
            ],
            dtype=np.float32,
        )
        dem_path = tmp_path / "extreme_dem.tif"
        _create_dem_file(dem_path, data)

        result = analyze_dem(dem_path)

        np.testing.assert_allclose(result.min_elevation, -400.0, rtol=1e-5)
        np.testing.assert_allclose(result.max_elevation, 8848.0, rtol=1e-5)

    def test_excludes_zero_values(self, tmp_path: Path) -> None:
        """Zero values are excluded (common fill value for clipped DEMs)."""
        # Simulate a clipped DEM where 0 is used as fill value
        data = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 500.0, 600.0],
                [0.0, 700.0, 800.0],
            ],
            dtype=np.float32,
        )
        dem_path = tmp_path / "dem_with_zeros.tif"
        # Create without explicit nodata to simulate common clipped DEM issue
        _create_dem_file(dem_path, data, nodata=-9999.0)

        result = analyze_dem(dem_path)

        # Zeros should be excluded; stats should reflect only 500-800 range
        np.testing.assert_allclose(result.min_elevation, 500.0, rtol=1e-5)
        np.testing.assert_allclose(result.max_elevation, 800.0, rtol=1e-5)


class TestHypsometricCurveContract:
    """Tests for hypsometric curve properties."""

    def test_has_101_points(self, simple_dem: Path) -> None:
        """Shape is (101,)."""
        result = analyze_dem(simple_dem)

        assert result.hypsometric_curve.shape == (101,)

    def test_monotonic_increasing(self, simple_dem: Path) -> None:
        """Each value >= previous."""
        result = analyze_dem(simple_dem)

        diffs = np.diff(result.hypsometric_curve)
        assert np.all(diffs >= 0), "Hypsometric curve should be monotonically increasing"

    def test_first_equals_min(self, simple_dem: Path) -> None:
        """hypsometric_curve[0] == min_elevation."""
        result = analyze_dem(simple_dem)

        np.testing.assert_allclose(
            result.hypsometric_curve[0],
            result.min_elevation,
            rtol=1e-5,
        )

    def test_last_equals_max(self, simple_dem: Path) -> None:
        """hypsometric_curve[100] == max_elevation."""
        result = analyze_dem(simple_dem)

        np.testing.assert_allclose(
            result.hypsometric_curve[100],
            result.max_elevation,
            rtol=1e-5,
        )

    def test_median_at_index_50(self, simple_dem: Path) -> None:
        """hypsometric_curve[50] == median_elevation."""
        result = analyze_dem(simple_dem)

        np.testing.assert_allclose(
            result.hypsometric_curve[50],
            result.median_elevation,
            rtol=1e-5,
        )
