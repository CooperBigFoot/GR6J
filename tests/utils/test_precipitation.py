"""Tests for solid precipitation utilities using the USACE formula.

Tests verify the correctness of solid fraction computation, solid precipitation
calculation, and mean annual solid precipitation estimation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pydrology.utils.precipitation import (
    compute_mean_annual_solid_precip,
    compute_solid_fraction,
    compute_solid_precip,
)


class TestComputeSolidFraction:
    """Tests for compute_solid_fraction function."""

    def test_all_snow_below_threshold(self) -> None:
        """Temperature well below t_snow returns fraction of 1.0."""
        temp = np.array([-5.0, -10.0, -2.0])

        result = compute_solid_fraction(temp)

        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], rtol=1e-10)

    def test_all_rain_above_threshold(self) -> None:
        """Temperature at or above t_rain returns fraction of 0.0."""
        temp = np.array([3.0, 5.0, 10.0])

        result = compute_solid_fraction(temp)

        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], rtol=1e-10)

    def test_linear_interpolation(self) -> None:
        """Temperatures in transition zone are linearly interpolated."""
        temp = np.array([0.0, 1.0, 2.0])

        result = compute_solid_fraction(temp)

        # With defaults t_snow=-1, t_rain=3 (range of 4):
        # T=0 -> (3-0)/4 = 0.75
        # T=1 -> (3-1)/4 = 0.5
        # T=2 -> (3-2)/4 = 0.25
        np.testing.assert_allclose(result, [0.75, 0.5, 0.25], rtol=1e-10)

    def test_exactly_at_boundaries(self) -> None:
        """Temperature exactly at boundaries returns expected values."""
        temp = np.array([-1.0, 3.0])

        result = compute_solid_fraction(temp)

        # T=-1 (t_snow) -> 1.0, T=3 (t_rain) -> 0.0
        np.testing.assert_allclose(result, [1.0, 0.0], rtol=1e-10)

    def test_custom_thresholds(self) -> None:
        """Non-default t_snow and t_rain thresholds work correctly."""
        temp = np.array([0.0, 2.0, 4.0])

        result = compute_solid_fraction(temp, t_snow=0.0, t_rain=4.0)

        # With t_snow=0, t_rain=4 (range of 4):
        # T=0 -> 1.0, T=2 -> 0.5, T=4 -> 0.0
        np.testing.assert_allclose(result, [1.0, 0.5, 0.0], rtol=1e-10)

    def test_output_bounded_zero_one(self) -> None:
        """Output values are always in [0, 1] range."""
        temp = np.array([-100.0, 0.0, 100.0])

        result = compute_solid_fraction(temp)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_invalid_threshold_order_raises(self) -> None:
        """t_snow >= t_rain raises ValueError."""
        temp = np.array([0.0])

        # Test equal thresholds
        with pytest.raises(ValueError, match="t_snow must be less than t_rain"):
            compute_solid_fraction(temp, t_snow=3.0, t_rain=3.0)

        # Test t_snow greater than t_rain
        with pytest.raises(ValueError, match="t_snow must be less than t_rain"):
            compute_solid_fraction(temp, t_snow=5.0, t_rain=3.0)


class TestComputeSolidPrecip:
    """Tests for compute_solid_precip function."""

    def test_all_snow_cold_temperature(self) -> None:
        """All snow conditions return full precipitation as solid."""
        precip = np.array([10.0, 5.0, 3.0])
        temp = np.array([-5.0, -5.0, -5.0])

        result = compute_solid_precip(precip, temp)

        np.testing.assert_allclose(result, precip, rtol=1e-10)

    def test_all_rain_warm_temperature(self) -> None:
        """All rain conditions return zero solid precipitation."""
        precip = np.array([10.0, 5.0, 3.0])
        temp = np.array([5.0, 10.0, 15.0])

        result = compute_solid_precip(precip, temp)

        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], rtol=1e-10)

    def test_mixed_precipitation(self) -> None:
        """Mixed conditions compute correct fractional solid precip."""
        precip = np.array([10.0, 10.0, 10.0])
        temp = np.array([-5.0, 1.0, 5.0])

        result = compute_solid_precip(precip, temp)

        # Expected: [10 * 1.0, 10 * 0.5, 10 * 0.0] = [10, 5, 0]
        np.testing.assert_allclose(result, [10.0, 5.0, 0.0], rtol=1e-10)

    def test_zero_precipitation(self) -> None:
        """Zero precipitation returns zero solid precipitation."""
        precip = np.array([0.0, 0.0, 0.0])
        temp = np.array([-5.0, 0.0, 5.0])

        result = compute_solid_precip(precip, temp)

        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], rtol=1e-10)

    def test_shape_mismatch_raises(self) -> None:
        """Different shapes for precip and temp raise ValueError."""
        precip = np.array([1.0, 2.0, 3.0])
        temp = np.array([0.0, 1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match=r"precip shape .* does not match temp shape"):
            compute_solid_precip(precip, temp)


class TestComputeMeanAnnualSolidPrecip:
    """Tests for compute_mean_annual_solid_precip function."""

    def test_all_snow_conditions(self) -> None:
        """All snow returns mean precip * 365.25."""
        precip = np.full(365, 5.0)
        temp = np.full(365, -5.0)

        result = compute_mean_annual_solid_precip(precip, temp)

        # Expected: 5.0 * 365.25 = 1826.25
        np.testing.assert_allclose(result, 1826.25, rtol=1e-10)

    def test_all_rain_conditions(self) -> None:
        """All rain returns 0.0."""
        precip = np.full(365, 5.0)
        temp = np.full(365, 10.0)

        result = compute_mean_annual_solid_precip(precip, temp)

        np.testing.assert_allclose(result, 0.0, rtol=1e-10)

    def test_empty_arrays_raises(self) -> None:
        """Empty arrays raise ValueError."""
        precip = np.array([])
        temp = np.array([])

        with pytest.raises(ValueError, match="Arrays must not be empty"):
            compute_mean_annual_solid_precip(precip, temp)

    def test_length_mismatch_raises(self) -> None:
        """Different length arrays raise ValueError."""
        precip = np.arange(10, dtype=np.float64)
        temp = np.arange(5, dtype=np.float64)

        with pytest.raises(ValueError, match=r"precip shape .* does not match temp shape"):
            compute_mean_annual_solid_precip(precip, temp)

    def test_single_day(self) -> None:
        """Works with single day of data."""
        precip = np.array([10.0])
        temp = np.array([-5.0])

        result = compute_mean_annual_solid_precip(precip, temp)

        # Expected: 10.0 * 365.25 = 3652.5
        np.testing.assert_allclose(result, 3652.5, rtol=1e-10)
