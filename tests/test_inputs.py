"""Tests for input data structures: ForcingData and Catchment.

Tests cover validation, immutability, type coercion, and boundary warnings
for the input containers used by the GR6J model.
"""

import logging

import numpy as np
import pytest
from pydantic import ValidationError

from gr6j import Catchment, ForcingData


def _make_dates(n: int, start: str = "2020-01-01") -> np.ndarray:
    """Create a datetime64 array with n days starting from the given date."""
    return np.arange(start, np.datetime64(start) + np.timedelta64(n, "D"), dtype="datetime64[D]")


class TestForcingData:
    """Tests for the ForcingData validated Pydantic model."""

    def test_creates_with_valid_arrays(self) -> None:
        """ForcingData instantiates correctly with valid time, precip, and pet arrays."""
        forcing = ForcingData(
            time=_make_dates(5),
            precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
            pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
        )

        assert len(forcing) == 5

    def test_creates_with_optional_temp(self) -> None:
        """ForcingData instantiates correctly when temp array is provided."""
        forcing = ForcingData(
            time=_make_dates(3),
            precip=np.array([10.0, 5.0, 0.0]),
            pet=np.array([3.0, 4.0, 5.0]),
            temp=np.array([15.0, 18.0, 12.0]),
        )

        assert forcing.temp is not None
        np.testing.assert_array_equal(forcing.temp, [15.0, 18.0, 12.0])

    def test_creates_without_temp(self) -> None:
        """ForcingData allows temp=None (optional temperature)."""
        forcing = ForcingData(
            time=_make_dates(3),
            precip=np.array([10.0, 5.0, 0.0]),
            pet=np.array([3.0, 4.0, 5.0]),
        )

        assert forcing.temp is None

    def test_len_returns_array_length(self) -> None:
        """__len__ method returns the number of timesteps."""
        forcing = ForcingData(
            time=_make_dates(7),
            precip=np.zeros(7),
            pet=np.zeros(7),
        )

        assert len(forcing) == 7

    # Validation - 1D requirement

    def test_rejects_2d_precip_array(self) -> None:
        """ValueError raised when precip array is 2D."""
        with pytest.raises(ValidationError, match="precip array must be 1D"):
            ForcingData(
                time=_make_dates(3),
                precip=np.array([[10.0, 5.0, 0.0]]),
                pet=np.array([3.0, 4.0, 5.0]),
            )

    def test_rejects_2d_pet_array(self) -> None:
        """ValueError raised when pet array is 2D."""
        with pytest.raises(ValidationError, match="pet array must be 1D"):
            ForcingData(
                time=_make_dates(3),
                precip=np.array([10.0, 5.0, 0.0]),
                pet=np.array([[3.0, 4.0, 5.0]]),
            )

    def test_rejects_2d_temp_array(self) -> None:
        """ValueError raised when temp array is 2D."""
        with pytest.raises(ValidationError, match="temp array must be 1D"):
            ForcingData(
                time=_make_dates(3),
                precip=np.array([10.0, 5.0, 0.0]),
                pet=np.array([3.0, 4.0, 5.0]),
                temp=np.array([[15.0, 18.0, 12.0]]),
            )

    def test_rejects_2d_time_array(self) -> None:
        """ValueError raised when time array is 2D."""
        with pytest.raises(ValidationError, match="time array must be 1D"):
            ForcingData(
                time=np.array([[np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]]),
                precip=np.array([10.0, 5.0]),
                pet=np.array([3.0, 4.0]),
            )

    # NaN rejection

    def test_rejects_nan_in_precip(self) -> None:
        """ValueError raised when precip array contains NaN."""
        with pytest.raises(ValidationError, match="NaN"):
            ForcingData(
                time=_make_dates(3),
                precip=np.array([10.0, np.nan, 5.0]),
                pet=np.array([3.0, 4.0, 5.0]),
            )

    def test_rejects_nan_in_pet(self) -> None:
        """ValueError raised when pet array contains NaN."""
        with pytest.raises(ValidationError, match="NaN"):
            ForcingData(
                time=_make_dates(3),
                precip=np.array([10.0, 5.0, 0.0]),
                pet=np.array([3.0, np.nan, 5.0]),
            )

    def test_rejects_nan_in_temp(self) -> None:
        """ValueError raised when temp array contains NaN."""
        with pytest.raises(ValidationError, match="NaN"):
            ForcingData(
                time=_make_dates(3),
                precip=np.array([10.0, 5.0, 0.0]),
                pet=np.array([3.0, 4.0, 5.0]),
                temp=np.array([15.0, np.nan, 12.0]),
            )

    # Length mismatch

    def test_rejects_length_mismatch_precip(self) -> None:
        """ValueError raised when precip length differs from time length."""
        with pytest.raises(ValidationError, match="precip length.*does not match"):
            ForcingData(
                time=_make_dates(5),
                precip=np.array([10.0, 5.0, 0.0]),  # Length 3, expected 5
                pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
            )

    def test_rejects_length_mismatch_pet(self) -> None:
        """ValueError raised when pet length differs from time length."""
        with pytest.raises(ValidationError, match="pet length.*does not match"):
            ForcingData(
                time=_make_dates(5),
                precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
                pet=np.array([3.0, 4.0]),  # Length 2, expected 5
            )

    def test_rejects_length_mismatch_temp(self) -> None:
        """ValueError raised when temp length differs from time length."""
        with pytest.raises(ValidationError, match="temp length.*does not match"):
            ForcingData(
                time=_make_dates(5),
                precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
                pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
                temp=np.array([15.0, 18.0]),  # Length 2, expected 5
            )

    # Type coercion

    def test_coerces_int_precip_to_float64(self) -> None:
        """Integer precip array is coerced to float64."""
        forcing = ForcingData(
            time=_make_dates(3),
            precip=np.array([10, 5, 0]),  # Integer array
            pet=np.array([3.0, 4.0, 5.0]),
        )

        assert forcing.precip.dtype == np.float64
        np.testing.assert_array_equal(forcing.precip, [10.0, 5.0, 0.0])

    def test_coerces_time_to_datetime64(self) -> None:
        """Time array is coerced to datetime64[ns]."""
        forcing = ForcingData(
            time=_make_dates(3),
            precip=np.array([10.0, 5.0, 0.0]),
            pet=np.array([3.0, 4.0, 5.0]),
        )

        assert np.issubdtype(forcing.time.dtype, np.datetime64)

    # Immutability

    def test_is_frozen(self) -> None:
        """ForcingData is immutable - assigning to fields should raise."""
        forcing = ForcingData(
            time=_make_dates(3),
            precip=np.array([10.0, 5.0, 0.0]),
            pet=np.array([3.0, 4.0, 5.0]),
        )

        with pytest.raises(ValidationError):
            forcing.precip = np.array([1.0, 2.0, 3.0])  # type: ignore[misc]


class TestCatchment:
    """Tests for the Catchment frozen dataclass."""

    def test_creates_with_valid_value(self) -> None:
        """Catchment instantiates correctly with mean_annual_solid_precip."""
        catchment = Catchment(mean_annual_solid_precip=150.0)

        assert catchment.mean_annual_solid_precip == 150.0

    def test_creates_with_optional_fields(self) -> None:
        """Catchment instantiates correctly with optional fields."""
        hypsometric = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            hypsometric_curve=hypsometric,
            input_elevation=500.0,
            n_layers=5,
        )

        assert catchment.mean_annual_solid_precip == 150.0
        np.testing.assert_array_equal(catchment.hypsometric_curve, hypsometric)
        assert catchment.input_elevation == 500.0
        assert catchment.n_layers == 5

    def test_default_values(self) -> None:
        """Catchment has correct default values for optional fields."""
        catchment = Catchment(mean_annual_solid_precip=150.0)

        assert catchment.hypsometric_curve is None
        assert catchment.input_elevation is None
        assert catchment.n_layers == 1

    # Bounds warnings

    def test_warns_when_below_bounds(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when mean_annual_solid_precip is below typical range."""
        with caplog.at_level(logging.WARNING):
            Catchment(mean_annual_solid_precip=-100.0)

        assert len(caplog.records) == 1
        assert "outside typical range" in caplog.text

    def test_warns_when_above_bounds(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when mean_annual_solid_precip exceeds typical range."""
        with caplog.at_level(logging.WARNING):
            Catchment(mean_annual_solid_precip=15000.0)

        assert len(caplog.records) == 1
        assert "outside typical range" in caplog.text

    def test_no_warning_within_bounds(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning logged when mean_annual_solid_precip is within typical range."""
        with caplog.at_level(logging.WARNING):
            Catchment(mean_annual_solid_precip=150.0)

        assert len(caplog.records) == 0

    # Immutability

    def test_is_frozen(self) -> None:
        """Catchment is immutable - assigning to fields should raise."""
        catchment = Catchment(mean_annual_solid_precip=150.0)

        with pytest.raises(AttributeError):
            catchment.mean_annual_solid_precip = 200.0  # type: ignore[misc]
