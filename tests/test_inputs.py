"""Tests for input data structures: ForcingData and Catchment.

Tests cover validation, immutability, and type coercion
for the input containers used by the GR6J model.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from pydrology import Catchment, ForcingData, PrecipGradientType


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
        hypsometric = np.linspace(200.0, 2000.0, 101)
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

    # Immutability

    def test_is_frozen(self) -> None:
        """Catchment is immutable - assigning to fields should raise."""
        catchment = Catchment(mean_annual_solid_precip=150.0)

        with pytest.raises(AttributeError):
            catchment.mean_annual_solid_precip = 200.0  # type: ignore[misc]


class TestCatchmentMultiLayerValidation:
    """Tests for Catchment multi-layer validation."""

    def test_rejects_multi_layer_without_hypsometric_curve(self) -> None:
        """ValueError when n_layers > 1 but hypsometric_curve is None."""
        with pytest.raises(ValueError, match="hypsometric_curve is required"):
            Catchment(
                mean_annual_solid_precip=150.0,
                n_layers=5,
                input_elevation=500.0,
            )

    def test_rejects_multi_layer_without_input_elevation(self) -> None:
        """ValueError when n_layers > 1 but input_elevation is None."""
        with pytest.raises(ValueError, match="input_elevation is required"):
            Catchment(
                mean_annual_solid_precip=150.0,
                n_layers=5,
                hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            )

    def test_rejects_wrong_hypsometric_curve_length(self) -> None:
        """ValueError when hypsometric_curve doesn't have 101 points."""
        with pytest.raises(ValueError, match="101 points"):
            Catchment(
                mean_annual_solid_precip=150.0,
                n_layers=5,
                hypsometric_curve=np.linspace(200.0, 2000.0, 50),
                input_elevation=500.0,
            )

    def test_accepts_valid_multi_layer_config(self) -> None:
        """Valid multi-layer configuration creates Catchment successfully."""
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=5,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )

        assert catchment.n_layers == 5

    def test_single_layer_needs_no_extra_fields(self) -> None:
        """n_layers=1 doesn't require hypsometric_curve or input_elevation."""
        catchment = Catchment(mean_annual_solid_precip=150.0)

        assert catchment.n_layers == 1
        assert catchment.hypsometric_curve is None
        assert catchment.input_elevation is None

    def test_gradient_fields_are_optional(self) -> None:
        """temp_gradient and precip_gradient default to None."""
        catchment = Catchment(mean_annual_solid_precip=150.0)

        assert catchment.temp_gradient is None
        assert catchment.precip_gradient is None

    def test_custom_gradients_are_stored(self) -> None:
        """Custom gradient values are correctly stored."""
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            temp_gradient=0.8,
            precip_gradient=0.0005,
        )

        assert catchment.temp_gradient == 0.8
        assert catchment.precip_gradient == 0.0005


class TestResolution:
    """Tests for the Resolution enum."""

    def test_hourly_less_than_daily(self) -> None:
        """hourly < daily comparison is True."""
        from pydrology import Resolution

        assert Resolution.hourly < Resolution.daily

    def test_daily_less_than_monthly(self) -> None:
        """daily < monthly comparison is True."""
        from pydrology import Resolution

        assert Resolution.daily < Resolution.monthly

    def test_monthly_less_than_annual(self) -> None:
        """monthly < annual comparison is True."""
        from pydrology import Resolution

        assert Resolution.monthly < Resolution.annual

    def test_daily_not_less_than_hourly(self) -> None:
        """daily < hourly comparison is False."""
        from pydrology import Resolution

        assert not Resolution.daily < Resolution.hourly

    def test_equal_resolution_not_less_than(self) -> None:
        """Resolution is not less than itself."""
        from pydrology import Resolution

        assert not Resolution.daily < Resolution.daily

    def test_less_than_or_equal_same(self) -> None:
        """Resolution is less than or equal to itself."""
        from pydrology import Resolution

        assert Resolution.daily <= Resolution.daily

    def test_days_per_timestep_hourly(self) -> None:
        """hourly has 1/24 days per timestep."""
        from pydrology import Resolution

        assert Resolution.hourly.days_per_timestep == pytest.approx(1 / 24)

    def test_days_per_timestep_daily(self) -> None:
        """daily has 1.0 days per timestep."""
        from pydrology import Resolution

        assert Resolution.daily.days_per_timestep == 1.0

    def test_days_per_timestep_monthly(self) -> None:
        """monthly has 30.4375 days per timestep."""
        from pydrology import Resolution

        assert Resolution.monthly.days_per_timestep == 30.4375

    def test_days_per_timestep_annual(self) -> None:
        """annual has 365.25 days per timestep."""
        from pydrology import Resolution

        assert Resolution.annual.days_per_timestep == 365.25

    def test_string_equality(self) -> None:
        """Resolution can be compared to string."""
        from pydrology import Resolution

        assert Resolution.daily == "daily"


class TestForcingDataResolution:
    """Tests for ForcingData resolution validation."""

    def test_default_resolution_is_daily(self) -> None:
        """ForcingData defaults to daily resolution."""
        from pydrology import Resolution

        forcing = ForcingData(
            time=_make_dates(5),
            precip=np.zeros(5),
            pet=np.zeros(5),
        )
        assert forcing.resolution == Resolution.daily

    def test_accepts_explicit_daily_resolution(self) -> None:
        """ForcingData accepts explicit daily resolution."""
        from pydrology import Resolution

        forcing = ForcingData(
            time=_make_dates(5),
            precip=np.zeros(5),
            pet=np.zeros(5),
            resolution=Resolution.daily,
        )
        assert forcing.resolution == Resolution.daily

    def test_accepts_monthly_resolution_with_monthly_data(self) -> None:
        """ForcingData accepts monthly resolution with monthly time steps."""
        from pydrology import Resolution

        time = np.arange("2020-01", "2020-07", dtype="datetime64[M]")
        forcing = ForcingData(
            time=time,
            precip=np.zeros(6),
            pet=np.zeros(6),
            resolution=Resolution.monthly,
        )
        assert forcing.resolution == Resolution.monthly

    def test_accepts_annual_resolution_with_annual_data(self) -> None:
        """ForcingData accepts annual resolution with annual time steps."""
        from pydrology import Resolution

        time = np.arange("2020", "2025", dtype="datetime64[Y]")
        forcing = ForcingData(
            time=time,
            precip=np.zeros(5),
            pet=np.zeros(5),
            resolution=Resolution.annual,
        )
        assert forcing.resolution == Resolution.annual

    def test_rejects_mismatched_resolution_daily_data_monthly_resolution(self) -> None:
        """ValueError when daily data has monthly resolution."""
        from pydrology import Resolution

        with pytest.raises(ValidationError, match="Time spacing.*does not match"):
            ForcingData(
                time=_make_dates(30),  # Daily data
                precip=np.zeros(30),
                pet=np.zeros(30),
                resolution=Resolution.monthly,  # But monthly resolution
            )

    def test_rejects_mismatched_resolution_monthly_data_daily_resolution(self) -> None:
        """ValueError when monthly data has daily resolution."""
        from pydrology import Resolution

        time = np.arange("2020-01", "2020-07", dtype="datetime64[M]")
        with pytest.raises(ValidationError, match="Time spacing.*does not match"):
            ForcingData(
                time=time,  # Monthly data
                precip=np.zeros(6),
                pet=np.zeros(6),
                resolution=Resolution.daily,  # But daily resolution
            )

    def test_single_timestep_skips_validation(self) -> None:
        """Single timestep skips resolution validation."""
        from pydrology import Resolution

        forcing = ForcingData(
            time=np.array(["2020-01-01"], dtype="datetime64[D]"),
            precip=np.array([10.0]),
            pet=np.array([3.0]),
            resolution=Resolution.monthly,  # Would fail with >1 timestep
        )
        assert len(forcing) == 1

    def test_tolerates_variable_month_lengths(self) -> None:
        """Monthly resolution tolerates 28-31 day months."""
        from pydrology import Resolution

        # Use actual month boundaries (28, 29, 30, 31 days)
        time = np.array(
            ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            dtype="datetime64[D]",
        )
        forcing = ForcingData(
            time=time,
            precip=np.zeros(4),
            pet=np.zeros(4),
            resolution=Resolution.monthly,
        )
        assert forcing.resolution == Resolution.monthly

    def test_resolution_is_immutable(self) -> None:
        """Resolution field is frozen."""
        from pydrology import Resolution

        forcing = ForcingData(
            time=_make_dates(5),
            precip=np.zeros(5),
            pet=np.zeros(5),
        )
        with pytest.raises(ValidationError):
            forcing.resolution = Resolution.monthly  # type: ignore[misc]


class TestForcingDataAggregate:
    """Tests for ForcingData aggregate method."""

    def test_aggregate_daily_to_monthly(self) -> None:
        """Aggregates daily data to monthly."""
        from pydrology import Resolution

        # Create 3 months of daily data (Jan, Feb, Mar 2020)
        time = np.arange("2020-01-01", "2020-04-01", dtype="datetime64[D]")
        n = len(time)
        forcing = ForcingData(
            time=time,
            precip=np.ones(n) * 10.0,  # 10mm/day
            pet=np.ones(n) * 3.0,  # 3mm/day
            temp=np.ones(n) * 15.0,  # 15C
        )
        monthly = forcing.aggregate(Resolution.monthly)
        assert monthly.resolution == Resolution.monthly
        assert len(monthly) == 3  # 3 months

    def test_aggregate_daily_to_annual(self) -> None:
        """Aggregates daily data to annual."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
        n = len(time)
        forcing = ForcingData(
            time=time,
            precip=np.ones(n) * 2.0,
            pet=np.ones(n) * 1.0,
        )
        annual = forcing.aggregate(Resolution.annual)
        assert annual.resolution == Resolution.annual
        assert len(annual) == 2  # 2 years

    def test_aggregate_sums_precip(self) -> None:
        """Aggregate sums precipitation."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
        n = len(time)  # 31 days in January
        forcing = ForcingData(
            time=time,
            precip=np.ones(n) * 10.0,  # 10mm/day
            pet=np.ones(n) * 3.0,
        )
        monthly = forcing.aggregate(Resolution.monthly)
        assert monthly.precip[0] == pytest.approx(310.0, rel=1e-3)  # 31 * 10

    def test_aggregate_sums_pet(self) -> None:
        """Aggregate sums PET."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
        n = len(time)  # 31 days
        forcing = ForcingData(
            time=time,
            precip=np.ones(n),
            pet=np.ones(n) * 3.0,  # 3mm/day
        )
        monthly = forcing.aggregate(Resolution.monthly)
        assert monthly.pet[0] == pytest.approx(93.0, rel=1e-3)  # 31 * 3

    def test_aggregate_means_temp(self) -> None:
        """Aggregate means temperature."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
        n = len(time)
        forcing = ForcingData(
            time=time,
            precip=np.ones(n),
            pet=np.ones(n),
            temp=np.ones(n) * 15.0,  # Constant 15C
        )
        monthly = forcing.aggregate(Resolution.monthly)
        assert monthly.temp is not None
        assert monthly.temp[0] == pytest.approx(15.0, rel=1e-3)

    def test_aggregate_handles_no_temp(self) -> None:
        """Aggregate works without temperature."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2020-04-01", dtype="datetime64[D]")
        n = len(time)
        forcing = ForcingData(
            time=time,
            precip=np.ones(n),
            pet=np.ones(n),
            temp=None,
        )
        monthly = forcing.aggregate(Resolution.monthly)
        assert monthly.temp is None

    def test_aggregate_custom_method(self) -> None:
        """Aggregate accepts custom aggregation methods."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")
        n = len(time)
        forcing = ForcingData(
            time=time,
            precip=np.ones(n) * 10.0,
            pet=np.ones(n) * 3.0,
        )
        monthly = forcing.aggregate(Resolution.monthly, methods={"precip": "mean"})
        assert monthly.precip[0] == pytest.approx(10.0, rel=1e-3)  # mean, not sum

    def test_aggregate_rejects_same_resolution(self) -> None:
        """Cannot aggregate to same resolution."""
        from pydrology import Resolution

        forcing = ForcingData(
            time=_make_dates(30),
            precip=np.zeros(30),
            pet=np.zeros(30),
        )
        with pytest.raises(ValueError, match="target must be coarser"):
            forcing.aggregate(Resolution.daily)

    def test_aggregate_rejects_finer_resolution(self) -> None:
        """Cannot aggregate to finer resolution."""
        from pydrology import Resolution

        time = np.arange("2020-01", "2020-07", dtype="datetime64[M]")
        forcing = ForcingData(
            time=time,
            precip=np.zeros(6),
            pet=np.zeros(6),
            resolution=Resolution.monthly,
        )
        with pytest.raises(ValueError, match="target must be coarser"):
            forcing.aggregate(Resolution.daily)

    def test_aggregate_returns_new_forcingdata(self) -> None:
        """Aggregate returns a new ForcingData instance."""
        from pydrology import Resolution

        time = np.arange("2020-01-01", "2020-04-01", dtype="datetime64[D]")
        n = len(time)
        forcing = ForcingData(
            time=time,
            precip=np.ones(n),
            pet=np.ones(n),
        )
        monthly = forcing.aggregate(Resolution.monthly)
        assert monthly is not forcing
        assert isinstance(monthly, ForcingData)

    def test_aggregate_monthly_to_annual(self) -> None:
        """Aggregates monthly data to annual."""
        from pydrology import Resolution

        time = np.arange("2020-01", "2022-01", dtype="datetime64[M]")
        n = len(time)  # 24 months
        forcing = ForcingData(
            time=time,
            precip=np.ones(n) * 50.0,  # 50mm/month
            pet=np.ones(n) * 30.0,  # 30mm/month
            resolution=Resolution.monthly,
        )
        annual = forcing.aggregate(Resolution.annual)
        assert annual.resolution == Resolution.annual
        assert len(annual) == 2
        # 12 months * 50mm = 600mm per year
        assert annual.precip[0] == pytest.approx(600.0, rel=1e-2)


class TestPrecipGradientType:
    """Tests for PrecipGradientType enum and Catchment integration."""

    def test_default_is_exponential(self) -> None:
        catchment = Catchment(mean_annual_solid_precip=150.0)
        assert catchment.precip_gradient_type == PrecipGradientType.exponential

    def test_linear_accepted(self) -> None:
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            precip_gradient_type=PrecipGradientType.linear,
        )
        assert catchment.precip_gradient_type == PrecipGradientType.linear

    def test_enum_values(self) -> None:
        assert PrecipGradientType.exponential.value == "exponential"
        assert PrecipGradientType.linear.value == "linear"


class TestMultiLayerWarnings:
    """Tests for multi-layer soft validation warnings."""

    def _make_hypsometric(self, min_elev: float = 200.0, max_elev: float = 2000.0) -> np.ndarray:
        return np.linspace(min_elev, max_elev, 101)

    def test_warns_n_layers_exceeds_five(self) -> None:
        with pytest.warns(UserWarning, match="exceeds the recommended maximum of 5"):
            Catchment(
                mean_annual_solid_precip=150.0,
                hypsometric_curve=self._make_hypsometric(),
                input_elevation=600.0,
                n_layers=6,
            )

    def test_no_warning_five_layers(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Catchment(
                mean_annual_solid_precip=150.0,
                hypsometric_curve=self._make_hypsometric(200.0, 800.0),
                input_elevation=600.0,
                n_layers=5,
            )

    def test_warns_precip_amplification_exceeds_2x(self) -> None:
        # Large elevation range -> large amplification
        with pytest.warns(UserWarning, match="amplification factor"):
            Catchment(
                mean_annual_solid_precip=150.0,
                hypsometric_curve=self._make_hypsometric(200.0, 4000.0),
                input_elevation=500.0,
                n_layers=3,
            )

    def test_no_warning_small_elevation_range(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Catchment(
                mean_annual_solid_precip=150.0,
                hypsometric_curve=self._make_hypsometric(400.0, 800.0),
                input_elevation=600.0,
                n_layers=3,
            )

    def test_amplification_warning_respects_custom_gradient(self) -> None:
        # Small custom gradient -> no amplification warning even with big range
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Catchment(
                mean_annual_solid_precip=150.0,
                hypsometric_curve=self._make_hypsometric(200.0, 3000.0),
                input_elevation=500.0,
                n_layers=3,
                precip_gradient=0.0001,  # very small -> amplification < 2x
            )

    def test_amplification_warning_respects_elevation_cap(self) -> None:
        # Above the cap, elevation differences don't increase amplification
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Input at 3800, max at 4200 -> both effectively capped near 4000
            Catchment(
                mean_annual_solid_precip=150.0,
                hypsometric_curve=self._make_hypsometric(3800.0, 4200.0),
                input_elevation=3800.0,
                n_layers=2,
            )
