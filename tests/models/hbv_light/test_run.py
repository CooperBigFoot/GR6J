"""Integration tests for pydrology.models.hbv_light.run module."""

import numpy as np
import pandas as pd
import pytest

from pydrology import Catchment, ForcingData, ModelOutput
from pydrology.models.hbv_light import (
    HBVLightFluxes,
    Parameters,
    State,
    compute_triangular_weights,
    run,
    step,
)

# Expected flux keys
EXPECTED_FLUX_KEYS = {
    "precip",
    "temp",
    "pet",
    "precip_rain",
    "precip_snow",
    "snow_pack",
    "snow_melt",
    "liquid_water_in_snow",
    "snow_input",
    "soil_moisture",
    "recharge",
    "actual_et",
    "upper_zone",
    "lower_zone",
    "q0",
    "q1",
    "q2",
    "percolation",
    "qgw",
    "streamflow",
}


@pytest.fixture
def typical_params() -> Parameters:
    """Typical HBV-light parameters."""
    return Parameters(
        tt=0.0,
        cfmax=3.0,
        sfcf=1.0,
        cwh=0.1,
        cfr=0.05,
        fc=250.0,
        lp=0.9,
        beta=2.0,
        k0=0.4,
        k1=0.1,
        k2=0.01,
        perc=1.0,
        uzl=20.0,
        maxbas=2.5,
    )


@pytest.fixture
def simple_forcing() -> ForcingData:
    """Simple forcing data for 5 days."""
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=5, freq="D").values,
        precip=np.array([10.0, 5.0, 0.0, 15.0, 2.0]),
        pet=np.array([3.0, 4.0, 5.0, 2.0, 3.5]),
        temp=np.array([5.0, 2.0, -5.0, 3.0, 1.0]),
    )


@pytest.fixture
def initialized_state(typical_params: Parameters) -> State:
    return State.initialize(typical_params)


@pytest.fixture
def uh_weights(typical_params: Parameters) -> np.ndarray:
    return compute_triangular_weights(typical_params.maxbas)


class TestStep:
    """Tests for the step() function."""

    def test_returns_state_and_fluxes(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_weights: np.ndarray,
    ) -> None:
        """step returns a tuple of (State, dict)."""
        result = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            temp=5.0,
            uh_weights=uh_weights,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        new_state, fluxes = result
        assert isinstance(new_state, State)
        assert isinstance(fluxes, dict)

    def test_state_immutability(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_weights: np.ndarray,
    ) -> None:
        """Original state should remain unchanged."""
        original_sm = initialized_state.zone_states[0, 2]
        original_suz = initialized_state.upper_zone

        new_state, _ = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            temp=5.0,
            uh_weights=uh_weights,
        )

        assert initialized_state.zone_states[0, 2] == original_sm
        assert initialized_state.upper_zone == original_suz
        assert new_state is not initialized_state

    def test_fluxes_contains_all_keys(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_weights: np.ndarray,
    ) -> None:
        """Fluxes dict contains all expected keys."""
        _, fluxes = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            temp=5.0,
            uh_weights=uh_weights,
        )

        assert set(fluxes.keys()) == EXPECTED_FLUX_KEYS
        for key, value in fluxes.items():
            assert isinstance(value, float), f"Flux '{key}' is not a float"


class TestStepTemperatureRequirement:
    """Tests for temperature validation."""

    # step() function accepts temp as a required float parameter, so no validation needed there
    # The validation happens in run() when forcing.temp is None
    pass


class TestRun:
    """Tests for the run() function."""

    def test_returns_model_output(self, typical_params: Parameters, simple_forcing: ForcingData) -> None:
        """run returns ModelOutput with correct fields."""
        result = run(typical_params, simple_forcing)

        assert isinstance(result, ModelOutput)
        assert isinstance(result.fluxes, HBVLightFluxes)
        assert set(result.fluxes.to_dict().keys()) == EXPECTED_FLUX_KEYS

    def test_output_length_matches_input(self, typical_params: Parameters) -> None:
        """Output has same length as input."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.random.default_rng(42).uniform(0, 20, 10),
            pet=np.random.default_rng(42).uniform(1, 6, 10),
            temp=np.random.default_rng(42).uniform(-5, 15, 10),
        )

        result = run(typical_params, forcing)
        assert len(result) == len(forcing)

    def test_uses_initial_state(self, typical_params: Parameters, simple_forcing: ForcingData) -> None:
        """Custom initial state is respected."""
        custom_state = State.initialize(typical_params)
        custom_state.zone_states[0, 2] = 200.0  # Different SM
        custom_state.upper_zone = 50.0

        result_custom = run(typical_params, simple_forcing, initial_state=custom_state)
        result_default = run(typical_params, simple_forcing)

        # First streamflow is from routing buffer (starts at 0 for both).
        # Check qgw (unrouted) which reflects the different upper zone.
        assert result_custom.fluxes.qgw[0] != result_default.fluxes.qgw[0]

    def test_multi_timestep_simulation(self, typical_params: Parameters) -> None:
        """Multi-day simulation produces valid outputs."""
        n_days = 15
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 25, n_days),
            pet=rng.uniform(2, 6, n_days),
            temp=rng.uniform(-10, 20, n_days),
        )

        result = run(typical_params, forcing)

        assert len(result) == n_days
        assert (result.streamflow >= 0).all()

        for key, values in result.fluxes.to_dict().items():
            assert np.all(np.isfinite(values)), f"Field '{key}' has non-finite values"

    def test_requires_temperature(self, typical_params: Parameters) -> None:
        """run raises ValueError if temperature is missing."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=5, freq="D").values,
            precip=np.array([10.0, 5.0, 0.0, 15.0, 2.0]),
            pet=np.array([3.0, 4.0, 5.0, 2.0, 3.5]),
            temp=None,
        )

        with pytest.raises(ValueError, match="requires temperature"):
            run(typical_params, forcing)


class TestStoreConstraints:
    """Tests for store constraint enforcement."""

    def test_soil_moisture_bounded_by_fc(self, typical_params: Parameters) -> None:
        """Soil moisture should stay within [0, FC]."""
        n_days = 30
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 50, n_days),
            pet=rng.uniform(0, 10, n_days),
            temp=rng.uniform(-5, 25, n_days),
        )

        result = run(typical_params, forcing)

        assert (result.fluxes.soil_moisture >= 0).all()
        assert (result.fluxes.soil_moisture <= typical_params.fc).all()

    def test_stores_non_negative(self, typical_params: Parameters) -> None:
        """Groundwater stores should be non-negative."""
        n_days = 30
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 30, n_days),
            pet=rng.uniform(0, 8, n_days),
            temp=rng.uniform(-10, 20, n_days),
        )

        result = run(typical_params, forcing)

        assert (result.fluxes.upper_zone >= 0).all()
        assert (result.fluxes.lower_zone >= 0).all()
        assert (result.fluxes.snow_pack >= 0).all()


class TestMassBalance:
    """Tests for water balance."""

    def test_water_balance(self, typical_params: Parameters) -> None:
        """Verify approximate water balance."""
        n_days = 100
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 20, n_days),
            pet=rng.uniform(2, 5, n_days),
            temp=rng.uniform(5, 20, n_days),  # No snow for simpler balance
        )

        initial_state = State.initialize(typical_params)
        result = run(typical_params, forcing, initial_state=initial_state)

        total_precip = forcing.precip.sum()
        total_streamflow = result.streamflow.sum()
        total_et = result.fluxes.actual_et.sum()

        # Storage changes
        initial_sm = 0.5 * typical_params.fc
        final_sm = result.fluxes.soil_moisture[-1]
        final_suz = result.fluxes.upper_zone[-1]
        final_slz = result.fluxes.lower_zone[-1]

        storage_change = (final_sm - initial_sm) + final_suz + final_slz

        # P = Q + ET + dS
        mass_balance_error = abs(total_precip - total_streamflow - total_et - storage_change)

        # Allow for routing buffer residual
        assert mass_balance_error < 0.1 * total_precip


class TestOutputFiniteness:
    """Tests for numerical stability."""

    def test_no_nan_or_inf(self, typical_params: Parameters) -> None:
        """No NaN or inf in outputs."""
        n_days = 50
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 30, n_days),
            pet=rng.uniform(0, 8, n_days),
            temp=rng.uniform(-20, 30, n_days),
        )

        result = run(typical_params, forcing)

        for key, values in result.fluxes.to_dict().items():
            assert np.all(np.isfinite(values)), f"Field '{key}' has NaN or inf"


class TestRunMultiZone:
    """Tests for multi-zone HBV-light execution."""

    @pytest.fixture
    def multi_zone_catchment(self) -> Catchment:
        """5-zone catchment with hypsometric curve."""
        return Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=5,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )

    @pytest.fixture
    def forcing_with_snow(self) -> ForcingData:
        """10-day forcing with varied temperatures for snow testing."""
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").to_numpy(),
            precip=np.array([10.0, 15.0, 5.0, 0.0, 20.0, 10.0, 0.0, 5.0, 15.0, 8.0]),
            pet=np.array([2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0]),
            temp=np.array([-5.0, -3.0, 0.0, 5.0, -8.0, -2.0, 3.0, 7.0, -1.0, 10.0]),
        )

    def test_run_multi_zone_output_length(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_with_snow: ForcingData,
    ) -> None:
        """run() with multi-zone catchment produces output of correct length."""
        result = run(typical_params, forcing_with_snow, catchment=multi_zone_catchment)

        assert len(result) == 10

    def test_run_multi_zone_produces_zone_outputs(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_with_snow: ForcingData,
    ) -> None:
        """run() with multi-zone catchment populates zone_outputs."""
        result = run(typical_params, forcing_with_snow, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None
        assert result.zone_outputs.n_zones == 5

    def test_higher_zones_colder_temperatures(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_with_snow: ForcingData,
    ) -> None:
        """Higher elevation zones receive colder extrapolated temperatures."""
        result = run(typical_params, forcing_with_snow, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None
        zone_temps = result.zone_outputs.zone_temp  # shape (10, 5)

        # At each timestep, zone temperatures should decrease with zone index
        for t in range(10):
            for z in range(4):
                assert zone_temps[t, z] > zone_temps[t, z + 1], \
                    f"Zone {z} should be warmer than zone {z+1} at timestep {t}"

    def test_single_zone_no_zone_outputs(
        self,
        typical_params: Parameters,
        simple_forcing: ForcingData,
    ) -> None:
        """Single-zone run produces no zone_outputs."""
        result = run(typical_params, simple_forcing)

        assert result.zone_outputs is None

    def test_single_zone_backward_compatible(
        self,
        typical_params: Parameters,
        simple_forcing: ForcingData,
    ) -> None:
        """Single-zone run with catchment produces same results as without."""
        # Run without catchment
        result_no_catchment = run(typical_params, simple_forcing)

        # Run with single-zone catchment (no elevation info)
        single_zone_catchment = Catchment(mean_annual_solid_precip=100.0, n_layers=1)
        result_single_zone = run(typical_params, simple_forcing, catchment=single_zone_catchment)

        # Streamflow should be identical
        np.testing.assert_array_almost_equal(
            result_no_catchment.streamflow,
            result_single_zone.streamflow,
        )

    def test_aggregation_is_area_weighted(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_with_snow: ForcingData,
    ) -> None:
        """Aggregated outputs are area-weighted averages of zone outputs."""
        result = run(typical_params, forcing_with_snow, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None

        # For uniform fractions (1/n_zones=0.2), aggregated snow_pack should be
        # mean of zone snow packs
        zone_snow_packs = result.zone_outputs.snow_pack  # shape (10, 5)
        expected_aggregated = np.mean(zone_snow_packs, axis=1)
        np.testing.assert_array_almost_equal(
            result.fluxes.snow_pack, expected_aggregated
        )


class TestZoneOutputs:
    """Tests for per-zone output structure."""

    @pytest.fixture
    def multi_zone_catchment(self) -> Catchment:
        """5-zone catchment."""
        return Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=5,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )

    @pytest.fixture
    def forcing_10_days(self) -> ForcingData:
        """10-day forcing data."""
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").to_numpy(),
            precip=np.full(10, 10.0),
            pet=np.full(10, 3.0),
            temp=np.full(10, 0.0),
        )

    def test_zone_outputs_shape(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Zone output arrays have shape (n_timesteps, n_zones)."""
        result = run(typical_params, forcing_10_days, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None
        assert result.zone_outputs.snow_pack.shape == (10, 5)
        assert result.zone_outputs.soil_moisture.shape == (10, 5)
        assert result.zone_outputs.zone_temp.shape == (10, 5)
        assert result.zone_outputs.zone_precip.shape == (10, 5)

    def test_zone_elevations_monotonic(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Zone elevations are monotonically increasing."""
        result = run(typical_params, forcing_10_days, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None
        elevations = result.zone_outputs.zone_elevations
        for i in range(len(elevations) - 1):
            assert elevations[i] < elevations[i + 1]

    def test_zone_fractions_sum_to_one(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Zone fractions sum to 1.0."""
        result = run(typical_params, forcing_10_days, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None
        assert sum(result.zone_outputs.zone_fractions) == pytest.approx(1.0)

    def test_all_zone_values_finite(
        self,
        typical_params: Parameters,
        multi_zone_catchment: Catchment,
    ) -> None:
        """All zone output values are finite."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=20, freq="D").to_numpy(),
            precip=np.random.default_rng(42).uniform(0, 30, 20),
            pet=np.random.default_rng(42).uniform(0, 8, 20),
            temp=np.random.default_rng(42).uniform(-20, 30, 20),
        )

        result = run(typical_params, forcing, catchment=multi_zone_catchment)

        assert result.zone_outputs is not None
        for key, values in result.zone_outputs.to_dict().items():
            assert np.all(np.isfinite(values)), f"Zone output '{key}' has non-finite values"


class TestValidationErrors:
    """Tests for array validation at the Rust boundary."""

    def test_run_rejects_wrong_length_params(self) -> None:
        """hbv_run raises ValueError for wrong-length params array."""
        from pydrology._core.hbv_light import hbv_run

        wrong_params = np.array([0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1])  # needs 14 elements
        precip = np.array([10.0, 5.0])
        pet = np.array([3.0, 4.0])
        temp = np.array([5.0, 2.0])

        with pytest.raises(ValueError, match="must have"):
            hbv_run(wrong_params, precip, pet, temp)

    def test_run_rejects_wrong_length_state(self) -> None:
        """hbv_run raises ValueError for wrong-length initial_state."""
        from pydrology._core.hbv_light import hbv_run

        params = np.array([0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5])
        precip = np.array([10.0, 5.0])
        pet = np.array([3.0, 4.0])
        temp = np.array([5.0, 2.0])
        wrong_state = np.array([0.0, 0.0, 125.0, 0.0, 0.0])  # needs 12 elements (1 zone * 3 + 9)

        with pytest.raises(ValueError, match="must have"):
            hbv_run(params, precip, pet, temp, initial_state=wrong_state)

    def test_step_rejects_wrong_length_params(self) -> None:
        """hbv_step raises ValueError for wrong-length params array."""
        from pydrology._core.hbv_light import hbv_step

        state = np.zeros(12)
        wrong_params = np.array([0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1])  # needs 14 elements
        uh_weights = np.array([0.25, 0.5, 0.25])

        with pytest.raises(ValueError, match="must have"):
            hbv_step(state, wrong_params, 10.0, 3.0, 5.0, uh_weights)

    def test_step_rejects_wrong_length_state(self) -> None:
        """hbv_step raises ValueError for wrong-length state array."""
        from pydrology._core.hbv_light import hbv_step

        wrong_state = np.array([0.0, 0.0, 125.0, 0.0, 0.0])  # needs 12 elements
        params = np.array([0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5])
        uh_weights = np.array([0.25, 0.5, 0.25])

        with pytest.raises(ValueError, match="must have"):
            hbv_step(wrong_state, params, 10.0, 3.0, 5.0, uh_weights)
