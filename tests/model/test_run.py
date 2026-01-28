"""Integration tests for gr6j.model.run module.

Tests the main orchestration functions step() and run() which execute
the complete GR6J model for single timesteps and timeseries respectively.
"""

import numpy as np
import pandas as pd
import pytest

from gr6j import Catchment, CemaNeige, ForcingData, ModelOutput, Parameters, State
from gr6j.cemaneige import CemaNeigeSingleLayerState
from gr6j.model.run import run, step

# Expected flux keys returned by step() - all 20 MISC outputs
EXPECTED_FLUX_KEYS = {
    "pet",
    "precip",
    "production_store",
    "net_rainfall",
    "storage_infiltration",
    "actual_et",
    "percolation",
    "effective_rainfall",
    "q9",
    "q1",
    "routing_store",
    "exchange",
    "actual_exchange_routing",
    "actual_exchange_direct",
    "actual_exchange_total",
    "qr",
    "qrexp",
    "exponential_store",
    "qd",
    "streamflow",
}

# Expected CemaNeige snow flux keys - all 12 snow outputs (including precip_raw)
EXPECTED_SNOW_FLUX_KEYS = {
    "precip_raw",
    "snow_pliq",
    "snow_psol",
    "snow_pack",
    "snow_thermal_state",
    "snow_gratio",
    "snow_pot_melt",
    "snow_melt",
    "snow_pliq_and_melt",
    "snow_temp",
    "snow_gthreshold",
    "snow_glocalmax",
}


@pytest.fixture
def typical_params() -> Parameters:
    """Typical GR6J parameters within valid ranges."""
    return Parameters(
        x1=350.0,  # Production store capacity [mm]
        x2=0.0,  # No intercatchment exchange
        x3=90.0,  # Routing store capacity [mm]
        x4=1.7,  # Unit hydrograph time constant [days]
        x5=0.0,  # Exchange threshold
        x6=5.0,  # Exponential store scale [mm]
    )


@pytest.fixture
def simple_forcing() -> ForcingData:
    """Simple ForcingData with precip and pet for 5 days."""
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=5, freq="D").values,
        precip=np.array([10.0, 5.0, 0.0, 15.0, 2.0]),
        pet=np.array([3.0, 4.0, 5.0, 2.0, 3.5]),
    )


@pytest.fixture
def initialized_state(typical_params: Parameters) -> State:
    """Initial model state from typical parameters."""
    return State.initialize(typical_params)


@pytest.fixture
def uh_ordinates(typical_params: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Pre-computed unit hydrograph ordinates."""
    from gr6j.model.unit_hydrographs import compute_uh_ordinates

    return compute_uh_ordinates(typical_params.x4)


class TestStep:
    """Tests for the step() function - single timestep execution."""

    def test_returns_new_state_and_fluxes(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_ordinates: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify step returns a tuple of (State, dict)."""
        uh1_ord, uh2_ord = uh_ordinates

        result = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            uh1_ordinates=uh1_ord,
            uh2_ordinates=uh2_ord,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        new_state, fluxes = result
        assert isinstance(new_state, State)
        assert isinstance(fluxes, dict)

    def test_state_is_new_instance(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_ordinates: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Original state should remain unchanged after step."""
        uh1_ord, uh2_ord = uh_ordinates

        # Store original values
        original_prod_store = initialized_state.production_store
        original_routing_store = initialized_state.routing_store
        original_exp_store = initialized_state.exponential_store
        original_uh1 = initialized_state.uh1_states.copy()
        original_uh2 = initialized_state.uh2_states.copy()

        new_state, _ = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            uh1_ordinates=uh1_ord,
            uh2_ordinates=uh2_ord,
        )

        # Verify original state is unchanged
        assert initialized_state.production_store == original_prod_store
        assert initialized_state.routing_store == original_routing_store
        assert initialized_state.exponential_store == original_exp_store
        np.testing.assert_array_equal(initialized_state.uh1_states, original_uh1)
        np.testing.assert_array_equal(initialized_state.uh2_states, original_uh2)

        # Verify new state is a different instance
        assert new_state is not initialized_state

    def test_fluxes_contains_all_expected_keys(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_ordinates: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Fluxes dictionary should contain all 20 MISC outputs."""
        uh1_ord, uh2_ord = uh_ordinates

        _, fluxes = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            uh1_ordinates=uh1_ord,
            uh2_ordinates=uh2_ord,
        )

        assert set(fluxes.keys()) == EXPECTED_FLUX_KEYS

        # All values should be floats
        for key, value in fluxes.items():
            assert isinstance(value, float), f"Flux '{key}' is not a float: {type(value)}"

    def test_streamflow_is_non_negative(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_ordinates: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Streamflow (Q) should always be >= 0."""
        uh1_ord, uh2_ord = uh_ordinates

        # Test with various input combinations
        test_inputs = [
            (10.0, 3.0),  # Normal rainfall
            (0.0, 5.0),  # Dry day with high PET
            (50.0, 0.0),  # Heavy rain, no PET
            (0.0, 0.0),  # No inputs
            (1.0, 10.0),  # Low rainfall, high PET
        ]

        state = initialized_state
        for precip, pet in test_inputs:
            state, fluxes = step(
                state=state,
                params=typical_params,
                precip=precip,
                pet=pet,
                uh1_ordinates=uh1_ord,
                uh2_ordinates=uh2_ord,
            )

            assert fluxes["streamflow"] >= 0.0, f"Negative streamflow {fluxes['streamflow']} for P={precip}, E={pet}"

    def test_zero_inputs_produce_valid_output(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_ordinates: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Zero precipitation and PET should produce valid (non-NaN) outputs."""
        uh1_ord, uh2_ord = uh_ordinates

        new_state, fluxes = step(
            state=initialized_state,
            params=typical_params,
            precip=0.0,
            pet=0.0,
            uh1_ordinates=uh1_ord,
            uh2_ordinates=uh2_ord,
        )

        # All flux values should be finite (not NaN or inf)
        for key, value in fluxes.items():
            assert np.isfinite(value), f"Flux '{key}' is not finite: {value}"

        # State values should be finite
        assert np.isfinite(new_state.production_store)
        assert np.isfinite(new_state.routing_store)
        assert np.isfinite(new_state.exponential_store)
        assert np.all(np.isfinite(new_state.uh1_states))
        assert np.all(np.isfinite(new_state.uh2_states))

        # With zero inputs and initial state, some fluxes should be zero
        assert fluxes["net_rainfall"] == 0.0
        assert fluxes["storage_infiltration"] == 0.0


class TestRun:
    """Tests for the run() function - timeseries execution."""

    def test_returns_model_output_with_correct_fields(
        self,
        typical_params: Parameters,
        simple_forcing: ForcingData,
    ) -> None:
        """Result should be ModelOutput with all expected GR6J fields."""
        result = run(typical_params, simple_forcing)

        assert isinstance(result, ModelOutput)
        assert set(result.gr6j.to_dict().keys()) == EXPECTED_FLUX_KEYS

    def test_output_length_matches_input(
        self,
        typical_params: Parameters,
    ) -> None:
        """Output should have the same length as input forcing."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.random.default_rng(42).uniform(0, 20, 10),
            pet=np.random.default_rng(42).uniform(1, 6, 10),
        )

        result = run(typical_params, forcing)

        assert len(result) == len(forcing)

    def test_uses_provided_initial_state(
        self,
        typical_params: Parameters,
        simple_forcing: ForcingData,
    ) -> None:
        """Custom initial state should be respected."""
        # Create a custom initial state with specific values
        custom_state = State(
            production_store=200.0,  # Different from 0.3 * x1 = 105
            routing_store=60.0,  # Different from 0.5 * x3 = 45
            exponential_store=10.0,  # Different from 0
            uh1_states=np.zeros(20),
            uh2_states=np.zeros(40),
        )

        result_custom = run(typical_params, simple_forcing, initial_state=custom_state)
        result_default = run(typical_params, simple_forcing)

        # Results should differ due to different initial states
        # Compare first row streamflow - should be different
        assert result_custom.gr6j.streamflow[0] != result_default.gr6j.streamflow[0]

    def test_uses_default_initial_state_when_none(
        self,
        typical_params: Parameters,
        simple_forcing: ForcingData,
    ) -> None:
        """When initial_state is None, State.initialize should be used."""
        # Run with explicit None
        result_none = run(typical_params, simple_forcing, initial_state=None)

        # Run with explicit initialized state (should match)
        default_state = State.initialize(typical_params)
        result_explicit = run(typical_params, simple_forcing, initial_state=default_state)

        # Results should be identical
        np.testing.assert_array_equal(result_none.gr6j.streamflow, result_explicit.gr6j.streamflow)

    def test_multi_timestep_simulation(
        self,
        typical_params: Parameters,
    ) -> None:
        """Run simulation for 10+ days and verify outputs are reasonable."""
        # Create 15 days of synthetic input data
        n_days = 15
        rng = np.random.default_rng(42)  # For reproducibility
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 25, n_days),
            pet=rng.uniform(2, 6, n_days),
        )

        result = run(typical_params, forcing)

        # Verify correct length
        assert len(result) == n_days

        # Streamflow should be non-negative throughout
        assert (result.gr6j.streamflow >= 0).all()

        # All values should be finite
        for key, values in result.gr6j.to_dict().items():
            assert np.all(np.isfinite(values)), f"Field '{key}' has non-finite values"

        # Production store should stay within bounds [0, x1]
        assert (result.gr6j.production_store >= 0).all()
        assert (result.gr6j.production_store <= typical_params.x1).all()

        # Routing store should stay within bounds [0, x3]
        assert (result.gr6j.routing_store >= 0).all()
        assert (result.gr6j.routing_store <= typical_params.x3).all()

        # Verify water balance makes sense: inputs should produce some outputs
        total_precip = forcing.precip.sum()
        total_streamflow = result.gr6j.streamflow.sum()
        total_et = result.gr6j.actual_et.sum()

        # Over a reasonable period, outputs should be > 0 given positive inputs
        assert total_streamflow > 0, "No streamflow generated despite precipitation"
        assert total_et > 0, "No ET occurred despite PET demand"

        # Total outputs should not exceed total inputs (mass balance check)
        # Allow some tolerance due to storage changes
        storage_change = (
            result.gr6j.production_store[-1]
            - 0.3 * typical_params.x1
            + result.gr6j.routing_store[-1]
            - 0.5 * typical_params.x3
            + result.gr6j.exponential_store[-1]
        )
        # Rough mass balance: P = Q + ET + dS (ignoring exchange for x2=0)
        mass_balance_error = abs(total_precip - total_streamflow - total_et - storage_change)
        # Error should be small relative to total inputs
        assert mass_balance_error < 0.1 * total_precip, (
            f"Mass balance error too large: {mass_balance_error:.2f} mm "
            f"(precip={total_precip:.2f}, Q={total_streamflow:.2f}, ET={total_et:.2f})"
        )


class TestRunWithSnow:
    """Tests for run() with CemaNeige snow module integration."""

    @pytest.fixture
    def typical_snow_params(self) -> Parameters:
        """Typical GR6J parameters with CemaNeige snow module."""
        return Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            snow=CemaNeige(ctg=0.97, kf=2.5),
        )

    @pytest.fixture
    def typical_catchment(self) -> Catchment:
        """Typical catchment with snow module properties."""
        return Catchment(mean_annual_solid_precip=150.0)

    @pytest.fixture
    def forcing_with_temp(self) -> ForcingData:
        """ForcingData with precip, pet, and temp for 5 days."""
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=5, freq="D").values,
            precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
            pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
            temp=np.array([-5.0, 0.0, 5.0, -2.0, 8.0]),
        )

    def test_backward_compatibility_without_snow(
        self,
        typical_params: Parameters,
        simple_forcing: ForcingData,
    ) -> None:
        """Existing run() calls work unchanged without snow parameter."""
        result = run(typical_params, simple_forcing)

        assert isinstance(result, ModelOutput)
        assert set(result.gr6j.to_dict().keys()) == EXPECTED_FLUX_KEYS
        assert result.snow is None

    def test_snow_parameter_adds_snow_output(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """When snow is enabled, output has snow field populated."""
        result = run(typical_snow_params, forcing_with_temp, catchment=typical_catchment)

        assert result.snow is not None
        assert set(result.snow.to_dict().keys()) == EXPECTED_SNOW_FLUX_KEYS

    def test_raises_when_temp_missing_with_snow(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
        simple_forcing: ForcingData,
    ) -> None:
        """ValueError when snow enabled but temp field missing."""
        # simple_forcing has only precip and pet, no temp
        with pytest.raises(ValueError, match="forcing.temp required when snow module enabled"):
            run(typical_snow_params, simple_forcing, catchment=typical_catchment)

    def test_raises_when_catchment_missing_with_snow(
        self,
        typical_snow_params: Parameters,
        forcing_with_temp: ForcingData,
    ) -> None:
        """ValueError when snow enabled but catchment not provided."""
        with pytest.raises(ValueError, match="catchment required when snow module enabled"):
            run(typical_snow_params, forcing_with_temp)

    def test_precip_raw_equals_input_precip(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """precip_raw field matches original input precipitation."""
        result = run(typical_snow_params, forcing_with_temp, catchment=typical_catchment)

        np.testing.assert_array_almost_equal(
            result.snow.precip_raw,
            forcing_with_temp.precip,
        )

    def test_precip_differs_from_precip_raw_with_snow(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """precip to GR6J differs from precip_raw (snow preprocessing)."""
        result = run(typical_snow_params, forcing_with_temp, catchment=typical_catchment)

        # At least some values should differ (cold days accumulate snow)
        # Not all precip passes through unchanged
        assert not np.allclose(result.gr6j.precip, result.snow.precip_raw)

    def test_cold_day_snow_accumulates(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Cold days with precip accumulate snow."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=3, freq="D").values,
            precip=np.array([10.0, 10.0, 10.0]),
            pet=np.array([2.0, 2.0, 2.0]),
            temp=np.array([-10.0, -10.0, -10.0]),  # Very cold
        )

        result = run(typical_snow_params, forcing, catchment=typical_catchment)

        # Snow pack should increase over cold period
        assert result.snow.snow_pack[-1] > result.snow.snow_pack[0]
        # All precip should be snow (solid_fraction = 1.0)
        np.testing.assert_array_almost_equal(
            result.snow.snow_psol,
            forcing.precip,
        )

    def test_warm_period_snow_passes_through(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Warm period with no snow: precip passes through unchanged."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=3, freq="D").values,
            precip=np.array([10.0, 10.0, 10.0]),
            pet=np.array([2.0, 2.0, 2.0]),
            temp=np.array([15.0, 15.0, 15.0]),  # Warm: all rain
        )

        result = run(typical_snow_params, forcing, catchment=typical_catchment)

        # All precip should be liquid rain
        np.testing.assert_array_almost_equal(
            result.snow.snow_pliq,
            forcing.precip,
        )
        # Snow pack should be zero
        assert (result.snow.snow_pack == 0.0).all()
        # pliq_and_melt = precip (no melt since no snow)
        np.testing.assert_array_almost_equal(
            result.snow.snow_pliq_and_melt,
            forcing.precip,
        )

    def test_snow_melt_produces_streamflow(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Snow accumulation followed by warm period produces melt."""
        # Longer simulation: snow accumulates, then melts over extended warm period
        cold_days = 5
        warm_days = 10
        n_days = cold_days + warm_days
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=np.array([20.0] * cold_days + [0.0] * warm_days),  # Precip then dry
            pet=np.array([1.0] * cold_days + [4.0] * warm_days),
            temp=np.array([-10.0] * cold_days + [10.0] * warm_days),  # Cold then warm
        )

        result = run(typical_snow_params, forcing, catchment=typical_catchment)

        # Snow should accumulate during cold period
        assert result.snow.snow_pack[cold_days - 1] > 0.0
        # Melt should occur during warm period (may need thermal state to warm up)
        assert result.snow.snow_melt[cold_days:].sum() > 0.0
        # Streamflow should be produced
        assert result.gr6j.streamflow.sum() > 0.0

    def test_uses_custom_initial_snow_state(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """Custom initial_snow_state is respected."""
        # Custom state with existing snow pack
        custom_snow_state = CemaNeigeSingleLayerState(
            g=100.0,  # 100mm snow pack
            etg=0.0,  # At melting point
            gthreshold=135.0,
            glocalmax=135.0,
        )

        result_custom = run(
            typical_snow_params,
            forcing_with_temp,
            catchment=typical_catchment,
            initial_snow_state=custom_snow_state,
        )
        result_default = run(
            typical_snow_params,
            forcing_with_temp,
            catchment=typical_catchment,
        )

        # First row should differ due to different initial snow
        assert result_custom.snow.snow_pack[0] != result_default.snow.snow_pack[0]

    def test_output_length_matches_input_with_snow(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Output length matches input when snow enabled."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=5, freq="D").values,
            precip=np.array([10.0] * 5),
            pet=np.array([3.0] * 5),
            temp=np.array([0.0] * 5),
        )

        result = run(typical_snow_params, forcing, catchment=typical_catchment)

        assert len(result) == len(forcing)

    def test_all_values_finite_with_snow(
        self,
        typical_snow_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """All output values are finite when snow enabled."""
        result = run(typical_snow_params, forcing_with_temp, catchment=typical_catchment)

        for key, values in result.gr6j.to_dict().items():
            assert np.all(np.isfinite(values)), f"GR6J field '{key}' has non-finite values"

        for key, values in result.snow.to_dict().items():
            assert np.all(np.isfinite(values)), f"Snow field '{key}' has non-finite values"
