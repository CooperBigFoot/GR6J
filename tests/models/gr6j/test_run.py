"""Integration tests for pydrology.models.gr6j.run module.

Tests the main orchestration functions step() and run() which execute
the complete GR6J model for single timesteps and timeseries respectively.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import ForcingData, ModelOutput, Parameters, State
from pydrology.models.gr6j import run, step

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
    from pydrology.processes.unit_hydrographs import compute_uh_ordinates

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
        assert set(result.fluxes.to_dict().keys()) == EXPECTED_FLUX_KEYS

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
        assert result_custom.streamflow[0] != result_default.streamflow[0]

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
        np.testing.assert_array_equal(result_none.streamflow, result_explicit.streamflow)

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
        assert (result.streamflow >= 0).all()

        # All values should be finite
        for key, values in result.fluxes.to_dict().items():
            assert np.all(np.isfinite(values)), f"Field '{key}' has non-finite values"

        # Production store should stay within bounds [0, x1]
        assert (result.fluxes.production_store >= 0).all()
        assert (result.fluxes.production_store <= typical_params.x1).all()

        # Routing store should stay within bounds [0, x3]
        assert (result.fluxes.routing_store >= 0).all()
        assert (result.fluxes.routing_store <= typical_params.x3).all()

        # Verify water balance makes sense: inputs should produce some outputs
        total_precip = forcing.precip.sum()
        total_streamflow = result.streamflow.sum()
        total_et = result.fluxes.actual_et.sum()

        # Over a reasonable period, outputs should be > 0 given positive inputs
        assert total_streamflow > 0, "No streamflow generated despite precipitation"
        assert total_et > 0, "No ET occurred despite PET demand"

        # Total outputs should not exceed total inputs (mass balance check)
        # Allow some tolerance due to storage changes
        storage_change = (
            result.fluxes.production_store[-1]
            - 0.3 * typical_params.x1
            + result.fluxes.routing_store[-1]
            - 0.5 * typical_params.x3
            + result.fluxes.exponential_store[-1]
        )
        # Rough mass balance: P = Q + ET + dS (ignoring exchange for x2=0)
        mass_balance_error = abs(total_precip - total_streamflow - total_et - storage_change)
        # Error should be small relative to total inputs
        assert mass_balance_error < 0.1 * total_precip, (
            f"Mass balance error too large: {mass_balance_error:.2f} mm "
            f"(precip={total_precip:.2f}, Q={total_streamflow:.2f}, ET={total_et:.2f})"
        )


class TestExchangeInvariant:
    """Tests that actual_exchange_total == actual_exchange_routing + actual_exchange_direct."""

    def test_actual_exchange_total_is_sum_of_components(
        self,
        typical_params: Parameters,
    ) -> None:
        """Run model and assert exchange invariant holds element-wise."""
        n_days = 30
        rng = np.random.default_rng(42)

        # Use params with x2 != 0 to get non-zero exchange
        params = Parameters(x1=350.0, x2=1.5, x3=90.0, x4=1.7, x5=0.5, x6=5.0)

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_days, freq="D").values,
            precip=rng.uniform(0, 25, n_days),
            pet=rng.uniform(2, 6, n_days),
        )

        result = run(params, forcing)

        expected = result.fluxes.actual_exchange_routing + result.fluxes.actual_exchange_direct
        np.testing.assert_allclose(
            result.fluxes.actual_exchange_total,
            expected,
            atol=1e-10,
            err_msg="actual_exchange_total != actual_exchange_routing + actual_exchange_direct",
        )


class TestValidationErrors:
    """Tests for array validation at the Rust boundary."""

    def test_run_rejects_wrong_length_params(self) -> None:
        """gr6j_run raises ValueError for wrong-length params array."""
        from pydrology._core.gr6j import gr6j_run

        wrong_params = np.array([350.0, 0.0, 90.0, 1.7])  # needs >= 6 elements
        precip = np.array([10.0, 5.0])
        pet = np.array([3.0, 4.0])

        with pytest.raises(ValueError, match="must have"):
            gr6j_run(wrong_params, precip, pet)

    def test_step_rejects_wrong_length_params(self) -> None:
        """gr6j_step raises ValueError for wrong-length params array."""
        from pydrology._core.gr6j import gr6j_step

        state = np.zeros(63)
        wrong_params = np.array([350.0, 0.0, 90.0, 1.7])  # needs >= 6 elements
        uh1 = np.zeros(20)
        uh2 = np.zeros(40)

        with pytest.raises(ValueError, match="must have"):
            gr6j_step(state, wrong_params, 10.0, 3.0, uh1, uh2)
