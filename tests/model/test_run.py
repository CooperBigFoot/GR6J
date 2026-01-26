"""Integration tests for gr6j.model.run module.

Tests the main orchestration functions step() and run() which execute
the complete GR6J model for single timesteps and timeseries respectively.
"""

import numpy as np
import pandas as pd
import pytest

from gr6j.model.run import run, step
from gr6j.model.types import Parameters, State

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
def simple_input_df() -> pd.DataFrame:
    """Simple DataFrame with precip and pet columns for 5 days."""
    return pd.DataFrame(
        {
            "precip": [10.0, 5.0, 0.0, 15.0, 2.0],
            "pet": [3.0, 4.0, 5.0, 2.0, 3.5],
        }
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

    def test_returns_dataframe_with_correct_columns(
        self,
        typical_params: Parameters,
        simple_input_df: pd.DataFrame,
    ) -> None:
        """Result DataFrame should have all flux columns."""
        result = run(typical_params, simple_input_df)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == EXPECTED_FLUX_KEYS

    def test_output_index_matches_input(
        self,
        typical_params: Parameters,
    ) -> None:
        """Output DataFrame should have the same index as input."""
        # Create input with a custom index
        custom_index = pd.date_range("2020-01-01", periods=10, freq="D")
        input_df = pd.DataFrame(
            {
                "precip": np.random.uniform(0, 20, 10),
                "pet": np.random.uniform(1, 6, 10),
            },
            index=custom_index,
        )

        result = run(typical_params, input_df)

        pd.testing.assert_index_equal(result.index, input_df.index)

    def test_raises_on_missing_precip_column(
        self,
        typical_params: Parameters,
    ) -> None:
        """Should raise ValueError when 'precip' column is missing."""
        input_df = pd.DataFrame({"pet": [3.0, 4.0, 5.0]})

        with pytest.raises(ValueError, match="precip"):
            run(typical_params, input_df)

    def test_raises_on_missing_pet_column(
        self,
        typical_params: Parameters,
    ) -> None:
        """Should raise ValueError when 'pet' column is missing."""
        input_df = pd.DataFrame({"precip": [10.0, 5.0, 0.0]})

        with pytest.raises(ValueError, match="pet"):
            run(typical_params, input_df)

    def test_uses_provided_initial_state(
        self,
        typical_params: Parameters,
        simple_input_df: pd.DataFrame,
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

        result_custom = run(typical_params, simple_input_df, initial_state=custom_state)
        result_default = run(typical_params, simple_input_df)

        # Results should differ due to different initial states
        # Compare first row streamflow - should be different
        assert result_custom["streamflow"].iloc[0] != result_default["streamflow"].iloc[0]

    def test_uses_default_initial_state_when_none(
        self,
        typical_params: Parameters,
        simple_input_df: pd.DataFrame,
    ) -> None:
        """When initial_state is None, State.initialize should be used."""
        # Run with explicit None
        result_none = run(typical_params, simple_input_df, initial_state=None)

        # Run with explicit initialized state (should match)
        default_state = State.initialize(typical_params)
        result_explicit = run(typical_params, simple_input_df, initial_state=default_state)

        # Results should be identical
        pd.testing.assert_frame_equal(result_none, result_explicit)

    def test_multi_timestep_simulation(
        self,
        typical_params: Parameters,
    ) -> None:
        """Run simulation for 10+ days and verify outputs are reasonable."""
        # Create 15 days of synthetic input data
        n_days = 15
        np.random.seed(42)  # For reproducibility
        input_df = pd.DataFrame(
            {
                "precip": np.random.uniform(0, 25, n_days),
                "pet": np.random.uniform(2, 6, n_days),
            }
        )

        result = run(typical_params, input_df)

        # Verify correct length
        assert len(result) == n_days

        # Streamflow should be non-negative throughout
        assert (result["streamflow"] >= 0).all()

        # All values should be finite
        assert result.notna().all().all()
        for col in result.columns:
            assert np.all(np.isfinite(result[col].values)), f"Column '{col}' has non-finite values"

        # Production store should stay within bounds [0, x1]
        assert (result["production_store"] >= 0).all()
        assert (result["production_store"] <= typical_params.x1).all()

        # Routing store should stay within bounds [0, x3]
        assert (result["routing_store"] >= 0).all()
        assert (result["routing_store"] <= typical_params.x3).all()

        # Verify water balance makes sense: inputs should produce some outputs
        total_precip = input_df["precip"].sum()
        total_streamflow = result["streamflow"].sum()
        total_et = result["actual_et"].sum()

        # Over a reasonable period, outputs should be > 0 given positive inputs
        assert total_streamflow > 0, "No streamflow generated despite precipitation"
        assert total_et > 0, "No ET occurred despite PET demand"

        # Total outputs should not exceed total inputs (mass balance check)
        # Allow some tolerance due to storage changes
        storage_change = (
            result["production_store"].iloc[-1]
            - 0.3 * typical_params.x1
            + result["routing_store"].iloc[-1]
            - 0.5 * typical_params.x3
            + result["exponential_store"].iloc[-1]
        )
        # Rough mass balance: P = Q + ET + dS (ignoring exchange for x2=0)
        mass_balance_error = abs(total_precip - total_streamflow - total_et - storage_change)
        # Error should be small relative to total inputs
        assert mass_balance_error < 0.1 * total_precip, (
            f"Mass balance error too large: {mass_balance_error:.2f} mm "
            f"(precip={total_precip:.2f}, Q={total_streamflow:.2f}, ET={total_et:.2f})"
        )
