"""Integration tests for pydrology.models.gr2m.run module.

Tests the main orchestration functions step() and run() which execute
the complete GR2M model for single timesteps and timeseries respectively.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import ForcingData, ModelOutput, Resolution
from pydrology.models.gr2m import Parameters, State, run, step

# Expected flux keys returned by step() - all 11 MISC outputs
EXPECTED_FLUX_KEYS = {
    "pet",
    "precip",
    "production_store",
    "rainfall_excess",
    "storage_fill",
    "actual_et",
    "percolation",
    "routing_input",
    "routing_store",
    "exchange",
    "streamflow",
}


@pytest.fixture
def typical_params() -> Parameters:
    """Typical GR2M parameters within valid ranges."""
    return Parameters(
        x1=500.0,  # Production store capacity [mm]
        x2=1.0,    # Groundwater exchange coefficient [-]
    )


@pytest.fixture
def monthly_forcing() -> ForcingData:
    """Monthly ForcingData with precip and pet for 12 months."""
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
        precip=np.array([80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),
        pet=np.array([20.0, 25.0, 40.0, 60.0, 80.0, 100.0, 110.0, 100.0, 70.0, 45.0, 25.0, 20.0]),
        resolution=Resolution.monthly,
    )


@pytest.fixture
def initialized_state(typical_params: Parameters) -> State:
    """Initial model state from typical parameters."""
    return State.initialize(typical_params)


class TestStep:
    """Tests for the step() function - single timestep execution."""

    def test_returns_new_state_and_fluxes(
        self,
        initialized_state: State,
        typical_params: Parameters,
    ) -> None:
        """Verify step returns a tuple of (State, dict)."""
        result = step(
            state=initialized_state,
            params=typical_params,
            precip=80.0,
            pet=20.0,
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
    ) -> None:
        """Original state should remain unchanged after step."""
        # Store original values
        original_prod_store = initialized_state.production_store
        original_routing_store = initialized_state.routing_store

        new_state, _ = step(
            state=initialized_state,
            params=typical_params,
            precip=80.0,
            pet=20.0,
        )

        # Verify original state is unchanged
        assert initialized_state.production_store == original_prod_store
        assert initialized_state.routing_store == original_routing_store

        # Verify new state is a different instance
        assert new_state is not initialized_state

    def test_fluxes_contains_all_expected_keys(
        self,
        initialized_state: State,
        typical_params: Parameters,
    ) -> None:
        """Fluxes dictionary should contain all 11 MISC outputs."""
        _, fluxes = step(
            state=initialized_state,
            params=typical_params,
            precip=80.0,
            pet=20.0,
        )

        assert set(fluxes.keys()) == EXPECTED_FLUX_KEYS

        # All values should be floats
        for key, value in fluxes.items():
            assert isinstance(value, float), f"Flux '{key}' is not a float: {type(value)}"

    def test_streamflow_is_non_negative(
        self,
        initialized_state: State,
        typical_params: Parameters,
    ) -> None:
        """Streamflow (Q) should always be >= 0."""
        # Test with various input combinations
        test_inputs = [
            (80.0, 20.0),   # Normal rainfall
            (0.0, 100.0),   # Dry month with high PET
            (200.0, 0.0),   # Heavy rain, no PET
            (0.0, 0.0),     # No inputs
            (10.0, 150.0),  # Low rainfall, high PET
        ]

        state = initialized_state
        for precip, pet in test_inputs:
            state, fluxes = step(
                state=state,
                params=typical_params,
                precip=precip,
                pet=pet,
            )

            assert fluxes["streamflow"] >= 0.0, f"Negative streamflow {fluxes['streamflow']} for P={precip}, E={pet}"

    def test_zero_inputs_produce_valid_output(
        self,
        initialized_state: State,
        typical_params: Parameters,
    ) -> None:
        """Zero precipitation and PET should produce valid (non-NaN) outputs."""
        new_state, fluxes = step(
            state=initialized_state,
            params=typical_params,
            precip=0.0,
            pet=0.0,
        )

        # All flux values should be finite (not NaN or inf)
        for key, value in fluxes.items():
            assert np.isfinite(value), f"Flux '{key}' is not finite: {value}"

        # State values should be finite
        assert np.isfinite(new_state.production_store)
        assert np.isfinite(new_state.routing_store)


class TestRun:
    """Tests for the run() function - timeseries execution."""

    def test_returns_model_output_with_correct_fields(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """Result should be ModelOutput with all expected GR2M fields."""
        result = run(typical_params, monthly_forcing)

        assert isinstance(result, ModelOutput)
        assert set(result.fluxes.to_dict().keys()) == EXPECTED_FLUX_KEYS

    def test_output_length_matches_input(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """Output should have the same length as input forcing."""
        result = run(typical_params, monthly_forcing)
        assert len(result) == len(monthly_forcing)

    def test_uses_provided_initial_state(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """Custom initial state should be respected."""
        # Create a custom initial state with specific values
        custom_state = State(
            production_store=300.0,  # Different from 0.3 * x1 = 150
            routing_store=100.0,     # Different from 0.3 * x1 = 150
        )

        result_custom = run(typical_params, monthly_forcing, initial_state=custom_state)
        result_default = run(typical_params, monthly_forcing)

        # Results should differ due to different initial states
        assert result_custom.streamflow[0] != result_default.streamflow[0]

    def test_uses_default_initial_state_when_none(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """When initial_state is None, State.initialize should be used."""
        # Run with explicit None
        result_none = run(typical_params, monthly_forcing, initial_state=None)

        # Run with explicit initialized state (should match)
        default_state = State.initialize(typical_params)
        result_explicit = run(typical_params, monthly_forcing, initial_state=default_state)

        # Results should be identical
        np.testing.assert_array_equal(result_none.streamflow, result_explicit.streamflow)


class TestResolutionValidation:
    """Tests for temporal resolution validation."""

    def test_accepts_monthly_resolution(
        self,
        typical_params: Parameters,
    ) -> None:
        """GR2M should accept monthly resolution forcing."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=6, freq="MS").values,
            precip=np.full(6, 50.0),
            pet=np.full(6, 30.0),
            resolution=Resolution.monthly,
        )

        result = run(typical_params, forcing)
        assert len(result) == 6

    def test_rejects_daily_resolution(
        self,
        typical_params: Parameters,
    ) -> None:
        """GR2M should reject daily resolution forcing."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=30, freq="D").values,
            precip=np.full(30, 5.0),
            pet=np.full(30, 3.0),
            resolution=Resolution.daily,
        )

        with pytest.raises(ValueError, match="GR2M supports resolutions"):
            run(typical_params, forcing)

    def test_rejects_hourly_resolution(
        self,
        typical_params: Parameters,
    ) -> None:
        """GR2M should reject hourly resolution forcing."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=24, freq="h").values,
            precip=np.full(24, 1.0),
            pet=np.full(24, 0.5),
            resolution=Resolution.hourly,
        )

        with pytest.raises(ValueError, match="GR2M supports resolutions"):
            run(typical_params, forcing)


class TestPhysicalConstraints:
    """Tests for physical constraints of the model."""

    def test_production_store_bounds(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """Production store should stay within reasonable bounds."""
        result = run(typical_params, monthly_forcing)

        # Store should be non-negative
        assert (result.fluxes.production_store >= 0).all()
        # Store should not exceed capacity (approximately)
        assert (result.fluxes.production_store <= typical_params.x1 * 1.1).all()

    def test_routing_store_bounds(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """Routing store should stay non-negative."""
        result = run(typical_params, monthly_forcing)

        # Store should be non-negative (can exceed X1 since X2 can amplify)
        assert (result.fluxes.routing_store >= 0).all()

    def test_streamflow_non_negative(
        self,
        typical_params: Parameters,
        monthly_forcing: ForcingData,
    ) -> None:
        """Streamflow should always be non-negative."""
        result = run(typical_params, monthly_forcing)
        assert (result.fluxes.streamflow >= 0).all()


class TestMassBalance:
    """Tests for water mass balance."""

    def test_approximate_mass_balance(
        self,
        typical_params: Parameters,
    ) -> None:
        """P ≈ Q + ET + dS over a long simulation."""
        # Use x2 = 1.0 (no exchange) for simpler mass balance
        params = Parameters(x1=500.0, x2=1.0)

        n_months = 24
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n_months, freq="MS").values,
            precip=rng.uniform(30, 150, n_months),
            pet=rng.uniform(20, 120, n_months),
            resolution=Resolution.monthly,
        )

        result = run(params, forcing)

        total_precip = forcing.precip.sum()
        total_streamflow = result.fluxes.streamflow.sum()
        total_et = result.fluxes.actual_et.sum()

        # Storage change (final - initial)
        initial_state = State.initialize(params)
        final_prod = result.fluxes.production_store[-1]
        final_rout = result.fluxes.routing_store[-1]
        storage_change = (
            (final_prod - initial_state.production_store)
            + (final_rout - initial_state.routing_store)
        )

        # Mass balance: P = Q + ET + dS (approximately)
        mass_balance_error = abs(total_precip - total_streamflow - total_et - storage_change)
        # Error should be small relative to total inputs (< 5%)
        assert mass_balance_error < 0.05 * total_precip, (
            f"Mass balance error too large: {mass_balance_error:.2f} mm "
            f"(P={total_precip:.2f}, Q={total_streamflow:.2f}, ET={total_et:.2f}, dS={storage_change:.2f})"
        )


class TestValidationErrors:
    """Tests for array validation at the Rust boundary."""

    def test_run_rejects_wrong_length_params(self) -> None:
        """gr2m_run raises ValueError for wrong-length params array."""
        from pydrology._core.gr2m import gr2m_run

        wrong_params = np.array([500.0])  # needs 2 elements
        precip = np.array([80.0, 70.0])
        pet = np.array([20.0, 25.0])

        with pytest.raises(ValueError, match="must have"):
            gr2m_run(wrong_params, precip, pet)

    def test_run_rejects_wrong_length_state(self) -> None:
        """gr2m_run raises ValueError for wrong-length initial_state."""
        from pydrology._core.gr2m import gr2m_run

        params = np.array([500.0, 1.0])
        precip = np.array([80.0, 70.0])
        pet = np.array([20.0, 25.0])
        wrong_state = np.array([150.0, 150.0, 0.0])  # needs 2 elements

        with pytest.raises(ValueError, match="must have"):
            gr2m_run(params, precip, pet, initial_state=wrong_state)

    def test_step_rejects_wrong_length_params(self) -> None:
        """gr2m_step raises ValueError for wrong-length params array."""
        from pydrology._core.gr2m import gr2m_step

        state = np.array([150.0, 150.0])
        wrong_params = np.array([500.0])  # needs 2 elements

        with pytest.raises(ValueError, match="must have"):
            gr2m_step(state, wrong_params, 80.0, 20.0)

    def test_step_rejects_wrong_length_state(self) -> None:
        """gr2m_step raises ValueError for wrong-length state array."""
        from pydrology._core.gr2m import gr2m_step

        wrong_state = np.array([150.0])  # needs 2 elements
        params = np.array([500.0, 1.0])

        with pytest.raises(ValueError, match="must have"):
            gr2m_step(wrong_state, params, 80.0, 20.0)
