"""Tests for CemaNeige step function.

Tests verify the complete snow model timestep execution according to
CEMANEIGE.md algorithm.
"""

import numpy as np
import pytest

from gr6j.cemaneige.run import cemaneige_step
from gr6j.cemaneige.types import CemaNeige, CemaNeigeSingleLayerState

EXPECTED_FLUX_KEYS = {
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
def typical_params() -> CemaNeige:
    """Typical CemaNeige parameters."""
    return CemaNeige(ctg=0.97, kf=2.5, mean_annual_solid_precip=150.0)


@pytest.fixture
def initialized_state(typical_params: CemaNeige) -> CemaNeigeSingleLayerState:
    """Initial model state from parameters."""
    return CemaNeigeSingleLayerState.initialize(typical_params.mean_annual_solid_precip)


class TestCemaNeigeStepe:
    """Tests for cemaneige_step function."""

    def test_returns_state_and_fluxes(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Verify step returns tuple of (state, dict)."""
        result = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=-5.0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        new_state, fluxes = result
        assert isinstance(new_state, CemaNeigeSingleLayerState)
        assert isinstance(fluxes, dict)

    def test_fluxes_contains_all_expected_keys(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Fluxes dictionary contains all 11 expected keys."""
        _, fluxes = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=-5.0,
        )

        assert set(fluxes.keys()) == EXPECTED_FLUX_KEYS

        # All values should be floats
        for key, value in fluxes.items():
            assert isinstance(value, float), f"Flux '{key}' is not a float: {type(value)}"

    def test_all_fluxes_are_finite(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """All flux values should be finite (not NaN or inf)."""
        _, fluxes = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=-5.0,
        )

        for key, value in fluxes.items():
            assert np.isfinite(value), f"Flux '{key}' is not finite: {value}"

    def test_snow_pack_non_negative(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Snow pack (g) should always be >= 0."""
        # Test various scenarios
        test_cases = [
            (-10.0, 10.0),  # Cold with snow
            (5.0, 10.0),  # Warm with rain
            (0.0, 0.0),  # No inputs
            (-5.0, 50.0),  # Heavy snow
        ]

        state = initialized_state
        for temp, precip in test_cases:
            state, fluxes = cemaneige_step(state, typical_params, precip, temp)
            assert state.g >= 0.0
            assert fluxes["snow_pack"] >= 0.0

    def test_thermal_state_at_or_below_zero(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Thermal state (etg) should always be <= 0."""
        test_cases = [
            (15.0, 0.0),  # Hot day
            (-20.0, 10.0),  # Very cold
            (5.0, 20.0),  # Warm rain
        ]

        state = initialized_state
        for temp, precip in test_cases:
            state, fluxes = cemaneige_step(state, typical_params, precip, temp)
            assert state.etg <= 0.0
            assert fluxes["snow_thermal_state"] <= 0.0

    def test_cold_day_accumulates_snow(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Cold day with precip should accumulate snow."""
        new_state, fluxes = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=-5.0,  # Cold: all snow
        )

        # All precip should be snow (solid_fraction = 1.0 at -5C)
        assert fluxes["snow_psol"] == pytest.approx(10.0)
        assert fluxes["snow_pliq"] == pytest.approx(0.0)
        # Snow pack should increase
        assert new_state.g > initialized_state.g

    def test_warm_day_no_snow_accumulation(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Warm day should not accumulate snow."""
        new_state, fluxes = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=10.0,  # Warm: all rain
        )

        # All precip should be rain
        assert fluxes["snow_psol"] == pytest.approx(0.0)
        assert fluxes["snow_pliq"] == pytest.approx(10.0)

    def test_mass_balance_psol_minus_melt_equals_delta_g(
        self,
        typical_params: CemaNeige,
    ) -> None:
        """Mass balance: Psol - Melt = change in snow pack."""
        # Start with some snow
        state = CemaNeigeSingleLayerState(
            g=50.0,
            etg=0.0,  # At melting point
            gthreshold=135.0,
            glocalmax=135.0,
        )

        new_state, fluxes = cemaneige_step(
            state=state,
            params=typical_params,
            precip=5.0,
            temp=5.0,  # Warm: melt occurs
        )

        delta_g = new_state.g - state.g
        psol = fluxes["snow_psol"]
        melt = fluxes["snow_melt"]

        # delta_g = psol - melt
        assert delta_g == pytest.approx(psol - melt)

    def test_pliq_and_melt_equals_sum(
        self,
        typical_params: CemaNeige,
    ) -> None:
        """snow_pliq_and_melt = snow_pliq + snow_melt."""
        state = CemaNeigeSingleLayerState(
            g=50.0,
            etg=0.0,
            gthreshold=135.0,
            glocalmax=135.0,
        )

        _, fluxes = cemaneige_step(
            state=state,
            params=typical_params,
            precip=10.0,
            temp=5.0,
        )

        assert fluxes["snow_pliq_and_melt"] == pytest.approx(fluxes["snow_pliq"] + fluxes["snow_melt"])

    def test_gratio_bounded_zero_one(
        self,
        typical_params: CemaNeige,
    ) -> None:
        """Gratio is always in [0, 1]."""
        test_states = [
            CemaNeigeSingleLayerState(g=0.0, etg=0.0, gthreshold=135.0, glocalmax=135.0),
            CemaNeigeSingleLayerState(g=50.0, etg=0.0, gthreshold=135.0, glocalmax=135.0),
            CemaNeigeSingleLayerState(g=200.0, etg=0.0, gthreshold=135.0, glocalmax=135.0),
        ]

        for state in test_states:
            _, fluxes = cemaneige_step(state, typical_params, 10.0, 0.0)
            assert 0.0 <= fluxes["snow_gratio"] <= 1.0

    def test_no_melt_when_snow_cold(
        self,
        typical_params: CemaNeige,
    ) -> None:
        """No melt when thermal state is below 0."""
        state = CemaNeigeSingleLayerState(
            g=100.0,
            etg=-5.0,  # Cold snow pack
            gthreshold=135.0,
            glocalmax=135.0,
        )

        _, fluxes = cemaneige_step(
            state=state,
            params=typical_params,
            precip=0.0,
            temp=5.0,  # Warm air
        )

        # No melt because etg < 0
        assert fluxes["snow_melt"] == 0.0
        assert fluxes["snow_pot_melt"] == 0.0

    def test_state_is_new_instance(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """New state is a different instance from input state."""
        new_state, _ = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=-5.0,
        )

        assert new_state is not initialized_state

    def test_temp_recorded_in_fluxes(
        self,
        initialized_state: CemaNeigeSingleLayerState,
        typical_params: CemaNeige,
    ) -> None:
        """Input temperature is recorded in fluxes."""
        _, fluxes = cemaneige_step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            temp=-7.5,
        )

        assert fluxes["snow_temp"] == -7.5
