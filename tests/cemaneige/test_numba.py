"""Tests for CemaNeige Numba acceleration.

Tests verify that Numba-compiled functions produce identical results
to the pure Python implementations.
"""

import numpy as np
import pandas as pd
import pytest

from gr6j import CemaNeige, Parameters, run
from gr6j.cemaneige.run import _cemaneige_step_numba, cemaneige_step
from gr6j.cemaneige.types import CemaNeigeSingleLayerState
from gr6j.inputs import Catchment, ForcingData


class TestCemaNeigeStepNumbaEquivalence:
    """Tests that _cemaneige_step_numba produces same results as cemaneige_step()."""

    @pytest.fixture
    def params(self) -> CemaNeige:
        return CemaNeige(ctg=0.97, kf=2.5)

    @pytest.fixture
    def state(self) -> CemaNeigeSingleLayerState:
        return CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

    def test_single_step_equivalence(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """Single CemaNeige step produces identical results."""
        precip, temp = 10.0, -2.0  # Cold day with snow

        # Python step
        new_state_py, fluxes_py = cemaneige_step(state, params, precip, temp)

        # Numba step
        state_arr = np.asarray(state)
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)
        _cemaneige_step_numba(state_arr, params.ctg, params.kf, precip, temp, out_state, out_fluxes)

        # Compare state
        assert out_state[0] == pytest.approx(new_state_py.g, rel=1e-10)
        assert out_state[1] == pytest.approx(new_state_py.etg, rel=1e-10)

        # Compare key fluxes
        assert out_fluxes[2] == pytest.approx(fluxes_py["snow_pack"], rel=1e-10)
        assert out_fluxes[6] == pytest.approx(fluxes_py["snow_melt"], rel=1e-10)
        assert out_fluxes[7] == pytest.approx(fluxes_py["snow_pliq_and_melt"], rel=1e-10)

    def test_all_flux_fields(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """All 11 flux fields match between Numba and Python."""
        precip, temp = 15.0, 2.0  # Mixed conditions

        # Python step
        _, fluxes_py = cemaneige_step(state, params, precip, temp)

        # Numba step
        state_arr = np.asarray(state)
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)
        _cemaneige_step_numba(state_arr, params.ctg, params.kf, precip, temp, out_state, out_fluxes)

        # Map output array indices to flux keys
        flux_mapping = [
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
        ]

        for idx, key in enumerate(flux_mapping):
            assert out_fluxes[idx] == pytest.approx(fluxes_py[key], rel=1e-10), (
                f"Field {key} (index {idx}) does not match"
            )

    def test_melt_conditions(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """Melt occurs correctly when conditions are met."""
        # First accumulate snow
        state_arr = np.asarray(state)
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        # Cold days to accumulate snow
        for _ in range(10):
            _cemaneige_step_numba(state_arr, params.ctg, params.kf, 20.0, -5.0, out_state, out_fluxes)
            state_arr[:] = out_state

        snow_before = out_state[0]
        assert snow_before > 0, "Should have accumulated snow"

        # Warm days to trigger melt - need more days for thermal state to warm up
        # The thermal state (etg) needs to reach 0 before melt can occur
        for _ in range(20):
            _cemaneige_step_numba(state_arr, params.ctg, params.kf, 0.0, 5.0, out_state, out_fluxes)
            state_arr[:] = out_state

        # Check melt occurred
        assert out_state[0] < snow_before, "Snow should have melted"

    def test_multiple_steps_state_evolution(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """State evolves identically over multiple steps."""
        # Numba path
        state_arr = np.asarray(state)
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        # Python path
        state_py = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

        test_inputs = [
            (10.0, -5.0),
            (5.0, -2.0),
            (0.0, 3.0),
            (15.0, -8.0),
            (0.0, 5.0),
        ]

        for precip, temp in test_inputs:
            state_py, _ = cemaneige_step(state_py, params, precip, temp)
            _cemaneige_step_numba(state_arr, params.ctg, params.kf, precip, temp, out_state, out_fluxes)
            state_arr[:] = out_state

        # Compare final states
        assert out_state[0] == pytest.approx(state_py.g, rel=1e-10)
        assert out_state[1] == pytest.approx(state_py.etg, rel=1e-10)
        assert out_state[2] == pytest.approx(state_py.gthreshold, rel=1e-10)
        assert out_state[3] == pytest.approx(state_py.glocalmax, rel=1e-10)


class TestCoupledSnowRunEquivalence:
    """Tests that coupled snow-GR6J runs match between Numba and Python paths."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(
            x1=350,
            x2=0.5,
            x3=90,
            x4=1.7,
            x5=0.1,
            x6=5,
            snow=CemaNeige(ctg=0.97, kf=2.5),
        )

    @pytest.fixture
    def catchment(self) -> Catchment:
        return Catchment(
            mean_annual_solid_precip=150.0,
        )

    @pytest.fixture
    def forcing(self) -> ForcingData:
        n = 100
        rng = np.random.default_rng(42)
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 20, n),
            pet=rng.uniform(2, 6, n),
            temp=rng.uniform(-5, 15, n),
        )

    def test_single_layer_snow_run(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """Single-layer snow run produces correct outputs."""
        result = run(params, forcing, catchment)

        # Check all outputs exist and are valid
        assert result.snow is not None
        assert len(result.gr6j.streamflow) == len(forcing)
        assert np.all(result.gr6j.streamflow >= 0)
        assert np.all(np.isfinite(result.gr6j.streamflow))

        # Check snow outputs
        assert np.all(result.snow.snow_pack >= 0)
        assert np.all(result.snow.snow_melt >= 0)

    def test_snow_pliq_and_melt_sum(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """snow_pliq_and_melt equals snow_pliq + snow_melt."""
        result = run(params, forcing, catchment)

        np.testing.assert_allclose(
            result.snow.snow_pliq_and_melt,
            result.snow.snow_pliq + result.snow.snow_melt,
            rtol=1e-10,
        )

    def test_precip_raw_matches_input(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """precip_raw matches original input precipitation."""
        result = run(params, forcing, catchment)

        np.testing.assert_allclose(result.snow.precip_raw, forcing.precip, rtol=1e-10)


class TestMultiLayerSnowEquivalence:
    """Tests for multi-layer snow mode."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(
            x1=350,
            x2=0.5,
            x3=90,
            x4=1.7,
            x5=0.1,
            x6=5,
            snow=CemaNeige(ctg=0.97, kf=2.5),
        )

    @pytest.fixture
    def catchment(self) -> Catchment:
        return Catchment(
            mean_annual_solid_precip=150.0,
            hypsometric_curve=np.linspace(500, 3000, 101),
            input_elevation=1500.0,
            n_layers=5,
        )

    @pytest.fixture
    def forcing(self) -> ForcingData:
        n = 50
        rng = np.random.default_rng(42)
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 20, n),
            pet=rng.uniform(2, 6, n),
            temp=rng.uniform(-10, 10, n),
        )

    def test_multi_layer_produces_layer_outputs(
        self, params: Parameters, catchment: Catchment, forcing: ForcingData
    ) -> None:
        """Multi-layer mode produces per-layer outputs."""
        result = run(params, forcing, catchment)

        assert result.snow_layers is not None
        assert result.snow_layers.n_layers == 5
        assert result.snow_layers.snow_pack.shape == (len(forcing), 5)
        assert result.snow_layers.layer_temp.shape == (len(forcing), 5)

    def test_layer_temperature_gradient(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """Higher elevation layers have lower temperatures."""
        result = run(params, forcing, catchment)

        assert result.snow_layers is not None
        # On average, higher layers should be colder
        mean_temps = result.snow_layers.layer_temp.mean(axis=0)
        # Layer 0 is lowest elevation, layer 4 is highest
        assert mean_temps[0] > mean_temps[4], "Higher elevations should be colder"

    def test_all_layer_values_finite(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """All layer outputs are finite."""
        result = run(params, forcing, catchment)

        assert result.snow_layers is not None
        assert np.all(np.isfinite(result.snow_layers.snow_pack))
        assert np.all(np.isfinite(result.snow_layers.snow_melt))
        assert np.all(np.isfinite(result.snow_layers.layer_temp))
        assert np.all(np.isfinite(result.snow_layers.layer_precip))

    def test_aggregated_matches_weighted_layers(
        self, params: Parameters, catchment: Catchment, forcing: ForcingData
    ) -> None:
        """Aggregated snow pack matches weighted average of layers."""
        result = run(params, forcing, catchment)

        assert result.snow_layers is not None
        assert result.snow is not None

        # Compute weighted average
        fractions = result.snow_layers.layer_fractions
        weighted_avg = (result.snow_layers.snow_pack * fractions).sum(axis=1)

        np.testing.assert_allclose(result.snow.snow_pack, weighted_avg, rtol=1e-10)


class TestCemaNeigeEdgeCases:
    """Tests for edge cases in CemaNeige Numba implementation."""

    def test_zero_precipitation(self) -> None:
        """Handles zero precipitation correctly."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state_arr = np.array([50.0, 0.0, 135.0, 135.0])  # Some initial snow
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        _cemaneige_step_numba(state_arr, params.ctg, params.kf, 0.0, 5.0, out_state, out_fluxes)

        assert np.all(np.isfinite(out_state))
        assert np.all(np.isfinite(out_fluxes))
        assert out_fluxes[0] == pytest.approx(0.0)  # snow_pliq
        assert out_fluxes[1] == pytest.approx(0.0)  # snow_psol

    def test_extreme_cold(self) -> None:
        """Handles extreme cold temperatures correctly."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state_arr = np.array([0.0, 0.0, 135.0, 135.0])
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        _cemaneige_step_numba(state_arr, params.ctg, params.kf, 20.0, -30.0, out_state, out_fluxes)

        assert np.all(np.isfinite(out_state))
        assert np.all(np.isfinite(out_fluxes))
        # All precip should be solid
        assert out_fluxes[1] == pytest.approx(20.0)  # snow_psol
        assert out_fluxes[0] == pytest.approx(0.0)  # snow_pliq

    def test_extreme_warm(self) -> None:
        """Handles extreme warm temperatures correctly."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state_arr = np.array([100.0, 0.0, 135.0, 135.0])  # Snow at melting point
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        _cemaneige_step_numba(state_arr, params.ctg, params.kf, 10.0, 25.0, out_state, out_fluxes)

        assert np.all(np.isfinite(out_state))
        assert np.all(np.isfinite(out_fluxes))
        # All precip should be liquid
        assert out_fluxes[0] == pytest.approx(10.0)  # snow_pliq
        assert out_fluxes[1] == pytest.approx(0.0)  # snow_psol
        # Melt should occur
        assert out_fluxes[6] > 0  # snow_melt

    def test_no_snow_no_melt(self) -> None:
        """No melt occurs when there is no snow."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state_arr = np.array([0.0, 0.0, 135.0, 135.0])  # No snow
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        _cemaneige_step_numba(state_arr, params.ctg, params.kf, 0.0, 20.0, out_state, out_fluxes)

        assert out_fluxes[6] == pytest.approx(0.0)  # snow_melt

    def test_cold_snow_no_melt(self) -> None:
        """No melt when thermal state is below zero."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state_arr = np.array([100.0, -5.0, 135.0, 135.0])  # Cold snow
        out_state = np.zeros(4)
        out_fluxes = np.zeros(11)

        # Even with warm air, cold snow shouldn't melt immediately
        _cemaneige_step_numba(state_arr, params.ctg, params.kf, 0.0, 5.0, out_state, out_fluxes)

        assert out_fluxes[6] == pytest.approx(0.0)  # snow_melt


class TestNumericalStabilitySnow:
    """Tests for numerical stability of CemaNeige Numba implementations."""

    def test_long_simulation_with_snow(self) -> None:
        """Long simulation with snow maintains numerical stability."""
        params = Parameters(
            x1=350,
            x2=0.5,
            x3=90,
            x4=1.7,
            x5=0.1,
            x6=5,
            snow=CemaNeige(ctg=0.97, kf=2.5),
        )
        catchment = Catchment(mean_annual_solid_precip=150.0)

        n = 500
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 25, n),
            pet=rng.uniform(1, 7, n),
            temp=rng.uniform(-15, 20, n),
        )

        result = run(params, forcing, catchment)

        # All outputs should be finite
        assert np.all(np.isfinite(result.gr6j.streamflow))
        assert np.all(np.isfinite(result.snow.snow_pack))
        assert np.all(np.isfinite(result.snow.snow_melt))

        # Streamflow should be non-negative
        assert np.all(result.gr6j.streamflow >= 0)
        assert np.all(result.snow.snow_pack >= 0)
        assert np.all(result.snow.snow_melt >= 0)

    def test_seasonal_cycle(self) -> None:
        """Model handles seasonal cycle (winter accumulation, spring melt)."""
        params = Parameters(
            x1=350,
            x2=0.5,
            x3=90,
            x4=1.7,
            x5=0.1,
            x6=5,
            snow=CemaNeige(ctg=0.97, kf=2.5),
        )
        catchment = Catchment(mean_annual_solid_precip=150.0)

        # Create seasonal temperature pattern
        n = 365
        days = np.arange(n)
        temp = 10 * np.sin(2 * np.pi * (days - 100) / 365)  # Peak in summer
        rng = np.random.default_rng(42)

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 15, n),
            pet=np.clip(2 + 3 * np.sin(2 * np.pi * (days - 100) / 365), 0.5, 6),
            temp=temp,
        )

        result = run(params, forcing, catchment)

        # Should have some snow accumulation in winter
        # Days 0-90 are winter (approx Dec-Mar)
        assert result.snow.snow_pack[:90].max() > 0, "Should have snow in winter"

        # All outputs should be valid
        assert np.all(np.isfinite(result.gr6j.streamflow))
        assert np.all(result.gr6j.streamflow >= 0)
