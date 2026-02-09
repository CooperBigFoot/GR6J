"""Tests for CemaNeige Rust backend equivalence.

Tests verify that the Rust-backed CemaNeige functions produce correct results.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import CemaNeige, Catchment, ForcingData
from pydrology.cemaneige.run import cemaneige_step
from pydrology.cemaneige.types import CemaNeigeSingleLayerState
from pydrology.models.gr6j_cemaneige import Parameters, run


class TestCemaNeigeStep:
    """Tests for cemaneige_step() correctness."""

    @pytest.fixture
    def params(self) -> CemaNeige:
        return CemaNeige(ctg=0.97, kf=2.5)

    @pytest.fixture
    def state(self) -> CemaNeigeSingleLayerState:
        return CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

    def test_cold_day_accumulates_snow(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """Cold day with precipitation accumulates snow."""
        precip, temp = 10.0, -2.0

        new_state, fluxes = cemaneige_step(state, params, precip, temp)

        assert new_state.g > state.g
        assert fluxes["snow_psol"] > 0
        assert np.isfinite(fluxes["snow_pack"])

    def test_all_flux_fields_present(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """All 11 flux fields are present and finite."""
        precip, temp = 15.0, 2.0

        _, fluxes = cemaneige_step(state, params, precip, temp)

        expected_keys = [
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

        for key in expected_keys:
            assert key in fluxes, f"Missing flux key: {key}"
            assert np.isfinite(fluxes[key]), f"Non-finite value for {key}"

    def test_melt_conditions(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """Melt occurs correctly when conditions are met."""
        # Accumulate snow
        current_state = state
        for _ in range(10):
            current_state, _ = cemaneige_step(current_state, params, 20.0, -5.0)

        snow_before = current_state.g
        assert snow_before > 0, "Should have accumulated snow"

        # Warm days to trigger melt
        for _ in range(20):
            current_state, _ = cemaneige_step(current_state, params, 0.0, 5.0)

        assert current_state.g < snow_before, "Snow should have melted"

    def test_multiple_steps_state_evolution(self, params: CemaNeige, state: CemaNeigeSingleLayerState) -> None:
        """State evolves correctly over multiple steps."""
        current_state = state

        test_inputs = [
            (10.0, -5.0),
            (5.0, -2.0),
            (0.0, 3.0),
            (15.0, -8.0),
            (0.0, 5.0),
        ]

        for precip, temp in test_inputs:
            current_state, _ = cemaneige_step(current_state, params, precip, temp)

        assert np.isfinite(current_state.g)
        assert np.isfinite(current_state.etg)
        assert np.isfinite(current_state.gthreshold)
        assert np.isfinite(current_state.glocalmax)


class TestCoupledSnowRunEquivalence:
    """Tests that coupled snow-GR6J runs produce valid results."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(
            x1=350,
            x2=0.5,
            x3=90,
            x4=1.7,
            x5=0.1,
            x6=5,
            ctg=0.97,
            kf=2.5,
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

        assert len(result.streamflow) == len(forcing)
        assert np.all(result.streamflow >= 0)
        assert np.all(np.isfinite(result.streamflow))

    def test_snow_pliq_and_melt_sum(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """snow_pliq_and_melt equals snow_pliq + snow_melt."""
        result = run(params, forcing, catchment)

        expected = result.snow.snow_pliq + result.snow.snow_melt
        np.testing.assert_array_almost_equal(result.snow.snow_pliq_and_melt, expected)

    def test_precip_raw_matches_input(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """precip_raw matches original input precipitation."""
        result = run(params, forcing, catchment)

        np.testing.assert_array_almost_equal(result.snow.precip_raw, forcing.precip)


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
            ctg=0.97,
            kf=2.5,
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
        assert result.snow_layers.snow_pack.shape == (len(forcing), catchment.n_layers)

    def test_layer_temperature_gradient(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """Higher elevation layers have lower temperatures."""
        result = run(params, forcing, catchment)

        layer_temps = result.snow_layers.layer_temp[0, :]
        for i in range(len(layer_temps) - 1):
            assert layer_temps[i] > layer_temps[i + 1]

    def test_all_layer_values_finite(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """All layer outputs are finite."""
        result = run(params, forcing, catchment)

        assert np.all(np.isfinite(result.snow_layers.snow_pack))
        assert np.all(np.isfinite(result.snow_layers.layer_temp))
        assert np.all(np.isfinite(result.snow_layers.snow_melt))

    def test_aggregated_matches_weighted_layers(
        self, params: Parameters, catchment: Catchment, forcing: ForcingData
    ) -> None:
        """Aggregated snow pack matches weighted average of layers."""
        result = run(params, forcing, catchment)

        weighted_avg = np.sum(
            result.snow_layers.snow_pack * result.snow_layers.layer_fractions, axis=1
        )
        np.testing.assert_array_almost_equal(result.snow.snow_pack, weighted_avg)


class TestCemaNeigeEdgeCases:
    """Tests for edge cases in CemaNeige implementation."""

    def test_zero_precipitation(self) -> None:
        """Handles zero precipitation correctly."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state = CemaNeigeSingleLayerState(g=50.0, etg=0.0, gthreshold=135.0, glocalmax=135.0)

        new_state, fluxes = cemaneige_step(state, params, 0.0, 5.0)

        assert np.isfinite(new_state.g)
        assert np.isfinite(new_state.etg)
        assert fluxes["snow_pliq"] == pytest.approx(0.0)
        assert fluxes["snow_psol"] == pytest.approx(0.0)

    def test_extreme_cold(self) -> None:
        """Handles extreme cold temperatures correctly."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state = CemaNeigeSingleLayerState(g=0.0, etg=0.0, gthreshold=135.0, glocalmax=135.0)

        new_state, fluxes = cemaneige_step(state, params, 20.0, -30.0)

        assert np.isfinite(new_state.g)
        assert fluxes["snow_psol"] == pytest.approx(20.0)
        assert fluxes["snow_pliq"] == pytest.approx(0.0)

    def test_extreme_warm(self) -> None:
        """Handles extreme warm temperatures correctly."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state = CemaNeigeSingleLayerState(g=100.0, etg=0.0, gthreshold=135.0, glocalmax=135.0)

        new_state, fluxes = cemaneige_step(state, params, 10.0, 25.0)

        assert np.isfinite(new_state.g)
        assert fluxes["snow_pliq"] == pytest.approx(10.0)
        assert fluxes["snow_psol"] == pytest.approx(0.0)
        assert fluxes["snow_melt"] > 0

    def test_no_snow_no_melt(self) -> None:
        """No melt occurs when there is no snow."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state = CemaNeigeSingleLayerState(g=0.0, etg=0.0, gthreshold=135.0, glocalmax=135.0)

        _, fluxes = cemaneige_step(state, params, 0.0, 20.0)

        assert fluxes["snow_melt"] == pytest.approx(0.0)

    def test_cold_snow_no_melt(self) -> None:
        """No melt when thermal state is below zero."""
        params = CemaNeige(ctg=0.97, kf=2.5)
        state = CemaNeigeSingleLayerState(g=100.0, etg=-5.0, gthreshold=135.0, glocalmax=135.0)

        _, fluxes = cemaneige_step(state, params, 0.0, 5.0)

        assert fluxes["snow_melt"] == pytest.approx(0.0)


class TestNumericalStabilitySnow:
    """Tests for numerical stability of CemaNeige implementations."""

    @pytest.fixture
    def params(self) -> Parameters:
        return Parameters(
            x1=350,
            x2=0.5,
            x3=90,
            x4=1.7,
            x5=0.1,
            x6=5,
            ctg=0.97,
            kf=2.5,
        )

    @pytest.fixture
    def catchment(self) -> Catchment:
        return Catchment(mean_annual_solid_precip=150.0)

    def test_long_simulation_with_snow(self, params: Parameters, catchment: Catchment) -> None:
        """Long simulation with snow maintains numerical stability."""
        n = 365
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 20, n),
            pet=rng.uniform(2, 6, n),
            temp=rng.uniform(-10, 20, n),
        )

        result = run(params, forcing, catchment)

        assert np.all(np.isfinite(result.streamflow))
        assert np.all(np.isfinite(result.snow.snow_pack))
        assert np.all(np.isfinite(result.snow.snow_melt))
        assert np.all(result.streamflow >= 0)

    def test_seasonal_cycle(self, params: Parameters, catchment: Catchment) -> None:
        """Model handles seasonal cycle (winter accumulation, spring melt)."""
        n = 365
        temps = np.zeros(n)
        temps[0:90] = -10.0
        temps[90:180] = np.linspace(-10, 15, 90)
        temps[180:270] = 15.0
        temps[270:365] = np.linspace(15, -10, 95)

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 5.0),
            pet=np.full(n, 3.0),
            temp=temps,
        )

        result = run(params, forcing, catchment)

        assert result.snow.snow_pack[89] > 0
        spring_melt = result.snow.snow_melt[90:180].sum()
        assert spring_melt > 0
        assert np.all(np.isfinite(result.streamflow))
        assert np.all(np.isfinite(result.snow.snow_pack))
        assert np.all(result.streamflow >= 0)
