"""Tests for CemaNeige Numba acceleration.

Tests verify that Numba-compiled functions produce identical results
to the pure Python implementations.
"""

import numpy as np
import pandas as pd
import pytest

from pydrology import CemaNeige, Catchment, ForcingData
from pydrology.cemaneige.run import _cemaneige_step_numba, cemaneige_step
from pydrology.cemaneige.types import CemaNeigeSingleLayerState
from pydrology.models.gr6j_cemaneige import Parameters, run


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

        # Check all outputs exist and are valid
        assert len(result.streamflow) == len(forcing)
        assert np.all(result.streamflow >= 0)
        assert np.all(np.isfinite(result.streamflow))

    def test_snow_pliq_and_melt_sum(self, params: Parameters, catchment: Catchment, forcing: ForcingData) -> None:
        """snow_pliq_and_melt equals snow_pliq + snow_melt."""
        result = run(params, forcing, catchment)

        # Verify the relationship: pliq_and_melt = pliq + melt
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

        # Check first timestep: layers should have decreasing temperature with elevation
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

        # Compute weighted average of layer snow packs
        weighted_avg = np.sum(
            result.snow_layers.snow_pack * result.snow_layers.layer_fractions, axis=1
        )
        np.testing.assert_array_almost_equal(result.snow.snow_pack, weighted_avg)


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

        # All outputs should be finite (no NaN/inf)
        assert np.all(np.isfinite(result.streamflow))
        assert np.all(np.isfinite(result.snow.snow_pack))
        assert np.all(np.isfinite(result.snow.snow_melt))

        # Streamflow should be non-negative
        assert np.all(result.streamflow >= 0)

    def test_seasonal_cycle(self, params: Parameters, catchment: Catchment) -> None:
        """Model handles seasonal cycle (winter accumulation, spring melt)."""
        n = 365
        # Create seasonal temperature pattern: cold winter (days 0-90), warm rest
        temps = np.zeros(n)
        temps[0:90] = -10.0  # Cold winter period
        temps[90:180] = np.linspace(-10, 15, 90)  # Spring warming
        temps[180:270] = 15.0  # Summer
        temps[270:365] = np.linspace(15, -10, 95)  # Fall cooling

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 5.0),  # Constant precip
            pet=np.full(n, 3.0),
            temp=temps,
        )

        result = run(params, forcing, catchment)

        # Snow should accumulate during cold period
        assert result.snow.snow_pack[89] > 0  # End of winter has snow

        # Melt should occur during spring/summer
        spring_melt = result.snow.snow_melt[90:180].sum()
        assert spring_melt > 0

        # No numerical instabilities
        assert np.all(np.isfinite(result.streamflow))
        assert np.all(np.isfinite(result.snow.snow_pack))
        assert np.all(result.streamflow >= 0)
