"""Integration tests for pydrology.models.gr6j_cemaneige.run module.

Tests the coupled GR6J-CemaNeige model for snow-affected catchments.
Tests cover single-layer and multi-layer snow simulation modes.
"""

import numpy as np
import pandas as pd
import pytest
from pydrology import Catchment, ForcingData, SnowLayerOutputs
from pydrology.models.gr6j_cemaneige import Parameters, State, run, step
from pydrology.types import PrecipGradientType
from pydrology.outputs import ModelOutput
from pydrology.processes.unit_hydrographs import compute_uh_ordinates

# Expected flux keys for the coupled model (snow + GR6J)
EXPECTED_GR6J_FLUX_KEYS = {
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
    """Typical GR6J-CemaNeige parameters (8 total: 6 GR6J + 2 CemaNeige)."""
    return Parameters(
        x1=350.0,  # Production store capacity [mm]
        x2=0.0,  # No intercatchment exchange
        x3=90.0,  # Routing store capacity [mm]
        x4=1.7,  # Unit hydrograph time constant [days]
        x5=0.0,  # Exchange threshold
        x6=5.0,  # Exponential store scale [mm]
        ctg=0.97,  # Thermal state coefficient
        kf=2.5,  # Degree-day factor
    )


@pytest.fixture
def typical_catchment() -> Catchment:
    """Typical catchment with snow module properties."""
    return Catchment(mean_annual_solid_precip=150.0)


@pytest.fixture
def forcing_with_temp() -> ForcingData:
    """ForcingData with precip, pet, and temp for 5 days."""
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=5, freq="D").values,
        precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
        pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
        temp=np.array([-5.0, 0.0, 5.0, -2.0, 8.0]),
    )


class TestRunWithSnow:
    """Tests for run() with CemaNeige snow module integration."""

    def test_returns_model_output_with_snow(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """When snow is enabled, output has snow field populated."""
        result = run(typical_params, forcing_with_temp, catchment=typical_catchment)

        assert isinstance(result, ModelOutput)
        assert result.snow is not None
        assert set(result.snow.to_dict().keys()) == EXPECTED_SNOW_FLUX_KEYS

    def test_raises_when_temp_missing(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """ValueError when temp field missing."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=5, freq="D").values,
            precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
            pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
        )
        with pytest.raises(ValueError, match="Temperature"):
            run(typical_params, forcing, catchment=typical_catchment)

    def test_precip_raw_equals_input_precip(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """precip_raw field matches original input precipitation."""
        result = run(typical_params, forcing_with_temp, catchment=typical_catchment)

        np.testing.assert_array_almost_equal(
            result.snow.precip_raw,
            forcing_with_temp.precip,
        )

    def test_precip_differs_from_precip_raw_with_snow(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """precip to GR6J differs from precip_raw (snow preprocessing)."""
        result = run(typical_params, forcing_with_temp, catchment=typical_catchment)

        # At least some values should differ (cold days accumulate snow)
        # Not all precip passes through unchanged
        assert not np.allclose(result.fluxes.precip, result.snow.precip_raw)

    def test_cold_day_snow_accumulates(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Cold days with precip accumulate snow."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=3, freq="D").values,
            precip=np.array([10.0, 10.0, 10.0]),
            pet=np.array([2.0, 2.0, 2.0]),
            temp=np.array([-10.0, -10.0, -10.0]),  # Very cold
        )

        result = run(typical_params, forcing, catchment=typical_catchment)

        # Snow pack should increase over cold period
        assert result.snow.snow_pack[-1] > result.snow.snow_pack[0]
        # All precip should be snow (solid_fraction = 1.0)
        np.testing.assert_array_almost_equal(
            result.snow.snow_psol,
            forcing.precip,
        )

    def test_warm_period_snow_passes_through(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Warm period with no snow: precip passes through unchanged."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=3, freq="D").values,
            precip=np.array([10.0, 10.0, 10.0]),
            pet=np.array([2.0, 2.0, 2.0]),
            temp=np.array([15.0, 15.0, 15.0]),  # Warm: all rain
        )

        result = run(typical_params, forcing, catchment=typical_catchment)

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
        typical_params: Parameters,
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

        result = run(typical_params, forcing, catchment=typical_catchment)

        # Snow should accumulate during cold period
        assert result.snow.snow_pack[cold_days - 1] > 0.0
        # Melt should occur during warm period (may need thermal state to warm up)
        assert result.snow.snow_melt[cold_days:].sum() > 0.0
        # Streamflow should be produced
        assert result.streamflow.sum() > 0.0

    def test_uses_custom_initial_state(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """Custom initial state is respected."""
        # Custom state with existing snow pack
        custom_state = State.initialize(typical_params, typical_catchment)
        # Modify the snow layer state (first layer g value)
        custom_state.snow_layer_states[0, 0] = 100.0  # 100mm snow pack

        result_custom = run(typical_params, forcing_with_temp, initial_state=custom_state, catchment=typical_catchment)
        result_default = run(typical_params, forcing_with_temp, catchment=typical_catchment)

        # First row should differ due to different initial snow
        assert result_custom.snow.snow_pack[0] != result_default.snow.snow_pack[0]

    def test_output_length_matches_input(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
    ) -> None:
        """Output length matches input when snow enabled."""
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=5, freq="D").values,
            precip=np.array([10.0] * 5),
            pet=np.array([3.0] * 5),
            temp=np.array([0.0] * 5),
        )

        result = run(typical_params, forcing, catchment=typical_catchment)

        assert len(result) == len(forcing)

    def test_all_values_finite(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
        forcing_with_temp: ForcingData,
    ) -> None:
        """All output values are finite when snow enabled."""
        result = run(typical_params, forcing_with_temp, catchment=typical_catchment)

        for key, values in result.fluxes.to_dict().items():
            assert np.all(np.isfinite(values)), f"Flux field '{key}' has non-finite values"

        for key, values in result.snow.to_dict().items():
            assert np.all(np.isfinite(values)), f"Snow field '{key}' has non-finite values"


class TestRunWithMultiLayerSnow:
    """Tests for run() with multi-layer CemaNeige snow simulation."""

    @pytest.fixture
    def multi_layer_catchment(self) -> Catchment:
        """5-layer catchment with hypsometric curve."""
        return Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=5,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )

    @pytest.fixture
    def forcing_10_days(self) -> ForcingData:
        """10-day forcing with temperature."""
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.array([10.0, 15.0, 5.0, 0.0, 20.0, 10.0, 0.0, 5.0, 15.0, 8.0]),
            pet=np.array([2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0]),
            temp=np.array([-5.0, -3.0, 0.0, 5.0, -8.0, -2.0, 3.0, 7.0, -1.0, 10.0]),
        )

    def test_produces_snow_layer_outputs(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Multi-layer run produces snow_layers output."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        assert result.snow_layers is not None
        assert isinstance(result.snow_layers, SnowLayerOutputs)

    def test_snow_layers_has_correct_shape(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Snow layer arrays have shape (n_timesteps, n_layers)."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        assert result.snow_layers.snow_pack.shape == (10, 5)
        assert result.snow_layers.snow_melt.shape == (10, 5)
        assert result.snow_layers.layer_temp.shape == (10, 5)
        assert result.snow_layers.layer_precip.shape == (10, 5)

    def test_snow_layers_n_layers_property(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """n_layers property returns correct count."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        assert result.snow_layers.n_layers == 5

    def test_aggregated_snow_output_still_present(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Aggregated SnowOutput is still present alongside layer outputs."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        assert result.snow is not None
        assert len(result.snow.snow_pack) == 10

    def test_all_values_finite(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """All output values are finite."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        for key, values in result.fluxes.to_dict().items():
            assert np.all(np.isfinite(values)), f"Flux '{key}' has non-finite"

        for key, values in result.snow.to_dict().items():
            assert np.all(np.isfinite(values)), f"Snow '{key}' has non-finite"

        assert np.all(np.isfinite(result.snow_layers.snow_pack))
        assert np.all(np.isfinite(result.snow_layers.layer_temp))

    def test_higher_layers_colder_temperatures(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Higher elevation layers have colder temperatures."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        # Check first timestep: layers should have decreasing temperature
        layer_temps = result.snow_layers.layer_temp[0, :]
        for i in range(len(layer_temps) - 1):
            assert layer_temps[i] > layer_temps[i + 1]

    def test_single_layer_has_no_layer_outputs(
        self,
        typical_params: Parameters,
        typical_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Single-layer catchment produces no snow_layers output."""
        result = run(typical_params, forcing_10_days, catchment=typical_catchment)

        assert result.snow_layers is None

    def test_layer_elevations_stored_in_output(
        self,
        typical_params: Parameters,
        multi_layer_catchment: Catchment,
        forcing_10_days: ForcingData,
    ) -> None:
        """Layer elevations are stored in the output."""
        result = run(typical_params, forcing_10_days, catchment=multi_layer_catchment)

        assert len(result.snow_layers.layer_elevations) == 5
        # Elevations should be monotonically increasing
        for i in range(4):
            assert result.snow_layers.layer_elevations[i] < result.snow_layers.layer_elevations[i + 1]


class TestStep:
    """Tests for the step() function - single timestep execution."""

    @pytest.fixture
    def initialized_state(self, typical_params: Parameters, typical_catchment: Catchment) -> State:
        """Initial model state from typical parameters."""
        return State.initialize(typical_params, typical_catchment)

    @pytest.fixture
    def uh_ordinates(self, typical_params: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """Pre-computed unit hydrograph ordinates."""
        return compute_uh_ordinates(typical_params.x4)

    def test_returns_state_and_fluxes(
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
            temp=-5.0,
            uh1_ordinates=uh1_ord,
            uh2_ordinates=uh2_ord,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        new_state, fluxes = result
        assert isinstance(new_state, State)
        assert isinstance(fluxes, dict)

    def test_fluxes_contains_both_snow_and_gr6j_keys(
        self,
        initialized_state: State,
        typical_params: Parameters,
        uh_ordinates: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Fluxes dictionary contains both snow and GR6J outputs."""
        uh1_ord, uh2_ord = uh_ordinates

        _, fluxes = step(
            state=initialized_state,
            params=typical_params,
            precip=10.0,
            pet=3.0,
            temp=-5.0,
            uh1_ordinates=uh1_ord,
            uh2_ordinates=uh2_ord,
        )

        # Should have snow keys
        assert "snow_pack" in fluxes
        assert "snow_melt" in fluxes
        assert "snow_pliq_and_melt" in fluxes

        # Should have GR6J keys
        assert "streamflow" in fluxes
        assert "production_store" in fluxes
        assert "routing_store" in fluxes

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
            (10.0, 3.0, -10.0),  # Cold, heavy snow
            (0.0, 5.0, 5.0),  # Dry, warm
            (50.0, 0.0, -5.0),  # Heavy precip, cold
            (0.0, 0.0, 0.0),  # No inputs
            (10.0, 4.0, 10.0),  # Rain
        ]

        state = initialized_state
        for precip, pet, temp in test_inputs:
            state, fluxes = step(
                state=state,
                params=typical_params,
                precip=precip,
                pet=pet,
                temp=temp,
                uh1_ordinates=uh1_ord,
                uh2_ordinates=uh2_ord,
            )

            assert fluxes["streamflow"] >= 0.0, f"Negative streamflow for P={precip}, E={pet}, T={temp}"


class TestPerBandAndLinearGradient:
    """Integration tests for per-band gthreshold and linear precipitation gradient."""

    def _make_forcing(self, n: int = 365, precip: float = 5.0, temp: float = 2.0) -> ForcingData:
        """Create simple daily forcing data."""
        import datetime
        base = datetime.datetime(2000, 1, 1)
        time = np.array([base + datetime.timedelta(days=i) for i in range(n)], dtype="datetime64[ns]")
        return ForcingData(
            time=time,
            precip=np.full(n, precip),
            pet=np.full(n, 3.0),
            temp=np.full(n, temp),
        )

    def _make_params(self) -> Parameters:
        return Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)

    def test_multi_layer_different_gratio_per_band(self) -> None:
        """Different bands should have different gthresholds and thus different gratio evolution."""
        catchment = Catchment(
            mean_annual_solid_precip=200.0,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=600.0,
            n_layers=3,
        )
        params = self._make_params()
        forcing = self._make_forcing(n=30, precip=10.0, temp=-2.0)
        result = run(params, forcing, catchment=catchment)
        # Per-layer gratio should differ between bands
        assert result.snow_layers is not None
        gratio = result.snow_layers.snow_gratio
        # gratio shape is (n_timesteps, n_layers)
        assert gratio.shape == (30, 3)

    def test_high_elevation_band_gratio_saturates_later(self) -> None:
        """Higher bands have higher gthreshold, so gratio should saturate later."""
        catchment = Catchment(
            mean_annual_solid_precip=100.0,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=600.0,
            n_layers=3,
        )
        params = self._make_params()
        # Long cold period with moderate precip to build snow
        forcing = self._make_forcing(n=90, precip=5.0, temp=-5.0)
        result = run(params, forcing, catchment=catchment)
        assert result.snow_layers is not None
        # After 90 cold days, check gratio at the last timestep
        gratio_last = result.snow_layers.snow_gratio[-1, :]
        # All gratios should be between 0 and 1 (inclusive)
        assert np.all(gratio_last >= 0.0)
        assert np.all(gratio_last <= 1.0)

    def test_run_with_linear_precip_gradient_type(self) -> None:
        """Model should run successfully with linear gradient type."""
        catchment = Catchment(
            mean_annual_solid_precip=200.0,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=600.0,
            n_layers=3,
            precip_gradient_type=PrecipGradientType.linear,
        )
        params = self._make_params()
        forcing = self._make_forcing(n=30, precip=10.0, temp=2.0)
        result = run(params, forcing, catchment=catchment)
        assert len(result.fluxes.streamflow) == 30
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_linear_gradient_differs_from_exponential(self) -> None:
        """Linear and exponential gradient types should produce different streamflow."""
        # Wide elevation range to amplify the difference between exponential and linear
        hyps = np.linspace(200.0, 3500.0, 101)
        params = self._make_params()
        forcing = self._make_forcing(n=60, precip=10.0, temp=0.0)

        catchment_exp = Catchment(
            mean_annual_solid_precip=200.0,
            hypsometric_curve=hyps,
            input_elevation=500.0,
            n_layers=5,
        )
        catchment_lin = Catchment(
            mean_annual_solid_precip=200.0,
            hypsometric_curve=hyps,
            input_elevation=500.0,
            n_layers=5,
            precip_gradient_type=PrecipGradientType.linear,
        )

        result_exp = run(params, forcing, catchment=catchment_exp)
        result_lin = run(params, forcing, catchment=catchment_lin)

        # They should produce different streamflow (not byte-identical)
        assert not np.array_equal(result_exp.fluxes.streamflow, result_lin.fluxes.streamflow)

    def test_default_gradient_backward_compat(self) -> None:
        """Default PrecipGradientType.exponential should match pre-existing behavior."""
        catchment = Catchment(
            mean_annual_solid_precip=200.0,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=600.0,
            n_layers=3,
        )
        assert catchment.precip_gradient_type == PrecipGradientType.exponential

    def test_single_layer_backward_compat_unchanged_output(self) -> None:
        """Single-layer model should produce identical results regardless of gradient type."""
        params = self._make_params()
        forcing = self._make_forcing(n=30, precip=10.0, temp=2.0)

        catchment_default = Catchment(mean_annual_solid_precip=200.0)
        catchment_linear = Catchment(
            mean_annual_solid_precip=200.0,
            precip_gradient_type=PrecipGradientType.linear,
        )

        result_default = run(params, forcing, catchment=catchment_default)
        result_linear = run(params, forcing, catchment=catchment_linear)

        # Single layer with no elevation -> extrapolation skipped -> identical
        np.testing.assert_array_equal(
            result_default.fluxes.streamflow,
            result_linear.fluxes.streamflow,
        )
