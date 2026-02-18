"""Tests for GR6J-CemaNeige data types: Parameters and State."""

import numpy as np
import pytest
from pydrology import Catchment
from pydrology.models.gr6j_cemaneige import Parameters, State


class TestParameters:
    """Tests for the Parameters frozen dataclass."""

    def test_creates_with_valid_parameters(self) -> None:
        """Parameters instantiates correctly with typical values."""
        params = Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
        )

        assert params.x1 == 350.0
        assert params.x2 == 0.0
        assert params.x3 == 90.0
        assert params.x4 == 1.7
        assert params.x5 == 0.0
        assert params.x6 == 5.0
        assert params.ctg == 0.97
        assert params.kf == 2.5

    def test_is_frozen_dataclass(self) -> None:
        """Parameters is immutable - assigning to fields should raise."""
        params = Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
        )

        with pytest.raises(AttributeError):
            params.x1 = 500.0  # type: ignore[misc]

    def test_to_array_shape(self) -> None:
        """Parameters.__array__ returns array of length 8."""
        params = Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
        )
        arr = np.asarray(params)
        assert arr.shape == (8,)

    def test_array_roundtrip(self) -> None:
        """Parameters can be reconstructed from array."""
        original = Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
        )
        arr = np.asarray(original)
        restored = Parameters.from_array(arr)

        assert restored.x1 == original.x1
        assert restored.x2 == original.x2
        assert restored.x3 == original.x3
        assert restored.x4 == original.x4
        assert restored.x5 == original.x5
        assert restored.x6 == original.x6
        assert restored.ctg == original.ctg
        assert restored.kf == original.kf

    def test_from_array_wrong_length_raises(self) -> None:
        """from_array with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Expected array of length"):
            Parameters.from_array(np.zeros(5))

    def test_array_values_order(self) -> None:
        """np.asarray(params) returns values in order [x1, x2, x3, x4, x5, x6, ctg, kf]."""
        params = Parameters(
            x1=350.0,
            x2=0.5,
            x3=90.0,
            x4=1.7,
            x5=0.1,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
        )
        arr = np.asarray(params)

        assert arr[0] == 350.0
        assert arr[1] == 0.5
        assert arr[2] == 90.0
        assert arr[3] == 1.7
        assert arr[4] == 0.1
        assert arr[5] == 5.0
        assert arr[6] == 0.97
        assert arr[7] == 2.5


class TestState:
    """Tests for the State mutable dataclass."""

    @pytest.fixture
    def valid_params(self) -> Parameters:
        """Provide valid parameters for state initialization."""
        return Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
        )

    @pytest.fixture
    def catchment(self) -> Catchment:
        """Provide a single-layer catchment."""
        return Catchment(mean_annual_solid_precip=150.0)

    def test_initialize_creates_correct_stores(self, valid_params: Parameters, catchment: Catchment) -> None:
        """State.initialize sets stores to expected fractions of parameters."""
        state = State.initialize(valid_params, catchment)

        assert state.production_store == pytest.approx(0.3 * 350.0)
        assert state.routing_store == pytest.approx(0.5 * 90.0)
        assert state.exponential_store == 0.0

    def test_initialize_creates_correct_uh_shapes(self, valid_params: Parameters, catchment: Catchment) -> None:
        """State.initialize creates UH state arrays with correct dimensions."""
        state = State.initialize(valid_params, catchment)

        assert state.uh1_states.shape == (20,)
        assert state.uh2_states.shape == (40,)

    def test_initialize_creates_snow_layer_states(self, valid_params: Parameters, catchment: Catchment) -> None:
        """State.initialize creates snow layer states with correct shape for single layer."""
        state = State.initialize(valid_params, catchment)

        assert state.snow_layer_states.shape == (1, 4)

    def test_n_layers_property(self, valid_params: Parameters, catchment: Catchment) -> None:
        """n_layers property returns 1 for single-layer catchment."""
        state = State.initialize(valid_params, catchment)

        assert state.n_layers == 1

    def test_array_roundtrip(self, valid_params: Parameters, catchment: Catchment) -> None:
        """State can be reconstructed from array."""
        original = State.initialize(valid_params, catchment)
        original.production_store = 123.45
        original.routing_store = 67.89
        original.exponential_store = -1.5
        original.uh1_states[0] = 1.0
        original.uh2_states[10] = 2.0

        arr = np.asarray(original)
        restored = State.from_array(arr, n_layers=1)

        assert restored.production_store == original.production_store
        assert restored.routing_store == original.routing_store
        assert restored.exponential_store == original.exponential_store
        np.testing.assert_array_equal(restored.uh1_states, original.uh1_states)
        np.testing.assert_array_equal(restored.uh2_states, original.uh2_states)
        np.testing.assert_array_equal(restored.snow_layer_states, original.snow_layer_states)

    def test_from_array_wrong_length_raises(self) -> None:
        """from_array with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Expected array of length"):
            State.from_array(np.zeros(10), n_layers=1)

    def test_multi_layer_initialize(self, valid_params: Parameters) -> None:
        """State.initialize with multi-layer catchment creates correct snow states."""
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=3,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )
        state = State.initialize(valid_params, catchment)

        assert state.n_layers == 3
        assert state.snow_layer_states.shape == (3, 4)

    def test_multi_layer_array_roundtrip(self, valid_params: Parameters) -> None:
        """Multi-layer state can be reconstructed from array."""
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=3,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )
        original = State.initialize(valid_params, catchment)
        original.production_store = 200.0
        original.routing_store = 30.0
        original.exponential_store = -0.5
        original.snow_layer_states[1, 0] = 50.0  # Set snow pack on second layer

        arr = np.asarray(original)
        restored = State.from_array(arr, n_layers=3)

        assert restored.production_store == original.production_store
        assert restored.routing_store == original.routing_store
        assert restored.exponential_store == original.exponential_store
        np.testing.assert_array_equal(restored.uh1_states, original.uh1_states)
        np.testing.assert_array_equal(restored.uh2_states, original.uh2_states)
        np.testing.assert_array_equal(restored.snow_layer_states, original.snow_layer_states)


class TestPerBandGthreshold:
    """Tests for per-band gthreshold initialization in State.initialize()."""

    def _make_catchment(self, n_layers: int = 5, **kwargs) -> Catchment:
        defaults = {
            "mean_annual_solid_precip": 200.0,
            "hypsometric_curve": np.linspace(200.0, 2000.0, 101),
            "input_elevation": 600.0,
            "n_layers": n_layers,
        }
        defaults.update(kwargs)
        return Catchment(**defaults)

    def test_multi_layer_per_band_gthreshold_varies(self) -> None:
        catchment = self._make_catchment()
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        state = State.initialize(params, catchment)
        gthresholds = state.snow_layer_states[:, 2]
        # Should not all be equal
        assert not np.allclose(gthresholds, gthresholds[0])
        # Should increase with layer index (elevations increase)
        for i in range(len(gthresholds) - 1):
            assert gthresholds[i] < gthresholds[i + 1]

    def test_single_layer_unchanged(self) -> None:
        catchment = Catchment(mean_annual_solid_precip=200.0)
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        state = State.initialize(params, catchment)
        expected = 0.9 * 200.0
        assert state.snow_layer_states[0, 2] == pytest.approx(expected)
        assert state.snow_layer_states[0, 3] == pytest.approx(expected)

    def test_glocalmax_matches_gthreshold(self) -> None:
        catchment = self._make_catchment()
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        state = State.initialize(params, catchment)
        np.testing.assert_array_equal(state.snow_layer_states[:, 2], state.snow_layer_states[:, 3])

    def test_custom_precip_gradient_wider_spread(self) -> None:
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        catchment_default = self._make_catchment()
        catchment_custom = self._make_catchment(precip_gradient=0.001)  # larger gradient
        state_default = State.initialize(params, catchment_default)
        state_custom = State.initialize(params, catchment_custom)
        # Larger gradient -> wider spread in gthresholds
        spread_default = state_default.snow_layer_states[-1, 2] - state_default.snow_layer_states[0, 2]
        spread_custom = state_custom.snow_layer_states[-1, 2] - state_custom.snow_layer_states[0, 2]
        assert spread_custom > spread_default
