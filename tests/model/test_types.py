"""Tests for GR6J data types: Parameters and State.

Tests cover validation, immutability, and initialization
for the core data structures used throughout the model.
"""

import dataclasses

import numpy as np
import pytest

from gr6j.model.types import Parameters, State


class TestParameters:
    """Tests for the Parameters frozen dataclass."""

    def test_creates_with_valid_parameters(self) -> None:
        """Parameters instantiates correctly with typical calibration values."""
        params = Parameters(
            x1=350.0,
            x2=0.5,
            x3=90.0,
            x4=1.7,
            x5=0.1,
            x6=5.0,
        )

        assert params.x1 == 350.0
        assert params.x2 == 0.5
        assert params.x3 == 90.0
        assert params.x4 == 1.7
        assert params.x5 == 0.1
        assert params.x6 == 5.0

    def test_is_frozen_dataclass(self) -> None:
        """Parameters is immutable - assigning to fields should raise."""
        params = Parameters(
            x1=350.0,
            x2=0.5,
            x3=90.0,
            x4=1.7,
            x5=0.1,
            x6=5.0,
        )

        with pytest.raises(AttributeError):
            params.x1 = 500.0  # type: ignore[misc]


class TestState:
    """Tests for the State mutable dataclass."""

    @pytest.fixture
    def valid_params(self) -> Parameters:
        """Provide valid parameters for state initialization."""
        return Parameters(
            x1=350.0,
            x2=0.5,
            x3=90.0,
            x4=1.7,
            x5=0.1,
            x6=5.0,
        )

    def test_initialize_creates_correct_fractions(self, valid_params: Parameters) -> None:
        """State.initialize sets stores to expected fractions of parameters."""
        state = State.initialize(valid_params)

        # S = 0.3 * X1 = 0.3 * 350 = 105
        assert state.production_store == pytest.approx(0.3 * 350.0)
        # R = 0.5 * X3 = 0.5 * 90 = 45
        assert state.routing_store == pytest.approx(0.5 * 90.0)
        # Exp = 0
        assert state.exponential_store == 0.0

    def test_initialize_creates_correct_uh_shapes(self, valid_params: Parameters) -> None:
        """State.initialize creates UH state arrays with correct dimensions."""
        state = State.initialize(valid_params)

        # UH1 has NH=20 elements
        assert state.uh1_states.shape == (20,)
        # UH2 has 2*NH=40 elements
        assert state.uh2_states.shape == (40,)

    def test_state_is_mutable(self, valid_params: Parameters) -> None:
        """State fields can be modified after initialization."""
        state = State.initialize(valid_params)

        # Modify scalar fields
        state.production_store = 200.0
        state.routing_store = 50.0
        state.exponential_store = -1.5

        assert state.production_store == 200.0
        assert state.routing_store == 50.0
        assert state.exponential_store == -1.5

    def test_uh_states_initialized_to_zeros(self, valid_params: Parameters) -> None:
        """Both UH state arrays are initialized to all zeros."""
        state = State.initialize(valid_params)

        np.testing.assert_array_equal(state.uh1_states, np.zeros(20))
        np.testing.assert_array_equal(state.uh2_states, np.zeros(40))


class TestParametersWithSnow:
    """Tests for Parameters.snow field and has_snow property."""

    def test_creates_without_snow_by_default(self):
        """Default Parameters has snow=None."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        assert params.snow is None

    def test_creates_with_snow_parameter(self):
        """Can create Parameters with snow enabled."""
        from gr6j import CemaNeige

        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5, snow=CemaNeige(ctg=0.97, kf=2.5))
        assert params.snow is not None
        assert params.snow.ctg == 0.97
        assert params.snow.kf == 2.5

    def test_has_snow_false_when_snow_none(self):
        """has_snow property returns False when snow=None."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        assert params.has_snow is False

    def test_has_snow_true_when_snow_set(self):
        """has_snow property returns True when snow is set."""
        from gr6j import CemaNeige

        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5, snow=CemaNeige(ctg=0.97, kf=2.5))
        assert params.has_snow is True

    def test_snow_is_frozen(self):
        """Cannot modify snow field after creation."""
        from gr6j import CemaNeige

        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5, snow=CemaNeige(ctg=0.97, kf=2.5))
        with pytest.raises(dataclasses.FrozenInstanceError):
            params.snow = None


class TestStateArrayProtocol:
    """Tests for State array conversion methods."""

    def test_state_to_array_shape(self) -> None:
        """State.__array__ returns array of length 63."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        state = State.initialize(params)
        arr = np.asarray(state)
        assert arr.shape == (63,)

    def test_state_to_array_values(self) -> None:
        """State.__array__ places values in correct positions."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        state = State.initialize(params)
        arr = np.asarray(state)

        assert arr[0] == state.production_store
        assert arr[1] == state.routing_store
        assert arr[2] == state.exponential_store
        np.testing.assert_array_equal(arr[3:23], state.uh1_states)
        np.testing.assert_array_equal(arr[23:63], state.uh2_states)

    def test_state_roundtrip(self) -> None:
        """State can be reconstructed from array."""
        params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        original = State.initialize(params)
        original.production_store = 123.45
        original.routing_store = 67.89
        original.exponential_store = -1.5
        original.uh1_states[0] = 1.0
        original.uh2_states[10] = 2.0

        arr = np.asarray(original)
        restored = State.from_array(arr)

        assert restored.production_store == original.production_store
        assert restored.routing_store == original.routing_store
        assert restored.exponential_store == original.exponential_store
        np.testing.assert_array_equal(restored.uh1_states, original.uh1_states)
        np.testing.assert_array_equal(restored.uh2_states, original.uh2_states)


class TestParametersArrayProtocol:
    """Tests for Parameters array conversion methods."""

    def test_params_to_array_shape(self) -> None:
        """Parameters.__array__ returns array of length 6."""
        params = Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)
        arr = np.asarray(params)
        assert arr.shape == (6,)

    def test_params_to_array_values(self) -> None:
        """Parameters.__array__ places values in correct order."""
        params = Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)
        arr = np.asarray(params)

        assert arr[0] == 350.0
        assert arr[1] == 0.5
        assert arr[2] == 90.0
        assert arr[3] == 1.7
        assert arr[4] == 0.1
        assert arr[5] == 5.0

    def test_params_roundtrip(self) -> None:
        """Parameters can be reconstructed from array."""
        original = Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5)
        arr = np.asarray(original)
        restored = Parameters.from_array(arr)

        assert restored.x1 == original.x1
        assert restored.x2 == original.x2
        assert restored.x3 == original.x3
        assert restored.x4 == original.x4
        assert restored.x5 == original.x5
        assert restored.x6 == original.x6
        assert restored.snow is None

    def test_params_roundtrip_preserves_snow(self) -> None:
        """Parameters.from_array can preserve snow parameter."""
        from gr6j import CemaNeige

        snow = CemaNeige(ctg=0.97, kf=2.5)
        original = Parameters(x1=350, x2=0.5, x3=90, x4=1.7, x5=0.1, x6=5, snow=snow)
        arr = np.asarray(original)
        restored = Parameters.from_array(arr, snow=snow)

        assert restored.snow is not None
        assert restored.snow.ctg == 0.97
        assert restored.snow.kf == 2.5
