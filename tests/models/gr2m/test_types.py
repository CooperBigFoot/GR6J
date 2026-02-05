"""Tests for GR2M data types: Parameters and State.

Tests cover validation, immutability, and initialization
for the core data structures used throughout the model.
"""

import numpy as np
import pytest

from pydrology.models.gr2m.types import Parameters, State


class TestParameters:
    """Tests for the Parameters frozen dataclass."""

    def test_creates_with_valid_parameters(self) -> None:
        """Parameters instantiates correctly with typical calibration values."""
        params = Parameters(
            x1=500.0,
            x2=1.0,
        )

        assert params.x1 == 500.0
        assert params.x2 == 1.0

    def test_is_frozen_dataclass(self) -> None:
        """Parameters is immutable - assigning to fields should raise."""
        params = Parameters(
            x1=500.0,
            x2=1.0,
        )

        with pytest.raises(AttributeError):
            params.x1 = 600.0  # type: ignore[misc]

    def test_boundary_values(self) -> None:
        """Parameters accepts boundary values from typical ranges."""
        # Minimum typical values
        params_min = Parameters(x1=1.0, x2=0.2)
        assert params_min.x1 == 1.0
        assert params_min.x2 == 0.2

        # Maximum typical values
        params_max = Parameters(x1=2500.0, x2=2.0)
        assert params_max.x1 == 2500.0
        assert params_max.x2 == 2.0


class TestState:
    """Tests for the State mutable dataclass."""

    @pytest.fixture
    def valid_params(self) -> Parameters:
        """Provide valid parameters for state initialization."""
        return Parameters(
            x1=500.0,
            x2=1.0,
        )

    def test_initialize_creates_correct_fractions(self, valid_params: Parameters) -> None:
        """State.initialize sets stores to expected fractions of X1."""
        state = State.initialize(valid_params)

        # Both stores = 0.3 * X1 = 0.3 * 500 = 150
        assert state.production_store == pytest.approx(0.3 * 500.0)
        assert state.routing_store == pytest.approx(0.3 * 500.0)

    def test_state_is_mutable(self, valid_params: Parameters) -> None:
        """State fields can be modified after initialization."""
        state = State.initialize(valid_params)

        # Modify scalar fields
        state.production_store = 200.0
        state.routing_store = 80.0

        assert state.production_store == 200.0
        assert state.routing_store == 80.0

    def test_manual_initialization(self) -> None:
        """State can be manually initialized with specific values."""
        state = State(
            production_store=123.45,
            routing_store=67.89,
        )

        assert state.production_store == 123.45
        assert state.routing_store == 67.89


class TestStateArrayProtocol:
    """Tests for State array conversion methods."""

    def test_state_to_array_shape(self) -> None:
        """State.__array__ returns array of length 2."""
        params = Parameters(x1=500, x2=1.0)
        state = State.initialize(params)
        arr = np.asarray(state)
        assert arr.shape == (2,)

    def test_state_to_array_values(self) -> None:
        """State.__array__ places values in correct positions."""
        params = Parameters(x1=500, x2=1.0)
        state = State.initialize(params)
        arr = np.asarray(state)

        assert arr[0] == state.production_store
        assert arr[1] == state.routing_store

    def test_state_roundtrip(self) -> None:
        """State can be reconstructed from array."""
        params = Parameters(x1=500, x2=1.0)
        original = State.initialize(params)
        original.production_store = 123.45
        original.routing_store = 67.89

        arr = np.asarray(original)
        restored = State.from_array(arr)

        assert restored.production_store == original.production_store
        assert restored.routing_store == original.routing_store


class TestParametersArrayProtocol:
    """Tests for Parameters array conversion methods."""

    def test_params_to_array_shape(self) -> None:
        """Parameters.__array__ returns array of length 2."""
        params = Parameters(x1=500, x2=1.0)
        arr = np.asarray(params)
        assert arr.shape == (2,)

    def test_params_to_array_values(self) -> None:
        """Parameters.__array__ places values in correct order."""
        params = Parameters(x1=500, x2=1.0)
        arr = np.asarray(params)

        assert arr[0] == 500.0
        assert arr[1] == 1.0

    def test_params_roundtrip(self) -> None:
        """Parameters can be reconstructed from array."""
        original = Parameters(x1=500, x2=1.0)
        arr = np.asarray(original)
        restored = Parameters.from_array(arr)

        assert restored.x1 == original.x1
        assert restored.x2 == original.x2
