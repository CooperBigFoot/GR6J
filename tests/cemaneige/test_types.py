"""Tests for CemaNeige data types: CemaNeige parameters and CemaNeigeSingleLayerState.

Tests cover validation, immutability, and initialization
for the core data structures used by the CemaNeige snow model.
"""

import numpy as np
import pytest

from gr6j.cemaneige.types import CemaNeige, CemaNeigeMultiLayerState, CemaNeigeSingleLayerState


class TestCemaNeige:
    """Tests for the CemaNeige frozen parameter dataclass."""

    def test_creates_with_valid_parameters(self) -> None:
        """CemaNeige instantiates correctly with typical values."""
        params = CemaNeige(ctg=0.97, kf=2.5)

        assert params.ctg == 0.97
        assert params.kf == 2.5

    def test_is_frozen_dataclass(self) -> None:
        """CemaNeige is immutable - assigning to fields should raise."""
        params = CemaNeige(ctg=0.97, kf=2.5)

        with pytest.raises(AttributeError):
            params.ctg = 0.5  # type: ignore[misc]


class TestCemaNeigeSingleLayerState:
    """Tests for the CemaNeigeSingleLayerState mutable dataclass."""

    def test_initialize_computes_gthreshold(self) -> None:
        """State.initialize sets gthreshold = 0.9 * mean_annual_solid_precip."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

        assert state.gthreshold == pytest.approx(0.9 * 150.0)  # 135.0

    def test_initialize_sets_glocalmax_equal_to_gthreshold(self) -> None:
        """Initially glocalmax equals gthreshold."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=200.0)

        assert state.glocalmax == state.gthreshold

    def test_initialize_starts_with_zero_snow(self) -> None:
        """Initial snow pack (g) is zero."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

        assert state.g == 0.0

    def test_initialize_starts_with_zero_thermal_state(self) -> None:
        """Initial thermal state (etg) is zero."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

        assert state.etg == 0.0

    def test_state_is_mutable(self) -> None:
        """State fields can be modified after initialization."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

        state.g = 100.0
        state.etg = -5.0

        assert state.g == 100.0
        assert state.etg == -5.0

    def test_creates_with_direct_values(self) -> None:
        """State can be created with direct attribute values."""
        state = CemaNeigeSingleLayerState(
            g=50.0,
            etg=-2.0,
            gthreshold=135.0,
            glocalmax=135.0,
        )

        assert state.g == 50.0
        assert state.etg == -2.0
        assert state.gthreshold == 135.0
        assert state.glocalmax == 135.0

    def test_initialize_with_zero_precip(self) -> None:
        """State initializes correctly with zero mean annual solid precipitation."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=0.0)

        assert state.g == 0.0
        assert state.etg == 0.0
        assert state.gthreshold == 0.0
        assert state.glocalmax == 0.0

    def test_initialize_with_large_precip(self) -> None:
        """State initializes correctly with large mean annual solid precipitation."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=5000.0)

        assert state.gthreshold == pytest.approx(0.9 * 5000.0)  # 4500.0
        assert state.glocalmax == pytest.approx(0.9 * 5000.0)

    def test_gthreshold_and_glocalmax_can_be_modified(self) -> None:
        """Threshold fields can be modified after initialization."""
        state = CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip=150.0)

        state.gthreshold = 200.0
        state.glocalmax = 250.0

        assert state.gthreshold == 200.0
        assert state.glocalmax == 250.0

    def test_all_fields_are_accessible(self) -> None:
        """All state fields are accessible as attributes."""
        state = CemaNeigeSingleLayerState(
            g=10.0,
            etg=-1.0,
            gthreshold=100.0,
            glocalmax=120.0,
        )

        assert hasattr(state, "g")
        assert hasattr(state, "etg")
        assert hasattr(state, "gthreshold")
        assert hasattr(state, "glocalmax")


class TestCemaNeigeMultiLayerState:
    """Tests for the CemaNeigeMultiLayerState multi-layer wrapper."""

    def test_initialize_creates_correct_number_of_layers(self) -> None:
        """Initialize creates the expected number of layer states."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=5, mean_annual_solid_precip=150.0)

        assert len(state) == 5

    def test_initialize_each_layer_has_zero_snow(self) -> None:
        """Each layer starts with zero snow pack."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=3, mean_annual_solid_precip=150.0)

        for i in range(3):
            assert state[i].g == 0.0

    def test_initialize_each_layer_has_correct_gthreshold(self) -> None:
        """Each layer's gthreshold is 0.9 * mean_annual_solid_precip."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=3, mean_annual_solid_precip=200.0)

        for i in range(3):
            assert state[i].gthreshold == pytest.approx(0.9 * 200.0)

    def test_getitem_returns_layer_state(self) -> None:
        """Indexing returns individual CemaNeigeSingleLayerState."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=3, mean_annual_solid_precip=150.0)

        layer = state[0]
        assert isinstance(layer, CemaNeigeSingleLayerState)

    def test_layer_states_are_independent(self) -> None:
        """Modifying one layer doesn't affect others."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=3, mean_annual_solid_precip=150.0)

        state[0].g = 100.0

        assert state[0].g == 100.0
        assert state[1].g == 0.0
        assert state[2].g == 0.0

    def test_creates_with_direct_layer_states(self) -> None:
        """Can be created directly from a list of layer states."""
        layers = [
            CemaNeigeSingleLayerState(g=10.0, etg=-1.0, gthreshold=100.0, glocalmax=100.0),
            CemaNeigeSingleLayerState(g=20.0, etg=-2.0, gthreshold=100.0, glocalmax=100.0),
        ]
        state = CemaNeigeMultiLayerState(layer_states=layers)

        assert len(state) == 2
        assert state[0].g == 10.0
        assert state[1].g == 20.0

    def test_single_layer_initialization(self) -> None:
        """Works correctly with n_layers=1."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=1, mean_annual_solid_precip=150.0)

        assert len(state) == 1


class TestCemaNeigeSingleLayerStateArrayProtocol:
    """Tests for CemaNeigeSingleLayerState array conversion."""

    def test_state_to_array_shape(self) -> None:
        """__array__ returns array of length 4."""
        state = CemaNeigeSingleLayerState(g=50, etg=-2, gthreshold=135, glocalmax=135)
        arr = np.asarray(state)
        assert arr.shape == (4,)

    def test_state_to_array_values(self) -> None:
        """__array__ places values in correct order."""
        state = CemaNeigeSingleLayerState(g=50, etg=-2, gthreshold=135, glocalmax=140)
        arr = np.asarray(state)

        assert arr[0] == 50.0
        assert arr[1] == -2.0
        assert arr[2] == 135.0
        assert arr[3] == 140.0

    def test_state_roundtrip(self) -> None:
        """State can be reconstructed from array."""
        original = CemaNeigeSingleLayerState(g=50, etg=-2, gthreshold=135, glocalmax=140)
        arr = np.asarray(original)
        restored = CemaNeigeSingleLayerState.from_array(arr)

        assert restored.g == original.g
        assert restored.etg == original.etg
        assert restored.gthreshold == original.gthreshold
        assert restored.glocalmax == original.glocalmax


class TestCemaNeigeMultiLayerStateArrayProtocol:
    """Tests for CemaNeigeMultiLayerState array conversion."""

    def test_state_to_array_shape(self) -> None:
        """__array__ returns 2D array with shape (n_layers, 4)."""
        state = CemaNeigeMultiLayerState.initialize(n_layers=5, mean_annual_solid_precip=150)
        arr = np.asarray(state)
        assert arr.shape == (5, 4)

    def test_state_to_array_values(self) -> None:
        """__array__ places layer values in correct positions."""
        layers = [
            CemaNeigeSingleLayerState(g=10, etg=-1, gthreshold=100, glocalmax=100),
            CemaNeigeSingleLayerState(g=20, etg=-2, gthreshold=100, glocalmax=110),
        ]
        state = CemaNeigeMultiLayerState(layer_states=layers)
        arr = np.asarray(state)

        assert arr[0, 0] == 10.0
        assert arr[0, 1] == -1.0
        assert arr[1, 0] == 20.0
        assert arr[1, 1] == -2.0

    def test_state_roundtrip(self) -> None:
        """Multi-layer state can be reconstructed from array."""
        layers = [
            CemaNeigeSingleLayerState(g=10, etg=-1, gthreshold=100, glocalmax=100),
            CemaNeigeSingleLayerState(g=20, etg=-2, gthreshold=150, glocalmax=160),
        ]
        original = CemaNeigeMultiLayerState(layer_states=layers)
        arr = np.asarray(original)
        restored = CemaNeigeMultiLayerState.from_array(arr)

        assert len(restored) == 2
        assert restored[0].g == 10.0
        assert restored[0].etg == -1.0
        assert restored[1].g == 20.0
        assert restored[1].gthreshold == 150.0
