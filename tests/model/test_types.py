"""Tests for GR6J data types: Parameters and State.

Tests cover validation, immutability, initialization, and boundary warnings
for the core data structures used throughout the model.
"""

import logging

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

    def test_warns_when_x1_below_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when x1 is below typical range (< 1.0)."""
        with caplog.at_level(logging.WARNING):
            Parameters(
                x1=0.5,  # Below minimum of 1.0
                x2=0.5,
                x3=90.0,
                x4=1.7,
                x5=0.1,
                x6=5.0,
            )

        assert len(caplog.records) == 1
        assert "x1=0.5000" in caplog.text
        assert "outside typical range" in caplog.text
        assert "[1.00, 2500.00]" in caplog.text

    def test_warns_when_x1_above_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when x1 exceeds typical range (> 2500)."""
        with caplog.at_level(logging.WARNING):
            Parameters(
                x1=3000.0,  # Above maximum of 2500
                x2=0.5,
                x3=90.0,
                x4=1.7,
                x5=0.1,
                x6=5.0,
            )

        assert len(caplog.records) == 1
        assert "x1=3000.0000" in caplog.text
        assert "outside typical range" in caplog.text

    def test_warns_when_x4_below_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when x4 is below typical range (< 0.5)."""
        with caplog.at_level(logging.WARNING):
            Parameters(
                x1=350.0,
                x2=0.5,
                x3=90.0,
                x4=0.2,  # Below minimum of 0.5
                x5=0.1,
                x6=5.0,
            )

        assert len(caplog.records) == 1
        assert "x4=0.2000" in caplog.text
        assert "outside typical range" in caplog.text
        assert "[0.50, 10.00]" in caplog.text

    def test_no_warning_within_typical_ranges(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warnings logged when all parameters are within typical ranges."""
        with caplog.at_level(logging.WARNING):
            Parameters(
                x1=350.0,
                x2=0.5,
                x3=90.0,
                x4=1.7,
                x5=0.1,
                x6=5.0,
            )

        assert len(caplog.records) == 0


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
