"""Tests for CemaNeige data types: CemaNeige parameters and CemaNeigeSingleLayerState.

Tests cover validation, immutability, initialization, and boundary warnings
for the core data structures used by the CemaNeige snow model.
"""

import logging

import pytest

from gr6j.cemaneige.types import CemaNeige, CemaNeigeSingleLayerState


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

    def test_warns_when_ctg_below_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when ctg is below 0."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=-0.1, kf=2.5)

        assert len(caplog.records) == 1
        assert "ctg" in caplog.text
        assert "outside typical range" in caplog.text

    def test_warns_when_ctg_above_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when ctg exceeds 1."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=1.5, kf=2.5)

        assert len(caplog.records) == 1
        assert "ctg" in caplog.text
        assert "outside typical range" in caplog.text

    def test_warns_when_kf_below_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when kf is below 0."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=0.97, kf=-1.0)

        assert len(caplog.records) == 1
        assert "kf" in caplog.text
        assert "outside typical range" in caplog.text

    def test_warns_when_kf_above_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when kf exceeds 200."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=0.97, kf=250.0)

        assert len(caplog.records) == 1
        assert "kf" in caplog.text
        assert "outside typical range" in caplog.text

    def test_no_warning_within_typical_ranges(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warnings logged when all parameters are within typical ranges."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=0.97, kf=2.5)

        assert len(caplog.records) == 0

    def test_bounds_class_variable_exists(self) -> None:
        """CemaNeige has BOUNDS class variable for parameter bounds."""
        assert hasattr(CemaNeige, "BOUNDS")
        assert "ctg" in CemaNeige.BOUNDS
        assert "kf" in CemaNeige.BOUNDS

    def test_bounds_contains_correct_ranges(self) -> None:
        """BOUNDS contains expected ranges for each parameter."""
        assert CemaNeige.BOUNDS["ctg"] == (0.0, 1.0)
        assert CemaNeige.BOUNDS["kf"] == (0.0, 200.0)

    def test_warns_multiple_parameters_out_of_range(self, caplog: pytest.LogCaptureFixture) -> None:
        """Multiple warnings logged when multiple parameters are out of range."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=-0.5, kf=300.0)

        assert len(caplog.records) == 2
        assert "ctg" in caplog.text
        assert "kf" in caplog.text

    def test_at_lower_boundary_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when parameter is exactly at lower boundary."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=0.0, kf=0.0)

        assert len(caplog.records) == 0

    def test_at_upper_boundary_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when parameter is exactly at upper boundary."""
        with caplog.at_level(logging.WARNING):
            CemaNeige(ctg=1.0, kf=200.0)

        assert len(caplog.records) == 0


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
