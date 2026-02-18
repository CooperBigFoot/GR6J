"""Tests for elevation utility functions.

Tests cover layer derivation from hypsometric curves and temperature/precipitation
extrapolation across elevation bands.
"""

import numpy as np
import pytest
from pydrology.utils.elevation import (
    derive_layers,
    extrapolate_precipitation,
    extrapolate_precipitation_linear,
    extrapolate_temperature,
)


class TestDeriveLayers:
    """Tests for derive_layers function."""

    def test_uniform_hypsometric_produces_uniform_elevations(self) -> None:
        """Flat catchment produces identical layer elevations."""
        hypsometric = np.full(101, 500.0)
        elevations, fractions = derive_layers(hypsometric, n_layers=5)

        assert len(elevations) == 5
        assert len(fractions) == 5
        np.testing.assert_array_almost_equal(elevations, 500.0)

    def test_fractions_are_uniform(self) -> None:
        """All layers have equal area fraction 1/n_layers."""
        hypsometric = np.linspace(100.0, 2000.0, 101)
        _, fractions = derive_layers(hypsometric, n_layers=5)

        np.testing.assert_array_almost_equal(fractions, 0.2)

    def test_fractions_sum_to_one(self) -> None:
        """Area fractions sum to 1.0."""
        hypsometric = np.linspace(100.0, 2000.0, 101)
        _, fractions = derive_layers(hypsometric, n_layers=3)

        assert sum(fractions) == pytest.approx(1.0)

    def test_elevations_increase_with_layer_index(self) -> None:
        """Layer elevations increase monotonically for ascending hypsometric curve."""
        hypsometric = np.linspace(100.0, 2000.0, 101)
        elevations, _ = derive_layers(hypsometric, n_layers=5)

        for i in range(len(elevations) - 1):
            assert elevations[i] < elevations[i + 1]

    def test_single_layer_gives_midpoint(self) -> None:
        """Single layer should be the midpoint of min and max elevation."""
        hypsometric = np.linspace(200.0, 1800.0, 101)
        elevations, fractions = derive_layers(hypsometric, n_layers=1)

        assert len(elevations) == 1
        assert fractions[0] == pytest.approx(1.0)
        assert elevations[0] == pytest.approx(1000.0)

    def test_two_layers_split_at_midpoint(self) -> None:
        """Two layers split the hypsometric curve at the 50th percentile."""
        hypsometric = np.linspace(0.0, 2000.0, 101)
        elevations, _ = derive_layers(hypsometric, n_layers=2)

        assert len(elevations) == 2
        # Lower band: midpoint of [0, 1000]
        assert elevations[0] == pytest.approx(500.0)
        # Upper band: midpoint of [1000, 2000]
        assert elevations[1] == pytest.approx(1500.0)

    def test_output_shapes(self) -> None:
        """Output arrays have correct shapes."""
        hypsometric = np.linspace(100.0, 2000.0, 101)
        elevations, fractions = derive_layers(hypsometric, n_layers=10)

        assert elevations.shape == (10,)
        assert fractions.shape == (10,)


class TestExtrapolateTemperature:
    """Tests for extrapolate_temperature function."""

    def test_same_elevation_returns_input(self) -> None:
        """No change when layer is at input elevation."""
        result = extrapolate_temperature(15.0, 500.0, 500.0)
        assert result == pytest.approx(15.0)

    def test_higher_elevation_colder(self) -> None:
        """Temperature decreases at higher elevation."""
        result = extrapolate_temperature(15.0, 500.0, 1500.0)
        # 1000m higher, gradient 0.6°C/100m -> 6°C cooler
        assert result == pytest.approx(15.0 - 6.0)

    def test_lower_elevation_warmer(self) -> None:
        """Temperature increases at lower elevation."""
        result = extrapolate_temperature(15.0, 1000.0, 500.0)
        # 500m lower, gradient 0.6°C/100m -> 3°C warmer
        assert result == pytest.approx(15.0 + 3.0)

    def test_custom_gradient(self) -> None:
        """Custom gradient is applied correctly."""
        result = extrapolate_temperature(10.0, 500.0, 1000.0, gradient=1.0)
        # 500m higher, gradient 1.0°C/100m -> 5°C cooler
        assert result == pytest.approx(5.0)

    def test_formula_correctness(self) -> None:
        """Verify formula: T_layer = T_input - gradient * (Z_layer - Z_input) / 100."""
        # Known values
        t_input = 20.0
        z_input = 300.0
        z_layer = 1300.0
        gradient = 0.6

        expected = t_input - gradient * (z_layer - z_input) / 100.0
        result = extrapolate_temperature(t_input, z_input, z_layer, gradient=gradient)
        assert result == pytest.approx(expected)


class TestExtrapolatePrecipitation:
    """Tests for extrapolate_precipitation function."""

    def test_same_elevation_returns_input(self) -> None:
        """No change when layer is at input elevation."""
        result = extrapolate_precipitation(10.0, 500.0, 500.0)
        assert result == pytest.approx(10.0)

    def test_higher_elevation_more_precip(self) -> None:
        """Precipitation increases at higher elevation (orographic enhancement)."""
        result = extrapolate_precipitation(10.0, 500.0, 1500.0)
        assert result > 10.0

    def test_lower_elevation_less_precip(self) -> None:
        """Precipitation decreases at lower elevation."""
        result = extrapolate_precipitation(10.0, 1000.0, 500.0)
        assert result < 10.0

    def test_zero_precip_stays_zero(self) -> None:
        """Zero precipitation stays zero regardless of elevation."""
        result = extrapolate_precipitation(0.0, 500.0, 2000.0)
        assert result == pytest.approx(0.0)

    def test_elevation_cap_applies(self) -> None:
        """Elevations above cap are capped before calculation."""
        # Both above cap -> no elevation difference -> no change
        result = extrapolate_precipitation(10.0, 5000.0, 6000.0)
        assert result == pytest.approx(10.0)

    def test_input_capped_but_layer_below(self) -> None:
        """Input above cap, layer below cap."""
        result = extrapolate_precipitation(10.0, 5000.0, 3000.0)
        # Effective input = 4000, effective layer = 3000
        # Lower elevation -> less precip
        assert result < 10.0

    def test_formula_correctness(self) -> None:
        """Verify formula: P_layer = P_input * exp(gradient * (Z_layer_eff - Z_input_eff))."""
        p_input = 10.0
        z_input = 500.0
        z_layer = 1500.0
        gradient = 0.00041

        expected = p_input * np.exp(gradient * (z_layer - z_input))
        result = extrapolate_precipitation(p_input, z_input, z_layer, gradient=gradient)
        assert result == pytest.approx(expected)

    def test_custom_gradient(self) -> None:
        """Custom gradient is applied correctly."""
        result_default = extrapolate_precipitation(10.0, 500.0, 1500.0)
        result_custom = extrapolate_precipitation(10.0, 500.0, 1500.0, gradient=0.001)

        # Higher gradient -> more enhancement
        assert result_custom > result_default


class TestExtrapolatePrecipitationLinear:
    """Tests for extrapolate_precipitation_linear function."""

    def test_same_elevation_returns_input(self) -> None:
        result = extrapolate_precipitation_linear(10.0, 500.0, 500.0)
        assert result == pytest.approx(10.0)

    def test_higher_elevation_more_precip(self) -> None:
        result = extrapolate_precipitation_linear(10.0, 500.0, 1500.0)
        # 10 * (1 + 0.0004 * 1000) = 14.0
        assert result == pytest.approx(14.0)

    def test_lower_elevation_less_precip(self) -> None:
        result = extrapolate_precipitation_linear(10.0, 1000.0, 500.0)
        # 10 * (1 + 0.0004 * (-500)) = 8.0
        assert result == pytest.approx(8.0)

    def test_clamps_to_zero(self) -> None:
        # Very large negative ΔZ -> would go negative without clamp
        result = extrapolate_precipitation_linear(10.0, 4000.0, 0.0, gradient=0.001)
        # 10 * (1 + 0.001 * (-4000)) = 10 * (-3.0) = -30 -> clamped to 0
        assert result == 0.0

    def test_zero_precip_stays_zero(self) -> None:
        result = extrapolate_precipitation_linear(0.0, 500.0, 2000.0)
        assert result == pytest.approx(0.0)

    def test_elevation_cap_applies(self) -> None:
        # Both above cap -> no elevation difference -> no change
        result = extrapolate_precipitation_linear(10.0, 5000.0, 6000.0)
        assert result == pytest.approx(10.0)
