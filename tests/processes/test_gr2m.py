"""Tests for GR2M core process functions.

Tests verify the mathematical correctness of each process function
according to AIRGR_MODEL_DEFINITION.md equations.
"""

import numpy as np
import pytest

from pydrology.models.gr2m.processes import (
    compute_streamflow,
    percolation,
    production_store_evaporation,
    production_store_rainfall,
    routing_store_update,
)


class TestProductionStoreRainfall:
    """Tests for production_store_rainfall function (Section 5.1)."""

    def test_store_increases_with_rainfall(self) -> None:
        """Production store should increase when rainfall is added."""
        precip = 50.0
        store = 100.0
        x1 = 500.0

        s1, p1, ps = production_store_rainfall(precip, store, x1)

        # Store should increase
        assert s1 > store
        # Some rainfall becomes excess
        assert p1 > 0.0
        # Some rainfall enters storage
        assert ps > 0.0
        # Water balance: P = P1 + PS
        assert np.isclose(p1 + ps, precip)

    def test_zero_precipitation(self) -> None:
        """Zero precipitation produces zero outputs."""
        precip = 0.0
        store = 100.0
        x1 = 500.0

        s1, p1, ps = production_store_rainfall(precip, store, x1)

        # Store unchanged
        assert np.isclose(s1, store)
        # No excess or storage
        assert p1 == 0.0
        assert ps == 0.0

    def test_full_store_more_excess(self) -> None:
        """Nearly full store produces more rainfall excess."""
        precip = 50.0
        x1 = 500.0

        # Empty store
        _, p1_empty, _ = production_store_rainfall(precip, 10.0, x1)
        # Full store
        _, p1_full, _ = production_store_rainfall(precip, 480.0, x1)

        assert p1_full > p1_empty

    def test_tanh_safeguard_high_precip(self) -> None:
        """High precipitation (WS > 13) should not cause overflow."""
        precip = 10000.0  # Very high to trigger safeguard
        store = 100.0
        x1 = 500.0

        s1, p1, ps = production_store_rainfall(precip, store, x1)

        # Results should be finite
        assert np.isfinite(s1)
        assert np.isfinite(p1)
        assert np.isfinite(ps)
        # Water balance
        assert np.isclose(p1 + ps, precip)

    def test_water_balance(self) -> None:
        """P = P1 + PS (rainfall = excess + storage)."""
        test_cases = [
            (10.0, 50.0, 300.0),
            (100.0, 100.0, 500.0),
            (200.0, 400.0, 1000.0),
        ]

        for precip, store, x1 in test_cases:
            s1, p1, ps = production_store_rainfall(precip, store, x1)
            assert np.isclose(p1 + ps, precip), f"Failed for P={precip}, S={store}, X1={x1}"


class TestProductionStoreEvaporation:
    """Tests for production_store_evaporation function (Section 5.2)."""

    def test_store_decreases_with_evaporation(self) -> None:
        """Production store should decrease when evaporation occurs."""
        pet = 30.0
        s1 = 200.0
        x1 = 500.0

        s2, ae = production_store_evaporation(pet, s1, x1)

        # Store should decrease
        assert s2 < s1
        # Actual ET should be positive
        assert ae > 0.0
        # Water balance
        assert np.isclose(s1 - s2, ae)

    def test_zero_pet(self) -> None:
        """Zero PET produces no change."""
        pet = 0.0
        s1 = 200.0
        x1 = 500.0

        s2, ae = production_store_evaporation(pet, s1, x1)

        assert np.isclose(s2, s1)
        assert ae == 0.0

    def test_high_pet_limited_by_store(self) -> None:
        """Actual ET is limited by available store."""
        pet = 1000.0  # Very high demand
        s1 = 50.0  # Low store
        x1 = 500.0

        s2, ae = production_store_evaporation(pet, s1, x1)

        # Store should not go negative
        assert s2 >= 0.0
        # AE limited by available water
        assert ae <= s1

    def test_tanh_safeguard_high_pet(self) -> None:
        """High PET (WS > 13) should not cause overflow."""
        pet = 10000.0  # Very high to trigger safeguard
        s1 = 200.0
        x1 = 500.0

        s2, ae = production_store_evaporation(pet, s1, x1)

        # Results should be finite
        assert np.isfinite(s2)
        assert np.isfinite(ae)
        # Store non-negative
        assert s2 >= 0.0

    def test_water_balance(self) -> None:
        """S1 = S2 + AE (initial = final + evaporation)."""
        test_cases = [
            (20.0, 100.0, 300.0),
            (50.0, 200.0, 500.0),
            (100.0, 400.0, 1000.0),
        ]

        for pet, s1, x1 in test_cases:
            s2, ae = production_store_evaporation(pet, s1, x1)
            assert np.isclose(s1, s2 + ae), f"Failed for E={pet}, S1={s1}, X1={x1}"


class TestPercolation:
    """Tests for percolation function (Section 5.3)."""

    def test_percolation_positive_for_positive_store(self) -> None:
        """Percolation should be positive when store is positive."""
        s2 = 200.0
        x1 = 500.0

        s_final, p2 = percolation(s2, x1)

        assert p2 > 0.0
        assert s_final < s2

    def test_zero_percolation_for_zero_store(self) -> None:
        """Zero store produces zero percolation."""
        s2 = 0.0
        x1 = 500.0

        s_final, p2 = percolation(s2, x1)

        assert p2 == 0.0
        assert s_final == 0.0

    def test_percolation_increases_with_store(self) -> None:
        """Higher store produces more percolation."""
        x1 = 500.0

        _, p2_low = percolation(50.0, x1)
        _, p2_high = percolation(400.0, x1)

        assert p2_high > p2_low

    def test_water_balance(self) -> None:
        """S2 = S_final + P2."""
        test_cases = [
            (50.0, 300.0),
            (200.0, 500.0),
            (400.0, 1000.0),
        ]

        for s2, x1 in test_cases:
            s_final, p2 = percolation(s2, x1)
            assert np.isclose(s2, s_final + p2), f"Failed for S2={s2}, X1={x1}"

    def test_negative_store_handled(self) -> None:
        """Negative store is clamped to zero."""
        s2 = -10.0
        x1 = 500.0

        s_final, p2 = percolation(s2, x1)

        assert s_final == 0.0
        assert p2 == 0.0


class TestRoutingStoreUpdate:
    """Tests for routing_store_update function (Section 5.4)."""

    def test_x2_greater_than_one_increases_water(self) -> None:
        """X2 > 1 causes water gain."""
        routing_store = 50.0
        p3 = 30.0
        x2 = 1.5

        r2, aexch = routing_store_update(routing_store, p3, x2)

        # R2 = X2 * (R + P3) = 1.5 * 80 = 120
        expected_r2 = x2 * (routing_store + p3)
        assert np.isclose(r2, expected_r2)
        # Exchange should be positive (gain)
        assert aexch > 0.0

    def test_x2_less_than_one_decreases_water(self) -> None:
        """X2 < 1 causes water loss."""
        routing_store = 50.0
        p3 = 30.0
        x2 = 0.8

        r2, aexch = routing_store_update(routing_store, p3, x2)

        # R2 = X2 * (R + P3) = 0.8 * 80 = 64
        expected_r2 = x2 * (routing_store + p3)
        assert np.isclose(r2, expected_r2)
        # Exchange should be negative (loss)
        assert aexch < 0.0

    def test_x2_equals_one_no_exchange(self) -> None:
        """X2 = 1 causes no exchange."""
        routing_store = 50.0
        p3 = 30.0
        x2 = 1.0

        r2, aexch = routing_store_update(routing_store, p3, x2)

        assert np.isclose(r2, routing_store + p3)
        assert np.isclose(aexch, 0.0)

    def test_zero_inflow(self) -> None:
        """Zero inflow still applies exchange."""
        routing_store = 100.0
        p3 = 0.0
        x2 = 1.2

        r2, aexch = routing_store_update(routing_store, p3, x2)

        assert np.isclose(r2, x2 * routing_store)

    def test_exchange_calculation(self) -> None:
        """AEXCH = R2 - R1."""
        routing_store = 60.0
        p3 = 40.0
        x2 = 1.3

        r2, aexch = routing_store_update(routing_store, p3, x2)

        r1 = routing_store + p3
        expected_aexch = r2 - r1
        assert np.isclose(aexch, expected_aexch)


class TestComputeStreamflow:
    """Tests for compute_streamflow function (Section 5.5)."""

    def test_quadratic_routing(self) -> None:
        """Q = R^2 / (R + 60)."""
        r2 = 100.0

        r_final, q = compute_streamflow(r2)

        expected_q = (r2 * r2) / (r2 + 60.0)
        assert np.isclose(q, expected_q)
        assert np.isclose(r_final, r2 - q)

    def test_zero_store_zero_flow(self) -> None:
        """Zero store produces zero streamflow."""
        r2 = 0.0

        r_final, q = compute_streamflow(r2)

        assert q == 0.0
        assert r_final == 0.0

    def test_streamflow_increases_with_store(self) -> None:
        """Higher store produces more streamflow."""
        _, q_low = compute_streamflow(20.0)
        _, q_high = compute_streamflow(200.0)

        assert q_high > q_low

    def test_water_balance(self) -> None:
        """R2 = R_final + Q."""
        test_cases = [10.0, 50.0, 100.0, 500.0]

        for r2 in test_cases:
            r_final, q = compute_streamflow(r2)
            assert np.isclose(r2, r_final + q), f"Failed for R2={r2}"

    def test_negative_store_handled(self) -> None:
        """Negative store is clamped to zero."""
        r2 = -10.0

        r_final, q = compute_streamflow(r2)

        assert r_final == 0.0
        assert q == 0.0

    def test_very_large_store(self) -> None:
        """Very large store produces finite results."""
        r2 = 100000.0

        r_final, q = compute_streamflow(r2)

        assert np.isfinite(q)
        assert np.isfinite(r_final)
        assert q > 0.0


class TestNumericalStability:
    """Tests for numerical stability of process functions."""

    def test_extreme_x1_values(self) -> None:
        """Extreme X1 values produce finite results."""
        precip = 50.0
        pet = 30.0
        store = 100.0

        for x1 in [1.0, 10.0, 1000.0, 2500.0]:
            s1, p1, ps = production_store_rainfall(precip, store, x1)
            s2, ae = production_store_evaporation(pet, s1, x1)
            s_final, p2 = percolation(s2, x1)

            assert np.isfinite(s1)
            assert np.isfinite(s2)
            assert np.isfinite(s_final)

    def test_extreme_x2_values(self) -> None:
        """Extreme X2 values produce finite results."""
        routing_store = 100.0
        p3 = 50.0

        for x2 in [0.2, 0.5, 1.0, 1.5, 2.0]:
            r2, aexch = routing_store_update(routing_store, p3, x2)
            r_final, q = compute_streamflow(r2)

            assert np.isfinite(r2)
            assert np.isfinite(q)

    def test_very_small_values(self) -> None:
        """Very small positive values produce finite results."""
        s1, p1, ps = production_store_rainfall(0.001, 0.001, 500.0)
        s2, ae = production_store_evaporation(0.001, 0.001, 500.0)
        s_final, p2 = percolation(0.001, 500.0)
        r2, aexch = routing_store_update(0.001, 0.001, 1.0)
        r_final, q = compute_streamflow(0.001)

        assert np.isfinite(s1)
        assert np.isfinite(s2)
        assert np.isfinite(s_final)
        assert np.isfinite(r2)
        assert np.isfinite(q)
