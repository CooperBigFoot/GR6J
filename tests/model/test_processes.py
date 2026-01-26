"""Tests for GR6J core process functions.

Tests verify the mathematical correctness of each process function
according to MODEL_DEFINITION.md equations.
"""

import numpy as np

from gr6j.model.processes import (
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)


class TestProductionStoreUpdate:
    """Tests for production_store_update function (Section 5.1)."""

    def test_rainfall_dominant_case(self) -> None:
        """When P > E, store increases and PR > 0."""
        precip = 20.0
        pet = 5.0
        production_store = 100.0
        x1 = 300.0

        new_store, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
            precip, pet, production_store, x1
        )

        # Store should increase (infiltration occurs)
        assert new_store > production_store
        # Actual ET equals PET when P >= E
        assert actual_et == pet
        # Net rainfall is P - E
        assert net_rainfall_pn == precip - pet
        # Effective rainfall should be positive (some rainfall runs off)
        assert effective_rainfall_pr > 0.0
        # PR should be less than Pn (some water infiltrates)
        assert effective_rainfall_pr < net_rainfall_pn

    def test_evapotranspiration_dominant_case(self) -> None:
        """When P < E, store decreases and PR = 0."""
        precip = 2.0
        pet = 10.0
        production_store = 150.0
        x1 = 300.0

        new_store, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
            precip, pet, production_store, x1
        )

        # Store should decrease (evaporation occurs)
        assert new_store < production_store
        # Net rainfall is zero when P < E
        assert net_rainfall_pn == 0.0
        # No effective rainfall when P < E
        assert effective_rainfall_pr == 0.0
        # Actual ET should be greater than P (includes evaporation from store)
        assert actual_et > precip

    def test_zero_precipitation(self) -> None:
        """When P = 0 and E > 0, water evaporates from store."""
        precip = 0.0
        pet = 8.0
        production_store = 200.0
        x1 = 300.0

        new_store, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
            precip, pet, production_store, x1
        )

        # Store should decrease
        assert new_store < production_store
        # No net rainfall or effective rainfall
        assert net_rainfall_pn == 0.0
        assert effective_rainfall_pr == 0.0
        # Actual ET equals evaporation from store (since P=0)
        evap_from_store = production_store - new_store
        assert np.isclose(actual_et, evap_from_store)

    def test_equal_precipitation_and_pet(self) -> None:
        """When P = E, this is boundary case (rainfall dominant branch)."""
        precip = 5.0
        pet = 5.0
        production_store = 100.0
        x1 = 300.0

        new_store, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
            precip, pet, production_store, x1
        )

        # P >= E triggers rainfall dominant case
        # Net rainfall is zero
        assert net_rainfall_pn == 0.0
        # Actual ET equals PET
        assert actual_et == pet
        # Store should remain unchanged (no net rainfall to add)
        assert np.isclose(new_store, production_store)
        # No effective rainfall
        assert effective_rainfall_pr == 0.0

    def test_actual_et_never_exceeds_available_water(self) -> None:
        """Actual ET should not exceed S + P (water balance)."""
        # Test with high ET demand and low store
        precip = 3.0
        pet = 50.0  # High ET demand
        production_store = 20.0  # Low store
        x1 = 300.0

        new_store, actual_et, _, _ = production_store_update(precip, pet, production_store, x1)

        # Available water is store + precip
        available_water = production_store + precip
        # Actual ET should not exceed available water
        assert actual_et <= available_water + 1e-10
        # Store should remain non-negative
        assert new_store >= -1e-10


class TestPercolation:
    """Tests for percolation function (Section 5.2)."""

    def test_percolation_increases_with_store_level(self) -> None:
        """Higher store level should produce more percolation."""
        x1 = 300.0

        # Low store level
        _, perc_low = percolation(50.0, x1)
        # High store level
        _, perc_high = percolation(200.0, x1)

        assert perc_high > perc_low

    def test_zero_percolation_at_zero_store(self) -> None:
        """When S = 0, percolation should be 0."""
        x1 = 300.0

        new_store, perc = percolation(0.0, x1)

        assert perc == 0.0
        assert new_store == 0.0

    def test_percolation_bounded_by_store(self) -> None:
        """Percolation should never exceed store level."""
        x1 = 300.0

        # Test various store levels
        for store in [10.0, 50.0, 100.0, 200.0, 300.0, 500.0]:
            new_store, perc = percolation(store, x1)

            # Percolation should not exceed store
            assert perc <= store + 1e-10
            # New store should be non-negative
            assert new_store >= -1e-10
            # Water balance: new_store + perc = original store
            assert np.isclose(new_store + perc, store)


class TestGroundwaterExchange:
    """Tests for groundwater_exchange function (Section 5.4)."""

    def test_positive_exchange_when_store_above_threshold(self) -> None:
        """When R/X3 > X5 with X2 > 0, exchange is positive (import)."""
        routing_store = 80.0
        x2 = 2.0  # Positive exchange coefficient
        x3 = 100.0
        x5 = 0.5  # Threshold: R/X3 = 0.8 > 0.5

        exchange = groundwater_exchange(routing_store, x2, x3, x5)

        # R/X3 = 0.8, X5 = 0.5, so R/X3 - X5 = 0.3
        # F = X2 * 0.3 = 2.0 * 0.3 = 0.6
        expected = x2 * (routing_store / x3 - x5)
        assert np.isclose(exchange, expected)
        assert exchange > 0.0

    def test_negative_exchange_when_store_below_threshold(self) -> None:
        """When R/X3 < X5 with X2 > 0, exchange is negative (export)."""
        routing_store = 30.0
        x2 = 2.0
        x3 = 100.0
        x5 = 0.5  # Threshold: R/X3 = 0.3 < 0.5

        exchange = groundwater_exchange(routing_store, x2, x3, x5)

        # R/X3 = 0.3, X5 = 0.5, so R/X3 - X5 = -0.2
        # F = X2 * (-0.2) = 2.0 * (-0.2) = -0.4
        expected = x2 * (routing_store / x3 - x5)
        assert np.isclose(exchange, expected)
        assert exchange < 0.0

    def test_zero_exchange_at_threshold(self) -> None:
        """When R/X3 = X5, exchange is zero."""
        x3 = 100.0
        x5 = 0.5
        routing_store = x5 * x3  # R/X3 = X5
        x2 = 2.0

        exchange = groundwater_exchange(routing_store, x2, x3, x5)

        assert np.isclose(exchange, 0.0)


class TestRoutingStoreUpdate:
    """Tests for routing_store_update function (Section 5.5)."""

    def test_outflow_increases_with_store(self) -> None:
        """Higher store level produces higher outflow."""
        x3 = 100.0

        # Low store + inflow
        _, qr_low, _ = routing_store_update(routing_store=20.0, uh1_output=10.0, exchange=0.0, x3=x3)
        # High store + same inflow
        _, qr_high, _ = routing_store_update(routing_store=80.0, uh1_output=10.0, exchange=0.0, x3=x3)

        assert qr_high > qr_low

    def test_store_cannot_go_negative(self) -> None:
        """Routing store should never become negative."""
        routing_store = 10.0
        uh1_output = 5.0
        exchange = -50.0  # Large negative exchange
        x3 = 100.0

        new_store, qr, actual_exchange = routing_store_update(routing_store, uh1_output, exchange, x3)

        # Store should be non-negative
        assert new_store >= 0.0
        # Actual exchange should be limited
        assert actual_exchange > exchange  # Less negative than potential

    def test_actual_exchange_limited_when_store_empties(self) -> None:
        """When store would go negative, actual exchange is limited."""
        routing_store = 10.0
        uh1_output = 5.0
        exchange = -100.0  # Would make store negative
        x3 = 100.0

        new_store, _, actual_exchange = routing_store_update(routing_store, uh1_output, exchange, x3)

        # Actual exchange should be -(routing_store + uh1_output)
        expected_exchange = -(routing_store + uh1_output)
        assert np.isclose(actual_exchange, expected_exchange)
        # Store should be zero
        assert np.isclose(new_store, 0.0)


class TestExponentialStoreUpdate:
    """Tests for exponential_store_update function (Section 5.6)."""

    def test_store_can_be_negative(self) -> None:
        """Exponential store is allowed to be negative."""
        exp_store = -5.0
        uh1_output = 2.0
        exchange = -3.0  # Net negative
        x6 = 5.0

        new_store, qrexp = exponential_store_update(exp_store, uh1_output, exchange, x6)

        # Store should still be negative
        # Input: -5 + 2 + (-3) = -6
        # Softplus output is small but positive for negative input
        assert new_store < 0.0
        # Outflow should still be positive (softplus property)
        assert qrexp >= 0.0

    def test_softplus_branch_normal_range(self) -> None:
        """Test softplus branch when AR is between -7 and 7."""
        exp_store = 0.0
        uh1_output = 10.0
        exchange = 0.0
        x6 = 5.0

        new_store, qrexp = exponential_store_update(exp_store, uh1_output, exchange, x6)

        # After inflow: store = 10.0
        # AR = 10/5 = 2.0 (in normal range)
        # QRExp = X6 * ln(exp(AR) + 1) = 5 * ln(exp(2) + 1)
        ar = 10.0 / x6
        expected_qrexp = x6 * np.log(np.exp(ar) + 1.0)
        assert np.isclose(qrexp, expected_qrexp)

    def test_high_ar_branch(self) -> None:
        """Test high AR branch (AR > 7)."""
        exp_store = 0.0
        uh1_output = 50.0  # Large input
        exchange = 0.0
        x6 = 5.0

        new_store, qrexp = exponential_store_update(exp_store, uh1_output, exchange, x6)

        # After inflow: store = 50.0
        # AR = 50/5 = 10.0 > 7, high AR branch
        # QRExp = Exp + X6/exp(AR)
        store_after_inflow = 50.0
        ar = store_after_inflow / x6
        expected_qrexp = store_after_inflow + x6 / np.exp(ar)
        assert np.isclose(qrexp, expected_qrexp)

    def test_low_ar_branch(self) -> None:
        """Test low AR branch (AR < -7)."""
        exp_store = -50.0  # Large negative store
        uh1_output = 2.0
        exchange = 0.0
        x6 = 5.0

        new_store, qrexp = exponential_store_update(exp_store, uh1_output, exchange, x6)

        # After inflow: store = -50 + 2 = -48
        # AR = -48/5 = -9.6 < -7, low AR branch
        # QRExp = X6 * exp(AR)
        store_after_inflow = -48.0
        ar = store_after_inflow / x6
        expected_qrexp = x6 * np.exp(ar)
        assert np.isclose(qrexp, expected_qrexp)
        # Outflow should be very small for large negative AR
        assert qrexp < 0.01


class TestDirectBranch:
    """Tests for direct_branch function (Section 5.7)."""

    def test_outflow_is_input_plus_exchange(self) -> None:
        """When sum is positive, outflow equals input + exchange."""
        uh2_output = 5.0
        exchange = 2.0

        qd, actual_exchange = direct_branch(uh2_output, exchange)

        # QD = UH2 + F when sum > 0
        assert np.isclose(qd, uh2_output + exchange)
        # Actual exchange equals potential exchange
        assert actual_exchange == exchange

    def test_outflow_clipped_to_zero(self) -> None:
        """When input + exchange < 0, outflow is clipped to zero."""
        uh2_output = 3.0
        exchange = -10.0  # Large negative exchange

        qd, actual_exchange = direct_branch(uh2_output, exchange)

        # Outflow should be zero
        assert qd == 0.0
        # Actual exchange should be limited
        assert actual_exchange > exchange

    def test_actual_exchange_limited_when_negative(self) -> None:
        """Actual exchange is limited to -uh2_output when outflow would be negative."""
        uh2_output = 5.0
        exchange = -20.0

        qd, actual_exchange = direct_branch(uh2_output, exchange)

        # Outflow clipped to zero
        assert qd == 0.0
        # Actual exchange is -uh2_output
        assert np.isclose(actual_exchange, -uh2_output)
