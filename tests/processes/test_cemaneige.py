"""Tests for CemaNeige core process functions.

Tests verify the mathematical correctness of each process function
according to CEMANEIGE.md equations.
"""

import numpy as np
import pytest

from pydrology.cemaneige.processes import (
    compute_actual_melt,
    compute_gratio,
    compute_potential_melt,
    compute_solid_fraction,
    partition_precipitation,
    update_thermal_state,
)


class TestComputeSolidFraction:
    """Tests for compute_solid_fraction (Section 5.1 USACE formula)."""

    def test_all_snow_below_threshold(self) -> None:
        """At T <= -1C, all precipitation is snow."""
        assert compute_solid_fraction(-1.0) == 1.0
        assert compute_solid_fraction(-5.0) == 1.0
        assert compute_solid_fraction(-10.0) == 1.0

    def test_all_rain_above_threshold(self) -> None:
        """At T >= 3C, all precipitation is rain."""
        assert compute_solid_fraction(3.0) == 0.0
        assert compute_solid_fraction(10.0) == 0.0
        assert compute_solid_fraction(25.0) == 0.0

    def test_mixed_at_zero_celsius(self) -> None:
        """At 0C, fraction is 0.75 (interpolation)."""
        # (3 - 0) / 4 = 0.75
        assert compute_solid_fraction(0.0) == pytest.approx(0.75)

    def test_mixed_at_one_celsius(self) -> None:
        """At 1C, fraction is 0.5 (midpoint)."""
        # (3 - 1) / 4 = 0.5
        assert compute_solid_fraction(1.0) == pytest.approx(0.5)

    def test_linear_interpolation(self) -> None:
        """Verify linear behavior in transition zone."""
        # At 2C: (3 - 2) / 4 = 0.25
        assert compute_solid_fraction(2.0) == pytest.approx(0.25)

    def test_boundary_at_negative_one(self) -> None:
        """Test boundary at exactly -1C."""
        assert compute_solid_fraction(-1.0) == 1.0
        # Just above -1C should start interpolating
        assert compute_solid_fraction(-0.99) < 1.0
        assert compute_solid_fraction(-0.99) == pytest.approx((3.0 - (-0.99)) / 4.0)

    def test_boundary_at_three_celsius(self) -> None:
        """Test boundary at exactly 3C."""
        assert compute_solid_fraction(3.0) == 0.0
        # Just below 3C should still have some snow
        assert compute_solid_fraction(2.99) > 0.0
        assert compute_solid_fraction(2.99) == pytest.approx((3.0 - 2.99) / 4.0)

    def test_fraction_bounded_zero_one(self) -> None:
        """Solid fraction is always in [0, 1]."""
        for temp in [-20.0, -10.0, -1.0, 0.0, 1.0, 2.0, 3.0, 10.0, 30.0]:
            result = compute_solid_fraction(temp)
            assert 0.0 <= result <= 1.0


class TestPartitionPrecipitation:
    """Tests for partition_precipitation (Section 5.1)."""

    def test_all_snow_partition(self) -> None:
        """When solid_fraction=1, all precip becomes snow."""
        pliq, psol = partition_precipitation(10.0, 1.0)
        assert pliq == 0.0
        assert psol == 10.0

    def test_all_rain_partition(self) -> None:
        """When solid_fraction=0, all precip becomes rain."""
        pliq, psol = partition_precipitation(10.0, 0.0)
        assert pliq == 10.0
        assert psol == 0.0

    def test_mixed_partition(self) -> None:
        """Correct split with partial solid fraction."""
        pliq, psol = partition_precipitation(20.0, 0.5)
        assert pliq == 10.0
        assert psol == 10.0

    def test_zero_precipitation(self) -> None:
        """Zero precip gives zero for both components."""
        pliq, psol = partition_precipitation(0.0, 0.5)
        assert pliq == 0.0
        assert psol == 0.0

    def test_mass_conservation(self) -> None:
        """pliq + psol always equals original precip."""
        for precip in [0.0, 5.0, 10.0, 100.0]:
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                pliq, psol = partition_precipitation(precip, frac)
                assert np.isclose(pliq + psol, precip)

    def test_quarter_fraction(self) -> None:
        """Verify correct partition at 0.25 solid fraction."""
        pliq, psol = partition_precipitation(100.0, 0.25)
        assert pliq == pytest.approx(75.0)
        assert psol == pytest.approx(25.0)

    def test_three_quarter_fraction(self) -> None:
        """Verify correct partition at 0.75 solid fraction."""
        pliq, psol = partition_precipitation(100.0, 0.75)
        assert pliq == pytest.approx(25.0)
        assert psol == pytest.approx(75.0)

    def test_large_precipitation(self) -> None:
        """Handles large precipitation values correctly."""
        pliq, psol = partition_precipitation(500.0, 0.6)
        assert pliq == pytest.approx(200.0)
        assert psol == pytest.approx(300.0)


class TestUpdateThermalState:
    """Tests for update_thermal_state (Section 5.3)."""

    def test_exponential_smoothing_formula(self) -> None:
        """Verify exponential smoothing: new_etg = ctg*etg + (1-ctg)*temp."""
        # ctg=0.9, etg=-5.0, temp=-2.0
        # new_etg = 0.9*(-5) + 0.1*(-2) = -4.5 - 0.2 = -4.7
        result = update_thermal_state(etg=-5.0, temp=-2.0, ctg=0.9)
        assert result == pytest.approx(-4.7)

    def test_thermal_state_capped_at_zero(self) -> None:
        """Thermal state cannot exceed 0C (melting point)."""
        # Warm temperature should try to push etg positive
        # ctg=0.5, etg=0.0, temp=10.0
        # Without cap: 0.5*0 + 0.5*10 = 5.0, but must be capped at 0
        result = update_thermal_state(etg=0.0, temp=10.0, ctg=0.5)
        assert result == 0.0

    def test_stays_negative_with_cold_temp(self) -> None:
        """Thermal state stays negative with cold temperatures."""
        result = update_thermal_state(etg=-5.0, temp=-10.0, ctg=0.9)
        assert result < 0.0

    def test_ctg_close_to_one_high_inertia(self) -> None:
        """CTG near 1 means slow thermal response."""
        # With ctg=0.99, state changes very slowly
        result = update_thermal_state(etg=-10.0, temp=5.0, ctg=0.99)
        # Should still be very negative (0.99 * -10 + 0.01 * 5 = -9.85)
        assert result == pytest.approx(-9.85)

    def test_ctg_close_to_zero_low_inertia(self) -> None:
        """CTG near 0 means rapid thermal response."""
        # With ctg=0.1, state tracks temperature closely
        result = update_thermal_state(etg=-10.0, temp=-2.0, ctg=0.1)
        # 0.1 * -10 + 0.9 * -2 = -1 - 1.8 = -2.8
        assert result == pytest.approx(-2.8)

    def test_ctg_zero_instant_response(self) -> None:
        """CTG=0 means thermal state equals temperature immediately."""
        result = update_thermal_state(etg=-20.0, temp=-5.0, ctg=0.0)
        # 0.0 * -20 + 1.0 * -5 = -5.0
        assert result == pytest.approx(-5.0)

    def test_ctg_one_no_change(self) -> None:
        """CTG=1 means thermal state never changes."""
        result = update_thermal_state(etg=-15.0, temp=10.0, ctg=1.0)
        # 1.0 * -15 + 0.0 * 10 = -15.0
        assert result == pytest.approx(-15.0)

    def test_warming_trend_capped(self) -> None:
        """Warming from negative toward zero is capped at zero."""
        # Start at -0.1C with warm temp, should cap at 0
        # ctg=0.5, etg=-0.1, temp=5.0
        # Without cap: 0.5 * -0.1 + 0.5 * 5 = -0.05 + 2.5 = 2.45
        result = update_thermal_state(etg=-0.1, temp=5.0, ctg=0.5)
        assert result == 0.0

    def test_at_zero_with_positive_temp_stays_zero(self) -> None:
        """When at 0C with positive temp, stays at 0C."""
        result = update_thermal_state(etg=0.0, temp=15.0, ctg=0.7)
        assert result == 0.0

    def test_at_zero_with_negative_temp_goes_negative(self) -> None:
        """When at 0C with negative temp, goes negative."""
        # ctg=0.7, etg=0.0, temp=-10.0
        # 0.7 * 0 + 0.3 * -10 = -3.0
        result = update_thermal_state(etg=0.0, temp=-10.0, ctg=0.7)
        assert result == pytest.approx(-3.0)


class TestComputePotentialMelt:
    """Tests for compute_potential_melt (Section 5.4)."""

    def test_no_melt_when_etg_negative(self) -> None:
        """No melt when thermal state is below 0C."""
        result = compute_potential_melt(etg=-1.0, temp=5.0, kf=2.5, snow_pack=100.0)
        assert result == 0.0

    def test_no_melt_when_temp_at_or_below_zero(self) -> None:
        """No melt when temperature is at or below melting point."""
        assert compute_potential_melt(etg=0.0, temp=0.0, kf=2.5, snow_pack=100.0) == 0.0
        assert compute_potential_melt(etg=0.0, temp=-5.0, kf=2.5, snow_pack=100.0) == 0.0

    def test_melt_when_both_conditions_met(self) -> None:
        """Melt occurs when etg=0 AND temp>0."""
        # pot_melt = kf * temp = 2.5 * 5.0 = 12.5
        result = compute_potential_melt(etg=0.0, temp=5.0, kf=2.5, snow_pack=100.0)
        assert result == pytest.approx(12.5)

    def test_potential_melt_capped_at_snow_pack(self) -> None:
        """Potential melt cannot exceed available snow."""
        # Would calculate 2.5 * 10 = 25, but only 5mm available
        result = compute_potential_melt(etg=0.0, temp=10.0, kf=2.5, snow_pack=5.0)
        assert result == 5.0

    def test_no_melt_with_zero_snow(self) -> None:
        """No melt when snow pack is zero."""
        result = compute_potential_melt(etg=0.0, temp=5.0, kf=2.5, snow_pack=0.0)
        assert result == 0.0

    def test_degree_day_factor_scaling(self) -> None:
        """Higher kf produces more melt."""
        melt_low_kf = compute_potential_melt(etg=0.0, temp=5.0, kf=1.0, snow_pack=100.0)
        melt_high_kf = compute_potential_melt(etg=0.0, temp=5.0, kf=5.0, snow_pack=100.0)
        assert melt_high_kf > melt_low_kf
        assert melt_low_kf == pytest.approx(5.0)  # 1.0 * 5.0
        assert melt_high_kf == pytest.approx(25.0)  # 5.0 * 5.0

    def test_temperature_scaling(self) -> None:
        """Higher temperature produces more melt."""
        melt_cool = compute_potential_melt(etg=0.0, temp=2.0, kf=2.0, snow_pack=100.0)
        melt_warm = compute_potential_melt(etg=0.0, temp=10.0, kf=2.0, snow_pack=100.0)
        assert melt_warm > melt_cool
        assert melt_cool == pytest.approx(4.0)  # 2.0 * 2.0
        assert melt_warm == pytest.approx(20.0)  # 2.0 * 10.0

    def test_no_melt_when_etg_slightly_negative(self) -> None:
        """Even small negative etg prevents melt (strict condition)."""
        result = compute_potential_melt(etg=-0.001, temp=5.0, kf=2.5, snow_pack=100.0)
        assert result == 0.0

    def test_melt_equals_snow_pack_exactly(self) -> None:
        """When pot_melt equals snow pack, result is the snow pack."""
        # kf * temp = 2.0 * 5.0 = 10.0 = snow_pack
        result = compute_potential_melt(etg=0.0, temp=5.0, kf=2.0, snow_pack=10.0)
        assert result == pytest.approx(10.0)

    def test_just_above_melt_threshold(self) -> None:
        """Melt occurs just above T_MELT (0C)."""
        result = compute_potential_melt(etg=0.0, temp=0.01, kf=2.5, snow_pack=100.0)
        assert result == pytest.approx(0.025)  # 2.5 * 0.01

    def test_typical_spring_conditions(self) -> None:
        """Realistic spring melt scenario."""
        # Warm day (8C), kf=3.5 mm/C/day, deep snow pack
        result = compute_potential_melt(etg=0.0, temp=8.0, kf=3.5, snow_pack=500.0)
        assert result == pytest.approx(28.0)  # 3.5 * 8.0


class TestComputeGratio:
    """Tests for compute_gratio (Section 5.5)."""

    def test_full_coverage_when_g_exceeds_threshold(self) -> None:
        """Gratio = 1 when snow pack >= threshold."""
        assert compute_gratio(snow_pack=150.0, gthreshold=100.0) == 1.0
        assert compute_gratio(snow_pack=100.0, gthreshold=100.0) == 1.0

    def test_partial_coverage_below_threshold(self) -> None:
        """Gratio = g/gthreshold when below threshold."""
        assert compute_gratio(snow_pack=50.0, gthreshold=100.0) == pytest.approx(0.5)
        assert compute_gratio(snow_pack=25.0, gthreshold=100.0) == pytest.approx(0.25)

    def test_zero_snow_gives_zero_gratio(self) -> None:
        """No snow means zero coverage."""
        assert compute_gratio(snow_pack=0.0, gthreshold=100.0) == 0.0

    def test_handles_zero_threshold(self) -> None:
        """Zero threshold edge case returns 0."""
        assert compute_gratio(snow_pack=10.0, gthreshold=0.0) == 0.0

    def test_gratio_bounded_zero_one(self) -> None:
        """Gratio is always in [0, 1]."""
        for g in [0.0, 50.0, 100.0, 200.0]:
            result = compute_gratio(g, 100.0)
            assert 0.0 <= result <= 1.0

    def test_at_three_quarters_threshold(self) -> None:
        """Gratio = 0.75 when snow is 75% of threshold."""
        assert compute_gratio(snow_pack=75.0, gthreshold=100.0) == pytest.approx(0.75)

    def test_just_below_threshold(self) -> None:
        """Snow just below threshold gives gratio just below 1."""
        assert compute_gratio(snow_pack=99.9, gthreshold=100.0) == pytest.approx(0.999)

    def test_well_above_threshold(self) -> None:
        """Snow well above threshold still gives gratio = 1."""
        assert compute_gratio(snow_pack=1000.0, gthreshold=100.0) == 1.0

    def test_small_threshold_value(self) -> None:
        """Works correctly with small threshold values."""
        assert compute_gratio(snow_pack=5.0, gthreshold=10.0) == pytest.approx(0.5)
        assert compute_gratio(snow_pack=10.0, gthreshold=10.0) == 1.0

    def test_large_threshold_value(self) -> None:
        """Works correctly with large threshold values."""
        assert compute_gratio(snow_pack=250.0, gthreshold=500.0) == pytest.approx(0.5)
        assert compute_gratio(snow_pack=600.0, gthreshold=500.0) == 1.0

    def test_zero_snow_zero_threshold(self) -> None:
        """Edge case: both snow and threshold are zero."""
        assert compute_gratio(snow_pack=0.0, gthreshold=0.0) == 0.0


class TestComputeActualMelt:
    """Tests for compute_actual_melt (Section 5.6)."""

    def test_full_melt_at_gratio_one(self) -> None:
        """At full coverage (gratio=1), melt = pot_melt."""
        # melt = ((1-0.1)*1 + 0.1) * 10 = (0.9 + 0.1) * 10 = 10
        result = compute_actual_melt(potential_melt=10.0, gratio=1.0)
        assert result == pytest.approx(10.0)

    def test_minimum_melt_at_gratio_zero(self) -> None:
        """At zero coverage, melt = MIN_SPEED * pot_melt = 10%."""
        # melt = ((1-0.1)*0 + 0.1) * 10 = 0.1 * 10 = 1.0
        result = compute_actual_melt(potential_melt=10.0, gratio=0.0)
        assert result == pytest.approx(1.0)

    def test_partial_melt_at_half_coverage(self) -> None:
        """At 50% coverage, melt is between min and max."""
        # melt = ((1-0.1)*0.5 + 0.1) * 10 = (0.45 + 0.1) * 10 = 5.5
        result = compute_actual_melt(potential_melt=10.0, gratio=0.5)
        assert result == pytest.approx(5.5)

    def test_zero_potential_melt(self) -> None:
        """Zero potential melt gives zero actual melt."""
        result = compute_actual_melt(potential_melt=0.0, gratio=1.0)
        assert result == 0.0

    def test_melt_scales_linearly_with_potential(self) -> None:
        """Actual melt scales linearly with potential melt."""
        melt1 = compute_actual_melt(potential_melt=10.0, gratio=0.5)
        melt2 = compute_actual_melt(potential_melt=20.0, gratio=0.5)
        assert melt2 == pytest.approx(2 * melt1)

    def test_quarter_coverage(self) -> None:
        """At 25% coverage, verify formula."""
        # melt = ((1-0.1)*0.25 + 0.1) * 10 = (0.225 + 0.1) * 10 = 3.25
        result = compute_actual_melt(potential_melt=10.0, gratio=0.25)
        assert result == pytest.approx(3.25)

    def test_three_quarter_coverage(self) -> None:
        """At 75% coverage, verify formula."""
        # melt = ((1-0.1)*0.75 + 0.1) * 10 = (0.675 + 0.1) * 10 = 7.75
        result = compute_actual_melt(potential_melt=10.0, gratio=0.75)
        assert result == pytest.approx(7.75)

    def test_melt_always_positive_with_positive_potential(self) -> None:
        """Melt is always positive when potential melt is positive."""
        for gratio in [0.0, 0.1, 0.5, 0.9, 1.0]:
            result = compute_actual_melt(potential_melt=10.0, gratio=gratio)
            assert result > 0.0

    def test_min_speed_guarantees_nonzero_melt(self) -> None:
        """MIN_SPEED ensures melt never drops below 10% of potential."""
        result = compute_actual_melt(potential_melt=100.0, gratio=0.0)
        # With MIN_SPEED=0.1, minimum melt is 10% of potential
        assert result == pytest.approx(10.0)

    def test_melt_bounded_by_gratio_range(self) -> None:
        """Melt varies between MIN_SPEED*pot_melt and pot_melt."""
        potential = 50.0
        min_melt = compute_actual_melt(potential_melt=potential, gratio=0.0)
        max_melt = compute_actual_melt(potential_melt=potential, gratio=1.0)

        # MIN_SPEED = 0.1, so min is 10% of potential
        assert min_melt == pytest.approx(0.1 * potential)
        assert max_melt == pytest.approx(potential)

        # Any gratio should give melt in this range
        for gratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = compute_actual_melt(potential_melt=potential, gratio=gratio)
            assert min_melt <= result <= max_melt

    def test_large_potential_melt(self) -> None:
        """Handles large potential melt values correctly."""
        result = compute_actual_melt(potential_melt=500.0, gratio=0.8)
        # melt = ((1-0.1)*0.8 + 0.1) * 500 = (0.72 + 0.1) * 500 = 410.0
        assert result == pytest.approx(410.0)

    def test_formula_consistency(self) -> None:
        """Verify the exact formula: ((1-MIN_SPEED)*gratio + MIN_SPEED) * pot_melt."""
        pot_melt = 25.0
        gratio = 0.6
        min_speed = 0.1  # From constants

        expected = ((1.0 - min_speed) * gratio + min_speed) * pot_melt
        result = compute_actual_melt(potential_melt=pot_melt, gratio=gratio)
        assert result == pytest.approx(expected)
