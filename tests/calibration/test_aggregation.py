"""Tests for daily-to-monthly aggregation utilities."""

import numpy as np
import pytest
from pydrology.calibration.aggregation import (
    AggregationContext,
    aggregate_array,
    build_aggregation_context,
)
from pydrology.types import Resolution


class TestBuildAggregationContext:
    """Tests for build_aggregation_context."""

    def test_complete_months(self) -> None:
        """Three full months should produce 3 groups."""
        # Jan 2020 (31d) + Feb 2020 (29d, leap) + Mar 2020 (31d) = 91 days
        time = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-04-01"),
            dtype="datetime64[D]",
        )
        ctx = build_aggregation_context(time, Resolution.monthly)
        assert ctx.n_groups == 3
        assert ctx.valid_mask.sum() == 91
        np.testing.assert_array_equal(ctx.n_per_group, [31, 29, 31])

    def test_partial_first_month_discarded(self) -> None:
        """Starting mid-January should discard January."""
        # Jan 15 -> Mar 31 = partial Jan + full Feb + full Mar
        time = np.arange(
            np.datetime64("2020-01-15"),
            np.datetime64("2020-04-01"),
            dtype="datetime64[D]",
        )
        ctx = build_aggregation_context(time, Resolution.monthly)
        assert ctx.n_groups == 2  # Feb + Mar only
        assert ctx.n_per_group[0] == 29  # Feb 2020 (leap year)
        assert ctx.n_per_group[1] == 31  # Mar 2020

    def test_partial_last_month_discarded(self) -> None:
        """Ending mid-March should discard March."""
        # Jan 1 -> Mar 15 = full Jan + full Feb + partial Mar
        time = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-03-16"),
            dtype="datetime64[D]",
        )
        ctx = build_aggregation_context(time, Resolution.monthly)
        assert ctx.n_groups == 2  # Jan + Feb only
        assert ctx.n_per_group[0] == 31  # Jan
        assert ctx.n_per_group[1] == 29  # Feb 2020

    def test_no_complete_months_raises(self) -> None:
        """Only partial month data should raise ValueError."""
        time = np.arange(
            np.datetime64("2020-01-10"),
            np.datetime64("2020-01-20"),
            dtype="datetime64[D]",
        )
        with pytest.raises(ValueError, match="No complete months"):
            build_aggregation_context(time, Resolution.monthly)

    def test_unsupported_resolution_raises(self) -> None:
        """Non-monthly target resolution should raise ValueError."""
        time = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-04-01"),
            dtype="datetime64[D]",
        )
        with pytest.raises(ValueError, match="Only monthly"):
            build_aggregation_context(time, Resolution.annual)


class TestAggregateArray:
    """Tests for aggregate_array."""

    @pytest.fixture
    def three_month_context(self) -> AggregationContext:
        """Context for Jan-Mar 2020 (91 complete days)."""
        time = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-04-01"),
            dtype="datetime64[D]",
        )
        return build_aggregation_context(time, Resolution.monthly)

    def test_aggregate_sum(self, three_month_context: AggregationContext) -> None:
        """Sum aggregation should produce monthly totals."""
        ctx = three_month_context
        # All ones: sum should equal days per month
        values = np.ones(91)
        result = aggregate_array(values, ctx, "sum")
        np.testing.assert_array_equal(result, [31.0, 29.0, 31.0])

    def test_aggregate_mean(self, three_month_context: AggregationContext) -> None:
        """Mean aggregation of constant array should return that constant."""
        ctx = three_month_context
        values = np.full(91, 5.0)
        result = aggregate_array(values, ctx, "mean")
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0])

    def test_aggregate_sum_known_values(self, three_month_context: AggregationContext) -> None:
        """Sum aggregation with known per-day values."""
        ctx = three_month_context
        # Jan: 1*31=31, Feb: 2*29=58, Mar: 3*31=93
        values = np.concatenate(
            [
                np.full(31, 1.0),
                np.full(29, 2.0),
                np.full(31, 3.0),
            ]
        )
        result = aggregate_array(values, ctx, "sum")
        np.testing.assert_allclose(result, [31.0, 58.0, 93.0])

    def test_aggregate_mean_known_values(self, three_month_context: AggregationContext) -> None:
        """Mean aggregation with known per-day values."""
        ctx = three_month_context
        values = np.concatenate(
            [
                np.full(31, 1.0),
                np.full(29, 2.0),
                np.full(31, 3.0),
            ]
        )
        result = aggregate_array(values, ctx, "mean")
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
