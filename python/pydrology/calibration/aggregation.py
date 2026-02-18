"""Daily-to-monthly aggregation utilities for cross-resolution calibration.

Provides pre-computed aggregation contexts and fast numpy-based aggregation
for comparing daily model output against monthly observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pydrology.types import Resolution


@dataclass(frozen=True)
class AggregationContext:
    """Pre-computed context for aggregating daily values to monthly groups.

    Attributes:
        valid_mask: Bool mask selecting days that belong to complete months.
        group_indices: Int array mapping each valid day to its month group index.
        n_per_group: Count of days in each complete month group.
        n_groups: Number of complete month groups.
    """

    valid_mask: np.ndarray  # bool, shape (n_days,)
    group_indices: np.ndarray  # int, shape (n_valid_days,)
    n_per_group: np.ndarray  # int, shape (n_groups,)
    n_groups: int


def build_aggregation_context(
    time: np.ndarray,
    target_resolution: Resolution,
) -> AggregationContext:
    """Build pre-computed aggregation context for daily-to-monthly mapping.

    Discards partial first/last months so only complete periods are used.

    Args:
        time: Daily datetime64 array (post-warmup portion of forcing.time).
        target_resolution: Target resolution (currently only monthly supported).

    Returns:
        AggregationContext with masks and indices for fast aggregation.

    Raises:
        ValueError: If target_resolution is not monthly or no complete months found.
    """
    if target_resolution != Resolution.monthly:
        msg = f"Only monthly target resolution is supported, got '{target_resolution.value}'"
        raise ValueError(msg)

    n_days = len(time)

    # Convert to month periods: e.g. 2020-01-15 -> 2020-01
    months = time.astype("datetime64[M]")

    # Get the day-of-month for each date (1-based)
    days_of_month = (time.astype("datetime64[D]") - months).astype(int) + 1

    # Find unique months
    unique_months = np.unique(months)

    # Check if first month is partial (first date is not the 1st)
    first_complete_idx = 0
    if days_of_month[0] != 1:
        first_complete_idx = 1

    # Check if last month is partial
    # Last day of last month: compare last date with last day of that month
    last_month = unique_months[-1]
    next_month = last_month + np.timedelta64(1, "M")
    last_day_of_last_month = (next_month.astype("datetime64[D]") - last_month.astype("datetime64[D]")).astype(int)
    # Find the last date in the last month
    last_month_mask = months == last_month
    last_date_day = days_of_month[last_month_mask][-1]

    last_complete_idx = len(unique_months)
    if last_date_day != last_day_of_last_month:
        last_complete_idx = len(unique_months) - 1

    complete_months = unique_months[first_complete_idx:last_complete_idx]

    if len(complete_months) == 0:
        msg = "No complete months found in the time array"
        raise ValueError(msg)

    # Build valid_mask: True for days belonging to complete months
    valid_mask = np.isin(months, complete_months)

    # Build group_indices: map each valid day to its group (0, 1, 2, ...)
    valid_months = months[valid_mask]
    # Use searchsorted to map month values to indices
    group_indices = np.searchsorted(complete_months, valid_months).astype(np.intp)

    # Count days per group
    n_per_group = np.bincount(group_indices, minlength=len(complete_months))

    return AggregationContext(
        valid_mask=valid_mask,
        group_indices=group_indices,
        n_per_group=n_per_group,
        n_groups=len(complete_months),
    )


def aggregate_array(
    values: np.ndarray,
    ctx: AggregationContext,
    method: Literal["sum", "mean"],
) -> np.ndarray:
    """Aggregate daily values to monthly using pre-computed context.

    Args:
        values: Daily values array (same length as the time array used to build ctx).
        ctx: Pre-computed aggregation context.
        method: Aggregation method — "sum" or "mean".

    Returns:
        Array of length ctx.n_groups with aggregated monthly values.
    """
    valid = values[ctx.valid_mask]
    sums = np.bincount(ctx.group_indices, weights=valid, minlength=ctx.n_groups).astype(np.float64)
    if method == "sum":
        return sums
    return sums / ctx.n_per_group
