"""Metrics registry for calibration objectives.

Provides a global registry mapping metric names to their implementations
and optimization directions.
"""

import logging
from collections.abc import Callable

from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Type alias for metric functions: (observed, simulated) -> score
type MetricFunction = Callable[[ArrayLike, ArrayLike], float]

# Global registry: {name: (function, direction)}
METRICS: dict[str, tuple[MetricFunction, str]] = {}


def register(direction: str) -> Callable[[MetricFunction], MetricFunction]:
    """Decorator to register a metric function.

    Args:
        direction: Either "maximize" or "minimize".

    Returns:
        Decorator that registers the function and returns it unchanged.

    Example:
        @register("maximize")
        def nse(observed, simulated):
            ...
    """
    if direction not in ("maximize", "minimize"):
        msg = f"direction must be 'maximize' or 'minimize', got '{direction}'"
        raise ValueError(msg)

    def decorator(func: MetricFunction) -> MetricFunction:
        METRICS[func.__name__] = (func, direction)
        return func

    return decorator


def get_metric(name: str) -> tuple[MetricFunction, str]:
    """Get a metric function and its direction by name.

    Args:
        name: The metric name (e.g., "nse", "kge").

    Returns:
        Tuple of (function, direction).

    Raises:
        KeyError: If metric name is not registered.
    """
    if name not in METRICS:
        available = ", ".join(sorted(METRICS.keys()))
        msg = f"Unknown metric '{name}'. Available: {available}"
        raise KeyError(msg)
    return METRICS[name]


def list_metrics() -> list[str]:
    """Return sorted list of registered metric names."""
    return sorted(METRICS.keys())


def validate_objectives(objectives: list[str] | dict[str, str]) -> dict[str, str]:
    """Validate and normalize objectives.

    Args:
        objectives: Metric names to optimize. Can be:
            - List of names (uses registered directions): ["nse", "log_nse"]
            - Dict with explicit directions (for overrides): {"nse": "minimize"}

    Returns:
        Normalized dict mapping metric names to directions.

    Raises:
        ValueError: If objectives is empty or contains unknown metrics.
    """
    if not objectives:
        msg = "objectives cannot be empty"
        raise ValueError(msg)

    # Normalize list to dict using registered directions
    if isinstance(objectives, list):
        result: dict[str, str] = {}
        for name in objectives:
            if name not in METRICS:
                available = ", ".join(sorted(METRICS.keys()))
                msg = f"Unknown metric '{name}'. Available: {available}"
                raise ValueError(msg)
            _, direction = METRICS[name]
            result[name] = direction
        return result

    # Dict form - validate and warn on direction mismatch
    for name, direction in objectives.items():
        if name not in METRICS:
            available = ", ".join(sorted(METRICS.keys()))
            msg = f"Unknown metric '{name}'. Available: {available}"
            raise ValueError(msg)

        _, registered_direction = METRICS[name]
        if direction != registered_direction:
            logger.warning(
                "Metric '%s' has registered direction '%s', but you specified '%s'. "
                "This is unusual - are you sure?",
                name,
                registered_direction,
                direction,
            )

    return objectives
