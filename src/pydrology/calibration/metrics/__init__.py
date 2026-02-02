"""Metrics subpackage for calibration objectives.

Provides hydrological metrics (NSE, KGE, etc.) and a registry system
for managing metric functions and their optimization directions.
"""

# Import functions first to trigger @register decorators
from .functions import kge, log_nse, mae, nse, pbias, rmse
from .registry import (
    METRICS,
    MetricFunction,
    get_metric,
    list_metrics,
    register,
    validate_objectives,
)

__all__ = [
    "METRICS",
    "MetricFunction",
    "get_metric",
    "kge",
    "list_metrics",
    "log_nse",
    "mae",
    "nse",
    "pbias",
    "register",
    "rmse",
    "validate_objectives",
]
