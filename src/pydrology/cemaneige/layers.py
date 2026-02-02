"""CemaNeige layer utility functions for multi-layer elevation bands.

Functions to derive layer properties from hypsometric curves and extrapolate
temperature/precipitation across elevation bands.
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import math

import pydrology._compat  # noqa: F401
import numpy as np
from numba import njit

from .constants import ELEV_CAP_PRECIP, GRAD_P_DEFAULT, GRAD_T_DEFAULT


def derive_layers(hypsometric_curve: np.ndarray, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Derive layer representative elevations and fractions from a hypsometric curve.

    Each layer represents an equal fraction of the catchment area (1/n_layers).
    The representative elevation for each layer is the midpoint of the elevation
    range spanned by that layer's percentile bounds.

    Args:
        hypsometric_curve: 101-point array of elevations at percentiles 0-100%
            (index i = elevation at percentile i%) [m].
        n_layers: Number of elevation bands to create.

    Returns:
        Tuple of (layer_elevations, layer_fractions):
        - layer_elevations: 1D array of representative elevation for each layer [m].
        - layer_fractions: 1D array of area fractions for each layer [-].
            Will be uniform: 1/n_layers.
    """
    layer_elevations = np.empty(n_layers)
    layer_fractions = np.full(n_layers, 1.0 / n_layers)

    percentiles = np.linspace(0, 100, 101)

    for i in range(n_layers):
        lower_percentile = i * 100.0 / n_layers
        upper_percentile = (i + 1) * 100.0 / n_layers

        lower_elev = np.interp(lower_percentile, percentiles, hypsometric_curve)
        upper_elev = np.interp(upper_percentile, percentiles, hypsometric_curve)

        layer_elevations[i] = (lower_elev + upper_elev) / 2.0

    return layer_elevations, layer_fractions


@njit(cache=True)
def extrapolate_temperature(
    input_temp: float,
    input_elevation: float,
    layer_elevation: float,
    gradient: float = GRAD_T_DEFAULT,
) -> float:
    """Extrapolate temperature to a different elevation.

    Uses a linear lapse rate to adjust temperature based on elevation difference.
    A positive gradient means temperature decreases with elevation.

    Args:
        input_temp: Temperature at input elevation [°C].
        input_elevation: Elevation of input measurement [m].
        layer_elevation: Target elevation for extrapolation [m].
        gradient: Temperature lapse rate [°C/100m]. Default is GRAD_T_DEFAULT.

    Returns:
        Extrapolated temperature at target elevation [°C].
    """
    return input_temp - gradient * (layer_elevation - input_elevation) / 100.0


@njit(cache=True)
def extrapolate_precipitation(
    input_precip: float,
    input_elevation: float,
    layer_elevation: float,
    gradient: float = GRAD_P_DEFAULT,
    elev_cap: float = ELEV_CAP_PRECIP,
) -> float:
    """Extrapolate precipitation to a different elevation using exponential gradient.

    Uses an exponential relationship to adjust precipitation based on elevation
    difference. Both elevations are capped at elev_cap before applying the formula
    to prevent unrealistic extrapolation at extreme elevations.

    Args:
        input_precip: Precipitation at input elevation [mm/day].
        input_elevation: Elevation of input measurement [m].
        layer_elevation: Target elevation for extrapolation [m].
        gradient: Precipitation gradient [m⁻¹]. Default is GRAD_P_DEFAULT.
        elev_cap: Maximum elevation for precipitation extrapolation [m].
            Default is ELEV_CAP_PRECIP.

    Returns:
        Extrapolated precipitation at target elevation [mm/day].
    """
    effective_input_elev = min(input_elevation, elev_cap)
    effective_layer_elev = min(layer_elevation, elev_cap)

    return input_precip * math.exp(gradient * (effective_layer_elev - effective_input_elev))
