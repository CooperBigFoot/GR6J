"""GR2M core process functions.

Pure functions implementing the mathematical equations for each GR2M model component.
All inputs and outputs are floats. These functions correspond to the airGR implementation.
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401
import numpy as np
from numba import njit

from .constants import MAX_TANH_ARG, ROUTING_DENOMINATOR


@njit(cache=True)
def production_store_rainfall(
    precip: float, store: float, x1: float
) -> tuple[float, float, float]:
    """Update production store for rainfall neutralization.

    Implements Section 5.1 of AIRGR_MODEL_DEFINITION.md.
    Rainfall is neutralized into the production store using a tanh function.

    Args:
        precip: Monthly precipitation (mm/month).
        store: Current production store level (mm).
        x1: Production store capacity parameter (mm).

    Returns:
        Tuple of (s1, p1, ps):
        - s1: Production store level after rainfall (mm).
        - p1: Rainfall excess / net precipitation (mm/month).
        - ps: Storage fill / infiltration (mm/month).
    """
    # Scaled precipitation with numerical safeguard
    ws = min(precip / x1, MAX_TANH_ARG)

    # Compute tanh using exponential form for Numba compatibility
    exp_2ws = np.exp(2.0 * ws)
    tws = (exp_2ws - 1.0) / (exp_2ws + 1.0)

    # Store ratio
    sr = store / x1

    # New store level after rainfall neutralization
    s1 = (store + x1 * tws) / (1.0 + sr * tws)

    # Rainfall excess (water not stored)
    p1 = precip + store - s1

    # Storage fill (water entering store)
    ps = precip - p1

    return s1, p1, ps


@njit(cache=True)
def production_store_evaporation(
    pet: float, s1: float, x1: float
) -> tuple[float, float]:
    """Extract evapotranspiration from production store.

    Implements Section 5.2 of AIRGR_MODEL_DEFINITION.md.
    Evapotranspiration is extracted from the store after rainfall neutralization.

    Args:
        pet: Monthly potential evapotranspiration (mm/month).
        s1: Production store level after rainfall (mm).
        x1: Production store capacity parameter (mm).

    Returns:
        Tuple of (s2, ae):
        - s2: Production store level after evaporation (mm).
        - ae: Actual evapotranspiration (mm/month).
    """
    # Scaled evapotranspiration with numerical safeguard
    ws = min(pet / x1, MAX_TANH_ARG)

    # Compute tanh using exponential form
    exp_2ws = np.exp(2.0 * ws)
    tws = (exp_2ws - 1.0) / (exp_2ws + 1.0)

    # Store ratio
    sr = s1 / x1

    # Store level after evaporation
    s2 = s1 * (1.0 - tws) / (1.0 + (1.0 - sr) * tws)

    # Actual evapotranspiration
    ae = s1 - s2

    return s2, ae


@njit(cache=True)
def percolation(s2: float, x1: float) -> tuple[float, float]:
    """Compute percolation from production store.

    Implements Section 5.3 of AIRGR_MODEL_DEFINITION.md.
    Uses a cube-root relationship for percolation.

    Args:
        s2: Production store level after evaporation (mm).
        x1: Production store capacity parameter (mm).

    Returns:
        Tuple of (s_final, p2):
        - s_final: Production store level after percolation (mm).
        - p2: Percolation amount (mm/month).
    """
    # Ensure non-negative store
    store = max(s2, 0.0)

    # Store ratio cubed
    sr = store / x1
    sr_cubed = sr * sr * sr

    # Percolation using cube root formula
    s_final = store / (1.0 + sr_cubed) ** (1.0 / 3.0)

    # Percolation amount
    p2 = store - s_final

    return s_final, p2


@njit(cache=True)
def routing_store_update(
    routing_store: float, p3: float, x2: float
) -> tuple[float, float]:
    """Update routing store with inflow and groundwater exchange.

    Implements Section 5.4 of AIRGR_MODEL_DEFINITION.md.
    The routing store receives water and applies groundwater exchange via X2.

    Args:
        routing_store: Current routing store level (mm).
        p3: Total water to routing (P1 + P2) (mm/month).
        x2: Groundwater exchange coefficient [-].

    Returns:
        Tuple of (r2, aexch):
        - r2: Routing store level after exchange (mm).
        - aexch: Actual groundwater exchange (mm/month).
    """
    # Routing store after inflow
    r1 = routing_store + p3

    # Apply groundwater exchange coefficient
    r2 = x2 * r1

    # Actual exchange (can be positive or negative)
    aexch = r2 - r1

    return r2, aexch


@njit(cache=True)
def compute_streamflow(r2: float) -> tuple[float, float]:
    """Compute streamflow from routing store.

    Implements Section 5.5 of AIRGR_MODEL_DEFINITION.md.
    Uses a quadratic reservoir routing equation.

    Args:
        r2: Routing store level after exchange (mm).

    Returns:
        Tuple of (r_final, q):
        - r_final: Routing store level after streamflow (mm).
        - q: Simulated streamflow (mm/month).
    """
    # Ensure non-negative store for routing
    store = max(r2, 0.0)

    # Quadratic routing: Q = R^2 / (R + 60)
    if store > 0.0:
        q = (store * store) / (store + ROUTING_DENOMINATOR)
    else:
        q = 0.0

    # Update routing store
    r_final = store - q

    return r_final, q
