"""GR6J core process functions.

Pure functions implementing the mathematical equations for each GR6J model component.
All inputs and outputs are floats. These functions correspond to sections in MODEL_DEFINITION.md.
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401
import numpy as np
from numba import njit

from .constants import (
    EXP_BRANCH_THRESHOLD,
    MAX_EXP_ARG,
    MAX_TANH_ARG,
    PERC_CONSTANT,
)


@njit(cache=True)
def production_store_update(
    precip: float, pet: float, production_store: float, x1: float
) -> tuple[float, float, float, float]:
    """Update the production store based on precipitation and evapotranspiration.

    Implements Section 5.1 of MODEL_DEFINITION.md. Handles two cases:
    - Case 1: P < E (evapotranspiration dominant)
    - Case 2: P >= E (rainfall dominant)

    Args:
        precip: Daily precipitation (mm/day).
        pet: Potential evapotranspiration (mm/day).
        production_store: Current production store level (mm).
        x1: Production store capacity parameter (mm).

    Returns:
        Tuple of (new_store, actual_et, net_rainfall_pn, effective_rainfall_pr):
        - new_store: Updated production store level (mm).
        - actual_et: Actual evapotranspiration (mm/day).
        - net_rainfall_pn: Net rainfall, 0 if P < E (mm/day).
        - effective_rainfall_pr: Effective rainfall contributing to runoff (mm/day).
    """
    store_ratio = production_store / x1

    if precip < pet:
        # Case 1: Evapotranspiration dominant (P < E)
        net_evap = pet - precip
        scaled_evap = min(net_evap / x1, MAX_TANH_ARG)
        tanh_ws = np.tanh(scaled_evap)

        # Evaporation from production store
        numerator = (2.0 - store_ratio) * tanh_ws
        denominator = 1.0 + (1.0 - store_ratio) * tanh_ws
        evap_from_store = production_store * numerator / denominator

        # Actual evapotranspiration = evaporation from store + precipitation
        actual_et = evap_from_store + precip

        # Update store
        new_store = production_store - evap_from_store

        # No effective rainfall when evapotranspiration dominant
        net_rainfall_pn = 0.0
        effective_rainfall_pr = 0.0
    else:
        # Case 2: Rainfall dominant (P >= E)
        net_rainfall_pn = precip - pet
        actual_et = pet

        scaled_precip = min(net_rainfall_pn / x1, MAX_TANH_ARG)
        tanh_ws = np.tanh(scaled_precip)

        # Storage infiltration
        numerator = (1.0 - store_ratio**2) * tanh_ws
        denominator = 1.0 + store_ratio * tanh_ws
        storage_infiltration = x1 * numerator / denominator

        # Rainfall excess (effective rainfall before percolation)
        effective_rainfall_pr = net_rainfall_pn - storage_infiltration

        # Update store
        new_store = production_store + storage_infiltration

    return new_store, actual_et, net_rainfall_pn, effective_rainfall_pr


@njit(cache=True)
def percolation(production_store: float, x1: float) -> tuple[float, float]:
    """Compute percolation from the production store.

    Implements Section 5.2 of MODEL_DEFINITION.md.
    Perc = S * (1 - (1 + (S / (9/4 * X1))^4)^(-0.25))

    Args:
        production_store: Current production store level (mm).
        x1: Production store capacity parameter (mm).

    Returns:
        Tuple of (new_store, percolation_amount):
        - new_store: Updated production store level after percolation (mm).
        - percolation_amount: Water percolating from store (mm/day).
    """
    # Ensure non-negative store
    store = max(production_store, 0.0)

    # Compute percolation using the standard GR6J formula
    # PERC_CONSTANT = (9/4)^4 = 25.62890625
    store_ratio_4 = (store / x1) ** 4
    percolation_amount = store * (1.0 - (1.0 + store_ratio_4 / PERC_CONSTANT) ** (-0.25))

    # Update store
    new_store = store - percolation_amount

    return new_store, percolation_amount


@njit(cache=True)
def groundwater_exchange(routing_store: float, x2: float, x3: float, x5: float) -> float:
    """Compute potential groundwater exchange.

    Implements Section 5.4 of MODEL_DEFINITION.md.
    F = X2 * (R/X3 - X5)

    Args:
        routing_store: Current routing store level (mm).
        x2: Intercatchment exchange coefficient (mm/day).
        x3: Routing store capacity (mm).
        x5: Intercatchment exchange threshold (dimensionless).

    Returns:
        Potential exchange F (mm/day). Positive = import, negative = export.
    """
    return x2 * (routing_store / x3 - x5)


@njit(cache=True)
def routing_store_update(
    routing_store: float, uh1_output: float, exchange: float, x3: float
) -> tuple[float, float, float]:
    """Update the routing store and compute outflow.

    Implements Section 5.5 of MODEL_DEFINITION.md.
    Receives (1-C) * uh1_output + exchange, applies non-negativity constraint,
    and computes non-linear outflow.

    Args:
        routing_store: Current routing store level (mm).
        uh1_output: Output from UH1 convolution for routing branch (mm/day).
            This should be (1-C) * StUH1(1), i.e., 60% of UH1 output.
        exchange: Potential groundwater exchange F (mm/day).
        x3: Routing store capacity (mm).

    Returns:
        Tuple of (new_store, outflow_qr, actual_exchange):
        - new_store: Updated routing store level after outflow (mm).
        - outflow_qr: Routing store outflow (mm/day).
        - actual_exchange: Actual exchange applied (may differ from potential
            if store would go negative) (mm/day).
    """
    # Inflow to routing store
    store_after_inflow = routing_store + uh1_output + exchange

    # Apply non-negativity constraint and track actual exchange
    if store_after_inflow >= 0.0:
        actual_exchange = exchange
        store = store_after_inflow
    else:
        # Store would go negative; limit exchange to prevent this
        actual_exchange = -(routing_store + uh1_output)
        store = 0.0

    # Compute non-linear outflow: QR = R * (1 - 1/(1 + (R/X3)^4)^0.25)
    if store > 0.0:
        store_ratio_4 = (store / x3) ** 4
        outflow_qr = store * (1.0 - (1.0 + store_ratio_4) ** (-0.25))
    else:
        outflow_qr = 0.0

    # Update store
    new_store = store - outflow_qr

    return new_store, outflow_qr, actual_exchange


@njit(cache=True)
def exponential_store_update(exp_store: float, uh1_output: float, exchange: float, x6: float) -> tuple[float, float]:
    """Update the exponential store and compute outflow.

    Implements Section 5.6 of MODEL_DEFINITION.md.
    Receives C * uh1_output + exchange. Note that exp_store can be negative.
    Uses softplus-like function with branch equations for numerical stability.

    Args:
        exp_store: Current exponential store level (mm). Can be negative.
        uh1_output: Output from UH1 convolution for exponential branch (mm/day).
            This should be C * StUH1(1), i.e., 40% of UH1 output.
        exchange: Potential groundwater exchange F (mm/day).
        x6: Exponential store scale parameter (mm).

    Returns:
        Tuple of (new_store, outflow_qrexp):
        - new_store: Updated exponential store level (mm). Can be negative.
        - outflow_qrexp: Exponential store outflow (mm/day).
    """
    # Inflow to exponential store (no non-negativity constraint)
    store = exp_store + uh1_output + exchange

    # Scaled store level with numerical safeguard
    # Using max/min instead of np.clip for numba compatibility
    ar = max(-MAX_EXP_ARG, min(store / x6, MAX_EXP_ARG))

    # Compute outflow using branch equations for numerical stability
    if ar > EXP_BRANCH_THRESHOLD:
        # Large positive AR: QRExp = Exp + X6/exp(AR)
        outflow_qrexp = store + x6 / np.exp(ar)
    elif ar < -EXP_BRANCH_THRESHOLD:
        # Large negative AR: QRExp = X6 * exp(AR)
        outflow_qrexp = x6 * np.exp(ar)
    else:
        # Normal range: softplus function QRExp = X6 * ln(exp(AR) + 1)
        outflow_qrexp = x6 * np.log(np.exp(ar) + 1.0)

    # Update store
    new_store = store - outflow_qrexp

    return new_store, outflow_qrexp


@njit(cache=True)
def direct_branch(uh2_output: float, exchange: float) -> tuple[float, float]:
    """Compute direct branch outflow.

    Implements Section 5.7 of MODEL_DEFINITION.md.
    Applies non-negativity constraint: QD = max(uh2_output + exchange, 0).

    Args:
        uh2_output: Output from UH2 convolution StUH2(1) (mm/day).
        exchange: Potential groundwater exchange F (mm/day).

    Returns:
        Tuple of (outflow_qd, actual_exchange):
        - outflow_qd: Direct branch outflow (mm/day).
        - actual_exchange: Actual exchange applied (may differ from potential
            if outflow would go negative) (mm/day).
    """
    combined = uh2_output + exchange

    if combined >= 0.0:
        outflow_qd = combined
        actual_exchange = exchange
    else:
        # Outflow would be negative; limit exchange
        outflow_qd = 0.0
        actual_exchange = -uh2_output

    return outflow_qd, actual_exchange
