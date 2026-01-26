"""GR6J model orchestration functions.

This module provides the main entry points for running the GR6J model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

import numpy as np
import pandas as pd

from .constants import B, C
from .processes import (
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)
from .types import Parameters, State
from .unit_hydrographs import compute_uh_ordinates, convolve_uh


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR6J model.

    Implements the complete GR6J algorithm following Section 6 of MODEL_DEFINITION.md:
    1. Production store update (evapotranspiration and infiltration)
    2. Percolation from production store
    3. Split effective rainfall to unit hydrographs
    4. Convolve through UH1 and UH2
    5. Compute groundwater exchange
    6. Update routing store
    7. Update exponential store
    8. Compute direct branch outflow
    9. Sum total streamflow

    Args:
        state: Current model state (stores and UH states).
        params: Model parameters (X1-X6).
        precip: Daily precipitation (mm/day).
        pet: Daily potential evapotranspiration (mm/day).
        uh1_ordinates: Pre-computed UH1 ordinates from compute_uh_ordinates().
        uh2_ordinates: Pre-computed UH2 ordinates from compute_uh_ordinates().

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs (see Section 8 of
          MODEL_DEFINITION.md for descriptions)
    """
    # 1. Production store update
    prod_store_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
        precip, pet, state.production_store, params.x1
    )

    # Compute storage infiltration (PS) for output
    # PS = Pn - PR (before percolation) when P >= E, otherwise 0
    storage_infiltration = net_rainfall_pn - effective_rainfall_pr if precip >= pet else 0.0

    # 2. Percolation
    prod_store_after_perc, percolation_amount = percolation(prod_store_after_ps, params.x1)

    # Add percolation to effective rainfall
    total_effective_rainfall = effective_rainfall_pr + percolation_amount

    # 3. Split effective rainfall to unit hydrographs
    uh1_input = B * total_effective_rainfall  # 90% to UH1 (slow branch)
    uh2_input = (1.0 - B) * total_effective_rainfall  # 10% to UH2 (fast branch)

    # 4. Convolve through unit hydrographs
    # Note: convolve_uh returns the OUTPUT first (before updating states)
    new_uh1_states, q9 = convolve_uh(state.uh1_states, uh1_input, uh1_ordinates)
    new_uh2_states, q1 = convolve_uh(state.uh2_states, uh2_input, uh2_ordinates)

    # 5. Compute groundwater exchange
    exchange_f = groundwater_exchange(state.routing_store, params.x2, params.x3, params.x5)

    # 6. Update routing store
    # Receives (1-C) * q9 = 60% of UH1 output
    routing_input = (1.0 - C) * q9
    new_routing_store, qr, actual_exchange_routing = routing_store_update(
        state.routing_store, routing_input, exchange_f, params.x3
    )

    # 7. Update exponential store
    # Receives C * q9 = 40% of UH1 output
    exp_input = C * q9
    new_exp_store, qrexp = exponential_store_update(state.exponential_store, exp_input, exchange_f, params.x6)

    # 8. Direct branch
    # Receives q1 (UH2 output) + exchange
    qd, actual_exchange_direct = direct_branch(q1, exchange_f)

    # 9. Total streamflow (with non-negativity)
    streamflow = max(qr + qrexp + qd, 0.0)

    # Compute total actual exchange
    # Note: From the Fortran MISC(15) = AExch1 + AExch2 + Exch
    # This represents the exchange applied to exponential store (which has no constraint)
    # plus the actual exchanges from routing and direct branches
    actual_exchange_total = actual_exchange_routing + actual_exchange_direct + exchange_f

    # Build new state
    new_state = State(
        production_store=prod_store_after_perc,
        routing_store=new_routing_store,
        exponential_store=new_exp_store,
        uh1_states=new_uh1_states,
        uh2_states=new_uh2_states,
    )

    # Build fluxes dictionary (matching Section 8 MISC outputs)
    fluxes: dict[str, float] = {
        "pet": pet,
        "precip": precip,
        "production_store": prod_store_after_perc,
        "net_rainfall": net_rainfall_pn,
        "storage_infiltration": storage_infiltration,
        "actual_et": actual_et,
        "percolation": percolation_amount,
        "effective_rainfall": total_effective_rainfall,
        "q9": q9,
        "q1": q1,
        "routing_store": new_routing_store,
        "exchange": exchange_f,
        "actual_exchange_routing": actual_exchange_routing,
        "actual_exchange_direct": actual_exchange_direct,
        "actual_exchange_total": actual_exchange_total,
        "qr": qr,
        "qrexp": qrexp,
        "exponential_store": new_exp_store,
        "qd": qd,
        "streamflow": streamflow,
    }

    return new_state, fluxes


def run(
    params: Parameters,
    data: pd.DataFrame,
    initial_state: State | None = None,
) -> pd.DataFrame:
    """Run the GR6J model over a timeseries.

    Executes the GR6J model for each timestep in the input data, returning
    a DataFrame with all model outputs.

    Args:
        params: Model parameters (X1-X6).
        data: Input DataFrame with 'precip' and 'pet' columns.
            Must have these columns with precipitation and potential
            evapotranspiration values in mm/day.
        initial_state: Initial model state. If None, uses State.initialize(params)
            which sets production store to 30% of X1, routing store to 50% of X3,
            and exponential store to 0.

    Returns:
        DataFrame with all model fluxes (same index as input data).
        Columns include: pet, precip, production_store, net_rainfall,
        storage_infiltration, actual_et, percolation, effective_rainfall,
        q9, q1, routing_store, exchange, actual_exchange_routing,
        actual_exchange_direct, actual_exchange_total, qr, qrexp,
        exponential_store, qd, streamflow.

    Raises:
        ValueError: If input data is missing required columns.

    Example:
        >>> params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        >>> data = pd.DataFrame({
        ...     'precip': [10.0, 5.0, 0.0],
        ...     'pet': [3.0, 4.0, 5.0]
        ... })
        >>> results = run(params, data)
        >>> results['streamflow']
        0    ...
        1    ...
        2    ...
        Name: streamflow, dtype: float64
    """
    # Validate input data
    required_columns = {"precip", "pet"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    # Initialize state if not provided
    state = State.initialize(params) if initial_state is None else initial_state

    # Compute unit hydrograph ordinates once
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    # Run model for each timestep
    all_fluxes: list[dict[str, float]] = []

    for idx in range(len(data)):
        precip = float(data["precip"].iloc[idx])
        pet = float(data["pet"].iloc[idx])

        state, fluxes = step(
            state=state,
            params=params,
            precip=precip,
            pet=pet,
            uh1_ordinates=uh1_ordinates,
            uh2_ordinates=uh2_ordinates,
        )

        all_fluxes.append(fluxes)

    # Convert to DataFrame with same index as input
    result = pd.DataFrame(all_fluxes, index=data.index)

    return result
