"""GR6J model orchestration functions.

This module provides the main entry points for running the GR6J model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401

import numpy as np
from numba import njit

from pydrology.types import ForcingData
from pydrology.outputs import ModelOutput
from .constants import B, C, SUPPORTED_RESOLUTIONS
from .outputs import GR6JFluxes
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

# Constants inlined for Numba compatibility
_B: float = 0.9  # Fraction of PR to UH1 (slow branch)
_C: float = 0.4  # Fraction of UH1 output to exponential store


@njit(cache=True)
def _step_numba(
    state_arr: np.ndarray,  # shape (63,) - modified in place
    params_arr: np.ndarray,  # shape (6,)
    precip: float,
    pet: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    output_arr: np.ndarray,  # shape (20,) - output written here
) -> None:
    """Execute one timestep of GR6J using arrays (Numba-optimized).

    State layout: [production_store, routing_store, exponential_store, uh1_states[20], uh2_states[40]]
    Params layout: [x1, x2, x3, x4, x5, x6]
    Output layout: [pet, precip, production_store, net_rainfall, storage_infiltration,
                    actual_et, percolation, effective_rainfall, q9, q1, routing_store,
                    exchange, actual_exchange_routing, actual_exchange_direct,
                    actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow]
    """
    # Unpack parameters
    x1 = params_arr[0]
    x2 = params_arr[1]
    x3 = params_arr[2]
    # x4 is not used in step (only for UH computation, which is done once)
    x5 = params_arr[4]
    x6 = params_arr[5]

    # Unpack state
    production_store = state_arr[0]
    routing_store = state_arr[1]
    exponential_store = state_arr[2]
    # uh1_states is state_arr[3:23], uh2_states is state_arr[23:63]

    # 1. Production store update
    prod_store_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
        precip, pet, production_store, x1
    )

    # Compute storage infiltration for output
    storage_infiltration = net_rainfall_pn - effective_rainfall_pr if precip >= pet else 0.0

    # 2. Percolation
    prod_store_after_perc, percolation_amount = percolation(prod_store_after_ps, x1)

    # Add percolation to effective rainfall
    total_effective_rainfall = effective_rainfall_pr + percolation_amount

    # 3. Split to unit hydrographs
    uh1_input = _B * total_effective_rainfall
    uh2_input = (1.0 - _B) * total_effective_rainfall

    # 4. Convolve through unit hydrographs
    # UH1 states are at indices 3:23
    q9 = state_arr[3]  # Output is first element
    new_uh1 = np.zeros(20)
    for k in range(19):
        new_uh1[k] = state_arr[4 + k] + uh1_ordinates[k] * uh1_input
    new_uh1[19] = uh1_ordinates[19] * uh1_input

    # UH2 states are at indices 23:63
    q1 = state_arr[23]  # Output is first element
    new_uh2 = np.zeros(40)
    for k in range(39):
        new_uh2[k] = state_arr[24 + k] + uh2_ordinates[k] * uh2_input
    new_uh2[39] = uh2_ordinates[39] * uh2_input

    # 5. Groundwater exchange
    exchange_f = groundwater_exchange(routing_store, x2, x3, x5)

    # 6. Update routing store
    routing_input = (1.0 - _C) * q9
    new_routing_store, qr, actual_exchange_routing = routing_store_update(routing_store, routing_input, exchange_f, x3)

    # 7. Update exponential store
    exp_input = _C * q9
    new_exp_store, qrexp = exponential_store_update(exponential_store, exp_input, exchange_f, x6)

    # 8. Direct branch
    qd, actual_exchange_direct = direct_branch(q1, exchange_f)

    # 9. Total streamflow
    streamflow = max(qr + qrexp + qd, 0.0)

    # Total actual exchange
    actual_exchange_total = actual_exchange_routing + actual_exchange_direct + exchange_f

    # Update state array in place
    state_arr[0] = prod_store_after_perc
    state_arr[1] = new_routing_store
    state_arr[2] = new_exp_store
    for k in range(20):
        state_arr[3 + k] = new_uh1[k]
    for k in range(40):
        state_arr[23 + k] = new_uh2[k]

    # Write outputs
    output_arr[0] = pet
    output_arr[1] = precip
    output_arr[2] = prod_store_after_perc
    output_arr[3] = net_rainfall_pn
    output_arr[4] = storage_infiltration
    output_arr[5] = actual_et
    output_arr[6] = percolation_amount
    output_arr[7] = total_effective_rainfall
    output_arr[8] = q9
    output_arr[9] = q1
    output_arr[10] = new_routing_store
    output_arr[11] = exchange_f
    output_arr[12] = actual_exchange_routing
    output_arr[13] = actual_exchange_direct
    output_arr[14] = actual_exchange_total
    output_arr[15] = qr
    output_arr[16] = qrexp
    output_arr[17] = new_exp_store
    output_arr[18] = qd
    output_arr[19] = streamflow


@njit(cache=True)
def _run_numba(
    state_arr: np.ndarray,  # shape (63,)
    params_arr: np.ndarray,  # shape (6,)
    precip_arr: np.ndarray,  # shape (n_timesteps,)
    pet_arr: np.ndarray,  # shape (n_timesteps,)
    uh1_ordinates: np.ndarray,  # shape (20,)
    uh2_ordinates: np.ndarray,  # shape (40,)
    outputs_arr: np.ndarray,  # shape (n_timesteps, 20)
) -> None:
    """Run GR6J over a timeseries using arrays (Numba-optimized).

    State is modified in place. Outputs are written to outputs_arr.
    """
    n_timesteps = len(precip_arr)
    output_single = np.zeros(20)

    for t in range(n_timesteps):
        _step_numba(
            state_arr,
            params_arr,
            precip_arr[t],
            pet_arr[t],
            uh1_ordinates,
            uh2_ordinates,
            output_single,
        )
        for i in range(20):
            outputs_arr[t, i] = output_single[i]


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
    forcing: ForcingData,
    initial_state: State | None = None,
) -> ModelOutput[GR6JFluxes]:
    """Run the GR6J model over a timeseries.

    Executes the GR6J model for each timestep in the input forcing data, returning
    a ModelOutput with all model outputs.

    Args:
        params: Model parameters (X1-X6).
        forcing: Input forcing data with precip and pet arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).

    Returns:
        ModelOutput containing GR6J flux outputs.
        Access streamflow via result.streamflow or result.fluxes.streamflow (numpy array).
        Convert to DataFrame via result.to_dataframe().

    Example:
        >>> params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        >>> forcing = ForcingData(
        ...     time=np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64'),
        ...     precip=np.array([10.0, 5.0, 0.0]),
        ...     pet=np.array([3.0, 4.0, 5.0]),
        ... )
        >>> result = run(params, forcing)
        >>> result.streamflow
        array([...])
    """
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR6J supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    # Initialize state if not provided
    state = State.initialize(params) if initial_state is None else initial_state

    # Compute unit hydrograph ordinates once
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    # Initialize output arrays
    n_timesteps = len(forcing)

    # Convert to arrays
    state_arr = np.asarray(state)
    params_arr = np.asarray(params)

    # Allocate output array
    outputs_arr = np.zeros((n_timesteps, 20), dtype=np.float64)

    # Run the Numba kernel
    _run_numba(
        state_arr,
        params_arr,
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        uh1_ordinates,
        uh2_ordinates,
        outputs_arr,
    )

    # Build output object from array
    gr6j_fluxes = GR6JFluxes(
        pet=outputs_arr[:, 0],
        precip=outputs_arr[:, 1],
        production_store=outputs_arr[:, 2],
        net_rainfall=outputs_arr[:, 3],
        storage_infiltration=outputs_arr[:, 4],
        actual_et=outputs_arr[:, 5],
        percolation=outputs_arr[:, 6],
        effective_rainfall=outputs_arr[:, 7],
        q9=outputs_arr[:, 8],
        q1=outputs_arr[:, 9],
        routing_store=outputs_arr[:, 10],
        exchange=outputs_arr[:, 11],
        actual_exchange_routing=outputs_arr[:, 12],
        actual_exchange_direct=outputs_arr[:, 13],
        actual_exchange_total=outputs_arr[:, 14],
        qr=outputs_arr[:, 15],
        qrexp=outputs_arr[:, 16],
        exponential_store=outputs_arr[:, 17],
        qd=outputs_arr[:, 18],
        streamflow=outputs_arr[:, 19],
    )

    return ModelOutput(
        time=forcing.time,
        fluxes=gr6j_fluxes,
        snow=None,
        snow_layers=None,
    )
