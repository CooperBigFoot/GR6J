"""GR2M model orchestration functions.

This module provides the main entry points for running the GR2M model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401

import numpy as np
from numba import njit

from pydrology.outputs import ModelOutput
from pydrology.types import ForcingData

from .constants import SUPPORTED_RESOLUTIONS
from .outputs import GR2MFluxes
from .processes import (
    compute_streamflow,
    percolation,
    production_store_evaporation,
    production_store_rainfall,
    routing_store_update,
)
from .types import Parameters, State


@njit(cache=True)
def _step_numba(
    state_arr: np.ndarray,  # shape (2,) - modified in place
    params_arr: np.ndarray,  # shape (2,)
    precip: float,
    pet: float,
    output_arr: np.ndarray,  # shape (11,) - output written here
) -> None:
    """Execute one timestep of GR2M using arrays (Numba-optimized).

    State layout: [production_store, routing_store]
    Params layout: [x1, x2]
    Output layout: [pet, precip, prod_store, p1, ps, ae, p2, p3, rout_store, exch, q]
    """
    # Unpack parameters
    x1 = params_arr[0]
    x2 = params_arr[1]

    # Unpack state
    production_store = state_arr[0]
    routing_store = state_arr[1]

    # 1. Production store - rainfall neutralization
    s1, p1, ps = production_store_rainfall(precip, production_store, x1)

    # 2. Production store - evaporation
    s2, ae = production_store_evaporation(pet, s1, x1)

    # 3. Percolation
    s_final, p2 = percolation(s2, x1)

    # 4. Total water to routing
    p3 = p1 + p2

    # 5. Routing store update with exchange
    r2, aexch = routing_store_update(routing_store, p3, x2)

    # 6. Streamflow
    r_final, q = compute_streamflow(r2)

    # Update state array in place
    state_arr[0] = s_final
    state_arr[1] = r_final

    # Write outputs (11 elements matching MISC array)
    output_arr[0] = pet
    output_arr[1] = precip
    output_arr[2] = s_final  # Production store after timestep
    output_arr[3] = p1       # Rainfall excess
    output_arr[4] = ps       # Storage fill
    output_arr[5] = ae       # Actual ET
    output_arr[6] = p2       # Percolation
    output_arr[7] = p3       # Routing input
    output_arr[8] = r_final  # Routing store after timestep
    output_arr[9] = aexch    # Exchange
    output_arr[10] = q       # Streamflow


@njit(cache=True)
def _run_numba(
    state_arr: np.ndarray,  # shape (2,)
    params_arr: np.ndarray,  # shape (2,)
    precip_arr: np.ndarray,  # shape (n_timesteps,)
    pet_arr: np.ndarray,  # shape (n_timesteps,)
    outputs_arr: np.ndarray,  # shape (n_timesteps, 11)
) -> None:
    """Run GR2M over a timeseries using arrays (Numba-optimized).

    State is modified in place. Outputs are written to outputs_arr.
    """
    n_timesteps = len(precip_arr)
    output_single = np.zeros(11)

    for t in range(n_timesteps):
        _step_numba(
            state_arr,
            params_arr,
            precip_arr[t],
            pet_arr[t],
            output_single,
        )
        for i in range(11):
            outputs_arr[t, i] = output_single[i]


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR2M model.

    Implements the complete GR2M algorithm:
    1. Production store update (rainfall neutralization)
    2. Evapotranspiration extraction
    3. Percolation from production store
    4. Route water through routing store with exchange
    5. Compute streamflow

    Args:
        state: Current model state (stores).
        params: Model parameters (X1, X2).
        precip: Monthly precipitation (mm/month).
        pet: Monthly potential evapotranspiration (mm/month).

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs
    """
    # Convert to arrays
    state_arr = np.asarray(state).copy()
    params_arr = np.asarray(params)
    output_arr = np.zeros(11)

    # Run the Numba kernel
    _step_numba(state_arr, params_arr, precip, pet, output_arr)

    # Build new state
    new_state = State.from_array(state_arr)

    # Build fluxes dictionary
    fluxes: dict[str, float] = {
        "pet": output_arr[0],
        "precip": output_arr[1],
        "production_store": output_arr[2],
        "rainfall_excess": output_arr[3],
        "storage_fill": output_arr[4],
        "actual_et": output_arr[5],
        "percolation": output_arr[6],
        "routing_input": output_arr[7],
        "routing_store": output_arr[8],
        "exchange": output_arr[9],
        "streamflow": output_arr[10],
    }

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
) -> ModelOutput[GR2MFluxes]:
    """Run the GR2M model over a timeseries.

    Executes the GR2M model for each timestep in the input forcing data, returning
    a ModelOutput with all model outputs.

    Args:
        params: Model parameters (X1, X2).
        forcing: Input forcing data with precip and pet arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).

    Returns:
        ModelOutput containing GR2M flux outputs.
        Access streamflow via result.streamflow or result.fluxes.streamflow (numpy array).
        Convert to DataFrame via result.to_dataframe().

    Raises:
        ValueError: If forcing resolution is not monthly.
    """
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR2M supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    # Initialize state if not provided
    state = State.initialize(params) if initial_state is None else initial_state

    # Initialize output arrays
    n_timesteps = len(forcing)

    # Convert to arrays
    state_arr = np.asarray(state)
    params_arr = np.asarray(params)

    # Allocate output array
    outputs_arr = np.zeros((n_timesteps, 11), dtype=np.float64)

    # Run the Numba kernel
    _run_numba(
        state_arr,
        params_arr,
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        outputs_arr,
    )

    # Build output object from array
    gr2m_fluxes = GR2MFluxes(
        pet=outputs_arr[:, 0],
        precip=outputs_arr[:, 1],
        production_store=outputs_arr[:, 2],
        rainfall_excess=outputs_arr[:, 3],
        storage_fill=outputs_arr[:, 4],
        actual_et=outputs_arr[:, 5],
        percolation=outputs_arr[:, 6],
        routing_input=outputs_arr[:, 7],
        routing_store=outputs_arr[:, 8],
        exchange=outputs_arr[:, 9],
        streamflow=outputs_arr[:, 10],
    )

    return ModelOutput(
        time=forcing.time,
        fluxes=gr2m_fluxes,
        snow=None,
        snow_layers=None,
    )
