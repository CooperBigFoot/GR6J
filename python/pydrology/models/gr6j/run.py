"""GR6J model orchestration functions.

This module provides the main entry points for running the GR6J model:
- step(): Execute a single timestep (Rust backend)
- run(): Execute the model over a timeseries (Rust backend)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import SUPPORTED_RESOLUTIONS
from .outputs import GR6JFluxes
from .types import Parameters, State

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import ForcingData


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR6J model.

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
        - fluxes: Dictionary containing all model outputs
    """
    import pydrology._core

    state_arr = np.asarray(state, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)

    new_state_arr, fluxes_dict = pydrology._core.gr6j.gr6j_step(
        state_arr,
        params_arr,
        precip,
        pet,
        np.ascontiguousarray(uh1_ordinates, dtype=np.float64),
        np.ascontiguousarray(uh2_ordinates, dtype=np.float64),
    )

    new_state = State.from_array(new_state_arr)

    # Convert numpy scalars to Python floats for consistency
    fluxes: dict[str, float] = {k: float(v) for k, v in fluxes_dict.items()}

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
) -> ModelOutput[GR6JFluxes]:
    """Run the GR6J model over a timeseries.

    Args:
        params: Model parameters (X1-X6).
        forcing: Input forcing data with precip and pet arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).

    Returns:
        ModelOutput containing GR6J flux outputs.
    """
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR6J supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    import pydrology._core
    from pydrology.outputs import ModelOutput

    params_arr = np.asarray(params, dtype=np.float64)

    initial_state_arr = None
    if initial_state is not None:
        initial_state_arr = np.asarray(initial_state, dtype=np.float64)

    result = pydrology._core.gr6j.gr6j_run(
        params_arr,
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        initial_state_arr,
    )

    gr6j_fluxes = GR6JFluxes(
        pet=result["pet"],
        precip=result["precip"],
        production_store=result["production_store"],
        net_rainfall=result["net_rainfall"],
        storage_infiltration=result["storage_infiltration"],
        actual_et=result["actual_et"],
        percolation=result["percolation"],
        effective_rainfall=result["effective_rainfall"],
        q9=result["q9"],
        q1=result["q1"],
        routing_store=result["routing_store"],
        exchange=result["exchange"],
        actual_exchange_routing=result["actual_exchange_routing"],
        actual_exchange_direct=result["actual_exchange_direct"],
        actual_exchange_total=result["actual_exchange_total"],
        qr=result["qr"],
        qrexp=result["qrexp"],
        exponential_store=result["exponential_store"],
        qd=result["qd"],
        streamflow=result["streamflow"],
    )

    return ModelOutput(
        time=forcing.time,
        fluxes=gr6j_fluxes,
        snow=None,
        snow_layers=None,
    )
