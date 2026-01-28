"""GR6J model orchestration functions.

This module provides the main entry points for running the GR6J model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

import numpy as np

from ..cemaneige import CemaNeigeSingleLayerState, cemaneige_step
from ..inputs import Catchment, ForcingData
from ..outputs import GR6JOutput, ModelOutput, SnowOutput
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
    forcing: ForcingData,
    catchment: Catchment | None = None,
    initial_state: State | None = None,
    initial_snow_state: CemaNeigeSingleLayerState | None = None,
) -> ModelOutput:
    """Run the GR6J model over a timeseries.

    Executes the GR6J model for each timestep in the input forcing data, returning
    a ModelOutput with all model outputs.

    Args:
        params: Model parameters (X1-X6), with optional snow module (params.snow).
        forcing: Input forcing data with precip, pet, and optionally temp arrays.
            When snow module is enabled (params.snow is set), temp is required.
        catchment: Catchment properties. Required when snow module is enabled
            for mean_annual_solid_precip initialization.
        initial_state: Initial model state. If None, uses State.initialize(params).
        initial_snow_state: Optional initial CemaNeige state. If None and
            snow module is enabled, uses CemaNeigeSingleLayerState.initialize().

    Returns:
        ModelOutput containing GR6J outputs and optionally snow outputs.
        Access streamflow via result.gr6j.streamflow (numpy array).
        Convert to DataFrame via result.to_dataframe().

    Raises:
        ValueError: If forcing.temp is None when snow module is enabled.
        ValueError: If catchment is None when snow module is enabled.

    Example:
        >>> params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5)
        >>> forcing = ForcingData(
        ...     time=np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64'),
        ...     precip=np.array([10.0, 5.0, 0.0]),
        ...     pet=np.array([3.0, 4.0, 5.0]),
        ... )
        >>> result = run(params, forcing)
        >>> result.gr6j.streamflow
        array([...])
    """
    # Validate snow module requirements
    if params.has_snow:
        if forcing.temp is None:
            raise ValueError("forcing.temp required when snow module enabled (params.snow is set)")
        if catchment is None:
            raise ValueError("catchment required when snow module enabled (params.snow is set)")

    # Initialize state if not provided
    state = State.initialize(params) if initial_state is None else initial_state

    # Initialize snow state if snow module enabled
    snow_state: CemaNeigeSingleLayerState | None = None
    if params.has_snow:
        snow_state = (
            CemaNeigeSingleLayerState.initialize(catchment.mean_annual_solid_precip)  # type: ignore  # validated above
            if initial_snow_state is None
            else initial_snow_state
        )

    # Compute unit hydrograph ordinates once
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    # Initialize output arrays
    n_timesteps = len(forcing)

    # GR6J outputs (20 fields)
    gr6j_outputs: dict[str, list[float]] = {
        "pet": [],
        "precip": [],
        "production_store": [],
        "net_rainfall": [],
        "storage_infiltration": [],
        "actual_et": [],
        "percolation": [],
        "effective_rainfall": [],
        "q9": [],
        "q1": [],
        "routing_store": [],
        "exchange": [],
        "actual_exchange_routing": [],
        "actual_exchange_direct": [],
        "actual_exchange_total": [],
        "qr": [],
        "qrexp": [],
        "exponential_store": [],
        "qd": [],
        "streamflow": [],
    }

    # Snow outputs (if enabled)
    snow_outputs: dict[str, list[float]] | None = None
    if params.has_snow:
        snow_outputs = {
            "precip_raw": [],
            "snow_pliq": [],
            "snow_psol": [],
            "snow_pack": [],
            "snow_thermal_state": [],
            "snow_gratio": [],
            "snow_pot_melt": [],
            "snow_melt": [],
            "snow_pliq_and_melt": [],
            "snow_temp": [],
            "snow_gthreshold": [],
            "snow_glocalmax": [],
        }

    # Run model for each timestep
    for idx in range(n_timesteps):
        precip = float(forcing.precip[idx])
        pet = float(forcing.pet[idx])

        # Store raw precip for output when snow enabled
        precip_raw = precip

        # Run snow module if enabled
        snow_fluxes: dict[str, float] = {}
        if params.has_snow and snow_state is not None:
            temp = float(forcing.temp[idx])  # type: ignore  # validated above
            snow_state, snow_fluxes = cemaneige_step(
                state=snow_state,
                params=params.snow,  # type: ignore  # validated via has_snow
                precip=precip,
                temp=temp,
            )
            # Use snow output as GR6J precipitation input
            precip = snow_fluxes["snow_pliq_and_melt"]

        state, fluxes = step(
            state=state,
            params=params,
            precip=precip,
            pet=pet,
            uh1_ordinates=uh1_ordinates,
            uh2_ordinates=uh2_ordinates,
        )

        # Append GR6J outputs
        for key, value in fluxes.items():
            gr6j_outputs[key].append(value)

        # Append snow outputs if enabled
        if params.has_snow and snow_outputs is not None:
            snow_outputs["precip_raw"].append(precip_raw)
            for key, value in snow_fluxes.items():
                snow_outputs[key].append(value)

    # Convert to numpy arrays and construct output objects
    gr6j_arrays = {k: np.array(v) for k, v in gr6j_outputs.items()}
    gr6j_output = GR6JOutput(**gr6j_arrays)

    snow_output: SnowOutput | None = None
    if snow_outputs is not None:
        snow_arrays = {k: np.array(v) for k, v in snow_outputs.items()}
        snow_output = SnowOutput(**snow_arrays)

    return ModelOutput(
        time=forcing.time,
        gr6j=gr6j_output,
        snow=snow_output,
    )
