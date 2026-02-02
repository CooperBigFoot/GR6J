"""GR6J model orchestration functions.

This module provides the main entry points for running the GR6J model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import gr6j._compat  # noqa: F401

import numpy as np
from numba import njit

from ..cemaneige import CemaNeigeMultiLayerState, CemaNeigeSingleLayerState, cemaneige_multi_layer_step, cemaneige_step
from ..cemaneige.run import _cemaneige_multi_layer_step_numba, _cemaneige_step_numba
from ..cemaneige.layers import derive_layers
from ..inputs import Catchment, ForcingData
from ..outputs import GR6JOutput, ModelOutput, SnowLayerOutputs, SnowOutput
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


@njit(cache=True)
def _run_with_snow_numba(
    state_arr: np.ndarray,  # GR6J state shape (63,)
    params_arr: np.ndarray,  # GR6J params shape (6,)
    snow_state_arr: np.ndarray,  # CemaNeige state shape (4,) for single layer
    ctg: float,
    kf: float,
    precip_arr: np.ndarray,  # shape (n_timesteps,)
    pet_arr: np.ndarray,  # shape (n_timesteps,)
    temp_arr: np.ndarray,  # shape (n_timesteps,)
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    gr6j_outputs_arr: np.ndarray,  # shape (n_timesteps, 20)
    snow_outputs_arr: np.ndarray,  # shape (n_timesteps, 12) - includes precip_raw
) -> None:
    """Run coupled CemaNeige-GR6J over a timeseries (single layer snow).

    Snow output layout: [precip_raw, pliq, psol, snow_pack, thermal_state, gratio,
                         pot_melt, melt, pliq_and_melt, temp, gthreshold, glocalmax]
    """
    n_timesteps = len(precip_arr)
    gr6j_output_single = np.zeros(20)
    snow_out_state = np.zeros(4)
    snow_out_fluxes = np.zeros(11)

    for t in range(n_timesteps):
        precip_raw = precip_arr[t]

        # Run CemaNeige
        _cemaneige_step_numba(
            snow_state_arr,
            ctg,
            kf,
            precip_raw,
            temp_arr[t],
            snow_out_state,
            snow_out_fluxes,
        )

        # Update snow state
        for j in range(4):
            snow_state_arr[j] = snow_out_state[j]

        # Use snow output (pliq_and_melt) as GR6J precip input
        precip_for_gr6j = snow_out_fluxes[7]  # pliq_and_melt

        # Run GR6J
        _step_numba(
            state_arr,
            params_arr,
            precip_for_gr6j,
            pet_arr[t],
            uh1_ordinates,
            uh2_ordinates,
            gr6j_output_single,
        )

        # Copy GR6J outputs
        for i in range(20):
            gr6j_outputs_arr[t, i] = gr6j_output_single[i]

        # Copy snow outputs with precip_raw as first element
        snow_outputs_arr[t, 0] = precip_raw
        for i in range(11):
            snow_outputs_arr[t, i + 1] = snow_out_fluxes[i]


@njit(cache=True)
def _run_with_multi_layer_snow_numba(
    state_arr: np.ndarray,  # GR6J state shape (63,)
    params_arr: np.ndarray,  # GR6J params shape (6,)
    layer_states_arr: np.ndarray,  # CemaNeige states shape (n_layers, 4)
    ctg: float,
    kf: float,
    precip_arr: np.ndarray,
    pet_arr: np.ndarray,
    temp_arr: np.ndarray,
    layer_elevations: np.ndarray,
    layer_fractions: np.ndarray,
    input_elevation: float,
    temp_gradient: float,
    precip_gradient: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    gr6j_outputs_arr: np.ndarray,  # shape (n_timesteps, 20)
    snow_outputs_arr: np.ndarray,  # shape (n_timesteps, 12)
    layer_outputs_arr: np.ndarray,  # shape (n_timesteps, n_layers, 7)
) -> None:
    """Run coupled CemaNeige-GR6J over a timeseries (multi-layer snow).

    Layer output layout per layer: [snow_pack, thermal_state, gratio, melt,
                                    pliq_and_melt, layer_temp, layer_precip]
    """
    n_timesteps = len(precip_arr)
    n_layers = layer_states_arr.shape[0]

    gr6j_output_single = np.zeros(20)
    out_layer_states = np.zeros((n_layers, 4))
    out_aggregated_fluxes = np.zeros(11)
    out_per_layer_fluxes = np.zeros((n_layers, 11))

    for t in range(n_timesteps):
        precip_raw = precip_arr[t]

        # Run multi-layer CemaNeige
        _cemaneige_multi_layer_step_numba(
            layer_states_arr,
            ctg,
            kf,
            precip_raw,
            temp_arr[t],
            layer_elevations,
            layer_fractions,
            input_elevation,
            temp_gradient,
            precip_gradient,
            out_layer_states,
            out_aggregated_fluxes,
            out_per_layer_fluxes,
        )

        # Update layer states
        for i in range(n_layers):
            for j in range(4):
                layer_states_arr[i, j] = out_layer_states[i, j]

        # Use aggregated snow output as GR6J precip input
        precip_for_gr6j = out_aggregated_fluxes[7]  # pliq_and_melt

        # Run GR6J
        _step_numba(
            state_arr,
            params_arr,
            precip_for_gr6j,
            pet_arr[t],
            uh1_ordinates,
            uh2_ordinates,
            gr6j_output_single,
        )

        # Copy GR6J outputs
        for i in range(20):
            gr6j_outputs_arr[t, i] = gr6j_output_single[i]

        # Copy aggregated snow outputs with precip_raw as first element
        snow_outputs_arr[t, 0] = precip_raw
        for i in range(11):
            snow_outputs_arr[t, i + 1] = out_aggregated_fluxes[i]

        # Copy per-layer outputs
        for i in range(n_layers):
            layer_outputs_arr[t, i, 0] = out_per_layer_fluxes[i, 2]  # snow_pack
            layer_outputs_arr[t, i, 1] = out_per_layer_fluxes[i, 3]  # thermal_state
            layer_outputs_arr[t, i, 2] = out_per_layer_fluxes[i, 4]  # gratio
            layer_outputs_arr[t, i, 3] = out_per_layer_fluxes[i, 6]  # melt
            layer_outputs_arr[t, i, 4] = out_per_layer_fluxes[i, 7]  # pliq_and_melt
            layer_outputs_arr[t, i, 5] = out_per_layer_fluxes[i, 8]  # layer_temp (snow_temp)
            # layer_precip = pliq + psol
            layer_outputs_arr[t, i, 6] = out_per_layer_fluxes[i, 0] + out_per_layer_fluxes[i, 1]


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
    initial_snow_state: CemaNeigeSingleLayerState | CemaNeigeMultiLayerState | None = None,
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
        initial_snow_state: Optional initial CemaNeige state. Accepts both
            CemaNeigeSingleLayerState (single-layer) and CemaNeigeMultiLayerState
            (multi-layer). If None and snow module is enabled, initializes
            automatically based on catchment.n_layers.

    Returns:
        ModelOutput containing GR6J outputs and optionally snow outputs.
        Access streamflow via result.gr6j.streamflow (numpy array).
        When multi-layer snow mode is used, per-layer outputs are available
        via result.snow_layers (SnowLayerOutputs).
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
    snow_state: CemaNeigeSingleLayerState | CemaNeigeMultiLayerState | None = None
    is_multi_layer = False
    layer_elevations: np.ndarray | None = None
    layer_fractions: np.ndarray | None = None

    if params.has_snow:
        assert catchment is not None  # validated above

        if catchment.n_layers > 1:
            is_multi_layer = True
            # Derive layer properties from hypsometric curve
            layer_elevations, layer_fractions = derive_layers(
                catchment.hypsometric_curve,
                catchment.n_layers,  # type: ignore[arg-type]
            )

            if initial_snow_state is None:
                snow_state = CemaNeigeMultiLayerState.initialize(
                    n_layers=catchment.n_layers,
                    mean_annual_solid_precip=catchment.mean_annual_solid_precip,
                )
            else:
                snow_state = initial_snow_state
        else:
            if initial_snow_state is None:
                snow_state = CemaNeigeSingleLayerState.initialize(catchment.mean_annual_solid_precip)
            else:
                snow_state = initial_snow_state

    # Compute unit hydrograph ordinates once
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    # Initialize output arrays
    n_timesteps = len(forcing)

    # Fast path: use Numba kernel for non-snow runs
    if not params.has_snow:
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
        gr6j_output = GR6JOutput(
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
            gr6j=gr6j_output,
            snow=None,
            snow_layers=None,
        )

    # Fast path: use Numba kernel for single-layer snow runs
    if params.has_snow and not is_multi_layer:
        assert snow_state is not None
        assert isinstance(snow_state, CemaNeigeSingleLayerState)

        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        snow_state_arr = np.asarray(snow_state)

        gr6j_outputs_arr = np.zeros((n_timesteps, 20), dtype=np.float64)
        snow_outputs_arr = np.zeros((n_timesteps, 12), dtype=np.float64)

        _run_with_snow_numba(
            state_arr,
            params_arr,
            snow_state_arr,
            params.snow.ctg,
            params.snow.kf,
            forcing.precip.astype(np.float64),
            forcing.pet.astype(np.float64),
            forcing.temp.astype(np.float64),
            uh1_ordinates,
            uh2_ordinates,
            gr6j_outputs_arr,
            snow_outputs_arr,
        )

        # Build GR6J output
        gr6j_output = GR6JOutput(
            pet=gr6j_outputs_arr[:, 0],
            precip=gr6j_outputs_arr[:, 1],
            production_store=gr6j_outputs_arr[:, 2],
            net_rainfall=gr6j_outputs_arr[:, 3],
            storage_infiltration=gr6j_outputs_arr[:, 4],
            actual_et=gr6j_outputs_arr[:, 5],
            percolation=gr6j_outputs_arr[:, 6],
            effective_rainfall=gr6j_outputs_arr[:, 7],
            q9=gr6j_outputs_arr[:, 8],
            q1=gr6j_outputs_arr[:, 9],
            routing_store=gr6j_outputs_arr[:, 10],
            exchange=gr6j_outputs_arr[:, 11],
            actual_exchange_routing=gr6j_outputs_arr[:, 12],
            actual_exchange_direct=gr6j_outputs_arr[:, 13],
            actual_exchange_total=gr6j_outputs_arr[:, 14],
            qr=gr6j_outputs_arr[:, 15],
            qrexp=gr6j_outputs_arr[:, 16],
            exponential_store=gr6j_outputs_arr[:, 17],
            qd=gr6j_outputs_arr[:, 18],
            streamflow=gr6j_outputs_arr[:, 19],
        )

        # Build snow output
        snow_output = SnowOutput(
            precip_raw=snow_outputs_arr[:, 0],
            snow_pliq=snow_outputs_arr[:, 1],
            snow_psol=snow_outputs_arr[:, 2],
            snow_pack=snow_outputs_arr[:, 3],
            snow_thermal_state=snow_outputs_arr[:, 4],
            snow_gratio=snow_outputs_arr[:, 5],
            snow_pot_melt=snow_outputs_arr[:, 6],
            snow_melt=snow_outputs_arr[:, 7],
            snow_pliq_and_melt=snow_outputs_arr[:, 8],
            snow_temp=snow_outputs_arr[:, 9],
            snow_gthreshold=snow_outputs_arr[:, 10],
            snow_glocalmax=snow_outputs_arr[:, 11],
        )

        return ModelOutput(
            time=forcing.time,
            gr6j=gr6j_output,
            snow=snow_output,
            snow_layers=None,
        )

    # Fast path: use Numba kernel for multi-layer snow runs
    if params.has_snow and is_multi_layer:
        assert snow_state is not None
        assert isinstance(snow_state, CemaNeigeMultiLayerState)
        assert layer_elevations is not None
        assert layer_fractions is not None
        assert catchment is not None

        state_arr = np.asarray(state)
        params_arr = np.asarray(params)
        layer_states_arr = np.asarray(snow_state)
        n_layers = layer_states_arr.shape[0]

        gr6j_outputs_arr = np.zeros((n_timesteps, 20), dtype=np.float64)
        snow_outputs_arr = np.zeros((n_timesteps, 12), dtype=np.float64)
        layer_outputs_arr = np.zeros((n_timesteps, n_layers, 7), dtype=np.float64)

        # Use default gradients if not specified
        temp_grad = catchment.temp_gradient if catchment.temp_gradient is not None else 0.6
        precip_grad = catchment.precip_gradient if catchment.precip_gradient is not None else 0.00041

        _run_with_multi_layer_snow_numba(
            state_arr,
            params_arr,
            layer_states_arr,
            params.snow.ctg,
            params.snow.kf,
            forcing.precip.astype(np.float64),
            forcing.pet.astype(np.float64),
            forcing.temp.astype(np.float64),
            layer_elevations.astype(np.float64),
            layer_fractions.astype(np.float64),
            float(catchment.input_elevation),
            temp_grad,
            precip_grad,
            uh1_ordinates,
            uh2_ordinates,
            gr6j_outputs_arr,
            snow_outputs_arr,
            layer_outputs_arr,
        )

        # Build GR6J output (same as single-layer)
        gr6j_output = GR6JOutput(
            pet=gr6j_outputs_arr[:, 0],
            precip=gr6j_outputs_arr[:, 1],
            production_store=gr6j_outputs_arr[:, 2],
            net_rainfall=gr6j_outputs_arr[:, 3],
            storage_infiltration=gr6j_outputs_arr[:, 4],
            actual_et=gr6j_outputs_arr[:, 5],
            percolation=gr6j_outputs_arr[:, 6],
            effective_rainfall=gr6j_outputs_arr[:, 7],
            q9=gr6j_outputs_arr[:, 8],
            q1=gr6j_outputs_arr[:, 9],
            routing_store=gr6j_outputs_arr[:, 10],
            exchange=gr6j_outputs_arr[:, 11],
            actual_exchange_routing=gr6j_outputs_arr[:, 12],
            actual_exchange_direct=gr6j_outputs_arr[:, 13],
            actual_exchange_total=gr6j_outputs_arr[:, 14],
            qr=gr6j_outputs_arr[:, 15],
            qrexp=gr6j_outputs_arr[:, 16],
            exponential_store=gr6j_outputs_arr[:, 17],
            qd=gr6j_outputs_arr[:, 18],
            streamflow=gr6j_outputs_arr[:, 19],
        )

        # Build snow output
        snow_output = SnowOutput(
            precip_raw=snow_outputs_arr[:, 0],
            snow_pliq=snow_outputs_arr[:, 1],
            snow_psol=snow_outputs_arr[:, 2],
            snow_pack=snow_outputs_arr[:, 3],
            snow_thermal_state=snow_outputs_arr[:, 4],
            snow_gratio=snow_outputs_arr[:, 5],
            snow_pot_melt=snow_outputs_arr[:, 6],
            snow_melt=snow_outputs_arr[:, 7],
            snow_pliq_and_melt=snow_outputs_arr[:, 8],
            snow_temp=snow_outputs_arr[:, 9],
            snow_gthreshold=snow_outputs_arr[:, 10],
            snow_glocalmax=snow_outputs_arr[:, 11],
        )

        # Build layer outputs
        snow_layer_output = SnowLayerOutputs(
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            snow_pack=layer_outputs_arr[:, :, 0],
            snow_thermal_state=layer_outputs_arr[:, :, 1],
            snow_gratio=layer_outputs_arr[:, :, 2],
            snow_melt=layer_outputs_arr[:, :, 3],
            snow_pliq_and_melt=layer_outputs_arr[:, :, 4],
            layer_temp=layer_outputs_arr[:, :, 5],
            layer_precip=layer_outputs_arr[:, :, 6],
        )

        return ModelOutput(
            time=forcing.time,
            gr6j=gr6j_output,
            snow=snow_output,
            snow_layers=snow_layer_output,
        )

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

    # Per-layer outputs for multi-layer mode
    layer_output_data: dict[str, list[list[float]]] | None = None
    if params.has_snow and is_multi_layer:
        assert layer_elevations is not None
        layer_output_data = {
            "snow_pack": [],
            "snow_thermal_state": [],
            "snow_gratio": [],
            "snow_melt": [],
            "snow_pliq_and_melt": [],
            "layer_temp": [],
            "layer_precip": [],
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
            temp = float(forcing.temp[idx])  # type: ignore[index]
            assert catchment is not None

            if is_multi_layer and isinstance(snow_state, CemaNeigeMultiLayerState):
                assert layer_elevations is not None
                assert layer_fractions is not None

                snow_state, snow_fluxes, per_layer_fluxes = cemaneige_multi_layer_step(
                    state=snow_state,
                    params=params.snow,  # type: ignore[arg-type]
                    precip=precip,
                    temp=temp,
                    layer_elevations=layer_elevations,
                    layer_fractions=layer_fractions,
                    input_elevation=catchment.input_elevation,  # type: ignore[arg-type]
                    temp_gradient=catchment.temp_gradient,
                    precip_gradient=catchment.precip_gradient,
                )

                # Store per-layer data
                if layer_output_data is not None:
                    for key in ["snow_pack", "snow_thermal_state", "snow_gratio", "snow_melt", "snow_pliq_and_melt"]:
                        layer_output_data[key].append([lf[key] for lf in per_layer_fluxes])
                    layer_output_data["layer_temp"].append([lf["snow_temp"] for lf in per_layer_fluxes])
                    layer_output_data["layer_precip"].append(
                        [lf["snow_pliq"] + lf["snow_psol"] for lf in per_layer_fluxes]
                    )
            else:
                assert isinstance(snow_state, CemaNeigeSingleLayerState)
                snow_state, snow_fluxes = cemaneige_step(
                    state=snow_state,
                    params=params.snow,  # type: ignore[arg-type]
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

    snow_layer_output: SnowLayerOutputs | None = None
    if layer_output_data is not None and layer_elevations is not None and layer_fractions is not None:
        snow_layer_output = SnowLayerOutputs(
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            snow_pack=np.array(layer_output_data["snow_pack"]),
            snow_thermal_state=np.array(layer_output_data["snow_thermal_state"]),
            snow_gratio=np.array(layer_output_data["snow_gratio"]),
            snow_melt=np.array(layer_output_data["snow_melt"]),
            snow_pliq_and_melt=np.array(layer_output_data["snow_pliq_and_melt"]),
            layer_temp=np.array(layer_output_data["layer_temp"]),
            layer_precip=np.array(layer_output_data["layer_precip"]),
        )

    return ModelOutput(
        time=forcing.time,
        gr6j=gr6j_output,
        snow=snow_output,
        snow_layers=snow_layer_output,
    )
