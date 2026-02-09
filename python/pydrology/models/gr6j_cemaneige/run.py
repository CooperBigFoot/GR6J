"""GR6J-CemaNeige coupled model orchestration functions.

This module provides the main entry points for running the coupled GR6J-CemaNeige model:
- step(): Execute a single timestep (with snow processing)
- run(): Execute the model over a timeseries

Design: Unified multi-layer approach. If catchment.input_elevation is None,
extrapolation is skipped (single-layer behavior). Single code path for all cases.
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401
import math

import numpy as np
from numba import njit

from pydrology.utils.elevation import (
    ELEV_CAP_PRECIP,
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
)
from pydrology.cemaneige.processes import (
    compute_actual_melt,
    compute_gratio,
    compute_potential_melt,
    compute_solid_fraction,
    partition_precipitation,
    update_thermal_state,
)
from pydrology.models.cemaneige.outputs import SnowLayerOutputs, SnowOutput
from pydrology.models.gr6j.run import _step_numba
from pydrology.models.gr6j.unit_hydrographs import compute_uh_ordinates
from pydrology.outputs import ModelOutput
from pydrology.types import Catchment, ForcingData

from .constants import SUPPORTED_RESOLUTIONS
from .outputs import GR6JCemaNeigeFluxes
from .types import Parameters, State

# Constants inlined for Numba compatibility
_GRAD_T_DEFAULT: float = GRAD_T_DEFAULT
_GRAD_P_DEFAULT: float = GRAD_P_DEFAULT
_ELEV_CAP_PRECIP: float = ELEV_CAP_PRECIP


@njit(cache=True)
def _cemaneige_layer_step_numba(
    layer_state: np.ndarray,  # shape (4,) - [g, etg, gthreshold, glocalmax]
    ctg: float,
    kf: float,
    precip: float,
    temp: float,
    out_state: np.ndarray,  # shape (4,) - output state
    out_fluxes: np.ndarray,  # shape (10,) - output fluxes [pliq, psol, snow_pack, thermal_state, gratio, pot_melt, melt, pliq_and_melt, temp, precip]
) -> None:
    """Execute one CemaNeige timestep for a single layer.

    Flux output layout: [pliq, psol, snow_pack, thermal_state, gratio, pot_melt, melt, pliq_and_melt, temp, precip]
    """
    # Unpack state
    g = layer_state[0]
    etg = layer_state[1]
    gthreshold = layer_state[2]
    glocalmax = layer_state[3]

    # 1. Compute solid fraction
    solid_fraction = compute_solid_fraction(temp)

    # 2. Partition precipitation
    pliq, psol = partition_precipitation(precip, solid_fraction)

    # 3. Accumulate snow
    g = g + psol

    # 4. Update thermal state
    etg = update_thermal_state(etg, temp, ctg)

    # 5. Compute potential melt
    pot_melt = compute_potential_melt(etg, temp, kf, g)

    # 6. Compute gratio before melt (for melt calculation)
    gratio_for_melt = compute_gratio(g, gthreshold)

    # 7. Compute actual melt
    melt = compute_actual_melt(pot_melt, gratio_for_melt)

    # 8. Update snow pack
    g = g - melt

    # 9. Compute output gratio (after melt)
    gratio_output = compute_gratio(g, gthreshold)

    # 10. Total liquid output
    pliq_and_melt = pliq + melt

    # Write output state
    out_state[0] = g
    out_state[1] = etg
    out_state[2] = gthreshold
    out_state[3] = glocalmax

    # Write fluxes
    out_fluxes[0] = pliq
    out_fluxes[1] = psol
    out_fluxes[2] = g  # snow_pack after melt
    out_fluxes[3] = etg  # thermal_state
    out_fluxes[4] = gratio_output
    out_fluxes[5] = pot_melt
    out_fluxes[6] = melt
    out_fluxes[7] = pliq_and_melt
    out_fluxes[8] = temp
    out_fluxes[9] = precip  # layer precip


@njit(cache=True)
def _run_unified_numba(
    state_arr: np.ndarray,  # shape (63 + n_layers*4,)
    params_arr: np.ndarray,  # shape (8,)
    precip_arr: np.ndarray,  # shape (n_timesteps,)
    pet_arr: np.ndarray,  # shape (n_timesteps,)
    temp_arr: np.ndarray,  # shape (n_timesteps,)
    uh1_ordinates: np.ndarray,  # shape (20,)
    uh2_ordinates: np.ndarray,  # shape (40,)
    n_layers: int,
    layer_elevations: np.ndarray,  # shape (n_layers,)
    layer_fractions: np.ndarray,  # shape (n_layers,)
    input_elevation: float,  # NaN if no extrapolation
    temp_gradient: float,
    precip_gradient: float,
    snow_outputs: np.ndarray,  # shape (n_timesteps, 10) - aggregated snow fluxes
    gr6j_outputs: np.ndarray,  # shape (n_timesteps, 20) - GR6J fluxes
    layer_outputs: np.ndarray,  # shape (n_timesteps, n_layers, 10) - per-layer snow fluxes
) -> None:
    """Run unified GR6J-CemaNeige over timeseries.

    If input_elevation is NaN, skip extrapolation (single-layer behavior).
    """
    n_timesteps = len(precip_arr)
    gr6j_params = params_arr[:6]  # x1-x6
    ctg = params_arr[6]
    kf = params_arr[7]

    # Working arrays for snow module
    layer_state = np.zeros(4)
    out_layer_state = np.zeros(4)
    out_layer_fluxes = np.zeros(10)
    aggregated_snow = np.zeros(10)
    gr6j_output_single = np.zeros(20)

    # Check if we need extrapolation
    skip_extrapolation = math.isnan(input_elevation)

    for t in range(n_timesteps):
        precip = precip_arr[t]
        temp = temp_arr[t]
        pet = pet_arr[t]

        # Reset aggregated snow fluxes
        for i in range(10):
            aggregated_snow[i] = 0.0

        # Process each snow layer
        for layer_idx in range(n_layers):
            # Get layer state from state_arr
            state_offset = 63 + layer_idx * 4
            for j in range(4):
                layer_state[j] = state_arr[state_offset + j]

            # Extrapolate temperature and precipitation if needed
            if skip_extrapolation:
                layer_temp = temp
                layer_precip = precip
            else:
                # Extrapolate temperature
                layer_temp = temp - temp_gradient * (layer_elevations[layer_idx] - input_elevation) / 100.0

                # Extrapolate precipitation
                effective_input_elev = min(input_elevation, _ELEV_CAP_PRECIP)
                effective_layer_elev = min(layer_elevations[layer_idx], _ELEV_CAP_PRECIP)
                layer_precip = precip * math.exp(precip_gradient * (effective_layer_elev - effective_input_elev))

            # Run CemaNeige step for this layer
            _cemaneige_layer_step_numba(
                layer_state, ctg, kf, layer_precip, layer_temp, out_layer_state, out_layer_fluxes
            )

            # Update state array
            for j in range(4):
                state_arr[state_offset + j] = out_layer_state[j]

            # Store per-layer outputs
            for j in range(10):
                layer_outputs[t, layer_idx, j] = out_layer_fluxes[j]

            # Aggregate (area-weighted)
            fraction = layer_fractions[layer_idx]
            for j in range(10):
                aggregated_snow[j] += out_layer_fluxes[j] * fraction

        # Store aggregated snow outputs
        for j in range(10):
            snow_outputs[t, j] = aggregated_snow[j]

        # Get pliq_and_melt as precipitation input for GR6J
        pliq_and_melt = aggregated_snow[7]

        # Run GR6J step (state_arr[:63] is modified in place)
        _step_numba(
            state_arr,  # GR6J uses first 63 elements
            gr6j_params,
            pliq_and_melt,  # Liquid water from snow module as precip
            pet,
            uh1_ordinates,
            uh2_ordinates,
            gr6j_output_single,
        )

        # Store GR6J outputs
        for j in range(20):
            gr6j_outputs[t, j] = gr6j_output_single[j]


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    layer_elevations: np.ndarray | None = None,
    layer_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR6J-CemaNeige coupled model.

    Processes snow accumulation/melt through CemaNeige, then routes
    the liquid water through GR6J.

    Args:
        state: Current combined model state.
        params: Model parameters (8 total).
        precip: Daily precipitation [mm/day].
        pet: Daily potential evapotranspiration [mm/day].
        temp: Daily air temperature [deg C].
        uh1_ordinates: Pre-computed UH1 ordinates.
        uh2_ordinates: Pre-computed UH2 ordinates.
        layer_elevations: Elevation of each layer [m]. Required for multi-layer.
        layer_fractions: Area fraction of each layer [-]. Required for multi-layer.
        input_elevation: Elevation of forcing data [m]. If None, skip extrapolation.
        temp_gradient: Temperature lapse rate [deg C/100m]. Default 0.6.
        precip_gradient: Precipitation gradient [m^-1]. Default 0.00041.

    Returns:
        Tuple of (new_state, fluxes) where fluxes contains all snow and GR6J outputs.
    """
    n_layers = state.n_layers

    # Set defaults for layer config
    if layer_elevations is None:
        layer_elevations = np.zeros(n_layers)
    if layer_fractions is None:
        layer_fractions = np.ones(n_layers) / n_layers
    if temp_gradient is None:
        temp_gradient = GRAD_T_DEFAULT
    if precip_gradient is None:
        precip_gradient = GRAD_P_DEFAULT

    # Convert state and params to arrays
    state_arr = np.asarray(state)
    params_arr = np.asarray(params)

    # Allocate outputs
    snow_outputs = np.zeros((1, 10))
    gr6j_outputs = np.zeros((1, 20))
    layer_outputs = np.zeros((1, n_layers, 10))

    # Use NaN to signal no extrapolation
    input_elev = float("nan") if input_elevation is None else input_elevation

    # Run single timestep
    _run_unified_numba(
        state_arr,
        params_arr,
        np.array([precip]),
        np.array([pet]),
        np.array([temp]),
        uh1_ordinates,
        uh2_ordinates,
        n_layers,
        layer_elevations,
        layer_fractions,
        input_elev,
        temp_gradient,
        precip_gradient,
        snow_outputs,
        gr6j_outputs,
        layer_outputs,
    )

    # Reconstruct state
    new_state = State.from_array(state_arr, n_layers)

    # Build fluxes dictionary
    fluxes: dict[str, float] = {
        # Snow outputs
        "precip_raw": precip,
        "snow_pliq": snow_outputs[0, 0],
        "snow_psol": snow_outputs[0, 1],
        "snow_pack": snow_outputs[0, 2],
        "snow_thermal_state": snow_outputs[0, 3],
        "snow_gratio": snow_outputs[0, 4],
        "snow_pot_melt": snow_outputs[0, 5],
        "snow_melt": snow_outputs[0, 6],
        "snow_pliq_and_melt": snow_outputs[0, 7],
        "snow_temp": snow_outputs[0, 8],
        # GR6J outputs
        "pet": gr6j_outputs[0, 0],
        "precip": gr6j_outputs[0, 1],
        "production_store": gr6j_outputs[0, 2],
        "net_rainfall": gr6j_outputs[0, 3],
        "storage_infiltration": gr6j_outputs[0, 4],
        "actual_et": gr6j_outputs[0, 5],
        "percolation": gr6j_outputs[0, 6],
        "effective_rainfall": gr6j_outputs[0, 7],
        "q9": gr6j_outputs[0, 8],
        "q1": gr6j_outputs[0, 9],
        "routing_store": gr6j_outputs[0, 10],
        "exchange": gr6j_outputs[0, 11],
        "actual_exchange_routing": gr6j_outputs[0, 12],
        "actual_exchange_direct": gr6j_outputs[0, 13],
        "actual_exchange_total": gr6j_outputs[0, 14],
        "qr": gr6j_outputs[0, 15],
        "qrexp": gr6j_outputs[0, 16],
        "exponential_store": gr6j_outputs[0, 17],
        "qd": gr6j_outputs[0, 18],
        "streamflow": gr6j_outputs[0, 19],
    }

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    catchment: Catchment,
    initial_state: State | None = None,
) -> ModelOutput[GR6JCemaNeigeFluxes]:
    """Run the GR6J-CemaNeige coupled model over a timeseries.

    Executes the coupled snow-hydrology model for each timestep in the input
    forcing data. Snow is processed through CemaNeige (with optional multi-layer
    elevation bands), and the liquid water output is routed through GR6J.

    Design decision: Unified multi-layer code path. If catchment.input_elevation
    is None, extrapolation is skipped (equivalent to single-layer behavior).

    Args:
        params: Model parameters (8 total: 6 GR6J + 2 CemaNeige).
        forcing: Input forcing data with precip, pet, and temp arrays.
        catchment: Catchment properties (mean_annual_solid_precip, elevation config).
        initial_state: Initial model state. If None, uses State.initialize().

    Returns:
        ModelOutput containing GR6JCemaNeigeFluxes, SnowOutput, and optionally
        SnowLayerOutputs (for multi-layer simulations).

    Raises:
        ValueError: If forcing.temp is None (temperature required for snow module).

    Example:
        >>> params = Parameters(x1=350, x2=0, x3=90, x4=1.7, x5=0, x6=5, ctg=0.5, kf=4.0)
        >>> catchment = Catchment(mean_annual_solid_precip=100.0)
        >>> forcing = ForcingData(
        ...     time=np.array(['2020-01-01', '2020-01-02'], dtype='datetime64'),
        ...     precip=np.array([10.0, 5.0]),
        ...     pet=np.array([3.0, 4.0]),
        ...     temp=np.array([2.0, -1.0]),
        ... )
        >>> result = run(params, forcing, catchment)
        >>> result.streamflow
        array([...])
    """
    # Validate temperature is provided
    if forcing.temp is None:
        msg = "Temperature data (forcing.temp) is required for the GR6J-CemaNeige model"
        raise ValueError(msg)

    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR6J-CemaNeige supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    n_timesteps = len(forcing)
    n_layers = catchment.n_layers

    # Derive layer properties if multi-layer
    if n_layers > 1 and catchment.hypsometric_curve is not None:
        layer_elevations, layer_fractions = derive_layers(catchment.hypsometric_curve, n_layers)
    else:
        # Single layer: use dummy values (extrapolation will be skipped)
        layer_elevations = np.zeros(n_layers)
        layer_fractions = np.ones(n_layers) / n_layers

    # Initialize state if not provided
    state = State.initialize(params, catchment) if initial_state is None else initial_state

    # Compute unit hydrograph ordinates once
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    # Set gradients (use defaults if not specified)
    temp_gradient = catchment.temp_gradient if catchment.temp_gradient is not None else GRAD_T_DEFAULT
    precip_gradient = catchment.precip_gradient if catchment.precip_gradient is not None else GRAD_P_DEFAULT

    # Use NaN to signal no extrapolation
    input_elevation = catchment.input_elevation if catchment.input_elevation is not None else float("nan")

    # Convert to arrays
    state_arr = np.asarray(state)
    params_arr = np.asarray(params)

    # Allocate output arrays
    snow_outputs = np.zeros((n_timesteps, 10), dtype=np.float64)
    gr6j_outputs = np.zeros((n_timesteps, 20), dtype=np.float64)
    layer_outputs = np.zeros((n_timesteps, n_layers, 10), dtype=np.float64)

    # Run the Numba kernel
    _run_unified_numba(
        state_arr,
        params_arr,
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        forcing.temp.astype(np.float64),
        uh1_ordinates,
        uh2_ordinates,
        n_layers,
        layer_elevations.astype(np.float64),
        layer_fractions.astype(np.float64),
        input_elevation,
        temp_gradient,
        precip_gradient,
        snow_outputs,
        gr6j_outputs,
        layer_outputs,
    )

    # Build combined flux output
    combined_fluxes = GR6JCemaNeigeFluxes(
        # Snow outputs (10 fields)
        precip_raw=forcing.precip.copy(),
        snow_pliq=snow_outputs[:, 0],
        snow_psol=snow_outputs[:, 1],
        snow_pack=snow_outputs[:, 2],
        snow_thermal_state=snow_outputs[:, 3],
        snow_gratio=snow_outputs[:, 4],
        snow_pot_melt=snow_outputs[:, 5],
        snow_melt=snow_outputs[:, 6],
        snow_pliq_and_melt=snow_outputs[:, 7],
        snow_temp=snow_outputs[:, 8],
        # GR6J outputs (20 fields)
        pet=gr6j_outputs[:, 0],
        precip=gr6j_outputs[:, 1],
        production_store=gr6j_outputs[:, 2],
        net_rainfall=gr6j_outputs[:, 3],
        storage_infiltration=gr6j_outputs[:, 4],
        actual_et=gr6j_outputs[:, 5],
        percolation=gr6j_outputs[:, 6],
        effective_rainfall=gr6j_outputs[:, 7],
        q9=gr6j_outputs[:, 8],
        q1=gr6j_outputs[:, 9],
        routing_store=gr6j_outputs[:, 10],
        exchange=gr6j_outputs[:, 11],
        actual_exchange_routing=gr6j_outputs[:, 12],
        actual_exchange_direct=gr6j_outputs[:, 13],
        actual_exchange_total=gr6j_outputs[:, 14],
        qr=gr6j_outputs[:, 15],
        qrexp=gr6j_outputs[:, 16],
        exponential_store=gr6j_outputs[:, 17],
        qd=gr6j_outputs[:, 18],
        streamflow=gr6j_outputs[:, 19],
    )

    # Build SnowOutput for backward compatibility
    snow_output = SnowOutput(
        precip_raw=forcing.precip.copy(),
        snow_pliq=snow_outputs[:, 0],
        snow_psol=snow_outputs[:, 1],
        snow_pack=snow_outputs[:, 2],
        snow_thermal_state=snow_outputs[:, 3],
        snow_gratio=snow_outputs[:, 4],
        snow_pot_melt=snow_outputs[:, 5],
        snow_melt=snow_outputs[:, 6],
        snow_pliq_and_melt=snow_outputs[:, 7],
        snow_temp=snow_outputs[:, 8],
        snow_gthreshold=np.full(n_timesteps, catchment.mean_annual_solid_precip * 0.9),
        snow_glocalmax=np.full(n_timesteps, catchment.mean_annual_solid_precip * 0.9),
    )

    # Build per-layer outputs if multi-layer
    snow_layers: SnowLayerOutputs | None = None
    if n_layers > 1:
        snow_layers = SnowLayerOutputs(
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            snow_pack=layer_outputs[:, :, 2],
            snow_thermal_state=layer_outputs[:, :, 3],
            snow_gratio=layer_outputs[:, :, 4],
            snow_melt=layer_outputs[:, :, 6],
            snow_pliq_and_melt=layer_outputs[:, :, 7],
            layer_temp=layer_outputs[:, :, 8],
            layer_precip=layer_outputs[:, :, 9],
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=combined_fluxes,
        snow=snow_output,
        snow_layers=snow_layers,
    )
