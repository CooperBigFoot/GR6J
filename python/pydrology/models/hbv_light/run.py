"""HBV-light model orchestration functions.

This module provides the main entry points for running the HBV-light model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

# ruff: noqa: I001, SIM108
# Import order matters: _compat must patch numpy before numba import
# SIM108: Ternary operators disabled in Numba functions for clarity
import pydrology._compat  # noqa: F401

import math

import numpy as np
from numba import njit

from pydrology.outputs import ModelOutput
from pydrology.types import Catchment, ForcingData
from pydrology.utils.elevation import (
    ELEV_CAP_PRECIP,
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
    extrapolate_precipitation,
    extrapolate_temperature,
)
from .outputs import HBVLightFluxes, HBVLightZoneOutputs
from .processes import (
    compute_actual_et,
    compute_melt,
    compute_recharge,
    compute_refreezing,
    partition_precipitation,
    update_snow_pack,
    update_soil_moisture,
)
from .routing import (
    compute_percolation,
    compute_triangular_weights,
    convolve_triangular,
    lower_zone_outflow,
    update_lower_zone,
    update_upper_zone,
    upper_zone_outflows,
)
from .constants import SUPPORTED_RESOLUTIONS
from .types import Parameters, State


@njit(cache=True)
def _step_numba(
    state_arr: np.ndarray,  # Modified in place
    params_arr: np.ndarray,
    precip: float,
    pet: float,
    temp: float,
    uh_weights: np.ndarray,
    output_arr: np.ndarray,  # Output written here (20 elements)
) -> None:
    """Execute one timestep of HBV-light using arrays."""
    # Unpack params (14 elements)
    tt = params_arr[0]
    cfmax = params_arr[1]
    sfcf = params_arr[2]
    cwh = params_arr[3]
    cfr = params_arr[4]
    fc = params_arr[5]
    lp = params_arr[6]
    beta = params_arr[7]
    k0 = params_arr[8]
    k1 = params_arr[9]
    k2 = params_arr[10]
    perc_max = params_arr[11]
    uzl = params_arr[12]
    # maxbas = params_arr[13]  # Not used in step (UH weights precomputed)

    # Unpack state for single zone (12 elements for n_zones=1)
    # Layout: [zone_states (3), upper_zone, lower_zone, routing_buffer (7)]
    sp = state_arr[0]  # snow pack
    lw = state_arr[1]  # liquid water in snow
    sm = state_arr[2]  # soil moisture
    suz = state_arr[3]  # upper zone
    slz = state_arr[4]  # lower zone
    # routing buffer is state_arr[5:12]

    # 1. Snow routine - precipitation partitioning
    if temp > tt:
        p_rain = precip
        p_snow = 0.0
    else:
        p_rain = 0.0
        p_snow = sfcf * precip

    # Compute melt
    if temp > tt:
        melt = cfmax * (temp - tt)
        if melt > sp:
            melt = sp
    else:
        melt = 0.0

    # Compute refreezing
    if temp < tt:
        refreeze = cfr * cfmax * (tt - temp)
        if refreeze > lw:
            refreeze = lw
    else:
        refreeze = 0.0

    # Update snow pack
    new_sp = sp + p_snow - melt + refreeze
    new_lw = lw + melt - refreeze
    lw_max = cwh * new_sp
    if new_lw > lw_max:
        snow_outflow = new_lw - lw_max
        new_lw = lw_max
    else:
        snow_outflow = 0.0
    if new_sp < 0.0:
        new_sp = 0.0
    if new_lw < 0.0:
        new_lw = 0.0

    # Total input to soil
    snow_input = p_rain + snow_outflow

    # 2. Soil routine - compute recharge
    if fc <= 0.0 or snow_input <= 0.0:
        recharge = 0.0
    else:
        sr = sm / fc
        if sr < 0.0:
            sr = 0.0
        elif sr > 1.0:
            sr = 1.0
        recharge = snow_input * (sr**beta)

    # Compute actual ET
    if fc <= 0.0 or lp <= 0.0:
        et_act = 0.0
    else:
        lp_threshold = lp * fc
        if sm >= lp_threshold:
            et_act = pet
        else:
            et_act = pet * sm / lp_threshold
        if et_act > sm:
            et_act = sm
        if et_act < 0.0:
            et_act = 0.0

    # Update soil moisture
    infiltration = snow_input - recharge
    new_sm = sm + infiltration - et_act
    if new_sm < 0.0:
        new_sm = 0.0
    elif new_sm > fc:
        new_sm = fc

    # 3. Response routine - upper zone outflows
    if suz > uzl:
        q0 = k0 * (suz - uzl)
    else:
        q0 = 0.0
    q1 = k1 * suz

    # Percolation
    if suz > perc_max:
        perc = perc_max
    else:
        perc = suz if suz > 0.0 else 0.0

    # Update upper zone
    new_suz = suz + recharge - q0 - q1 - perc
    if new_suz < 0.0:
        new_suz = 0.0

    # Lower zone
    q2 = k2 * slz
    new_slz = slz + perc - q2
    if new_slz < 0.0:
        new_slz = 0.0

    qgw = q0 + q1 + q2

    # 4. Routing - get routing buffer and convolve
    # Output is first buffer element
    qsim = state_arr[5]

    # Create new buffer, shifted left
    new_buffer = np.zeros(7, dtype=np.float64)
    for i in range(6):
        new_buffer[i] = state_arr[6 + i]

    # Add current input contribution using weights
    n_weights = len(uh_weights)
    for i in range(min(n_weights, 7)):
        new_buffer[i] += qgw * uh_weights[i]

    # Update state array in place
    state_arr[0] = new_sp
    state_arr[1] = new_lw
    state_arr[2] = new_sm
    state_arr[3] = new_suz
    state_arr[4] = new_slz
    for i in range(7):
        state_arr[5 + i] = new_buffer[i]

    # Write outputs (20 elements)
    output_arr[0] = precip
    output_arr[1] = temp
    output_arr[2] = pet
    output_arr[3] = p_rain
    output_arr[4] = p_snow
    output_arr[5] = new_sp
    output_arr[6] = melt
    output_arr[7] = new_lw
    output_arr[8] = snow_input
    output_arr[9] = new_sm
    output_arr[10] = recharge
    output_arr[11] = et_act
    output_arr[12] = new_suz
    output_arr[13] = new_slz
    output_arr[14] = q0
    output_arr[15] = q1
    output_arr[16] = q2
    output_arr[17] = perc
    output_arr[18] = qgw
    output_arr[19] = qsim


@njit(cache=True)
def _run_numba(
    state_arr: np.ndarray,
    params_arr: np.ndarray,
    precip_arr: np.ndarray,
    pet_arr: np.ndarray,
    temp_arr: np.ndarray,
    uh_weights: np.ndarray,
    outputs_arr: np.ndarray,  # shape (n_timesteps, 20)
) -> None:
    """Run HBV-light over a timeseries."""
    n_timesteps = len(precip_arr)
    output_single = np.zeros(20)

    for t in range(n_timesteps):
        _step_numba(
            state_arr,
            params_arr,
            precip_arr[t],
            pet_arr[t],
            temp_arr[t],
            uh_weights,
            output_single,
        )
        for i in range(20):
            outputs_arr[t, i] = output_single[i]


@njit(cache=True)
def _run_numba_multizone(
    state_arr: np.ndarray,  # Modified in place
    params_arr: np.ndarray,
    precip_arr: np.ndarray,
    pet_arr: np.ndarray,
    temp_arr: np.ndarray,
    uh_weights: np.ndarray,
    n_zones: int,
    zone_elevations: np.ndarray,  # shape (n_zones,)
    zone_fractions: np.ndarray,  # shape (n_zones,)
    input_elevation: float,  # NaN if no extrapolation
    temp_gradient: float,
    precip_gradient: float,
    elev_cap: float,
    outputs_arr: np.ndarray,  # shape (n_timesteps, 20)
    zone_outputs_arr: np.ndarray,  # shape (n_timesteps, n_zones, 11)
) -> None:
    """Run HBV-light over a timeseries with multi-zone elevation bands."""
    n_timesteps = len(precip_arr)

    # Unpack params
    tt = params_arr[0]
    cfmax = params_arr[1]
    sfcf = params_arr[2]
    cwh = params_arr[3]
    cfr = params_arr[4]
    fc = params_arr[5]
    lp = params_arr[6]
    beta = params_arr[7]
    k0 = params_arr[8]
    k1 = params_arr[9]
    k2 = params_arr[10]
    perc_max = params_arr[11]
    uzl = params_arr[12]

    # Check if extrapolation needed
    skip_extrapolation = math.isnan(input_elevation)

    # State layout: [zone_states (n_zones*3), upper_zone, lower_zone, routing_buffer (7)]
    suz_idx = n_zones * 3
    slz_idx = n_zones * 3 + 1
    buffer_start = n_zones * 3 + 2

    for t in range(n_timesteps):
        precip = precip_arr[t]
        temp = temp_arr[t]
        pet = pet_arr[t]

        # Aggregation accumulators
        agg_p_rain = 0.0
        agg_p_snow = 0.0
        agg_melt = 0.0
        agg_snow_input = 0.0
        agg_recharge = 0.0
        agg_et_act = 0.0
        agg_sm = 0.0
        agg_sp = 0.0
        agg_lw = 0.0

        # Process each zone
        for zone_idx in range(n_zones):
            fraction = zone_fractions[zone_idx]

            # Extrapolate forcing
            if skip_extrapolation:
                zone_temp = temp
                zone_precip = precip
            else:
                zone_temp = temp - temp_gradient * (zone_elevations[zone_idx] - input_elevation) / 100.0
                eff_input = min(input_elevation, elev_cap)
                eff_zone = min(zone_elevations[zone_idx], elev_cap)
                zone_precip = precip * math.exp(precip_gradient * (eff_zone - eff_input))

            # Get zone state indices
            sp_idx = zone_idx * 3
            lw_idx = zone_idx * 3 + 1
            sm_idx = zone_idx * 3 + 2

            sp = state_arr[sp_idx]
            lw = state_arr[lw_idx]
            sm = state_arr[sm_idx]

            # Snow routine
            if zone_temp > tt:
                p_rain = zone_precip
                p_snow = 0.0
            else:
                p_rain = 0.0
                p_snow = sfcf * zone_precip

            if zone_temp > tt:
                melt = cfmax * (zone_temp - tt)
                if melt > sp:
                    melt = sp
            else:
                melt = 0.0

            if zone_temp < tt:
                refreeze = cfr * cfmax * (tt - zone_temp)
                if refreeze > lw:
                    refreeze = lw
            else:
                refreeze = 0.0

            new_sp = sp + p_snow - melt + refreeze
            new_lw = lw + melt - refreeze
            lw_max = cwh * new_sp
            if new_lw > lw_max:
                snow_outflow = new_lw - lw_max
                new_lw = lw_max
            else:
                snow_outflow = 0.0
            if new_sp < 0.0:
                new_sp = 0.0
            if new_lw < 0.0:
                new_lw = 0.0

            snow_input = p_rain + snow_outflow

            # Soil routine
            if fc <= 0.0 or snow_input <= 0.0:
                recharge = 0.0
            else:
                sr = sm / fc
                if sr < 0.0:
                    sr = 0.0
                elif sr > 1.0:
                    sr = 1.0
                recharge = snow_input * (sr**beta)

            if fc <= 0.0 or lp <= 0.0:
                et_act = 0.0
            else:
                lp_threshold = lp * fc
                if sm >= lp_threshold:
                    et_act = pet
                else:
                    et_act = pet * sm / lp_threshold
                if et_act > sm:
                    et_act = sm
                if et_act < 0.0:
                    et_act = 0.0

            infiltration = snow_input - recharge
            new_sm = sm + infiltration - et_act
            if new_sm < 0.0:
                new_sm = 0.0
            elif new_sm > fc:
                new_sm = fc

            # Update zone state
            state_arr[sp_idx] = new_sp
            state_arr[lw_idx] = new_lw
            state_arr[sm_idx] = new_sm

            # Store per-zone outputs (11 fields)
            zone_outputs_arr[t, zone_idx, 0] = zone_temp
            zone_outputs_arr[t, zone_idx, 1] = zone_precip
            zone_outputs_arr[t, zone_idx, 2] = new_sp
            zone_outputs_arr[t, zone_idx, 3] = new_lw
            zone_outputs_arr[t, zone_idx, 4] = melt
            zone_outputs_arr[t, zone_idx, 5] = snow_input
            zone_outputs_arr[t, zone_idx, 6] = new_sm
            zone_outputs_arr[t, zone_idx, 7] = recharge
            zone_outputs_arr[t, zone_idx, 8] = et_act
            zone_outputs_arr[t, zone_idx, 9] = p_rain
            zone_outputs_arr[t, zone_idx, 10] = p_snow

            # Aggregate (area-weighted)
            agg_p_rain += p_rain * fraction
            agg_p_snow += p_snow * fraction
            agg_melt += melt * fraction
            agg_snow_input += snow_input * fraction
            agg_recharge += recharge * fraction
            agg_et_act += et_act * fraction
            agg_sm += new_sm * fraction
            agg_sp += new_sp * fraction
            agg_lw += new_lw * fraction

        # Response routine (lumped)
        suz = state_arr[suz_idx]
        slz = state_arr[slz_idx]

        if suz > uzl:
            q0 = k0 * (suz - uzl)
        else:
            q0 = 0.0
        q1 = k1 * suz

        if suz > perc_max:
            perc = perc_max
        else:
            perc = suz if suz > 0.0 else 0.0

        new_suz = suz + agg_recharge - q0 - q1 - perc
        if new_suz < 0.0:
            new_suz = 0.0

        q2 = k2 * slz
        new_slz = slz + perc - q2
        if new_slz < 0.0:
            new_slz = 0.0

        qgw = q0 + q1 + q2

        # Routing
        qsim = state_arr[buffer_start]

        for i in range(6):
            state_arr[buffer_start + i] = state_arr[buffer_start + i + 1]
        state_arr[buffer_start + 6] = 0.0

        n_weights = len(uh_weights)
        for i in range(min(n_weights, 7)):
            state_arr[buffer_start + i] += qgw * uh_weights[i]

        state_arr[suz_idx] = new_suz
        state_arr[slz_idx] = new_slz

        # Write aggregated outputs
        outputs_arr[t, 0] = precip
        outputs_arr[t, 1] = temp
        outputs_arr[t, 2] = pet
        outputs_arr[t, 3] = agg_p_rain
        outputs_arr[t, 4] = agg_p_snow
        outputs_arr[t, 5] = agg_sp
        outputs_arr[t, 6] = agg_melt
        outputs_arr[t, 7] = agg_lw
        outputs_arr[t, 8] = agg_snow_input
        outputs_arr[t, 9] = agg_sm
        outputs_arr[t, 10] = agg_recharge
        outputs_arr[t, 11] = agg_et_act
        outputs_arr[t, 12] = new_suz
        outputs_arr[t, 13] = new_slz
        outputs_arr[t, 14] = q0
        outputs_arr[t, 15] = q1
        outputs_arr[t, 16] = q2
        outputs_arr[t, 17] = perc
        outputs_arr[t, 18] = qgw
        outputs_arr[t, 19] = qsim


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    catchment: Catchment | None = None,
    uh_weights: np.ndarray | None = None,
    zone_elevations: np.ndarray | None = None,
    zone_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the HBV-light model.

    Implements the complete HBV-light algorithm:
    1. Snow routine (precipitation partitioning, melt, refreezing)
    2. Soil moisture routine (recharge, evapotranspiration)
    3. Response routine (upper/lower zone outflows)
    4. Routing (triangular unit hydrograph)

    When zone_elevations and zone_fractions are provided with n_zones > 1,
    the model runs in semi-distributed mode with elevation bands.

    Args:
        state: Current model state (stores and routing buffer).
        params: Model parameters.
        precip: Daily precipitation [mm/day].
        pet: Daily potential evapotranspiration [mm/day].
        temp: Daily mean temperature [C].
        catchment: Optional catchment properties (unused, for API compatibility).
        uh_weights: Pre-computed UH weights. If None, computed from params.maxbas.
        zone_elevations: Representative elevation of each zone [m]. Shape (n_zones,).
        zone_fractions: Area fraction of each zone [-]. Shape (n_zones,).
        input_elevation: Elevation of input measurement [m]. Required for extrapolation.
        temp_gradient: Temperature lapse rate [C/100m]. Default is GRAD_T_DEFAULT.
        precip_gradient: Precipitation gradient [m^-1]. Default is GRAD_P_DEFAULT.

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs (area-weighted aggregates)
    """
    # Compute UH weights if not provided
    if uh_weights is None:
        uh_weights = compute_triangular_weights(params.maxbas)

    # Get number of zones from state
    n_zones = state.n_zones

    # Set default zone config if not provided
    if zone_elevations is None:
        zone_elevations = np.zeros(n_zones, dtype=np.float64)
    if zone_fractions is None:
        zone_fractions = np.ones(n_zones, dtype=np.float64) / n_zones

    # Set default gradients
    if temp_gradient is None:
        temp_gradient = GRAD_T_DEFAULT
    if precip_gradient is None:
        precip_gradient = GRAD_P_DEFAULT

    # Check if extrapolation should be skipped
    skip_extrapolation = input_elevation is None

    # Aggregation accumulators
    agg_p_rain = 0.0
    agg_p_snow = 0.0
    agg_melt = 0.0
    agg_snow_input = 0.0
    agg_recharge = 0.0
    agg_et_act = 0.0
    agg_sm = 0.0
    agg_sp = 0.0
    agg_lw = 0.0

    # New zone states array
    new_zone_states = np.zeros((n_zones, 3), dtype=np.float64)

    # Process each zone
    for zone_idx in range(n_zones):
        fraction = zone_fractions[zone_idx]

        # Extrapolate forcing
        if skip_extrapolation:
            zone_temp = temp
            zone_precip = precip
        else:
            zone_temp = extrapolate_temperature(temp, input_elevation, zone_elevations[zone_idx], temp_gradient)
            zone_precip = extrapolate_precipitation(
                precip, input_elevation, zone_elevations[zone_idx], precip_gradient, ELEV_CAP_PRECIP
            )

        # Get current zone state
        sp = state.zone_states[zone_idx, 0]
        lw = state.zone_states[zone_idx, 1]
        sm = state.zone_states[zone_idx, 2]

        # 1. Snow routine
        p_rain, p_snow = partition_precipitation(zone_precip, zone_temp, params.tt, params.sfcf)
        melt = compute_melt(zone_temp, params.tt, params.cfmax, sp)
        refreeze = compute_refreezing(zone_temp, params.tt, params.cfmax, params.cfr, lw)
        new_sp, new_lw, snow_outflow = update_snow_pack(
            sp,
            lw,
            p_snow,
            melt,
            refreeze,
            params.cwh,
        )

        # Total input to soil
        snow_input = p_rain + snow_outflow

        # 2. Soil routine
        recharge = compute_recharge(snow_input, sm, params.fc, params.beta)
        et_act = compute_actual_et(pet, sm, params.fc, params.lp)
        new_sm = update_soil_moisture(sm, snow_input, recharge, et_act, params.fc)

        # Store new zone state
        new_zone_states[zone_idx, 0] = new_sp
        new_zone_states[zone_idx, 1] = new_lw
        new_zone_states[zone_idx, 2] = new_sm

        # Aggregate (area-weighted)
        agg_p_rain += p_rain * fraction
        agg_p_snow += p_snow * fraction
        agg_melt += melt * fraction
        agg_snow_input += snow_input * fraction
        agg_recharge += recharge * fraction
        agg_et_act += et_act * fraction
        agg_sm += new_sm * fraction
        agg_sp += new_sp * fraction
        agg_lw += new_lw * fraction

    # 3. Response routine (lumped, using aggregated recharge)
    q0, q1 = upper_zone_outflows(state.upper_zone, params.k0, params.k1, params.uzl)
    perc = compute_percolation(state.upper_zone, params.perc)
    new_suz = update_upper_zone(state.upper_zone, agg_recharge, q0, q1, perc)

    q2 = lower_zone_outflow(state.lower_zone, params.k2)
    new_slz = update_lower_zone(state.lower_zone, perc, q2)

    qgw = q0 + q1 + q2

    # 4. Routing
    new_buffer, qsim = convolve_triangular(qgw, state.routing_buffer, uh_weights)

    # Build new state
    new_state = State(
        zone_states=new_zone_states,
        upper_zone=new_suz,
        lower_zone=new_slz,
        routing_buffer=new_buffer,
    )

    # Build fluxes dictionary (area-weighted aggregates)
    fluxes: dict[str, float] = {
        "precip": precip,
        "temp": temp,
        "pet": pet,
        "precip_rain": agg_p_rain,
        "precip_snow": agg_p_snow,
        "snow_pack": agg_sp,
        "snow_melt": agg_melt,
        "liquid_water_in_snow": agg_lw,
        "snow_input": agg_snow_input,
        "soil_moisture": agg_sm,
        "recharge": agg_recharge,
        "actual_et": agg_et_act,
        "upper_zone": new_suz,
        "lower_zone": new_slz,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "percolation": perc,
        "qgw": qgw,
        "streamflow": qsim,
    }

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
    catchment: Catchment | None = None,
) -> ModelOutput[HBVLightFluxes]:
    """Run the HBV-light model over a timeseries.

    Executes the HBV-light model for each timestep in the input forcing data,
    returning a ModelOutput with all model outputs.

    When a Catchment with n_layers > 1 is provided, the model runs in semi-distributed
    mode with elevation bands. Temperature and precipitation are extrapolated to each
    zone, snow and soil routines run independently per zone, and recharge is aggregated
    before the response routine.

    Args:
        params: Model parameters.
        forcing: Input forcing data with precip, pet, and temp arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).
        catchment: Optional catchment properties for elevation band support.
            If provided with n_layers > 1 and hypsometric_curve, runs in multi-zone mode.

    Returns:
        ModelOutput containing HBVLightFluxes outputs.
        For multi-zone runs, zone_outputs contains per-zone details.
        Access streamflow via result.streamflow or result.fluxes.streamflow.
        Convert to DataFrame via result.to_dataframe().

    Raises:
        ValueError: If forcing.temp is None.
    """
    # Validate temperature is provided
    if forcing.temp is None:
        raise ValueError("HBV-light requires temperature data (forcing.temp)")

    # Validate resolution
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"HBV-light supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    # Determine zone configuration
    if catchment is not None and catchment.n_layers > 1 and catchment.hypsometric_curve is not None:
        n_zones = catchment.n_layers
        zone_elevations, zone_fractions = derive_layers(catchment.hypsometric_curve, n_zones)
        input_elevation = catchment.input_elevation if catchment.input_elevation is not None else float("nan")
        temp_gradient = catchment.temp_gradient if catchment.temp_gradient is not None else GRAD_T_DEFAULT
        precip_gradient = catchment.precip_gradient if catchment.precip_gradient is not None else GRAD_P_DEFAULT
    else:
        n_zones = 1
        zone_elevations = np.zeros(1, dtype=np.float64)
        zone_fractions = np.ones(1, dtype=np.float64)
        input_elevation = float("nan")  # Skip extrapolation
        temp_gradient = GRAD_T_DEFAULT
        precip_gradient = GRAD_P_DEFAULT

    # Initialize state if not provided
    state = State.initialize(params, n_zones) if initial_state is None else initial_state

    # Compute UH weights once
    uh_weights = compute_triangular_weights(params.maxbas)

    # Initialize output arrays
    n_timesteps = len(forcing)

    # Convert to arrays
    state_arr = np.asarray(state)
    params_arr = np.asarray(params)

    # Allocate output arrays
    outputs_arr = np.zeros((n_timesteps, 20), dtype=np.float64)
    zone_outputs_arr = np.zeros((n_timesteps, n_zones, 11), dtype=np.float64)

    # Run the multi-zone Numba kernel
    _run_numba_multizone(
        state_arr,
        params_arr,
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        forcing.temp.astype(np.float64),
        uh_weights,
        n_zones,
        zone_elevations.astype(np.float64),
        zone_fractions.astype(np.float64),
        input_elevation,
        temp_gradient,
        precip_gradient,
        ELEV_CAP_PRECIP,
        outputs_arr,
        zone_outputs_arr,
    )

    # Build output object from array
    hbv_fluxes = HBVLightFluxes(
        precip=outputs_arr[:, 0],
        temp=outputs_arr[:, 1],
        pet=outputs_arr[:, 2],
        precip_rain=outputs_arr[:, 3],
        precip_snow=outputs_arr[:, 4],
        snow_pack=outputs_arr[:, 5],
        snow_melt=outputs_arr[:, 6],
        liquid_water_in_snow=outputs_arr[:, 7],
        snow_input=outputs_arr[:, 8],
        soil_moisture=outputs_arr[:, 9],
        recharge=outputs_arr[:, 10],
        actual_et=outputs_arr[:, 11],
        upper_zone=outputs_arr[:, 12],
        lower_zone=outputs_arr[:, 13],
        q0=outputs_arr[:, 14],
        q1=outputs_arr[:, 15],
        q2=outputs_arr[:, 16],
        percolation=outputs_arr[:, 17],
        qgw=outputs_arr[:, 18],
        streamflow=outputs_arr[:, 19],
    )

    # Build per-zone outputs if multi-zone
    zone_outputs: HBVLightZoneOutputs | None = None
    if n_zones > 1:
        zone_outputs = HBVLightZoneOutputs(
            zone_elevations=zone_elevations,
            zone_fractions=zone_fractions,
            zone_temp=zone_outputs_arr[:, :, 0],
            zone_precip=zone_outputs_arr[:, :, 1],
            snow_pack=zone_outputs_arr[:, :, 2],
            liquid_water_in_snow=zone_outputs_arr[:, :, 3],
            snow_melt=zone_outputs_arr[:, :, 4],
            snow_input=zone_outputs_arr[:, :, 5],
            soil_moisture=zone_outputs_arr[:, :, 6],
            recharge=zone_outputs_arr[:, :, 7],
            actual_et=zone_outputs_arr[:, :, 8],
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=hbv_fluxes,
        snow=None,
        snow_layers=None,
        zone_outputs=zone_outputs,
    )
