"""CemaNeige snow model orchestration functions.

This module provides the main entry points for running the CemaNeige snow model:
- cemaneige_step(): Execute a single timestep for single-layer mode
- cemaneige_multi_layer_step(): Execute a single timestep for multi-layer mode
"""

from __future__ import annotations

import numpy as np

from .layers import extrapolate_precipitation, extrapolate_temperature
from .processes import (
    compute_actual_melt,
    compute_gratio,
    compute_potential_melt,
    compute_solid_fraction,
    partition_precipitation,
    update_thermal_state,
)
from .types import CemaNeige, CemaNeigeMultiLayerState, CemaNeigeSingleLayerState


def cemaneige_step(
    state: CemaNeigeSingleLayerState,
    params: CemaNeige,
    precip: float,
    temp: float,
) -> tuple[CemaNeigeSingleLayerState, dict[str, float]]:
    """Execute one timestep of the CemaNeige snow model.

    Implements the complete CemaNeige algorithm following Section 7 of CEMANEIGE.md.

    Algorithm steps:
    1. Compute solid_fraction from temperature (USACE formula)
    2. Partition precipitation into liquid (pliq) and solid (psol)
    3. Accumulate snow: g = g + psol
    4. Update thermal state (exponential smoothing, cap at 0)
    5. Compute potential melt (degree-day, when etg=0 AND temp>0)
    6. Compute gratio before melt
    7. Compute actual melt (modulated by gratio + MIN_SPEED floor)
    8. Update snow pack: g = g - melt
    9. Compute output gratio (after melt)
    10. Compute total liquid output: pliq_and_melt = pliq + melt

    Args:
        state: Current CemaNeige state (g, etg, gthreshold, glocalmax).
        params: CemaNeige parameters (ctg, kf, mean_annual_solid_precip).
        precip: Total precipitation [mm/day].
        temp: Air temperature [°C].

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated CemaNeigeSingleLayerState after timestep
        - fluxes: Dictionary containing all 11 model outputs (see Section 13
          of CEMANEIGE.md):
            - snow_pliq: Liquid precipitation [mm/day]
            - snow_psol: Solid precipitation [mm/day]
            - snow_pack: Snow pack water equivalent after melt [mm]
            - snow_thermal_state: Thermal state [°C]
            - snow_gratio: Snow cover fraction after melt [-]
            - snow_pot_melt: Potential melt [mm/day]
            - snow_melt: Actual melt [mm/day]
            - snow_pliq_and_melt: Total liquid output to GR6J [mm/day]
            - snow_temp: Air temperature [°C]
            - snow_gthreshold: Melt threshold [mm]
            - snow_glocalmax: Local maximum for hysteresis [mm]
    """
    # 1. Compute solid fraction from temperature
    solid_fraction = compute_solid_fraction(temp)

    # 2. Partition precipitation
    pliq, psol = partition_precipitation(precip, solid_fraction)

    # 3. Accumulate snow
    g = state.g + psol

    # 4. Update thermal state
    etg = update_thermal_state(state.etg, temp, params.ctg)

    # 5. Compute potential melt
    pot_melt = compute_potential_melt(etg, temp, params.kf, g)

    # 6. Compute gratio before melt (for melt calculation)
    gratio_for_melt = compute_gratio(g, state.gthreshold)

    # 7. Compute actual melt
    melt = compute_actual_melt(pot_melt, gratio_for_melt)

    # 8. Update snow pack
    g = g - melt

    # 9. Compute output gratio (after melt)
    gratio_output = compute_gratio(g, state.gthreshold)

    # 10. Compute total liquid output
    pliq_and_melt = pliq + melt

    # Build new state (keep gthreshold and glocalmax unchanged for standard mode)
    new_state = CemaNeigeSingleLayerState(
        g=g,
        etg=etg,
        gthreshold=state.gthreshold,
        glocalmax=state.glocalmax,
    )

    # Build fluxes dictionary (11 keys with snow_ prefix)
    fluxes: dict[str, float] = {
        "snow_pliq": pliq,
        "snow_psol": psol,
        "snow_pack": g,
        "snow_thermal_state": etg,
        "snow_gratio": gratio_output,
        "snow_pot_melt": pot_melt,
        "snow_melt": melt,
        "snow_pliq_and_melt": pliq_and_melt,
        "snow_temp": temp,
        "snow_gthreshold": state.gthreshold,
        "snow_glocalmax": state.glocalmax,
    }

    return new_state, fluxes


def _aggregate_layer_fluxes(
    all_layer_fluxes: list[dict[str, float]],
    layer_fractions: list[float],
) -> dict[str, float]:
    """Compute area-weighted average of per-layer fluxes.

    Args:
        all_layer_fluxes: List of flux dictionaries, one per layer.
        layer_fractions: Area fraction of each layer [-].

    Returns:
        Single dictionary with area-weighted average fluxes.
    """
    keys = all_layer_fluxes[0].keys()
    aggregated: dict[str, float] = {}
    for key in keys:
        aggregated[key] = sum(
            fluxes[key] * fraction for fluxes, fraction in zip(all_layer_fluxes, layer_fractions, strict=True)
        )
    return aggregated


def cemaneige_multi_layer_step(
    state: CemaNeigeMultiLayerState,
    params: CemaNeige,
    precip: float,
    temp: float,
    layer_elevations: list[float] | np.ndarray,
    layer_fractions: list[float] | np.ndarray,
    input_elevation: float,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
) -> tuple[CemaNeigeMultiLayerState, dict[str, float], list[dict[str, float]]]:
    """Execute one timestep of multi-layer CemaNeige.

    Runs the CemaNeige snow model independently on each elevation band,
    extrapolating temperature and precipitation to each layer's elevation,
    then aggregates the results using area-weighted averaging.

    Args:
        state: Current multi-layer state.
        params: CemaNeige parameters (ctg, kf).
        precip: Total precipitation at input elevation [mm/day].
        temp: Air temperature at input elevation [°C].
        layer_elevations: Representative elevation of each layer [m].
        layer_fractions: Area fraction of each layer [-].
        input_elevation: Elevation of the input forcing data [m].
        temp_gradient: Temperature lapse rate [°C/100m]. If None, uses default.
        precip_gradient: Precipitation gradient [m⁻¹]. If None, uses default.

    Returns:
        Tuple of (new_state, aggregated_fluxes, per_layer_fluxes):
        - new_state: Updated CemaNeigeMultiLayerState
        - aggregated_fluxes: Area-weighted average fluxes (same keys as cemaneige_step)
        - per_layer_fluxes: List of flux dicts, one per layer
    """
    n_layers = len(state)
    new_layer_states: list[CemaNeigeSingleLayerState] = []
    per_layer_fluxes: list[dict[str, float]] = []

    for i in range(n_layers):
        # Extrapolate temperature to this layer's elevation
        layer_temp_kwargs: dict[str, float] = {}
        if temp_gradient is not None:
            layer_temp_kwargs["gradient"] = temp_gradient
        layer_temp = extrapolate_temperature(temp, input_elevation, float(layer_elevations[i]), **layer_temp_kwargs)

        # Extrapolate precipitation to this layer's elevation
        layer_precip_kwargs: dict[str, float] = {}
        if precip_gradient is not None:
            layer_precip_kwargs["gradient"] = precip_gradient
        layer_precip = extrapolate_precipitation(
            precip, input_elevation, float(layer_elevations[i]), **layer_precip_kwargs
        )

        # Run single-layer step for this band
        new_layer_state, layer_fluxes = cemaneige_step(
            state=state[i],
            params=params,
            precip=layer_precip,
            temp=layer_temp,
        )

        new_layer_states.append(new_layer_state)
        per_layer_fluxes.append(layer_fluxes)

    # Aggregate fluxes across layers
    fractions = [float(f) for f in layer_fractions]
    aggregated_fluxes = _aggregate_layer_fluxes(per_layer_fluxes, fractions)

    new_state = CemaNeigeMultiLayerState(layer_states=new_layer_states)

    return new_state, aggregated_fluxes, per_layer_fluxes
