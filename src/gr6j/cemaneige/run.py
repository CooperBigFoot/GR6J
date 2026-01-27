"""CemaNeige snow model orchestration functions.

This module provides the main entry point for running the CemaNeige snow model:
- cemaneige_step(): Execute a single timestep
"""

from .processes import (
    compute_actual_melt,
    compute_gratio,
    compute_potential_melt,
    compute_solid_fraction,
    partition_precipitation,
    update_thermal_state,
)
from .types import CemaNeige, CemaNeigeSingleLayerState


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
