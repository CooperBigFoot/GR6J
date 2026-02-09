"""CemaNeige snow module process functions.

Pure functions implementing the mathematical equations for each CemaNeige model component.
All inputs and outputs are floats. These functions correspond to sections in CEMANEIGE.md.
"""

_T_MELT: float = 0.0  # Melt threshold [°C]
_MIN_SPEED: float = 0.1  # Minimum melt fraction [-]



def compute_solid_fraction(temp: float) -> float:
    """Compute fraction of precipitation falling as snow using USACE formula.

    Implements Section 5.1 of CEMANEIGE.md. Uses linear interpolation
    between T_SNOW (-1C) and T_RAIN (3C) thresholds.

    Args:
        temp: Air temperature [°C].

    Returns:
        Fraction of precipitation that is solid [0, 1].
        - 1.0 when temp <= -1C (all snow)
        - 0.0 when temp >= 3C (all rain)
        - Linear interpolation between
    """
    if temp <= -1.0:
        return 1.0
    if temp >= 3.0:
        return 0.0
    return (3.0 - temp) / 4.0



def partition_precipitation(precip: float, solid_fraction: float) -> tuple[float, float]:
    """Split precipitation into liquid and solid components.

    Implements Section 5.1 of CEMANEIGE.md. Partitions total precipitation
    based on the solid fraction determined from temperature.

    Args:
        precip: Total precipitation [mm/day].
        solid_fraction: Fraction of precipitation that is solid [0, 1].

    Returns:
        Tuple of (pliq, psol):
        - pliq: Liquid precipitation (rain) [mm/day]
        - psol: Solid precipitation (snow) [mm/day]
    """
    pliq = (1.0 - solid_fraction) * precip
    psol = solid_fraction * precip
    return pliq, psol



def update_thermal_state(etg: float, temp: float, ctg: float) -> float:
    """Update snow pack thermal state using exponential smoothing.

    Implements Section 5.3 of CEMANEIGE.md. The thermal state represents
    the weighted temperature history of the snow pack, controlling when
    melt can occur.

    The thermal state is capped at 0C because snow cannot be warmer than
    the melting point. This ensures melt only occurs when the snow pack
    has absorbed enough heat to reach 0C.

    Args:
        etg: Current thermal state [°C]. Must be <= 0.
        temp: Air temperature [°C].
        ctg: Thermal state weighting coefficient (CTG parameter) [-].
            - CTG near 1: High thermal inertia, slow response
            - CTG near 0: Low thermal inertia, tracks air temperature

    Returns:
        Updated thermal state [°C], capped at 0.
    """
    new_etg = ctg * etg + (1.0 - ctg) * temp
    return min(new_etg, 0.0)



def compute_potential_melt(etg: float, temp: float, kf: float, snow_pack: float) -> float:
    """Compute potential snow melt using degree-day method.

    Implements Section 5.4 of CEMANEIGE.md. Melt only occurs when two
    conditions are met simultaneously:
    1. The snow pack thermal state has reached 0C (etg == 0)
    2. Air temperature exceeds the melt threshold (temp > T_MELT)

    The physical interpretation is that the snow pack must first absorb
    enough heat to reach the melting point before melt can begin.

    Args:
        etg: Thermal state [°C]. Melt requires etg == 0.
        temp: Air temperature [°C].
        kf: Degree-day melt factor (Kf parameter) [mm/°C/day].
        snow_pack: Current snow pack water equivalent [mm].

    Returns:
        Potential melt [mm/day], capped at available snow pack.
        Returns 0 if melt conditions are not met.
    """
    if etg == 0.0 and temp > _T_MELT:
        pot_melt = kf * temp
        return min(pot_melt, snow_pack)
    return 0.0



def compute_gratio(snow_pack: float, gthreshold: float) -> float:
    """Compute snow cover fraction (Gratio).

    Implements Section 5.5 of CEMANEIGE.md. The Gratio represents the
    effective snow cover fraction of the catchment, which modulates
    the melt rate.

    When the snow pack equals or exceeds the threshold, the catchment
    is considered fully snow-covered. Below the threshold, the snow
    cover fraction decreases linearly.

    Args:
        snow_pack: Current snow pack water equivalent [mm].
        gthreshold: Snow pack threshold for full coverage [mm].
            Typically 0.9 * MeanAnnualSolidPrecip.

    Returns:
        Snow cover fraction [0, 1].
        - 1.0 when snow_pack >= gthreshold
        - snow_pack / gthreshold when snow_pack < gthreshold
        - 0.0 if gthreshold is 0 (edge case handling)
    """
    if gthreshold == 0.0:
        return 0.0
    if snow_pack >= gthreshold:
        return 1.0
    return snow_pack / gthreshold



def compute_actual_melt(potential_melt: float, gratio: float) -> float:
    """Compute actual snow melt modulated by snow cover fraction.

    Implements Section 5.6 of CEMANEIGE.md. The actual melt is scaled
    by the snow cover fraction (Gratio), with a minimum melt efficiency
    of MIN_SPEED (10%) even when snow cover is minimal.

    Physical interpretation:
    - When Gratio = 1 (full cover): Melt = 100% of potential
    - When Gratio = 0 (nearly depleted): Melt = MIN_SPEED (10%) of potential
    - The MIN_SPEED ensures melt never completely stops while snow exists

    Args:
        potential_melt: Potential melt from degree-day calculation [mm/day].
        gratio: Snow cover fraction [0, 1].

    Returns:
        Actual melt [mm/day].
    """
    melt = ((1.0 - _MIN_SPEED) * gratio + _MIN_SPEED) * potential_melt
    return melt
