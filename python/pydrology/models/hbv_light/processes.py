"""HBV-light snow and soil moisture process functions.

Numba-compiled functions implementing the snow routine (precipitation partitioning,
accumulation, melt, refreezing) and soil moisture routine (recharge, evapotranspiration).
"""

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401

from numba import njit


@njit(cache=True)
def partition_precipitation(precip: float, temp: float, tt: float, sfcf: float) -> tuple[float, float]:
    """Partition precipitation into rain and snow.

    Args:
        precip: Total precipitation [mm/day].
        temp: Air temperature [deg C].
        tt: Threshold temperature [deg C].
        sfcf: Snowfall correction factor [-].

    Returns:
        Tuple of (p_rain, p_snow) in mm/day.
    """
    if temp > tt:
        return precip, 0.0
    else:
        return 0.0, sfcf * precip


@njit(cache=True)
def compute_melt(temp: float, tt: float, cfmax: float, snow_pack: float) -> float:
    """Compute snowmelt using degree-day method.

    Args:
        temp: Air temperature [deg C].
        tt: Threshold temperature [deg C].
        cfmax: Degree-day factor [mm/deg C/day].
        snow_pack: Current snow pack [mm].

    Returns:
        Melt rate [mm/day], limited by available snow.
    """
    if temp > tt:
        melt = cfmax * (temp - tt)
        return min(melt, snow_pack)
    return 0.0


@njit(cache=True)
def compute_refreezing(temp: float, tt: float, cfmax: float, cfr: float, liquid_water: float) -> float:
    """Compute refreezing of liquid water in snowpack.

    Args:
        temp: Air temperature [deg C].
        tt: Threshold temperature [deg C].
        cfmax: Degree-day factor [mm/deg C/day].
        cfr: Refreezing coefficient [-].
        liquid_water: Liquid water in snowpack [mm].

    Returns:
        Refreezing rate [mm/day], limited by available liquid water.
    """
    if temp < tt:
        refreeze = cfr * cfmax * (tt - temp)
        return min(refreeze, liquid_water)
    return 0.0


@njit(cache=True)
def update_snow_pack(
    sp: float,
    lw: float,
    p_snow: float,
    melt: float,
    refreeze: float,
    cwh: float,
) -> tuple[float, float, float]:
    """Update snow pack and liquid water states.

    Args:
        sp: Current snow pack [mm].
        lw: Current liquid water in snow [mm].
        p_snow: Snowfall [mm/day].
        melt: Snowmelt [mm/day].
        refreeze: Refreezing [mm/day].
        cwh: Water holding capacity of snow [-].

    Returns:
        Tuple of (new_sp, new_lw, outflow):
        - new_sp: Updated snow pack [mm]
        - new_lw: Updated liquid water in snow [mm]
        - outflow: Water released from snow to soil [mm/day]
    """
    # Update snow pack: add snowfall, subtract melt, add refrozen water
    new_sp = sp + p_snow - melt + refreeze

    # Update liquid water: add melt, subtract refreezing
    new_lw = lw + melt - refreeze

    # Compute maximum liquid water storage
    lw_max = cwh * new_sp

    # Compute outflow (excess liquid water)
    if new_lw > lw_max:
        outflow = new_lw - lw_max
        new_lw = lw_max
    else:
        outflow = 0.0

    # Ensure non-negative
    new_sp = max(0.0, new_sp)
    new_lw = max(0.0, new_lw)

    return new_sp, new_lw, outflow


@njit(cache=True)
def compute_recharge(soil_input: float, sm: float, fc: float, beta: float) -> float:
    """Compute groundwater recharge from soil.

    Uses the HBV soil moisture / runoff relationship:
    Recharge/Input = (SM/FC)^BETA

    Args:
        soil_input: Water input to soil (rain + snowmelt) [mm/day].
        sm: Current soil moisture [mm].
        fc: Field capacity [mm].
        beta: Shape coefficient [-].

    Returns:
        Recharge to groundwater [mm/day].
    """
    if fc <= 0.0 or soil_input <= 0.0:
        return 0.0

    # Saturation ratio, clamped to [0, 1]
    sr = min(max(sm / fc, 0.0), 1.0)

    # Recharge fraction
    recharge = soil_input * (sr**beta)

    return recharge


@njit(cache=True)
def compute_actual_et(pet: float, sm: float, fc: float, lp: float) -> float:
    """Compute actual evapotranspiration from soil.

    ET is reduced linearly when SM < LP * FC.

    Args:
        pet: Potential evapotranspiration [mm/day].
        sm: Current soil moisture [mm].
        fc: Field capacity [mm].
        lp: Limit for potential ET as fraction of FC [-].

    Returns:
        Actual evapotranspiration [mm/day].
    """
    if fc <= 0.0 or lp <= 0.0:
        return 0.0

    lp_threshold = lp * fc

    # Linear reduction below threshold, full PET above
    et_act = pet if sm >= lp_threshold else pet * sm / lp_threshold

    # Cannot extract more than available
    et_act = min(et_act, max(sm, 0.0))

    return et_act


@njit(cache=True)
def update_soil_moisture(
    sm: float,
    soil_input: float,
    recharge: float,
    et_act: float,
    fc: float,
) -> float:
    """Update soil moisture state.

    Args:
        sm: Current soil moisture [mm].
        soil_input: Water input to soil [mm/day].
        recharge: Recharge leaving to groundwater [mm/day].
        et_act: Actual evapotranspiration [mm/day].
        fc: Field capacity [mm].

    Returns:
        Updated soil moisture [mm], bounded [0, FC].
    """
    # Infiltration is the non-recharge portion
    infiltration = soil_input - recharge

    # Update soil moisture
    new_sm = sm + infiltration - et_act

    # Bound to [0, FC]
    new_sm = max(0.0, min(new_sm, fc))

    return new_sm
