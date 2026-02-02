"""Utility functions for computing solid precipitation using the USACE formula."""

import numpy as np

from pydrology.cemaneige.constants import T_RAIN, T_SNOW

DAYS_PER_YEAR = 365.25


def compute_solid_fraction(
    temp: np.ndarray,
    t_snow: float = T_SNOW,
    t_rain: float = T_RAIN,
) -> np.ndarray:
    """Compute solid precipitation fraction using USACE formula.

    The fraction ranges from 1.0 (all snow) when temp <= t_snow to
    0.0 (all rain) when temp >= t_rain, with linear interpolation between.

    Args:
        temp: Temperature array [°C].
        t_snow: Temperature at or below which all precipitation is snow [°C].
        t_rain: Temperature at or above which all precipitation is rain [°C].

    Returns:
        Solid fraction array with values in [0, 1].

    Raises:
        ValueError: If t_snow >= t_rain.
    """
    if t_snow >= t_rain:
        raise ValueError("t_snow must be less than t_rain")

    fraction = (t_rain - temp) / (t_rain - t_snow)
    return np.clip(fraction, 0.0, 1.0)


def compute_solid_precip(
    precip: np.ndarray,
    temp: np.ndarray,
    t_snow: float = T_SNOW,
    t_rain: float = T_RAIN,
) -> np.ndarray:
    """Compute solid precipitation from total precipitation and temperature.

    Args:
        precip: Total precipitation array [mm/day].
        temp: Temperature array [°C].
        t_snow: Temperature at or below which all precipitation is snow [°C].
        t_rain: Temperature at or above which all precipitation is rain [°C].

    Returns:
        Solid precipitation array [mm/day].

    Raises:
        ValueError: If t_snow >= t_rain.
        ValueError: If precip and temp have different shapes.
    """
    if precip.shape != temp.shape:
        raise ValueError(f"precip shape {precip.shape} does not match temp shape {temp.shape}")

    solid_fraction = compute_solid_fraction(temp, t_snow, t_rain)
    return precip * solid_fraction


def compute_mean_annual_solid_precip(
    precip: np.ndarray,
    temp: np.ndarray,
    t_snow: float = T_SNOW,
    t_rain: float = T_RAIN,
) -> float:
    """Compute mean annual solid precipitation.

    Calculates the mean daily solid precipitation and converts to annual.

    Args:
        precip: Daily total precipitation array [mm/day].
        temp: Daily temperature array [°C].
        t_snow: Temperature at or below which all precipitation is snow [°C].
        t_rain: Temperature at or above which all precipitation is rain [°C].

    Returns:
        Mean annual solid precipitation [mm/year].

    Raises:
        ValueError: If t_snow >= t_rain.
        ValueError: If precip and temp have different shapes.
        ValueError: If arrays are empty.
    """
    if precip.size == 0:
        raise ValueError("Arrays must not be empty")

    solid_precip = compute_solid_precip(precip, temp, t_snow, t_rain)
    return float(np.mean(solid_precip) * DAYS_PER_YEAR)
