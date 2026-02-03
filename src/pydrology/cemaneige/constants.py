"""CemaNeige numerical constants.

Fixed values for the CemaNeige snow accumulation and melt model.
Values derived from CEMANEIGE.md technical definition.
"""

# CemaNeige-specific constants
T_MELT: float = 0.0  # Melt threshold [°C]
MIN_SPEED: float = 0.1  # Minimum melt fraction [-]
T_SNOW: float = -1.0  # All snow threshold [°C] for USACE formula
T_RAIN: float = 3.0  # All rain threshold [°C] for USACE formula
GTHRESHOLD_FACTOR: float = 0.9  # Fraction of mean annual solid precip for standard mode

# Re-export shared elevation constants for backward compatibility
from pydrology.utils.elevation import (  # noqa: E402
    ELEV_CAP_PRECIP,
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
)

__all__ = [
    "T_MELT",
    "MIN_SPEED",
    "T_SNOW",
    "T_RAIN",
    "GTHRESHOLD_FACTOR",
    "ELEV_CAP_PRECIP",
    "GRAD_P_DEFAULT",
    "GRAD_T_DEFAULT",
]
