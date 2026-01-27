"""CemaNeige numerical constants.

Fixed values for the CemaNeige snow accumulation and melt model.
Values derived from CEMANEIGE.md technical definition.
"""

T_MELT: float = 0.0  # Melt threshold [°C]
MIN_SPEED: float = 0.1  # Minimum melt fraction [-]
T_SNOW: float = -1.0  # All snow threshold [°C] for USACE formula
T_RAIN: float = 3.0  # All rain threshold [°C] for USACE formula
GTHRESHOLD_FACTOR: float = 0.9  # Fraction of mean annual solid precip for standard mode
