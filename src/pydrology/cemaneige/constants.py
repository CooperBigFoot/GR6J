"""CemaNeige numerical constants.

Fixed values for the CemaNeige snow accumulation and melt model.
Values derived from CEMANEIGE.md technical definition.
"""

T_MELT: float = 0.0  # Melt threshold [°C]
MIN_SPEED: float = 0.1  # Minimum melt fraction [-]
T_SNOW: float = -1.0  # All snow threshold [°C] for USACE formula
T_RAIN: float = 3.0  # All rain threshold [°C] for USACE formula
GTHRESHOLD_FACTOR: float = 0.9  # Fraction of mean annual solid precip for standard mode

GRAD_T_DEFAULT: float = 0.6  # Default temperature lapse rate [°C/100m]
GRAD_P_DEFAULT: float = 0.00041  # Default precipitation gradient [m⁻¹]
ELEV_CAP_PRECIP: float = 4000.0  # Maximum elevation for precipitation extrapolation [m]
