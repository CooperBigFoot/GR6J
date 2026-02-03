"""CemaNeige layer utility functions (re-exported from pydrology.utils.elevation).

This module provides backward compatibility. New code should import from
pydrology.utils.elevation directly.
"""

from pydrology.utils.elevation import (
    ELEV_CAP_PRECIP,
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
    extrapolate_precipitation,
    extrapolate_temperature,
)

__all__ = [
    "ELEV_CAP_PRECIP",
    "GRAD_P_DEFAULT",
    "GRAD_T_DEFAULT",
    "derive_layers",
    "extrapolate_precipitation",
    "extrapolate_temperature",
]
