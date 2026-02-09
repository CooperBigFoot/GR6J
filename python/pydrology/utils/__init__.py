"""Utility modules for pydrology."""

from pydrology.utils.data_interface import CaravanDataSource
from pydrology.utils.dem import DEMStatistics, analyze_dem
from pydrology.utils.elevation import (
    ELEV_CAP_PRECIP,
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
    extrapolate_precipitation,
    extrapolate_temperature,
)
from pydrology.utils.precipitation import (
    compute_mean_annual_solid_precip,
    compute_solid_fraction,
    compute_solid_precip,
)

__all__ = [
    "CaravanDataSource",
    "DEMStatistics",
    "ELEV_CAP_PRECIP",
    "GRAD_P_DEFAULT",
    "GRAD_T_DEFAULT",
    "analyze_dem",
    "compute_mean_annual_solid_precip",
    "compute_solid_fraction",
    "compute_solid_precip",
    "derive_layers",
    "extrapolate_precipitation",
    "extrapolate_temperature",
]
