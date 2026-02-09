"""GR6J-CemaNeige coupled model subpackage.

Public API for the coupled GR6J hydrological model with CemaNeige snow module.
Provides an 8-parameter model combining rainfall-runoff (GR6J) with snow
accumulation and melt (CemaNeige).

Design: Unified multi-layer code path. If catchment.input_elevation is None,
extrapolation is skipped (single-layer behavior). One code path handles all cases.
"""

from .constants import (
    DEFAULT_BOUNDS,
    PARAM_NAMES,
    SNOW_LAYER_STATE_SIZE,
    STATE_SIZE_BASE,
    SUPPORTED_RESOLUTIONS,
    compute_state_size,
)
from .outputs import GR6JCemaNeigeFluxes
from .run import _run_unified_numba, run, step
from .types import Parameters, State

# STATE_SIZE for the default single-layer case (required by registry)
STATE_SIZE: int = compute_state_size(n_layers=1)

__all__ = [
    "DEFAULT_BOUNDS",
    "GR6JCemaNeigeFluxes",
    "PARAM_NAMES",
    "Parameters",
    "SNOW_LAYER_STATE_SIZE",
    "STATE_SIZE",
    "STATE_SIZE_BASE",
    "SUPPORTED_RESOLUTIONS",
    "State",
    "_run_unified_numba",
    "compute_state_size",
    "run",
    "step",
]

# Auto-register with the model registry
import pydrology.models.gr6j_cemaneige as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("gr6j_cemaneige", _self)
