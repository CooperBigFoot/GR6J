"""GR6J model subpackage.

Public API for the GR6J hydrological model.
"""

from .constants import DEFAULT_BOUNDS, PARAM_NAMES, STATE_SIZE, SUPPORTED_RESOLUTIONS
from .outputs import GR6JFluxes, GR6JOutput
from .run import _run_numba, _step_numba, run, step
from .types import Parameters, State

__all__ = [
    "DEFAULT_BOUNDS",
    "GR6JFluxes",
    "GR6JOutput",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "State",
    "SUPPORTED_RESOLUTIONS",
    "_run_numba",
    "_step_numba",
    "run",
    "step",
]

# Auto-register with the model registry
import pydrology.models.gr6j as _self
from pydrology.registry import register

register("gr6j", _self)
