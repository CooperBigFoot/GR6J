"""GR2M model subpackage.

Public API for the GR2M hydrological model.
"""

from .constants import DEFAULT_BOUNDS, PARAM_NAMES, STATE_SIZE, SUPPORTED_RESOLUTIONS
from .outputs import GR2MFluxes
from .run import _run_numba, _step_numba, run, step
from .types import Parameters, State

__all__ = [
    "DEFAULT_BOUNDS",
    "GR2MFluxes",
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
import pydrology.models.gr2m as _self
from pydrology.registry import register

register("gr2m", _self)
