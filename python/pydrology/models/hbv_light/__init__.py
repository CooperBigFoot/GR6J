"""HBV-light model subpackage.

Public API for the HBV-light hydrological model.
"""

from .constants import DEFAULT_BOUNDS, PARAM_NAMES, STATE_SIZE, SUPPORTED_RESOLUTIONS
from .outputs import HBVLightFluxes, HBVLightZoneOutputs
from .run import _run_numba, _step_numba, run, step
from .types import Parameters, State

__all__ = [
    "DEFAULT_BOUNDS",
    "HBVLightFluxes",
    "HBVLightZoneOutputs",
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
import pydrology.models.hbv_light as _self
from pydrology.registry import register

register("hbv_light", _self)
