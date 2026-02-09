"""PyDrology hydrological modeling package.

A collection of lumped conceptual rainfall-runoff models for daily streamflow simulation.
Includes GR6J (Génie Rural à 6 paramètres Journalier) and optional CemaNeige snow module.
"""

import pydrology.models.gr6j_cemaneige  # noqa: F401 - triggers auto-registration
import pydrology.models.hbv_light  # noqa: F401 - triggers auto-registration
import pydrology.models.gr2m  # noqa: F401 - triggers auto-registration
from pydrology.calibration import ObservedData, Solution, calibrate, list_metrics
from pydrology.cemaneige import (
    CemaNeige,
    CemaNeigeMultiLayerState,
    CemaNeigeSingleLayerState,
    cemaneige_multi_layer_step,
    cemaneige_step,
)
from pydrology.models.gr6j import Parameters, State, run, step
from pydrology.outputs import GR6JFluxes, GR6JOutput, ModelOutput, SnowLayerOutputs, SnowOutput
from pydrology.registry import get_model, get_model_info, list_models
from pydrology.types import Catchment, ForcingData, Resolution

__all__ = [
    "Catchment",
    "CemaNeige",
    "CemaNeigeMultiLayerState",
    "CemaNeigeSingleLayerState",
    "ForcingData",
    "GR6JFluxes",
    "GR6JOutput",
    "ModelOutput",
    "ObservedData",
    "Parameters",
    "Resolution",
    "SnowLayerOutputs",
    "SnowOutput",
    "Solution",
    "State",
    "calibrate",
    "cemaneige_multi_layer_step",
    "cemaneige_step",
    "get_model",
    "get_model_info",
    "list_metrics",
    "list_models",
    "run",
    "step",
]
