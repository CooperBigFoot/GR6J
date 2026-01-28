"""GR6J hydrological model.

A lumped conceptual rainfall-runoff model for daily streamflow simulation.
Génie Rural à 6 paramètres Journalier.

Includes optional CemaNeige snow module for cold-climate catchments.
"""

from .cemaneige import CemaNeige, CemaNeigeSingleLayerState, cemaneige_step
from .inputs import Catchment, ForcingData
from .model import Parameters, State, run, step
from .outputs import GR6JOutput, ModelOutput, SnowOutput

__all__ = [
    "Catchment",
    "CemaNeige",
    "CemaNeigeSingleLayerState",
    "ForcingData",
    "GR6JOutput",
    "ModelOutput",
    "Parameters",
    "SnowOutput",
    "State",
    "cemaneige_step",
    "run",
    "step",
]
