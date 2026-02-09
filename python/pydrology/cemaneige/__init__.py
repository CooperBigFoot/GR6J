"""CemaNeige snow accumulation and melt module.

This subpackage implements the CemaNeige snow model for coupling with GR6J.
"""

from .layers import derive_layers, extrapolate_precipitation, extrapolate_temperature
from .run import cemaneige_multi_layer_step, cemaneige_step
from .types import CemaNeige, CemaNeigeMultiLayerState, CemaNeigeSingleLayerState

__all__ = [
    "CemaNeige",
    "CemaNeigeMultiLayerState",
    "CemaNeigeSingleLayerState",
    "cemaneige_multi_layer_step",
    "cemaneige_step",
    "derive_layers",
    "extrapolate_precipitation",
    "extrapolate_temperature",
]
