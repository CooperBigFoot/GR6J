"""GR2M model flux outputs as arrays.

This module provides dataclasses for organizing and accessing GR2M model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class GR2MFluxes:
    """GR2M model flux outputs as arrays.

    All arrays have the same length as the input forcing data.
    Field order matches the airGR MISC array indices (1-11).

    Attributes:
        pet: Potential evapotranspiration [mm/month]. MISC(1)
        precip: Precipitation input [mm/month]. MISC(2)
        production_store: Production store level after timestep [mm]. MISC(3)
        rainfall_excess: Rainfall excess / net precipitation P1 [mm/month]. MISC(4)
        storage_fill: Storage infiltration PS [mm/month]. MISC(5)
        actual_et: Actual evapotranspiration AE [mm/month]. MISC(6)
        percolation: Percolation from production store P2 [mm/month]. MISC(7)
        routing_input: Total water to routing P3 = P1 + P2 [mm/month]. MISC(8)
        routing_store: Routing store level after timestep [mm]. MISC(9)
        exchange: Groundwater exchange AEXCH [mm/month]. MISC(10)
        streamflow: Total simulated streamflow Q [mm/month]. MISC(11)
    """

    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    rainfall_excess: np.ndarray
    storage_fill: np.ndarray
    actual_et: np.ndarray
    percolation: np.ndarray
    routing_input: np.ndarray
    routing_store: np.ndarray
    exchange: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}
