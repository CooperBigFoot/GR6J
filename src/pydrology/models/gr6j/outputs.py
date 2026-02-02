"""GR6J model flux outputs as arrays.

This module provides dataclasses for organizing and accessing GR6J model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class GR6JFluxes:
    """GR6J model flux outputs as arrays.

    All arrays have the same length as the input forcing data.

    Attributes:
        pet: Potential evapotranspiration [mm/day].
        precip: Precipitation input to GR6J [mm/day]. When snow module is
            enabled, this is the liquid water output from CemaNeige.
        production_store: Production store level after timestep [mm].
        net_rainfall: Net rainfall after interception [mm/day].
        storage_infiltration: Water infiltrating to production store [mm/day].
        actual_et: Actual evapotranspiration [mm/day].
        percolation: Percolation from production store [mm/day].
        effective_rainfall: Total effective rainfall after percolation [mm/day].
        q9: Output from UH1 (90% branch) [mm/day].
        q1: Output from UH2 (10% branch) [mm/day].
        routing_store: Routing store level after timestep [mm].
        exchange: Groundwater exchange potential [mm/day].
        actual_exchange_routing: Actual exchange from routing store [mm/day].
        actual_exchange_direct: Actual exchange from direct branch [mm/day].
        actual_exchange_total: Total actual exchange [mm/day].
        qr: Outflow from routing store [mm/day].
        qrexp: Outflow from exponential store [mm/day].
        exponential_store: Exponential store level after timestep [mm].
        qd: Direct branch outflow [mm/day].
        streamflow: Total simulated streamflow [mm/day].
    """

    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    net_rainfall: np.ndarray
    storage_infiltration: np.ndarray
    actual_et: np.ndarray
    percolation: np.ndarray
    effective_rainfall: np.ndarray
    q9: np.ndarray
    q1: np.ndarray
    routing_store: np.ndarray
    exchange: np.ndarray
    actual_exchange_routing: np.ndarray
    actual_exchange_direct: np.ndarray
    actual_exchange_total: np.ndarray
    qr: np.ndarray
    qrexp: np.ndarray
    exponential_store: np.ndarray
    qd: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


# Backward compatibility alias
GR6JOutput = GR6JFluxes
