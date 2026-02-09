"""GR6J-CemaNeige coupled model flux outputs.

This module provides the dataclass for organizing and accessing
combined snow and GR6J model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class GR6JCemaNeigeFluxes:
    """Combined GR6J and CemaNeige flux outputs as arrays.

    All arrays have the same length as the input forcing data.
    Contains 10 snow fields + 20 GR6J fields = 30 total fields.

    Snow-related attributes (from CemaNeige):
        precip_raw: Original precipitation before snow processing [mm/day].
        snow_pliq: Liquid precipitation (rain) [mm/day].
        snow_psol: Solid precipitation (snow) [mm/day].
        snow_pack: Snow pack water equivalent after melt [mm].
        snow_thermal_state: Thermal state of snow pack [deg C].
        snow_gratio: Snow cover fraction after melt [-].
        snow_pot_melt: Potential melt [mm/day].
        snow_melt: Actual melt [mm/day].
        snow_pliq_and_melt: Total liquid output to GR6J [mm/day].
        snow_temp: Air temperature [deg C].

    GR6J-related attributes:
        pet: Potential evapotranspiration [mm/day].
        precip: Precipitation input to GR6J (= snow_pliq_and_melt) [mm/day].
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

    # Snow-related outputs (10 fields)
    precip_raw: np.ndarray
    snow_pliq: np.ndarray
    snow_psol: np.ndarray
    snow_pack: np.ndarray
    snow_thermal_state: np.ndarray
    snow_gratio: np.ndarray
    snow_pot_melt: np.ndarray
    snow_melt: np.ndarray
    snow_pliq_and_melt: np.ndarray
    snow_temp: np.ndarray

    # GR6J-related outputs (20 fields)
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
