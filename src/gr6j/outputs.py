"""Structured output dataclasses for GR6J model results.

This module provides dataclasses for organizing and accessing model outputs:
- GR6JOutput: GR6J model flux outputs
- SnowOutput: CemaNeige snow module flux outputs
- ModelOutput: Combined model output with time index
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GR6JOutput:
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


@dataclass(frozen=True)
class SnowOutput:
    """CemaNeige snow module flux outputs as arrays.

    All arrays have the same length as the input forcing data.

    Attributes:
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
        snow_gthreshold: Melt threshold [mm].
        snow_glocalmax: Local maximum for hysteresis [mm].
    """

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
    snow_gthreshold: np.ndarray
    snow_glocalmax: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class ModelOutput:
    """Complete model output combining GR6J and optional snow outputs.

    Provides a unified interface for accessing all model results with time
    indexing and conversion to pandas DataFrame.

    Attributes:
        time: Datetime array for each timestep.
        gr6j: GR6J model outputs.
        snow: Optional CemaNeige outputs (present if snow module was enabled).
    """

    time: np.ndarray
    gr6j: GR6JOutput
    snow: SnowOutput | None = None

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with time index.

        Combines all GR6J outputs (and snow outputs if present) into a single
        DataFrame with time as the index.

        Returns:
            DataFrame with all flux outputs and time as index.
        """
        data = self.gr6j.to_dict()

        if self.snow is not None:
            data.update(self.snow.to_dict())

        df = pd.DataFrame(data, index=self.time)
        df.index.name = "time"

        return df
