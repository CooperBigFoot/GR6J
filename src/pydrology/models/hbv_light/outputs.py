"""HBV-light model flux outputs as arrays.

This module provides dataclasses for organizing and accessing HBV-light model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class HBVLightFluxes:
    """HBV-light model flux outputs as arrays.

    All arrays have the same length as the input forcing data.

    Attributes:
        precip: Precipitation input [mm/day].
        temp: Temperature input [°C].
        pet: Potential evapotranspiration [mm/day].
        precip_rain: Liquid precipitation [mm/day].
        precip_snow: Solid precipitation after SFCF correction [mm/day].
        snow_pack: Snow pack water equivalent after timestep [mm].
        snow_melt: Snowmelt [mm/day].
        liquid_water_in_snow: Liquid water held in snow pack [mm].
        snow_input: Total input to soil (rain + melt outflow) [mm/day].
        soil_moisture: Soil moisture after timestep [mm].
        recharge: Recharge to groundwater [mm/day].
        actual_et: Actual evapotranspiration [mm/day].
        upper_zone: Upper groundwater zone storage after timestep [mm].
        lower_zone: Lower groundwater zone storage after timestep [mm].
        q0: Surface/quick flow from upper zone [mm/day].
        q1: Interflow from upper zone [mm/day].
        q2: Baseflow from lower zone [mm/day].
        percolation: Percolation from upper to lower zone [mm/day].
        qgw: Total groundwater runoff before routing [mm/day].
        streamflow: Total simulated streamflow after routing [mm/day].
    """

    # Inputs
    precip: np.ndarray
    temp: np.ndarray
    pet: np.ndarray

    # Snow routine outputs
    precip_rain: np.ndarray
    precip_snow: np.ndarray
    snow_pack: np.ndarray
    snow_melt: np.ndarray
    liquid_water_in_snow: np.ndarray
    snow_input: np.ndarray

    # Soil routine outputs
    soil_moisture: np.ndarray
    recharge: np.ndarray
    actual_et: np.ndarray

    # Response routine outputs
    upper_zone: np.ndarray
    lower_zone: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    percolation: np.ndarray
    qgw: np.ndarray

    # Final output
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class HBVLightZoneOutputs:
    """Per-zone outputs for multi-zone HBV-light simulations.

    Contains 2D arrays with shape (n_timesteps, n_zones) for detailed
    zone-by-zone analysis. The catchment-aggregated values are still
    available in HBVLightFluxes.

    In HBV-light, the snow and soil routines are computed independently
    per elevation zone, then aggregated by area-weighting. The response
    routine (groundwater) operates on the aggregated recharge.

    Attributes:
        zone_elevations: Representative elevation of each zone [m]. Shape (n_zones,).
        zone_fractions: Area fraction of each zone [-]. Shape (n_zones,).
        zone_temp: Extrapolated temperature per zone [°C]. Shape (n_timesteps, n_zones).
        zone_precip: Extrapolated precipitation per zone [mm/day]. Shape (n_timesteps, n_zones).
        snow_pack: Snow pack water equivalent per zone [mm]. Shape (n_timesteps, n_zones).
        liquid_water_in_snow: Liquid water held in snow per zone [mm]. Shape (n_timesteps, n_zones).
        snow_melt: Snowmelt per zone [mm/day]. Shape (n_timesteps, n_zones).
        snow_input: Total input to soil per zone (rain + melt outflow) [mm/day]. Shape (n_timesteps, n_zones).
        soil_moisture: Soil moisture per zone [mm]. Shape (n_timesteps, n_zones).
        recharge: Recharge to groundwater per zone [mm/day]. Shape (n_timesteps, n_zones).
        actual_et: Actual evapotranspiration per zone [mm/day]. Shape (n_timesteps, n_zones).
    """

    # Zone metadata
    zone_elevations: np.ndarray  # Shape (n_zones,)
    zone_fractions: np.ndarray  # Shape (n_zones,)

    # Per-zone forcings (extrapolated)
    zone_temp: np.ndarray  # Shape (n_timesteps, n_zones)
    zone_precip: np.ndarray  # Shape (n_timesteps, n_zones)

    # Per-zone snow routine outputs
    snow_pack: np.ndarray  # Shape (n_timesteps, n_zones)
    liquid_water_in_snow: np.ndarray  # Shape (n_timesteps, n_zones)
    snow_melt: np.ndarray  # Shape (n_timesteps, n_zones)
    snow_input: np.ndarray  # Shape (n_timesteps, n_zones)

    # Per-zone soil routine outputs
    soil_moisture: np.ndarray  # Shape (n_timesteps, n_zones)
    recharge: np.ndarray  # Shape (n_timesteps, n_zones)
    actual_et: np.ndarray  # Shape (n_timesteps, n_zones)

    @property
    def n_zones(self) -> int:
        """Return the number of elevation zones."""
        return len(self.zone_elevations)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}
