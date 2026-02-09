"""CemaNeige snow module flux outputs as arrays.

This module provides dataclasses for organizing and accessing CemaNeige outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


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
class SnowLayerOutputs:
    """Per-layer snow outputs for multi-layer CemaNeige simulations.

    Contains 2D arrays with shape (n_timesteps, n_layers) for detailed
    layer-by-layer analysis. The catchment-aggregated values are still
    available in SnowOutput.

    Attributes:
        layer_elevations: Representative elevation of each layer [m]. Shape (n_layers,).
        layer_fractions: Area fraction of each layer [-]. Shape (n_layers,).
        snow_pack: Snow pack water equivalent per layer [mm]. Shape (n_timesteps, n_layers).
        snow_thermal_state: Thermal state per layer [deg C]. Shape (n_timesteps, n_layers).
        snow_gratio: Snow cover fraction per layer [-]. Shape (n_timesteps, n_layers).
        snow_melt: Actual melt per layer [mm/day]. Shape (n_timesteps, n_layers).
        snow_pliq_and_melt: Total liquid output per layer [mm/day]. Shape (n_timesteps, n_layers).
        layer_temp: Extrapolated temperature per layer [deg C]. Shape (n_timesteps, n_layers).
        layer_precip: Extrapolated precipitation per layer [mm/day]. Shape (n_timesteps, n_layers).
    """

    layer_elevations: np.ndarray  # Shape (n_layers,)
    layer_fractions: np.ndarray  # Shape (n_layers,)
    snow_pack: np.ndarray  # Shape (n_timesteps, n_layers)
    snow_thermal_state: np.ndarray  # Shape (n_timesteps, n_layers)
    snow_gratio: np.ndarray  # Shape (n_timesteps, n_layers)
    snow_melt: np.ndarray  # Shape (n_timesteps, n_layers)
    snow_pliq_and_melt: np.ndarray  # Shape (n_timesteps, n_layers)
    layer_temp: np.ndarray  # Shape (n_timesteps, n_layers)
    layer_precip: np.ndarray  # Shape (n_timesteps, n_layers)

    @property
    def n_layers(self) -> int:
        """Return the number of elevation layers."""
        return len(self.layer_elevations)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}
