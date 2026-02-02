"""Input data structures for the GR6J model.

This module defines validated input containers:
- ForcingData: Time series forcing data (precipitation, PET, temperature)
- Catchment: Static catchment properties for snow module configuration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ForcingData(BaseModel):
    """Validated forcing data for the GR6J model.

    All arrays must be 1D with the same length. NaN values are rejected.
    Numeric arrays are coerced to float64.

    Attributes:
        time: Datetime array for each timestep (datetime64).
        precip: Precipitation [mm/day].
        pet: Potential evapotranspiration [mm/day].
        temp: Temperature [C]. Required when snow module is enabled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    time: np.ndarray  # datetime64
    precip: np.ndarray  # [mm/day]
    pet: np.ndarray  # [mm/day]
    temp: np.ndarray | None = None  # [C]

    @field_validator("time", mode="before")
    @classmethod
    def validate_time(cls, v: np.ndarray) -> np.ndarray:
        """Validate time array: must be 1D and coerced to datetime64."""
        arr = np.asarray(v)
        if arr.ndim != 1:
            msg = f"time array must be 1D, got {arr.ndim}D"
            raise ValueError(msg)
        return arr.astype("datetime64[ns]")

    @field_validator("precip", mode="before")
    @classmethod
    def validate_precip(cls, v: np.ndarray) -> np.ndarray:
        """Validate precip array: must be 1D float64 with no NaN values."""
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            msg = f"precip array must be 1D, got {arr.ndim}D"
            raise ValueError(msg)
        if np.any(np.isnan(arr)):
            msg = "precip array contains NaN values"
            raise ValueError(msg)
        return arr

    @field_validator("pet", mode="before")
    @classmethod
    def validate_pet(cls, v: np.ndarray) -> np.ndarray:
        """Validate pet array: must be 1D float64 with no NaN values."""
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            msg = f"pet array must be 1D, got {arr.ndim}D"
            raise ValueError(msg)
        if np.any(np.isnan(arr)):
            msg = "pet array contains NaN values"
            raise ValueError(msg)
        return arr

    @field_validator("temp", mode="before")
    @classmethod
    def validate_temp(cls, v: np.ndarray | None) -> np.ndarray | None:
        """Validate temp array: must be 1D float64 with no NaN values if provided."""
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            msg = f"temp array must be 1D, got {arr.ndim}D"
            raise ValueError(msg)
        if np.any(np.isnan(arr)):
            msg = "temp array contains NaN values"
            raise ValueError(msg)
        return arr

    @model_validator(mode="after")
    def validate_array_lengths(self) -> ForcingData:
        """Ensure all arrays have the same length."""
        n = len(self.time)
        if len(self.precip) != n:
            msg = f"precip length {len(self.precip)} does not match time length {n}"
            raise ValueError(msg)
        if len(self.pet) != n:
            msg = f"pet length {len(self.pet)} does not match time length {n}"
            raise ValueError(msg)
        if self.temp is not None and len(self.temp) != n:
            msg = f"temp length {len(self.temp)} does not match time length {n}"
            raise ValueError(msg)
        return self

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)


def _validate_multi_layer_config(catchment: Catchment) -> None:
    """Validate that multi-layer configuration is complete.

    When n_layers > 1, hypsometric_curve and input_elevation are required.
    Raises ValueError if the configuration is incomplete.
    """
    if catchment.n_layers > 1:
        if catchment.hypsometric_curve is None:
            msg = "hypsometric_curve is required when n_layers > 1"
            raise ValueError(msg)
        if catchment.input_elevation is None:
            msg = "input_elevation is required when n_layers > 1"
            raise ValueError(msg)
        if len(catchment.hypsometric_curve) != 101:
            msg = f"hypsometric_curve must have 101 points (percentiles 0-100), got {len(catchment.hypsometric_curve)}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Catchment:
    """Static catchment properties for the GR6J model.

    Attributes:
        mean_annual_solid_precip: Mean annual solid precipitation [mm/year].
            Required when snow module is enabled.
        hypsometric_curve: Optional elevation distribution for multi-layer snow.
        input_elevation: Optional elevation of forcing data [m].
        n_layers: Number of elevation bands for snow (default 1).
        temp_gradient: Temperature lapse rate [°C/100m]. If None, uses default 0.6.
        precip_gradient: Precipitation gradient [m⁻¹]. If None, uses default 0.00041.
    """

    mean_annual_solid_precip: float  # [mm/year]

    # For future multi-layer CemaNeige (optional)
    hypsometric_curve: np.ndarray | None = None
    input_elevation: float | None = None
    n_layers: int = 1
    temp_gradient: float | None = None  # Temperature lapse rate [°C/100m]
    precip_gradient: float | None = None  # Precipitation gradient [m⁻¹]

    def __post_init__(self) -> None:
        """Validate multi-layer configuration."""
        _validate_multi_layer_config(self)
