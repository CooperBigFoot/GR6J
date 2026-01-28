"""Input data structures for the GR6J model.

This module defines validated input containers:
- ForcingData: Time series forcing data (precipitation, PET, temperature)
- Catchment: Static catchment properties for snow module configuration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

logger = logging.getLogger(__name__)


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


# Catchment property bounds for validation warnings
_CATCHMENT_BOUNDS: dict[str, tuple[float, float]] = {
    "mean_annual_solid_precip": (0.0, 10000.0),
}


def _warn_if_outside_bounds(catchment: Catchment) -> None:
    """Log warnings for catchment properties outside typical ranges.

    This does not raise errors - properties outside bounds may still be valid
    for specific catchments or research purposes.
    """
    for name, (lower, upper) in _CATCHMENT_BOUNDS.items():
        value = getattr(catchment, name)
        if value < lower or value > upper:
            logger.warning(
                "Catchment property %s=%.4f is outside typical range [%.2f, %.2f]",
                name,
                value,
                lower,
                upper,
            )


@dataclass(frozen=True)
class Catchment:
    """Static catchment properties for the GR6J model.

    Attributes:
        mean_annual_solid_precip: Mean annual solid precipitation [mm/year].
            Required when snow module is enabled.
        hypsometric_curve: Optional elevation distribution for multi-layer snow.
        input_elevation: Optional elevation of forcing data [m].
        n_layers: Number of elevation bands for snow (default 1).
    """

    mean_annual_solid_precip: float  # [mm/year]

    # For future multi-layer CemaNeige (optional)
    hypsometric_curve: np.ndarray | None = None
    input_elevation: float | None = None
    n_layers: int = 1

    # Class-level reference to bounds for external access
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = _CATCHMENT_BOUNDS

    def __post_init__(self) -> None:
        """Validate catchment properties and warn if outside typical ranges."""
        _warn_if_outside_bounds(self)
