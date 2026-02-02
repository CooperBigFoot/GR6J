"""Data types for calibration.

Defines validated containers for observed data and calibration results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ObservedData(BaseModel):
    """Validated observed streamflow data for calibration.

    All arrays must be 1D with the same length. NaN values are rejected.
    Numeric arrays are coerced to float64, time to datetime64[ns].

    Attributes:
        time: Datetime array for each observation (datetime64).
        streamflow: Observed streamflow [mm/day].
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    time: np.ndarray  # datetime64[ns]
    streamflow: np.ndarray  # [mm/day]

    @field_validator("time", mode="before")
    @classmethod
    def validate_time(cls, v: np.ndarray) -> np.ndarray:
        """Validate time array: must be 1D and coerced to datetime64[ns]."""
        arr = np.asarray(v)
        if arr.ndim != 1:
            msg = f"time array must be 1D, got {arr.ndim}D"
            raise ValueError(msg)
        return arr.astype("datetime64[ns]")

    @field_validator("streamflow", mode="before")
    @classmethod
    def validate_streamflow(cls, v: np.ndarray) -> np.ndarray:
        """Validate streamflow array: must be 1D float64 with no NaN values."""
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            msg = f"streamflow array must be 1D, got {arr.ndim}D"
            raise ValueError(msg)
        if np.any(np.isnan(arr)):
            msg = "streamflow array contains NaN values"
            raise ValueError(msg)
        return arr

    @model_validator(mode="after")
    def validate_array_lengths(self) -> ObservedData:
        """Ensure time and streamflow arrays have the same length."""
        if len(self.time) != len(self.streamflow):
            msg = f"streamflow length {len(self.streamflow)} does not match time length {len(self.time)}"
            raise ValueError(msg)
        return self

    def __len__(self) -> int:
        """Return the number of observations."""
        return len(self.time)


@dataclass(frozen=True)
class Solution:
    """Result from calibration optimization.

    Contains the optimized parameters and their objective scores.
    Model-agnostic: works with any registered model.

    Attributes:
        model: Model identifier (e.g., "gr6j", "gr6j_cemaneige").
        parameters: The optimized model parameters (type depends on model).
        score: Dictionary mapping objective names to their values.
    """

    model: str
    parameters: Any  # Model-specific Parameters type
    score: dict[str, float]
