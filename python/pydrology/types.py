"""Input data structures for the GR6J model.

This module defines validated input containers:
- ForcingData: Time series forcing data (precipitation, PET, temperature)
- Catchment: Static catchment properties for snow module configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Resolution(str, Enum):
    """Temporal resolution of forcing data."""

    hourly = "hourly"
    daily = "daily"
    monthly = "monthly"
    annual = "annual"

    @property
    def days_per_timestep(self) -> float:
        return {
            Resolution.hourly: 1 / 24,
            Resolution.daily: 1.0,
            Resolution.monthly: 30.4375,
            Resolution.annual: 365.25,
        }[self]

    @property
    def _ordinal(self) -> int:
        return [Resolution.hourly, Resolution.daily, Resolution.monthly, Resolution.annual].index(self)

    def __lt__(self, other: Resolution) -> bool:
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._ordinal < other._ordinal

    def __le__(self, other: Resolution) -> bool:
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._ordinal <= other._ordinal


_RESOLUTION_TOLERANCES: dict[Resolution, tuple[float, float]] = {
    Resolution.hourly: (0.9, 1.1),  # ~54-66 minutes in hours
    Resolution.daily: (22.0, 26.0),  # 22-26 hours
    Resolution.monthly: (27 * 24, 32 * 24),  # 27-32 days in hours
    Resolution.annual: (360 * 24, 370 * 24),  # 360-370 days in hours
}


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
    resolution: Resolution = Resolution.daily

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

    @model_validator(mode="after")
    def validate_time_resolution(self) -> ForcingData:
        if len(self.time) <= 1:
            return self
        median_gap_hours = float(np.median(np.diff(self.time)) / np.timedelta64(1, "h"))
        min_hours, max_hours = _RESOLUTION_TOLERANCES[self.resolution]
        if not (min_hours <= median_gap_hours <= max_hours):
            msg = (
                f"Time spacing (median {median_gap_hours:.1f} hours) does not match "
                f"resolution '{self.resolution.value}' (expected {min_hours}-{max_hours} hours)"
            )
            raise ValueError(msg)
        return self

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)

    def aggregate(
        self,
        target: Resolution,
        methods: dict[str, str] | None = None,
    ) -> ForcingData:
        """Aggregate forcing data to coarser temporal resolution.

        Args:
            target: Target resolution (must be coarser than current).
            methods: Aggregation methods per field. Defaults: precip=sum, pet=sum, temp=mean.

        Returns:
            New ForcingData at the target resolution.

        Raises:
            ValueError: If target is not coarser than current resolution.
            ImportError: If polars is not installed.
        """
        if target <= self.resolution:
            raise ValueError(
                f"Cannot aggregate from {self.resolution.value} to {target.value}; "
                f"target must be coarser than current resolution"
            )

        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("Polars is required for aggregation: uv add polars") from e

        # Default aggregation methods
        methods = {"precip": "sum", "pet": "sum", "temp": "mean", **(methods or {})}

        # Map resolution to polars frequency
        freq_map = {
            Resolution.daily: "1d",
            Resolution.monthly: "1mo",
            Resolution.annual: "1y",
        }
        freq = freq_map.get(target)
        if freq is None:
            raise ValueError(f"Aggregation to {target.value} is not supported")

        # Build DataFrame
        data = {
            "time": self.time.astype("datetime64[us]"),
            "precip": self.precip,
            "pet": self.pet,
        }
        if self.temp is not None:
            data["temp"] = self.temp

        df = pl.DataFrame(data)

        # Build aggregation expressions
        agg_exprs = []
        for col, method in methods.items():
            if col == "temp" and self.temp is None:
                continue
            if col not in df.columns:
                continue
            if method == "sum":
                agg_exprs.append(pl.col(col).sum().alias(col))
            elif method == "mean":
                agg_exprs.append(pl.col(col).mean().alias(col))
            else:
                raise ValueError(f"Unknown aggregation method '{method}' for column '{col}'")

        # Perform group_by_dynamic aggregation
        result = df.sort("time").group_by_dynamic("time", every=freq).agg(agg_exprs)

        # Extract arrays
        new_time = result["time"].to_numpy().astype("datetime64[ns]")
        new_precip = result["precip"].to_numpy()
        new_pet = result["pet"].to_numpy()
        new_temp = result["temp"].to_numpy() if self.temp is not None else None

        return ForcingData(
            time=new_time,
            precip=new_precip,
            pet=new_pet,
            temp=new_temp,
            resolution=target,
        )


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
