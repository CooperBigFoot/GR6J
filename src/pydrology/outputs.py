"""Structured output dataclasses for model results.

This module provides dataclasses for organizing and accessing model outputs:
- GR6JOutput: GR6J model flux outputs (alias for GR6JFluxes)
- SnowOutput: CemaNeige snow module flux outputs
- SnowLayerOutputs: Per-layer snow outputs for multi-layer simulations
- ModelOutput: Combined model output with time index (generic over flux type)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

# Re-export from model-specific modules for backward compatibility
from pydrology.models.cemaneige.outputs import SnowLayerOutputs, SnowOutput
from pydrology.models.gr6j.outputs import GR6JFluxes, GR6JOutput

__all__ = ["GR6JFluxes", "GR6JOutput", "ModelOutput", "SnowLayerOutputs", "SnowOutput"]

# Type variable for flux types
F = TypeVar("F")


@dataclass(frozen=True)
class ModelOutput(Generic[F]):
    """Complete model output combining flux outputs and optional snow outputs.

    Generic over the flux type F, allowing different models to use their own
    flux output types while sharing the same ModelOutput structure.

    Provides a unified interface for accessing all model results with time
    indexing and conversion to pandas DataFrame.

    Attributes:
        time: Datetime array for each timestep.
        fluxes: Model flux outputs (type depends on the model).
        snow: Optional CemaNeige outputs (present if snow module was enabled).
        snow_layers: Optional per-layer snow outputs for multi-layer simulations.
    """

    time: np.ndarray
    fluxes: F
    snow: SnowOutput | None = None
    snow_layers: SnowLayerOutputs | None = None

    @property
    def streamflow(self) -> np.ndarray:
        """Return the streamflow array from flux outputs.

        Returns:
            Streamflow array [mm/day].
        """
        return self.fluxes.streamflow  # type: ignore[attr-defined, return-value]

    @property
    def gr6j(self) -> F:
        """Backward compatibility alias for fluxes.

        Deprecated: Use .fluxes instead.
        """
        return self.fluxes

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with time index.

        Combines all flux outputs (and snow outputs if present) into a single
        DataFrame with time as the index.

        Returns:
            DataFrame with all flux outputs and time as index.
        """
        data = self.fluxes.to_dict()  # type: ignore[attr-defined]

        if self.snow is not None:
            data.update(self.snow.to_dict())

        df = pd.DataFrame(data, index=self.time)
        df.index.name = "time"

        return df
