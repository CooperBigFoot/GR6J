"""GR2M data structures for parameters and state variables.

This module defines the core data types used by the GR2M hydrological model:
- Parameters: The 2 calibrated model parameters
- State: The mutable state variables tracked during simulation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """GR2M calibrated parameters.

    Both parameters that define the model behavior. This is a frozen dataclass
    to prevent accidental modification during simulation.

    Attributes:
        x1: Production store capacity [mm].
        x2: Groundwater exchange coefficient [-].
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Groundwater exchange coefficient [-]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to a 1D array for array protocol.

        Layout: [x1, x2] (2 elements)
        """
        arr = np.array([self.x1, self.x2], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Reconstruct Parameters from array."""
        return cls(
            x1=float(arr[0]),
            x2=float(arr[1]),
        )


@dataclass
class State:
    """GR2M model state variables.

    Mutable state that evolves during simulation. Contains the two stores.

    Attributes:
        production_store: S - soil moisture store level [mm].
        routing_store: R - groundwater/routing store level [mm].
    """

    production_store: float  # S - soil moisture [mm]
    routing_store: float  # R - groundwater [mm]

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        """Create initial state from parameters.

        Uses standard initialization fractions:
        - Production store at 30% of X1 capacity
        - Routing store at 30% of X1 (since X2 is dimensionless)

        Args:
            params: Model parameters to derive initial state from.

        Returns:
            Initialized State object ready for simulation.
        """
        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.3 * params.x1,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for array protocol.

        Layout: [production_store, routing_store]
        Total: 2 elements
        """
        arr = np.array([self.production_store, self.routing_store], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        """Reconstruct State from array."""
        return cls(
            production_store=float(arr[0]),
            routing_store=float(arr[1]),
        )
