"""GR6J data structures for parameters and state variables.

This module defines the core data types used by the GR6J hydrological model:
- Parameters: The 6 calibrated model parameters
- State: The mutable state variables tracked during simulation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import NH


@dataclass(frozen=True)
class Parameters:
    """GR6J calibrated parameters.

    All 6 parameters that define the model behavior. This is a frozen dataclass
    to prevent accidental modification during simulation.

    Attributes:
        x1: Production store capacity [mm].
        x2: Intercatchment exchange coefficient [mm/day].
        x3: Routing store capacity [mm].
        x4: Unit hydrograph time constant [days].
        x5: Intercatchment exchange threshold [-].
        x6: Exponential store scale parameter [mm].
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Intercatchment exchange coefficient [mm/day]
    x3: float  # Routing store capacity [mm]
    x4: float  # Unit hydrograph time constant [days]
    x5: float  # Intercatchment exchange threshold [-]
    x6: float  # Exponential store scale parameter [mm]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to a 1D array for array protocol.

        Layout: [x1, x2, x3, x4, x5, x6] (6 elements)
        """
        arr = np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Reconstruct Parameters from array."""
        return cls(
            x1=float(arr[0]),
            x2=float(arr[1]),
            x3=float(arr[2]),
            x4=float(arr[3]),
            x5=float(arr[4]),
            x6=float(arr[5]),
        )


@dataclass
class State:
    """GR6J model state variables.

    Mutable state that evolves during simulation. Contains the three stores
    and the unit hydrograph convolution states.

    Attributes:
        production_store: S - soil moisture store level [mm].
        routing_store: R - groundwater/routing store level [mm].
        exponential_store: Exp - slow drainage store, can be negative [mm].
        uh1_states: Convolution states for UH1 (20 elements).
        uh2_states: Convolution states for UH2 (40 elements).
    """

    production_store: float  # S - soil moisture [mm]
    routing_store: float  # R - groundwater [mm]
    exponential_store: float  # Exp - slow drainage, can be negative [mm]
    uh1_states: np.ndarray  # 20-element array for UH1
    uh2_states: np.ndarray  # 40-element array for UH2

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        """Create initial state from parameters.

        Uses standard initialization fractions:
        - Production store at 30% capacity
        - Routing store at 50% capacity
        - Exponential store at zero
        - Unit hydrograph states all zero

        Args:
            params: Model parameters to derive initial state from.

        Returns:
            Initialized State object ready for simulation.
        """
        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.5 * params.x3,
            exponential_store=0.0,
            uh1_states=np.zeros(NH),
            uh2_states=np.zeros(2 * NH),
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for array protocol.

        Layout: [production_store, routing_store, exponential_store, uh1_states[0:20], uh2_states[0:40]]
        Total: 63 elements
        """
        arr = np.empty(63, dtype=np.float64)
        arr[0] = self.production_store
        arr[1] = self.routing_store
        arr[2] = self.exponential_store
        arr[3:23] = self.uh1_states
        arr[23:63] = self.uh2_states
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        """Reconstruct State from array."""
        return cls(
            production_store=float(arr[0]),
            routing_store=float(arr[1]),
            exponential_store=float(arr[2]),
            uh1_states=arr[3:23].copy(),
            uh2_states=arr[23:63].copy(),
        )
