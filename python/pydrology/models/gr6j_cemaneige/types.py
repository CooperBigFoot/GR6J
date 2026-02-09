"""GR6J-CemaNeige coupled model data structures.

This module defines the core data types for the coupled GR6J-CemaNeige model:
- Parameters: 8 calibrated parameters (6 GR6J + 2 CemaNeige, flat structure)
- State: Combined GR6J state and multi-layer snow state
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydrology.cemaneige.constants import GTHRESHOLD_FACTOR
from pydrology.models.gr6j.constants import NH
from pydrology.types import Catchment

from .constants import PARAM_NAMES, SNOW_LAYER_STATE_SIZE, STATE_SIZE_BASE


@dataclass(frozen=True)
class Parameters:
    """GR6J-CemaNeige calibrated parameters (8 total, flat structure).

    Combines all 6 GR6J parameters with 2 CemaNeige snow parameters.
    This is a frozen dataclass to prevent accidental modification during simulation.

    Attributes:
        x1: Production store capacity [mm].
        x2: Intercatchment exchange coefficient [mm/day].
        x3: Routing store capacity [mm].
        x4: Unit hydrograph time constant [days].
        x5: Intercatchment exchange threshold [-].
        x6: Exponential store scale parameter [mm].
        ctg: Thermal state weighting coefficient (CemaNeige) [-].
            Controls the inertia of the snow pack thermal state.
        kf: Degree-day melt factor (CemaNeige) [mm/°C/day].
            Controls the rate of snowmelt per degree above the melt threshold.
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Intercatchment exchange coefficient [mm/day]
    x3: float  # Routing store capacity [mm]
    x4: float  # Unit hydrograph time constant [days]
    x5: float  # Intercatchment exchange threshold [-]
    x6: float  # Exponential store scale parameter [mm]
    ctg: float  # Thermal state weighting coefficient [-]
    kf: float  # Degree-day melt factor [mm/°C/day]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to a 1D array for Numba.

        Layout: [x1, x2, x3, x4, x5, x6, ctg, kf] (8 elements)
        """
        arr = np.array(
            [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.ctg, self.kf],
            dtype=np.float64,
        )
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Reconstruct Parameters from array.

        Args:
            arr: 1D array of shape (8,) with parameter values.

        Returns:
            Parameters instance.
        """
        if len(arr) != len(PARAM_NAMES):
            msg = f"Expected array of length {len(PARAM_NAMES)}, got {len(arr)}"
            raise ValueError(msg)
        return cls(
            x1=float(arr[0]),
            x2=float(arr[1]),
            x3=float(arr[2]),
            x4=float(arr[3]),
            x5=float(arr[4]),
            x6=float(arr[5]),
            ctg=float(arr[6]),
            kf=float(arr[7]),
        )


@dataclass
class State:
    """Combined GR6J and CemaNeige state variables.

    Mutable state that evolves during simulation. Contains the three GR6J stores,
    unit hydrograph convolution states, and per-layer snow states.

    Attributes:
        production_store: S - soil moisture store level [mm].
        routing_store: R - groundwater/routing store level [mm].
        exponential_store: Exp - slow drainage store, can be negative [mm].
        uh1_states: Convolution states for UH1 (20 elements).
        uh2_states: Convolution states for UH2 (40 elements).
        snow_layer_states: Per-layer snow state array, shape (n_layers, 4).
            Each layer: [snow_pack, thermal_state, g_threshold, glocalmax].
    """

    production_store: float  # S - soil moisture [mm]
    routing_store: float  # R - groundwater [mm]
    exponential_store: float  # Exp - slow drainage, can be negative [mm]
    uh1_states: np.ndarray  # 20-element array for UH1
    uh2_states: np.ndarray  # 40-element array for UH2
    snow_layer_states: np.ndarray  # shape (n_layers, 4)

    @property
    def n_layers(self) -> int:
        """Return the number of snow layers."""
        return self.snow_layer_states.shape[0]

    @classmethod
    def initialize(cls, params: Parameters, catchment: Catchment) -> State:
        """Create initial state from parameters and catchment properties.

        Uses standard initialization fractions:
        - Production store at 30% capacity
        - Routing store at 50% capacity
        - Exponential store at zero
        - Unit hydrograph states all zero
        - Snow layers initialized based on mean annual solid precipitation

        Args:
            params: Model parameters.
            catchment: Catchment properties including mean_annual_solid_precip and n_layers.

        Returns:
            Initialized State object ready for simulation.
        """
        n_layers = catchment.n_layers

        # Initialize snow layer states
        gthreshold = GTHRESHOLD_FACTOR * catchment.mean_annual_solid_precip
        snow_layer_states = np.zeros((n_layers, SNOW_LAYER_STATE_SIZE), dtype=np.float64)
        for i in range(n_layers):
            snow_layer_states[i, 0] = 0.0  # g (snow pack)
            snow_layer_states[i, 1] = 0.0  # etg (thermal state)
            snow_layer_states[i, 2] = gthreshold  # gthreshold
            snow_layer_states[i, 3] = gthreshold  # glocalmax

        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.5 * params.x3,
            exponential_store=0.0,
            uh1_states=np.zeros(NH),
            uh2_states=np.zeros(2 * NH),
            snow_layer_states=snow_layer_states,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for Numba.

        Layout:
            [0]: production_store
            [1]: routing_store
            [2]: exponential_store
            [3:23]: uh1_states (20 elements)
            [23:63]: uh2_states (40 elements)
            [63:63+n_layers*4]: snow_layer_states flattened

        Total: 63 + n_layers * 4 elements
        """
        n_layers = self.n_layers
        total_size = STATE_SIZE_BASE + n_layers * SNOW_LAYER_STATE_SIZE
        arr = np.empty(total_size, dtype=np.float64)
        arr[0] = self.production_store
        arr[1] = self.routing_store
        arr[2] = self.exponential_store
        arr[3:23] = self.uh1_states
        arr[23:63] = self.uh2_states
        arr[63:] = self.snow_layer_states.flatten()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray, n_layers: int) -> State:
        """Reconstruct State from array.

        Args:
            arr: 1D array with state values.
            n_layers: Number of snow layers to reconstruct.

        Returns:
            State instance.
        """
        expected_size = STATE_SIZE_BASE + n_layers * SNOW_LAYER_STATE_SIZE
        if len(arr) != expected_size:
            msg = f"Expected array of length {expected_size}, got {len(arr)}"
            raise ValueError(msg)

        snow_layer_states = arr[63:].reshape(n_layers, SNOW_LAYER_STATE_SIZE)

        return cls(
            production_store=float(arr[0]),
            routing_store=float(arr[1]),
            exponential_store=float(arr[2]),
            uh1_states=arr[3:23].copy(),
            uh2_states=arr[23:63].copy(),
            snow_layer_states=snow_layer_states.copy(),
        )
