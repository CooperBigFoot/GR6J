"""CemaNeige data structures for parameters and state variables.

This module defines the core data types used by the CemaNeige snow model:
- CemaNeige: The calibrated model parameters
- CemaNeigeSingleLayerState: The mutable state variables for single-layer mode
- CemaNeigeMultiLayerState: The mutable state wrapper for multi-layer mode
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CemaNeige:
    """CemaNeige calibrated parameters.

    Parameters that define the snow model behavior. This is a frozen dataclass
    to prevent accidental modification during simulation.

    Note: mean_annual_solid_precip is now specified via the Catchment class,
    as it is a static catchment property rather than a calibration parameter.

    Attributes:
        ctg: Thermal state weighting coefficient [-]. Controls the inertia of
            the snow pack thermal state.
        kf: Degree-day melt factor [mm/°C/day]. Controls the rate of snowmelt
            per degree above the melt threshold.
    """

    ctg: float  # Thermal state weighting coefficient [-]
    kf: float  # Degree-day melt factor [mm/°C/day]


@dataclass
class CemaNeigeSingleLayerState:
    """CemaNeige single-layer model state variables.

    Mutable state that evolves during simulation. Contains the snow pack
    water equivalent, thermal state, and hysteresis tracking variables.

    Attributes:
        g: Snow pack water equivalent [mm]. The total water stored in the
            snow pack. Constraint: g >= 0.
        etg: Thermal state of the snow pack [°C]. Represents the cold content
            of the snow. Constraint: etg <= 0.
        gthreshold: Melt threshold [mm]. The snow pack level at which melt
            rate begins to decrease due to patchiness effects.
        glocalmax: Local maximum snow pack [mm]. Tracks the maximum snow
            accumulation since the last complete melt, used for hysteresis.
    """

    g: float  # Snow pack water equivalent [mm]
    etg: float  # Thermal state [°C]
    gthreshold: float  # Melt threshold [mm]
    glocalmax: float  # Local maximum for hysteresis [mm]

    @classmethod
    def initialize(cls, mean_annual_solid_precip: float) -> CemaNeigeSingleLayerState:
        """Create initial state from mean annual solid precipitation.

        Initializes with:
        - g = 0 (no initial snow pack)
        - etg = 0 (neutral thermal state)
        - gthreshold = 0.9 * mean_annual_solid_precip
        - glocalmax = gthreshold

        Args:
            mean_annual_solid_precip: Mean annual solid precipitation [mm].
                Used to compute the initial gthreshold value.

        Returns:
            Initialized CemaNeigeSingleLayerState object ready for simulation.
        """
        from .constants import GTHRESHOLD_FACTOR

        gthreshold = GTHRESHOLD_FACTOR * mean_annual_solid_precip
        return cls(
            g=0.0,
            etg=0.0,
            gthreshold=gthreshold,
            glocalmax=gthreshold,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for array protocol.

        Layout: [g, etg, gthreshold, glocalmax] (4 elements)
        """
        arr = np.array([self.g, self.etg, self.gthreshold, self.glocalmax], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> CemaNeigeSingleLayerState:
        """Reconstruct state from array."""
        return cls(
            g=float(arr[0]),
            etg=float(arr[1]),
            gthreshold=float(arr[2]),
            glocalmax=float(arr[3]),
        )


@dataclass
class CemaNeigeMultiLayerState:
    """CemaNeige multi-layer model state.

    Wraps multiple single-layer states, one per elevation band.
    This is mutable since the layer states evolve during simulation.

    Attributes:
        layer_states: List of single-layer state objects, one per elevation band.
    """

    layer_states: list[CemaNeigeSingleLayerState]

    @classmethod
    def initialize(
        cls,
        n_layers: int,
        mean_annual_solid_precip: float,
        *,
        layer_elevations: list[float] | np.ndarray | None = None,
        input_elevation: float | None = None,
        precip_gradient: float | None = None,
        use_linear: bool = False,
    ) -> CemaNeigeMultiLayerState:
        """Create initial multi-layer state.

        Each layer is initialized independently. When layer_elevations and
        input_elevation are provided, gthreshold is scaled per band using
        the same precipitation gradient applied to forcing.

        Args:
            n_layers: Number of elevation bands.
            mean_annual_solid_precip: Mean annual solid precipitation [mm/year].
            layer_elevations: Representative elevation of each layer [m].
            input_elevation: Elevation of the forcing data [m].
            precip_gradient: Precipitation gradient [m⁻¹]. If None, uses default.
            use_linear: If True, use linear gradient instead of exponential.

        Returns:
            Initialized CemaNeigeMultiLayerState with n_layers independent states.
        """
        import math

        from .constants import GTHRESHOLD_FACTOR
        from pydrology.utils.elevation import ELEV_CAP_PRECIP, GRAD_P_DEFAULT, GRAD_P_LINEAR_DEFAULT

        base_gthreshold = GTHRESHOLD_FACTOR * mean_annual_solid_precip

        can_scale = (
            layer_elevations is not None
            and input_elevation is not None
            and n_layers > 1
        )

        if can_scale:
            if precip_gradient is not None:
                grad = precip_gradient
            elif use_linear:
                grad = GRAD_P_LINEAR_DEFAULT
            else:
                grad = GRAD_P_DEFAULT

            input_eff = min(input_elevation, ELEV_CAP_PRECIP)
            layer_states = []
            for i in range(n_layers):
                layer_eff = min(float(layer_elevations[i]), ELEV_CAP_PRECIP)
                if use_linear:
                    ratio = max(0.0, 1.0 + grad * (layer_eff - input_eff))
                else:
                    ratio = math.exp(grad * (layer_eff - input_eff))
                gt = base_gthreshold * ratio
                layer_states.append(CemaNeigeSingleLayerState(
                    g=0.0, etg=0.0, gthreshold=gt, glocalmax=gt,
                ))
        else:
            layer_states = [
                CemaNeigeSingleLayerState.initialize(mean_annual_solid_precip)
                for _ in range(n_layers)
            ]

        return cls(layer_states=layer_states)

    def __len__(self) -> int:
        """Return the number of layers."""
        return len(self.layer_states)

    def __getitem__(self, index: int) -> CemaNeigeSingleLayerState:
        """Get a specific layer state by index."""
        return self.layer_states[index]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert multi-layer state to a 2D array for array protocol.

        Shape: (n_layers, 4) where columns are [g, etg, gthreshold, glocalmax]
        """
        n_layers = len(self.layer_states)
        arr = np.empty((n_layers, 4), dtype=np.float64)
        for i, layer in enumerate(self.layer_states):
            arr[i, 0] = layer.g
            arr[i, 1] = layer.etg
            arr[i, 2] = layer.gthreshold
            arr[i, 3] = layer.glocalmax
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> CemaNeigeMultiLayerState:
        """Reconstruct multi-layer state from 2D array."""
        layer_states = [
            CemaNeigeSingleLayerState(
                g=float(arr[i, 0]),
                etg=float(arr[i, 1]),
                gthreshold=float(arr[i, 2]),
                glocalmax=float(arr[i, 3]),
            )
            for i in range(arr.shape[0])
        ]
        return cls(layer_states=layer_states)
