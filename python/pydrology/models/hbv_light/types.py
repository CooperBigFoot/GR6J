"""HBV-light data structures for parameters and state variables.

This module defines the core data types used by the HBV-light hydrological model:
- Parameters: The 14 calibrated model parameters
- State: The mutable state variables tracked during simulation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """HBV-light calibrated parameters.

    Attributes:
        tt: Threshold temperature for rain/snow partition [C].
        cfmax: Degree-day factor for snowmelt [mm/C/d].
        sfcf: Snowfall correction factor [-].
        cwh: Water holding capacity of snow [-].
        cfr: Refreezing coefficient [-].
        fc: Field capacity / maximum soil moisture storage [mm].
        lp: Limit for potential evapotranspiration as fraction of FC [-].
        beta: Shape coefficient for soil moisture runoff [-].
        k0: Surface/quick flow recession coefficient [1/d].
        k1: Interflow recession coefficient [1/d].
        k2: Baseflow recession coefficient [1/d].
        perc: Maximum percolation rate to lower zone [mm/d].
        uzl: Upper zone threshold for K0 flow [mm].
        maxbas: Length of triangular unit hydrograph [d].
    """

    tt: float
    cfmax: float
    sfcf: float
    cwh: float
    cfr: float
    fc: float
    lp: float
    beta: float
    k0: float
    k1: float
    k2: float
    perc: float
    uzl: float
    maxbas: float

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to 1D array for Numba."""
        arr = np.array(
            [
                self.tt,
                self.cfmax,
                self.sfcf,
                self.cwh,
                self.cfr,
                self.fc,
                self.lp,
                self.beta,
                self.k0,
                self.k1,
                self.k2,
                self.perc,
                self.uzl,
                self.maxbas,
            ],
            dtype=np.float64,
        )
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Reconstruct Parameters from array."""
        return cls(
            tt=float(arr[0]),
            cfmax=float(arr[1]),
            sfcf=float(arr[2]),
            cwh=float(arr[3]),
            cfr=float(arr[4]),
            fc=float(arr[5]),
            lp=float(arr[6]),
            beta=float(arr[7]),
            k0=float(arr[8]),
            k1=float(arr[9]),
            k2=float(arr[10]),
            perc=float(arr[11]),
            uzl=float(arr[12]),
            maxbas=float(arr[13]),
        )


@dataclass
class State:
    """HBV-light model state variables.

    Mutable state that evolves during simulation. Contains per-zone states
    for snow and soil, lumped groundwater stores, and routing buffer.

    Attributes:
        zone_states: Per-zone states, shape (n_zones, 3).
                     Each row: [snow_pack, liquid_water_in_snow, soil_moisture].
        upper_zone: SUZ - Upper groundwater zone storage [mm].
        lower_zone: SLZ - Lower groundwater zone storage [mm].
        routing_buffer: Buffer for triangular unit hydrograph convolution, 7 elements.
    """

    zone_states: np.ndarray  # shape (n_zones, 3): [SP, LW, SM] per zone
    upper_zone: float  # SUZ [mm]
    lower_zone: float  # SLZ [mm]
    routing_buffer: np.ndarray  # shape (7,) for MAXBAS convolution

    @property
    def n_zones(self) -> int:
        """Number of elevation zones."""
        return self.zone_states.shape[0]

    @classmethod
    def initialize(cls, params: Parameters, n_zones: int = 1) -> State:
        """Create initial state from parameters.

        Uses standard initialization:
        - Snow pack at zero
        - Liquid water in snow at zero
        - Soil moisture at 50% of field capacity
        - Upper zone at zero
        - Lower zone at zero
        - Routing buffer at zero
        """
        zone_states = np.zeros((n_zones, 3), dtype=np.float64)
        # Initialize soil moisture at 50% of FC (column index 2)
        zone_states[:, 2] = 0.5 * params.fc

        return cls(
            zone_states=zone_states,
            upper_zone=0.0,
            lower_zone=0.0,
            routing_buffer=np.zeros(7, dtype=np.float64),
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to 1D array for Numba.

        Layout: [zone_states.flatten(), upper_zone, lower_zone, routing_buffer]
        """
        n_zones = self.n_zones
        size = n_zones * 3 + 2 + 7
        arr = np.empty(size, dtype=np.float64)

        # Flatten zone states
        arr[: n_zones * 3] = self.zone_states.flatten()
        arr[n_zones * 3] = self.upper_zone
        arr[n_zones * 3 + 1] = self.lower_zone
        arr[n_zones * 3 + 2 :] = self.routing_buffer

        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray, n_zones: int = 1) -> State:
        """Reconstruct State from array."""
        zone_states = arr[: n_zones * 3].reshape(n_zones, 3).copy()
        upper_zone = float(arr[n_zones * 3])
        lower_zone = float(arr[n_zones * 3 + 1])
        routing_buffer = arr[n_zones * 3 + 2 : n_zones * 3 + 9].copy()

        return cls(
            zone_states=zone_states,
            upper_zone=upper_zone,
            lower_zone=lower_zone,
            routing_buffer=routing_buffer,
        )
