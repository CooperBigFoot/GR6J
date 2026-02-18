"""HBV-light daily rainfall-runoff model.

Public API for the HBV-light hydrological model. Combines constants, types,
outputs, routing, and run/step functions into a single module.
Core computations are executed in Rust via PyO3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from pydrology.types import PrecipGradientType, Resolution

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model parameter names in canonical order
PARAM_NAMES: tuple[str, ...] = (
    "tt",
    "cfmax",
    "sfcf",
    "cwh",
    "cfr",
    "fc",
    "lp",
    "beta",
    "k0",
    "k1",
    "k2",
    "perc",
    "uzl",
    "maxbas",
)

# Literature-based parameter bounds
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "tt": (-2.5, 2.5),
    "cfmax": (0.5, 10.0),
    "sfcf": (0.4, 1.4),
    "cwh": (0.0, 0.2),
    "cfr": (0.0, 0.2),
    "fc": (50.0, 700.0),
    "lp": (0.3, 1.0),
    "beta": (1.0, 6.0),
    "k0": (0.05, 0.99),
    "k1": (0.01, 0.5),
    "k2": (0.001, 0.2),
    "perc": (0.0, 6.0),
    "uzl": (0.0, 100.0),
    "maxbas": (1.0, 7.0),
}

# Maximum routing buffer size (MAXBAS parameter upper bound, determines UH length)
MAXBAS_MAX: int = 7

# State size constants
ZONE_STATE_SIZE: int = 3
LUMPED_STATE_SIZE: int = 2
ROUTING_BUFFER_SIZE: int = 7


def compute_state_size(n_zones: int = 1) -> int:
    """Compute total state size for given number of elevation zones.

    State layout: [zone_states (n_zones * 3), SUZ, SLZ, routing_buffer (7)]
    """
    return n_zones * ZONE_STATE_SIZE + LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE


# Total state vector size for single-zone (lumped) model
STATE_SIZE: int = compute_state_size(1)  # = 12

# Physics constant
T_MELT: float = 0.0

SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


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
        """Convert parameters to 1D array for array protocol."""
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
        if len(arr) != len(PARAM_NAMES):
            msg = f"Expected array of length {len(PARAM_NAMES)}, got {len(arr)}"
            raise ValueError(msg)
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
        """Convert state to 1D array for array protocol.

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


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HBVLightFluxes:
    """HBV-light model flux outputs as arrays.

    All arrays have the same length as the input forcing data.

    Attributes:
        precip: Precipitation input [mm/day].
        temp: Temperature input [deg C].
        pet: Potential evapotranspiration [mm/day].
        precip_rain: Liquid precipitation [mm/day].
        precip_snow: Solid precipitation after SFCF correction [mm/day].
        snow_pack: Snow pack water equivalent after timestep [mm].
        snow_melt: Snowmelt [mm/day].
        liquid_water_in_snow: Liquid water held in snow pack [mm].
        snow_input: Total input to soil (rain + melt outflow) [mm/day].
        soil_moisture: Soil moisture after timestep [mm].
        recharge: Recharge to groundwater [mm/day].
        actual_et: Actual evapotranspiration [mm/day].
        upper_zone: Upper groundwater zone storage after timestep [mm].
        lower_zone: Lower groundwater zone storage after timestep [mm].
        q0: Surface/quick flow from upper zone [mm/day].
        q1: Interflow from upper zone [mm/day].
        q2: Baseflow from lower zone [mm/day].
        percolation: Percolation from upper to lower zone [mm/day].
        qgw: Total groundwater runoff before routing [mm/day].
        streamflow: Total simulated streamflow after routing [mm/day].
    """

    # Inputs
    precip: np.ndarray
    temp: np.ndarray
    pet: np.ndarray

    # Snow routine outputs
    precip_rain: np.ndarray
    precip_snow: np.ndarray
    snow_pack: np.ndarray
    snow_melt: np.ndarray
    liquid_water_in_snow: np.ndarray
    snow_input: np.ndarray

    # Soil routine outputs
    soil_moisture: np.ndarray
    recharge: np.ndarray
    actual_et: np.ndarray

    # Response routine outputs
    upper_zone: np.ndarray
    lower_zone: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    percolation: np.ndarray
    qgw: np.ndarray

    # Final output
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class HBVLightZoneOutputs:
    """Per-zone outputs for multi-zone HBV-light simulations.

    Contains 2D arrays with shape (n_timesteps, n_zones) for detailed
    zone-by-zone analysis.

    Attributes:
        zone_elevations: Representative elevation of each zone [m]. Shape (n_zones,).
        zone_fractions: Area fraction of each zone [-]. Shape (n_zones,).
        zone_temp: Extrapolated temperature per zone [deg C]. Shape (n_timesteps, n_zones).
        zone_precip: Extrapolated precipitation per zone [mm/day]. Shape (n_timesteps, n_zones).
        snow_pack: Snow pack water equivalent per zone [mm]. Shape (n_timesteps, n_zones).
        liquid_water_in_snow: Liquid water held in snow per zone [mm]. Shape (n_timesteps, n_zones).
        snow_melt: Snowmelt per zone [mm/day]. Shape (n_timesteps, n_zones).
        snow_input: Total input to soil per zone [mm/day]. Shape (n_timesteps, n_zones).
        soil_moisture: Soil moisture per zone [mm]. Shape (n_timesteps, n_zones).
        recharge: Recharge to groundwater per zone [mm/day]. Shape (n_timesteps, n_zones).
        actual_et: Actual evapotranspiration per zone [mm/day]. Shape (n_timesteps, n_zones).
    """

    # Zone metadata
    zone_elevations: np.ndarray  # Shape (n_zones,)
    zone_fractions: np.ndarray  # Shape (n_zones,)

    # Per-zone forcings (extrapolated)
    zone_temp: np.ndarray  # Shape (n_timesteps, n_zones)
    zone_precip: np.ndarray  # Shape (n_timesteps, n_zones)

    # Per-zone snow routine outputs
    snow_pack: np.ndarray  # Shape (n_timesteps, n_zones)
    liquid_water_in_snow: np.ndarray  # Shape (n_timesteps, n_zones)
    snow_melt: np.ndarray  # Shape (n_timesteps, n_zones)
    snow_input: np.ndarray  # Shape (n_timesteps, n_zones)

    # Per-zone soil routine outputs
    soil_moisture: np.ndarray  # Shape (n_timesteps, n_zones)
    recharge: np.ndarray  # Shape (n_timesteps, n_zones)
    actual_et: np.ndarray  # Shape (n_timesteps, n_zones)

    @property
    def n_zones(self) -> int:
        """Return the number of elevation zones."""
        return len(self.zone_elevations)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def compute_triangular_weights(maxbas: float) -> np.ndarray:
    """Compute triangular unit hydrograph weights.

    Uses analytical integration of the triangular function to compute
    weights that sum to 1.0. Supports fractional MAXBAS values.

    Args:
        maxbas: Length of triangular function [days].

    Returns:
        Array of weights with length ceil(MAXBAS), summing to 1.0.
    """
    n = int(math.ceil(maxbas))
    if n < 1:
        n = 1

    weights = np.zeros(n, dtype=np.float64)
    half = maxbas / 2.0

    for i in range(n):
        t_start = float(i)
        t_end = min(float(i + 1), maxbas)

        if t_end <= t_start:
            continue

        weight = 0.0

        # Handle rising limb portion
        if t_start < half:
            t_r_end = min(t_end, half)
            weight += (t_r_end**2 - t_start**2) / (maxbas**2)

        # Handle falling limb portion
        if t_end > half:
            t_f_start = max(t_start, half)
            weight += 2.0 * (t_end - t_f_start) / maxbas - (t_end**2 - t_f_start**2) / (maxbas**2)

        weights[i] = weight

    # Normalize to ensure sum = 1.0 (handles numerical precision)
    total = weights.sum()
    if total > 0:
        weights /= total

    return weights


# ---------------------------------------------------------------------------
# Run / Step
# ---------------------------------------------------------------------------


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    catchment: object | None = None,
    uh_weights: np.ndarray | None = None,
    zone_elevations: np.ndarray | None = None,
    zone_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the HBV-light model.

    Args:
        state: Current model state (stores and routing buffer).
        params: Model parameters.
        precip: Daily precipitation [mm/day].
        pet: Daily potential evapotranspiration [mm/day].
        temp: Daily mean temperature [C].
        catchment: Optional catchment properties (unused, for API compatibility).
        uh_weights: Pre-computed UH weights. If None, computed from params.maxbas.
        zone_elevations: Representative elevation of each zone [m]. Shape (n_zones,).
        zone_fractions: Area fraction of each zone [-]. Shape (n_zones,).
        input_elevation: Elevation of input measurement [m]. Required for extrapolation.
        temp_gradient: Temperature lapse rate [C/100m]. Default is GRAD_T_DEFAULT.
        precip_gradient: Precipitation gradient [m^-1]. Default is GRAD_P_DEFAULT.

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs (area-weighted aggregates)

    Delegates to the compiled Rust backend for a single timestep.
    """
    from pydrology._core import hbv_light as _rust

    if uh_weights is None:
        uh_weights = compute_triangular_weights(params.maxbas)

    state_arr = np.ascontiguousarray(state, dtype=np.float64)
    params_arr = np.ascontiguousarray(params, dtype=np.float64)

    new_state_arr, fluxes = _rust.hbv_step(
        state_arr,
        params_arr,
        precip,
        pet,
        temp,
        uh_weights,
    )

    new_state = State.from_array(np.asarray(new_state_arr), state.n_zones)

    fluxes_converted: dict[str, float] = {k: float(v) for k, v in fluxes.items()}
    return new_state, fluxes_converted


def run(
    params: Parameters,
    forcing: object,
    initial_state: State | None = None,
    catchment: object | None = None,
) -> ModelOutput[HBVLightFluxes]:
    """Run the HBV-light model over a timeseries.

    Args:
        params: Model parameters.
        forcing: Input forcing data with precip, pet, and temp arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).
        catchment: Optional catchment properties for elevation band support.

    Returns:
        ModelOutput containing HBVLightFluxes outputs.

    Raises:
        ValueError: If forcing.temp is None.

    Delegates to the compiled Rust backend for the full simulation loop.
    """
    from pydrology._core import hbv_light as _rust
    from pydrology.outputs import ModelOutput
    from pydrology.utils.elevation import (
        GRAD_P_DEFAULT,
        GRAD_T_DEFAULT,
        derive_layers,
    )

    if forcing.temp is None:
        raise ValueError("HBV-light requires temperature data (forcing.temp)")

    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"HBV-light supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    # Determine zone configuration
    if catchment is not None and catchment.n_layers > 1 and catchment.hypsometric_curve is not None:
        n_zones = catchment.n_layers
        zone_elevations, zone_fractions = derive_layers(catchment.hypsometric_curve, n_zones)
        input_elevation = catchment.input_elevation
        temp_gradient = catchment.temp_gradient if catchment.temp_gradient is not None else GRAD_T_DEFAULT
        if catchment.precip_gradient is not None:
            precip_gradient = catchment.precip_gradient
        elif catchment.precip_gradient_type == PrecipGradientType.linear:
            from pydrology.utils.elevation import GRAD_P_LINEAR_DEFAULT
            precip_gradient = GRAD_P_LINEAR_DEFAULT
        else:
            precip_gradient = GRAD_P_DEFAULT
        precip_gradient_type_str = catchment.precip_gradient_type.value
    else:
        n_zones = 1
        zone_elevations = None
        zone_fractions = None
        input_elevation = None
        temp_gradient = None
        precip_gradient = None
        precip_gradient_type_str = None

    # Prepare arrays
    params_arr = np.ascontiguousarray(params, dtype=np.float64)
    precip_arr = np.ascontiguousarray(forcing.precip, dtype=np.float64)
    pet_arr = np.ascontiguousarray(forcing.pet, dtype=np.float64)
    temp_arr = np.ascontiguousarray(forcing.temp, dtype=np.float64)

    # Initial state
    state_arr = None
    if initial_state is not None:
        state_arr = np.ascontiguousarray(initial_state, dtype=np.float64)

    # Call Rust backend (returns a single merged dict)
    result = _rust.hbv_run(
        params_arr,
        precip_arr,
        pet_arr,
        temp_arr,
        initial_state=state_arr,
        n_zones=n_zones,
        zone_elevations=zone_elevations.astype(np.float64) if zone_elevations is not None else None,
        zone_fractions=zone_fractions.astype(np.float64) if zone_fractions is not None else None,
        input_elevation=input_elevation,
        temp_gradient=temp_gradient,
        precip_gradient=precip_gradient,
        precip_gradient_type=precip_gradient_type_str,
    )

    # Build output object
    hbv_fluxes = HBVLightFluxes(
        precip=np.asarray(result["precip"]),
        temp=np.asarray(result["temp"]),
        pet=np.asarray(result["pet"]),
        precip_rain=np.asarray(result["precip_rain"]),
        precip_snow=np.asarray(result["precip_snow"]),
        snow_pack=np.asarray(result["snow_pack"]),
        snow_melt=np.asarray(result["snow_melt"]),
        liquid_water_in_snow=np.asarray(result["liquid_water_in_snow"]),
        snow_input=np.asarray(result["snow_input"]),
        soil_moisture=np.asarray(result["soil_moisture"]),
        recharge=np.asarray(result["recharge"]),
        actual_et=np.asarray(result["actual_et"]),
        upper_zone=np.asarray(result["upper_zone"]),
        lower_zone=np.asarray(result["lower_zone"]),
        q0=np.asarray(result["q0"]),
        q1=np.asarray(result["q1"]),
        q2=np.asarray(result["q2"]),
        percolation=np.asarray(result["percolation"]),
        qgw=np.asarray(result["qgw"]),
        streamflow=np.asarray(result["streamflow"]),
    )

    # Build per-zone outputs if multi-zone
    zone_outputs: HBVLightZoneOutputs | None = None
    if "zone_n_timesteps" in result:
        n_t = int(result["zone_n_timesteps"])
        n_z = int(result["zone_n_zones"])
        zone_outputs = HBVLightZoneOutputs(
            zone_elevations=np.asarray(result["zone_elevations"]),
            zone_fractions=np.asarray(result["zone_fractions"]),
            zone_temp=np.asarray(result["zone_temp"]).reshape(n_t, n_z),
            zone_precip=np.asarray(result["zone_precip"]).reshape(n_t, n_z),
            snow_pack=np.asarray(result["zone_snow_pack"]).reshape(n_t, n_z),
            liquid_water_in_snow=np.asarray(result["zone_liquid_water_in_snow"]).reshape(n_t, n_z),
            snow_melt=np.asarray(result["zone_snow_melt"]).reshape(n_t, n_z),
            snow_input=np.asarray(result["zone_snow_input"]).reshape(n_t, n_z),
            soil_moisture=np.asarray(result["zone_soil_moisture"]).reshape(n_t, n_z),
            recharge=np.asarray(result["zone_recharge"]).reshape(n_t, n_z),
            actual_et=np.asarray(result["zone_actual_et"]).reshape(n_t, n_z),
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=hbv_fluxes,
        snow=None,
        snow_layers=None,
        zone_outputs=zone_outputs,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_BOUNDS",
    "HBVLightFluxes",
    "HBVLightZoneOutputs",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "State",
    "SUPPORTED_RESOLUTIONS",
    "run",
    "step",
]

# Auto-register
import pydrology.models.hbv_light as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("hbv_light", _self)
