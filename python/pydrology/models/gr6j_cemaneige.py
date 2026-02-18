"""GR6J-CemaNeige coupled model.

Public API for the coupled GR6J hydrological model with CemaNeige snow module.
Provides an 8-parameter model combining rainfall-runoff (GR6J) with snow
accumulation and melt (CemaNeige).
Core computations are executed in Rust via PyO3.

Design: Unified multi-layer code path. If catchment.input_elevation is None,
extrapolation is skipped (single-layer behavior). One code path handles all cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from pydrology.cemaneige.constants import GTHRESHOLD_FACTOR
from pydrology.models.gr6j import NH
from pydrology.types import Catchment, PrecipGradientType, Resolution

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import ForcingData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model contract constants (8 parameters: 6 GR6J + 2 CemaNeige)
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "ctg", "kf")

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
    "x4": (0.5, 10.0),
    "x5": (-4.0, 4.0),
    "x6": (1.0, 50.0),
    "ctg": (0.0, 1.0),
    "kf": (0.0, 10.0),
}

# State sizes
STATE_SIZE_BASE: int = 63  # GR6J state size (3 stores + 20 UH1 + 40 UH2)
SNOW_LAYER_STATE_SIZE: int = 4  # Per-layer snow state [g, etg, gthreshold, glocalmax]


def compute_state_size(n_layers: int) -> int:
    """Compute total state size for given number of snow layers.

    Args:
        n_layers: Number of elevation bands for snow module.

    Returns:
        Total state array size.
    """
    return STATE_SIZE_BASE + n_layers * SNOW_LAYER_STATE_SIZE


SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)

# STATE_SIZE for the default single-layer case (required by registry)
STATE_SIZE: int = compute_state_size(n_layers=1)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


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
        kf: Degree-day melt factor (CemaNeige) [mm/deg C/day].
            Controls the rate of snowmelt per degree above the melt threshold.
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Intercatchment exchange coefficient [mm/day]
    x3: float  # Routing store capacity [mm]
    x4: float  # Unit hydrograph time constant [days]
    x5: float  # Intercatchment exchange threshold [-]
    x6: float  # Exponential store scale parameter [mm]
    ctg: float  # Thermal state weighting coefficient [-]
    kf: float  # Degree-day melt factor [mm/deg C/day]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to a 1D array for array protocol.

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
        base_gthreshold = GTHRESHOLD_FACTOR * catchment.mean_annual_solid_precip
        snow_layer_states = np.zeros((n_layers, SNOW_LAYER_STATE_SIZE), dtype=np.float64)

        # Per-band gthreshold scaling when multi-layer with elevation data
        can_scale = (
            n_layers > 1
            and catchment.hypsometric_curve is not None
            and catchment.input_elevation is not None
        )
        if can_scale:
            from pydrology.utils.elevation import ELEV_CAP_PRECIP, GRAD_P_DEFAULT, GRAD_P_LINEAR_DEFAULT, derive_layers

            layer_elevations, _ = derive_layers(catchment.hypsometric_curve, n_layers)
            use_linear = catchment.precip_gradient_type == PrecipGradientType.linear
            if catchment.precip_gradient is not None:
                precip_gradient = catchment.precip_gradient
            elif use_linear:
                precip_gradient = GRAD_P_LINEAR_DEFAULT
            else:
                precip_gradient = GRAD_P_DEFAULT
            input_elev_eff = min(catchment.input_elevation, ELEV_CAP_PRECIP)
            for i in range(n_layers):
                layer_elev_eff = min(float(layer_elevations[i]), ELEV_CAP_PRECIP)
                if use_linear:
                    ratio = max(0.0, 1.0 + precip_gradient * (layer_elev_eff - input_elev_eff))
                else:
                    ratio = math.exp(precip_gradient * (layer_elev_eff - input_elev_eff))
                gthreshold_i = base_gthreshold * ratio
                snow_layer_states[i, 2] = gthreshold_i
                snow_layer_states[i, 3] = gthreshold_i
        else:
            for i in range(n_layers):
                snow_layer_states[i, 2] = base_gthreshold
                snow_layer_states[i, 3] = base_gthreshold

        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.5 * params.x3,
            exponential_store=0.0,
            uh1_states=np.zeros(NH),
            uh2_states=np.zeros(2 * NH),
            snow_layer_states=snow_layer_states,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for array protocol.

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


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GR6JCemaNeigeFluxes:
    """Combined GR6J and CemaNeige flux outputs as arrays.

    All arrays have the same length as the input forcing data.
    Contains 10 snow fields + 20 GR6J fields = 30 total fields.

    Snow-related attributes (from CemaNeige):
        precip_raw: Original precipitation before snow processing [mm/day].
        snow_pliq: Liquid precipitation (rain) [mm/day].
        snow_psol: Solid precipitation (snow) [mm/day].
        snow_pack: Snow pack water equivalent after melt [mm].
        snow_thermal_state: Thermal state of snow pack [deg C].
        snow_gratio: Snow cover fraction after melt [-].
        snow_pot_melt: Potential melt [mm/day].
        snow_melt: Actual melt [mm/day].
        snow_pliq_and_melt: Total liquid output to GR6J [mm/day].
        snow_temp: Air temperature [deg C].

    GR6J-related attributes:
        pet: Potential evapotranspiration [mm/day].
        precip: Precipitation input to GR6J (= snow_pliq_and_melt) [mm/day].
        production_store: Production store level after timestep [mm].
        net_rainfall: Net rainfall after interception [mm/day].
        storage_infiltration: Water infiltrating to production store [mm/day].
        actual_et: Actual evapotranspiration [mm/day].
        percolation: Percolation from production store [mm/day].
        effective_rainfall: Total effective rainfall after percolation [mm/day].
        q9: Output from UH1 (90% branch) [mm/day].
        q1: Output from UH2 (10% branch) [mm/day].
        routing_store: Routing store level after timestep [mm].
        exchange: Groundwater exchange potential [mm/day].
        actual_exchange_routing: Actual exchange from routing store [mm/day].
        actual_exchange_direct: Actual exchange from direct branch [mm/day].
        actual_exchange_total: Total actual exchange [mm/day].
        qr: Outflow from routing store [mm/day].
        qrexp: Outflow from exponential store [mm/day].
        exponential_store: Exponential store level after timestep [mm].
        qd: Direct branch outflow [mm/day].
        streamflow: Total simulated streamflow [mm/day].
    """

    # Snow-related outputs (10 fields)
    precip_raw: np.ndarray
    snow_pliq: np.ndarray
    snow_psol: np.ndarray
    snow_pack: np.ndarray
    snow_thermal_state: np.ndarray
    snow_gratio: np.ndarray
    snow_pot_melt: np.ndarray
    snow_melt: np.ndarray
    snow_pliq_and_melt: np.ndarray
    snow_temp: np.ndarray

    # GR6J-related outputs (20 fields)
    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    net_rainfall: np.ndarray
    storage_infiltration: np.ndarray
    actual_et: np.ndarray
    percolation: np.ndarray
    effective_rainfall: np.ndarray
    q9: np.ndarray
    q1: np.ndarray
    routing_store: np.ndarray
    exchange: np.ndarray
    actual_exchange_routing: np.ndarray
    actual_exchange_direct: np.ndarray
    actual_exchange_total: np.ndarray
    qr: np.ndarray
    qrexp: np.ndarray
    exponential_store: np.ndarray
    qd: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


# ---------------------------------------------------------------------------
# Run / Step
# ---------------------------------------------------------------------------


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    layer_elevations: np.ndarray | None = None,
    layer_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
    precip_gradient_type: str | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR6J-CemaNeige coupled model.

    Processes snow accumulation/melt through CemaNeige, then routes
    the liquid water through GR6J.

    Args:
        state: Current combined model state.
        params: Model parameters (8 total).
        precip: Daily precipitation [mm/day].
        pet: Daily potential evapotranspiration [mm/day].
        temp: Daily air temperature [deg C].
        uh1_ordinates: Pre-computed UH1 ordinates.
        uh2_ordinates: Pre-computed UH2 ordinates.
        layer_elevations: Elevation of each layer [m]. Required for multi-layer.
        layer_fractions: Area fraction of each layer [-]. Required for multi-layer.
        input_elevation: Elevation of forcing data [m]. If None, skip extrapolation.
        temp_gradient: Temperature lapse rate [deg C/100m]. Default 0.6.
        precip_gradient: Precipitation gradient [m^-1]. Default 0.00041.

    Returns:
        Tuple of (new_state, fluxes) where fluxes contains all snow and GR6J outputs.

    Delegates to the compiled Rust backend for a single timestep.
    """
    from pydrology._core import cemaneige as _rust

    n_layers = state.n_layers

    # Set defaults for layer config
    if layer_elevations is None:
        layer_elevations = np.zeros(n_layers)
    if layer_fractions is None:
        layer_fractions = np.ones(n_layers) / n_layers

    # Convert state and params to arrays
    state_arr = np.ascontiguousarray(state, dtype=np.float64)
    params_arr = np.ascontiguousarray(params, dtype=np.float64)

    new_state_arr, fluxes_dict = _rust.gr6j_cemaneige_step(
        state_arr,
        params_arr,
        precip,
        pet,
        temp,
        np.ascontiguousarray(uh1_ordinates, dtype=np.float64),
        np.ascontiguousarray(uh2_ordinates, dtype=np.float64),
        np.ascontiguousarray(layer_elevations, dtype=np.float64),
        np.ascontiguousarray(layer_fractions, dtype=np.float64),
        input_elevation,
        temp_gradient,
        precip_gradient,
        precip_gradient_type=precip_gradient_type,
    )

    # Reconstruct state
    new_state = State.from_array(new_state_arr, n_layers)

    # Convert numpy scalars to Python floats for consistency
    fluxes: dict[str, float] = {k: float(v) for k, v in fluxes_dict.items()}

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
    *,
    catchment: Catchment | None = None,
) -> ModelOutput[GR6JCemaNeigeFluxes]:
    """Run the GR6J-CemaNeige coupled model over a timeseries.

    Executes the coupled snow-hydrology model for each timestep in the input
    forcing data. Snow is processed through CemaNeige (with optional multi-layer
    elevation bands), and the liquid water output is routed through GR6J.

    Args:
        params: Model parameters (8 total: 6 GR6J + 2 CemaNeige).
        forcing: Input forcing data with precip, pet, and temp arrays.
        catchment: Catchment properties (mean_annual_solid_precip, elevation config).
        initial_state: Initial model state. If None, uses State.initialize().

    Returns:
        ModelOutput containing GR6JCemaNeigeFluxes, SnowOutput, and optionally
        SnowLayerOutputs (for multi-layer simulations).

    Raises:
        ValueError: If forcing.temp is None (temperature required for snow module).

    Delegates to the compiled Rust backend for the full simulation loop.
    """
    from pydrology._core import cemaneige as _rust
    from pydrology.models.cemaneige.outputs import SnowLayerOutputs, SnowOutput
    from pydrology.outputs import ModelOutput
    from pydrology.utils.elevation import (
        GRAD_P_DEFAULT,
        GRAD_T_DEFAULT,
        derive_layers,
    )

    # Validate catchment is provided
    if catchment is None:
        msg = "catchment is required for GR6J-CemaNeige (provides mean_annual_solid_precip and layer config)"
        raise ValueError(msg)

    # Validate temperature is provided
    if forcing.temp is None:
        msg = "Temperature data (forcing.temp) is required for the GR6J-CemaNeige model"
        raise ValueError(msg)

    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR6J-CemaNeige supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    n_timesteps = len(forcing)
    n_layers = catchment.n_layers

    # Derive layer properties if multi-layer
    if n_layers > 1 and catchment.hypsometric_curve is not None:
        layer_elevations, layer_fractions = derive_layers(catchment.hypsometric_curve, n_layers)
    else:
        # Single layer: use dummy values (extrapolation will be skipped)
        layer_elevations = np.zeros(n_layers)
        layer_fractions = np.ones(n_layers) / n_layers

    # Initialize state if not provided
    state = State.initialize(params, catchment) if initial_state is None else initial_state

    # Set gradients (use defaults if not specified)
    temp_gradient = catchment.temp_gradient if catchment.temp_gradient is not None else GRAD_T_DEFAULT

    # Resolve gradient type and default
    precip_gradient_type_str = catchment.precip_gradient_type.value
    if catchment.precip_gradient is not None:
        precip_gradient = catchment.precip_gradient
    elif catchment.precip_gradient_type == PrecipGradientType.linear:
        from pydrology.utils.elevation import GRAD_P_LINEAR_DEFAULT
        precip_gradient = GRAD_P_LINEAR_DEFAULT
    else:
        precip_gradient = GRAD_P_DEFAULT

    # Use NaN to signal no extrapolation
    input_elevation = catchment.input_elevation if catchment.input_elevation is not None else float("nan")

    # Convert to arrays
    initial_state_arr = np.ascontiguousarray(state, dtype=np.float64)
    params_arr = np.ascontiguousarray(params, dtype=np.float64)

    # Call Rust backend (returns a single merged dict)
    result = _rust.gr6j_cemaneige_run(
        params_arr,
        np.ascontiguousarray(forcing.precip, dtype=np.float64),
        np.ascontiguousarray(forcing.pet, dtype=np.float64),
        np.ascontiguousarray(forcing.temp, dtype=np.float64),
        initial_state_arr,
        None,  # uh1_ordinates (computed internally)
        None,  # uh2_ordinates (computed internally)
        n_layers,
        np.ascontiguousarray(layer_elevations, dtype=np.float64),
        np.ascontiguousarray(layer_fractions, dtype=np.float64),
        input_elevation,
        temp_gradient,
        precip_gradient,
        catchment.mean_annual_solid_precip,
        precip_gradient_type=precip_gradient_type_str,
    )

    # Build combined flux output
    combined_fluxes = GR6JCemaNeigeFluxes(
        # Snow outputs (10 fields)
        precip_raw=forcing.precip.copy(),
        snow_pliq=result["snow_pliq"],
        snow_psol=result["snow_psol"],
        snow_pack=result["snow_pack"],
        snow_thermal_state=result["snow_thermal_state"],
        snow_gratio=result["snow_gratio"],
        snow_pot_melt=result["snow_pot_melt"],
        snow_melt=result["snow_melt"],
        snow_pliq_and_melt=result["snow_pliq_and_melt"],
        snow_temp=result["snow_temp"],
        # GR6J outputs (20 fields)
        pet=result["pet"],
        precip=result["precip"],
        production_store=result["production_store"],
        net_rainfall=result["net_rainfall"],
        storage_infiltration=result["storage_infiltration"],
        actual_et=result["actual_et"],
        percolation=result["percolation"],
        effective_rainfall=result["effective_rainfall"],
        q9=result["q9"],
        q1=result["q1"],
        routing_store=result["routing_store"],
        exchange=result["exchange"],
        actual_exchange_routing=result["actual_exchange_routing"],
        actual_exchange_direct=result["actual_exchange_direct"],
        actual_exchange_total=result["actual_exchange_total"],
        qr=result["qr"],
        qrexp=result["qrexp"],
        exponential_store=result["exponential_store"],
        qd=result["qd"],
        streamflow=result["streamflow"],
    )

    # Build SnowOutput for backward compatibility
    snow_output = SnowOutput(
        precip_raw=forcing.precip.copy(),
        snow_pliq=result["snow_pliq"],
        snow_psol=result["snow_psol"],
        snow_pack=result["snow_pack"],
        snow_thermal_state=result["snow_thermal_state"],
        snow_gratio=result["snow_gratio"],
        snow_pot_melt=result["snow_pot_melt"],
        snow_melt=result["snow_melt"],
        snow_pliq_and_melt=result["snow_pliq_and_melt"],
        snow_temp=result["snow_temp"],
        snow_gthreshold=np.full(n_timesteps, catchment.mean_annual_solid_precip * 0.9),
        snow_glocalmax=np.full(n_timesteps, catchment.mean_annual_solid_precip * 0.9),
    )

    # Build per-layer outputs if multi-layer
    snow_layers: SnowLayerOutputs | None = None
    if n_layers > 1:
        snow_layers = SnowLayerOutputs(
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            snow_pack=result["layer_snow_pack"],
            snow_thermal_state=result["layer_snow_thermal_state"],
            snow_gratio=result["layer_snow_gratio"],
            snow_melt=result["layer_snow_melt"],
            snow_pliq_and_melt=result["layer_snow_pliq_and_melt"],
            layer_temp=result["layer_temp"],
            layer_precip=result["layer_precip"],
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=combined_fluxes,
        snow=snow_output,
        snow_layers=snow_layers,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_BOUNDS",
    "GR6JCemaNeigeFluxes",
    "PARAM_NAMES",
    "Parameters",
    "SNOW_LAYER_STATE_SIZE",
    "STATE_SIZE",
    "STATE_SIZE_BASE",
    "SUPPORTED_RESOLUTIONS",
    "State",
    "compute_state_size",
    "run",
    "step",
]

# Auto-register
import pydrology.models.gr6j_cemaneige as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("gr6j_cemaneige", _self)
