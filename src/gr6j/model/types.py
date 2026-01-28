"""GR6J data structures for parameters and state variables.

This module defines the core data types used by the GR6J hydrological model:
- Parameters: The 6 calibrated model parameters
- State: The mutable state variables tracked during simulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .constants import NH

if TYPE_CHECKING:
    from ..cemaneige import CemaNeige

logger = logging.getLogger(__name__)


# Parameter bounds for validation warnings
_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
    "x4": (0.5, 10.0),
    "x5": (-4.0, 4.0),
    "x6": (0.01, 20.0),
}


def _warn_if_outside_bounds(params: Parameters) -> None:
    """Log warnings for parameters outside typical calibration ranges.

    This does not raise errors - parameters outside bounds may still be valid
    for specific catchments or research purposes.
    """
    for name, (lower, upper) in _PARAMETER_BOUNDS.items():
        value = getattr(params, name)
        if value < lower or value > upper:
            logger.warning(
                "Parameter %s=%.4f is outside typical range [%.2f, %.2f]",
                name,
                value,
                lower,
                upper,
            )


@dataclass(frozen=True)
class Parameters:
    """GR6J calibrated parameters.

    All 6 parameters that define the model behavior, plus optional snow module
    configuration. This is a frozen dataclass to prevent accidental modification
    during simulation.

    Attributes:
        x1: Production store capacity [mm]. Typical range [1, 2500].
        x2: Intercatchment exchange coefficient [mm/day]. Typical range [-5, 5].
        x3: Routing store capacity [mm]. Typical range [1, 1000].
        x4: Unit hydrograph time constant [days]. Typical range [0.5, 10].
        x5: Intercatchment exchange threshold [-]. Typical range [-4, 4].
        x6: Exponential store scale parameter [mm]. Typical range [0.01, 20].
        snow: Optional CemaNeige parameters for snow module. When provided,
            the model will preprocess precipitation through the snow module.
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Intercatchment exchange coefficient [mm/day]
    x3: float  # Routing store capacity [mm]
    x4: float  # Unit hydrograph time constant [days]
    x5: float  # Intercatchment exchange threshold [-]
    x6: float  # Exponential store scale parameter [mm]

    snow: CemaNeige | None = None  # Optional snow module parameters

    # Class-level reference to bounds for external access
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = _PARAMETER_BOUNDS

    def __post_init__(self) -> None:
        """Validate parameters and warn if outside typical ranges."""
        _warn_if_outside_bounds(self)

    @property
    def has_snow(self) -> bool:
        """Return True if snow module is enabled."""
        return self.snow is not None


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
