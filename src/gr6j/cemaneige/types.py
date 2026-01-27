"""CemaNeige data structures for parameters and state variables.

This module defines the core data types used by the CemaNeige snow model:
- CemaNeige: The calibrated model parameters
- CemaNeigeSingleLayerState: The mutable state variables for single-layer mode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


# Parameter bounds for validation warnings
_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "ctg": (0.0, 1.0),
    "kf": (0.0, 200.0),
    "mean_annual_solid_precip": (0.0, 10000.0),
}


def _warn_if_outside_bounds(params: CemaNeige) -> None:
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
class CemaNeige:
    """CemaNeige calibrated parameters.

    Parameters that define the snow model behavior. This is a frozen dataclass
    to prevent accidental modification during simulation.

    Attributes:
        ctg: Thermal state weighting coefficient [-]. Controls the inertia of
            the snow pack thermal state. Typical range [0, 1].
        kf: Degree-day melt factor [mm/°C/day]. Controls the rate of snowmelt
            per degree above the melt threshold. Typical range [1, 10], but
            can extend to [0, 200] for extreme cases.
        mean_annual_solid_precip: Mean annual solid precipitation [mm]. Used
            to compute the gthreshold value for hysteresis. Typical range
            [0, 10000].
    """

    ctg: float  # Thermal state weighting coefficient [-]
    kf: float  # Degree-day melt factor [mm/°C/day]
    mean_annual_solid_precip: float  # Mean annual solid precipitation [mm]

    # Class-level reference to bounds for external access
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = _PARAMETER_BOUNDS

    def __post_init__(self) -> None:
        """Validate parameters and warn if outside typical ranges."""
        _warn_if_outside_bounds(self)


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
