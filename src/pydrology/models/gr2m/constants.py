"""GR2M numerical constants.

These are fixed values used throughout the GR2M model computations.
Values are derived from the original Fortran implementation (airGR).
"""

from pydrology.types import Resolution

# Numerical safeguards to prevent overflow
MAX_TANH_ARG: float = 13.0  # Maximum argument for tanh in production store

# Routing constant
ROUTING_DENOMINATOR: float = 60.0  # Constant in quadratic routing equation

# Model contract constants
PARAM_NAMES: tuple[str, ...] = ("x1", "x2")
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),  # Production store capacity [mm]
    "x2": (0.2, 2.0),     # Groundwater exchange coefficient [-]
}
STATE_SIZE: int = 2
SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.monthly,)
