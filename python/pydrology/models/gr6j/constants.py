"""GR6J numerical constants.

These are fixed values used throughout the GR6J model computations.
Values are derived from the original Fortran implementation (airGR).
"""

from pydrology.types import Resolution

# Routing split fractions
B: float = 0.9  # Fraction of PR to UH1 (slow branch)
C: float = 0.4  # Fraction of UH1 output to exponential store

# Unit hydrograph parameters
D: float = 2.5  # S-curve exponent
NH: int = 20  # UH1 length (days), UH2 is 2*NH = 40 days

# Percolation constant: (9/4)^4 = 2.25^4
PERC_CONSTANT: float = 25.62890625

# Numerical safeguards to prevent overflow
MAX_TANH_ARG: float = 13.0  # Maximum argument for tanh in production store
MAX_EXP_ARG: float = 33.0  # Maximum AR for exponential store clipping
EXP_BRANCH_THRESHOLD: float = 7.0  # Threshold for exponential store branch equations

# Model contract constants
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
    "x4": (0.5, 10.0),
    "x5": (-4.0, 4.0),
    "x6": (1.0, 50.0),
}
STATE_SIZE: int = 63
SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)
