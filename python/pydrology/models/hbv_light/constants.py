"""HBV-light numerical constants.

These are fixed values used throughout the HBV-light model computations.
Includes parameter bounds based on literature values and state size constants
for the snow routine, soil moisture routine, and response routine.
"""

from pydrology.types import Resolution

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
    "tt": (-2.5, 2.5),  # Threshold temperature [°C]
    "cfmax": (0.5, 10.0),  # Degree-day factor [mm/°C/d]
    "sfcf": (0.4, 1.4),  # Snowfall correction factor [-]
    "cwh": (0.0, 0.2),  # Water holding capacity of snow [-]
    "cfr": (0.0, 0.2),  # Refreezing coefficient [-]
    "fc": (50.0, 700.0),  # Field capacity [mm]
    "lp": (0.3, 1.0),  # Limit for potential ET [-]
    "beta": (1.0, 6.0),  # Shape coefficient [-]
    "k0": (0.05, 0.99),  # Surface flow recession [1/d]
    "k1": (0.01, 0.5),  # Interflow recession [1/d]
    "k2": (0.001, 0.2),  # Baseflow recession [1/d]
    "perc": (0.0, 6.0),  # Maximum percolation rate [mm/d]
    "uzl": (0.0, 100.0),  # Upper zone threshold [mm]
    "maxbas": (1.0, 7.0),  # Routing time [d]
}

# Maximum routing buffer size (MAXBAS parameter upper bound, determines UH length)
MAXBAS_MAX: int = 7

# State size constants
ZONE_STATE_SIZE: int = 3  # SP, LW, SM per elevation zone (snow pack, liquid water, soil moisture)
LUMPED_STATE_SIZE: int = 2  # SUZ, SLZ (upper zone storage, lower zone storage)
ROUTING_BUFFER_SIZE: int = 7  # Fixed buffer for triangular unit hydrograph


def compute_state_size(n_zones: int = 1) -> int:
    """Compute total state size for given number of elevation zones.

    State layout: [zone_states (n_zones * 3), SUZ, SLZ, routing_buffer (7)]
    """
    return n_zones * ZONE_STATE_SIZE + LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE


# Total state vector size for single-zone (lumped) model
STATE_SIZE: int = compute_state_size(1)  # = 12

# Physics constant
T_MELT: float = 0.0  # Default threshold temperature for snow/rain partitioning [°C]

SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)
