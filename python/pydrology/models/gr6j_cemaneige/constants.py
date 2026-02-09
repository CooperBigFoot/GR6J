"""GR6J-CemaNeige coupled model constants.

Fixed values and model contract constants for the coupled GR6J-CemaNeige model.
Combines GR6J routing parameters with CemaNeige snow module parameters.
"""

from pydrology.types import Resolution

# Model contract constants (8 parameters: 6 GR6J + 2 CemaNeige)
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "ctg", "kf")

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),  # Production store capacity [mm]
    "x2": (-5.0, 5.0),  # Intercatchment exchange coefficient [mm/day]
    "x3": (1.0, 1000.0),  # Routing store capacity [mm]
    "x4": (0.5, 10.0),  # Unit hydrograph time constant [days]
    "x5": (-4.0, 4.0),  # Intercatchment exchange threshold [-]
    "x6": (1.0, 50.0),  # Exponential store scale parameter [mm]
    "ctg": (0.0, 1.0),  # Thermal state weighting coefficient [-]
    "kf": (0.0, 10.0),  # Degree-day melt factor [mm/°C/day]
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
