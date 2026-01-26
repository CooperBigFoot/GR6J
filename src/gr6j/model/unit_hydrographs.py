"""GR6J unit hydrograph functions.

This module implements the S-curve based unit hydrographs (UH1 and UH2) used
in the GR6J rainfall-runoff model for temporal distribution of effective rainfall.

UH1 is faster (length NH=20 days), UH2 is slower (length 2*NH=40 days).
"""

import numpy as np

from .constants import NH, D


def compute_uh_ordinates(x4: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit hydrograph ordinates for UH1 and UH2.

    The ordinates are derived from S-curve functions that describe the
    cumulative response. UH1 has a faster response (base time X4) while
    UH2 has a slower response (base time 2*X4).

    Args:
        x4: Unit hydrograph time constant (days). Controls routing speed.

    Returns:
        Tuple of (uh1_ordinates, uh2_ordinates) where:
            - uh1_ordinates: Array of length NH (20) for the slow branch
            - uh2_ordinates: Array of length 2*NH (40) for the fast branch

    Notes:
        The S-curves are defined as:

        SS1(i):
            - 0 if i <= 0
            - (i/X4)^D if 0 < i < X4
            - 1 if i >= X4

        SS2(i):
            - 0 if i <= 0
            - 0.5 * (i/X4)^D if 0 < i <= X4
            - 1 - 0.5 * (2 - i/X4)^D if X4 < i < 2*X4
            - 1 if i >= 2*X4

        Where D = 2.5 (fixed exponent).

        Ordinates are computed as: UH(i) = SS(i) - SS(i-1)
    """
    uh1_ordinates = np.zeros(NH, dtype=np.float64)
    uh2_ordinates = np.zeros(2 * NH, dtype=np.float64)

    # Compute UH1 ordinates
    for i in range(1, NH + 1):
        ss1_i = _ss1(i, x4)
        ss1_i_minus_1 = _ss1(i - 1, x4)
        uh1_ordinates[i - 1] = ss1_i - ss1_i_minus_1

    # Compute UH2 ordinates
    for i in range(1, 2 * NH + 1):
        ss2_i = _ss2(i, x4)
        ss2_i_minus_1 = _ss2(i - 1, x4)
        uh2_ordinates[i - 1] = ss2_i - ss2_i_minus_1

    return uh1_ordinates, uh2_ordinates


def _ss1(i: float, x4: float) -> float:
    """Compute UH1 S-curve value at position i.

    Args:
        i: Position in the unit hydrograph (days).
        x4: Unit hydrograph time constant (days).

    Returns:
        S-curve value between 0 and 1.
    """
    if i <= 0:
        return 0.0
    elif i < x4:
        return (i / x4) ** D
    else:
        return 1.0


def _ss2(i: float, x4: float) -> float:
    """Compute UH2 S-curve value at position i.

    Args:
        i: Position in the unit hydrograph (days).
        x4: Unit hydrograph time constant (days).

    Returns:
        S-curve value between 0 and 1.
    """
    if i <= 0:
        return 0.0
    elif i <= x4:
        return 0.5 * (i / x4) ** D
    elif i < 2 * x4:
        return 1.0 - 0.5 * (2.0 - i / x4) ** D
    else:
        return 1.0


def convolve_uh(uh_states: np.ndarray, pr_input: float, uh_ordinates: np.ndarray) -> tuple[np.ndarray, float]:
    """Perform unit hydrograph convolution for one time step.

    This function updates the unit hydrograph state array by shifting values
    and adding the contribution of new rainfall input weighted by the UH ordinates.

    Args:
        uh_states: Current unit hydrograph state array (will not be modified).
        pr_input: Effective rainfall input for this time step (mm/day).
        uh_ordinates: Unit hydrograph ordinates (from compute_uh_ordinates).

    Returns:
        Tuple of (new_states, output) where:
            - new_states: Updated state array after convolution
            - output: The UH output for this time step (mm/day), which is
              the value of uh_states[0] BEFORE the update

    Notes:
        The convolution follows the algorithm:
            1. Output is the first element of the current states
            2. For k = 0 to len-2: new_states[k] = uh_states[k+1] + uh_ordinates[k] * pr_input
            3. For the last element: new_states[-1] = uh_ordinates[-1] * pr_input
    """
    # Output is the first element before updating
    output = uh_states[0]

    # Create new state array
    new_states = np.zeros_like(uh_states)

    # Update states: shift and add weighted input
    n = len(uh_states)
    for k in range(n - 1):
        new_states[k] = uh_states[k + 1] + uh_ordinates[k] * pr_input

    # Last state only gets the weighted input (no shift from beyond array)
    new_states[-1] = uh_ordinates[-1] * pr_input

    return new_states, output
