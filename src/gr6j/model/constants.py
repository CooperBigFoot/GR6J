"""GR6J numerical constants.

These are fixed values used throughout the GR6J model computations.
Values are derived from the original Fortran implementation (airGR).
"""

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
