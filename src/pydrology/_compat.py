"""NumPy/Numba compatibility shim.

Patches deprecated NumPy function names that were removed in NumPy 2.0+ but are
still referenced by Numba 0.63.x. This module must be imported before numba to
ensure the compatibility aliases exist when Numba loads its numpy extensions.

This workaround is temporary and should be removed once Numba fully supports
NumPy 2.0+.
"""

import numpy as np

# Mapping of deprecated names to their NumPy 2.0+ replacements
_DEPRECATION_ALIASES: dict[str, str] = {
    "trapz": "trapezoid",
    "in1d": "isin",
    "product": "prod",
    "row_stack": "vstack",
    "cumproduct": "cumprod",
    "sometrue": "any",
    "alltrue": "all",
}

for old_name, new_name in _DEPRECATION_ALIASES.items():
    if not hasattr(np, old_name) and hasattr(np, new_name):
        setattr(np, old_name, getattr(np, new_name))
