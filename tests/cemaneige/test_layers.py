"""Backward compatibility tests for cemaneige.layers imports.

The actual elevation tests have moved to tests/utils/test_elevation.py.
These tests verify that the re-exports from cemaneige.layers still work.
"""

import pytest


class TestBackwardCompatibility:
    """Verify backward compatible imports from cemaneige.layers."""

    def test_derive_layers_importable(self) -> None:
        """derive_layers can be imported from cemaneige.layers."""
        from pydrology.cemaneige.layers import derive_layers

        assert callable(derive_layers)

    def test_extrapolate_temperature_importable(self) -> None:
        """extrapolate_temperature can be imported from cemaneige.layers."""
        from pydrology.cemaneige.layers import extrapolate_temperature

        assert callable(extrapolate_temperature)

    def test_extrapolate_precipitation_importable(self) -> None:
        """extrapolate_precipitation can be imported from cemaneige.layers."""
        from pydrology.cemaneige.layers import extrapolate_precipitation

        assert callable(extrapolate_precipitation)

    def test_constants_importable(self) -> None:
        """Elevation constants can be imported from cemaneige.layers."""
        from pydrology.cemaneige.layers import (
            ELEV_CAP_PRECIP,
            GRAD_P_DEFAULT,
            GRAD_T_DEFAULT,
        )

        assert GRAD_T_DEFAULT == 0.6
        assert GRAD_P_DEFAULT == 0.00041
        assert ELEV_CAP_PRECIP == 4000.0
