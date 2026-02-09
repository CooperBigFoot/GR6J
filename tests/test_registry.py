"""Tests for pydrology model registry.

Tests cover model registration, retrieval, listing, and contract validation.
"""

from types import ModuleType

import numpy as np
import pytest

from pydrology import registry
from pydrology.types import Resolution


class TestModelRegistry:
    """Tests for model registry functions."""

    def test_list_models_returns_registered_models(self) -> None:
        """list_models returns a list of registered model names."""
        models = registry.list_models()

        assert isinstance(models, list)
        # At minimum, gr6j should be registered
        assert "gr6j" in models

    def test_list_models_returns_sorted_list(self) -> None:
        """list_models returns a sorted list."""
        models = registry.list_models()

        assert models == sorted(models)

    def test_get_model_returns_module(self) -> None:
        """get_model returns a module for registered model."""
        model = registry.get_model("gr6j")

        assert isinstance(model, ModuleType)
        assert hasattr(model, "run")
        assert hasattr(model, "step")
        assert hasattr(model, "Parameters")
        assert hasattr(model, "State")

    def test_get_model_raises_for_unknown_model(self) -> None:
        """get_model raises KeyError for unknown model name."""
        with pytest.raises(KeyError, match="Unknown model"):
            registry.get_model("nonexistent_model")

    def test_get_model_info_returns_dict(self) -> None:
        """get_model_info returns a dictionary with model metadata."""
        info = registry.get_model_info("gr6j")

        assert isinstance(info, dict)
        assert "param_names" in info
        assert "default_bounds" in info
        assert "state_size" in info
    def test_get_model_info_param_names(self) -> None:
        """get_model_info returns correct parameter names for GR6J."""
        info = registry.get_model_info("gr6j")

        assert info["param_names"] == ("x1", "x2", "x3", "x4", "x5", "x6")

    def test_get_model_info_default_bounds(self) -> None:
        """get_model_info returns default bounds as dict mapping param names to (min, max)."""
        info = registry.get_model_info("gr6j")

        bounds = info["default_bounds"]
        assert isinstance(bounds, dict)
        assert len(bounds) == 6  # 6 parameters for GR6J
        # Keys should match param names
        assert set(bounds.keys()) == {"x1", "x2", "x3", "x4", "x5", "x6"}
        # Each bound should be (min, max)
        for param_name, bound in bounds.items():
            assert len(bound) == 2
            assert bound[0] < bound[1], f"{param_name}: min should be < max"

    def test_get_model_info_state_size(self) -> None:
        """get_model_info returns state size as integer."""
        info = registry.get_model_info("gr6j")

        assert isinstance(info["state_size"], int)
        assert info["state_size"] > 0

    def test_get_model_info_raises_for_unknown(self) -> None:
        """get_model_info raises KeyError for unknown model."""
        with pytest.raises(KeyError, match="Unknown model"):
            registry.get_model_info("nonexistent_model")


class TestGR6JCemaNeigeRegistration:
    """Tests for the gr6j_cemaneige model registration."""

    def test_gr6j_cemaneige_is_registered(self) -> None:
        """gr6j_cemaneige model is automatically registered."""
        models = registry.list_models()

        assert "gr6j_cemaneige" in models

    def test_gr6j_cemaneige_has_8_parameters(self) -> None:
        """gr6j_cemaneige has 8 parameters (6 GR6J + 2 CemaNeige)."""
        info = registry.get_model_info("gr6j_cemaneige")

        assert len(info["param_names"]) == 8
        assert len(info["default_bounds"]) == 8

    def test_gr6j_cemaneige_param_names(self) -> None:
        """gr6j_cemaneige has correct parameter names."""
        info = registry.get_model_info("gr6j_cemaneige")

        expected = ("x1", "x2", "x3", "x4", "x5", "x6", "ctg", "kf")
        assert info["param_names"] == expected


class TestModelContractValidation:
    """Tests that verify registered models follow the contract."""

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige"])
    def test_model_has_required_exports(self, model_name: str) -> None:
        """Registered model has all required exports."""
        model = registry.get_model(model_name)

        # Required exports
        assert hasattr(model, "PARAM_NAMES")
        assert hasattr(model, "DEFAULT_BOUNDS")
        assert hasattr(model, "STATE_SIZE")
        assert hasattr(model, "Parameters")
        assert hasattr(model, "State")
        assert hasattr(model, "run")
        assert hasattr(model, "step")

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige"])
    def test_parameters_class_has_array_protocol(self, model_name: str) -> None:
        """Parameters class supports array conversion."""
        model = registry.get_model(model_name)

        assert hasattr(model.Parameters, "from_array")
        assert hasattr(model.Parameters, "__array__")

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige"])
    def test_state_class_has_array_protocol(self, model_name: str) -> None:
        """State class supports array conversion."""
        model = registry.get_model(model_name)

        assert hasattr(model.State, "from_array")
        assert hasattr(model.State, "__array__")

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige"])
    def test_param_names_matches_bounds_length(self, model_name: str) -> None:
        """PARAM_NAMES length matches DEFAULT_BOUNDS length."""
        model = registry.get_model(model_name)

        assert len(model.PARAM_NAMES) == len(model.DEFAULT_BOUNDS)

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige"])
    def test_state_size_is_positive_integer(self, model_name: str) -> None:
        """STATE_SIZE is a positive integer."""
        model = registry.get_model(model_name)

        assert isinstance(model.STATE_SIZE, int)
        assert model.STATE_SIZE > 0


class TestRegisterFunction:
    """Tests for the register function itself."""

    def test_register_validates_missing_exports(self) -> None:
        """register raises ValueError for module missing required exports."""

        class FakeModule:
            pass

        with pytest.raises(ValueError, match="missing required exports"):
            registry.register("fake_model", FakeModule())  # type: ignore[arg-type]

    def test_register_validates_parameters_methods(self) -> None:
        """register raises ValueError if Parameters lacks required methods."""

        class FakeParameters:
            pass  # Missing from_array and __array__

        class FakeState:
            @classmethod
            def from_array(cls, arr: np.ndarray) -> "FakeState":
                return cls()

            def __array__(self) -> np.ndarray:
                return np.array([])

        class FakeModule:
            PARAM_NAMES = ("x1",)
            DEFAULT_BOUNDS = ((0, 1),)
            STATE_SIZE = 1
            Parameters = FakeParameters
            State = FakeState
            SUPPORTED_RESOLUTIONS = (Resolution.daily,)

            @staticmethod
            def run() -> None:
                pass

            @staticmethod
            def step() -> None:
                pass

        with pytest.raises(ValueError, match="Parameters class is missing"):
            registry.register("fake_model", FakeModule())  # type: ignore[arg-type]

    def test_register_validates_state_methods(self) -> None:
        """register raises ValueError if State lacks required methods."""

        class FakeParameters:
            @classmethod
            def from_array(cls, arr: np.ndarray) -> "FakeParameters":
                return cls()

            def __array__(self) -> np.ndarray:
                return np.array([])

        class FakeState:
            pass  # Missing from_array and __array__

        class FakeModule:
            PARAM_NAMES = ("x1",)
            DEFAULT_BOUNDS = ((0, 1),)
            STATE_SIZE = 1
            Parameters = FakeParameters
            State = FakeState
            SUPPORTED_RESOLUTIONS = (Resolution.daily,)

            @staticmethod
            def run() -> None:
                pass

            @staticmethod
            def step() -> None:
                pass

        with pytest.raises(ValueError, match="State class is missing"):
            registry.register("fake_model", FakeModule())  # type: ignore[arg-type]


class TestSupportedResolutionsContract:
    """Tests for SUPPORTED_RESOLUTIONS model contract."""

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige", "hbv_light"])
    def test_model_has_supported_resolutions(self, model_name: str) -> None:
        """Registered model has SUPPORTED_RESOLUTIONS attribute."""
        model = registry.get_model(model_name)
        assert hasattr(model, "SUPPORTED_RESOLUTIONS")

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige", "hbv_light"])
    def test_supported_resolutions_is_tuple(self, model_name: str) -> None:
        """SUPPORTED_RESOLUTIONS is a tuple."""
        model = registry.get_model(model_name)
        assert isinstance(model.SUPPORTED_RESOLUTIONS, tuple)

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige", "hbv_light"])
    def test_supported_resolutions_contains_resolution_enums(self, model_name: str) -> None:
        """SUPPORTED_RESOLUTIONS contains Resolution enum values."""
        model = registry.get_model(model_name)
        for res in model.SUPPORTED_RESOLUTIONS:
            assert isinstance(res, Resolution)

    @pytest.mark.parametrize("model_name", ["gr6j", "gr6j_cemaneige", "hbv_light"])
    def test_get_model_info_includes_supported_resolutions(self, model_name: str) -> None:
        """get_model_info returns supported_resolutions."""
        info = registry.get_model_info(model_name)
        assert "supported_resolutions" in info
