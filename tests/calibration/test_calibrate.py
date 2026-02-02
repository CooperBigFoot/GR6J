"""Tests for the calibrate function."""

import numpy as np
import pytest

from pydrology import Catchment, ForcingData, Parameters
from pydrology.calibration import ObservedData, Solution, calibrate
from pydrology.calibration.calibrate import (
    _validate_bounds,
    _validate_warmup,
)
from pydrology.registry import get_model


class TestValidateBounds:
    """Tests for _validate_bounds helper."""

    def test_no_bounds_and_no_default_raises(self) -> None:
        """Must provide bounds or set use_default_bounds=True."""
        model_module = get_model("gr6j")
        with pytest.raises(ValueError, match="Must provide bounds or set use_default_bounds"):
            _validate_bounds(None, use_default_bounds=False, model_module=model_module)

    def test_missing_gr6j_params_raises(self) -> None:
        """Missing GR6J parameters should raise ValueError."""
        model_module = get_model("gr6j")
        bounds = {"x1": (1, 2500), "x2": (-5, 5)}  # Missing x3-x6
        with pytest.raises(ValueError, match="Missing bounds"):
            _validate_bounds(bounds, use_default_bounds=False, model_module=model_module)

    def test_missing_snow_params_raises(self) -> None:
        """Missing snow parameters for gr6j_cemaneige should raise ValueError."""
        model_module = get_model("gr6j_cemaneige")
        bounds = {
            "x1": (1, 2500),
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
        }  # Missing ctg, kf
        with pytest.raises(ValueError, match="Missing bounds"):
            _validate_bounds(bounds, use_default_bounds=False, model_module=model_module)

    def test_lower_ge_upper_raises(self) -> None:
        """Lower bound >= upper bound should raise ValueError."""
        model_module = get_model("gr6j")
        bounds = {
            "x1": (2500, 1),  # Invalid: lower > upper
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
        }
        with pytest.raises(ValueError, match="Lower bound must be less than upper"):
            _validate_bounds(bounds, use_default_bounds=False, model_module=model_module)

    def test_valid_bounds_gr6j(self) -> None:
        """Valid GR6J bounds should return the bounds."""
        model_module = get_model("gr6j")
        bounds = {
            "x1": (1, 2500),
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
        }
        result = _validate_bounds(bounds, use_default_bounds=False, model_module=model_module)
        assert result == bounds

    def test_valid_bounds_gr6j_cemaneige(self) -> None:
        """Valid GR6J-CemaNeige bounds should return the bounds."""
        model_module = get_model("gr6j_cemaneige")
        bounds = {
            "x1": (1, 2500),
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
            "ctg": (0, 1),
            "kf": (0, 10),
        }
        result = _validate_bounds(bounds, use_default_bounds=False, model_module=model_module)
        assert result == bounds

    def test_use_default_bounds_gr6j(self) -> None:
        """use_default_bounds=True should return model's DEFAULT_BOUNDS."""
        model_module = get_model("gr6j")
        result = _validate_bounds(None, use_default_bounds=True, model_module=model_module)
        assert result == model_module.DEFAULT_BOUNDS

    def test_use_default_bounds_gr6j_cemaneige(self) -> None:
        """use_default_bounds=True should return model's DEFAULT_BOUNDS."""
        model_module = get_model("gr6j_cemaneige")
        result = _validate_bounds(None, use_default_bounds=True, model_module=model_module)
        assert result == model_module.DEFAULT_BOUNDS


class TestValidateWarmup:
    """Tests for _validate_warmup helper."""

    def test_negative_warmup_raises(self) -> None:
        """Negative warmup should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            _validate_warmup(warmup=-1, forcing_length=100, observed_length=100)

    def test_length_mismatch_raises(self) -> None:
        """Observed length != forcing - warmup should raise ValueError."""
        with pytest.raises(ValueError, match="must equal"):
            _validate_warmup(warmup=10, forcing_length=100, observed_length=80)  # Should be 90

    def test_valid_warmup(self) -> None:
        """Valid warmup should pass."""
        _validate_warmup(warmup=10, forcing_length=100, observed_length=90)


class TestCalibrate:
    """Integration tests for calibrate function."""

    @pytest.fixture
    def simple_forcing(self) -> ForcingData:
        """Create simple forcing data for testing."""
        n_days = 30  # Short for fast tests
        return ForcingData(
            time=np.datetime64("2020-01-01") + np.arange(n_days),
            precip=np.random.default_rng(42).exponential(5.0, n_days),
            pet=np.full(n_days, 3.5),
        )

    @pytest.fixture
    def simple_observed(self, simple_forcing: ForcingData) -> ObservedData:
        """Create observed data matching forcing minus warmup."""
        warmup = 10
        n_obs = len(simple_forcing) - warmup
        return ObservedData(
            time=simple_forcing.time[warmup:],
            streamflow=np.random.default_rng(42).uniform(1.0, 5.0, n_obs),
        )

    @pytest.fixture
    def simple_bounds(self) -> dict[str, tuple[float, float]]:
        """Simple parameter bounds for GR6J."""
        return {
            "x1": (100, 500),
            "x2": (-2, 2),
            "x3": (50, 200),
            "x4": (1, 5),
            "x5": (-2, 2),
            "x6": (1, 10),
        }

    def test_single_objective_returns_solution(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Single objective should return a Solution."""
        result = calibrate(
            model="gr6j",
            forcing=simple_forcing,
            observed=simple_observed,
            objectives=["nse"],
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
        )
        assert isinstance(result, Solution)
        assert result.model == "gr6j"
        assert "nse" in result.score
        assert isinstance(result.parameters, Parameters)

    def test_multi_objective_returns_list(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Multiple objectives should return a list of Solutions."""
        result = calibrate(
            model="gr6j",
            forcing=simple_forcing,
            observed=simple_observed,
            objectives=["nse", "log_nse"],
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, Solution) for s in result)
        assert all(s.model == "gr6j" for s in result)
        assert all("nse" in s.score and "log_nse" in s.score for s in result)

    def test_deterministic_with_seed(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Same seed should produce same results."""
        kwargs = {
            "model": "gr6j",
            "forcing": simple_forcing,
            "observed": simple_observed,
            "objectives": {"nse": "maximize"},
            "bounds": simple_bounds,
            "warmup": 10,
            "population_size": 10,
            "generations": 3,
            "seed": 12345,
        }
        result1 = calibrate(**kwargs)
        result2 = calibrate(**kwargs)
        assert result1.parameters.x1 == result2.parameters.x1
        assert result1.score["nse"] == result2.score["nse"]

    def test_use_default_bounds(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
    ) -> None:
        """Calibration with use_default_bounds=True should work."""
        result = calibrate(
            model="gr6j",
            forcing=simple_forcing,
            observed=simple_observed,
            objectives=["nse"],
            use_default_bounds=True,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
        )
        assert isinstance(result, Solution)
        assert result.model == "gr6j"

    def test_unknown_model_raises_keyerror(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
    ) -> None:
        """Unknown model should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            calibrate(
                model="unknown_model",
                forcing=simple_forcing,
                observed=simple_observed,
                objectives=["nse"],
                use_default_bounds=True,
                warmup=10,
                population_size=10,
                generations=3,
                seed=42,
            )

    def test_gr6j_cemaneige_requires_catchment(self) -> None:
        """GR6J-CemaNeige should require catchment parameter."""
        n_days = 30
        forcing = ForcingData(
            time=np.datetime64("2020-01-01") + np.arange(n_days),
            precip=np.random.default_rng(42).exponential(5.0, n_days),
            pet=np.full(n_days, 3.5),
            temp=np.random.default_rng(42).uniform(-5.0, 10.0, n_days),
        )
        warmup = 10
        observed = ObservedData(
            time=forcing.time[warmup:],
            streamflow=np.random.default_rng(42).uniform(1.0, 5.0, n_days - warmup),
        )

        with pytest.raises(ValueError, match="requires catchment"):
            calibrate(
                model="gr6j_cemaneige",
                forcing=forcing,
                observed=observed,
                objectives=["nse"],
                use_default_bounds=True,
                warmup=warmup,
                population_size=10,
                generations=3,
                seed=42,
            )

    def test_gr6j_cemaneige_calibration(self) -> None:
        """GR6J-CemaNeige calibration should work with catchment."""
        n_days = 30
        forcing = ForcingData(
            time=np.datetime64("2020-01-01") + np.arange(n_days),
            precip=np.random.default_rng(42).exponential(5.0, n_days),
            pet=np.full(n_days, 3.5),
            temp=np.random.default_rng(42).uniform(-5.0, 10.0, n_days),
        )
        catchment = Catchment(mean_annual_solid_precip=150.0)
        warmup = 10
        observed = ObservedData(
            time=forcing.time[warmup:],
            streamflow=np.random.default_rng(42).uniform(1.0, 5.0, n_days - warmup),
        )

        result = calibrate(
            model="gr6j_cemaneige",
            forcing=forcing,
            observed=observed,
            objectives=["nse"],
            use_default_bounds=True,
            catchment=catchment,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=42,
        )

        assert isinstance(result, Solution)
        assert result.model == "gr6j_cemaneige"
        # Check that we have 8 parameters (6 GR6J + 2 CemaNeige)
        assert hasattr(result.parameters, "ctg")
        assert hasattr(result.parameters, "kf")

    def test_progress_true_runs_without_error(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Calibration with progress=True should run without error."""
        result = calibrate(
            model="gr6j",
            forcing=simple_forcing,
            observed=simple_observed,
            objectives=["nse"],
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
            progress=True,
        )
        assert isinstance(result, Solution)

    def test_progress_false_runs_without_error(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Calibration with progress=False should run without error."""
        result = calibrate(
            model="gr6j",
            forcing=simple_forcing,
            observed=simple_observed,
            objectives=["nse"],
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
            progress=False,
        )
        assert isinstance(result, Solution)

    def test_progress_does_not_affect_results(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Results should be identical with progress=True vs progress=False."""
        kwargs = {
            "model": "gr6j",
            "forcing": simple_forcing,
            "observed": simple_observed,
            "objectives": ["nse"],
            "bounds": simple_bounds,
            "warmup": 10,
            "population_size": 10,
            "generations": 3,
            "seed": 42,
        }
        result_with_progress = calibrate(**kwargs, progress=True)
        result_without_progress = calibrate(**kwargs, progress=False)

        # Results should be deterministic with same seed
        assert result_with_progress.parameters.x1 == result_without_progress.parameters.x1
        assert result_with_progress.score["nse"] == result_without_progress.score["nse"]

    def test_multi_objective_progress_works(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Multi-objective calibration should work with progress bar."""
        result = calibrate(
            model="gr6j",
            forcing=simple_forcing,
            observed=simple_observed,
            objectives=["nse", "log_nse"],
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
            progress=True,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, Solution) for s in result)
