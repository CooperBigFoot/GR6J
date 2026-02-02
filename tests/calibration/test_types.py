"""Tests for calibration data types."""

import numpy as np
import pytest
from pydantic import ValidationError

from pydrology import Parameters
from pydrology.calibration.types import ObservedData, Solution


class TestObservedData:
    """Tests for ObservedData validation."""

    def test_valid_creation(self) -> None:
        """Valid data should create ObservedData successfully."""
        time = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        streamflow = np.array([1.0, 2.0, 3.0])
        obs = ObservedData(time=time, streamflow=streamflow)
        assert len(obs) == 3

    def test_2d_time_rejected(self) -> None:
        """2D time array should raise ValidationError."""
        time = np.array([["2020-01-01"], ["2020-01-02"]], dtype="datetime64")
        streamflow = np.array([1.0, 2.0])
        with pytest.raises(ValidationError, match="must be 1D"):
            ObservedData(time=time, streamflow=streamflow)

    def test_2d_streamflow_rejected(self) -> None:
        """2D streamflow array should raise ValidationError."""
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        streamflow = np.array([[1.0], [2.0]])
        with pytest.raises(ValidationError, match="must be 1D"):
            ObservedData(time=time, streamflow=streamflow)

    def test_nan_streamflow_rejected(self) -> None:
        """NaN values in streamflow should raise ValidationError."""
        time = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        streamflow = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValidationError, match="NaN"):
            ObservedData(time=time, streamflow=streamflow)

    def test_length_mismatch_rejected(self) -> None:
        """Mismatched array lengths should raise ValidationError."""
        time = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        streamflow = np.array([1.0, 2.0])  # Only 2 elements
        with pytest.raises(ValidationError, match="does not match"):
            ObservedData(time=time, streamflow=streamflow)

    def test_dtype_coercion(self) -> None:
        """Arrays should be coerced to correct dtypes."""
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        streamflow = np.array([1, 2])  # int, should coerce to float64
        obs = ObservedData(time=time, streamflow=streamflow)
        assert obs.time.dtype == np.dtype("datetime64[ns]")
        assert obs.streamflow.dtype == np.float64

    def test_frozen(self) -> None:
        """ObservedData should be immutable."""
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        streamflow = np.array([1.0, 2.0])
        obs = ObservedData(time=time, streamflow=streamflow)
        with pytest.raises(ValidationError):
            obs.streamflow = np.array([3.0, 4.0])

    def test_len(self) -> None:
        """__len__ should return number of observations."""
        time = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64")
        streamflow = np.array([1.0, 2.0, 3.0, 4.0])
        obs = ObservedData(time=time, streamflow=streamflow)
        assert len(obs) == 4


class TestSolution:
    """Tests for Solution dataclass."""

    def test_creation(self) -> None:
        """Solution should be creatable with model, parameters, and score."""
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0)
        score = {"nse": 0.85, "kge": 0.78}
        sol = Solution(model="gr6j", parameters=params, score=score)
        assert sol.model == "gr6j"
        assert sol.parameters.x1 == 350.0
        assert sol.score["nse"] == 0.85

    def test_frozen(self) -> None:
        """Solution should be immutable (frozen dataclass)."""
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0)
        score = {"nse": 0.85}
        sol = Solution(model="gr6j", parameters=params, score=score)
        with pytest.raises(AttributeError, match="cannot assign"):
            sol.score = {"nse": 0.90}
