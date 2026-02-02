"""Tests for hydrological metric functions."""

import numpy as np
import pytest

from pydrology.calibration.metrics import kge, log_nse, mae, nse, pbias, rmse


class TestNSE:
    """Tests for Nash-Sutcliffe Efficiency."""

    def test_perfect_match(self) -> None:
        """Perfect simulation should give NSE = 1."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nse(obs, sim) == pytest.approx(1.0)

    def test_mean_simulation(self) -> None:
        """Simulating the mean should give NSE = 0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.full(5, np.mean(obs))
        assert nse(obs, sim) == pytest.approx(0.0)

    def test_zero_variance_returns_neg_inf(self) -> None:
        """Zero variance in observed should return -inf."""
        obs = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nse(obs, sim) == -np.inf

    def test_poor_simulation_negative(self) -> None:
        """Poor simulation should give negative NSE."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed
        assert nse(obs, sim) < 0


class TestLogNSE:
    """Tests for log-transformed NSE."""

    def test_perfect_match(self) -> None:
        """Perfect simulation should give log-NSE close to 1."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert log_nse(obs, sim) == pytest.approx(1.0, rel=1e-5)

    def test_zero_variance_returns_neg_inf(self) -> None:
        """Zero variance in log-observed should return -inf."""
        obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert log_nse(obs, sim) == -np.inf

    def test_handles_zeros(self) -> None:
        """Should handle zeros without error (small constant added)."""
        obs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        sim = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = log_nse(obs, sim)
        assert np.isfinite(result)


class TestKGE:
    """Tests for Kling-Gupta Efficiency."""

    def test_perfect_match(self) -> None:
        """Perfect simulation should give KGE = 1."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert kge(obs, sim) == pytest.approx(1.0)

    def test_bias_affects_kge(self) -> None:
        """Biased simulation should reduce KGE."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # Positive bias
        result = kge(obs, sim)
        assert result < 1.0

    def test_variability_affects_kge(self) -> None:
        """Different variability should reduce KGE."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([2.0, 2.5, 3.0, 3.5, 4.0])  # Same mean, less variability
        result = kge(obs, sim)
        assert result < 1.0

    def test_zero_variance_obs_handled(self) -> None:
        """Zero variance observed should not raise."""
        obs = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kge(obs, sim)
        assert np.isfinite(result)


class TestPBIAS:
    """Tests for Percent Bias."""

    def test_perfect_match(self) -> None:
        """Perfect simulation should give PBIAS = 0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert pbias(obs, sim) == pytest.approx(0.0)

    def test_overestimation_positive(self) -> None:
        """Overestimation should give positive PBIAS."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All +1
        result = pbias(obs, sim)
        assert result > 0

    def test_underestimation_negative(self) -> None:
        """Underestimation should give negative PBIAS."""
        obs = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # All -1
        result = pbias(obs, sim)
        assert result < 0

    def test_zero_observed_returns_inf(self) -> None:
        """Zero sum observed should return inf."""
        obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert pbias(obs, sim) == np.inf


class TestRMSE:
    """Tests for Root Mean Square Error."""

    def test_perfect_match(self) -> None:
        """Perfect simulation should give RMSE = 0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(obs, sim) == pytest.approx(0.0)

    def test_constant_error(self) -> None:
        """Constant error should give known RMSE."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All +1
        assert rmse(obs, sim) == pytest.approx(1.0)

    def test_always_nonnegative(self) -> None:
        """RMSE should always be non-negative."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([5.0, 1.0, 2.0])
        assert rmse(obs, sim) >= 0


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_match(self) -> None:
        """Perfect simulation should give MAE = 0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mae(obs, sim) == pytest.approx(0.0)

    def test_constant_error(self) -> None:
        """Constant error should give known MAE."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All +1
        assert mae(obs, sim) == pytest.approx(1.0)

    def test_symmetric_error(self) -> None:
        """MAE should treat positive and negative errors equally."""
        obs = np.array([2.0, 2.0])
        sim = np.array([1.0, 3.0])  # -1 and +1
        assert mae(obs, sim) == pytest.approx(1.0)

    def test_always_nonnegative(self) -> None:
        """MAE should always be non-negative."""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([5.0, 1.0, 2.0])
        assert mae(obs, sim) >= 0
