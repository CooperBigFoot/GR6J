"""Tests for the metrics registry."""

import logging

import pytest

from pydrology.calibration.metrics import (
    METRICS,
    get_metric,
    list_metrics,
    register,
    validate_objectives,
)


class TestRegistry:
    """Tests for metric registration and retrieval."""

    def test_metrics_populated(self) -> None:
        """Registry should contain registered metrics."""
        # Functions module registers 6 metrics
        assert len(METRICS) >= 6
        assert "nse" in METRICS
        assert "kge" in METRICS

    def test_get_metric_returns_function_and_direction(self) -> None:
        """get_metric returns (function, direction) tuple."""
        func, direction = get_metric("nse")
        assert callable(func)
        assert direction in ("maximize", "minimize")

    def test_get_metric_unknown_raises_keyerror(self) -> None:
        """get_metric raises KeyError for unknown metrics."""
        with pytest.raises(KeyError, match="Unknown metric 'nonexistent'"):
            get_metric("nonexistent")

    def test_list_metrics_returns_sorted_names(self) -> None:
        """list_metrics returns sorted list of metric names."""
        names = list_metrics()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "nse" in names
        assert "kge" in names

    def test_register_invalid_direction_raises(self) -> None:
        """register raises ValueError for invalid direction."""
        with pytest.raises(ValueError, match="direction must be"):
            register("invalid")


class TestValidateObjectives:
    """Tests for validate_objectives function."""

    def test_list_form_returns_dict_with_registered_directions(self) -> None:
        """List form should return dict with registered directions."""
        result = validate_objectives(["nse", "kge"])
        assert result == {"nse": "maximize", "kge": "maximize"}

    def test_list_form_single_metric(self) -> None:
        """List form works with a single metric."""
        result = validate_objectives(["rmse"])
        assert result == {"rmse": "minimize"}

    def test_list_form_unknown_raises(self) -> None:
        """Unknown metric in list should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            validate_objectives(["nse", "unknown"])

    def test_list_form_empty_raises(self) -> None:
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_objectives([])

    def test_dict_form_returns_same_dict(self) -> None:
        """Dict form should return the same dict for backward compatibility."""
        result = validate_objectives({"nse": "maximize"})
        assert result == {"nse": "maximize"}

    def test_dict_form_multiple_metrics(self) -> None:
        """Dict form works with multiple metrics."""
        result = validate_objectives({"nse": "maximize", "rmse": "minimize"})
        assert result == {"nse": "maximize", "rmse": "minimize"}

    def test_empty_dict_raises(self) -> None:
        """Empty objectives dict should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_objectives({})

    def test_dict_form_unknown_metric_raises(self) -> None:
        """Unknown metric in dict should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            validate_objectives({"unknown_metric": "maximize"})

    def test_dict_form_wrong_direction_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Wrong direction should log a warning but still return dict."""
        with caplog.at_level(logging.WARNING):
            result = validate_objectives({"nse": "minimize"})  # NSE is registered as maximize
        assert "unusual" in caplog.text.lower() or "direction" in caplog.text.lower()
        assert result == {"nse": "minimize"}
