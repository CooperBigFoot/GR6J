"""Tests for the progress module."""

from unittest.mock import MagicMock, patch

import pytest

from pydrology.calibration.progress import ProgressTracker, progress_context


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    @patch("pydrology.calibration.progress.tqdm")
    def test_init_creates_progress_bar(self, mock_tqdm: MagicMock) -> None:
        """ProgressTracker should create a tqdm bar on init."""
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse"],
            negate_flags=[True],
        )

        mock_tqdm.assert_called_once_with(total=100, desc="Calibrating", unit="gen")
        assert tracker._bar is mock_bar

    @patch("pydrology.calibration.progress.tqdm")
    def test_ga_callback_returns_false(self, mock_tqdm: MagicMock) -> None:
        """GA callback should always return False (never stops early)."""
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse"],
            negate_flags=[True],
        )

        mock_result = MagicMock()
        mock_result.best = (MagicMock(), -0.85)

        result = tracker.ga_callback(mock_result, generation=1)

        assert result is False

    @patch("pydrology.calibration.progress.tqdm")
    def test_ga_callback_unnegates_maximized_fitness(self, mock_tqdm: MagicMock) -> None:
        """GA callback should un-negate fitness for maximization objectives.

        negate_flags=[True] means we're maximizing (ctrl-freak minimizes internally).
        If raw fitness is -0.85, displayed value should be 0.85.
        """
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse"],
            negate_flags=[True],  # Maximize: ctrl-freak negates, we un-negate
        )

        mock_result = MagicMock()
        mock_result.best = (MagicMock(), -0.85)

        tracker.ga_callback(mock_result, generation=1)

        # Should display 0.85 (un-negated from -0.85)
        mock_bar.set_postfix.assert_called_once_with(best="0.8500")
        mock_bar.update.assert_called_once_with(1)

    @patch("pydrology.calibration.progress.tqdm")
    def test_ga_callback_keeps_minimized_fitness(self, mock_tqdm: MagicMock) -> None:
        """GA callback should keep fitness as-is for minimization objectives.

        negate_flags=[False] means minimize, fitness stays as-is.
        """
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["rmse"],
            negate_flags=[False],  # Minimize: no negation
        )

        mock_result = MagicMock()
        mock_result.best = (MagicMock(), 2.5)

        tracker.ga_callback(mock_result, generation=1)

        # Should display 2.5 as-is (no un-negation needed)
        mock_bar.set_postfix.assert_called_once_with(best="2.5000")
        mock_bar.update.assert_called_once_with(1)

    @patch("pydrology.calibration.progress.tqdm")
    def test_nsga2_callback_returns_false(self, mock_tqdm: MagicMock) -> None:
        """NSGA2 callback should always return False."""
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse", "log_nse"],
            negate_flags=[True, True],
        )

        mock_result = MagicMock()
        mock_result.pareto_front = [MagicMock() for _ in range(15)]

        result = tracker.nsga2_callback(mock_result, generation=1)

        assert result is False

    @patch("pydrology.calibration.progress.tqdm")
    def test_nsga2_callback_shows_pareto_size(self, mock_tqdm: MagicMock) -> None:
        """NSGA2 callback should display Pareto front size."""
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse", "log_nse"],
            negate_flags=[True, True],
        )

        mock_result = MagicMock()
        mock_result.pareto_front = [MagicMock() for _ in range(15)]

        tracker.nsga2_callback(mock_result, generation=1)

        mock_bar.set_postfix.assert_called_once_with(pareto_size=15)
        mock_bar.update.assert_called_once_with(1)

    @patch("pydrology.calibration.progress.tqdm")
    def test_finalize_ga_closes_bar(self, mock_tqdm: MagicMock) -> None:
        """finalize_ga should close the progress bar."""
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse"],
            negate_flags=[True],
        )

        mock_result = MagicMock()
        mock_result.best = (MagicMock(), -0.9)

        tracker.finalize_ga(mock_result)

        mock_bar.set_postfix.assert_called_once_with(best="0.9000")
        mock_bar.close.assert_called_once()

    @patch("pydrology.calibration.progress.tqdm")
    def test_finalize_nsga2_closes_bar(self, mock_tqdm: MagicMock) -> None:
        """finalize_nsga2 should close the progress bar."""
        mock_bar = MagicMock()
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse", "log_nse"],
            negate_flags=[True, True],
        )

        mock_result = MagicMock()
        mock_result.pareto_front = [MagicMock() for _ in range(20)]

        tracker.finalize_nsga2(mock_result)

        mock_bar.set_postfix.assert_called_once_with(pareto_size=20)
        mock_bar.close.assert_called_once()

    @patch("pydrology.calibration.progress.tqdm")
    def test_close_is_idempotent(self, mock_tqdm: MagicMock) -> None:
        """Calling close() multiple times should not raise."""
        mock_bar = MagicMock()
        mock_bar.disable = False
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse"],
            negate_flags=[True],
        )

        # First close should work
        tracker.close()
        mock_bar.close.assert_called_once()

        # Simulate bar being disabled after close
        mock_bar.disable = True

        # Second close should not call close again
        tracker.close()
        mock_bar.close.assert_called_once()  # Still only one call

    @patch("pydrology.calibration.progress.tqdm")
    def test_close_handles_already_closed_bar(self, mock_tqdm: MagicMock) -> None:
        """close() should handle already-closed bar gracefully."""
        mock_bar = MagicMock()
        mock_bar.disable = True  # Already disabled/closed
        mock_tqdm.return_value = mock_bar

        tracker = ProgressTracker(
            total=100,
            objective_names=["nse"],
            negate_flags=[True],
        )

        # Should not raise even though bar is already "closed"
        tracker.close()

        # close() should not be called when disable is True
        mock_bar.close.assert_not_called()


class TestProgressContext:
    """Tests for progress_context context manager."""

    @patch("pydrology.calibration.progress.tqdm")
    def test_yields_progress_tracker(self, mock_tqdm: MagicMock) -> None:
        """Context manager should yield a ProgressTracker."""
        mock_bar = MagicMock()
        mock_bar.disable = False
        mock_tqdm.return_value = mock_bar

        with progress_context(100, ["nse"], [True]) as tracker:
            assert isinstance(tracker, ProgressTracker)

    @patch("pydrology.calibration.progress.tqdm")
    def test_closes_tracker_on_normal_exit(self, mock_tqdm: MagicMock) -> None:
        """Tracker should be closed when context exits normally."""
        mock_bar = MagicMock()
        mock_bar.disable = False
        mock_tqdm.return_value = mock_bar

        with progress_context(100, ["nse"], [True]):
            pass  # Normal exit

        mock_bar.close.assert_called_once()

    @patch("pydrology.calibration.progress.tqdm")
    def test_closes_tracker_on_exception(self, mock_tqdm: MagicMock) -> None:
        """Tracker should be closed even when exception is raised."""
        mock_bar = MagicMock()
        mock_bar.disable = False
        mock_tqdm.return_value = mock_bar

        with pytest.raises(ValueError, match="test error"), progress_context(100, ["nse"], [True]):
            raise ValueError("test error")

        # Bar should still be closed despite the exception
        mock_bar.close.assert_called_once()
