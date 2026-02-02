"""Progress bar tracking for GR6J calibration using tqdm."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from ctrl_freak import GAResult, NSGA2Result


class ProgressTracker:
    """Track calibration progress with a tqdm progress bar.

    Provides callbacks for GA and NSGA2 optimization algorithms to update
    the progress bar with current best fitness or Pareto front size.

    Parameters
    ----------
    total
        Total number of generations to track.
    objective_names
        Names of the objective functions being optimized.
    negate_flags
        Flags indicating which objectives were negated for minimization.
        If True, the displayed value will be un-negated (multiplied by -1).

    """

    def __init__(
        self,
        total: int,
        objective_names: list[str],
        negate_flags: list[bool],
    ) -> None:
        self._total = total
        self._objective_names = objective_names
        self._negate_flags = negate_flags
        self._bar = tqdm(total=total, desc="Calibrating", unit="gen")

    def ga_callback(self, result: GAResult, generation: int) -> bool:
        """Callback for single-objective GA optimization.

        Parameters
        ----------
        result
            The current GA result containing best individual and fitness.
        generation
            The current generation number.

        Returns
        -------
        bool
            Always False (never stops early).

        """
        best_fitness = result.best[1]
        # Un-negate if the objective was negated for minimization
        if self._negate_flags[0]:
            best_fitness = -best_fitness
        formatted_value = f"{best_fitness:.4f}"
        self._bar.set_postfix(best=formatted_value)
        self._bar.update(1)
        return False

    def nsga2_callback(self, result: NSGA2Result, generation: int) -> bool:
        """Callback for multi-objective NSGA2 optimization.

        Parameters
        ----------
        result
            The current NSGA2 result containing the Pareto front.
        generation
            The current generation number.

        Returns
        -------
        bool
            Always False (never stops early).

        """
        pareto_size = len(result.pareto_front)
        self._bar.set_postfix(pareto_size=pareto_size)
        self._bar.update(1)
        return False

    def finalize_ga(self, result: GAResult) -> None:
        """Finalize the progress bar after GA optimization completes.

        Parameters
        ----------
        result
            The final GA result.

        """
        best_fitness = result.best[1]
        if self._negate_flags[0]:
            best_fitness = -best_fitness
        formatted_value = f"{best_fitness:.4f}"
        self._bar.set_postfix(best=formatted_value)
        self._bar.close()

    def finalize_nsga2(self, result: NSGA2Result) -> None:
        """Finalize the progress bar after NSGA2 optimization completes.

        Parameters
        ----------
        result
            The final NSGA2 result.

        """
        pareto_size = len(result.pareto_front)
        self._bar.set_postfix(pareto_size=pareto_size)
        self._bar.close()

    def close(self) -> None:
        """Close the progress bar if not already closed."""
        if not self._bar.disable:
            self._bar.close()


@contextmanager
def progress_context(
    total: int,
    objective_names: list[str],
    negate_flags: list[bool],
) -> Generator[ProgressTracker, None, None]:
    """Context manager for progress tracking during calibration.

    Ensures the progress bar is properly closed even if an exception occurs.

    Parameters
    ----------
    total
        Total number of generations to track.
    objective_names
        Names of the objective functions being optimized.
    negate_flags
        Flags indicating which objectives were negated for minimization.

    Yields
    ------
    ProgressTracker
        The progress tracker instance.

    Examples
    --------
    >>> with progress_context(100, ["NSE"], [True]) as tracker:
    ...     # Run optimization with tracker.ga_callback
    ...     pass

    """
    tracker = ProgressTracker(total, objective_names, negate_flags)
    try:
        yield tracker
    finally:
        tracker.close()
