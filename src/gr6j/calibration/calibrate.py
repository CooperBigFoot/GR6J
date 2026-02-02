"""Core calibration function wiring GR6J to ctrl-freak.

Provides automatic parameter optimization via evolutionary algorithms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..cemaneige import CemaNeige
from ..inputs import Catchment, ForcingData
from ..model import Parameters, State, run
from .metrics import get_metric, validate_objectives
from .types import ObservedData, Solution

if TYPE_CHECKING:
    from numpy.random import Generator


# Parameter names in order for vector construction
_GR6J_PARAMS = ("x1", "x2", "x3", "x4", "x5", "x6")
_SNOW_PARAMS = ("ctg", "kf")


def _validate_bounds(bounds: dict[str, tuple[float, float]], snow: bool) -> None:
    """Validate bounds dictionary.

    Args:
        bounds: Dict mapping parameter names to (lower, upper) tuples.
        snow: Whether snow parameters should be included.

    Raises:
        ValueError: If bounds are invalid.
    """
    if not bounds:
        msg = "bounds cannot be empty"
        raise ValueError(msg)

    required: set[str] = set(_GR6J_PARAMS)
    if snow:
        required |= set(_SNOW_PARAMS)

    provided: set[str] = set(bounds.keys())
    missing = required - provided
    if missing:
        msg = f"Missing bounds for parameters: {sorted(missing)}"
        raise ValueError(msg)

    for name, (lower, upper) in bounds.items():
        if lower >= upper:
            msg = f"Lower bound must be less than upper bound for '{name}': {lower} >= {upper}"
            raise ValueError(msg)


def _validate_snow_config(snow: bool, forcing: ForcingData, catchment: Catchment | None) -> None:
    """Validate snow configuration.

    Args:
        snow: Whether snow mode is enabled.
        forcing: Forcing data.
        catchment: Catchment properties.

    Raises:
        ValueError: If snow is enabled but temp or catchment is missing.
    """
    if not snow:
        return

    if forcing.temp is None:
        msg = "forcing.temp required when snow=True"
        raise ValueError(msg)

    if catchment is None:
        msg = "catchment required when snow=True"
        raise ValueError(msg)


def _validate_warmup(warmup: int, forcing_length: int, observed_length: int) -> None:
    """Validate warmup period.

    Args:
        warmup: Number of warmup days.
        forcing_length: Length of forcing data.
        observed_length: Length of observed data.

    Raises:
        ValueError: If warmup is invalid.
    """
    if warmup < 0:
        msg = f"warmup must be non-negative, got {warmup}"
        raise ValueError(msg)

    expected_obs_length = forcing_length - warmup
    if observed_length != expected_obs_length:
        msg = (
            f"observed length ({observed_length}) must equal "
            f"forcing length ({forcing_length}) minus warmup ({warmup}) = {expected_obs_length}"
        )
        raise ValueError(msg)


def _array_to_parameters(x: np.ndarray, snow: bool) -> Parameters:
    """Convert parameter vector to Parameters object.

    Args:
        x: Parameter vector (6 elements without snow, 8 with snow).
        snow: Whether snow parameters are included.

    Returns:
        Parameters object with optional CemaNeige.
    """
    snow_obj = None
    if snow:
        snow_obj = CemaNeige(ctg=float(x[6]), kf=float(x[7]))

    return Parameters(
        x1=float(x[0]),
        x2=float(x[1]),
        x3=float(x[2]),
        x4=float(x[3]),
        x5=float(x[4]),
        x6=float(x[5]),
        snow=snow_obj,
    )


def calibrate(
    forcing: ForcingData,
    observed: ObservedData,
    objectives: list[str] | dict[str, str],
    bounds: dict[str, tuple[float, float]],
    catchment: Catchment | None = None,
    snow: bool = False,
    initial_state: State | None = None,
    warmup: int = 365,
    population_size: int = 50,
    generations: int = 100,
    seed: int | None = None,
    progress: bool = True,
    callback: Callable | None = None,
    n_workers: int = 1,
) -> Solution | list[Solution]:
    """Calibrate GR6J parameters using evolutionary optimization.

    Supports single-objective (GA) and multi-objective (NSGA-II) optimization.
    Single objective returns a single Solution; multiple objectives return
    a list of Pareto-optimal Solutions.

    Args:
        forcing: Forcing data (precip, pet, optionally temp).
        observed: Observed streamflow for the post-warmup period.
        objectives: Metric names to optimize. Can be:
            - List of names (uses registered directions): ["nse", "log_nse"]
            - Dict with explicit directions (for overrides): {"nse": "minimize"}
        bounds: Dict mapping parameter names to (lower, upper) tuples.
            Must include x1-x6, plus ctg/kf if snow=True.
        catchment: Catchment properties. Required when snow=True.
        snow: Whether to calibrate with CemaNeige snow module.
        initial_state: Initial GR6J state. If None, uses State.initialize().
        warmup: Number of days to discard for spin-up. Default 365.
        population_size: EA population size. Default 50.
        generations: Number of EA generations. Default 100.
        seed: Random seed for reproducibility.
        progress: Whether to display a progress bar. Default True.
        callback: Optional callback for monitoring. Only used when progress=False.
            For GA, receives (GAResult, gen); for NSGA-II, receives (NSGA2Result, gen).
            Return True to stop early.
        n_workers: Number of parallel workers for evaluation. Use 1 for sequential
            execution (default), -1 for all CPU cores, or any positive integer.

    Returns:
        Single-objective: Solution with best parameters and score.
        Multi-objective: List of Pareto-optimal Solutions.

    Raises:
        ValueError: If inputs are invalid (empty objectives, missing bounds, etc.).

    Example:
        >>> from gr6j import ForcingData, calibrate, ObservedData
        >>> # Single-objective calibration
        >>> result = calibrate(
        ...     forcing=forcing,
        ...     observed=observed,
        ...     objectives=["nse"],
        ...     bounds={"x1": (1, 2500), "x2": (-5, 5), ...},
        ...     warmup=365,
        ... )
        >>> print(result.parameters.x1, result.score["nse"])
    """
    # Lazy import ctrl-freak to keep it optional at import time
    from ctrl_freak import ga, nsga2, polynomial_mutation, sbx_crossover

    from .progress import progress_context

    # Validate and normalize inputs
    objectives_dict = validate_objectives(objectives)
    _validate_bounds(bounds, snow)
    _validate_snow_config(snow, forcing, catchment)
    _validate_warmup(warmup, len(forcing), len(observed))

    # Build parameter order and bounds arrays
    param_names: list[str] = list(_GR6J_PARAMS)
    if snow:
        param_names.extend(_SNOW_PARAMS)

    lower_bounds = np.array([bounds[name][0] for name in param_names])
    upper_bounds = np.array([bounds[name][1] for name in param_names])
    bounds_tuple = (lower_bounds, upper_bounds)

    # Get metric functions and determine if we need to negate for minimization
    # ctrl-freak always minimizes, so we negate maximization objectives
    objective_names = list(objectives_dict.keys())
    metric_funcs = []
    negate_flags = []
    for name in objective_names:
        func, _ = get_metric(name)
        metric_funcs.append(func)
        # Negate if we want to maximize (since ctrl-freak minimizes)
        negate_flags.append(objectives_dict[name] == "maximize")

    # Create operators with seed for reproducibility
    crossover = sbx_crossover(eta=15.0, bounds=bounds_tuple, seed=seed)
    mutate = polynomial_mutation(eta=20.0, bounds=bounds_tuple, seed=seed)

    # Create initialization function
    def init(rng: Generator) -> np.ndarray:
        return rng.uniform(lower_bounds, upper_bounds)

    # Create single-objective evaluation function for GA
    def evaluate_single(x: np.ndarray) -> float:
        params = _array_to_parameters(x, snow)
        output = run(params, forcing, catchment=catchment, initial_state=initial_state)
        sim = output.gr6j.streamflow[warmup:]
        obs = observed.streamflow
        score = metric_funcs[0](obs, sim)
        return -score if negate_flags[0] else score

    # Create multi-objective evaluation function for NSGA2
    def evaluate_multi(x: np.ndarray) -> np.ndarray:
        params = _array_to_parameters(x, snow)
        output = run(params, forcing, catchment=catchment, initial_state=initial_state)
        sim = output.gr6j.streamflow[warmup:]
        obs = observed.streamflow
        scores = []
        for func, negate in zip(metric_funcs, negate_flags, strict=True):
            score = func(obs, sim)
            scores.append(-score if negate else score)
        return np.array(scores)

    # Run optimization
    if len(objective_names) == 1:
        # Single-objective GA
        if progress:
            with progress_context(generations, objective_names, negate_flags) as tracker:
                result = ga(
                    init=init,
                    evaluate=evaluate_single,
                    crossover=crossover,
                    mutate=mutate,
                    pop_size=population_size,
                    n_generations=generations,
                    seed=seed,
                    callback=tracker.ga_callback,
                    n_workers=n_workers,
                )
                tracker.finalize_ga(result)
        else:
            result = ga(
                init=init,
                evaluate=evaluate_single,
                crossover=crossover,
                mutate=mutate,
                pop_size=population_size,
                n_generations=generations,
                seed=seed,
                callback=callback,
                n_workers=n_workers,
            )
        best_x, best_fitness = result.best
        best_params = _array_to_parameters(best_x, snow)

        # Compute actual (non-negated) scores for output
        output = run(best_params, forcing, catchment=catchment, initial_state=initial_state)
        sim = output.gr6j.streamflow[warmup:]
        obs = observed.streamflow
        score_dict = {name: float(func(obs, sim)) for name, func in zip(objective_names, metric_funcs, strict=True)}

        return Solution(parameters=best_params, score=score_dict)

    else:
        # Multi-objective NSGA-II
        if progress:
            with progress_context(generations, objective_names, negate_flags) as tracker:
                result = nsga2(
                    init=init,
                    evaluate=evaluate_multi,
                    crossover=crossover,
                    mutate=mutate,
                    pop_size=population_size,
                    n_generations=generations,
                    seed=seed,
                    callback=tracker.nsga2_callback,
                    n_workers=n_workers,
                )
                tracker.finalize_nsga2(result)
        else:
            result = nsga2(
                init=init,
                evaluate=evaluate_multi,
                crossover=crossover,
                mutate=mutate,
                pop_size=population_size,
                n_generations=generations,
                seed=seed,
                callback=callback,
                n_workers=n_workers,
            )

        # Convert Pareto front to Solutions
        solutions = []
        for individual in result.pareto_front:
            params = _array_to_parameters(individual.x, snow)

            # Compute actual (non-negated) scores for output
            output = run(params, forcing, catchment=catchment, initial_state=initial_state)
            sim = output.gr6j.streamflow[warmup:]
            obs = observed.streamflow
            score_dict = {name: float(func(obs, sim)) for name, func in zip(objective_names, metric_funcs, strict=True)}

            solutions.append(Solution(parameters=params, score=score_dict))

        return solutions
