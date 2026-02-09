"""Core calibration function wiring hydrological models to ctrl-freak.

Provides automatic parameter optimization via evolutionary algorithms.
Supports any model registered in the pydrology model registry.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from pydrology.registry import get_model
from pydrology.types import Catchment, ForcingData

from .metrics import get_metric, validate_objectives
from .types import ObservedData, Solution

if TYPE_CHECKING:
    from numpy.random import Generator


def _validate_bounds(
    bounds: dict[str, tuple[float, float]] | None,
    use_default_bounds: bool,
    model_module: ModuleType,
) -> dict[str, tuple[float, float]]:
    """Validate and resolve bounds dictionary.

    Args:
        bounds: Dict mapping parameter names to (lower, upper) tuples, or None.
        use_default_bounds: Whether to use model's default bounds if bounds is None.
        model_module: The model module containing PARAM_NAMES and DEFAULT_BOUNDS.

    Returns:
        Resolved bounds dictionary.

    Raises:
        ValueError: If bounds are invalid or not provided when required.
    """
    param_names = model_module.PARAM_NAMES

    if bounds is not None:
        # Validate provided bounds
        missing = set(param_names) - set(bounds.keys())
        if missing:
            msg = f"Missing bounds for parameters: {sorted(missing)}"
            raise ValueError(msg)

        # Validate lower < upper for each bound
        for name, (lower, upper) in bounds.items():
            if lower >= upper:
                msg = f"Lower bound must be less than upper bound for '{name}': {lower} >= {upper}"
                raise ValueError(msg)

        return bounds

    elif use_default_bounds:
        return model_module.DEFAULT_BOUNDS

    else:
        msg = "Must provide bounds or set use_default_bounds=True"
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


def _model_requires_catchment(model: str) -> bool:
    """Check if a model requires catchment parameter for run().

    Args:
        model: Model identifier string.

    Returns:
        True if the model's run() function requires a catchment parameter.
    """
    # Models with snow modules require catchment for snow layer configuration
    return model == "gr6j_cemaneige"


def calibrate(
    model: str,
    forcing: ForcingData,
    observed: ObservedData,
    objectives: list[str] | dict[str, str],
    bounds: dict[str, tuple[float, float]] | None = None,
    use_default_bounds: bool = False,
    catchment: Catchment | None = None,
    initial_state: Any | None = None,
    warmup: int = 365,
    population_size: int = 50,
    generations: int = 100,
    seed: int | None = None,
    progress: bool = True,
    callback: Callable | None = None,
    n_workers: int = 1,
) -> Solution | list[Solution]:
    """Calibrate model parameters using evolutionary optimization.

    Supports single-objective (GA) and multi-objective (NSGA-II) optimization.
    Single objective returns a single Solution; multiple objectives return
    a list of Pareto-optimal Solutions.

    Args:
        model: Model identifier (e.g., "gr6j", "gr6j_cemaneige").
        forcing: Forcing data (precip, pet, optionally temp).
        observed: Observed streamflow for the post-warmup period.
        objectives: Metric names to optimize. Can be:
            - List of names (uses registered directions): ["nse", "log_nse"]
            - Dict with explicit directions (for overrides): {"nse": "minimize"}
        bounds: Dict mapping parameter names to (lower, upper) tuples.
            Must include all parameters for the selected model.
            If None, use_default_bounds must be True.
        use_default_bounds: Whether to use model's default bounds if bounds is None.
        catchment: Catchment properties. Required for models with snow module
            (e.g., "gr6j_cemaneige").
        initial_state: Initial model state. Type depends on the model.
            If None, uses the model's State.initialize().
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
        KeyError: If the model name is not registered.
        ValueError: If inputs are invalid (empty objectives, missing bounds, etc.).

    Example:
        >>> from pydrology import ForcingData, calibrate, ObservedData
        >>> # Single-objective calibration with GR6J
        >>> result = calibrate(
        ...     model="gr6j",
        ...     forcing=forcing,
        ...     observed=observed,
        ...     objectives=["nse"],
        ...     use_default_bounds=True,
        ...     warmup=365,
        ... )
        >>> print(result.parameters.x1, result.score["nse"])
        >>>
        >>> # Calibration with GR6J-CemaNeige (snow model)
        >>> result = calibrate(
        ...     model="gr6j_cemaneige",
        ...     forcing=forcing,  # Must include temp
        ...     observed=observed,
        ...     objectives=["nse"],
        ...     use_default_bounds=True,
        ...     catchment=catchment,  # Required for snow models
        ...     warmup=365,
        ... )
    """
    # Lazy import ctrl-freak to keep it optional at import time
    from ctrl_freak import ga, nsga2, polynomial_mutation, sbx_crossover

    from .progress import progress_context

    # Get model module from registry
    model_module = get_model(model)
    run_fn = model_module.run
    requires_catchment = _model_requires_catchment(model)

    # Validate catchment requirement
    if requires_catchment and catchment is None:
        msg = f"Model '{model}' requires catchment parameter"
        raise ValueError(msg)

    # Validate and normalize inputs
    objectives_dict = validate_objectives(objectives)
    resolved_bounds = _validate_bounds(bounds, use_default_bounds, model_module)
    _validate_warmup(warmup, len(forcing), len(observed))

    # Build parameter order and bounds arrays
    param_names: list[str] = list(model_module.PARAM_NAMES)

    lower_bounds = np.array([resolved_bounds[name][0] for name in param_names])
    upper_bounds = np.array([resolved_bounds[name][1] for name in param_names])
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

    # Helper to run the model with the correct signature
    def run_model(params: Any) -> Any:
        if requires_catchment:
            return run_fn(params, forcing, catchment, initial_state=initial_state)
        else:
            return run_fn(params, forcing, initial_state=initial_state)

    # Create single-objective evaluation function for GA
    def evaluate_single(x: np.ndarray) -> float:
        params = model_module.Parameters.from_array(x)
        output = run_model(params)
        sim = output.streamflow[warmup:]
        obs = observed.streamflow
        score = metric_funcs[0](obs, sim)
        return -score if negate_flags[0] else score

    # Create multi-objective evaluation function for NSGA2
    def evaluate_multi(x: np.ndarray) -> np.ndarray:
        params = model_module.Parameters.from_array(x)
        output = run_model(params)
        sim = output.streamflow[warmup:]
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
        best_params = model_module.Parameters.from_array(best_x)

        # Compute actual (non-negated) scores for output
        output = run_model(best_params)
        sim = output.streamflow[warmup:]
        obs = observed.streamflow
        score_dict = {name: float(func(obs, sim)) for name, func in zip(objective_names, metric_funcs, strict=True)}

        return Solution(model=model, parameters=best_params, score=score_dict)

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
            params = model_module.Parameters.from_array(individual.x)

            # Compute actual (non-negated) scores for output
            output = run_model(params)
            sim = output.streamflow[warmup:]
            obs = observed.streamflow
            score_dict = {name: float(func(obs, sim)) for name, func in zip(objective_names, metric_funcs, strict=True)}

            solutions.append(Solution(model=model, parameters=params, score=score_dict))

        return solutions
