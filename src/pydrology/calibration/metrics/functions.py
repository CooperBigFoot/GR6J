"""Hydrological metrics for calibration objectives.

All metrics take (observed, simulated) arrays and return a scalar score.
"""

import numpy as np
from numpy.typing import ArrayLike

from .registry import register


@register("maximize")
def nse(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Range: (-inf, 1], where 1 is perfect match.
    """
    obs = np.asarray(observed)
    sim = np.asarray(simulated)
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        return -np.inf
    return float(1.0 - numerator / denominator)


@register("maximize")
def log_nse(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Nash-Sutcliffe Efficiency on log-transformed flows.

    Emphasizes low-flow performance by log-transforming both series.
    Small constant (1e-6) added to avoid log(0).

    Range: (-inf, 1], where 1 is perfect match.
    """
    obs = np.asarray(observed)
    sim = np.asarray(simulated)
    # Add small constant to avoid log(0)
    log_obs = np.log(obs + 1e-6)
    log_sim = np.log(sim + 1e-6)
    numerator = np.sum((log_obs - log_sim) ** 2)
    denominator = np.sum((log_obs - np.mean(log_obs)) ** 2)
    if denominator == 0:
        return -np.inf
    return float(1.0 - numerator / denominator)


@register("maximize")
def kge(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Kling-Gupta Efficiency.

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    Where:
        r = Pearson correlation coefficient
        alpha = std(sim) / std(obs)
        beta = mean(sim) / mean(obs)

    Range: (-inf, 1], where 1 is perfect match.
    """
    obs = np.asarray(observed)
    sim = np.asarray(simulated)

    # Correlation
    r = 0.0 if np.std(obs) == 0 or np.std(sim) == 0 else float(np.corrcoef(obs, sim)[0, 1])

    # Variability ratio
    alpha = 0.0 if np.std(obs) == 0 else float(np.std(sim) / np.std(obs))

    # Bias ratio
    beta = 0.0 if np.mean(obs) == 0 else float(np.mean(sim) / np.mean(obs))

    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


@register("minimize")
def pbias(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Percent Bias.

    PBIAS = 100 * sum(sim - obs) / sum(obs)

    Positive PBIAS = overestimation, negative = underestimation.
    Optimal value is 0.
    """
    obs = np.asarray(observed)
    sim = np.asarray(simulated)
    if np.sum(obs) == 0:
        return np.inf
    return float(100.0 * np.sum(sim - obs) / np.sum(obs))


@register("minimize")
def rmse(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Root Mean Square Error.

    RMSE = sqrt(mean((obs - sim)^2))

    Range: [0, inf), where 0 is perfect match.
    """
    obs = np.asarray(observed)
    sim = np.asarray(simulated)
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


@register("minimize")
def mae(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Mean Absolute Error.

    MAE = mean(|obs - sim|)

    Range: [0, inf), where 0 is perfect match.
    """
    obs = np.asarray(observed)
    sim = np.asarray(simulated)
    return float(np.mean(np.abs(obs - sim)))
