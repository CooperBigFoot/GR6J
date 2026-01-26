"""GR6J hydrological model.

A lumped conceptual rainfall-runoff model for daily streamflow simulation.
Génie Rural à 6 paramètres Journalier.
"""

from .model import Parameters, State, run, step

__all__ = ["Parameters", "State", "step", "run"]
