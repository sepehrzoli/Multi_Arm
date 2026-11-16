"""Public interface for the best-arm identification framework."""

__all__ = [
    "Bandit",
    "BaseEnvironment",
    "BernoulliEnvironment",
    "NormalEnvironment",
    "BaseAlgorithm",
    "BaseTracker",
    "STOPPING_CONDITION_REGISTRY",
    "TRACKER_REGISTRY",
    "ALGO_REGISTRY",
    "run_experiment",
    "__version__",
]

__version__ = "0.1.2"

from .bandit import Bandit
from .environment import BaseEnvironment, BernoulliEnvironment, NormalEnvironment
from .experiments import run_experiment
from .model import ALGO_REGISTRY, TRACKER_REGISTRY, BaseAlgorithm, BaseTracker
from .stopping_conditions import STOPPING_CONDITION_REGISTRY
