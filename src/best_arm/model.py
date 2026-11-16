"""Base interfaces and registries for algorithms and tracking policies."""

from typing import Dict, Optional, Type

from .environment import BaseEnvironment


class BaseTracker:
    """Base configuration and arm-selection interface for tracking policies."""

    def __init__(self, num_arms: int, **kwargs):
        self.num_arms = num_arms

    def select_arm(self, counts, rewards, env: BaseEnvironment):
        """Select an arm from the current observations and environment."""
        raise NotImplementedError


class BaseAlgorithm:
    """Base configuration and fixed-confidence execution interface for algorithms."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        tracker: Optional[BaseTracker] = None,
        **kwargs,
    ):
        self.env = env
        self.confidence = confidence
        self.tracker = tracker

    def run(self):
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        raise NotImplementedError


# Registry imports follow the base classes because algorithm modules import them.
from .algorithms.adahedge import AdaHedgeBestResponse
from .algorithms.anchored_top_two import AT2, IAT2
from .algorithms.exp_gap_elim import ExpGapElimination
from .algorithms.frank_wolfe import FWS
from .algorithms.gradient_ascent import GradientAscent
from .algorithms.lil_ucb import LilUCB
from .algorithms.lucb1 import LUCB1
from .algorithms.pmax_best_seen import PMaxBestSeen
from .algorithms.top_two import TopTwoBAI
from .algorithms.track_and_stop import (
    BestChallengerTracking,
    CTracking,
    DOracleTracking,
    DTracking,
    GTracking,
    HeuristicTracking,
    NoisyTracking,
    OptimisticTracking,
    RealOracleTracking,
    SamplingTracking,
    TrackAndStop,
    UniformTracking,
)
from .algorithms.tskkt import TTSKKT
from .algorithms.tspps import TSPPS

TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    "c_tracking": CTracking,
    "d_tracking": DTracking,
    "heuristic_tracking": HeuristicTracking,
    "g_tracking": GTracking,
    "d_oracle_tracking": DOracleTracking,
    "real_oracle_tracking": RealOracleTracking,
    "sampling_tracking": SamplingTracking,
    "bc": BestChallengerTracking,
    "optimistic_tracking": OptimisticTracking,
    "noisy_tracking": NoisyTracking,
    "uniform_tracking": UniformTracking,
}

ALGO_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "track_and_stop": TrackAndStop,
    "exp_gap_elim": ExpGapElimination,
    "lil_ucb": LilUCB,
    "lucb": LUCB1,
    "lazyma": GradientAscent,
    "fws": FWS,
    "top_two": TopTwoBAI,
    "adahedge_br": AdaHedgeBestResponse,
    "iat2": IAT2,
    "at2": AT2,
    "ttskkt": TTSKKT,
    "tspps": TSPPS,
    "pmax": PMaxBestSeen,
}
