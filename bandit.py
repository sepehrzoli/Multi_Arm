from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from environment import ENV_REGISTRY, BaseEnvironment
from model import TRACKER_REGISTRY, ALGO_REGISTRY, BaseTracker

defctor = Dict[str, Any]


class Bandit:
    def __init__(
        self,
        env_name: str,
        algo_name: str,
        env_args: Dict[str, Any],
        algo_args: Dict[str, Any],
        tracker_name: Optional[str] = None,
        tracker_args: Optional[Dict[str, Any]] = None,
    ):
        if env_name not in ENV_REGISTRY:
            raise ValueError(f"Unknown environment: {env_name}")
        self.env: BaseEnvironment = ENV_REGISTRY[env_name](**env_args)

        self.tracker: Optional[BaseTracker] = None
        if tracker_name:
            if tracker_name not in TRACKER_REGISTRY:
                raise ValueError(f"Unknown tracker: {tracker_name}")
            self.tracker = TRACKER_REGISTRY[tracker_name](self.env.num_arms, **(tracker_args or {}))

        if algo_name not in ALGO_REGISTRY:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        self.algo_cls = ALGO_REGISTRY[algo_name]

        self.algo_args: Dict[str, Any] = {"env": self.env, "confidence": algo_args.get("confidence")}
        for key, val in algo_args.items():
            if key != "confidence":
                self.algo_args[key] = val
        if self.tracker:
            self.algo_args["tracker"] = self.tracker

    def run(self) -> Tuple[int, np.ndarray, np.ndarray]:
        algo = self.algo_cls(**self.algo_args)
        return algo.run()

    def run_n(self, n: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Run n independent trials"""
        return [self.run() for _ in range(n)]