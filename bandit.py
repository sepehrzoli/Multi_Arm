# bandit.py

import os, glob, json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from environment import ENV_REGISTRY, BaseEnvironment
from model       import TRACKER_REGISTRY, ALGO_REGISTRY, BaseTracker
from stopping_conditions import STOPPING_CONDITION_REGISTRY, BaseStoppingCondition

def _next_env_index(prefix: str = "env_", suffix: str = ".json") -> int:
    existing = glob.glob(f"{prefix}*{suffix}")
    idxs = []
    for fn in existing:
        name, _ = os.path.splitext(os.path.basename(fn))
        num = name[len(prefix):]
        if num.isdigit():
            idxs.append(int(num))
    return max(idxs, default=0) + 1

def _make_env_filename() -> str:
    i = _next_env_index()
    return f"env_{i:04d}.json"

class Bandit:
    def __init__(
        self,
        *,
        env_name:       Optional[str]            = None,
        algo_name:      str,
        env_args:       Optional[Dict[str, Any]] = None,
        algo_args:      Optional[Dict[str, Any]] = None,
        tracker_name:   Optional[str]            = None,
        tracker_args:   Optional[Dict[str, Any]] = None,
        stopping_name:  Optional[str]            = None,
        stopping_args:  Optional[Dict[str, Any]] = None,
        env_instance:   Optional[BaseEnvironment]= None,
    ):
        # ── 1) Environment ────────────────────────────────
        if env_instance is not None:
            self.env = env_instance
        else:
            if env_name not in ENV_REGISTRY:
                raise ValueError(f"Unknown environment: {env_name}")
            self.env = ENV_REGISTRY[env_name](**(env_args or {}))
            # auto‐save
            path = _make_env_filename()
            self.save_env(path)
            print(f"[Bandit] auto-saved environment to {path}")

        # ── 2) Tracker (for TrackAndStop only) ────────────
        self.tracker: Optional[BaseTracker] = None
        if tracker_name:
            if tracker_name not in TRACKER_REGISTRY:
                raise ValueError(f"Unknown tracker: {tracker_name}")
            self.tracker = TRACKER_REGISTRY[tracker_name](
                self.env.num_arms, **(tracker_args or {})
            )

        # ── 3) Stopping condition (for TrackAndStop & LilUCB) ───
        self.stopping: Optional[BaseStoppingCondition] = None
        if stopping_name:
            if stopping_name not in STOPPING_CONDITION_REGISTRY:
                raise ValueError(f"Unknown stopping condition: {stopping_name}")
            self.stopping = STOPPING_CONDITION_REGISTRY[stopping_name](
                **(stopping_args or {})
            )

        # ── 4) Algorithm ───────────────────────────────────
        if algo_name not in ALGO_REGISTRY:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        self.algo_cls = ALGO_REGISTRY[algo_name]

        # collect args
        self.algo_args: Dict[str, Any] = {
            "env":        self.env,
            "confidence": (algo_args or {}).get("confidence", 0.1),
        }
        for k, v in (algo_args or {}).items():
            if k != "confidence":
                self.algo_args[k] = v
        if self.tracker:
            self.algo_args["tracker"] = self.tracker
        if self.stopping:
            self.algo_args["stopping_condition"] = self.stopping

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        algo = self.algo_cls(**self.algo_args)
        return algo.run()

    def run_n(self, n: int) -> List[Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]]:
        return [self.run() for _ in range(n)]

    # ── Serialization ────────────────────────────────

    def save_env(self, path: str) -> None:
        data = self.env.to_dict()
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_env(path: str) -> BaseEnvironment:
        with open(path) as f:
            data = json.load(f)
        return BaseEnvironment.from_dict(data)

    @classmethod
    def from_saved_env(
        cls,
        path:         str,
        algo_name:    str,
        algo_args:    Optional[Dict[str, Any]] = None,
        tracker_name: Optional[str]            = None,
        stopping_name: Optional[str]            = None,
        stopping_args: Optional[Dict[str, Any]] = None,
        tracker_args: Optional[Dict[str, Any]] = None,
    ) -> "Bandit":
        env = cls.load_env(path)
        return cls(
            env_instance  = env,
            algo_name     = algo_name,
            algo_args     = algo_args,
            tracker_name  = tracker_name,
            tracker_args  = tracker_args,
            stopping_name = stopping_name,
            stopping_args = stopping_args,
        )
