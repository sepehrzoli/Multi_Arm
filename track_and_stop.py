# track_and_stop.py
import math
import numpy as np
from typing import Tuple, List
from environment import BaseEnvironment
from stopping_conditions import BaseStoppingCondition
from utils import _kl_bernoulli, _kl_gaussian, _compute_w_star
from model import BaseTracker, BaseAlgorithm

class CTracking(BaseTracker):
    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        t = int(counts.sum())
        if t == 0:
            return np.random.randint(self.num_arms)
        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat, env=env)
        return int(np.argmax(t * w_star - counts))

class DTracking(BaseTracker):
    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        t = int(counts.sum())
        if t == 0:
            return np.random.randint(self.num_arms)
        thresh = math.sqrt(t) - self.num_arms / 2.0
        under = [a for a in range(self.num_arms) if counts[a] < thresh]
        if under:
            return int(min(under, key=lambda a: counts[a]))
        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat, env=env)
        return int(np.argmax(t * w_star - counts))

class TrackAndStop(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        tracker: BaseTracker,
        stopping_condition: BaseStoppingCondition
    ):
        super().__init__(env=env, confidence=confidence, tracker=tracker)
        self.stopping_condition = stopping_condition

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        counts  = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # 1) initial exploration
        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1
            rewards[a] += r
        history.append(counts / counts.sum())

        # 2) main loop
        while not self.stopping_condition.should_stop(
            counts, rewards, self.env, self.confidence
        ):
            arm = self.tracker.select_arm(counts, rewards, self.env)
            r = self.env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history
