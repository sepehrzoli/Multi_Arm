# lil_ucb.py

import numpy as np
from typing import Tuple, List
from environment import BaseEnvironment
from model import BaseAlgorithm
from stopping_conditions import BaseStoppingCondition


class LilUCB(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition

    def _bonus(self, n: int) -> float:
        δ = self.confidence
        numerator = np.log(1.01 * n + 2)
        denom = ((1/1005) * δ) ** (1 / 1.01)
        inner = np.log(numerator / denom)
        return 2.2 * np.sqrt((0.505 / n) * inner)

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        counts  = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1
            rewards[a] += r
            history.append(counts / counts.sum())

        while not self.stopping_condition.should_stop(
            counts, rewards, self.env, self.confidence
        ):
            mu_hat  = rewards / np.maximum(1, counts)
            bonuses = np.array([self._bonus(counts[i]) for i in range(K)])
            arm     = int(np.argmax(mu_hat + bonuses))

            r = self.env.sample(arm)
            counts[arm]  += 1
            rewards[arm] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(counts))
        return best, counts, rewards, history
