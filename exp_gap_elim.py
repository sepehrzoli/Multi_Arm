# exp_gap_elim.py

import math
import numpy as np
from typing import Tuple, List
from environment import BaseEnvironment
from model import BaseAlgorithm

class ExpGapElimination(BaseAlgorithm):

    def __init__(self, env: BaseEnvironment, confidence: float):
        # no tracker needed
        super().__init__(env=env, confidence=confidence)

    def _median_elimination(
        self,
        S: List[int],
        epsilon: float,
        delta: float,
        counts: np.ndarray,
        rewards: np.ndarray,
    ) -> Tuple[int, float]:
        eps_l = epsilon / 4.0
        delta_l = delta / 2.0
        active = list(S)

        while True:
            n_pulls = math.ceil(4.0 / eps_l ** 2 * math.log(3.0 / delta_l))
            for arm in active:
                for _ in range(n_pulls):
                    r = self.env.sample(arm)
                    counts[arm] += 1
                    rewards[arm] += r

            means = {a: rewards[a] / counts[a] for a in active}
            median_val = np.median(list(means.values()))
            active = [a for a in active if means[a] >= median_val]

            if len(active) == 1:
                best = active[0]
                return best, means[best]

            eps_l *= 0.75
            delta_l *= 0.5

    def run(self) -> Tuple[int, np.ndarray, np.ndarray]:
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        S = list(range(K))
        r = 1

        while len(S) > 1:
            eps_r = ((2.0 ** -r) / 4.0)
            delta_r = self.confidence / (50.0 * r**3)
            t_r = math.ceil(2.0 / (eps_r ** 2) * math.log(2.0 / delta_r))
            for arm in S:
                for _ in range(t_r):
                    rwd = self.env.sample(arm)
                    counts[arm] += 1
                    rewards[arm] += rwd
            ref, _ = self._median_elimination(
                S, eps_r / 2.0, delta_r, counts, rewards
            )
            means = {a: rewards[a] / counts[a] for a in S}
            mu_ref = means[ref]
            S = [a for a in S if means[a] >= mu_ref - eps_r]
            r += 1

        best_arm = S[0]
        return best_arm, counts, rewards, [[1]]
        #TODO: change later
