"""Exponential-gap elimination with optional external stopping."""

import math
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition


class ExpGapElimination(BaseAlgorithm):
    """Eliminate arms in rounds using a median-elimination reference arm."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: Optional[BaseStoppingCondition] = None,
    ):
        # no tracker needed
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition

    def _median_elimination(
        self,
        S: List[int],
        epsilon: float,
        delta: float,
        counts: np.ndarray,
        rewards: np.ndarray,
        # propagate stopping condition
        stopping_condition: Optional[BaseStoppingCondition],
    ) -> Tuple[int, float, bool]:
        eps_l = epsilon / 4.0
        delta_l = delta / 2.0
        active = list(S)

        while True:
            n_pulls = math.ceil(4.0 / eps_l**2 * math.log(3.0 / delta_l))
            for arm in active:
                for _ in range(n_pulls):
                    r = self.env.sample(arm)
                    counts[arm] += 1
                    rewards[arm] += r
                    if stopping_condition and stopping_condition.should_stop(
                        counts, rewards, self.env, self.confidence
                    ):
                        # early stop: pick best among active
                        means = {a: rewards[a] / counts[a] for a in active}
                        best = int(max(means, key=lambda k: means[k]))
                        return best, means[best], True

            means = {a: rewards[a] / counts[a] for a in active}
            median_val = np.median(list(means.values()))
            active = [a for a in active if means[a] >= median_val]

            if len(active) == 1:
                best = active[0]
                return best, means[best], False

            eps_l *= 0.75
            delta_l *= 0.5

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        S = list(range(K))
        r = 1
        history: List[np.ndarray] = []

        # initial exploration if using stopping condition
        if self.stopping_condition is not None:
            for a in range(K):
                rw = self.env.sample(a)
                counts[a] += 1
                rewards[a] += rw
            if self.stopping_condition.should_stop(
                counts, rewards, self.env, self.confidence
            ):
                best_arm = int(np.argmax(rewards / counts))
                return best_arm, counts, rewards, [[1]]

        while len(S) > 1:
            eps_r = (2.0**-r) / 4.0
            delta_r = self.confidence / (50.0 * r**3)
            t_r = math.ceil(2.0 / (eps_r**2) * math.log(2.0 / delta_r))

            # sample each arm in S t_r times with per-sample stopping check
            for arm in S:
                for _ in range(t_r):
                    rwd = self.env.sample(arm)
                    counts[arm] += 1
                    rewards[arm] += rwd
                    if self.stopping_condition and self.stopping_condition.should_stop(
                        counts, rewards, self.env, self.confidence
                    ):
                        best_arm = int(np.argmax(rewards / counts))
                        return best_arm, counts, rewards, [[1]]
            if self.stopping_condition is None:
                # Call median elimination without an external stopping check.
                ref, _mean = self._median_elimination(
                    S, eps_r / 2.0, delta_r, counts, rewards, None
                )[:2]
                means = {a: rewards[a] / counts[a] for a in S}
                mu_ref = means[ref]
                S = [a for a in S if means[a] >= mu_ref - eps_r]
            else:
                ref, _mean, stopped_early = self._median_elimination(
                    S, eps_r / 2.0, delta_r, counts, rewards, self.stopping_condition
                )
                if stopped_early:
                    best_arm = int(np.argmax(rewards / counts))
                    return best_arm, counts, rewards, [[1]]
                means = {a: rewards[a] / counts[a] for a in S}
                mu_ref = means[ref]
                S = [a for a in S if means[a] >= mu_ref - eps_r]

            r += 1

            # final check after updating S
            if self.stopping_condition and self.stopping_condition.should_stop(
                counts, rewards, self.env, self.confidence
            ):
                best_arm = int(np.argmax(rewards / counts))
                return best_arm, counts, rewards, [[1]]

        best_arm = S[0]
        # Preserve the algorithm's placeholder allocation history.
        if self.stopping_condition is None:
            return best_arm, counts, rewards, [[1]]
        else:
            return best_arm, counts, rewards, [[1]]
