import numpy as np
from typing import Tuple, List
from environment import BaseEnvironment
from model       import BaseAlgorithm
from stopping_conditions import BaseStoppingCondition

class LUCB1(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition

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
            p_hat = rewards / counts
            t = int(counts.sum())
            δ = self.confidence
            k1 = 5.0 / 4.0

            betas = np.zeros(K, dtype=float)
            for i in range(K):
                u = counts[i]
                arg = k1 * K * (t ** 4) / δ
                betas[i] = np.sqrt((1.0 / (2.0 * u)) * np.log(arg))

            h_star = int(np.argmax(p_hat))
            scores = p_hat + betas
            scores[h_star] = -np.inf
            l_star = int(np.argmax(scores))

            for arm in (h_star, l_star):
                r = self.env.sample(arm)
                counts[arm]  += 1
                rewards[arm] += r
                history.append(counts / counts.sum())

        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history