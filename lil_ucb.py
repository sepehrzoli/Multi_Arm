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
        stopping_condition: BaseStoppingCondition,
        epsilon: float = 0.01,
        beta: float = 1,
        lambda_param: float = 9.0
    ):
        # store ν
        super().__init__(env=env, confidence=confidence)
        self.v = confidence
        self.stopping_condition = stopping_condition

        # algorithm params
        self.epsilon = epsilon
        self.beta = beta
        self.lambda_param = lambda_param


        c_e = (
                      ((2.0 + self.epsilon) / self.epsilon)
                      * (1.0 / np.log(1.0 + self.epsilon))
              ) ** (1.0 + self.epsilon)

        self.delta = (
                (np.sqrt(1.0 + self.v) - 1.0) ** 2
                / (4.0 * c_e)
        )

        if hasattr(env, "sigma"):
            self.sigma2 = float(env.sigma ** 2)
        else:
            self.sigma2 = 0.25

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        counts  = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        for i in range(K):
            r = self.env.sample(i)
            counts[i] += 1
            rewards[i] += r
            history.append(counts / counts.sum())

        while not self.stopping_condition.should_stop(
            counts, rewards, self.env, self.v
        ):
            p_hat = rewards / counts
            total = counts.sum()

            scores = np.empty(K, dtype=float)
            for i in range(K):
                Ti = counts[i]
                inner = np.log((1.0 + self.epsilon) * Ti)
                term = np.log(inner / self.delta)
                bonus = (
                    (1.0 + self.beta)
                    * (1.0 + np.sqrt(self.epsilon))
                    * np.sqrt(2.0 * self.sigma2 * (1.0 + self.epsilon) * term / Ti)
                )
                scores[i] = p_hat[i] + bonus

            I_t = int(np.argmax(scores))
            r = self.env.sample(I_t)
            counts[I_t]  += 1
            rewards[I_t] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(counts))
        return best, counts, rewards, history
