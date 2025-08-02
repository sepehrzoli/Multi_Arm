# lil_ucb.py

import numpy as np
from typing import Tuple, List, Optional
from environment import BaseEnvironment
from model import BaseAlgorithm
from stopping_conditions import BaseStoppingCondition, LilUCBStoppingCondition


class LilUCB(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: Optional[BaseStoppingCondition] = None,
        mode: str = "theory",  # "theory" or "heuristic"
        epsilon: Optional[float] = None,
        beta: Optional[float] = None,
        lambda_param: Optional[float] = None,
    ):
        super().__init__(env=env, confidence=confidence)
        self.v = confidence

        K = env.num_arms

        # derive defaults from mode
        if mode == "theory":
            default_epsilon = 0.01
            default_beta = 1.0
            default_lambda = ((2.0 + default_beta) / default_beta) ** 2
        elif mode == "heuristic":
            default_epsilon = 0
            default_beta = 0.5
            default_lambda = 1.0 + 10.0 / K
        else:
            raise ValueError(f"Unknown mode: {mode}; use 'theory' or 'heuristic'")

        # allow explicit overrides
        self.epsilon = epsilon if epsilon is not None else default_epsilon
        self.beta = beta if beta is not None else default_beta
        self.lambda_param = lambda_param if lambda_param is not None else default_lambda

        # stopping condition: external takes precedence, but if it's LilUCBStoppingCondition,
        # ensure its lambda matches (unless the user explicitly passed a stopping condition
        # with its own desired lambda, in which case they can construct it themselves).
        if stopping_condition is None:
            self.stopping_condition = LilUCBStoppingCondition(lambda_param=self.lambda_param)
        elif isinstance(stopping_condition, LilUCBStoppingCondition):
            # sync its lambda to the current lambda_param
            stopping_condition.lambda_param = self.lambda_param
            self.stopping_condition = stopping_condition
        else:
            self.stopping_condition = stopping_condition

        # delta computation (only meaningful when epsilon > 0)
        if self.epsilon > 0:
            c_e = (
                ((2.0 + self.epsilon) / self.epsilon)
                * (1.0 / np.log(1.0 + self.epsilon))
            ) ** (1.0 + self.epsilon)
            self.delta = ((np.sqrt(1.0 + self.v) - 1.0) ** 2) / (4.0 * c_e)
        else:
            self.delta = self.v / 5

        if hasattr(env, "sigma"):
            self.sigma2 = float(env.sigma ** 2)
        else:
            self.sigma2 = 0.25

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # initial one pull per arm
        for i in range(K):
            r = self.env.sample(i)
            counts[i] += 1
            rewards[i] += r

        history.append(counts / counts.sum())

        while not self.stopping_condition.should_stop(counts, rewards, self.env, self.v):
            p_hat = rewards / counts
            total = counts.sum()

            scores = np.empty(K, dtype=float)
            for i in range(K):
                Ti = counts[i]
                inner = np.log((1.0 + self.epsilon) * Ti) if self.epsilon > 0 else np.log(1.0 + Ti)
                term = np.log(inner / self.delta) if self.epsilon > 0 else np.log(inner / max(self.delta, 1e-12))
                bonus = (
                    (1.0 + self.beta)
                    * (1.0 + np.sqrt(self.epsilon))
                    * np.sqrt(2.0 * self.sigma2 * (1.0 + self.epsilon) * term / Ti)
                    if self.epsilon > 0
                    else (1.0 + self.beta) * np.sqrt(2.0 * self.sigma2 * term / Ti)
                )
                scores[i] = p_hat[i] + bonus

            I_t = int(np.argmax(scores))
            r = self.env.sample(I_t)
            counts[I_t] += 1
            rewards[I_t] += r
            history.append(counts / counts.sum())

        # best arm is the one with highest empirical mean
        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history
