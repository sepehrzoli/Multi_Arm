import math
import numpy as np
from typing import Tuple, List
from model import BaseAlgorithm
from environment import BaseEnvironment

class TopTwoBAI(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        tracker=None,
        stopping_condition=None,
        **kwargs
    ):
        super().__init__(env=env, confidence=confidence, tracker=tracker)
        self.beta = kwargs['beta']
        self.leader_type = kwargs['leader_type']
        self.challenger_type = kwargs['challenger_type']
        self.stop_cond = stopping_condition
        K = env.num_arms
        self.N = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K, dtype=float)
        self.n = 0
        self.history: List[np.ndarray] = []
        if hasattr(env, 'probs'):
            self.distribution = 'bernoulli'
            self.sigma2 = None
        else:
            self.distribution = 'gaussian'
            self.sigma2 = float(env.sigma) ** 2

    def _initialize(self) -> None:
        K = self.env.num_arms
        for a in range(K):
            r = self.env.sample(a)
            self.N[a] += 1
            self.sum_rewards[a] += r
            self.n += 1
        self.history.append(self.N / self.n)

    def _empirical_means(self) -> np.ndarray:
        return self.sum_rewards / self.N

    def _transport_cost(self, i: int, j: int) -> float:
        N_i, N_j = self.N[i], self.N[j]
        mu_i = self.sum_rewards[i] / N_i
        mu_j = self.sum_rewards[j] / N_j
        x_star = (N_i * mu_i + N_j * mu_j) / (N_i + N_j)
        if self.distribution == 'bernoulli':
            from utils import _kl_bernoulli as kl
            return N_i * kl(mu_i, x_star) + N_j * kl(mu_j, x_star)
        else:
            from utils import _kl_gaussian as klf
            return (N_i * N_j) / (2 * self.sigma2 * (N_i + N_j)) * (mu_i - mu_j) ** 2

    def _choose_leader(self) -> int:
        mus = self._empirical_means()
        if self.leader_type == 'EB':
            return int(np.argmax(mus))
        thetas: List[float] = []
        for i in range(self.env.num_arms):
            N_i = self.N[i]
            mu_i = self.sum_rewards[i] / N_i
            if self.distribution == 'bernoulli':
                alpha = 1 + self.sum_rewards[i]
                beta_param = 1 + N_i - self.sum_rewards[i]
                thetas.append(np.random.beta(alpha, beta_param))
            else:
                thetas.append(np.random.normal(mu_i, math.sqrt(self.sigma2 / N_i)))
        return int(np.argmax(thetas))

    def _choose_challenger(self, B_n: int) -> int:
        K = self.env.num_arms
        if self.challenger_type == 'TC':
            costs = [
                (self._transport_cost(B_n, j) if j != B_n else math.inf)
                for j in range(K)
            ]
            min_cost = min(costs)
            tol = 1e-6
            candidates = [j for j, c in enumerate(costs) if c <= min_cost + tol]
            return int(np.random.choice(candidates))
        if self.challenger_type == 'TCI':
            costs = [
                (
                    self._transport_cost(B_n, j) + math.log(self.N[j])
                    if j != B_n else math.inf
                )
                for j in range(K)
            ]
            min_cost = min(costs)
            tol = 1e-6
            candidates = [j for j, c in enumerate(costs) if c <= min_cost + tol]
            return int(np.random.choice(candidates))
        # RS
        while True:
            thetas = []
            for i in range(K):
                N_i = self.N[i]
                mu_i = self.sum_rewards[i] / N_i
                if self.distribution == 'bernoulli':
                    alpha = 1 + self.sum_rewards[i]
                    beta_param = 1 + N_i - self.sum_rewards[i]
                    thetas.append(np.random.beta(alpha, beta_param))
                else:
                    thetas.append(np.random.normal(mu_i, math.sqrt(self.sigma2 / N_i)))
            c = int(np.argmax(thetas))
            if c != B_n:
                return c

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        self._initialize()
        while not self.stop_cond.should_stop(
            self.N, self.sum_rewards, self.env, self.confidence
        ):
            B_n = self._choose_leader()
            if np.random.rand() < self.beta:
                arm = B_n
            else:
                arm = self._choose_challenger(B_n)
            r = self.env.sample(arm)
            self.N[arm] += 1
            self.sum_rewards[arm] += r
            self.n += 1
            self.history.append(self.N / self.n)
        best = int(np.argmax(self.sum_rewards / self.N))
        return best, self.N, self.sum_rewards, self.history