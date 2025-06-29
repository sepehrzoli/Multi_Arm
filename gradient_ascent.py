import numpy as np
import math
from typing import Tuple, List
from environment import BaseEnvironment
from model import BaseAlgorithm
from stopping_conditions import BaseStoppingCondition
from utils import _kl_bernoulli, _kl_gaussian

class GradientAscent(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        M: float = float('inf')
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        self.M = M
        self.K = env.num_arms

        self.counts = np.zeros(self.K, dtype=int)
        self.rewards = np.zeros(self.K, dtype=float)
        self.history: List[np.ndarray] = []

        self.pi = np.ones(self.K) / self.K
        self.G = np.zeros(self.K)
        self.tilde_w = self.pi.copy()
        self.w_prime = self.pi.copy()
        self.cum_w = np.zeros(self.K)

        self.t = 0

        name = env.__class__.__name__.lower()
        if name.startswith("normal"):
            sigma2 = float(getattr(env, "sigma", 1.0)) ** 2
            self.kl = lambda p, q: _kl_gaussian(p, q, sigma2)
        else:
            self.kl = _kl_bernoulli

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        for a in range(self.K):
            r = self.env.sample(a)
            self.t += 1
            self.counts[a] += 1
            self.rewards[a] += r
            self.history.append(self.counts / self.counts.sum())
            self.cum_w += self.w_prime

        while True:
            mu_hat = self.rewards / self.counts
            i_star = int(np.argmax(mu_hat))
            best_b = None
            best_val = float('inf')
            best_mix = 0.0
            wi = self.tilde_w[i_star]
            for b in range(self.K):
                if b == i_star:
                    continue
                wb = self.tilde_w[b]
                if wi + wb == 0:
                    continue
                mix = (wi * mu_hat[i_star] + wb * mu_hat[b]) / (wi + wb)
                val = wi * self.kl(mu_hat[i_star], mix) + wb * self.kl(mu_hat[b], mix)
                if val < best_val:
                    best_val, best_b, best_mix = val, b, mix

            g = np.zeros(self.K)
            if best_b is not None:
                g[i_star] = self.kl(mu_hat[i_star], best_mix)
                g[best_b] = self.kl(mu_hat[best_b], best_mix)
            else:
                print("None")

            self.G += g

            eta = 1.0 / math.sqrt(self.t + 1)
            log_w = eta * self.G
            log_w -= np.max(log_w)
            exp_w = np.exp(log_w)
            self.tilde_w = exp_w / np.sum(exp_w)

            gamma = 1.0 / (4.0 * math.sqrt(self.t))
            self.w_prime = (1 - gamma) * self.tilde_w + gamma * self.pi

            self.cum_w += self.w_prime

            deficits = self.cum_w - self.counts
            a_next = int(np.argmax(deficits))

            r = self.env.sample(a_next)
            self.t += 1
            self.counts[a_next] += 1
            self.rewards[a_next] += r
            self.history.append(self.counts / self.counts.sum())

            if self.stopping_condition.should_stop(
                self.counts, self.rewards, self.env, self.confidence
            ):
                break

        i_hat = int(np.argmax(self.rewards / self.counts))
        return i_hat, self.counts, self.rewards, self.history
