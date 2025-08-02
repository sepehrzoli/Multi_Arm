import math
from typing import List, Tuple
import numpy as np

from environment import BaseEnvironment, BernoulliEnvironment, NormalEnvironment
from model import BaseAlgorithm
from stopping_conditions import BaseStoppingCondition
from utils import binary_search
from utils import find_upper_bound, find_lower_bound, kl_div


class AdaHedge:
    def __init__(self, K: int):
        self.K = K
        self.L = np.zeros(K, dtype=float)  # cumulative losses
        self.delta = 0.01

    def act(self) -> np.ndarray:
        η = math.log(self.K) / self.delta
        m = self.L.min()
        u_unn = np.exp(-η * (self.L - m))
        return u_unn / u_unn.sum()

    def incur(self, loss: np.ndarray) -> None:
        u = self.act()
        η = math.log(self.K) / self.delta
        def M(arr: np.ndarray) -> float:
            m0 = arr.min()
            exps = np.exp(-η * (arr - m0))
            return m0 - (1/η) * math.log(exps.mean())
        
        Mpre = M(self.L)
        self.L += loss
        Mpst = M(self.L)
        m = Mpst - Mpre
        mixl = float(u.dot(loss))
        self.delta += mixl - m

class AdaHedgeBestResponse(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        exploration_bonus=None
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        # exploration bonus f(t); default log(t+1)
        self.f = exploration_bonus or (lambda t: math.log(t+1))

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        if isinstance(self.env, NormalEnvironment):
            sigma2 = float(self.env.sigma ** 2)
        else:
            sigma2 = None
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1; rewards[a] += r

        k_player = AdaHedge(K)
        sum_w = np.ones(K, dtype=float)
        history: List[np.ndarray] = []
        while True:
            t = int(counts.sum())
            mu_hat = rewards / counts
            if self.stopping_condition.should_stop(counts, rewards, self.env, self.confidence):
                break
            ft = self.f(t)
            alpha = np.zeros(K); beta = np.zeros(K)
            for k in range(K):
                v = ft / counts[k]
                alpha[k] = find_lower_bound(mu_hat[k], v, env=self.env, sigma2=sigma2)
                beta[k]  = find_upper_bound(mu_hat[k], v, env=self.env, sigma2=sigma2)

            i_t = int(np.argmax(mu_hat))
            w_t = k_player.act()

            w_i = w_t[i_t]
            best_val = float('inf')
            lambda_s = mu_hat.copy()
            for j in range(K):
                if j == i_t:
                    continue
                w_j = w_t[j]
                theta = (w_i * mu_hat[i_t] + w_j * mu_hat[j]) / (w_i + w_j)
                val = (w_i * kl_div(mu_hat[i_t], theta, env=self.env, sigma2=sigma2)
                       + w_j * kl_div(mu_hat[j], theta, env=self.env, sigma2=sigma2))
                if val < best_val:
                    best_val = val
                    lambda_s = mu_hat.copy()
                    lambda_s[i_t] = theta
                    lambda_s[j] = theta

            U = np.zeros(K)
            for k in range(K):
                du = kl_div(beta[k], lambda_s[k], env=self.env, sigma2=sigma2)
                dd = kl_div(alpha[k], lambda_s[k], env=self.env, sigma2=sigma2)
                bonus = ft / counts[k]
                U[k] = max(du, dd, bonus)

            k_player.incur(-U)
            sum_w += w_t
            arm = int(np.argmin(counts - sum_w))
            r = self.env.sample(arm)
            counts[arm] += 1; rewards[arm] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history
