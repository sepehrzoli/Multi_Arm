# gradient_ascent.py

import numpy as np
import math
from scipy.optimize import linprog

from environment import BaseEnvironment
from model import BaseAlgorithm
from utils import _kl_bernoulli, _kl_gaussian

class FWS(BaseAlgorithm):

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition,
        r_exponent: float = 0.9,
    ):
        super().__init__(env=env, confidence=confidence)
        self.stop = stopping_condition
        self.r_exp = r_exponent
        name = env.__class__.__name__.lower()
        if name.startswith("normal"):
            sigma = getattr(env, "sigma", 1.0)
            self.sigma2 = float(sigma ** 2)
            self.kl = lambda p, q: _kl_gaussian(p, q, self.sigma2)
        else:
            self.kl = _kl_bernoulli

        self.K = env.num_arms

    def _solve_zero_sum(self, M: np.ndarray) -> np.ndarray:
        K, J = M.shape

        c = np.zeros(K + 1)
        c[-1] = -1.0

        A_ub = np.zeros((J, K + 1))
        A_ub[:, :K] = -M.T
        A_ub[:, K] = 1.0
        b_ub = np.zeros(J)

        A_eq = np.zeros((1, K + 1))
        A_eq[0, :K] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * K + [(None, None)]

        res = linprog(
            c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        z = res.x[:K]
        z[z < 1e-12] = 0.0
        return z

    def run(self):
        K = self.K
        env = self.env
        delta = self.confidence
        stop = self.stop

        counts = np.zeros(K, int)
        rewards = np.zeros(K, float)
        history = []

        for a in range(K):
            r = env.sample(a)
            counts[a] += 1
            rewards[a] += r
            history.append(counts / counts.sum())
        t = K

        omega = counts / t
        x = np.ones(K) / K

        # 2) Main loop
        while not stop.should_stop(counts, rewards, env, delta):
            t += 1
            mu_hat = rewards / counts

            is_forced = False
            if (t % K == 0) and (math.isqrt(t // K) ** 2 == t // K):
                is_forced = True
            elif not hasattr(env, "sigma"):
                if np.any(mu_hat <= 0.0) or np.any(mu_hat >= 1.0):
                    is_forced = True

            if is_forced:
                z = np.ones(K) / K
            else:
                i_star = int(np.argmax(mu_hat))

                f_vals = []
                for j in range(K):
                    if j == i_star:
                        continue
                    w_i, w_j = x[i_star], x[j]
                    m_j = (w_i * mu_hat[i_star] + w_j * mu_hat[j]) / (w_i + w_j)
                    f_j = w_i * self.kl(mu_hat[i_star], m_j) + w_j * self.kl(mu_hat[j], m_j)
                    f_vals.append((j, f_j, m_j))

                Fmin = min(f for _, f, _ in f_vals)
                r_t = (t ** (-self.r_exp)) / K

                grads = []
                for j, f_j, m_j in f_vals:
                    if f_j < Fmin + r_t:
                        g = np.zeros(K)
                        g[i_star] = self.kl(mu_hat[i_star], m_j)
                        g[j]      = self.kl(mu_hat[j], m_j)
                        grads.append((j, g))

                J = len(grads)
                M = np.zeros((K, J))
                for idx, (_, g) in enumerate(grads):
                    inner = x.dot(g)
                    M[:, idx] = g - inner

                z = self._solve_zero_sum(M)

            x = ((t - 1) / t) * x + (1 / t) * z

            omega = counts / (t - 1)
            ratios = x / omega
            A_t = int(np.argmax(ratios))

            r = env.sample(A_t)
            counts[A_t] += 1
            rewards[A_t] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history
