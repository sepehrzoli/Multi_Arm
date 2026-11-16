"""Thompson top-two sampling with KKT-based allocation."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition
from ..utils import _kl_bernoulli, _kl_gaussian


class TTSKKT(BaseAlgorithm):
    """Choose a posterior leader and a KKT-based challenger and sampling coin."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        optimism: float = 0.5,
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        self.optimism = float(optimism)
        self.K = env.num_arms

        # pick the right KL + posterior family
        name = env.__class__.__name__.lower()
        if hasattr(env, "sigma") or name.startswith("normal"):
            self.sigma2 = float(getattr(env, "sigma", 1.0)) ** 2
            self._kl = lambda p, q: _kl_gaussian(p, q, self.sigma2)
            self._is_bernoulli = False
        else:
            self.sigma2 = None
            self._kl = _kl_bernoulli
            self._is_bernoulli = True

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        counts, rewards, history, _, _, _ = self._run_core(
            horizon=None, record_stats=False
        )
        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Return the selected arm and post-bootstrap statistics for a fixed horizon.

        The arrays contain Chernoff statistics, allocation proportions, and
        empirical means. Bootstrap length follows the selected algorithm.
        """
        counts, rewards, history, worst_curve, prop_history, mean_history = (
            self._run_core(horizon=horizon, record_stats=True)
        )
        best_arm = int(np.argmax(rewards / counts))
        return best_arm, worst_curve, prop_history, mean_history

    # Shared engine.

    def _run_core(
        self,
        *,
        horizon: Optional[int],
        record_stats: bool,
    ):
        K, env, δ = self.K, self.env, self.confidence

        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # Bootstrap: one pull per arm (respect horizon)
        pulls = 0
        for a in range(K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(a)
            counts[a] += 1
            rewards[a] += r
            history.append(counts / counts.sum())
            pulls += 1

        # If horizon ended during bootstrap, finalize early
        if horizon is not None and pulls >= horizon:
            if record_stats:
                return (
                    counts,
                    rewards,
                    history,
                    np.zeros(0),
                    np.zeros((0, K)),
                    np.zeros((0, K)),
                )
            return counts, rewards, history, None, None, None

        # Recorders for run_fixed
        if record_stats:
            chernoff = ChernoffStoppingCondition()
            worst_curve: List[float] = []
            prop_hist: List[np.ndarray] = []
            mean_hist: List[np.ndarray] = []

        # Main loop
        while True:
            if horizon is None:
                if self.stopping_condition.should_stop(counts, rewards, env, δ):
                    break
            else:
                if pulls >= horizon:
                    break

            t = int(counts.sum())
            mu_hat = rewards / counts
            psi = counts / max(t, 1)

            # Step 1: posterior sample per arm
            if self._is_bernoulli:
                # Beta(1 + S, 1 + N - S)
                alpha = 1.0 + rewards
                beta = 1.0 + counts - rewards
                theta_sample = np.random.beta(alpha, beta)
            else:
                # Normal(mu_hat, sigma^2 / N)
                std = np.sqrt(self.sigma2 / np.maximum(counts, 1))
                theta_sample = np.random.normal(mu_hat, std)

            # Rank thetas; leader index i*, others J
            A = np.argsort(-theta_sample)
            i_star = int(A[0])
            others = A[1:]

            # Step 2: build costs to choose challenger j*
            # C(j) = psi_i d(mu_i, x) + psi_j d(mu_j, x) + optimism * log(1 / (1/N_i + 1/N_j)) / t
            psi_i = float(psi[i_star])
            mu_i = float(mu_hat[i_star])
            n_i = int(counts[i_star])

            best_cost = float("inf")
            j_star = None

            for j in others:
                psi_j = float(psi[j])
                mu_j = float(mu_hat[j])
                n_j = int(counts[j])

                denom = psi_i + psi_j
                x = (psi_i * mu_i + psi_j * mu_j) / (denom if denom > 0 else 1.0)

                cost_base = psi_i * self._kl(mu_i, x) + psi_j * self._kl(mu_j, x)
                bonus = (
                    self.optimism
                    * math.log(1.0 / (1.0 / max(n_i, 1) + 1.0 / max(n_j, 1)))
                    / max(t, 1)
                )
                cost = cost_base + bonus

                if cost < best_cost:
                    best_cost = cost
                    j_star = int(j)

            # Step 3: top-two coin between (i*, j*)
            psi_j = float(psi[j_star])
            mu_j = float(mu_hat[j_star])
            n_j = int(counts[j_star])
            denom = psi_i + psi_j
            x_ij = (psi_i * mu_i + psi_j * mu_j) / (denom if denom > 0 else 1.0)

            coin_num = psi_i * self._kl(mu_i, x_ij)
            coin_den = coin_num + psi_j * self._kl(mu_j, x_ij)
            p_i = coin_num / coin_den if coin_den > 0 else 0.5
            arm = i_star if (np.random.rand() < p_i) else j_star

            # Pull and update
            r = env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            history.append(counts / counts.sum())
            pulls += 1

            # Optional stats
            if record_stats:
                worst_curve.append(chernoff.compute_statistic(counts, rewards, env))
                prop_hist.append(counts / counts.sum())
                mean_hist.append(rewards / np.maximum(counts, 1))

        if record_stats:
            worst_curve = np.asarray(worst_curve, dtype=float)
            prop_history = np.vstack(prop_hist) if prop_hist else np.zeros((0, K))
            mean_history = np.vstack(mean_hist) if mean_hist else np.zeros((0, K))
        else:
            worst_curve = prop_history = mean_history = None

        return counts, rewards, history, worst_curve, prop_history, mean_history
