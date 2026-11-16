"""Thompson sampling with posterior pair selection."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition
from ..utils import _kl_bernoulli, _kl_gaussian


class TSPPS(BaseAlgorithm):
    """Select a posterior leader, a probabilistic challenger, and a sampling coin."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        self.K = env.num_arms

        # pick KL + posterior family
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
        K, env, delta = self.K, self.env, self.confidence

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

        # Early exit if horizon exhausted during bootstrap
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

        # Recorders (for run_fixed)
        if record_stats:
            chernoff = ChernoffStoppingCondition()
            worst_curve: List[float] = []
            prop_hist: List[np.ndarray] = []
            mean_hist: List[np.ndarray] = []

        # Main loop
        while True:
            if horizon is None:
                if self.stopping_condition.should_stop(counts, rewards, env, delta):
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

            # Rank by samples; leader i*
            A = np.argsort(-theta_sample)
            i_star = int(A[0])
            others = A[1:]

            # Step 2: pair probabilities over (i*, j)
            vals = np.empty(len(others), dtype=float)
            psi_i = float(psi[i_star])
            mu_i = float(mu_hat[i_star])
            n_i = int(counts[i_star])

            if self._is_bernoulli:
                # vals[j] = -t * [psi_i d(mu_i, x) + psi_j d(mu_j, x)]
                for idx, j in enumerate(others):
                    psi_j = float(psi[j])
                    mu_j = float(mu_hat[j])
                    denom = psi_i + psi_j
                    x = (psi_i * mu_i + psi_j * mu_j) / (denom if denom > 0 else 1.0)
                    vals[idx] = -t * (
                        psi_i * self._kl(mu_i, x) + psi_j * self._kl(mu_j, x)
                    )
            else:
                # vals[j] = log(1 - Φ(x)), x = (μ_i - μ_j)/(σ*sqrt(1/N_i + 1/N_j))
                sigma = math.sqrt(self.sigma2)
                root2 = math.sqrt(2.0)
                for idx, j in enumerate(others):
                    n_j = int(counts[j])
                    denom = sigma * math.sqrt(1.0 / max(n_i, 1) + 1.0 / max(n_j, 1))
                    x = (mu_i - float(mu_hat[j])) / (denom if denom > 0 else 1.0)
                    vals[idx] = math.log(0.5 * float(math.erfc(x / root2)))

            vmax = np.max(vals)
            probs = np.exp(vals - vmax)
            s = probs.sum()
            if s <= 0 or not np.isfinite(s):
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / s

            j_star = (
                int(np.random.choice(others, p=probs))
                if len(others) > 1
                else int(others[0])
            )

            # Step 3: biased coin between (i*, j*)
            psi_j = float(psi[j_star])
            mu_j = float(mu_hat[j_star])
            denom = psi_i + psi_j
            x_ij = (psi_i * mu_i + psi_j * mu_j) / (denom if denom > 0 else 1.0)

            coin_num = psi_i * self._kl(mu_i, x_ij)
            coin_den = coin_num + psi_j * self._kl(mu_j, x_ij)
            p_i = coin_num / coin_den if coin_den > 0 else 0.5
            arm = i_star if (np.random.rand() < p_i) else j_star

            # Pull & update
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
