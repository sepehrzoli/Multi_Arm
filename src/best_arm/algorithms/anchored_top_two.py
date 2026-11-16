"""Anchored and improved anchored top-two sampling algorithms."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition
from ..utils import _kl_bernoulli, _kl_gaussian


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)


class _AT2Base(BaseAlgorithm):
    """Shared sampling engine for anchored top-two variants."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        alpha: Optional[float] = None,  # forced-exploration exponent in (0,1)
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        self.alpha = 0.5 if alpha is None else float(alpha)

        name = env.__class__.__name__.lower()
        if hasattr(env, "sigma") or name.startswith("normal"):
            sigma2 = float(getattr(env, "sigma", 1.0)) ** 2
            self._kl = lambda p, q: _kl_gaussian(p, q, sigma2)
        else:
            self._kl = _kl_bernoulli

        self.K = env.num_arms

    @staticmethod
    def _x_weighted(mu_i: float, mu_j: float, n_i: int, n_j: int) -> float:
        denom = n_i + n_j
        if denom <= 0:
            return 0.5 * (mu_i + mu_j)
        return (n_i * mu_i + n_j * mu_j) / denom

    def _index_against(
        self, mu: np.ndarray, counts: np.ndarray, i_hat: int, a: int
    ) -> float:
        n_i, n_a = counts[i_hat], counts[a]
        x = self._x_weighted(mu[i_hat], mu[a], n_i, n_a)
        return n_i * self._kl(mu[i_hat], x) + n_a * self._kl(mu[a], x)

    def _anchor_g(self, mu: np.ndarray, counts: np.ndarray, i_hat: int) -> float:
        g_sum = 0.0
        for a in range(self.K):
            if a == i_hat:
                continue
            n_i, n_a = counts[i_hat], counts[a]
            x = self._x_weighted(mu[i_hat], mu[a], n_i, n_a)
            num = self._kl(mu[i_hat], x)
            den = self._kl(mu[a], x)
            g_sum += _safe_div(num, den)
        return g_sum - 1.0

    def _select_arm(self, counts: np.ndarray, rewards: np.ndarray) -> int:
        N = int(counts.sum())
        if N < self.K:
            return N if N < self.K else int(np.argmin(counts))

        threshold = (max(N, 1)) ** self.alpha
        if np.any(counts < threshold):
            return int(np.argmin(counts))

        mu = rewards / np.maximum(counts, 1)
        i_hat = int(np.argmax(mu))

        if self._anchor_g(mu, counts, i_hat) > 0.0:
            return i_hat

        best_arm = None
        best_score = float("inf")
        for a in range(self.K):
            if a == i_hat:
                continue
            I_a = self._index_against(mu, counts, i_hat, a)
            score = self._challenger_score(I_a, counts[a])
            if score < best_score:
                best_score = score
                best_arm = a
        return int(best_arm)

    def _challenger_score(self, index_value: float, count_a: int) -> float:
        raise NotImplementedError

    # Core engine.

    def _run_core(
        self,
        *,
        horizon: Optional[int],
        record_stats: bool,
    ):
        K, env = self.K, self.env

        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # Bootstrap (respect horizon if provided)
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

        # Optional recorders
        if record_stats:
            ch = ChernoffStoppingCondition()
            worst_curve: List[float] = []
            prop_hist: List[np.ndarray] = []
            mean_hist: List[np.ndarray] = []

        # Main loop (fixed-confidence vs fixed-horizon)
        while True:
            if horizon is None:
                if self.stopping_condition.should_stop(
                    counts, rewards, env, self.confidence
                ):
                    break
            else:
                if pulls >= horizon:
                    break

            arm = self._select_arm(counts, rewards)
            r = env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            history.append(counts / counts.sum())
            pulls += 1

            if record_stats:
                worst_curve.append(ch.compute_statistic(counts, rewards, env))
                prop_hist.append(counts / counts.sum())
                mean_hist.append(rewards / np.maximum(counts, 1))

        if record_stats:
            worst_curve = np.asarray(worst_curve, dtype=float)
            prop_history = np.vstack(prop_hist) if prop_hist else np.zeros((0, K))
            mean_history = np.vstack(mean_hist) if mean_hist else np.zeros((0, K))
        else:
            worst_curve = prop_history = mean_history = None

        return counts, rewards, history, worst_curve, prop_history, mean_history

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


class AT2(_AT2Base):
    """Anchored top-two sampling with an unadjusted challenger index."""

    def _challenger_score(self, index_value: float, count_a: int) -> float:
        return index_value


class IAT2(_AT2Base):
    """Anchored top-two sampling with a logarithmic challenger adjustment."""

    def _challenger_score(self, index_value: float, count_a: int) -> float:
        return index_value + math.log(max(count_a, 1))
