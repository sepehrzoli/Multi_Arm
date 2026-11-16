"""Gaussian best-seen probability estimates and sampling policies."""

from typing import List, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm, BaseTracker
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition

# Utilities.


def _stdnorm_cdf(z: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF Φ(z) using an erf approximation (Abramowitz–Stegun 7.1.26).
    No reliance on np.erf/np.erfc to keep compatibility across NumPy builds.
    """
    x = z / np.sqrt(2.0)

    # erf(x) ≈ sign(x) * (1 - (((((a5 t + a4) t + a3) t + a2) t + a1) t) e^{-x^2}),
    # with t = 1 / (1 + p |x|)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    erf_approx = np.sign(x) * (1.0 - poly * np.exp(-x * x))

    phi = 0.5 * (1.0 + erf_approx)
    # Keep within [0,1] to be safe numerically
    return np.clip(phi, 0.0, 1.0)


def _prob_arm_is_max(
    i: int, mu_hat: np.ndarray, n: np.ndarray, sigma: float, gh_nodes: int = 50
) -> float:
    """
    Approximate P(arm i is the maximum) for independent Gaussians
        X_j ~ N(mu_hat_j, sigma^2 / n_j),  with all n_j >= 1.
    Uses Gauss–Hermite quadrature; returns a clipped probability in [0,1].
    """
    K = mu_hat.shape[0]
    assert n.shape[0] == K
    if np.any(n <= 0):
        raise ValueError("All arms must have at least one sample (n_j >= 1).")

    sig = sigma / np.sqrt(n)  # σ_j for each arm
    sig_i = sig[i]

    # Gauss–Hermite nodes/weights for ∫ e^{-y^2} g(y) dy ≈ Σ w_k g(x_k)
    x, w = np.polynomial.hermite.hermgauss(gh_nodes)  # shapes (M,), (M,)

    # Map nodes to x for arm i
    x_i = mu_hat[i] + np.sqrt(2.0) * sig_i * x  # shape (M,)

    # For each node, product over j≠i of Φ( (x_i - μ_j)/σ_j )
    inv_sig = 1.0 / sig
    z = (x_i[:, None] - mu_hat[None, :]) * inv_sig[None, :]  # (M, K)
    F = _stdnorm_cdf(z)
    F = np.clip(F, 1e-300, 1.0)  # stabilize logs
    logF = np.log(F)
    sum_logF = np.sum(logF, axis=1) - logF[:, i]  # exclude i
    prod_excl_i = np.exp(sum_logF)  # (M,)

    val = (1.0 / np.sqrt(np.pi)) * np.sum(w * prod_excl_i)
    return float(np.clip(val, 0.0, 1.0))


def _prob_best_seen_is_correct(
    mu_hat: np.ndarray, n: np.ndarray, sigma: float, i_best: int, gh_nodes: int = 50
) -> float:
    """Probability that the currently best-seen arm i_best is truly the best."""
    return _prob_arm_is_max(i_best, mu_hat, n, sigma, gh_nodes)


# Tracker.


class PMaxBestSeenTracker(BaseTracker):
    """
    Tracker that picks the arm which (if pulled once more) maximizes the
    probability that the current empirical best remains truly the best.
    Works only for Gaussian environments (requires env.sigma).
    """

    def __init__(self, num_arms: int, quad_nodes: int = 50):
        super().__init__(num_arms)
        self.quad_nodes = int(quad_nodes)

    def select_arm(
        self, counts: np.ndarray, rewards: np.ndarray, env: BaseEnvironment
    ) -> int:
        # Safety: only for Gaussian envs with known sigma
        """Select an arm from the current observations and environment."""
        if not hasattr(env, "sigma"):
            raise ValueError(
                "PMaxBestSeen works only for Gaussian environments with attribute `sigma`."
            )
        sigma = float(env.sigma)

        K = self.num_arms
        t = int(counts.sum())

        # Forced exploration: ensure each arm has >=1 sample
        for a in range(K):
            if counts[a] <= 0:
                return int(a)

        # Empirical means
        mu_hat = rewards / np.maximum(1, counts)

        # Current best-seen arm
        i_best = int(np.argmax(mu_hat))

        # Myopic look-ahead: test each candidate single increment
        best_a = 0
        best_val = -np.inf
        for a in range(K):
            n_hyp = counts.copy()
            n_hyp[a] += 1
            p = _prob_best_seen_is_correct(
                mu_hat=mu_hat,
                n=n_hyp,
                sigma=sigma,
                i_best=i_best,
                gh_nodes=self.quad_nodes,
            )
            if p > best_val + 1e-15:
                best_val = p
                best_a = a
            elif abs(p - best_val) <= 1e-15:
                # Tie-breaker: prefer the arm with smaller count (higher variance)
                if counts[a] < counts[best_a]:
                    best_a = a
        return int(best_a)


# Algorithm.


class PMaxBestSeen(BaseAlgorithm):
    """Sample Gaussian arms to increase confidence in the current empirical best."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        quad_nodes: int = 50,
    ):
        tracker = PMaxBestSeenTracker(num_arms=env.num_arms, quad_nodes=quad_nodes)
        super().__init__(env=env, confidence=confidence, tracker=tracker)
        if not hasattr(env, "sigma"):
            raise ValueError(
                "PMaxBestSeen requires a Gaussian environment with known `sigma`."
            )
        self.stopping_condition = stopping_condition
        self.quad_nodes = int(quad_nodes)

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Normal / fixed-confidence mode.
        Returns: best, counts, rewards, history (allocations over time).
        """
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # 1) initial exploration: pull each arm once
        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1
            rewards[a] += r
        history.append(counts / counts.sum())

        # 2) main loop until stopping condition triggers
        while not self.stopping_condition.should_stop(
            counts, rewards, self.env, self.confidence
        ):
            arm = self.tracker.select_arm(counts, rewards, self.env)
            r = self.env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(rewards / np.maximum(1, counts)))
        return best, counts, rewards, history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Return the selected arm and post-bootstrap statistics for a fixed horizon.

        The arrays contain Chernoff statistics, allocation proportions, and
        empirical means. Bootstrap length follows the selected algorithm.
        """
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)

        # 1) bootstrap.
        pulls = 0
        numberEx = 5
        for a in range(numberEx * K):
            if pulls >= horizon:
                break
            r = self.env.sample(a % K)
            counts[a % K] += 1
            rewards[a % K] += r
            pulls += 1

        remaining = max(horizon - pulls, 0)
        worst_curve = np.zeros(remaining, dtype=float)
        prop_history = np.zeros((remaining, K), dtype=float)
        mean_history = np.zeros((remaining, K), dtype=float)

        chernoff = ChernoffStoppingCondition()
        t_idx = 0

        # 2) continue until horizon.
        while pulls < horizon:
            arm = self.tracker.select_arm(counts, rewards, self.env)
            r = self.env.sample(arm)

            counts[arm] += 1
            rewards[arm] += r
            pulls += 1

            worst_curve[t_idx] = chernoff.compute_statistic(counts, rewards, self.env)
            prop_history[t_idx] = counts / counts.sum()
            mean_history[t_idx] = rewards / np.maximum(1, counts)
            t_idx += 1

        best_arm = int(np.argmax(rewards / np.maximum(1, counts)))
        return best_arm, worst_curve, prop_history, mean_history
