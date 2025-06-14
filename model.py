from typing import Type, Dict, Any, Optional, Sequence
import math
import numpy as np
from environment import BaseEnvironment
def _kl_bernoulli(p, q):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _compute_w_star(mu: Sequence[float],
                    tol: float = 1e-9,
                    max_iter_outer: int = 60,
                    max_iter_inner: int = 60) -> np.ndarray:

    mu = np.asarray(mu, dtype=float)
    K   = mu.size
    if K < 2:
        return np.array([1.0])

    best_idx = int(np.argmax(mu))
    order    = [best_idx] + [i for i in range(K) if i != best_idx]
    u        = mu[order]          # reordered means
    u_best   = u[0]

    # --------------------------------------------------------------
    # 2.  Helper: g_a(x) and its inversion x_from_y
    # --------------------------------------------------------------
    def g(a: int, x: float) -> float:
        """g_a(x) as in Eq. (4)."""
        m = (u_best + x * u[a]) / (1.0 + x)
        return _kl_bernoulli(u_best, m) + x * _kl_bernoulli(u[a], m)

    def x_from_y(a: int, y: float) -> float:
        """
        Find x ≥ 0 s.t. g_a(x) = y  by bisection.
        g_a is strictly increasing, so the solution is unique.
        """
        lo, hi = 0.0, 1.0
        # Expand hi until g_a(hi) ≥ y
        while g(a, hi) < y:
            hi *= 2.0
        for _ in range(max_iter_inner):
            mid = 0.5 * (lo + hi)
            if g(a, mid) < y:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    # --------------------------------------------------------------
    # 3.  Outer root search: find y* with F(y*) = 1
    # --------------------------------------------------------------
    second_best = max(u[1:])
    y_lo, y_hi  = 0.0, _kl_bernoulli(u_best, second_best)

    for _ in range(max_iter_outer):
        y_mid = 0.5 * (y_lo + y_hi)
        F_y = 0.0
        for a in range(1, K):
            x_a = x_from_y(a, y_mid)
            m = (u_best + x_a * u[a]) / (1.0 + x_a)
            num = _kl_bernoulli(u_best, m)
            den = _kl_bernoulli(u[a], m)
            if den < 1e-12:
                continue  # skip or assume contribution ~ 0
            F_y += num / den
        if F_y < 1.0:
            y_lo = y_mid
        else:
            y_hi = y_mid
        if y_hi - y_lo < tol:
            break
    y_star = 0.5 * (y_lo + y_hi)

    x_vec = [1.0] + [x_from_y(a, y_star) for a in range(1, K)]
    w_tmp = np.array(x_vec, dtype=float) / sum(x_vec)

    w_star = np.empty_like(w_tmp)
    for idx_reordered, idx_original in enumerate(order):
        w_star[idx_original] = w_tmp[idx_reordered]

    return w_star


class BaseTracker:
    """Abstract base"""
    def __init__(self, num_arms: int, **kwargs):
        self.num_arms = num_arms

    def select_arm(self, counts: np.ndarray, rewards: np.ndarray) -> int:
        raise NotImplementedError


class CTracking(BaseTracker):
    def select_arm(self, counts: np.ndarray, rewards: np.ndarray) -> int:
        t = int(counts.sum())
        if t == 0:
            return np.random.randint(self.num_arms)
        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat)
        deficits = t * w_star - counts
        return int(np.argmax(deficits))


class DTracking(BaseTracker):
    def select_arm(self, counts: np.ndarray, rewards: np.ndarray) -> int:
        t = int(counts.sum())
        if t == 0:
            return np.random.randint(self.num_arms)
        threshold = math.sqrt(t) - self.num_arms / 2.0
        under = [a for a in range(self.num_arms) if counts[a] < threshold]
        if under:
            return int(min(under, key=lambda a: counts[a]))
        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat)
        deficits = t * w_star - counts
        return int(np.argmax(deficits))

# Registry for trackers
TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    "c_tracking": CTracking,
    "d_tracking": DTracking,
}

#  Algorithms

class BaseAlgorithm:
    """Abstract base"""
    def __init__(self, env: BaseEnvironment, confidence: float, tracker: Optional[BaseTracker] = None, **kwargs):
        self.env = env
        self.confidence = confidence
        self.tracker = tracker

    def run(self):
        raise NotImplementedError


class TrackAndStop(BaseAlgorithm):
    def __init__(self, env: BaseEnvironment, confidence: float, tracker: BaseTracker):
        super().__init__(env=env, confidence=confidence, tracker=tracker)

    def run(self) -> (int, np.ndarray, np.ndarray):
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)

        # Initial exploration
        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1
            rewards[a] += r

        # Main loop
        while True:
            arm = self.tracker.select_arm(counts, rewards)
            r = self.env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            if self._stop_condition(counts, rewards):
                break

        best_arm = int(np.argmax(rewards / counts))
        return best_arm, counts, rewards

    # Chernoff
    def _stop_condition(self, counts: np.ndarray, rewards: np.ndarray) -> bool:
        t = int(counts.sum())
        K = counts.size
        if t < K:
            return False
        mu_hat = rewards / np.maximum(1, counts)
        Z = -math.inf
        for a in range(K):
            worst = math.inf
            for b in range(K):
                if a == b:
                    continue
                n_ab = counts[a] + counts[b]
                mix = (counts[a] * mu_hat[a] + counts[b] * mu_hat[b]) / n_ab
                num = counts[a] * _kl_bernoulli(mu_hat[a], mix) + counts[b] * _kl_bernoulli(mu_hat[b], mix)
                worst = min(worst, num)
            Z = max(Z, worst)
        beta = math.log(2.0 * t * (K - 1) / self.confidence)
        return Z > beta

# Registry
ALGO_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "track_and_stop": TrackAndStop,
}