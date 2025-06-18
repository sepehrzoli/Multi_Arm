# track_and_stop.py

import math
import numpy as np
from typing import Tuple, List
from environment import BaseEnvironment
from utils import _kl_bernoulli, _kl_gaussian, _compute_w_star
from model import BaseTracker, BaseAlgorithm

class CTracking(BaseTracker):
    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        t = int(counts.sum())
        if t == 0:
            return np.random.randint(self.num_arms)
        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat, env=env)
        return int(np.argmax(t * w_star - counts))

class DTracking(BaseTracker):
    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        t = int(counts.sum())
        if t == 0:
            return np.random.randint(self.num_arms)
        thresh = math.sqrt(t) - self.num_arms / 2.0
        under = [a for a in range(self.num_arms) if counts[a] < thresh]
        if under:
            return int(min(under, key=lambda a: counts[a]))
        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat, env=env)
        return int(np.argmax(t * w_star - counts))

class TrackAndStop(BaseAlgorithm):
    def __init__(self, env: BaseEnvironment, confidence: float, tracker: BaseTracker):
        super().__init__(env=env, confidence=confidence, tracker=tracker)

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        self.sampling_history: List[np.ndarray] = []

        # initial exploration: one pull each
        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1
            rewards[a] += r
        # record after initial K pulls
        self.sampling_history.append(counts / counts.sum())

        # main loop
        while True:
            arm = self.tracker.select_arm(counts, rewards, self.env)
            r = self.env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            # record fraction after each pull
            self.sampling_history.append(counts / counts.sum())
            if self._stop_condition(counts, rewards):
                break

        best_arm = int(np.argmax(rewards / counts))
        return best_arm, counts, rewards, self.sampling_history

    def _stop_condition(self, counts, rewards) -> bool:
        if self.env.__class__.__name__.lower().startswith("normal"):
            sigma2 = float(getattr(self.env, "sigma", 1.0) ** 2)
            kl = lambda p, q: _kl_gaussian(p, q, sigma2)
        else:
            kl = _kl_bernoulli

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
                val = counts[a] * kl(mu_hat[a], mix) + counts[b] * kl(mu_hat[b], mix)
                worst = min(worst, val)
            Z = max(Z, worst)
        beta = math.log(2.0 * t * (K - 1) / self.confidence)
        return Z > beta
