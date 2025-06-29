# track_and_stop.py
import math
import numpy as np
from typing import Tuple, List
from environment import BaseEnvironment
from stopping_conditions import BaseStoppingCondition
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


class HeuristicTracking(BaseTracker):

    def select_arm(self, counts, rewards, env):
        t = int(counts.sum())
        K = self.num_arms

        if t > 0 and t % K == 0:
            return int(np.random.randint(K))


        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat, env=env)

        return int(np.random.choice(K, p=w_star))
class GTracking(BaseTracker):

    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        t = int(counts.sum())
        K = self.num_arms

        if t > 0 and t % K == 0:
            return int(np.random.randint(K))

        mu_hat = rewards / np.maximum(1, counts)
        w_star = _compute_w_star(mu_hat, env=env)

        p = counts / t

        d = w_star - p
        mask = d < 0
        if np.any(mask):
            alphas = p[mask] / (p[mask] - w_star[mask])
            alpha_max = np.min(alphas)
        else:
            alpha_max = 1.0

        x = p + alpha_max * d

        x = np.clip(x, 0.0, None)
        x = x / x.sum()

        return int(np.random.choice(K, p=x))

class DOracleTracking(BaseTracker):
    def __init__(self, num_arms: int):
        super().__init__(num_arms)
        self._w_star = None

    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        t = int(counts.sum())
        K = self.num_arms
        if t == 0:
            return np.random.randint(K)

        thresh = math.sqrt(t) - K / 2.0
        under = [a for a in range(K) if counts[a] < thresh]
        if under:
            return int(min(under, key=lambda a: counts[a]))

        if hasattr(env, "mu"):
            mu = np.asarray(env.mu, dtype=float)
        else:
            mu = np.asarray(env.probs, dtype=float)


        w_star = _compute_w_star(mu, env=env)

        return int(np.argmax(t * w_star - counts))

class RealOracleTracking(BaseTracker):

    def __init__(self, num_arms: int):
        super().__init__(num_arms)
        self._w_star = None

    def select_arm(self, counts, rewards, env: BaseEnvironment) -> int:
        if self._w_star is None:
            if hasattr(env, "mu"):
                mu = np.asarray(env.mu, dtype=float)
            else:
                mu = np.asarray(env.probs, dtype=float)
            self._w_star = _compute_w_star(mu, env=env)

        return int(np.random.choice(self.num_arms, p=self._w_star))
class TrackAndStop(BaseAlgorithm):
    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        tracker: BaseTracker,
        stopping_condition: BaseStoppingCondition
    ):
        super().__init__(env=env, confidence=confidence, tracker=tracker)
        self.stopping_condition = stopping_condition

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        K = self.env.num_arms
        counts  = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # 1) initial exploration
        for a in range(K):
            r = self.env.sample(a)
            counts[a] += 1
            rewards[a] += r
        history.append(counts / counts.sum())

        # 2) main loop
        while not self.stopping_condition.should_stop(
            counts, rewards, self.env, self.confidence
        ):
            arm = self.tracker.select_arm(counts, rewards, self.env)
            r = self.env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            history.append(counts / counts.sum())

        best = int(np.argmax(rewards / counts))
        return best, counts, rewards, history
