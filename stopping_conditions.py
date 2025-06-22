import math
import numpy as np
from typing import Dict, Any
from environment import BaseEnvironment
from utils import _kl_bernoulli, _kl_gaussian

class BaseStoppingCondition:
    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float
    ) -> bool:
        raise NotImplementedError

class ChernoffStoppingCondition(BaseStoppingCondition):
    def should_stop(self, counts, rewards, env, confidence):
        # exactly the old TrackAndStop _stop_condition
        if env.__class__.__name__.lower().startswith("normal"):
            sigma2 = float(getattr(env, "sigma", 1.0) ** 2)
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
                mix = (counts[a]*mu_hat[a] + counts[b]*mu_hat[b]) / n_ab
                val = counts[a]*kl(mu_hat[a], mix) + counts[b]*kl(mu_hat[b], mix)
                worst = min(worst, val)
            Z = max(Z, worst)

        beta = math.log((math.log(t)+1) / confidence)
        return Z > beta

class LilUCBStoppingCondition(BaseStoppingCondition):
    def should_stop(self, counts, rewards, env, confidence):
        # original Lil'UCB stop: while all(counts[i] < 1 + 9*(total - counts[i]))
        K = counts.size
        total = counts.sum()
        return not all(counts[i] < 1 + 9 * (total - counts[i]) for i in range(K))

class LilStoppingCondition(BaseStoppingCondition):
    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float
    ) -> bool:
        K = counts.size
        if np.any(counts < 1):
            return False

        mu_hat = rewards / counts
        if env.__class__.__name__.lower().startswith("normal"):
            sigma2 = float(getattr(env, "sigma", 1.0) ** 2)
        else:
            sigma2 = 0.25

        ε = self.eps
        δ = math.log(1+ε) * (((confidence * ε)/(2+ε)) ** (1/(1+ε)))
        log_denom = δ / K

        Bs = np.zeros(K, dtype=float)
        for i in range(K):
            T = counts[i]
            inner_log = np.log((1 + ε) * T + 2)
            numerator   = 2 * sigma2 * (1 + ε) * np.log(2 * inner_log / log_denom)
            Bs[i] = (1 + math.sqrt(ε)) * math.sqrt(numerator / T)

        i_hat = int(np.argmax(mu_hat))
        lhs = mu_hat[i_hat] - Bs[i_hat]
        rhs = np.max(np.delete(mu_hat + Bs, i_hat))
        return lhs >= rhs

# add to registry
STOPPING_CONDITION_REGISTRY: Dict[str, Any] = {
    "chernoff":          ChernoffStoppingCondition,
    "lil_ucb_original":  LilUCBStoppingCondition,
    "lil_stopping":      LilStoppingCondition,
}
