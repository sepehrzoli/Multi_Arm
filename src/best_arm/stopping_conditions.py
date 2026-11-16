"""Stopping rules and their configuration registry."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .environment import BaseEnvironment
from .utils import _kl_bernoulli, _kl_gaussian


class BaseStoppingCondition:
    """Interface for deciding when accumulated observations are sufficient."""

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float,
    ) -> bool:
        raise NotImplementedError


class ChernoffStoppingCondition(BaseStoppingCondition):
    """Compare a pairwise Chernoff statistic with a stopping threshold."""

    def __init__(self, mode: str = "practice", tol: float = 1e-5):
        if mode not in ("theory", "practice"):
            raise ValueError("mode must be 'theory' or 'practice'")
        self.mode = mode
        self.tol = tol

    def compute_statistic(
        self, counts: np.ndarray, rewards: np.ndarray, env: BaseEnvironment
    ) -> float:
        """Compute the Chernoff ‘worst’ statistic (without the beta threshold)."""
        K = counts.size
        if np.sum(counts) < K:
            return 0.0

        # choose KL divergence
        if hasattr(env, "sigma"):
            sigma2 = float(env.sigma) ** 2
            kl = lambda p, q: _kl_gaussian(p, q, sigma2)
        else:
            kl = _kl_bernoulli

        mu_hat = rewards / counts
        max_mu = mu_hat.max()
        best_arms = [i for i in range(K) if mu_hat[i] >= max_mu - self.tol]
        best = int(np.random.choice(best_arms))

        NB, SB = counts[best], rewards[best]
        muB = SB / NB
        worst = math.inf
        for i in range(K):
            if i == best:
                continue
            n_ab = NB + counts[i]
            mix = (NB * muB + counts[i] * mu_hat[i]) / n_ab
            val = NB * kl(muB, mix) + counts[i] * kl(mu_hat[i], mix)
            worst = min(worst, val)
        return worst

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float,
    ) -> bool:
        K = counts.size
        t = int(counts.sum())
        if t < K:
            return False

        worst = self.compute_statistic(counts, rewards, env)

        if self.mode == "practice":
            beta = math.log((math.log(t) + 1) / confidence)
        else:  # theory
            if not hasattr(env, "sigma"):
                x = 0.5 * math.log((K - 1) / confidence)
                C_exp = x + 4 * math.log(1 + x + math.sqrt(2 * x))
                beta = 6 * math.log(math.log(t / 2) + 1) + 2 * C_exp
            else:
                beta = 4 * math.log(math.log(t / 2) + 4)

        return worst > beta


class LilUCBStoppingCondition(BaseStoppingCondition):
    """Stop when one arm's pull count dominates the remaining counts."""

    def __init__(self, lambda_param: float | None = None):
        # default to theory lambda (beta=1): ((2+1)/1)^2 = 9
        if lambda_param is None:
            self.lambda_param = ((2.0 + 1.0) / 1.0) ** 2
        else:
            self.lambda_param = lambda_param

    def should_stop(self, counts: np.ndarray, rewards: np.ndarray, env, confidence):
        K = counts.size
        total = counts.sum()
        # Continue while for all i: T_i < 1 + lambda * sum_{j != i} T_j
        # So stop when exists i with T_i >= 1 + lambda * (total - T_i)
        return any(
            counts[i] >= 1 + self.lambda_param * (total - counts[i]) for i in range(K)
        )


class LilStoppingCondition(BaseStoppingCondition):
    """Stop when the leading arm's LIL interval separates from its challengers."""

    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float,
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
        δ = math.log(1 + ε) * (((confidence * ε) / (2 + ε)) ** (1 / (1 + ε)))
        log_denom = δ / K

        Bs = np.zeros(K, dtype=float)
        for i in range(K):
            T = counts[i]
            inner_log = np.log((1 + ε) * T + 2)
            numerator = 2 * sigma2 * (1 + ε) * np.log(2 * inner_log / log_denom)
            Bs[i] = (1 + math.sqrt(ε)) * math.sqrt(numerator / T)

        i_hat = int(np.argmax(mu_hat))
        lhs = mu_hat[i_hat] - Bs[i_hat]
        rhs = np.max(np.delete(mu_hat + Bs, i_hat))
        return lhs >= rhs


class LUCBStoppingCondition(BaseStoppingCondition):
    """Stop when the LUCB confidence gap is below the configured precision."""

    def __init__(self, epsilon: float = 0.02):
        # epsilon here is the required precision gap before stopping
        self.epsilon = epsilon
        # constant from the paper
        self.k1 = 5.0 / 4.0

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float,
    ) -> bool:
        K = counts.size
        if np.any(counts < 1):
            return False

        p_hat = rewards / counts
        t = int(counts.sum())
        δ = confidence

        betas = np.zeros(K, dtype=float)
        for i in range(K):
            u = counts[i]
            if u <= 0:
                continue
            arg = self.k1 * K * (t**4) / δ
            betas[i] = math.sqrt((1.0 / (2.0 * u)) * math.log(arg))

        h_star = int(np.argmax(p_hat))
        scores = p_hat + betas
        scores[h_star] = -np.inf
        l_star = int(np.argmax(scores))

        lhs = p_hat[l_star] + betas[l_star]
        rhs = p_hat[h_star] - betas[h_star]
        return (lhs - rhs) < self.epsilon


class LSAndLUCBStoppingCondition(BaseStoppingCondition):
    """Combine LIL and LUCB stopping rules with split confidence budgets."""

    def __init__(self, lil_eps: float = 0.01, lucb_epsilon: float = 0.0):
        self.ls = LilStoppingCondition(eps=lil_eps)
        self.lucb = LUCBStoppingCondition(epsilon=lucb_epsilon)

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float,
    ) -> bool:
        δ_ls = confidence / 2.0
        δ_lucb = confidence / 2.0

        ok_ls = self.ls.should_stop(counts, rewards, env, δ_ls)
        ok_lucb = self.lucb.should_stop(counts, rewards, env, δ_lucb)
        return ok_ls or ok_lucb


class TopTwoStoppingCondition(BaseStoppingCondition):
    """Stop when every challenger exceeds a transport-cost threshold."""

    def should_stop(
        self,
        counts: np.ndarray,
        rewards: np.ndarray,
        env: BaseEnvironment,
        confidence: float,
    ) -> bool:
        K = counts.size
        t = int(counts.sum())
        n = int(counts.sum())
        if n < K:
            return False
        mu_hat = rewards / counts
        i_hat = int(np.argmax(mu_hat))

        def transport_cost(i: int, j: int) -> float:
            N_i, N_j = counts[i], counts[j]
            mu_i, mu_j = mu_hat[i], mu_hat[j]
            x_star = (N_i * mu_i + N_j * mu_j) / (N_i + N_j)
            if hasattr(env, "probs"):
                return N_i * _kl_bernoulli(mu_i, x_star) + N_j * _kl_bernoulli(
                    mu_j, x_star
                )
            else:
                return (N_i * N_j / (2 * float(env.sigma) ** 2 * (N_i + N_j))) * (
                    mu_i - mu_j
                ) ** 2

        minW = math.inf
        for j in range(K):
            if j == i_hat:
                continue
            w = transport_cost(i_hat, j)
            minW = min(minW, w)

        if hasattr(env, "probs"):
            threshold = math.log(2 * t * (K - 1) / confidence)
        else:
            threshold = math.log((K - 1) / confidence) + 0.5 * math.log(1 + math.log(n))
        return minW > threshold


STOPPING_CONDITION_REGISTRY: Dict[str, Any] = {
    "chernoff": ChernoffStoppingCondition,
    "lil_ucb_original": LilUCBStoppingCondition,
    "lil_stopping": LilStoppingCondition,
    "lucb": LUCBStoppingCondition,
    "ls_and_lucb": LSAndLUCBStoppingCondition,
    "top_two": TopTwoStoppingCondition,
}
