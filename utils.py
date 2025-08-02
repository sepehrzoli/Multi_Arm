# utils.py
import math
from typing import Sequence, Optional
import numpy as np
from environment import BaseEnvironment

def _kl_bernoulli(p: float, q: float) -> float:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    q = np.clip(q, 1e-8, 1 - 1e-8)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def _kl_gaussian(p: float, q: float, sigma2: float = 1.0) -> float:
    return (p - q) ** 2 / (2.0 * sigma2)

def kl_div(
    d1: float,
    d2: float,
    *,
    env: Optional[BaseEnvironment] = None,
    sigma2: Optional[float] = None
) -> float:
    if sigma2 is not None:
        return _kl_gaussian(d1, d2, sigma2)
    if env is not None and hasattr(env, "sigma"):
        sigma2_val = float(getattr(env, "sigma") ** 2)
        return _kl_gaussian(d1, d2, sigma2_val)
    return _kl_bernoulli(d1, d2)

def _compute_w_star(
    mu: Sequence[float],
    *,
    env: BaseEnvironment,
    tol: float = 1e-3,
    max_iter_outer: int = 13,
    max_iter_inner: int = 13
) -> np.ndarray:
    if env.__class__.__name__.lower().startswith("normal"):
        sigma2 = float(getattr(env, "sigma", 1.0) ** 2)
        kl = lambda p, q: _kl_gaussian(p, q, sigma2)
    else:
        kl = _kl_bernoulli

    mu = np.asarray(mu, dtype=float)
    K = mu.size
    if K < 2:
        return np.array([1.0])

    best = int(np.argmax(mu))
    order = [best] + [i for i in range(K) if i != best]
    u = mu[order]
    u_best = u[0]

    def g(a: int, x: float) -> float:
        m = (u_best + x * u[a]) / (1.0 + x)
        return kl(u_best, m) + x * kl(u[a], m)

    def x_from_y(a: int, y: float) -> float:
        lo, hi = 0.0, 1.0
        while g(a, hi) < y:
            hi *= 2.0
        for _ in range(max_iter_inner):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if g(a, mid) < y else (lo, mid)
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    y_lo, y_hi = 0.0, kl(u_best, max(u[1:]))
    for _ in range(max_iter_outer):
        y_mid = 0.5 * (y_lo + y_hi)
        F = 0.0
        for a in range(1, K):
            x_a = x_from_y(a, y_mid)
            m = (u_best + x_a * u[a]) / (1.0 + x_a)
            num, den = kl(u_best, m), kl(u[a], m)
            if den > 1e-12:
                F += num / den
        (y_lo, y_hi) = (y_mid, y_hi) if F < 1.0 else (y_lo, y_mid)
        if y_hi - y_lo < tol:
            break

    y_star = 0.5 * (y_lo + y_hi)
    x_vec = [1.0] + [x_from_y(a, y_star) for a in range(1, K)]
    w_tmp = np.array(x_vec) / sum(x_vec)

    w_star = np.empty_like(w_tmp)
    for i_reord, i_orig in enumerate(order):
        w_star[i_orig] = w_tmp[i_reord]
    return w_star

def _I_alpha(mu1: float, mu2: float, alpha: float, kl) -> float:
    m = alpha * mu1 + (1 - alpha) * mu2
    return alpha * kl(mu1, m) + (1 - alpha) * kl(mu2, m)


def lower_bound(mu: Sequence[float], *, env: "BaseEnvironment") -> float:
    mu = np.asarray(mu, dtype=float)
    K = mu.size
    if K < 2:
        return 0.0

    if env.__class__.__name__.lower().startswith("normal"):
        sigma2 = float(getattr(env, "sigma", 1.0) ** 2)
        kl = lambda p, q: _kl_gaussian(p, q, sigma2)
    else:
        kl = _kl_bernoulli

    best = int(np.argmax(mu))
    w_star = _compute_w_star(mu, env=env)
    w_best = w_star[best]

    min_val = float("inf")
    for a in range(K):
        if a == best:
            continue
        w_a = w_star[a]
        s = w_best + w_a
        if s <= 0:
            continue
        alpha = w_best / s
        I = _I_alpha(mu[best], mu[a], alpha, kl)
        val = s * I
        if val < min_val:
            min_val = val

    if min_val <= 0:
        return float("inf")
    return 1.0 / min_val


def fixed_confidence_lower_bound(mu, *, env, delta: float) -> float:
    T_star = lower_bound(mu, env=env)
    d = max(1e-12, min(delta, 1 - delta))
    kl_delta = _kl_bernoulli(d, 1 - d)
    return T_star * kl_delta


def binary_search(f, lo: float, hi: float, eps: float = 1e-5, maxiter: int = 15) -> float:
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            return mid
        v = f(mid)
        if v > eps:
            hi = mid
        elif v < -eps:
            lo = mid
        else:
            return mid
    return 0.5 * (lo + hi)

def optimistic_oracle(mu_hat: np.ndarray, counts: np.ndarray, env: BaseEnvironment) -> np.ndarray:
    t = int(np.sum(counts))
    if t <= 0:
        raise ValueError("Total count must be positive")
    logt = math.log(t)

    K = len(mu_hat)
    if np.allclose(mu_hat, mu_hat[0]):
        return np.ones_like(mu_hat, dtype=float) / K

    optimistic_weights = []
    for j in range(K):
        mu_optimistic = np.array([
            find_upper_bound(mu_hat[k], logt / counts[k], env=env) if k == j else find_lower_bound(mu_hat[k], logt / counts[k], env=env)
            for k in range(K)
        ], dtype=float)
        w_j = _compute_w_star(mu_optimistic, env=env)
        optimistic_weights.append(w_j)

    return min(optimistic_weights, key=lambda w: np.sum(w))


def find_upper_bound(
    mu: float,
    v: float,
    *,
    env: Optional[BaseEnvironment] = None,
    sigma2: Optional[float] = None
) -> float:
    if (sigma2 is not None) or (env is not None and hasattr(env, "sigma")):
        if sigma2 is None:
            sigma2 = float(getattr(env, "sigma") ** 2)
        return mu + np.sqrt(2 * sigma2 * v)
    if mu >= 1.0:
        return 1.0
    return binary_search(lambda x: _kl_bernoulli(mu, x) - v, mu, 1.0)


def find_lower_bound(
    mu: float,
    v: float,
    *,
    env: Optional[BaseEnvironment] = None,
    sigma2: Optional[float] = None
) -> float:
    if (sigma2 is not None) or (env is not None and hasattr(env, "sigma")):
        if sigma2 is None:
            sigma2 = float(getattr(env, "sigma") ** 2)
        return mu - np.sqrt(2 * sigma2 * v)
    if mu <= 0.0:
        return 0.0
    return binary_search(lambda x: v - _kl_bernoulli(mu, x), 0.0, mu)




