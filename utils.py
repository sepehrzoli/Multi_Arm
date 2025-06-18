# utils.py

from typing import Sequence
import numpy as np
from environment import BaseEnvironment

def _kl_bernoulli(p: float, q: float) -> float:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def _kl_gaussian(p: float, q: float, sigma2: float = 1.0) -> float:
    return (p - q) ** 2 / (2.0 * sigma2)

def _compute_w_star(
    mu: Sequence[float],
    *,
    env: BaseEnvironment,
    tol: float = 1e-9,
    max_iter_outer: int = 60,
    max_iter_inner: int = 60
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
