"""Frank-Wolfe sampling with a linear-programming allocation oracle."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition
from ..utils import _kl_bernoulli, _kl_gaussian


@dataclass
class _FWSState:
    """Local state for one execution of the sampling algorithm."""

    counts: np.ndarray
    rewards: np.ndarray
    history: List[np.ndarray]
    t: int  # conceptual step counter (after bootstrap)
    x: np.ndarray  # current mixture (on simplex)


class FWS(BaseAlgorithm):
    """Allocate samples using Frank-Wolfe updates and a zero-sum linear program."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        r_exponent: float = 0.9,
    ):
        super().__init__(env=env, confidence=confidence)
        self.stop = stopping_condition
        self.r_exp = r_exponent

        name = env.__class__.__name__.lower()
        if name.startswith("normal"):
            sigma = getattr(env, "sigma", 1.0)
            self.sigma2 = float(sigma**2)
            self.kl = lambda p, q: _kl_gaussian(p, q, self.sigma2)
        else:
            self.sigma2 = None
            self.kl = _kl_bernoulli

        self.K = env.num_arms

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        state, _, _, _ = self._run_core(horizon=None, record_stats=False)
        best = int(np.argmax(state.rewards / state.counts))
        return best, state.counts, state.rewards, state.history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Frank–Wolfe Sampling (FWS) for exactly `horizon` pulls.

        Returns
        -------
        best_arm      : int
        worst_curve   : np.ndarray
        prop_history  : np.ndarray
        mean_history  : np.ndarray
        """
        state, worst_curve, prop_history, mean_history = self._run_core(
            horizon=horizon, record_stats=True
        )
        best_arm = int(np.argmax(state.rewards / state.counts))
        return best_arm, worst_curve, prop_history, mean_history

    # Core engine.

    def _run_core(
        self,
        *,
        horizon: Optional[int],
        record_stats: bool,
    ) -> Tuple[
        _FWSState, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        K, env, delta = self.K, self.env, self.confidence

        # fresh local state
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # 1) Bootstrap one pull per arm (respect horizon)
        pulls = 0
        for a in range(K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(a)
            counts[a] += 1
            rewards[a] += r
            history.append(counts / counts.sum())
            pulls += 1

        # If horizon exhausted during bootstrap, we’re done gathering stats.
        if horizon is not None and pulls >= horizon:
            state = _FWSState(
                counts=counts,
                rewards=rewards,
                history=history,
                t=pulls,
                x=np.ones(K) / K,
            )
            if record_stats:
                return state, np.zeros(0), np.zeros((0, K)), np.zeros((0, K))
            return state, None, None, None

        # step counter & mixture
        t = max(pulls, 1)  # keep consistent with original code
        x = np.ones(K) / K

        # Optional recorders
        worst_curve_rec: List[float] = []
        prop_hist_rec: List[np.ndarray] = []
        mean_hist_rec: List[np.ndarray] = []
        chernoff = ChernoffStoppingCondition() if record_stats else None

        # 2) Main loop
        while True:
            # stopping rule (confidence mode only)
            if horizon is None and self.stop.should_stop(counts, rewards, env, delta):
                break
            if horizon is not None and pulls >= horizon:
                break

            # conceptual step increment BEFORE computing z, as in original
            t += 1
            mu_hat = rewards / counts

            # choose z (forced exploration or via LP)
            z = self._choose_z(mu_hat, x, t)

            # Frank–Wolfe update of mixture x
            x = ((t - 1) / t) * x + (1 / t) * z

            # pick arm by ratio rule (x / omega)
            omega = counts / (t - 1)
            ratios = x / np.maximum(omega, 1e-12)
            A_t = int(np.argmax(ratios))

            # pull and update
            r = env.sample(A_t)
            counts[A_t] += 1
            rewards[A_t] += r
            history.append(counts / counts.sum())
            pulls += 1

            # optional stats
            if record_stats and chernoff is not None:
                worst_curve_rec.append(chernoff.compute_statistic(counts, rewards, env))
                prop_hist_rec.append(counts / counts.sum())
                mean_hist_rec.append(rewards / np.maximum(counts, 1))

        state = _FWSState(counts=counts, rewards=rewards, history=history, t=t, x=x)

        if record_stats:
            worst_curve = np.asarray(worst_curve_rec, dtype=float)
            prop_history = (
                np.vstack(prop_hist_rec) if prop_hist_rec else np.zeros((0, K))
            )
            mean_history = (
                np.vstack(mean_hist_rec) if mean_hist_rec else np.zeros((0, K))
            )
        else:
            worst_curve = prop_history = mean_history = None

        return state, worst_curve, prop_history, mean_history

    # Helpers.

    def _choose_z(self, mu_hat: np.ndarray, x: np.ndarray, t: int) -> np.ndarray:
        """Implements the 'z' selection (forced exploration or LP over gradients)."""
        K, env = self.K, self.env

        # Forced exploration triggers:
        is_forced = False
        if (t % K == 0) and (math.isqrt(t // K) ** 2 == t // K):
            is_forced = True
        elif self.sigma2 is None:  # Bernoulli case
            if np.any(mu_hat <= 0.0) or np.any(mu_hat >= 1.0):
                is_forced = True

        if is_forced:
            return np.ones(K) / K

        # Otherwise build gradients around the best arm
        i_star = int(np.argmax(mu_hat))

        f_vals = []
        for j in range(K):
            if j == i_star:
                continue
            w_i, w_j = x[i_star], x[j]
            m_j = (w_i * mu_hat[i_star] + w_j * mu_hat[j]) / max(w_i + w_j, 1e-12)
            f_j = w_i * self.kl(mu_hat[i_star], m_j) + w_j * self.kl(mu_hat[j], m_j)
            f_vals.append((j, f_j, m_j))

        # select near-minimizers within tolerance r_t
        Fmin = min(f for _, f, _ in f_vals)
        r_t = (t ** (-self.r_exp)) / K

        grads = []
        for j, f_j, m_j in f_vals:
            if f_j < Fmin + r_t:
                g = np.zeros(K)
                g[i_star] = self.kl(mu_hat[i_star], m_j)
                g[j] = self.kl(mu_hat[j], m_j)
                grads.append((j, g))

        # set up the zero-sum LP: min_z max_g ⟨g - ⟨x,g⟩1, z⟩
        J = len(grads)
        if J == 0:
            return np.ones(K) / K  # very rare, but keep robust

        M = np.zeros((K, J))
        for idx, (_, g) in enumerate(grads):
            inner = x.dot(g)
            M[:, idx] = g - inner

        return self._solve_zero_sum(M)

    def _solve_zero_sum(self, M: np.ndarray) -> np.ndarray:
        """Solve: min_z max_j ⟨M[:, j], z⟩ s.t. z in simplex."""
        K, J = M.shape

        c = np.zeros(K + 1)
        c[-1] = -1.0  # minimize upper bound 'v' with negative sign -> maximize v

        A_ub = np.zeros((J, K + 1))
        A_ub[:, :K] = -M.T
        A_ub[:, K] = 1.0
        b_ub = np.zeros(J)

        A_eq = np.zeros((1, K + 1))
        A_eq[0, :K] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * K + [(None, None)]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        z = res.x[:K]
        z[z < 1e-12] = 0.0
        return z
