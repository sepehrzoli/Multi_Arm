"""AdaHedge learning and best-response sampling for best-arm identification."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment, NormalEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition
from ..utils import find_lower_bound, find_upper_bound, kl_div


class AdaHedge:
    """Maintain an adaptive exponential-weights distribution over cumulative losses."""

    def __init__(self, K: int):
        self.K = K
        self.L = np.zeros(K, dtype=float)  # cumulative losses
        self.delta = 0.01

    def act(self) -> np.ndarray:
        η = math.log(self.K) / self.delta
        m = self.L.min()
        u_unn = np.exp(-η * (self.L - m))
        return u_unn / u_unn.sum()

    def incur(self, loss: np.ndarray) -> None:
        u = self.act()
        η = math.log(self.K) / self.delta

        def M(arr: np.ndarray) -> float:
            m0 = arr.min()
            exps = np.exp(-η * (arr - m0))
            return m0 - (1 / η) * math.log(exps.mean())

        Mpre = M(self.L)
        self.L += loss
        Mpst = M(self.L)
        m = Mpst - Mpre
        mixl = float(u.dot(loss))
        self.delta += mixl - m


@dataclass
class _AHState:
    """Local state for one execution of the sampling algorithm."""

    counts: np.ndarray
    rewards: np.ndarray
    history: List[np.ndarray]
    k_player: AdaHedge
    sum_w: np.ndarray  # cumulative mixtures


class AdaHedgeBestResponse(BaseAlgorithm):
    """Allocate samples using AdaHedge and an adversarial best response."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        exploration_bonus=None,
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        # exploration bonus f(t); default log(t+1)
        self.f = exploration_bonus or (lambda t: math.log(t + 1))

        # cache distribution type info
        self.sigma2 = (
            float(getattr(env, "sigma", 0.0) ** 2)
            if isinstance(env, NormalEnvironment)
            else None
        )

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        state, _, _, _ = self._run_core(horizon=None, record_stats=False)
        best = int(np.argmax(state.rewards / state.counts))
        return best, state.counts, state.rewards, state.history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run AdaHedge-BR for exactly `horizon` pulls.

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
        _AHState, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        K, env = self.env.num_arms, self.env

        # fresh local state
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # Bootstrap one pull per arm (respect horizon)
        pulls = 0
        for a in range(K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(a)
            counts[a] += 1
            rewards[a] += r
            history.append(counts / counts.sum())
            pulls += 1

        # If horizon exhausted during bootstrap, return immediately
        if horizon is not None and pulls >= horizon:
            state = _AHState(
                counts=counts,
                rewards=rewards,
                history=history,
                k_player=AdaHedge(K),
                sum_w=np.ones(K),
            )
            if record_stats:
                return state, np.zeros(0), np.zeros((0, K)), np.zeros((0, K))
            return state, None, None, None

        # Players & accumulators
        k_player = AdaHedge(K)
        sum_w = np.ones(K, dtype=float)

        # Optional recorders
        worst_curve_rec: List[float] = []
        prop_hist_rec: List[np.ndarray] = []
        mean_hist_rec: List[np.ndarray] = []
        chernoff = ChernoffStoppingCondition() if record_stats else None

        # Main loop
        while True:
            t = int(counts.sum())
            mu_hat = rewards / counts

            # stopping (confidence mode only)
            if horizon is None and self.stopping_condition.should_stop(
                counts, rewards, env, self.confidence
            ):
                break
            if horizon is not None and pulls >= horizon:
                break

            ft = self.f(t)

            # confidence intervals
            alpha = np.zeros(K)
            beta = np.zeros(K)
            for k in range(K):
                v = ft / counts[k]
                alpha[k] = find_lower_bound(mu_hat[k], v, env=env, sigma2=self.sigma2)
                beta[k] = find_upper_bound(mu_hat[k], v, env=env, sigma2=self.sigma2)

            # leader & mixture
            i_t = int(np.argmax(mu_hat))
            w_t = k_player.act()

            # adversary best response: set lambda_s on (i_t, j)
            w_i = w_t[i_t]
            best_val = float("inf")
            lambda_s = mu_hat.copy()
            for j in range(K):
                if j == i_t:
                    continue
                w_j = w_t[j]
                theta = (w_i * mu_hat[i_t] + w_j * mu_hat[j]) / (w_i + w_j)
                val = w_i * kl_div(
                    mu_hat[i_t], theta, env=env, sigma2=self.sigma2
                ) + w_j * kl_div(mu_hat[j], theta, env=env, sigma2=self.sigma2)
                if val < best_val:
                    best_val = val
                    lambda_s[i_t] = theta
                    lambda_s[j] = theta

            # compute U vector
            U = np.zeros(K)
            for k in range(K):
                du = kl_div(beta[k], lambda_s[k], env=env, sigma2=self.sigma2)
                dd = kl_div(alpha[k], lambda_s[k], env=env, sigma2=self.sigma2)
                bonus = ft / counts[k]
                U[k] = max(du, dd, bonus)

            # update AdaHedge player & choose arm by deficit
            k_player.incur(-U)
            sum_w += w_t
            arm = int(np.argmin(counts - sum_w))

            # pull
            r = env.sample(arm)
            counts[arm] += 1
            rewards[arm] += r
            history.append(counts / counts.sum())
            pulls += 1

            # optional statistics
            if record_stats and chernoff is not None:
                worst_curve_rec.append(chernoff.compute_statistic(counts, rewards, env))
                prop_hist_rec.append(counts / counts.sum())
                mean_hist_rec.append(rewards / np.maximum(counts, 1))

        state = _AHState(
            counts=counts,
            rewards=rewards,
            history=history,
            k_player=k_player,
            sum_w=sum_w,
        )

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
