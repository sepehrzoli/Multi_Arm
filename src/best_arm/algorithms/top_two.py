"""Top-two sampling with configurable leader and challenger selection."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import ChernoffStoppingCondition


@dataclass
class _TTState:
    """Local state for one execution of the sampling algorithm."""

    N: np.ndarray  # counts per arm
    sum_rewards: np.ndarray  # cumulative rewards per arm
    n: int  # total pulls
    history: List[np.ndarray]  # proportions snapshots


class TopTwoBAI(BaseAlgorithm):
    """Sample between a selected leader and challenger using a configured coin."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        tracker=None,
        stopping_condition=None,
        **kwargs,
    ):
        super().__init__(env=env, confidence=confidence, tracker=tracker)
        # required kwargs
        self.beta = kwargs["beta"]
        self.leader_type = kwargs["leader_type"]  # "EB" or Thompson-style
        self.challenger_type = kwargs["challenger_type"]  # "TC", "TCI", or "RS"
        self.stop_cond = stopping_condition

        K = env.num_arms
        # public state (kept for backward-compat after run())
        self.N = np.zeros(K, dtype=int)
        self.sum_rewards = np.zeros(K, dtype=float)
        self.n = 0
        self.history: List[np.ndarray] = []

        # distribution info
        if hasattr(env, "probs"):
            self.distribution = "bernoulli"
            self.sigma2 = None
        else:
            self.distribution = "gaussian"
            self.sigma2 = float(env.sigma) ** 2

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        state, _, _, _ = self._run_core(horizon=None, record_stats=False)
        # expose state like before
        self.N, self.sum_rewards, self.n, self.history = (
            state.N,
            state.sum_rewards,
            state.n,
            state.history,
        )
        best = int(np.argmax(self.sum_rewards / self.N))
        return best, self.N, self.sum_rewards, self.history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the algorithm for a *fixed* number of pulls (`horizon`).

        Returns
        -------
        best_arm      : int
        worst_curve   : np.ndarray, shape=(T',)
        prop_history  : np.ndarray, shape=(T', K)
        mean_history  : np.ndarray, shape=(T', K)
        where T' = max(horizon - pulls_during_bootstrap, 0)
        """
        state, worst_curve, prop_history, mean_history = self._run_core(
            horizon=horizon, record_stats=True
        )
        # keep public state in sync
        self.N, self.sum_rewards, self.n, self.history = (
            state.N,
            state.sum_rewards,
            state.n,
            state.history,
        )
        best_arm = int(np.argmax(self.sum_rewards / self.N))
        return best_arm, worst_curve, prop_history, mean_history

    # Core engine.

    def _run_core(
        self,
        *,
        horizon: Optional[int],
        record_stats: bool,
    ):
        K, env = self.env.num_arms, self.env

        # fresh local state
        state = _TTState(
            N=np.zeros(K, dtype=int),
            sum_rewards=np.zeros(K, dtype=float),
            n=0,
            history=[],
        )

        # 1) Bootstrap: one pull per arm (respect horizon if provided)
        pulls = 0
        for a in range(K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(a)
            state.N[a] += 1
            state.sum_rewards[a] += r
            state.n += 1
            pulls += 1
        # original code recorded a single snapshot after bootstrap
        if state.n > 0:
            state.history.append(state.N / state.n)

        # early exit if horizon exhausted during bootstrap
        if horizon is not None and pulls >= horizon:
            if record_stats:
                return state, np.zeros(0), np.zeros((0, K)), np.zeros((0, K))
            return state, None, None, None

        # Optional recorders (for run_fixed)
        worst_curve_rec: List[float] = []
        prop_hist_rec: List[np.ndarray] = []
        mean_hist_rec: List[np.ndarray] = []
        chernoff = ChernoffStoppingCondition() if record_stats else None

        # 2) Main loop
        while True:
            # stopping (only in fixed-confidence mode)
            if horizon is None:
                if self.stop_cond.should_stop(
                    state.N, state.sum_rewards, env, self.confidence
                ):
                    break
            else:
                if pulls >= horizon:
                    break

            # select arm via Top-Two logic
            B_n = self._choose_leader(state)
            arm = (
                B_n
                if np.random.rand() < self.beta
                else self._choose_challenger(state, B_n)
            )

            # pull and update
            r = env.sample(arm)
            state.N[arm] += 1
            state.sum_rewards[arm] += r
            state.n += 1
            pulls += 1
            state.history.append(state.N / state.n)

            # optional statistics (for plotting/analysis)
            if record_stats and chernoff is not None:
                worst_curve_rec.append(
                    chernoff.compute_statistic(state.N, state.sum_rewards, env)
                )
                prop_hist_rec.append(state.N / state.n)
                mean_hist_rec.append(state.sum_rewards / np.maximum(state.N, 1))

        # finalize outputs
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

    def _empirical_means(self, state: _TTState) -> np.ndarray:
        return state.sum_rewards / np.maximum(state.N, 1)

    def _transport_cost(self, state: _TTState, i: int, j: int) -> float:
        N_i, N_j = state.N[i], state.N[j]
        mu_i = state.sum_rewards[i] / N_i
        mu_j = state.sum_rewards[j] / N_j
        x_star = (N_i * mu_i + N_j * mu_j) / (N_i + N_j)
        if self.distribution == "bernoulli":
            from ..utils import _kl_bernoulli as kl

            return N_i * kl(mu_i, x_star) + N_j * kl(mu_j, x_star)
        else:
            # closed-form Gaussian transport cost equals KL pair-wise GLR
            return (N_i * N_j) / (2 * self.sigma2 * (N_i + N_j)) * (mu_i - mu_j) ** 2

    def _choose_leader(self, state: _TTState) -> int:
        mus = self._empirical_means(state)
        if self.leader_type == "EB":
            return int(np.argmax(mus))

        # Thompson-style leader
        thetas: List[float] = []
        for i in range(self.env.num_arms):
            N_i = state.N[i]
            mu_i = state.sum_rewards[i] / N_i
            if self.distribution == "bernoulli":
                alpha = 1 + state.sum_rewards[i]
                beta_param = 1 + N_i - state.sum_rewards[i]
                thetas.append(np.random.beta(alpha, beta_param))
            else:
                thetas.append(np.random.normal(mu_i, math.sqrt(self.sigma2 / N_i)))
        return int(np.argmax(thetas))

    def _choose_challenger(self, state: _TTState, B_n: int) -> int:
        K = self.env.num_arms
        if self.challenger_type == "TC":
            costs = [
                (self._transport_cost(state, B_n, j) if j != B_n else math.inf)
                for j in range(K)
            ]
            min_cost = min(costs)
            tol = 1e-6
            candidates = [j for j, c in enumerate(costs) if c <= min_cost + tol]
            return int(np.random.choice(candidates))

        if self.challenger_type == "TCI":
            costs = [
                (
                    self._transport_cost(state, B_n, j) + math.log(state.N[j])
                    if j != B_n
                    else math.inf
                )
                for j in range(K)
            ]
            min_cost = min(costs)
            tol = 1e-6
            candidates = [j for j, c in enumerate(costs) if c <= min_cost + tol]
            return int(np.random.choice(candidates))

        # "RS": resample until challenger != leader
        while True:
            thetas = []
            for i in range(K):
                N_i = state.N[i]
                mu_i = state.sum_rewards[i] / N_i
                if self.distribution == "bernoulli":
                    alpha = 1 + state.sum_rewards[i]
                    beta_param = 1 + N_i - state.sum_rewards[i]
                    thetas.append(np.random.beta(alpha, beta_param))
                else:
                    thetas.append(np.random.normal(mu_i, math.sqrt(self.sigma2 / N_i)))
            c = int(np.argmax(thetas))
            if c != B_n:
                return c
