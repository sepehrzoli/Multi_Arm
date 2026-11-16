"""LilUCB sampling in theoretical and heuristic modes."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import (
    BaseStoppingCondition,
    ChernoffStoppingCondition,
    LilUCBStoppingCondition,
)


@dataclass
class _LilState:
    """Local state for one execution of the sampling algorithm."""

    counts: np.ndarray
    rewards: np.ndarray
    history: List[np.ndarray]


class LilUCB(BaseAlgorithm):
    """Select arms using law-of-the-iterated-logarithm confidence bonuses."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: Optional[BaseStoppingCondition] = None,
        mode: str = "theory",  # "theory" or "heuristic"
        epsilon: Optional[float] = None,
        beta: Optional[float] = None,
        lambda_param: Optional[float] = None,
    ):
        super().__init__(env=env, confidence=confidence)
        self.v = confidence

        K = env.num_arms

        # derive defaults from mode
        if mode == "theory":
            default_epsilon = 0.01
            default_beta = 1.0
            default_lambda = ((2.0 + default_beta) / default_beta) ** 2
        elif mode == "heuristic":
            default_epsilon = 0
            default_beta = 0.5
            default_lambda = 1.0 + 10.0 / K
        else:
            raise ValueError(f"Unknown mode: {mode}; use 'theory' or 'heuristic'")

        # allow explicit overrides
        self.epsilon = epsilon if epsilon is not None else default_epsilon
        self.beta = beta if beta is not None else default_beta
        self.lambda_param = lambda_param if lambda_param is not None else default_lambda

        # stopping condition
        if stopping_condition is None:
            self.stopping_condition = LilUCBStoppingCondition(
                lambda_param=self.lambda_param
            )
        elif isinstance(stopping_condition, LilUCBStoppingCondition):
            stopping_condition.lambda_param = self.lambda_param
            self.stopping_condition = stopping_condition
        else:
            self.stopping_condition = stopping_condition

        # delta (only meaningful when epsilon > 0)
        if self.epsilon > 0:
            c_e = (
                ((2.0 + self.epsilon) / self.epsilon)
                * (1.0 / np.log(1.0 + self.epsilon))
            ) ** (1.0 + self.epsilon)
            self.delta = ((np.sqrt(1.0 + self.v) - 1.0) ** 2) / (4.0 * c_e)
        else:
            self.delta = self.v / 5

        # variance proxy
        self.sigma2 = (
            float(getattr(env, "sigma", 0.5) ** 2) if hasattr(env, "sigma") else 0.25
        )

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        state, _, _, _ = self._run_core(horizon=None, record_stats=False)
        best = int(np.argmax(state.rewards / state.counts))
        return best, state.counts, state.rewards, state.history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run LilUCB for exactly `horizon` pulls.

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
        _LilState, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        K, env = self.env.num_arms, self.env

        # fresh state
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # 1) Bootstrap (respect horizon)
        pulls = 0
        for i in range(K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(i)
            counts[i] += 1
            rewards[i] += r
            history.append(counts / counts.sum())
            pulls += 1

        # If horizon exhausted during bootstrap, return early
        if horizon is not None and pulls >= horizon:
            state = _LilState(counts=counts, rewards=rewards, history=history)
            if record_stats:
                return state, np.zeros(0), np.zeros((0, K)), np.zeros((0, K))
            return state, None, None, None

        # optional recorders
        worst_curve_rec: List[float] = []
        prop_hist_rec: List[np.ndarray] = []
        mean_hist_rec: List[np.ndarray] = []
        chernoff = ChernoffStoppingCondition() if record_stats else None

        # 2) Main loop
        while True:
            if horizon is None:
                if self.stopping_condition.should_stop(counts, rewards, env, self.v):
                    break
            else:
                if pulls >= horizon:
                    break

            # compute scores
            p_hat = rewards / counts
            scores = np.empty(K, dtype=float)
            for i in range(K):
                Ti = counts[i]
                if self.epsilon > 0:
                    inner = np.log((1.0 + self.epsilon) * Ti)
                    term = np.log(inner / self.delta)
                    bonus = (
                        (1.0 + self.beta)
                        * (1.0 + np.sqrt(self.epsilon))
                        * np.sqrt(2.0 * self.sigma2 * (1.0 + self.epsilon) * term / Ti)
                    )
                else:
                    inner = np.log(1.0 + Ti)
                    term = np.log(inner / max(self.delta, 1e-12))
                    bonus = (1.0 + self.beta) * np.sqrt(2.0 * self.sigma2 * term / Ti)
                scores[i] = p_hat[i] + bonus

            I_t = int(np.argmax(scores))

            # pull and update
            r = env.sample(I_t)
            counts[I_t] += 1
            rewards[I_t] += r
            history.append(counts / counts.sum())
            pulls += 1

            # optional stats
            if record_stats and chernoff is not None:
                worst_curve_rec.append(chernoff.compute_statistic(counts, rewards, env))
                prop_hist_rec.append(counts / counts.sum())
                mean_hist_rec.append(rewards / np.maximum(counts, 1))

        state = _LilState(counts=counts, rewards=rewards, history=history)

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
