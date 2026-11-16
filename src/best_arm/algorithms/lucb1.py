"""LUCB1 sampling with configurable stopping."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition


@dataclass
class _LUCBState:
    """Local state for one execution of the sampling algorithm."""

    counts: np.ndarray
    rewards: np.ndarray
    history: List[np.ndarray]


class LUCB1(BaseAlgorithm):
    """Sample the empirical leader and its strongest confidence-bound challenger."""

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        state, _, _, _ = self._run_core(horizon=None, record_stats=False)
        best = int(np.argmax(state.rewards / state.counts))
        return best, state.counts, state.rewards, state.history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute LUCB1 for exactly `horizon` pulls.

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
        _LUCBState, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        K, env, δ = self.env.num_arms, self.env, self.confidence

        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        # 1) Bootstrap
        pulls = 0
        for a in range(K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(a)
            counts[a] += 1
            rewards[a] += r
            history.append(counts / counts.sum())
            pulls += 1

        if horizon is not None and pulls >= horizon:
            state = _LUCBState(counts=counts, rewards=rewards, history=history)
            if record_stats:
                return state, np.zeros(0), np.zeros((0, K)), np.zeros((0, K))
            return state, None, None, None

        # optional recorders
        worst_curve_rec: List[float] = []
        prop_hist_rec: List[np.ndarray] = []
        mean_hist_rec: List[np.ndarray] = []
        chernoff = ChernoffStoppingCondition() if record_stats else None

        k1 = 5.0 / 4.0

        # 2) Main loop
        while True:
            if horizon is None:
                if self.stopping_condition.should_stop(counts, rewards, env, δ):
                    break
            else:
                if pulls >= horizon:
                    break

            p_hat = rewards / counts
            t = int(counts.sum())

            # confidence radii
            betas = np.sqrt((1.0 / (2.0 * counts)) * np.log(k1 * K * (t**4) / δ))

            # leader & most dangerous challenger
            h_star = int(np.argmax(p_hat))
            scores = p_hat + betas
            scores[h_star] = -np.inf
            l_star = int(np.argmax(scores))

            # pull both (h_star, l_star); record after each pull
            for arm in (h_star, l_star):
                if horizon is not None and pulls >= horizon:
                    break
                r = env.sample(arm)
                counts[arm] += 1
                rewards[arm] += r
                history.append(counts / counts.sum())
                pulls += 1

                if record_stats and chernoff is not None:
                    worst_curve_rec.append(
                        chernoff.compute_statistic(counts, rewards, env)
                    )
                    prop_hist_rec.append(counts / counts.sum())
                    mean_hist_rec.append(rewards / np.maximum(counts, 1))

        state = _LUCBState(counts=counts, rewards=rewards, history=history)

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
