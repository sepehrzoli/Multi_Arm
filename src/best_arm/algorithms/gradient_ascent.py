"""Gradient-ascent sampling with cumulative allocation tracking."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..environment import BaseEnvironment
from ..model import BaseAlgorithm
from ..stopping_conditions import BaseStoppingCondition, ChernoffStoppingCondition
from ..utils import _kl_bernoulli, _kl_gaussian


@dataclass
class _GAState:
    """Local state for one execution of the sampling algorithm."""

    counts: np.ndarray
    rewards: np.ndarray
    history: List[np.ndarray]
    pi: np.ndarray
    G: np.ndarray
    tilde_w: np.ndarray
    w_prime: np.ndarray
    cum_w: np.ndarray
    t: int = 0  # total pulls so far


class GradientAscent(BaseAlgorithm):
    """
    Gradient-Ascent allocation for best-arm identification.

    `run` and `run_fixed` share the same sampling engine.
    Selection is always via the deficit rule: argmax(cum_w - counts).
    """

    def __init__(
        self,
        env: BaseEnvironment,
        confidence: float,
        stopping_condition: BaseStoppingCondition,
        M: float = float("inf"),
    ):
        super().__init__(env=env, confidence=confidence)
        self.stopping_condition = stopping_condition
        self.M = M
        self.K = env.num_arms

        # public state after run()
        self.counts = np.zeros(self.K, dtype=int)
        self.rewards = np.zeros(self.K, dtype=float)
        self.history: List[np.ndarray] = []

        # pick the right KL
        name = env.__class__.__name__.lower()
        if name.startswith("normal"):
            sigma2 = float(getattr(env, "sigma", 1.0)) ** 2
            self.kl = lambda p, q: _kl_gaussian(p, q, sigma2)
        else:
            self.kl = _kl_bernoulli

    # Public API.

    def run(self) -> Tuple[int, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return the selected arm, pull counts, reward sums, and allocation history."""
        state, _, _, _ = self._run_core(
            horizon=None,  # stop by condition
            record_stats=False,  # no plotting payload
        )
        self.counts, self.rewards, self.history = (
            state.counts,
            state.rewards,
            state.history,
        )
        i_hat = int(np.argmax(self.rewards / self.counts))
        return i_hat, self.counts, self.rewards, self.history

    def run_fixed(self, horizon: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Gradient-Ascent sampling for exactly `horizon` pulls.

        Returns
        -------
        best_arm      : int
        worst_curve   : np.ndarray,  shape=(T',)
        prop_history  : np.ndarray,  shape=(T', K)
        mean_history  : np.ndarray,  shape=(T', K)
        where T' = max(horizon - pulls_during_bootstrap, 0)
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
        _GAState, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Shared driver for both modes.

        Parameters
        ----------
        horizon      : None -> stop via stopping_condition; int -> fixed pulls
        record_stats : if True, return (worst_curve, prop_history, mean_history)
        """
        K = self.K
        env = self.env

        # fresh local state (never mutates self.* mid-run)
        state = self._new_state(K)

        # 1) Bootstrap: one pull per arm (respect horizon)
        pulls = self._bootstrap(state, env, horizon)

        # Optional recorders
        worst_curve_rec: List[float] = []
        prop_hist_rec: List[np.ndarray] = []
        mean_hist_rec: List[np.ndarray] = []
        chernoff = ChernoffStoppingCondition() if record_stats else None

        # 2) Main loop
        while True:
            if horizon is not None and pulls >= horizon:
                break

            # one GA step (deficit rule)
            self._step(state, env)
            pulls += 1

            # stop only in confidence mode
            if horizon is None and self.stopping_condition.should_stop(
                state.counts, state.rewards, env, self.confidence
            ):
                break

            # optional statistics
            if record_stats and chernoff is not None:
                worst_curve_rec.append(
                    chernoff.compute_statistic(state.counts, state.rewards, env)
                )
                prop_hist_rec.append(state.counts / state.counts.sum())
                mean_hist_rec.append(state.rewards / np.maximum(state.counts, 1))

        # finalize payloads
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

    def _new_state(self, K: int) -> _GAState:
        counts = np.zeros(K, dtype=int)
        rewards = np.zeros(K, dtype=float)
        history: List[np.ndarray] = []

        pi = np.ones(K) / K
        G = np.zeros(K)
        tilde_w = pi.copy()
        w_prime = pi.copy()
        cum_w = np.zeros(K)

        return _GAState(
            counts=counts,
            rewards=rewards,
            history=history,
            pi=pi,
            G=G,
            tilde_w=tilde_w,
            w_prime=w_prime,
            cum_w=cum_w,
            t=0,
        )

    def _bootstrap(
        self, state: _GAState, env: BaseEnvironment, horizon: Optional[int]
    ) -> int:
        """Pull each arm once; respect finite horizon."""
        pulls = 0
        for a in range(self.K):
            if horizon is not None and pulls >= horizon:
                break
            r = env.sample(a)
            state.t += 1
            state.counts[a] += 1
            state.rewards[a] += r
            state.history.append(state.counts / state.counts.sum())
            state.cum_w += state.w_prime
            pulls += 1
        return pulls

    def _compute_g(self, mu_hat: np.ndarray, tilde_w: np.ndarray) -> np.ndarray:
        """Compute oracle gradient g (non-zeros at i* and best challenger)."""
        K = self.K
        i_star = int(np.argmax(mu_hat))
        wi = float(tilde_w[i_star])

        best_b = None
        best_val = float("inf")
        best_mix = 0.0

        for b in range(K):
            if b == i_star:
                continue
            wb = float(tilde_w[b])
            if wi + wb == 0.0:
                continue
            mix = (wi * mu_hat[i_star] + wb * mu_hat[b]) / (wi + wb)
            val = wi * self.kl(mu_hat[i_star], mix) + wb * self.kl(mu_hat[b], mix)
            if val < best_val:
                best_val, best_b, best_mix = val, b, mix

        g = np.zeros(K)
        if best_b is not None:
            g[i_star] = self.kl(mu_hat[i_star], best_mix)
            g[best_b] = self.kl(mu_hat[best_b], best_mix)
        return g

    def _update_weights(self, state: _GAState) -> None:
        """Update tilde_w, w_prime, and cum_w with AdaGrad-like step & mixing."""
        eta = 1.0 / math.sqrt(state.t + 1)  # matches original
        log_w = eta * state.G
        log_w -= np.max(log_w)
        exp_w = np.exp(log_w)
        state.tilde_w = exp_w / exp_w.sum()

        gamma = 1.0 / (4.0 * math.sqrt(state.t if state.t > 0 else 1.0))
        state.w_prime = (1 - gamma) * state.tilde_w + gamma * state.pi
        state.cum_w += state.w_prime

    def _choose_next_arm(self, state: _GAState) -> int:
        """Deficit rule (default for both modes)."""
        deficits = state.cum_w - state.counts
        return int(np.argmax(deficits))

    def _step(self, state: _GAState, env: BaseEnvironment) -> None:
        """One allocation + sampling step using the deficit rule."""
        mu_hat = state.rewards / np.maximum(state.counts, 1)  # safe during bootstrap
        g = self._compute_g(mu_hat, state.tilde_w)
        state.G += g

        self._update_weights(state)
        a_next = self._choose_next_arm(state)

        # pull and update
        r = env.sample(a_next)
        state.t += 1
        state.counts[a_next] += 1
        state.rewards[a_next] += r
        state.history.append(state.counts / state.counts.sum())
