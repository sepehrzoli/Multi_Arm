from __future__ import annotations

import math

from bandit import Bandit
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import lower_bound, fixed_confidence_lower_bound, _compute_w_star


def run_experiment(
    env_path: str,
    algo_name: str,
    algo_args: dict,
    tracker_name: str | None,
    stopping_name: str | None,
    runs: int = 10,
    delta: float = 0.1,
    expected_best: int = 0,
) -> dict:
    b = Bandit.from_saved_env(
        path=env_path,
        algo_name=algo_name,
        algo_args=algo_args,
        tracker_name=tracker_name,
        stopping_name=stopping_name,
    )

    env = b.env
    if hasattr(env, "probs"):
        mu = np.array(env.probs)
    elif hasattr(env, "mu"):
        mu = np.array(env.mu)
    else:
        raise ValueError("Unknown environment structure for extracting true means")

    # true information-theoretic lower bounds
    T_star = lower_bound(mu, env=env)
    full_lb = fixed_confidence_lower_bound(mu, env=env, delta=delta)

    print(f"[Lower bound] T*(mu) = {T_star:.6f}")
    print(f"[Fixed-confidence bound with delta={delta}] T*(mu) * kl(delta,1-delta) = {full_lb:.6f}")

    cumulative_hist = np.zeros_like(mu, dtype=float)
    pulls = 0
    tsum = 0.0
    wrong = 0
    max_pulls = 0

    last_history = None  # will keep the history from the last run for plotting
    for i in range(runs):
        start = time.perf_counter()
        best, counts, rewards, history = b.run()
        elapsed = time.perf_counter() - start

        if best != expected_best:
            wrong += 1
        total_pulls = counts.sum()
        pulls += total_pulls
        max_pulls = max(max_pulls, total_pulls)
        tsum += elapsed
        last_hist = history[-1]
        cumulative_hist += last_hist
        last_history = history  # overwrite so after loop it is the final run's trajectory

        print(
            f"Run {i+1:2d} | Algo: {algo_name} | Best: {best} | Pulls: {total_pulls} | "
            f"Hist(last)={last_hist} | Avg pulls so far: {pulls/(i+1):.2f} | "
            f"Avg time: {tsum/(i+1):.4f} | Wrong: {wrong}"
        )

    avg_hist = cumulative_hist / runs
    summary = {
        "T_star": T_star,
        "fixed_conf_lb": full_lb,
        "avg_pulls": pulls / runs,
        "max_pulls": max_pulls,
        "avg_time": tsum / runs,
        "wrong": wrong,
        "avg_last_hist": avg_hist,
    }

    print("Average final empirical proportions:", avg_hist)

    # --- plotting empirical proportions over time for last run + true w* ---
    if last_history is not None:
        K = len(mu)
        # time axis: number of updates (assuming each history entry corresponds to one sample after initialization)
        times = np.arange(1, len(last_history) + 1)

        fig, ax = plt.subplots()
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key().get("color", None)

        # plot empirical proportions
        for k in range(K):
            series = [h[k] for h in last_history]
            color = None
            if colors:
                color = colors[k % len(colors)]
            ax.plot(times, series, label=f"arm {k}", linewidth=1.5, color=color)

        # compute true w*
        w_star = _compute_w_star(mu, env=env)

        # overlay w* as horizontal dashed lines matching colors
        for k in range(K):
            color = None
            if colors:
                color = colors[k % len(colors)]
            ax.hlines(
                w_star[k],
                xmin=times[0],
                xmax=times[-1],
                linestyles="--",
                label=f"w*_{k}",
                linewidth=1.0,
                color=color,
            )

        ax.set_xlabel("Iteration (total pulls)")
        ax.set_ylabel("Empirical proportion")
        ax.set_title(f"Empirical proportions over time ({algo_name})")
        ax.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return summary

if __name__ == "__main__":  # Stopping files + not exp_gap
    summary = run_experiment(
        env_path="env_0002.json",
        algo_name="top_two",
        algo_args={"confidence": 1e-10,
                   "leader_type": "TS",
                   "challenger_type": "TC",
                   "beta": 0.5},
        tracker_name=None,
        stopping_name="chernoff",
        runs=1,
        delta=0.1,
        expected_best=0,  # adjust if you know the true best arm index
    )
    summary = run_experiment(
        env_path="env_0002.json",
        algo_name="top_two",
        algo_args={"confidence": 1e-10,
                   "leader_type": "TS",
                   "challenger_type": "TC",
                   "beta": 0.5},
        tracker_name=None,
        stopping_name="chernoff",
        runs=1,
        delta=0.1,
        expected_best=0,  # adjust if you know the true best arm index
    )
    summary = run_experiment(
        env_path="env_0002.json",
        algo_name="top_two",
        algo_args={"confidence": 1e-10,
                   "leader_type": "TS",
                   "challenger_type": "TC",
                   "beta" : 0.5},
        tracker_name=None,
        stopping_name="chernoff",
        runs=1,
        delta=0.1,
        expected_best=0,  # adjust if you know the true best arm index
    )
    print(summary)
    summary = run_experiment(
        env_path="env_0002.json",
        algo_name="track_and_stop",
        algo_args={"confidence": 1e-30},
        tracker_name="g_tracking",
        stopping_name="chernoff",
        runs=1,
        delta=0.1,
        expected_best=0,  # adjust if you know the true best arm index
    )
    print(summary)
    # 1) Parameters
    horizon = 50000
    n_runs = 40
    deltas = [10 ** (-k) for k in range(1, 21)]  # 0.1, 1e-2, …, 1e-20
    t = np.arange(1, horizon + 1)
    for delta in deltas:
        beta = np.log((np.log(t) + 1) / delta)
    # 2) Build two Bandit instances (d_tracking and real_oracle_tracking)
    b_d = Bandit.from_saved_env(
        path="env_0002.json",
        algo_name="track_and_stop",
        algo_args={"confidence": 0.1},  # placeholder, not used in run_fixed
        tracker_name="d_tracking",
        stopping_name="chernoff"
    )
    b_o = Bandit.from_saved_env(
        path="env_0002.json",
        algo_name="track_and_stop",
        algo_args={"confidence": 0.1},
        tracker_name="real_oracle_tracking",
        stopping_name="chernoff"
    )

    # 3) Run fixed‑horizon to collect all “worst” curves for both trackers
    all_worsts_d = np.zeros((n_runs, horizon))
    all_worsts_o = np.zeros((n_runs, horizon))
    for i in range(n_runs):
        start = time.perf_counter()
        all_worsts_o[i] = b_o.run_fixed(horizon)
        all_worsts_d[i] = b_d.run_fixed(horizon)
        elapsed2 = time.perf_counter() - start
        print("Run ", i+1, " completed, it took", elapsed2, " seconds.")

    # 4) Compute average trajectories
    avg_worst_d = np.nanmean(all_worsts_d, axis=0)
    avg_worst_o = np.nanmean(all_worsts_o, axis=0)

    # 5) Find stopping time for each δ and each tracker
    t_stops_d = []
    t_stops_o = []
    for delta in deltas:
        beta = np.log((np.log(t) + 1) / delta)
        # D-Tracking
        mask_d = avg_worst_d > beta
        hits_d = np.where(mask_d)[0]
        t_stops_d.append((hits_d[0] + 1) if hits_d.size > 0 else np.nan)
        # Real-Oracle-Tracking
        mask_o = avg_worst_o > beta
        hits_o = np.where(mask_o)[0]
        t_stops_o.append((hits_o[0] + 1) if hits_o.size > 0 else np.nan)

    # 6) Plot both curves in one figure
    plt.figure(figsize=(10, 6))
    plt.plot(t, avg_worst_d, color='blue', label="D‑Tracking")
    plt.plot(t, avg_worst_o, color='red', label="Real‑Oracle‑Tracking")

    # Overlay stopping times for each δ
    for t_stop in t_stops_d:
        if not np.isnan(t_stop):
            plt.axvline(t_stop, color='blue', linestyle="--", alpha=0.3)
    for t_stop in t_stops_o:
        if not np.isnan(t_stop):
            plt.axvline(t_stop, color='red', linestyle="--", alpha=0.3)

    plt.xlabel("Iteration")
    plt.ylabel("Average Chernoff statistic (worst)")
    plt.title(
        "Average Chernoff worst vs. time for two trackers\n"
        "D‑Tracking (blue) and Real‑Oracle‑Tracking (red)\n"
        "Vertical lines = stopping t* for δ=10⁻¹…10⁻²⁰"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
