"""Run, summarize, merge, save, and plot bandit experiments."""

from __future__ import annotations

import json
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .bandit import Bandit
from .utils import _compute_w_star, fixed_confidence_lower_bound, lower_bound


def run_experiment_multi(
    bandit_factories: Sequence[Callable[[], Bandit]],
    labels: Sequence[str],
    runs: int = 10,
    show_plot: bool = True,
    verbose: bool = True,
    *,
    merge_from_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run each algorithm produced by `bandit_factories` for `runs` repetitions
    using `.run()`. Summarize, plot a compact box+points figure, and save a JSON.

    If `merge_from_json` is provided and points to an existing JSON saved by this
    function, results for algorithms whose labels match will be **concatenated**
    with the previous runs (pulls & times), and statistics will be recomputed
    over the combined runs. Algorithms present only in the JSON are kept as-is.

    The saved JSON file name is built from the *current* `labels` and written to
    `results/<joined_labels>.json`.
    """
    if len(bandit_factories) != len(labels):
        raise ValueError("`bandit_factories` and `labels` must have the same length")
    if runs < 1:
        raise ValueError("`runs` must be >= 1")

    # Inspect env / delta from the first factory.
    probe = bandit_factories[0]()
    env = probe.env
    if hasattr(env, "probs"):
        mu = np.array(env.probs, dtype=float)
    elif hasattr(env, "mu"):
        mu = np.array(env.mu, dtype=float)
    else:
        raise ValueError("Unknown environment structure for extracting true means")

    delta = float(probe.algo_args.get("confidence", probe.algo_args.get("delta", 0.1)))
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1); got {delta!r}")

    K = env.num_arms
    true_best = int(np.argmax(mu))
    T_star = float(lower_bound(mu, env=env))
    full_lb = float(fixed_confidence_lower_bound(mu, env=env, delta=delta))
    del probe

    # Storage for *new* runs.
    new_alg: Dict[str, Any] = {}
    pulls_data: list[list[int]] = []

    for make, label in zip(bandit_factories, labels):
        pulls_total = 0
        time_total = 0.0
        wrong = 0
        max_pulls = 0
        cumulative_hist = np.zeros(K, dtype=float)
        pulls_list: list[int] = []
        times_list: list[float] = []

        for i in range(runs):
            b = make()
            start = time.perf_counter()
            best, counts, rewards, history = b.run()
            elapsed = time.perf_counter() - start

            pulls = int(counts.sum())
            pulls_list.append(pulls)
            times_list.append(elapsed)

            pulls_total += pulls
            max_pulls = max(max_pulls, pulls)
            time_total += elapsed
            cumulative_hist += history[-1]
            if best != true_best:
                wrong += 1

            if verbose:
                print(
                    f"[{label}] run {i + 1:02d} | best={best} | pulls={pulls} | time={elapsed:.4f}s"
                )

        avg_hist = cumulative_hist / max(runs, 1)
        var_pulls = (
            float(np.var(np.asarray(pulls_list, float), ddof=1))
            if len(pulls_list) > 1
            else 0.0
        )

        new_alg[label] = {
            "avg_pulls": pulls_total / runs,
            "max_pulls": max_pulls,
            "avg_time": time_total / runs,
            "wrong": wrong,
            "avg_last_hist": avg_hist,
            "pulls_per_run": pulls_list,
            "times_per_run": times_list,
            "var_pulls": var_pulls,
        }
        pulls_data.append(pulls_list)

    # Optional MERGE from an existing JSON.
    merged_alg: Dict[str, Any] = {}

    # Load old results if requested & present
    old: Dict[str, Any] = {}
    if merge_from_json:
        p = Path(merge_from_json)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    old = json.load(f)
            except Exception as e:
                if verbose:
                    print(f"[merge] Could not load JSON ({p}): {e}")

    old_alg = old.get("algorithms", {}) if isinstance(old, dict) else {}

    # Merge per label (combine when both exist; keep JSON-only algs)
    for label, new_stats in new_alg.items():
        if label in old_alg:
            prev = old_alg[label]
            prev_pulls = list(prev.get("pulls_per_run", []))
            prev_times = list(prev.get("times_per_run", []))
            prev_wrong = int(prev.get("wrong", 0))
            prev_max = int(prev.get("max_pulls", 0))
            prev_hist = np.asarray(prev.get("avg_last_hist", np.zeros(K)), dtype=float)

            # sanity check K dimension on hist
            if prev_hist.shape != (K,):
                raise ValueError(
                    f"[merge] avg_last_hist for '{label}' has wrong shape: {prev_hist.shape}, expected {(K,)}"
                )

            # concatenate runs
            all_pulls = prev_pulls + new_stats["pulls_per_run"]
            all_times = prev_times + new_stats["times_per_run"]

            n_prev = len(prev_pulls)
            n_new = len(new_stats["pulls_per_run"])
            n_all = n_prev + n_new

            # weighted average of avg_last_hist
            new_hist = np.asarray(new_stats["avg_last_hist"], dtype=float)
            comb_hist = (prev_hist * n_prev + new_hist * n_new) / max(n_all, 1)

            merged_alg[label] = {
                "avg_pulls": float(np.mean(all_pulls)) if n_all else 0.0,
                "max_pulls": int(max(prev_max, new_stats["max_pulls"])) if n_all else 0,
                "avg_time": float(np.mean(all_times)) if n_all else 0.0,
                "wrong": int(prev_wrong + new_stats["wrong"]),
                "avg_last_hist": comb_hist,
                "pulls_per_run": all_pulls,
                "times_per_run": all_times,
                "var_pulls": float(np.var(np.asarray(all_pulls, float), ddof=1))
                if n_all > 1
                else 0.0,
            }
        else:
            merged_alg[label] = new_stats

    # Keep the algorithms that exist only in the JSON (not run now)
    for label, prev in old_alg.items():
        if label not in merged_alg:
            # ensure numpy arrays if any are converted back later
            prev_hist = np.asarray(prev.get("avg_last_hist", np.zeros(K)), dtype=float)
            merged_alg[label] = {
                **prev,
                "avg_last_hist": prev_hist,
            }

    # Pretty boxplot (robust & readable) — uses UNION of algorithms.
    if show_plot and merged_alg:
        # 1) Build label order: inputs first (if present), then JSON-only ones
        plot_labels: list[str] = []
        seen: set[str] = set()
        for lbl in labels:
            if lbl in merged_alg and lbl not in seen:
                plot_labels.append(lbl)
                seen.add(lbl)
        for lbl in merged_alg.keys():  # preserves JSON insertion order
            if lbl not in seen:
                plot_labels.append(lbl)
                seen.add(lbl)

        pulls_data_plot = [merged_alg[lbl]["pulls_per_run"] for lbl in plot_labels]
        if not any(len(x) for x in pulls_data_plot):
            print("[plot] nothing to plot (no runs in merged data).")
        else:
            fig, ax = plt.subplots(figsize=(9.5, 5.8))

            colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            if not colours:
                colours = [
                    "#4c78a8",
                    "#f58518",
                    "#54a24b",
                    "#e45756",
                    "#72b7b2",
                    "#ff9da6",
                ]

            boxprops = dict(linewidth=1.2, edgecolor="black")
            whiskerprops = dict(linewidth=1.1, color="black")
            capprops = dict(linewidth=1.1, color="black")
            medianprops = dict(linewidth=1.6, color="black")

            bplot = ax.boxplot(
                pulls_data_plot,
                notch=True,
                patch_artist=True,
                showfliers=False,  # we’ll annotate clipped ones
                widths=0.55,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                medianprops=medianprops,
            )

            for i, box in enumerate(bplot["boxes"]):
                box.set_facecolor(colours[i % len(colours)])
                box.set_alpha(0.35)

            # robust y-limit at P99 across ALL merged points
            all_pts = np.concatenate(
                [np.asarray(x, float) for x in pulls_data_plot if len(x)]
            )
            y_cap = np.quantile(all_pts, 0.99)
            y_cap = max(y_cap, full_lb * 1.05, T_star * 1.05)

            outlier_marker_y = y_cap * 0.985
            clipped_total = 0

            for i, pts in enumerate(pulls_data_plot, start=1):
                c = colours[(i - 1) % len(colours)]
                pts = np.asarray(pts, float)
                x = i + 0.16 * (2 * np.random.rand(len(pts)) - 1)  # jitter
                y_draw = np.minimum(pts, y_cap)
                ax.scatter(
                    x,
                    y_draw,
                    s=28,
                    color=c,
                    edgecolor="black",
                    linewidths=0.25,
                    alpha=0.85,
                    zorder=3,
                )
                if len(pts):
                    ax.scatter(
                        i,
                        float(np.mean(pts)),
                        marker="*",
                        s=120,
                        color="black",
                        zorder=4,
                    )

                n_clip = int(np.sum(pts > y_cap))
                clipped_total += n_clip
                if n_clip > 0:
                    ax.plot(
                        i, outlier_marker_y, marker="^", color=c, markersize=7, zorder=5
                    )
                    ax.text(
                        i,
                        outlier_marker_y,
                        f"  ↑{n_clip}",
                        va="center",
                        ha="left",
                        fontsize=8.5,
                        color=c,
                        weight="bold",
                    )

            lb_line = ax.axhline(T_star, color="#1f77b4", lw=2.0, label="lower bd")
            pr_line = ax.axhline(full_lb, color="#ff7f0e", lw=2.0, label="practical")

            ax.set_xticks(np.arange(1, len(plot_labels) + 1))
            ax.set_xticklabels(plot_labels, fontsize=11)
            ax.set_ylabel("Stopping pulls", fontsize=12)
            title = (
                "Stopping-time distribution by algorithm (merged)\n"
                "Box = IQR, whiskers = range, ★ = mean, dots = runs"
            )
            if clipped_total:
                title += f"  (clipped {clipped_total} outlier{'s' if clipped_total > 1 else ''} at P99)"
            ax.set_title(title, fontsize=13)

            ax.set_ylim(0, y_cap * 1.06)
            ax.grid(alpha=0.25, axis="y")
            ax.margins(x=0.03)

            leg = ax.legend(handles=[lb_line, pr_line], loc="upper left", frameon=True)
            leg.get_frame().set_alpha(0.9)

            fig.tight_layout()
            plt.show()

    # Final payload.
    results = {
        "global": {
            "T_star": T_star,
            "fixed_conf_lb": full_lb,
            "delta_used": delta,
            "true_best": true_best,
        },
        "algorithms": merged_alg,
    }

    # Save merged JSON to results/<joined_labels>.json.
    def _slug(s: str) -> str:
        import re as _re

        return _re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()

    fname = "_".join(_slug(lbl) for lbl in labels) or "summary"
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fname}.json"

    # JSON-safe serialization
    json_ready = {
        "global": results["global"],
        "algorithms": {
            label: {
                **stats,
                "avg_last_hist": np.asarray(stats["avg_last_hist"], float).tolist(),
            }
            for label, stats in results["algorithms"].items()
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)

    if verbose:
        print(f"[run_experiment_multi] Saved merged results → {out_path.resolve()}")

    return results


def run_experiment(
    env_path: str,
    algo_name: str,
    algo_args: dict,
    tracker_name: str | None,
    stopping_name: str | None,
    runs: int = 10,
    show_plot: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run `runs` independent repetitions of the bandit with the given configuration.
    - delta is taken from algo_args["confidence"] (preferred) or algo_args["delta"], default 0.1
    - expected best arm is taken from the environment (argmax of true means)
    """
    # 1) Build bandit from saved env
    b = Bandit.from_saved_env(
        path=env_path,
        algo_name=algo_name,
        algo_args=algo_args,
        tracker_name=tracker_name,
        stopping_name=stopping_name,
    )

    # 2) Extract true means from env
    env = b.env
    if hasattr(env, "probs"):
        mu = np.array(env.probs, dtype=float)
    elif hasattr(env, "mu"):
        mu = np.array(env.mu, dtype=float)
    else:
        raise ValueError("Unknown environment structure for extracting true means")

    # 3) Determine delta from algo_args (preferred key: 'confidence')
    delta = algo_args.get("confidence", None)
    if delta is None:
        delta = algo_args.get("delta", 0.1)
    # basic sanity clamp
    if not (0.0 < float(delta) < 1.0):
        raise ValueError(f"delta must be in (0,1); got {delta!r}")

    # 4) Compute information-theoretic lower bounds
    T_star = lower_bound(mu, env=env)
    full_lb = fixed_confidence_lower_bound(mu, env=env, delta=float(delta))

    if verbose:
        print(f"[Lower bound] T*(mu) = {T_star:.6f}")
        print(
            f"[Fixed-confidence bound with delta={delta}] T*(mu) * kl(delta,1-delta) = {full_lb:.6f}"
        )

    # 5) Repeated runs
    cumulative_hist = np.zeros_like(mu, dtype=float)
    pulls_total = 0
    time_total = 0.0
    wrong = 0
    max_pulls = 0
    last_history = None

    true_best = int(np.argmax(mu))

    for i in range(runs):
        start = time.perf_counter()
        best, counts, rewards, history = b.run()
        elapsed = time.perf_counter() - start

        if best != true_best:
            wrong += 1

        total_pulls = int(counts.sum())
        pulls_total += total_pulls
        max_pulls = max(max_pulls, total_pulls)
        time_total += elapsed

        cumulative_hist += history[-1]
        last_history = history

        if verbose:
            print(
                f"Run {i + 1:2d} | Algo: {algo_name} | Best: {best} | Pulls: {total_pulls} | "
                f"Hist(last)={history[-1]} | Avg pulls: {pulls_total / (i + 1):.2f} | "
                f"Avg time: {time_total / (i + 1):.4f} | Wrong: {wrong}"
            )

    avg_hist = cumulative_hist / runs

    summary = {
        "T_star": T_star,
        "fixed_conf_lb": full_lb,
        "delta_used": float(delta),
        "true_best": true_best,
        "avg_pulls": pulls_total / runs,
        "max_pulls": max_pulls,
        "avg_time": time_total / runs,
        "wrong": wrong,
        "avg_last_hist": avg_hist,
    }

    if verbose:
        print("Average final empirical proportions:", avg_hist)

    # 6) Plot empirical proportions + true w* with matching colors
    if show_plot and last_history is not None:
        K = len(mu)
        times = np.arange(1, len(last_history) + 1)

        fig, ax = plt.subplots()

        # get default color cycle (may be None if backend has no cycle)
        prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
        colors = None
        if prop_cycle is not None:
            by_key = prop_cycle.by_key()
            colors = by_key.get("color", None)

        # empirical proportions
        for k in range(K):
            series = [h[k] for h in last_history]
            color = None
            if colors:
                color = colors[k % len(colors)]
            ax.plot(times, series, label=f"arm {k}", linewidth=1.5, color=color)

        # true w*
        w_star = _compute_w_star(mu, env=env)
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
        fig.tight_layout()
        plt.show()

    return summary


def _pad_worst_curve(w: np.ndarray, horizon: int, K: int) -> np.ndarray:
    full = np.full(horizon, np.nan, dtype=float)
    steps = max(horizon - K, 0)
    if steps and w.size:
        full[K : K + min(steps, w.shape[0])] = w[:steps]
    return full


def _pad_prop_history(ph: np.ndarray, horizon: int, K: int) -> np.ndarray:
    full = np.full((horizon, K), np.nan, dtype=float)
    steps = max(horizon - K, 0)
    if steps and ph.size:
        full[K : K + min(steps, ph.shape[0]), :] = ph[:steps, :]
    return full


def _combine_mean_var(
    mean1: np.ndarray,
    var1: np.ndarray,
    n1: int,
    mean2: np.ndarray,
    var2: np.ndarray,
    n2: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Parallel (per-element) combination of two sample-mean/variance summaries.
    Returns (mean, var, n_total) with sample variance (ddof=1) when n_total>1.
    """
    if n1 == 0:
        return mean2, var2, n2
    if n2 == 0:
        return mean1, var1, n1
    n_tot = n1 + n2
    delta = mean2 - mean1
    mean = mean1 + (n2 / n_tot) * delta
    # Convert sample var -> M2 (sum of squared deviations)
    M2_1 = (n1 - 1) * var1
    M2_2 = (n2 - 1) * var2
    M2 = M2_1 + M2_2 + (n1 * n2 / n_tot) * (delta**2)
    var = np.where(n_tot > 1, M2 / (n_tot - 1), np.zeros_like(mean))
    return mean, var, n_tot


def run_and_plot_average_stats(
    bandit_factories: Sequence[Callable[[], Bandit]],
    labels: Sequence[str],
    horizon: int,
    n_runs: int = 40,
    deltas: Sequence[float] | None = None,
    *,
    ax_chernoff: Optional["plt.Axes"] = None,
    show: bool = True,
    merge_from_json: Optional[str] = None,
    progress_every: int = 10,
) -> tuple[
    Dict[str, np.ndarray],  # avg_props_timeseries
    Dict[str, np.ndarray],  # avg_final_props
    Dict[str, np.ndarray],  # w_star_by_algo
    Dict[str, float],  # avg_runtime_secs
]:
    """Run fixed-horizon repetitions, merge summaries, save JSON, and plot statistics.

    Return average allocation time series, final allocations, target weights,
    and runtimes, each keyed by algorithm label.

    Report counters and estimated remaining simulation time every
    ``progress_every`` runs per algorithm; use zero to disable progress output.
    """
    if len(bandit_factories) != len(labels):
        raise ValueError("`bandit_factories` and `labels` must have the same length")
    if progress_every < 0:
        raise ValueError("`progress_every` must be nonnegative")

    # Probe env/K from the first factory
    probe = bandit_factories[0]()
    K = probe.env.num_arms
    env_ref = probe.env
    del probe

    if horizon <= K:
        raise ValueError(
            f"horizon must be > K to record post-bootstrap stats (h={horizon}, K={K})."
        )

    if deltas is None:
        # Default: wide range of confidence values (1e-1 ... 1e-20)
        deltas = [10.0 ** (-k) for k in range(1, 21)]

    # time axis and colours
    t_axis = np.arange(1, horizon + 1, dtype=float)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_algos = len(bandit_factories)

    # Storage for NEW runs
    worst_runs = [np.full((n_runs, horizon), np.nan, float) for _ in range(n_algos)]
    prop_runs = [np.full((n_runs, horizon, K), np.nan, float) for _ in range(n_algos)]
    mean_runs = [np.full((n_runs, horizon, K), np.nan, float) for _ in range(n_algos)]
    final_props_runs = [np.full((n_runs, K), np.nan, float) for _ in range(n_algos)]
    runtimes_runs = [np.zeros(n_runs, dtype=float) for _ in range(n_algos)]
    w_star_by_algo: Dict[str, np.ndarray] = {}

    # Monte-Carlo.
    import time as _time

    simulation_start = _time.perf_counter()
    for r in range(n_runs):
        for j, (make, label) in enumerate(zip(bandit_factories, labels)):
            b = make()
            if b.env.num_arms != K:
                raise ValueError(
                    "All factories must use environments with the same number of arms (K)."
                )

            t0 = _time.perf_counter()
            best, worst, prop, means = b.run_fixed(horizon)
            runtimes_runs[j][r] = _time.perf_counter() - t0

            # place outputs on global timeline: first K steps stay NaN
            steps = min(max(horizon - K, 0), worst.shape[0])
            if steps:
                worst_runs[j][r, K : K + steps] = worst[:steps]
                prop_runs[j][r, K : K + steps, :] = prop[:steps, :]
                mean_runs[j][r, K : K + steps, :] = means[:steps, :]
                final_props_runs[j][r, :] = prop[-1, :]

            # record w* (once per algorithm)
            if label not in w_star_by_algo:
                if hasattr(b.env, "mu"):
                    mu_vec = np.asarray(b.env.mu, dtype=float)
                elif hasattr(b.env, "probs"):
                    mu_vec = np.asarray(b.env.probs, dtype=float)
                else:
                    raise ValueError(
                        "Environment must expose `mu` (normal) or `probs` (bernoulli)."
                    )
                w_star_by_algo[label] = _compute_w_star(mu_vec, env=b.env)

        completed = r + 1
        if progress_every and (completed % progress_every == 0 or completed == n_runs):
            remaining_runs = n_runs - completed
            for j, label in enumerate(labels):
                average_seconds = float(np.mean(runtimes_runs[j][:completed]))
                remaining_seconds = remaining_runs * average_seconds
                print(
                    f"[{label}] {completed:,}/{n_runs:,} runs"
                    f" | average {average_seconds:.3f}s/run"
                    f" | remaining compute ~{timedelta(seconds=round(remaining_seconds))}",
                    flush=True,
                )
            elapsed = _time.perf_counter() - simulation_start
            remaining = remaining_runs * elapsed / completed
            print(
                f"[combined] {completed * n_algos:,}/{n_runs * n_algos:,} executions"
                f" ({completed / n_runs:.1%})"
                f" | elapsed {timedelta(seconds=round(elapsed))}"
                f" | ETA ~{timedelta(seconds=round(remaining))}"
                " (final summaries and plot export follow)",
                flush=True,
            )

    # Diagnostics computed from new runs.
    # 1) Per-delta stopping pulls using the Chernoff 'worst' statistic
    new_stop_pulls_per_delta: Dict[str, Dict[str, list[int]]] = {}
    new_avg_stop_pulls_per_delta: Dict[str, Dict[str, float]] = {}

    # Precompute beta_t(delta) on the global time axis once per delta
    beta_by_delta = {
        f"{d:.12g}": np.log((np.log(t_axis) + 1.0) / float(d)) for d in deltas
    }

    for j, label in enumerate(labels):
        per_delta = {k: [] for k in beta_by_delta.keys()}
        wr = worst_runs[j]  # shape (n_runs, horizon)

        for r in range(n_runs):
            w_curve = wr[r, :]  # includes NaNs for t <= K
            for d_key, beta_t in beta_by_delta.items():
                valid = np.isfinite(w_curve)
                hit = np.where(np.logical_and(valid, w_curve > beta_t))[0]
                t_star = int(hit[0] + 1) if hit.size else int(horizon)
                per_delta[d_key].append(t_star)

        new_stop_pulls_per_delta[label] = per_delta
        new_avg_stop_pulls_per_delta[label] = {
            k: float(np.mean(v)) if len(v) else float("nan")
            for k, v in per_delta.items()
        }

    # 2) Error probability vs time  (and 3) Simple regret vs time)
    if hasattr(env_ref, "mu"):
        mu_vec_plot = np.asarray(env_ref.mu, dtype=float)
    elif hasattr(env_ref, "probs"):
        mu_vec_plot = np.asarray(env_ref.probs, dtype=float)
    else:
        raise ValueError("Environment must expose `mu` or `probs`.")
    true_best = int(np.argmax(mu_vec_plot))
    best_mean = float(mu_vec_plot[true_best])

    new_avg_error_ts: Dict[str, np.ndarray] = {}
    new_avg_simple_regret_ts: Dict[str, np.ndarray] = {}

    for j, label in enumerate(labels):
        m = mean_runs[j]  # (n_runs, horizon, K) with NaN for t<=K
        valid = np.any(np.isfinite(m), axis=2)  # (n_runs, horizon)
        m_filled = np.nan_to_num(m, nan=-np.inf)
        est_best = np.argmax(m_filled, axis=2)  # (n_runs, horizon)
        est_best = np.where(valid, est_best, -1)

        correct = (est_best == true_best) & valid
        denom = np.sum(valid, axis=0).astype(float)  # (horizon,)
        pcs = np.divide(
            np.sum(correct, axis=0).astype(float),
            denom,
            out=np.full(horizon, np.nan, float),
            where=denom > 0,
        )
        err = 1.0 - pcs  # error probability
        new_avg_error_ts[label] = err

        est_b_safe = np.where(valid, est_best, true_best)
        mu_hat = mu_vec_plot[est_b_safe]  # (n_runs, horizon)
        sr = np.where(valid, best_mean - mu_hat, np.nan)
        new_avg_simple_regret_ts[label] = np.nanmean(sr, axis=0)

    # Summaries from NEW runs (existing aggregates).
    def _combine_mean_var(
        mean1: np.ndarray,
        var1: np.ndarray,
        n1: int,
        mean2: np.ndarray,
        var2: np.ndarray,
        n2: int,
    ):
        if n1 == 0:
            return mean2, var2, n2
        if n2 == 0:
            return mean1, var1, n1
        n_tot = n1 + n2
        delta = mean2 - mean1
        mean = mean1 + (n2 / n_tot) * delta
        # Convert sample var -> M2 (sum of squared deviations)
        M2_1 = (n1 - 1) * var1
        M2_2 = (n2 - 1) * var2
        M2 = M2_1 + M2_2 + (n1 * n2 / n_tot) * (delta**2)
        var = np.where(n_tot > 1, M2 / (n_tot - 1), np.zeros_like(mean))
        return mean, var, n_tot

    new_avg_worst: Dict[str, np.ndarray] = {}
    new_var_worst: Dict[str, np.ndarray] = {}
    new_avg_props: Dict[str, np.ndarray] = {}
    new_avg_final_props: Dict[str, np.ndarray] = {}
    new_avg_runtime: Dict[str, float] = {}
    new_n: Dict[str, int] = {label: n_runs for label in labels}

    for j, label in enumerate(labels):
        tail = worst_runs[j][:, K:]  # (n_runs, horizon-K)
        mean_tail = np.nanmean(tail, axis=0)
        var_tail = (
            np.nanvar(tail, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_tail)
        )

        avg_worst = np.full(horizon, np.nan, float)
        avg_worst[K:] = mean_tail
        var_worst = np.full(horizon, np.nan, float)
        var_worst[K:] = var_tail
        new_avg_worst[label] = avg_worst
        new_var_worst[label] = var_worst

        tail_props = prop_runs[j][:, K:, :]  # (n_runs, horizon-K, K)
        mean_props = np.nanmean(tail_props, axis=0)  # (horizon-K, K)
        mat = np.full((horizon, K), np.nan, float)
        mat[K:, :] = mean_props
        new_avg_props[label] = mat

        new_avg_final_props[label] = np.nanmean(final_props_runs[j], axis=0)
        new_avg_runtime[label] = float(np.mean(runtimes_runs[j]))

    # Optional MERGE from existing JSON.

    merged_avg_worst = dict(new_avg_worst)
    merged_var_worst = dict(new_var_worst)
    merged_avg_props = dict(new_avg_props)
    merged_avg_runtime = dict(new_avg_runtime)
    merged_avg_final = dict(new_avg_final_props)
    merged_n = dict(new_n)
    merged_stop_pulls_per_delta = dict(new_stop_pulls_per_delta)
    merged_avg_stop_pulls_per_delta = dict(new_avg_stop_pulls_per_delta)
    merged_avg_error_ts = dict(new_avg_error_ts)
    merged_avg_simple_regret_ts = dict(new_avg_simple_regret_ts)

    old: Dict[str, Any] = {}
    if merge_from_json:
        p = Path(merge_from_json)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    old = json.load(f)
            except Exception as e:
                print(f"[merge] Could not load JSON ({p}): {e}")

    old_alg = old.get("algorithms", {}) if isinstance(old, dict) else {}

    for label in labels:
        if label in old_alg:
            prev = old_alg[label]
            prev_aw = np.asarray(prev.get("avg_worst", []), float)
            prev_vw = np.asarray(prev.get("var_worst", []), float)
            prev_ap = np.asarray(prev.get("avg_props", []), float)
            prev_n = int(prev.get("n_runs", 0))
            prev_rt = float(prev.get("avg_runtime_secs", 0.0))
            if (
                prev_aw.shape != (horizon,)
                or prev_vw.shape != (horizon,)
                or prev_ap.shape != (horizon, K)
            ):
                raise ValueError(
                    f"[merge] Shape mismatch for '{label}' in JSON. "
                    f"Expected {(horizon,)}, {(horizon,)}, {(horizon, K)}; "
                    f"got {prev_aw.shape}, {prev_vw.shape}, {prev_ap.shape}"
                )

            mean1, var1, n1 = prev_aw, prev_vw, prev_n
            mean2, var2, n2 = new_avg_worst[label], new_var_worst[label], new_n[label]
            comb_mean, comb_var, n_tot = _combine_mean_var(
                mean1, var1, n1, mean2, var2, n2
            )

            merged_avg_worst[label] = comb_mean
            merged_var_worst[label] = comb_var
            merged_n[label] = n_tot

            merged_avg_props[label] = prev_ap * (
                prev_n / max(n_tot, 1)
            ) + new_avg_props[label] * (new_n[label] / max(n_tot, 1))

            merged_avg_final[label] = merged_avg_props[label][-1, :]
            merged_avg_runtime[label] = (
                prev_rt * prev_n + new_avg_runtime[label] * new_n[label]
            ) / max(n_tot, 1)

            prev_sp = prev.get("stop_pulls_per_delta", {}) or {}
            merged_dict = {
                k: list(v) for k, v in new_stop_pulls_per_delta[label].items()
            }
            for k, lst in prev_sp.items():
                if k in merged_dict:
                    merged_dict[k] = list(lst) + list(merged_dict[k])
                else:
                    merged_dict[k] = list(lst)
            merged_stop_pulls_per_delta[label] = merged_dict
            merged_avg_stop_pulls_per_delta[label] = {
                k: float(np.mean(v)) if len(v) else float("nan")
                for k, v in merged_dict.items()
            }

            prev_err = np.asarray(prev.get("avg_error_probability", []), float)
            prev_sr = np.asarray(prev.get("avg_simple_regret", []), float)
            if prev_err.shape == (horizon,) and prev_sr.shape == (horizon,):
                w1 = prev_n / max(n_tot, 1)
                w2 = new_n[label] / max(n_tot, 1)
                merged_avg_error_ts[label] = np.where(
                    np.isfinite(prev_err) | np.isfinite(new_avg_error_ts[label]),
                    np.nan_to_num(prev_err) * w1
                    + np.nan_to_num(new_avg_error_ts[label]) * w2,
                    np.nan,
                )
                merged_avg_simple_regret_ts[label] = np.where(
                    np.isfinite(prev_sr) | np.isfinite(new_avg_simple_regret_ts[label]),
                    np.nan_to_num(prev_sr) * w1
                    + np.nan_to_num(new_avg_simple_regret_ts[label]) * w2,
                    np.nan,
                )
            else:
                merged_avg_error_ts[label] = new_avg_error_ts[label]
                merged_avg_simple_regret_ts[label] = new_avg_simple_regret_ts[label]

    for label, prev in old_alg.items():
        if label not in merged_avg_worst:
            merged_avg_worst[label] = np.asarray(prev.get("avg_worst", []), float)
            merged_var_worst[label] = np.asarray(prev.get("var_worst", []), float)
            merged_avg_props[label] = np.asarray(prev.get("avg_props", []), float)
            merged_avg_final[label] = (
                merged_avg_props[label][-1, :]
                if merged_avg_props[label].size
                else np.array([])
            )
            merged_avg_runtime[label] = float(prev.get("avg_runtime_secs", 0.0))
            merged_n[label] = int(prev.get("n_runs", 0))

            if "stop_pulls_per_delta" in prev:
                merged_stop_pulls_per_delta[label] = {
                    k: list(v) for k, v in prev["stop_pulls_per_delta"].items()
                }
                merged_avg_stop_pulls_per_delta[label] = {
                    k: float(np.mean(v)) if len(v) else float("nan")
                    for k, v in merged_stop_pulls_per_delta[label].items()
                }
            if "avg_error_probability" in prev and "avg_simple_regret" in prev:
                merged_avg_error_ts[label] = np.asarray(
                    prev["avg_error_probability"], float
                )
                merged_avg_simple_regret_ts[label] = np.asarray(
                    prev["avg_simple_regret"], float
                )

    # Save merged JSON.

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "_".join([re.sub(r"[^A-Za-z0-9_.-]+", "_", s) for s in labels])
    out_path = out_dir / f"{fname}.json"
    env_kind = (
        "bernoulli"
        if hasattr(env_ref, "probs")
        else ("normal" if hasattr(env_ref, "mu") else "unknown")
    )

    json_ready = {
        "meta": {
            "horizon": horizon,
            "K": int(K),
            "env_kind": env_kind,
            "deltas": [float(d) for d in deltas],
        },
        "algorithms": {
            label: {
                "avg_worst": merged_avg_worst[label].tolist(),
                "var_worst": merged_var_worst[label].tolist(),
                "avg_props": merged_avg_props[label].tolist(),  # (horizon × K)
                "avg_runtime_secs": float(merged_avg_runtime[label]),
                "n_runs": int(merged_n[label]),
                "stop_pulls_per_delta": {
                    k: list(map(int, v))
                    for k, v in merged_stop_pulls_per_delta.get(label, {}).items()
                },
                "avg_stop_pulls_per_delta": {
                    k: float(v)
                    for k, v in merged_avg_stop_pulls_per_delta.get(label, {}).items()
                },
                "avg_error_probability": merged_avg_error_ts.get(
                    label, np.full(horizon, np.nan)
                ).tolist(),
                "avg_simple_regret": merged_avg_simple_regret_ts.get(
                    label, np.full(horizon, np.nan)
                ).tolist(),
            }
            for label in merged_avg_worst.keys()
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)
    print(f"[run_fixed] Saved merged results → {out_path.resolve()}")

    # Plots from the MERGED data.
    # Chernoff plot (all algorithms, merged)
    created_A = ax_chernoff is None
    if created_A:
        _, ax_chernoff = plt.subplots(figsize=(10, 6))

    # order: inputs first, then JSON-only
    plot_labels = []
    seen = set()
    for lbl in labels:
        if lbl in merged_avg_worst and lbl not in seen:
            plot_labels.append(lbl)
            seen.add(lbl)
    for lbl in merged_avg_worst.keys():
        if lbl not in seen:
            plot_labels.append(lbl)
            seen.add(lbl)

    for j, label in enumerate(plot_labels):
        col = colours[j % len(colours)]
        aw = merged_avg_worst[label]
        ax_chernoff.plot(t_axis, aw, color=col, label=label)
        for d in deltas:
            beta = np.log((np.log(t_axis) + 1.0) / d)
            hit = np.where(aw > beta)[0]
            if hit.size:
                t_star = hit[0] + 1
                ax_chernoff.axvline(
                    t_star, color=col, linestyle="--", alpha=0.6, lw=1.2, zorder=5
                )

    ax_chernoff.set_xlabel("Pull index $t$")
    ax_chernoff.set_ylabel("Average Chernoff statistic $W_t$")
    ax_chernoff.set_title(
        f"Average Chernoff worst over merged runs, horizon={horizon}n"
        r"Dashed lines: $t^ast$ where $W_t > beta_t(delta)=ln((ln t+1)/delta)$"
    )
    ax_chernoff.legend()
    ax_chernoff.grid(alpha=0.3)
    plt.tight_layout()

    # Allocation plots (one per algorithm, merged; use w* from env_ref)
    if hasattr(env_ref, "mu"):
        mu_vec_plot = np.asarray(env_ref.mu, dtype=float)
    elif hasattr(env_ref, "probs"):
        mu_vec_plot = np.asarray(env_ref.probs, dtype=float)
    else:
        raise ValueError("Environment must expose `mu` or `probs`.")
    w_star_plot = _compute_w_star(mu_vec_plot, env=env_ref)

    for j, label in enumerate(plot_labels):
        fig, ax = plt.subplots(figsize=(10, 6))
        props = merged_avg_props[label]  # (horizon × K)
        for a in range(K):
            col = colours[a % len(colours)]
            ax.plot(t_axis, props[:, a], color=col, lw=1.8, label=f"Arm {a} (emp.)")
        ax.plot(
            t_axis,
            np.repeat(w_star_plot[None, :], repeats=horizon, axis=0),
            linestyle="--",
            alpha=0.5,
        )
        ax.set_title(f"Average allocation over merged runs — {label}")
        ax.set_xlabel("Pull index $t$")
        ax.set_ylabel("Proportion of pulls")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        plt.tight_layout()

    # Error probability across time (all algorithms in one plot)
    figE, axE = plt.subplots(figsize=(10, 6))
    for j, label in enumerate(plot_labels):
        col = colours[j % len(colours)]
        ep = merged_avg_error_ts.get(label, np.full(horizon, np.nan))
        axE.plot(t_axis, ep, label=label, color=col)
    axE.set_xlabel("Pull index $t$")
    axE.set_ylabel("Error probability (incorrect best arm)")
    axE.set_title(
        "Error probability vs. time (fraction of runs choosing a suboptimal arm)"
    )
    axE.set_ylim(0, 1)
    axE.grid(alpha=0.3)
    axE.legend()
    plt.tight_layout()

    # Simple regret across time (all algorithms in one plot)
    figR, axR = plt.subplots(figsize=(10, 6))
    for j, label in enumerate(plot_labels):
        col = colours[j % len(colours)]
        sr = merged_avg_simple_regret_ts.get(label, np.full(horizon, np.nan))
        axR.plot(t_axis, sr, label=label, color=col)
    axR.set_xlabel("Pull index $t$")
    axR.set_ylabel("Simple regret")
    axR.set_title("Simple regret vs. time")
    axR.grid(alpha=0.3)
    axR.legend()
    plt.tight_layout()

    if show:
        plt.show()

    # Build return values (from merged).
    avg_props_timeseries = {lbl: merged_avg_props[lbl] for lbl in merged_avg_props}
    avg_final_props = {lbl: merged_avg_final[lbl] for lbl in merged_avg_final}
    avg_runtime_secs = {lbl: merged_avg_runtime[lbl] for lbl in merged_avg_runtime}

    # Supply w* for algorithms we actually ran; for JSON-only, reuse env_ref
    for lbl in merged_avg_props:
        if lbl not in w_star_by_algo:
            w_star_by_algo[lbl] = w_star_plot

    return avg_props_timeseries, avg_final_props, w_star_by_algo, avg_runtime_secs
