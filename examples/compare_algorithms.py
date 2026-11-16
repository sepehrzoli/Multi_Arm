"""Compare fixed-budget trials and save JSON plus a final-allocation figure."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from best_arm import Bandit

ENVIRONMENT = Path(__file__).resolve().parent / "environments" / "bernoulli.json"
ALGORITHMS = (
    ("track_and_stop", "d_tracking"),
    ("ttskkt", None),
    ("tspps", None),
)


def compare(runs, horizon, seed):
    """Run fresh bandits and aggregate allocations, errors, and wall-clock time."""
    environment = Bandit.load_env(str(ENVIRONMENT))
    if runs < 1 or horizon <= environment.num_arms:
        raise ValueError("runs must be positive and horizon must exceed the arm count")
    true_best = int(np.argmax(environment.probs))
    summary = {
        "environment": environment.to_dict(),
        "runs": runs,
        "horizon": horizon,
        "seed": seed,
        "true_best": true_best,
        "algorithms": {},
    }
    for name, tracker in ALGORITHMS:
        final_allocations = []
        selected = []
        runtimes = []
        for repetition in range(runs):
            # Match initial seeds across methods; each method consumes its own draws.
            np.random.seed(seed + repetition)
            bandit = Bandit.from_saved_env(
                path=str(ENVIRONMENT),
                algo_name=name,
                algo_args={"confidence": 0.1},
                tracker_name=tracker,
                stopping_name="chernoff",
            )
            start = time.perf_counter()
            best, _, allocations, _ = bandit.run_fixed(horizon)
            runtimes.append(time.perf_counter() - start)
            selected.append(int(best))
            final_allocations.append(allocations[-1])
        summary["algorithms"][name] = {
            "selected_arms": selected,
            "wrong": sum(best != true_best for best in selected),
            "mean_final_allocation": np.mean(final_allocations, axis=0).tolist(),
            "mean_runtime_seconds": float(np.mean(runtimes)),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed + args.runs > 2**32:
        parser.error("seed and repetition seeds must be in NumPy's uint32 range")
    try:
        summary = compare(args.runs, args.horizon, args.seed)
    except ValueError as error:
        parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "comparison.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(figsize=(7, 4))
    arm_count = len(summary["environment"]["state"]["probs"])
    positions = np.arange(arm_count)
    width = 0.25
    for index, (name, result) in enumerate(summary["algorithms"].items()):
        axes.bar(
            positions + (index - 1) * width,
            result["mean_final_allocation"],
            width=width,
            label=name,
        )
        print(f"{name}: {result['wrong']}/{args.runs} wrong selections")
    axes.set_xticks(positions)
    axes.set_xlabel("Arm (0-based)")
    axes.set_ylabel("Mean fraction of pulls")
    axes.set_ylim(0, 1)
    axes.set_title(f"Final allocations: {args.runs} trials, {args.horizon} pulls each")
    axes.legend()
    figure.tight_layout()
    image_path = args.output / "allocations.png"
    figure.savefig(image_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(figure)
    print("Saved", json_path.resolve())
    print("Saved", image_path.resolve())


if __name__ == "__main__":
    main()
