"""Identify a Bernoulli best arm, then inspect a fixed-budget allocation."""

from pathlib import Path

import numpy as np

from best_arm import Bandit

ENVIRONMENT = Path(__file__).resolve().parent / "environments" / "bernoulli.json"


def make_bandit():
    """Construct fresh components for an independent trial."""
    return Bandit.from_saved_env(
        path=str(ENVIRONMENT),
        algo_name="track_and_stop",
        algo_args={"confidence": 0.1},
        tracker_name="d_tracking",
        stopping_name="chernoff",
    )


def main():
    np.random.seed(7)
    best, counts, rewards, history = make_bandit().run()
    print("True best arm: 1 (indices start at 0)")
    print("Selected arm:", best)
    print("Pulls per arm:", counts)
    print("Estimated means:", rewards / counts)
    print("Final allocation:", history[-1])

    best, statistics, allocations, means = make_bandit().run_fixed(horizon=300)
    print("\nFixed budget: 300 pulls")
    print("Selected arm:", best)
    print("Final allocation:", allocations[-1])
    print("Final means:", means[-1])
    print("Final Chernoff statistic:", statistics[-1])


if __name__ == "__main__":
    main()
