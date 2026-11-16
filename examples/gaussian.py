"""Run the Gaussian-only probability-of-maximum algorithm on a saved problem."""

from pathlib import Path

import numpy as np

from best_arm import Bandit


def main():
    np.random.seed(7)
    environment = Path(__file__).resolve().parent / "environments" / "gaussian.json"
    bandit = Bandit.from_saved_env(
        path=str(environment),
        algo_name="pmax",
        algo_args={"confidence": 0.1},
        stopping_name="chernoff",
    )
    best, statistics, allocations, means = bandit.run_fixed(horizon=300)
    print("True means:", bandit.env.mu)
    print("Standard deviation:", bandit.env.sigma)
    print("Selected arm:", best)
    print("Final estimated means:", means[-1])
    print("Final allocation:", allocations[-1])
    print("Recorded steps after the 5-pulls-per-arm bootstrap:", len(statistics))


if __name__ == "__main__":
    main()
