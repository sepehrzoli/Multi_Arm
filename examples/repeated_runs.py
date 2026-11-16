"""Summarize three fixed-confidence trials using a stateless tracker."""

import json
from pathlib import Path

import numpy as np

from best_arm import run_experiment


def main():
    np.random.seed(7)
    environment = Path(__file__).resolve().parent / "environments" / "bernoulli.json"
    summary = run_experiment(
        env_path=str(environment),
        algo_name="track_and_stop",
        algo_args={"confidence": 0.1},
        tracker_name="d_tracking",
        stopping_name="chernoff",
        runs=3,
        show_plot=False,
    )
    print(
        json.dumps(
            summary,
            indent=2,
            default=lambda value: value.tolist(),
        )
    )


if __name__ == "__main__":
    main()
