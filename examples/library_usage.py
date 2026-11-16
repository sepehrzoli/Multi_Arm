"""Use the installed library without any repository files or saved environments."""

import numpy as np

from best_arm import Bandit, BaseEnvironment, __version__


def main():
    np.random.seed(7)
    environment = BaseEnvironment.from_dict(
        {"env_name": "bernoulli", "state": {"probs": [0.2, 0.8, 0.4]}}
    )
    bandit = Bandit(
        env_instance=environment,
        algo_name="track_and_stop",
        algo_args={"confidence": 0.1},
        tracker_name="d_tracking",
        stopping_name="chernoff",
    )
    best, statistics, allocations, means = bandit.run_fixed(horizon=300)
    print("best_arm version:", __version__)
    print("Selected arm:", best)
    print("Final estimated means:", means[-1])
    print("Final allocation:", allocations[-1])
    print("Final Chernoff statistic:", statistics[-1])


if __name__ == "__main__":
    main()
