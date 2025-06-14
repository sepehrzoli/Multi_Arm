from bandit import Bandit


if __name__ == "__main__":
    bandit = Bandit(
        env_name="bernoulli",
        tracker_name="c_tracking",
        algo_name="track_and_stop",
        env_args={"probabilities": [0.1, 0.5, 0.8]},
        algo_args={"confidence": 0.05},
    )
    for i in range(100):
        result = bandit.run()
        print("Run ",i,": ","Best Arm: ", result[0], " counts: ", result[1], " rewards: ", result[2], " Means: ", result[2]/result[1])
