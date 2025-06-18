# main.py

import numpy as np
from bandit import Bandit

def run_and_report(env_name, algo_name, tracker_name=None, env_args=None, algo_args=None, n_trials=10):
    env_args  = env_args or {}
    algo_args = algo_args or {}

    # instantiate once, so we can inspect the true params
    bandit = Bandit(env_name, algo_name, env_args=env_args, algo_args=algo_args, tracker_name=tracker_name)
    # grab ground truth
    if hasattr(bandit.env, "probs"):
        truth = np.array(bandit.env.probs)
    else:
        truth = np.array(bandit.env.mu)
    true_best = int(np.argmax(truth))

    successes = 0
    fraction_accum = np.zeros(bandit.env.num_arms)

    for i in range(n_trials):
        best, counts, rewards, history = bandit.run()
        final_frac = history[-1]  # vector of length K

        successes += (best == true_best)
        fraction_accum += final_frac

        print(f"[{i+1:2d}] best={best} (true={true_best}), pulls={int(counts.sum())}")

    avg_frac = fraction_accum / n_trials
    print(f"\n→ {algo_name} on {env_name} (n={n_trials}):")
    print(f"   Success rate: {successes}/{n_trials} = {successes/n_trials:.2f}")
    print(f"   Avg. sampling fractions per arm: {avg_frac.tolist()}\n")


if __name__ == "__main__":
    # run Track-and-Stop with DTracking on 3‐arm Bernoulli (competition)
    run_and_report(
        env_name="bernoulli",
        algo_name="track_and_stop",
        tracker_name="d_tracking",
        env_args={"num_arms": 3, "env_type": "competition"},
        algo_args={"confidence": 0.05},
        n_trials=10,
    )
    # run Track-and-Stop with CTracking on 3‐arm Bernoulli (competition)
    run_and_report(
        env_name="bernoulli",
        algo_name="track_and_stop",
        tracker_name="c_tracking",
        env_args={"num_arms": 3, "env_type": "competition"},
        algo_args={"confidence": 0.05},
        n_trials=10,
    )



    # run Exp‐Gap‐Elim on 5‐arm Normal (linear)
    run_and_report(
        env_name="normal",
        algo_name="exp_gap_elim",
        env_args={"num_arms": 5, "env_type": "linear", "sigma": 1.0},
        algo_args={"confidence": 0.05},
        n_trials=10,
    )
