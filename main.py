from bandit import Bandit
import time
if __name__ == "__main__":  # Stopping files + not exp_gap
    b2 = Bandit.from_saved_env(
        path="env_0003.json",
        algo_name="lil_ucb",
        algo_args={"confidence": 0.1},
        tracker_name=None,
        stopping_name="lil_ucb_original"
    )
    pulls = 0
    tsum = 0
    t = 0
    wrong = 0
    for i in range(300):
        start = time.perf_counter()
        best2, counts2, rewards2, hist2 = b2.run()
        elapsed2 = time.perf_counter() - start
        if best2 != 0:
            wrong += 1
        pulls = pulls + counts2.sum()
        tsum += elapsed2
        t += 1
        print("Run ",i+1,"Track and Stop D on same env:", best2, counts2, rewards2, counts2.sum(), hist2[-1], pulls/t, tsum/t, wrong)

    wrong = 0
    b2 = Bandit.from_saved_env(
        path="env_0002.json",
        algo_name="track_and_stop",
        algo_args={"confidence": 0.1},
        tracker_name="d_tracking",
        stopping_name="chernoff"
    )
    pulls = 0
    tsum = 0
    t = 0
    for i in range(300):
        start = time.perf_counter()
        best2, counts2, rewards2, hist2 = b2.run()
        elapsed2 = time.perf_counter() - start
        if best2 != 0:
            wrong += 1
        pulls = pulls + counts2.sum()
        tsum += elapsed2
        t += 1
        print("Run ", i + 1, "Track and Stop D on same env:", best2, counts2, rewards2, counts2.sum(), hist2[-1],
              pulls / t, tsum / t, wrong)
