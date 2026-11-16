# Multi_Arm: best-arm identification

A Python research framework for testing **best-arm identification** in stochastic
multi-armed bandits. Combine reward environments, sampling algorithms, tracking
policies, and stopping rules; inspect how pulls are allocated; and compare methods
on the same problem.

The importable package is `best_arm`. It includes 13 algorithms, 11 tracking
policies, Bernoulli and Gaussian rewards, and experiment utilities built on NumPy,
SciPy, and Matplotlib.

## Install

Use **Python 3.9–3.12** (the current SciPy dependency range supports these versions).
Clone the repository and install it in a virtual environment:

```bash
git clone https://github.com/sepehrzoli/Multi_Arm.git
cd Multi_Arm
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install the local package:

```bash
python -m pip install -e .
python examples/quickstart.py
```

The quickstart uses a small saved Bernoulli environment, identifies an arm, and
prints its pull counts and estimated means. Arm indices start at **0**; the true
best arm in the supplied example is arm **1**. Results are stochastic. The examples
set a NumPy seed, so results can be reproduced within the same dependency versions.

## First experiment

Run this from the repository root after installation:

```python
import numpy as np
from best_arm import Bandit

np.random.seed(7)
bandit = Bandit.from_saved_env(
    path="examples/environments/bernoulli.json",
    algo_name="track_and_stop",
    algo_args={"confidence": 0.1},
    tracker_name="d_tracking",
    stopping_name="chernoff",
)

best, counts, rewards, history = bandit.run()
print("Selected arm:", best)
print("Pulls per arm:", counts)
print("Estimated means:", rewards / counts)
```

`run()` continues until the stopping rule fires and returns the selected arm,
per-arm pull counts, per-arm reward sums, and allocation history. `confidence` is
the error parameter, usually denoted delta: smaller values generally need more
samples. Stopping runs have no global pull cap; closely spaced arms can take a
long time. Use a fixed budget when trying a new configuration:

```python
best, statistics, allocations, means = bandit.run_fixed(horizon=300)
print("Final allocation:", allocations[-1])
```

The returned arrays record the Chernoff statistic, allocation proportions, and
empirical means **after the algorithm's bootstrap**. Their lengths therefore
depend on the algorithm. Use a horizon comfortably above the number of arms
(`pmax` initially samples each arm five times). Exponential-gap elimination does
not implement `run_fixed`.

## Command line

The installed `multi-arm` command runs repeated fixed-confidence experiments:

```bash
multi-arm --env examples/environments/bernoulli.json --algo track_and_stop --tracker d_tracking --stopping chernoff --runs 3 --delta 0.1 --seed 7
```

`python -m best_arm` is equivalent and works when the console script is not on
your PATH. The command prints JSON with the true best arm, average/max pulls,
average runtime, number of wrong selections, average final allocation, and
information-theoretic reference bounds. The true best arm is inferred from the
saved environment. Use `--show-plot` to display allocation proportions for the
last run.

Algorithm-specific constructor options go in `--algo-args` as a JSON object;
`confidence` in that object takes precedence over `--delta`. For example, in a
POSIX shell:

```bash
multi-arm --env examples/environments/bernoulli.json --algo top_two --algo-args '{"leader_type":"EB","challenger_type":"TC","beta":0.5}' --runs 3 --seed 7
```

For detailed configuration from PowerShell or Python, the Python examples avoid
shell-specific JSON quoting. `multi-arm --help` lists registry names and options.

## Examples

All example scripts locate their input files relative to the script and can be
run from any working directory after the package is installed.

| Example | Command | What it demonstrates |
| --- | --- | --- |
| Quickstart | `python examples/quickstart.py` | One stopped Bernoulli run and one fixed-budget run |
| Gaussian rewards | `python examples/gaussian.py` | Gaussian environment, variance, and the Gaussian-only `pmax` method |
| Repeated experiments | `python examples/repeated_runs.py` | Summary statistics and the `run_experiment` Python API |
| Algorithm comparison | `python examples/compare_algorithms.py` | Fresh bandits per repetition; fixed-budget comparison of Track-and-Stop, TTSKKT, and TSPPS |

The comparison defaults to 3 repetitions of 300 pulls per algorithm and writes
`output/comparison.json` and `output/allocations.png`. Customize it with
`--runs 10 --horizon 1000 --seed 7 --output output/my-comparison`; add `--show`
to display the plot. For a machine without a display, set `MPLBACKEND=Agg`.
The figure shows mean final allocations; it is a demonstration, not a benchmark
claim. Runtime and error rates from a few trials are noisy.

## Algorithms and components

Pass these exact names to `algo_name` (or `--algo`):

| Name | Implementation / configuration |
| --- | --- |
| `track_and_stop` | Track-and-Stop; requires a tracker and a stopping rule |
| `exp_gap_elim` | Exponential-gap elimination; stopping runs only |
| `lil_ucb` | lil'UCB; `mode="theory"` or `"heuristic"` |
| `lucb` | LUCB |
| `lazyma` | Gradient-ascent allocation |
| `fws` | Frank-Wolfe sampling |
| `top_two` | Configurable top-two sampling; requires `leader_type`, `challenger_type`, and `beta` in `algo_args` |
| `adahedge_br` | AdaHedge with best responses |
| `iat2`, `at2` | Improved / anchored top-two sampling |
| `ttskkt` | Thompson top-two sampling with KKT allocation; optional `optimism` |
| `tspps` | Thompson sampling with posterior probability sampling |
| `pmax` | Best-seen probability-of-maximum sampling; **Gaussian only** |

`top_two` accepts leader choices `EB` or `TS` and challenger choices `TC`, `TCI`,
or `RS`. Those options belong to `top_two`, not `ttskkt`.

Tracking policies: `c_tracking`, `d_tracking`, `heuristic_tracking`, `g_tracking`,
`d_oracle_tracking`, `real_oracle_tracking`, `sampling_tracking`, `bc`,
`optimistic_tracking`, `noisy_tracking`, and `uniform_tracking`. Oracle trackers
use true environment means, so they serve as simulation references.

Stopping rules: `chernoff`, `lil_ucb_original`, `lil_stopping`, `lucb`,
`ls_and_lucb`, and `top_two`. Pass compatible rules to each method rather than
assuming every rule/algorithm pairing has the same theoretical guarantees. The
default Chernoff implementation uses `mode="practice"`; the Python API accepts
`stopping_args={"mode": "theory"}`. These are research implementations; empirical
smoke checks do not establish correctness or confidence guarantees.

Inspect the live registries in Python:

```python
from best_arm import ALGO_REGISTRY, TRACKER_REGISTRY, STOPPING_CONDITION_REGISTRY

print(list(ALGO_REGISTRY))
print(list(TRACKER_REGISTRY))
print(list(STOPPING_CONDITION_REGISTRY))
```

## Your own environments

Saved environments are small JSON files containing distribution parameters:

```json
{"env_name": "bernoulli", "state": {"probs": [0.2, 0.8, 0.4]}}
```

For Gaussian rewards, use `{"env_name": "normal", "state": {"mu": [0.0, 1.5, 0.5],
"sigma": 1.0}}`. `sigma` is the shared **standard deviation**. Use at least two
arms, valid probabilities, and a positive standard deviation. The saved format
stores parameters, not observations or random-generator state.

You can also generate a scenario:

```python
from best_arm import Bandit, BernoulliEnvironment

env = BernoulliEnvironment(num_arms=5, env_type="best_gap")
bandit = Bandit(
    env_instance=env,
    algo_name="track_and_stop",
    tracker_name="d_tracking",
    stopping_name="chernoff",
    algo_args={"confidence": 0.1},
)
bandit.save_env("my_environment.json")
```

Both Bernoulli and Gaussian generators support `uniform`, `best_gap`,
`competition`, `linear`, `1_sparse`, and `alpha_<value>` scenarios. Passing
`env_name`/`env_args` directly to `Bandit` also generates an environment and
automatically saves `env_XXXX.json` in the working directory.

## API notes

- Construct a **fresh `Bandit` per independent trial when using stateful
  trackers**, such as C-tracking. `Bandit.run_n` and `run_experiment` reuse their
  configured tracker/stopping instances. The comparison example uses factories;
  the repeated-run example uses stateless D-tracking.
- The ratings adapter is available as
  `best_arm.environment.BernoulliFromRatingsEnvironment` and can be supplied via
  `env_instance`. It does not support saved-environment JSON serialization.
- Exponential-gap elimination currently returns a placeholder allocation history.
- `best_arm.experiments` also contains `run_experiment_multi` and
  `run_and_plot_average_stats` for plotting, merging, and writing JSON summaries
  under `results/`. The latter aligns histories assuming a one-pull-per-arm
  bootstrap; its time alignment is unsuitable for `pmax`'s longer bootstrap.
  The comparison example uses `run_fixed` directly.

## Development

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
python -m build
```

Smoke tests cover registered algorithms on small Bernoulli/Gaussian problems,
the CLI, saved-environment round trips, and executable examples. CI runs the
checks on Python 3.9 and 3.12, including a Windows job.

```text
src/best_arm/       Public package and algorithm implementations
examples/          Runnable demos and small, synthetic environment fixtures
tests/             Integration and smoke tests
.github/workflows/  Continuous integration
```

The public repository includes the code, examples, tests, packaging metadata,
and documentation needed to install and try the framework. Virtual environments,
editor settings, generated results, local research drivers, and unfinished paper
drafts are excluded.

## License

[MIT](LICENSE). Copyright Sepehr Zolfaghari.
