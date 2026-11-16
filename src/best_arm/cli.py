"""Command-line interface for saved-environment experiments."""

import argparse
import json

import numpy as np

from .experiments import run_experiment
from .model import ALGO_REGISTRY, TRACKER_REGISTRY
from .stopping_conditions import STOPPING_CONDITION_REGISTRY


def main(argv=None):
    """Parse experiment options and print the resulting summary as JSON."""
    p = argparse.ArgumentParser(
        prog="multi-arm", description="Run multi-armed bandit experiments."
    )
    p.add_argument(
        "--env",
        required=True,
        help="Path to a saved Bernoulli or Gaussian environment JSON",
    )
    p.add_argument("--algo", required=True, choices=ALGO_REGISTRY)
    p.add_argument(
        "--algo-args", default="{}", help='JSON dict, e.g. {"confidence":0.1}'
    )
    p.add_argument("--tracker", default=None, choices=TRACKER_REGISTRY)
    p.add_argument(
        "--stopping", default="chernoff", choices=STOPPING_CONDITION_REGISTRY
    )
    p.add_argument("--runs", type=int, default=10)
    p.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Confidence parameter, overridden by confidence in --algo-args",
    )
    p.add_argument("--seed", type=int, default=None, help="NumPy random seed")
    p.add_argument("--show-plot", action="store_true")

    args = p.parse_args(argv)
    try:
        algo_args = json.loads(args.algo_args)
        if not isinstance(algo_args, dict):
            raise ValueError
    except (ValueError, TypeError):
        p.error("--algo-args must be a JSON object, e.g. '{\"confidence\":0.1}'")

    algo_args.setdefault("confidence", args.delta)
    confidence = algo_args["confidence"]
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 < confidence < 1
    ):
        p.error("confidence / --delta must be a number strictly between 0 and 1")
    if args.runs < 1:
        p.error("--runs must be at least 1")
    if args.algo == "track_and_stop" and args.tracker is None:
        p.error("track_and_stop requires --tracker (for example, d_tracking)")
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            p.error("--seed must be between 0 and 2**32 - 1")
        np.random.seed(args.seed)

    summary = run_experiment(
        env_path=args.env,
        algo_name=args.algo,
        algo_args=algo_args,
        tracker_name=args.tracker,
        stopping_name=args.stopping,
        runs=args.runs,
        show_plot=bool(args.show_plot),
    )

    # Pretty print numeric arrays
    def _fmt(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    print(json.dumps({k: _fmt(v) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
