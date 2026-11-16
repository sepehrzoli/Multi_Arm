"""Exercise the public package, CLI, and examples on small synthetic problems."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from best_arm import ALGO_REGISTRY, Bandit
from best_arm.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "examples" / "environments"


def make_bandit(name="track_and_stop", distribution="bernoulli"):
    arguments = {"confidence": 0.1}
    if name == "top_two":
        arguments.update(leader_type="EB", challenger_type="TC", beta=0.5)
    return Bandit.from_saved_env(
        path=str(ENVIRONMENTS / f"{distribution}.json"),
        algo_name=name,
        algo_args=arguments,
        tracker_name="d_tracking" if name == "track_and_stop" else None,
        stopping_name="chernoff",
    )


class PublicAPITests(unittest.TestCase):
    def test_fixed_budget_algorithms(self):
        """All supported algorithms return finite, normalized post-bootstrap data."""
        for distribution in ("bernoulli", "gaussian"):
            for name in ALGO_REGISTRY:
                if name == "exp_gap_elim" or (
                    name == "pmax" and distribution == "bernoulli"
                ):
                    continue
                with self.subTest(distribution=distribution, algorithm=name):
                    np.random.seed(7)
                    bandit = make_bandit(name, distribution)
                    best, statistics, allocations, means = bandit.run_fixed(40)
                    self.assertIn(best, range(bandit.env.num_arms))
                    self.assertGreater(len(statistics), 0)
                    self.assertEqual(allocations.shape, means.shape)
                    self.assertEqual(allocations.shape, (len(statistics), 3))
                    self.assertTrue(np.isfinite(statistics).all())
                    self.assertTrue(np.isfinite(means).all())
                    self.assertTrue((allocations >= 0).all())
                    np.testing.assert_allclose(allocations.sum(axis=1), 1.0)

    def test_stopped_run_returns_counts_and_rewards(self):
        np.random.seed(7)
        best, counts, rewards, history = make_bandit().run()
        self.assertIn(best, range(3))
        self.assertTrue((counts > 0).all())
        self.assertTrue((rewards >= 0).all())
        self.assertTrue((rewards <= counts).all())
        np.testing.assert_allclose(history[-1], counts / counts.sum())

    def test_saved_environment_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            for distribution in ("bernoulli", "gaussian"):
                with self.subTest(distribution=distribution):
                    bandit = make_bandit(distribution=distribution)
                    saved = Path(temporary) / f"{distribution}.json"
                    bandit.save_env(str(saved))
                    loaded = Bandit.load_env(str(saved))
                    self.assertEqual(loaded.to_dict(), bandit.env.to_dict())

    def test_exp_gap_rejects_fixed_budget(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            make_bandit("exp_gap_elim").run_fixed(40)

    def test_exp_gap_stopped_run(self):
        np.random.seed(7)
        best, counts, rewards, _ = make_bandit("exp_gap_elim").run()
        self.assertIn(best, range(3))
        self.assertTrue((counts > 0).all())
        self.assertTrue((rewards <= counts).all())


class CLITests(unittest.TestCase):
    def arguments(self):
        return [
            "--env",
            str(ENVIRONMENTS / "bernoulli.json"),
            "--algo",
            "track_and_stop",
            "--tracker",
            "d_tracking",
            "--runs",
            "1",
            "--seed",
            "7",
        ]

    def test_delta_reaches_experiment_and_prints_json(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli_main(self.arguments() + ["--delta", "0.2"])
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["delta_used"], 0.2)
        self.assertEqual(summary["true_best"], 1)
        self.assertGreater(summary["avg_pulls"], 0)

    def test_algorithm_confidence_overrides_delta(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli_main(
                self.arguments()
                + ["--delta", "0.2", "--algo-args", '{"confidence":0.1}']
            )
        self.assertEqual(json.loads(output.getvalue())["delta_used"], 0.1)

    def test_invalid_input_has_actionable_error(self):
        cases = [
            (["--runs", "0"], "at least 1"),
            (["--delta", "1.0"], "between 0 and 1"),
            (["--seed", "-1"], "--seed must be between"),
            (["--algo-args", "[]"], "JSON object"),
            (["--algo-args", '{"confidence":"bad"}'], "between 0 and 1"),
        ]
        for extra, message in cases:
            with self.subTest(arguments=extra):
                error = io.StringIO()
                with (
                    contextlib.redirect_stderr(error),
                    self.assertRaises(SystemExit) as raised,
                ):
                    cli_main(self.arguments() + extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, error.getvalue())

    def test_top_two_prints_only_json(self):
        output = io.StringIO()
        arguments = self.arguments()
        arguments[arguments.index("track_and_stop")] = "top_two"
        with contextlib.redirect_stdout(output):
            cli_main(
                arguments
                + [
                    "--algo-args",
                    '{"leader_type":"EB","challenger_type":"TC","beta":0.5}',
                ]
            )
        self.assertEqual(json.loads(output.getvalue())["true_best"], 1)

    def test_track_and_stop_requires_tracker(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            cli_main(
                [
                    "--env",
                    str(ENVIRONMENTS / "bernoulli.json"),
                    "--algo",
                    "track_and_stop",
                ]
            )
        self.assertIn("requires --tracker", error.getvalue())

    def test_module_entry_point_outside_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-m", "best_arm", *self.arguments()],
                cwd=temporary,
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            self.assertEqual(json.loads(completed.stdout)["delta_used"], 0.1)


class ExampleTests(unittest.TestCase):
    def run_example(self, name, temporary, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "examples" / name), *arguments],
            cwd=temporary,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )

    def test_examples_run_outside_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("quickstart.py", "gaussian.py", "repeated_runs.py"):
                with self.subTest(example=name):
                    completed = self.run_example(name, temporary)
                    self.assertTrue(completed.stdout.strip())

    def test_comparison_writes_summary_and_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_example(
                "compare_algorithms.py", temporary, "--runs", "2", "--horizon", "40"
            )
            output = Path(temporary) / "output"
            summary = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["runs"], 2)
            self.assertEqual(summary["horizon"], 40)
            self.assertEqual(
                set(summary["algorithms"]), {"track_and_stop", "ttskkt", "tspps"}
            )
            for result in summary["algorithms"].values():
                self.assertEqual(len(result["selected_arms"]), 2)
                self.assertAlmostEqual(sum(result["mean_final_allocation"]), 1.0)
            self.assertGreater((output / "allocations.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
