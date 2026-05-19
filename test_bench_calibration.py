"""Tests for engine/bench_calibration.py (Task 6, 2026-05-19)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.bench_calibration import (
    BetaBinomial,
    TaskResult,
    apply_human_labels,
    beta_inv_cdf,
    calibrate_results,
    load_repeat_results,
    regularized_beta,
)


class TestBetaMath(unittest.TestCase):
    """Sanity-check the no-scipy Beta functions against known values."""

    def test_regularized_beta_uniform(self):
        # I_x(1, 1) == x (Beta(1,1) is uniform, so its CDF is identity)
        for x in (0.1, 0.25, 0.5, 0.75, 0.9):
            self.assertAlmostEqual(regularized_beta(x, 1.0, 1.0), x, places=5)

    def test_regularized_beta_endpoints(self):
        self.assertEqual(regularized_beta(0.0, 2.0, 3.0), 0.0)
        self.assertEqual(regularized_beta(1.0, 2.0, 3.0), 1.0)

    def test_regularized_beta_known(self):
        # I_0.5(2, 5) ≈ 0.890625 (closed-form for small ints)
        self.assertAlmostEqual(
            regularized_beta(0.5, 2.0, 5.0), 0.890625, places=4
        )

    def test_inverse_round_trip(self):
        for a in (1.0, 2.0, 5.0):
            for b in (1.0, 2.0, 5.0):
                for q in (0.025, 0.5, 0.975):
                    x = beta_inv_cdf(q, a, b)
                    p = regularized_beta(x, a, b)
                    self.assertAlmostEqual(p, q, places=4)


class TestBetaBinomial(unittest.TestCase):
    def test_uniform_prior_mean(self):
        # 42/42 with uniform prior -> Beta(43, 1), mean 43/44 = 0.977...
        post = BetaBinomial.from_observations(42, 42)
        self.assertAlmostEqual(post.mean, 43 / 44, places=4)

    def test_credible_interval_width_shrinks_with_n(self):
        # All-pass at small n -> wide CI; all-pass at large n -> narrow CI
        small = BetaBinomial.from_observations(3, 3)
        large = BetaBinomial.from_observations(100, 100)
        lo_s, hi_s = small.credible_interval(0.95)
        lo_l, hi_l = large.credible_interval(0.95)
        self.assertGreater(hi_s - lo_s, hi_l - lo_l)
        # CI lower bound should be much higher with more trials
        self.assertLess(lo_s, lo_l)

    def test_invalid_observations(self):
        with self.assertRaises(ValueError):
            BetaBinomial.from_observations(5, 3)
        with self.assertRaises(ValueError):
            BetaBinomial.from_observations(-1, 3)

    def test_zero_passes_gives_low_estimate(self):
        post = BetaBinomial.from_observations(0, 10)
        # Beta(1, 11), mean = 1/12 ≈ 0.083
        self.assertAlmostEqual(post.mean, 1 / 12, places=4)
        lo, hi = post.credible_interval(0.95)
        self.assertLess(lo, 0.01)
        self.assertLess(hi, 0.3)

    def test_as_dict_keys(self):
        post = BetaBinomial.from_observations(40, 42)
        d = post.as_dict(0.95)
        self.assertIn("alpha", d)
        self.assertIn("beta", d)
        self.assertIn("mean", d)
        self.assertIn("ci95_lo", d)
        self.assertIn("ci95_hi", d)
        self.assertLessEqual(d["ci95_lo"], d["mean"])
        self.assertLessEqual(d["mean"], d["ci95_hi"])


class TestCalibrateResults(unittest.TestCase):
    def test_per_task_and_suite(self):
        results = [
            TaskResult("t1", 3, 3),
            TaskResult("t2", 2, 3),
            TaskResult("t3", 3, 3),
        ]
        out = calibrate_results(results)
        self.assertIn("tasks", out)
        self.assertIn("suite", out)
        self.assertEqual(set(out["tasks"]), {"t1", "t2", "t3"})
        suite = out["suite"]
        self.assertEqual(suite["passed"], 8)
        self.assertEqual(suite["trials"], 9)
        self.assertLessEqual(suite["ci95_lo"], suite["mean"])
        self.assertLessEqual(suite["mean"], suite["ci95_hi"])

    def test_prior_recorded(self):
        out = calibrate_results([TaskResult("t1", 1, 1)], prior_alpha=2, prior_beta=8)
        self.assertEqual(out["prior"], {"alpha": 2, "beta": 8})


class TestLoadRepeatResults(unittest.TestCase):
    def test_by_task_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            p.write_text(json.dumps({
                "by_task": {
                    "task_a": {"pass_count": 5, "trial_count": 5},
                    "task_b": {"pass_count": 3, "trial_count": 5},
                }
            }))
            results = load_repeat_results(p)
        self.assertEqual(len(results), 2)
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["task_a"].passed, 5)
        self.assertEqual(by_name["task_a"].trials, 5)
        self.assertEqual(by_name["task_b"].passed, 3)
        self.assertEqual(by_name["task_b"].trials, 5)

    def test_flat_results_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            p.write_text(json.dumps({
                "results": [
                    {"name": "t1", "passed": True},
                    {"name": "t1", "passed": False},
                    {"name": "t2", "passed": True},
                ]
            }))
            results = load_repeat_results(p)
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["t1"].passed, 1)
        self.assertEqual(by_name["t1"].trials, 2)
        self.assertEqual(by_name["t2"].passed, 1)
        self.assertEqual(by_name["t2"].trials, 1)


class TestHumanLabelCalibration(unittest.TestCase):
    def test_no_labels_falls_back_to_auto(self):
        results = [TaskResult("t1", 5, 5)]
        out = apply_human_labels(results, human_labels={})
        self.assertEqual(out["tasks"]["t1"]["human_label_count"], 0)
        self.assertEqual(
            out["tasks"]["t1"]["auto"]["mean"],
            out["tasks"]["t1"]["calibrated"]["mean"],
        )

    def test_perfect_agreement_matches_auto(self):
        results = [TaskResult("t1", 5, 5)]
        labels = {"t1": {"agree": 3, "checked": 3}}
        out = apply_human_labels(results, labels)
        # Perfect agreement on 3/3 -> agreement posterior mean = 4/5 (Beta(4,1))
        # Calibrated = auto.mean * 0.8 (NOT 1.0 — Bayesian shrinks toward prior)
        auto_mean = out["tasks"]["t1"]["auto"]["mean"]
        cal_mean = out["tasks"]["t1"]["calibrated"]["mean"]
        # Calibrated should be lower than auto when human-label count is small
        self.assertLess(cal_mean, auto_mean)

    def test_disagreement_lowers_estimate(self):
        # Auto says 5/5; human says auto was wrong on 2 of 3 checked
        results = [TaskResult("t1", 5, 5)]
        labels_high = {"t1": {"agree": 3, "checked": 3}}  # full agreement
        labels_low = {"t1": {"agree": 1, "checked": 3}}   # mostly disagreement
        out_high = apply_human_labels(results, labels_high)
        out_low = apply_human_labels(results, labels_low)
        self.assertLess(
            out_low["tasks"]["t1"]["calibrated"]["mean"],
            out_high["tasks"]["t1"]["calibrated"]["mean"],
        )

    def test_label_count_recorded(self):
        results = [TaskResult("t1", 5, 5)]
        labels = {"t1": {"agree": 3, "checked": 5}}
        out = apply_human_labels(results, labels)
        self.assertEqual(out["tasks"]["t1"]["human_label_count"], 5)


class TestCLI(unittest.TestCase):
    def test_cli_runs_clean(self):
        from engine.bench_calibration import _cli
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.json"
            results_path.write_text(json.dumps({
                "by_task": {
                    "task_a": {"pass_count": 5, "trial_count": 5},
                    "task_b": {"pass_count": 4, "trial_count": 5},
                }
            }))
            out_path = Path(tmp) / "out.json"
            rc = _cli([str(results_path), "--out", str(out_path)])
            self.assertEqual(rc, 0)
            summary = json.loads(out_path.read_text())
            self.assertEqual(summary["suite"]["passed"], 9)
            self.assertEqual(summary["suite"]["trials"], 10)


if __name__ == "__main__":
    unittest.main()
