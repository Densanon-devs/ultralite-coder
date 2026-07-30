"""Bayesian calibration framework for the Phase 14 agentic bench (Task 6, 2026-05-19).

Sources:
- arXiv 2604.27082 — Bayesian framework for LLM end-of-life migration
- HuggingFace EvalEval blog 2026-04-29 — k=5 consistency band

Why this exists. The bench currently reports a single pass count (e.g.
"41/42 (97.6%)"). That's a point estimate. Two problems:

1. Variance is invisible. Was 41/42 a lucky run? A repeat-3 = 122/126 already
   exists in CI but doesn't surface a credible interval on the *true*
   pass rate the model would achieve in expectation.
2. Auto-grader bias is invisible. The bench's pass/fail check is code, not
   a human. If a tiny fraction of "passes" are actually wrong outputs that
   happen to satisfy the checker, the auto-grader's pass rate is an upper
   bound on the true pass rate, not an estimate.

This module provides:

- ``BetaBinomial`` — Beta(α + k, β + n-k) posterior on the true pass rate
  given k successes / n trials. Default prior Beta(1, 1) is uninformative;
  swap to a stronger prior when you have history.
- ``calibrate_results(...)`` — Load a benchmark_agentic.py JSON
  result file (or its by_task aggregate from --repeat N), produce per-task
  posteriors + a suite-level posterior.
- ``apply_human_labels(...)`` — Optional. Reconcile auto-grader passes
  against a small (~20-30) human-labeled subset; emit a calibrated
  pass-rate estimate that accounts for grader bias.

Pure-Python, no scipy dependency. The Beta CDF / inverse-CDF are
approximated via the regularized incomplete beta function ``I_x(a, b)``
computed by Lentz's continued-fraction algorithm + binary search for the
inverse — accurate to better than 1e-6 over realistic ranges. This is
overkill for a benchmark calibration utility but keeps the dependency
surface zero.

Output cap: 95% credible interval [q_lo, q_hi] where q_lo = I^{-1}(0.025)
and q_hi = I^{-1}(0.975). Mean of Beta(a, b) is a / (a + b).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Beta math (no scipy) ─────────────────────────────────────────────


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued-fraction expansion for the incomplete beta function.

    Adapted from Numerical Recipes' BETACF. Used inside ``regularized_beta``.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    """Return I_x(a, b), the regularized incomplete beta function.

    This is the CDF of Beta(a, b) at x. Domain: 0 <= x <= 1.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        -_log_beta(a, b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_inv_cdf(p: float, a: float, b: float) -> float:
    """Inverse CDF of Beta(a, b) at probability p via bisection.

    Used to compute credible interval endpoints. Tolerance 1e-6 on x.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):  # 2^-60 < 1e-6
        mid = 0.5 * (lo + hi)
        cdf = regularized_beta(mid, a, b)
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── Beta-Binomial posterior ─────────────────────────────────────────


@dataclass
class BetaBinomial:
    """Beta(alpha, beta) posterior on a Bernoulli rate."""

    alpha: float
    beta: float

    @classmethod
    def from_observations(
        cls, successes: int, trials: int, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> "BetaBinomial":
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError(
                f"invalid observation: {successes}/{trials}"
            )
        return cls(alpha=prior_alpha + successes, beta=prior_beta + trials - successes)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval at the given level (default 95%)."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = 0.5 * (1.0 - level)
        return (
            beta_inv_cdf(tail, self.alpha, self.beta),
            beta_inv_cdf(1.0 - tail, self.alpha, self.beta),
        )

    def as_dict(self, level: float = 0.95) -> dict:
        lo, hi = self.credible_interval(level)
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            f"ci{int(level * 100)}_lo": lo,
            f"ci{int(level * 100)}_hi": hi,
        }


# ── Benchmark-JSON adapters ─────────────────────────────────────────


@dataclass
class TaskResult:
    name: str
    passed: int
    trials: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.trials if self.trials else 0.0


def load_repeat_results(json_path: str | Path) -> list[TaskResult]:
    """Read benchmark_agentic.py --repeat N output JSON.

    The runner writes either a single-result document (no `by_task`) or a
    repeat-aggregated document with `by_task: {name: {pass_count, trial_count}}`.
    Both shapes are handled.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out: list[TaskResult] = []
    if "by_task" in data and isinstance(data["by_task"], dict):
        for name, stats in data["by_task"].items():
            passed = int(stats.get("pass_count", stats.get("passed", 0)))
            trials = int(stats.get("trial_count", stats.get("trials", 1)))
            out.append(TaskResult(name=name, passed=passed, trials=trials))
        return out
    # Fallback: flat task list with `passed` booleans
    results = data.get("results") or data.get("tasks") or []
    counters: dict[str, list[int]] = {}
    for r in results:
        name = r.get("name") or r.get("task")
        if name is None:
            continue
        counters.setdefault(name, [0, 0])
        counters[name][1] += 1
        if r.get("passed") or r.get("pass") is True:
            counters[name][0] += 1
    for name, (passed, trials) in counters.items():
        out.append(TaskResult(name=name, passed=passed, trials=trials))
    return out


def calibrate_results(
    results: list[TaskResult],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    level: float = 0.95,
) -> dict:
    """Produce per-task + suite-level posteriors on the pass rate.

    Suite-level posterior pools across tasks at the trial level (Beta-Binomial
    with shared prior). For a more realistic "task is a random effect"
    treatment use a hierarchical model — out of scope for v1.
    """
    per_task: dict[str, dict] = {}
    total_passed, total_trials = 0, 0
    for r in results:
        post = BetaBinomial.from_observations(
            r.passed, r.trials, prior_alpha, prior_beta
        )
        per_task[r.name] = {
            "passed": r.passed,
            "trials": r.trials,
            **post.as_dict(level),
        }
        total_passed += r.passed
        total_trials += r.trials
    suite = BetaBinomial.from_observations(
        total_passed, total_trials, prior_alpha, prior_beta
    )
    return {
        "tasks": per_task,
        "suite": {
            "passed": total_passed,
            "trials": total_trials,
            **suite.as_dict(level),
        },
        "prior": {"alpha": prior_alpha, "beta": prior_beta},
        "credible_level": level,
    }


# ── Human-label calibration ─────────────────────────────────────────


def apply_human_labels(
    results: list[TaskResult],
    human_labels: dict[str, dict],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    level: float = 0.95,
) -> dict:
    """Reconcile auto-grader passes against a hand-labeled subset.

    ``human_labels`` is a mapping ``{task_name: {"agree": int, "checked": int}}``.
    ``checked`` is how many trials were human-reviewed for that task; ``agree``
    is how many of those the human agreed with the auto-grader on. The
    calibrated pass rate is the auto-grader rate times the per-task agreement
    ratio, with both wrapped in their own Beta posteriors so the joint
    credible interval reflects label scarcity.

    Conservative model: treat auto-grader pass + human disagreement as a
    false positive. The calibrated estimate is
    p_true ≈ p_auto × p_agreement
    where p_agreement = Beta(α + agree, β + checked - agree).

    The resulting interval is wider than the auto-grader interval alone
    when labeled data is scarce — surfacing exactly the right uncertainty.
    """
    out_tasks: dict[str, dict] = {}
    for r in results:
        auto = BetaBinomial.from_observations(
            r.passed, r.trials, prior_alpha, prior_beta
        )
        hl = human_labels.get(r.name, {})
        agree = int(hl.get("agree", 0))
        checked = int(hl.get("checked", 0))
        if checked <= 0:
            # No human labels for this task — fall back to auto-grader.
            out_tasks[r.name] = {
                "auto": auto.as_dict(level),
                "calibrated": auto.as_dict(level),
                "human_label_count": 0,
            }
            continue
        agreement = BetaBinomial.from_observations(
            agree, checked, prior_alpha, prior_beta
        )
        # Posterior mean is multiplicative; for the interval we use Monte
        # Carlo since two Beta variables' product has no closed form.
        cal_mean = auto.mean * agreement.mean
        # Conservative CI: use the product of CI lower bounds and the
        # product of upper bounds. Wider than the true product distribution
        # but doesn't underclaim uncertainty.
        a_lo, a_hi = auto.credible_interval(level)
        ag_lo, ag_hi = agreement.credible_interval(level)
        out_tasks[r.name] = {
            "auto": auto.as_dict(level),
            "agreement": agreement.as_dict(level),
            "calibrated": {
                "mean": cal_mean,
                f"ci{int(level * 100)}_lo": a_lo * ag_lo,
                f"ci{int(level * 100)}_hi": a_hi * ag_hi,
            },
            "human_label_count": checked,
        }
    return {"tasks": out_tasks, "credible_level": level}


# ── CLI ─────────────────────────────────────────────────────────────


def _cli(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Bayesian calibration of benchmark_agentic.py results."
    )
    p.add_argument("results_json", help="benchmark_agentic.py output JSON")
    p.add_argument(
        "--human-labels", default=None,
        help="Optional JSON: {task_name: {agree: N, checked: M}}",
    )
    p.add_argument(
        "--prior-alpha", type=float, default=1.0,
        help="Prior alpha (default 1 = uniform)",
    )
    p.add_argument(
        "--prior-beta", type=float, default=1.0,
        help="Prior beta (default 1 = uniform)",
    )
    p.add_argument(
        "--level", type=float, default=0.95,
        help="Credible interval level (default 0.95)",
    )
    p.add_argument(
        "--out", default=None,
        help="Write the calibration JSON here (default: stdout)",
    )
    args = p.parse_args(argv)

    results = load_repeat_results(args.results_json)
    if not results:
        print("No results found in input JSON.")
        return 2

    summary = calibrate_results(
        results, args.prior_alpha, args.prior_beta, args.level
    )
    if args.human_labels:
        labels = json.loads(Path(args.human_labels).read_text(encoding="utf-8"))
        summary["calibrated"] = apply_human_labels(
            results, labels, args.prior_alpha, args.prior_beta, args.level
        )

    out_text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
