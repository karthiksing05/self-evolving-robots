"""`sea reward` recomputes the in-loop reward from the logs alone (spec 8.2, 12.2)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sea.contracts import Report, RunSpec
from sea.loop import evolve
from sea.verify.reward import main as reward_main

ROOT = Path(__file__).resolve().parents[1]


def tiny_spec(**kw) -> RunSpec:
    """A two-iteration offline run small enough for the fast suite."""
    base = yaml.safe_load((ROOT / "configs" / "exp" / "smoke.yaml").read_text())
    base.update({"iter_sample": 2, "splits": {"train": 6, "val": 2, "test": 2, "compose": 2},
                 "s1_trials": 1, "trial_repeats": 1,
                 "budgets": {"design_iterations": 2, "sim_hours": 40},
                 "ablations": {"reference_regression": False}})
    base.update(kw)
    return RunSpec.model_validate(base)


def test_reward_recompute_matches_loop(tmp_path: Path) -> None:
    run = tmp_path / "audit"
    final = evolve(tiny_spec(), run, log=lambda s: None)
    recomputed = reward_main(run)
    iters = sorted(run.glob("iter_*/report.json"))
    assert len(iters) == len(recomputed["iterations"]) >= 1
    for path, rec in zip(iters, recomputed["iterations"], strict=True):
        rep = Report.model_validate_json(path.read_text())
        assert rec["iteration"] == rep.iteration
        assert rec["coverage_reward"] == rep.coverage_reward
        assert rec["reward"] == rep.reward  # exact float equality on the JSON values
    assert recomputed["final"]["coverage_reward"] == final["coverage_reward"]
    assert recomputed["final"]["reward"] == final["reward"]
    for name in ("spec.yaml", "harness.json", "design_000.json", "library_000.json", "timing.json",
                 "final/report.json", "final/summary.json"):
        assert (run / name).exists(), name
    for it in iters:
        for name in ("results.jsonl", "cost.json", "proposal.json", "decision.json"):
            assert (it.parent / name).exists(), f"{it.parent.name}/{name}"
    assert "timing" not in json.loads((run / "iter_000" / "report.json").read_text())
