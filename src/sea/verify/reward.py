"""Standalone reward audit (spec 8.2): ``main(run_dir) -> dict``.

Recomputes ``coverage_reward`` and ``reward`` from the run's files alone:

    run_dir/spec.yaml                    RunSpec
    run_dir/design_<k>.json              Design in effect from iteration k on
    run_dir/iter_<k>/results.jsonl       one ControlResult per line
    run_dir/iter_<k>/cost.json           {"cumulative": CostVector, ...}
    run_dir/final/{results.jsonl, cost.json, design.json}   (optional)

A design file need not exist for every iteration: iteration k uses the newest ``design_<j>``
with j <= k.  This module imports nothing from ``sea`` except ``contracts`` and
``verify.coverage``; ``python -m sea.verify.reward runs/<id>`` prints the result as JSON.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from sea.contracts import ControlResult, CostVector, Design, RunSpec
from sea.verify import coverage

_ITER = re.compile(r"^iter_(\d+)$")
_DESIGN = re.compile(r"^design_(\d+)\.json$")


def _results(path: Path) -> list[ControlResult]:
    lines = path.read_text().splitlines() if path.exists() else []
    return [ControlResult.model_validate_json(line) for line in lines if line.strip()]


def _cumulative(path: Path) -> CostVector:
    data = json.loads(path.read_text()) if path.exists() else {}
    return CostVector.model_validate(data.get("cumulative", {}))


def _designs(run_dir: Path) -> dict[int, Design]:
    out = {}
    for p in run_dir.iterdir():
        m = _DESIGN.match(p.name)
        if m:
            out[int(m.group(1))] = Design.model_validate_json(p.read_text())
    return out


def _design_at(designs: dict[int, Design], k: int) -> Design | None:
    keys = [j for j in designs if j <= k]
    return designs[max(keys)] if keys else None


def score(spec: RunSpec, results: list[ControlResult], cumulative: CostVector, design: Design | None) -> dict:
    cov = coverage.coverage_reward(results)
    mass = design.est.added_mass_g if design is not None else 0.0
    return {
        "control_coverage": coverage.control_coverage(results),
        "pass_rate": coverage.pass_rate(results),
        "coverage_reward": cov,
        "reward": coverage.reward(cov, cumulative.print_minutes, spec.budgets.print_minutes, mass,
                                  spec.limits.max_tool_mass_g, spec.ablations.cost_terms),
    }


def main(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    spec = RunSpec.model_validate(yaml.safe_load((run_dir / "spec.yaml").read_text()))
    designs = _designs(run_dir)
    iters = sorted((int(m.group(1)), p) for p in run_dir.iterdir() if (m := _ITER.match(p.name)) and p.is_dir())
    out: dict = {"iterations": [], "final": None}
    for k, d in iters:
        row = score(spec, _results(d / "results.jsonl"), _cumulative(d / "cost.json"), _design_at(designs, k))
        out["iterations"].append({"iteration": k, **row})
    final = run_dir / "final"
    if (final / "results.jsonl").exists():
        design = (Design.model_validate_json((final / "design.json").read_text())
                  if (final / "design.json").exists() else _design_at(designs, max(designs, default=0)))
        out["final"] = score(spec, _results(final / "results.jsonl"), _cumulative(final / "cost.json"), design)
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=2))
