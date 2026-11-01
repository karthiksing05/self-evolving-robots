"""Coverage metrics, the reward, and the standalone reward audit (spec 8.2, 8.3)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from sea.contracts import (
    LAMBDA_MASS,
    LAMBDA_PRINT,
    BuildEstimate,
    ControlResult,
    CostVector,
    Design,
    HardLimits,
    RunSpec,
    Vec3,
)
from sea.verify import coverage, reward
from sea.verify.coverage import VERDICTS


def res(verdict: str, cls: str = "push_button", cid: str = "c0", panel: str = "p0") -> ControlResult:
    return ControlResult(panel_id=panel, control_id=cid, cls=cls, verdict=verdict, peak_force=1.0,
                         travel_reached=0.5, detents=0, time_s=1.0)


def metrics(results):
    return coverage.control_coverage(results), coverage.pass_rate(results), coverage.coverage_reward(results)


# ---------------------------------------------------------------- hand-computed examples
def test_zero_controls():
    assert metrics([]) == (0.0, 0.0, 0.0)


def test_all_pass():
    assert metrics([res("pass")] * 4) == (1.0, 1.0, 1.0)


def test_all_inconclusive():
    assert metrics([res("inconclusive")] * 3) == (0.0, 0.0, 0.0)


def test_mixed():
    rs = [res("pass"), res("pass"), res("stuck"), res("inconclusive"), res("no_applicable_primitive")]
    cov, pr, cr = metrics(rs)
    assert cov == 3 / 5
    assert pr == 2 / 3
    assert cr == pytest.approx(cov * pr)
    assert cr == 2 / 5


@given(st.lists(st.sampled_from(VERDICTS), max_size=40))
def test_coverage_reward_is_passes_over_n(verdicts):
    rs = [res(v) for v in verdicts]
    cr = coverage.coverage_reward(rs)
    assert 0.0 <= cr <= 1.0
    assert cr == (verdicts.count("pass") / len(verdicts) if verdicts else 0.0)
    assert cr == pytest.approx(coverage.control_coverage(rs) * coverage.pass_rate(rs))


# ---------------------------------------------------------------- reward
def test_reward_terms():
    assert coverage.reward(0.8, 60.0, 240.0, 20.0, 40.0) == pytest.approx(0.8 - LAMBDA_PRINT * 0.25 - LAMBDA_MASS * 0.5)
    assert coverage.reward(0.8, 60.0, 240.0, 20.0, 40.0, cost_terms=False) == 0.8
    assert coverage.reward(0.8, 60.0, 0.0, 20.0, 0.0) == 0.8  # zero budgets drop their terms
    assert coverage.reward(0.0, 480.0, 240.0, 40.0, 40.0) < 0.0  # reward can be negative


def test_summarize():
    rs = [res("pass", "push_button"), res("stuck", "push_button", "c1"), res("pass", "door", "c2"),
          res("inconclusive", "knob", "c3")]
    s = coverage.summarize(rs)
    assert (s["n"], s["covered"], s["passes"]) == (4, 3, 2)
    assert s["coverage_reward"] == 0.5
    assert s["by_verdict"]["pass"] == 2 and s["by_verdict"]["unreachable"] == 0
    assert set(s["by_verdict"]) == set(VERDICTS)
    assert s["by_class"]["push_button"] == {"n": 2, "passes": 1, "by_verdict": {"pass": 1, "stuck": 1}}
    json.dumps(s)


# ---------------------------------------------------------------- standalone audit
LIMITS = HardLimits(workspace_box=(Vec3(x=0.1, y=-0.2, z=-0.1), Vec3(x=0.4, y=0.2, z=0.25)))


def _write_run(tmp: Path, cost_terms: bool = True, final_design: bool = True) -> RunSpec:
    spec = RunSpec(seed=1, limits=LIMITS)
    spec = spec.model_copy(update={"ablations": spec.ablations.model_copy(update={"cost_terms": cost_terms})})
    (tmp / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json")))
    d0 = Design(id="d0")
    d2 = Design(id="d2", parent="d0", est=BuildEstimate(print_minutes=30.0, added_mass_g=12.0))
    (tmp / "design_000.json").write_text(d0.model_dump_json())
    (tmp / "design_002.json").write_text(d2.model_dump_json())
    verdict_sets = [["pass", "stuck", "inconclusive"], ["pass", "pass", "stuck"], ["pass", "pass", "pass"]]
    for k, vs in enumerate(verdict_sets):
        d = tmp / f"iter_{k:03d}"
        d.mkdir()
        (d / "results.jsonl").write_text("\n".join(res(v, cid=f"c{i}").model_dump_json() for i, v in enumerate(vs)) + "\n")
        (d / "cost.json").write_text(json.dumps({"cumulative": CostVector(print_minutes=15.0 * k).model_dump(),
                                                 "iteration": CostVector().model_dump()}))
    (tmp / "iter_notes").mkdir()  # not an iteration directory
    fin = tmp / "final"
    fin.mkdir()
    (fin / "results.jsonl").write_text(res("pass").model_dump_json() + "\n" + res("unreachable", cid="c1").model_dump_json())
    (fin / "cost.json").write_text(json.dumps({"cumulative": CostVector(print_minutes=40.0).model_dump()}))
    if final_design:
        (fin / "design.json").write_text(d2.model_copy(update={"id": "dfinal"}).model_dump_json())
    return spec


def test_reward_main_recomputes_every_iteration(tmp_path):
    spec = _write_run(tmp_path)
    out = reward.main(tmp_path)
    assert [r["iteration"] for r in out["iterations"]] == [0, 1, 2]
    it0, it1, it2 = out["iterations"]
    assert it0["coverage_reward"] == 1 / 3 and it0["control_coverage"] == 2 / 3 and it0["pass_rate"] == 1 / 2
    assert it0["reward"] == coverage.reward(1 / 3, 0.0, spec.budgets.print_minutes, 0.0, 40.0)
    assert it1["reward"] == coverage.reward(2 / 3, 15.0, spec.budgets.print_minutes, 0.0, 40.0)  # design_000 still
    assert it2["reward"] == coverage.reward(1.0, 30.0, spec.budgets.print_minutes, 12.0, 40.0)  # design_002
    assert out["final"]["coverage_reward"] == 0.5
    assert out["final"]["reward"] == coverage.reward(0.5, 40.0, spec.budgets.print_minutes, 12.0, 40.0)


def test_reward_main_ablation_and_missing_final_design(tmp_path):
    _write_run(tmp_path, cost_terms=False, final_design=False)
    out = reward.main(str(tmp_path))
    assert [r["reward"] for r in out["iterations"]] == [r["coverage_reward"] for r in out["iterations"]]
    assert out["final"]["reward"] == 0.5


def test_reward_main_without_final_or_designs(tmp_path):
    spec = RunSpec(seed=1, limits=LIMITS)
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json")))
    (tmp_path / "iter_000").mkdir()  # an iteration that wrote nothing yet
    out = reward.main(tmp_path)
    assert out["final"] is None
    assert out["iterations"] == [{"iteration": 0, "control_coverage": 0.0, "pass_rate": 0.0,
                                  "coverage_reward": 0.0, "reward": 0.0}]


def test_reward_module_imports_only_contracts_and_coverage():
    tree = ast.parse(Path(reward.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    sea_mods = {m for m in mods if m.startswith("sea")}
    assert sea_mods <= {"sea.contracts", "sea.verify"}
    from_verify = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "sea.verify"
                   for a in n.names}
    assert from_verify <= {"coverage"}
