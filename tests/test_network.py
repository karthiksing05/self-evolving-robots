"""The task network (docs/goals.md, D-037): the goal vocabulary, the contracts, and the loop as a network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest
import yaml

from sea.compose.checks import CheckSyntaxError
from sea.compose.library import Library
from sea.contracts import Budgets, CostVector, Decision, Prediction, Proposal, RunSpec, SensorSuite
from sea.contracts import Design as BodyDesign
from sea.loop import evolve
from sea.plan import network as N
from sea.plan.network import Goal, GoalLog, Task
from sea.sim.sensors import mounted_parts
from sea.verify import goals as G

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- the vocabulary (judge tier)
def test_goal_tests_parse_from_a_fixed_vocabulary():
    atoms = G.parse("pass_share(train, 0.8) & still_passes(reference)")
    assert G.pretty(atoms) == "pass_share(train, 0.8) & still_passes(reference)"
    assert G.sets(atoms) == {"train", "reference"} and G.pretty(G.parse("accepted()")) == "accepted()"
    for bad in ("coverage(train)", "pass_share(train, 1.5)", "accepted() x", "still_passes(train", "pass_share(1, 1)"):
        with pytest.raises(CheckSyntaxError):
            G.parse(bad)


def test_goal_tests_read_only_what_the_judge_decided():
    ev = G.Evidence(members={"a": ["p:1", "p:2", "p:3", "p:4"], "none": []},
                    verdicts={"p:1": "pass", "p:2": "pass", "p:3": "pass", "p:4": "stuck"},
                    before={"p:1": "pass", "p:4": "pass"})
    assert G.holds("pass_share(a, 0.75)", ev) and not G.holds("pass_share(a, 0.8)", ev)
    assert not G.holds("still_passes(a)", ev)  # p:4 passed before and fails now
    assert G.holds("still_passes(a)", G.Evidence(members=ev.members, verdicts=ev.verdicts, before={"p:1": "pass"}))
    assert not G.holds("pass_share(none, 0.5)", ev) and G.holds("pass_share(none, 0)", ev)
    assert not G.holds("accepted()", ev)  # no trial ran
    assert G.holds("accepted() & pass_share(a, 0.5)", G.Evidence(members=ev.members, verdicts=ev.verdicts, accepted=True))
    with pytest.raises(KeyError):
        G.holds("pass_share(b, 0.5)", ev)


# ---------------------------------------------------------------- the contracts (policy tier)
SLICE = CostVector(sim_seconds=600, llm_usd=2.0, print_minutes=60)


def _net(**changes: Goal) -> list[Goal]:
    goals = {
        "root": Goal(id="root", test="pass_share(train, 1)", task=Task(kind="network", method="coverage"), slice=SLICE),
        "p:c": Goal(id="p:c", parent="root", test="moved_at_least($target, travel)", target="p:c",
                    task=Task(kind="skill", program="press_button")),
        "why": Goal(id="why", parent="root", test="pass_share(train, 1)", serves=("p:c",),
                    task=Task(kind="analyze", question="does the pad slip?", needs=("jaw_pads",)),
                    slice=CostVector(sim_seconds=60)),
        "fix": Goal(id="fix", parent="root", test="accepted() & still_passes(train)", serves=("p:c",),
                    task=Task(kind="design", change="finger"), slice=CostVector(sim_seconds=300, print_minutes=60)),
    }
    goals.update(changes)
    return list(goals.values())


def _errors(goals: list[Goal], pads: bool = True) -> list[str]:
    body = BodyDesign(id="d", sensors=SensorSuite(jaw_pads=pads))
    return N.validate(goals, mounted=mounted_parts(body), programs=set(Library.seed().prims), sets={"train"})


def test_a_well_formed_network_validates_and_slices_are_shared_down():
    goals = _net()
    assert _errors(goals) == []
    by_id = {g.id: g for g in goals}
    assert N.effective_slice("p:c", by_id) == SLICE  # no slice of its own: it draws on the root's
    assert N.fits(N.cost_sum([by_id["why"].slice, by_id["fix"].slice]), SLICE)


@pytest.mark.parametrize("changes, pads, expect", [
    ({"x": Goal(id="x", test="accepted()", task=Task(kind="design"), slice=SLICE)}, True, "exactly one root"),
    ({"x": Goal(id="x", parent="nobody", test="accepted()", task=Task(kind="design"))}, True, "unknown parent"),
    ({"x": Goal(id="x", parent="root", test="accepted()", serves=("ghost",), task=Task(kind="design"))}, True,
     "serves unknown goal"),
    ({"x": Goal(id="x", parent="root", test="i_am_done()", task=Task(kind="analyze"))}, True, "vocabulary"),
    ({"x": Goal(id="x", parent="root", test="moved_at_least($target, travel)", task=Task(kind="analyze"))}, True,
     "vocabulary"),  # a control check with no control is not a test
    ({"x": Goal(id="x", parent="root", test="still_passes(regression)", task=Task(kind="analyze"))}, True,
     "unknown sets"),
    ({"x": Goal(id="x", parent="root", test="accepted()", target="p:c", task=Task(kind="skill", program="fly"))},
     True, "vocabulary"),
    ({"x": Goal(id="x", parent="root", test="held_for($target, 1.0)", target="p:c", task=Task(kind="skill",
                                                                                              program="fly"))},
     True, "not in the library"),
    ({}, False, "unmounted sensors jaw_pads"),
    ({"root": Goal(id="root", test="pass_share(train, 1)", task=Task(kind="network"))}, True, "needs a slice"),
    ({"fix": Goal(id="fix", parent="root", test="accepted()", task=Task(kind="design"),
                  slice=CostVector(print_minutes=90))}, True, "slices exceed"),
    ({"a": Goal(id="a", parent="b", test="accepted()", task=Task(kind="design")),
      "b": Goal(id="b", parent="a", test="accepted()", task=Task(kind="design"))}, True, "cycle"),
])
def test_validation_names_what_would_stop_dispatch(changes, pads, expect):
    errors = _errors(_net(**changes), pads=pads)
    assert any(expect in e for e in errors), errors


def test_duplicate_ids_are_refused():
    goals = _net()
    assert any("duplicate id" in e for e in _errors([*goals, goals[1]]))


def test_every_impasse_has_a_sub_goal_rule_and_slices_are_the_ledger_budgets():
    assert set(N.IMPASSE_SUBGOAL) == set(get_args(N.Impasse))
    assert N.IMPASSE_SUBGOAL["structural"] == "design" and N.IMPASSE_SUBGOAL["reflex"] == "analyze"
    assert N.IMPASSE_SUBGOAL["slice_exhausted"] is None
    s = N.run_slice(Budgets(sim_hours=10, llm_usd=20, reserve_fraction=0.2))
    assert s.sim_seconds == pytest.approx(10 * 3600 * 0.8) and s.llm_usd == pytest.approx(16.0)
    assert N.overspent(s, CostVector(llm_usd=17.0, servo_heat_j=1e6)) == ["llm_usd"]  # heat is priced, not sliced
    assert N.cost_diff(CostVector(llm_usd=3), CostVector(llm_usd=1)).llm_usd == 2


def test_the_log_replays_and_a_one_shot_node_closes_on_the_acceptance_rule(tmp_path: Path):
    from sea.sim.panels import reference_panel

    log = GoalLog(tmp_path / "goals.jsonl")
    net = N.DegenerateNetwork(log, Budgets(), [reference_panel()])
    prop = Proposal(kind="new_tool", rung=3, rationale="r", hypothesis="a hook reaches the tab",
                    prediction=Prediction(confidence=0.5))
    yes = Decision(accepted=True, rule="mean", delta_mean=0.2, delta_lo=0.1, delta_hi=0.3)
    net.one_shot(0, ["ref:c1"], prop, yes, CostVector(llm_usd=0.4))
    net.one_shot(1, ["ref:c1"], None, None, CostVector())
    assert net.close(1, "iterations") == "exhausted"  # nothing was served, so the root's test fails
    goals, reports, status = N.replay(log.path)
    assert status == {"root": "exhausted", "s2.000": "passed", "s2.001": "failed"}
    assert goals["s2.000"].task.change == "new_tool" and goals["s2.000"].why == "a hook reaches the tab"
    assert [r.verdict for r in reports] == ["pass", "fail"] and reports[0].spent.llm_usd == 0.4


# ---------------------------------------------------------------- the loop as a network
@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    base = yaml.safe_load((ROOT / "configs" / "exp" / "smoke.yaml").read_text())
    base.update({"iter_sample": 2, "splits": {"train": 6, "val": 2, "test": 2, "compose": 2}, "s1_trials": 1,
                 "trial_repeats": 1, "budgets": {"design_iterations": 2, "sim_hours": 40},
                 "ablations": {"reference_regression": False}})
    out = tmp_path_factory.mktemp("net") / "run"
    evolve(RunSpec.model_validate(base), out, log=lambda s: None)
    return out


def test_the_loop_is_recorded_as_the_degenerate_network(run: Path):
    goals, reports, status = N.replay(run / "goals.jsonl")
    assert N.validate(list(goals.values()), programs=set(Library.seed().prims), sets={"train"}) == []
    root = goals["root"]
    assert root.test == "pass_share(train, 1)" and status["root"] in ("passed", "exhausted")
    s2 = [g for g in goals.values() if g.task.kind == "design"]
    assert s2 and all(g.test == "accepted()" and g.serves for g in s2)  # System 2 answered an impasse
    for it_dir in sorted(run.glob("iter_*")):
        it = int(it_dir.name[-3:])
        results = [json.loads(line) for line in (it_dir / "results.jsonl").read_text().splitlines()]
        mine = {r.goal: r for r in reports if r.iteration == it and r.goal != f"s2.{it:03d}"}
        assert {f"{r['panel_id']}:{r['control_id']}": r["verdict"] for r in results} == {
            k: r.verdict for k, r in mine.items()}  # one leaf report per control System 1 served
        assert all(goals[k].parent == "root" and goals[k].task.kind == "skill" for k in mine)
        for r in results:
            if r["escalated"]:
                assert mine[f"{r['panel_id']}:{r['control_id']}"].impasse == r["escalation_reason"]
        sim = json.loads((it_dir / "cost.json").read_text())["iteration"]["sim_seconds"]
        assert sum(r.spent.sim_seconds for r in mine.values()) == pytest.approx(sim, abs=1e-2)
        decision = json.loads((it_dir / "decision.json").read_text())
        if f"s2.{it:03d}" in goals:
            assert status[f"s2.{it:03d}"] == ("passed" if decision and decision["accepted"] else "failed")
