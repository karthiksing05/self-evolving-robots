"""A world judged on its score (D-093): the loop keeps what raises the points and loses no part past the noise."""

from __future__ import annotations

import json

from sea.evolve.core import Edit, Outcome, ProposalBatch
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, points_of, run
from sea.worlds.me2110 import KIT, VARIANTS, ME2110World
from tests.test_evolve import Reach, _p


class Scored(Reach):
    """The stick scores its length x 100 in a match, split into a part for reach and a part that long sticks lose."""

    objective = "points"
    points_noise = 5.0
    sprint_bars = (30.0, 60.0)
    max_points = 100.0

    def solve(self, body, task, seed=0):
        x = body.parts["stick"].params["length"]
        far, near = 100 * x, 20.0 if x <= 0.25 else 0.0
        broke = x > 0.29
        return Outcome(task=task.id, success=False, score=x, points=0.0 if broke else far + near,
                       states={"points.far": far, "points.near": near}, sim_s=0.01)


def _edit(x):
    return _p("params", [Edit(slot="stick", params={"length": x})])


def _go(tmp_path, proposals, iterations=1):
    prop = Scripted([ProposalBatch(proposals=proposals)])
    s = run(RunConfig(world="toy", iterations=iterations, merge=False), tmp_path, proposer=prop, world=Scored(),
            log=lambda m: None)
    return s, json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])["trials"], prop


def test_a_rise_is_kept_and_a_fall_is_not(tmp_path):
    s, t, prop = _go(tmp_path, [_edit(0.15), _edit(0.22)])
    assert t[0]["result"].startswith("rejected: 40.0 -> 35.0 points") and not t[0]["eligible"]
    assert t[1]["result"].startswith("KEPT: 40.0 -> 42.0 points") and t[1]["points_gain"] == 2.0
    assert s["curve"][0]["points"] == 40.0 and s["curve"][1]["points"] == 42.0
    assert s["points"]["train"] == 42.0 and s["points"]["test"] == 42.0 and s["points"]["bars_met_test"] == [30.0]
    assert "The current body scores 40.0 of 100 points in a match (the brief's bars: 30, 60)" in prop.prompts[0]


def test_a_rise_that_loses_a_part_goes_on_the_frontier(tmp_path):
    s, t, _ = _go(tmp_path, [_edit(0.27)])  # 27 far + 0 near = 27 falls; the toy's rise with a part lost needs more
    assert not t[0]["eligible"]
    w = Scored()
    w.solve = lambda body, task, seed=0: Outcome(  # a total that rises while one part falls by more than the noise
        task=task.id, success=False, points=50.0 if body.parts["stick"].params["length"] != 0.2 else 40.0,
        states={"points.a": 50.0, "points.b": 0.0} if body.parts["stick"].params["length"] != 0.2
        else {"points.a": 30.0, "points.b": 10.0})
    run(RunConfig(world="toy", iterations=1, merge=False), tmp_path / "f", proposer=Scripted([ProposalBatch(
        proposals=[_edit(0.3)])]), world=w, log=lambda m: None)
    t = json.loads((tmp_path / "f" / "iterations.jsonl").read_text().splitlines()[0])["trials"]
    assert "points.b dropped more than 5" in t[0]["result"] and "kept on the frontier" in t[0]["result"]


def test_a_broken_rule_scores_nothing(tmp_path):
    _, t, _ = _go(tmp_path, [_edit(0.3)])
    assert t[0]["points"] == 0.0 and not t[0]["eligible"]


def test_points_are_means_over_the_tasks_that_carry_them():
    tasks = [t for t in Reach().tasks() if t.split == "test"]
    out = {t.id: Outcome(task=t.id, success=False, points=float(k), states={"points.a": float(k)})
           for k, t in enumerate(tasks[:2])}
    out[tasks[2].id] = Outcome(task=tasks[2].id, success=False)  # a task that carries no points does not dilute them
    assert points_of(tasks, out) == (0.5, {"points.a": 0.5})


def test_me2110_has_the_brief_s_kit_tracks_and_objective():
    w = ME2110World()
    assert KIT["mousetraps"] == 4 and w.objective == "points" and w.sprint_bars == (60.0, 150.0, 215.0) and w.max_points == 347
    assert {v.barrier for v in VARIANTS["test"]} == {6.0, 7.0, 8.0, 8.75}
    assert {v.rpm for v in VARIANTS["test"]} >= {2.0, 5.0} and {v.mu for v in VARIANTS["test"]} >= {0.4, 0.6}
    t = next(t for t in w.tasks() if t.id == "match_test3")
    assert t.data["barrier_in"] == 8.0 and t.data["rpm"] == 3.5 and t.data["mu"] == 0.6
