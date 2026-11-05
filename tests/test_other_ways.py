"""A failure is not charged to a mechanism until it has been used in the other ways it can be (D-102)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sea.compose.library import Library
from sea.control import local_search as LS
from sea.control.executor import HOME_Q, stock_ik
from sea.control.motor import grasp_poses, held_way
from sea.evolve.loop import make_world
from sea.sim import env
from sea.worlds import gripper as G

DOWN, ROLL, POINT = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([0.2, 0.0, 0.06])


def test_a_way_of_holding_moves_what_it_says():
    point, down, roll = held_way(POINT, DOWN, ROLL)
    assert np.allclose(point, POINT) and np.allclose(down, DOWN) and np.allclose(roll, ROLL)
    point, down, roll = held_way(POINT, DOWN, ROLL, along=0.03)
    assert np.isclose(np.linalg.norm(point - POINT), 0.03) and abs(float((point - POINT) @ ROLL)) < 1e-12
    _, down, roll = held_way(POINT, DOWN, ROLL, yaw=0.2)
    assert np.allclose(down, DOWN) and np.isclose(float(roll @ ROLL), np.cos(0.2))
    _, down, roll = held_way(POINT, DOWN, ROLL, lean=0.2)
    assert np.allclose(roll, ROLL) and np.isclose(float(down @ DOWN), np.cos(0.2))


def test_the_grasp_offers_other_ways_and_a_button_press_does_not():
    lib = Library.seed().prims
    assert len(LS.modes_of(lib["pick_up"].body, lib)) == len(lib["grasp"].modes) > 0
    others = [p for p in lib.values() if p.kind == "composite" and p.name != "pick_up"]
    assert others and not any(LS.modes_of(p.body, lib) for p in others)


def test_every_way_offered_is_a_pose_the_arm_can_hold():
    """A pose the 5-joint arm cannot reach reads as a new orientation while being only an offset (D-102)."""
    world = make_world("gripper")
    fingers = G.to_design(world.stock()).fingers
    ik = stock_ik()
    for way in G._ways():
        for spec, contact, down, roll in G._grasp_scenes():
            point, d, r = held_way(contact, down, roll, **way)
            target, *_ = grasp_poses(spec, fingers, point, d, r)
            _, res, ang = ik.track(target, d, HOME_Q.copy(), roll_axis=r)
            assert res < 10.0 and ang < 6.0, (way, spec.size_mm, res, ang)


def _stub(monkeypatch, verdicts):
    """A runner whose System 1 returns these verdicts in turn, recording what each search was told to hold fixed."""
    seen = []
    results = iter(verdicts)

    def run_system1(sim, prog, cid, ecfg, judge_fn, **kw):
        seen.append(ecfg.param_overrides)
        v = next(results)
        return LS.S1Outcome(SimpleNamespace(verdict=v, s1_trials=2, model_copy=lambda update: SimpleNamespace(verdict=v, **update)),
                            SimpleNamespace(), 2, {"grasp": {"depth": 0.03}}, [], None if v == "pass" else "s1_exhausted", "grasp")

    monkeypatch.setattr(LS, "run_system1", run_system1)
    monkeypatch.setattr(LS, "objective", lambda r, t, travel: -10.0 if r.verdict == "pass" else 0.0)
    monkeypatch.setattr(env, "reset_episode", lambda sim: None)
    lib = Library.seed()
    runner = SimpleNamespace(cfg=SimpleNamespace(library=lib, m=2, max_param_delta=0.3, reparam={}), panel=SimpleNamespace(id="p"),
                             seed=0, sim=SimpleNamespace(controls={"c": SimpleNamespace(spec=SimpleNamespace(travel=0.03))}),
                             config=lambda control: SimpleNamespace(param_overrides={}))
    first = env.ControlRun(SimpleNamespace(verdict="stuck", s1_trials=2, model_copy=lambda update: SimpleNamespace(verdict="stuck", **update)),
                           SimpleNamespace(), "pick_up")
    return runner, first, lib.prims["pick_up"].body, seen


def test_it_stops_at_the_first_way_that_passes_and_the_replay_keeps_it(monkeypatch):
    runner, first, prog, seen = _stub(monkeypatch, ["stuck", "pass"])
    run = env.other_ways(runner, SimpleNamespace(id="c"), prog, first, None)
    assert run.result.verdict == "pass" and run.ways == 3 and len(seen) == 2
    assert run.params["grasp"] == {"along": 0.03, "depth": 0.03}, "a replay must hold the object the same way"


def test_a_failure_says_every_way_was_tried(monkeypatch):
    n = len(Library.seed().prims["grasp"].modes)
    runner, first, prog, seen = _stub(monkeypatch, ["stuck"] * n)
    run = env.other_ways(runner, SimpleNamespace(id="c"), prog, first, None)
    assert run.result.verdict == "stuck" and run.ways == n + 1 and len(seen) == n


def test_a_pass_searches_nothing_more(monkeypatch):
    runner, first, prog, seen = _stub(monkeypatch, [])
    passed = env.ControlRun(SimpleNamespace(verdict="pass", s1_trials=0), SimpleNamespace(), "pick_up")
    assert env.other_ways(runner, SimpleNamespace(id="c"), prog, passed, None) is passed and not seen


@pytest.mark.parametrize("task,seed", [("reg_mustard_bottle", 1)])
def test_the_stock_jaw_says_how_many_ways_it_failed(task, seed):
    world = make_world("gripper")
    t = next(x for x in world.tasks() if x.id == task)
    out = world.solve(world.stock(), t, seed)
    assert out.success or f"tried {len(G._ways())} ways" in out.detail


def test_each_way_draws_its_own_search_samples(monkeypatch):
    """Ways searched with one seed try the same samples, so they fail together and are not 7 tries (D-105)."""
    from sea.control import local_search

    seeds = []

    def run(sim, prog, cid, cfg, judge, *, seed, **kw):
        seeds.append(seed)
        return SimpleNamespace(params={}, result=first.result, trace=first.trace, escalate="")

    monkeypatch.setattr(local_search, "run_system1", run)
    monkeypatch.setattr(local_search, "objective", lambda *a: 1.0)
    monkeypatch.setattr(env, "reset_episode", lambda sim: None)
    lib = Library.seed().prims
    first = env.ControlRun(SimpleNamespace(verdict="stuck", s1_trials=1, model_copy=lambda update: None), None, "pick_up", {})
    runner = SimpleNamespace(
        cfg=SimpleNamespace(library=SimpleNamespace(prims=lib), m=9, max_param_delta=0.5, reparam=None), panel=SimpleNamespace(id="p"),
        sim=SimpleNamespace(controls={"c": SimpleNamespace(spec=SimpleNamespace(travel=0.01))}), seed=7,
        config=lambda c: SimpleNamespace(param_overrides=None))
    env.other_ways(runner, SimpleNamespace(id="c"), lib["pick_up"].body, first, None)
    assert len(seeds) == len(lib["grasp"].modes) and len(set(seeds)) == len(seeds) and 7 not in seeds

