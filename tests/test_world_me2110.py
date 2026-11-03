"""The ME 2110 world (sea.worlds.me2110, D-073): the arena, the kit's mechanisms, the brief's rules as laws, the loop."""

from __future__ import annotations

import math

import pytest

from sea.evolve.core import Body, Edit, World, apply
from sea.evolve.loop import RunConfig, run
from sea.sim import me2110_model as MM
from sea.worlds.me2110 import BUDGET, KIT, TASKS, VARIANTS, ME2110World, _robot, _uses, envelope, fits_box

W = ME2110World()
CAT = {p.name: p for p in W.catalogue()}
TRAIN = [t for t in W.tasks() if t.split == "train"]
TASK = {t.group: t for t in TRAIN}
V = VARIANTS["train"][0]


def _body(*edits: Edit) -> Body:
    return apply(W.stock(), list(edits), CAT, "t")


def _match(body: Body, horizon: float = 8.0, **ctl: float) -> MM.Result:
    return MM.Match(_robot(body), V, ctl, horizon=horizon).run()


def test_protocol_tasks_and_catalogue():
    assert isinstance(W, World)
    tasks = W.tasks()
    groups = {t.group for t in tasks}
    assert groups == {*TASKS, "match"} and len(TRAIN) == 7 and len(tasks) == 21
    assert len({p.name for p in W.catalogue()}) == len(W.catalogue())  # the loop keys parts by name
    stock = W.stock()
    assert set(stock.parts) == {p.slot for p in W.catalogue()} and W.check(stock) == []
    assert fits_box(envelope(stock))
    for slot, part in stock.parts.items():
        pt = CAT[part.type]
        assert pt.stock and pt.slot == slot
        assert all(pt.params[k].lo <= v <= pt.params[k].hi for k, v in part.params.items())
    for pt in W.catalogue():  # physical docs; capabilities never name a task
        assert pt.doc and pt.provides and all(p.doc and p.lo < p.hi for p in pt.params.values())
        assert not any(g in pt.provides for g in (*TASKS, "match", "mermaid"))
    brief = W.brief(stock, TRAIN)
    for fact in ("folded", "kit actuators used: 2 of 2 gearmotors", "parts bought", "in/s at full duty", "push bar"):
        assert fact in brief
    assert "swap" not in brief


def test_the_rules_are_laws_before_trial():
    assert any("go/no-go box" in e for e in W.check(_body(Edit(slot="frame", params={"length": 20.0, "width": 20.0}))))
    three = _body(Edit(slot="cannon", type="gate_chute", params={"exit_z": 9.5, "reach": 9.0}),
                  Edit(slot="flag", type="piston_boom", params={"boom_len": 8.0, "pivot_z": 12.0, "raise_deg": 40.0}),
                  Edit(slot="heart", type="hook_boom", params={"boom_len": 6.0, "hook_depth": 3.0, "drop_deg": 20.0}))
    assert _uses(three)["pistons"] == 3 > KIT["pistons"] and any("3 pistons" in e for e in W.check(three))
    cam = _body(Edit(slot="release", type="cam_timer"))
    assert _uses(cam)["motors"] == 3 and any("3 motors" in e for e in W.check(cam))
    car = _body(Edit(slot="drive", type="mousetrap_car", params={"wheel_r": 1.5, "axle_r": 0.3, "lever": 4.0}))
    assert any("drive_trip trips only when a wheels2 drive stops" in e for e in W.check(car))
    assert W.check(car.model_copy(update={"parts": {**car.parts, **_body(Edit(slot="release", type="cam_timer")).parts
                                                    | {"drive": car.parts["drive"]}}})) == []
    short = _body(Edit(slot="cannon", type="gate_chute", params={"exit_z": 6.0, "reach": 3.0}))
    assert any("6.7 in" in e for e in W.check(short))
    long_bands = _body(Edit(slot="flag", type="band_mast", params={"stages": 2, "stage_len": 14.0, "lean_deg": 10.0,
                                                                    "bands": 1}))
    assert any("stretched" in e for e in W.check(long_bands))
    assert W._cost(W.stock()) < BUDGET


def test_the_stock_leaves_the_dock_and_its_bar_slides_a_figure_it_cannot_topple():
    stock = W.stock()
    assert W.solve(stock, TASK["activate"]).success
    for g in ("treasure", "flag", "cannon", "heart"):  # no mechanism: refused without a simulation
        out = W.solve(stock, TASK[g])
        assert not out.success and out.mechanism == f"no_{g}_mechanism" and out.attempts == 1
    r = _match(stock, y0=6.0, park_x=17.0, duty=0.6)
    assert r.points["mermaids"] == 5.0 and "moved" in r.notes["mermaids"] and "standing" in r.notes["mermaids"]


def test_a_bar_pushed_above_a_figures_centre_of_mass_topples_it():
    wide = _body(Edit(slot="mermaid", params={"bar_z": 5.5, "bar_w": 24.0}))
    assert W.check(wide) == []
    assert _match(wide, y0=0.0, park_x=17.0, duty=0.6).points["mermaids"] == 24.0
    assert _match(wide, y0=0.0, park_x=14.0, duty=0.6).points["mermaids"] == 0.0  # it stops short


def test_mousetrap_arms_swing_and_strike():
    arms = _body(Edit(slot="mermaid", type="trap_arms", params={"arm_len": 10.0, "arm_z": 5.0}))
    assert W.check(arms) == [] and _uses(arms)["mousetraps"] == 2
    r = _match(arms, y0=0.0, park_x=8.0, duty=0.6)
    assert r.points["mermaids"] == 24.0 and any("released mermaid" in e for _, e in r.events)


def test_the_tank_spends_air_per_stroke_and_the_rods_sit_on_their_stops():
    body = _body(Edit(slot="cannon", type="gate_chute", params={"exit_z": 9.5, "reach": 9.8}),
                 Edit(slot="frame", params={"width": 8.0}))
    mt = MM.Match(_robot(body), V, {"y0": -4.0, "park_x": 11.5, "duty": 0.5}, horizon=1.0)
    p0 = mt.p_tank
    mt.stroke("cannon", True)
    drop = (p0 - mt.p_tank) / (p0 + 1.013e5)
    assert 0.02 < drop < 0.04  # 29 mL of a 1 L tank, isothermal
    mt.run()
    assert abs(mt.d.qpos[mt.m.jnt_qposadr[mt.jid["gate"]]]) < 0.002  # the gate held shut on its stop


def test_the_cannon_chute_drops_balls_over_the_barrier_into_the_port():
    body = _body(Edit(slot="cannon", type="gate_chute", params={"exit_z": 9.5, "reach": 9.8}),
                 Edit(slot="frame", params={"width": 8.0, "deck_z": 8.0}))
    assert W.check(body) == []
    r = _match(body, horizon=30.0, y0=-4.0, park_x=11.5, duty=0.5, gate_delay=0.0, gate_open=0.8)
    assert any("IR sensor" in e for _, e in r.events)
    assert r.points["cannon"] >= 18.0 and not r.laws


def test_a_magnet_catches_only_a_flag_flat_on_a_pole_face():
    body = _body(Edit(slot="flag", type="band_mast", params={"stages": 4, "stage_len": 9.0, "lean_deg": 40.0,
                                                               "bands": 4}), Edit(slot="frame", params={"deck_z": 6.0}))
    mt = MM.Match(_robot(body), V, {"y0": 3.0, "park_x": 12.5, "duty": 0.5}, horizon=0.1)
    mt.run()
    assert mt.clip and not mt.magnet  # clipped to the mast, nowhere near the pole


@pytest.mark.slow
def test_the_loop_runs_on_the_world(tmp_path):
    s = run(RunConfig(world="me2110", arm="s1", iterations=1), tmp_path / "s1", world=ME2110World(), log=lambda m: None)
    assert s["final_held_train"] == ["activate"] and len(s["curve"]) == 1
    assert math.isclose(len(s["train_groups"]), 7)
