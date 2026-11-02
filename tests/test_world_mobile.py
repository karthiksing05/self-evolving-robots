"""The mobile exploration world (sea.worlds.mobile): physics, the stock and gold bodies, resources, the loop."""

from __future__ import annotations

import dataclasses
import itertools
import time

import numpy as np

from sea.evolve.core import Body, Edit, World, apply
from sea.evolve.loop import RunConfig, held_groups, run
from sea.evolve.resources import check
from sea.worlds.mobile import PROBE, ROUTES, SURF, MobileWorld, _attempt, _Calm, _spec, _step_max

W = MobileWorld()
CAT = {p.name: p for p in W.catalogue()}
TRAIN = [t for t in W.tasks() if t.split == "train"]
GROUPS = {t.group for t in TRAIN}
TASK = {t.group: t for t in TRAIN}
GOLD = (Edit(slot="locomotion", type="tracks", params={"track_length": 22.0, "idler_height": 6.6}),
        Edit(slot="chassis", params={"width": 11.0, "length": 22.0, "ride_height": 3.0}),
        Edit(slot="battery", params={"capacity": 14.4}), Edit(slot="sensor", type="ultrasonic"))


def _body(*edits: Edit) -> Body:
    return apply(W.stock(), list(edits), CAT, "t")


def _held(body: Body) -> tuple[set[str], dict]:
    outs = {t.id: W.solve(body, t) for t in TRAIN}
    return held_groups(TRAIN, outs), {t.group: outs[t.id] for t in TRAIN}


def _cost(body: Body):
    stock = W.stock()
    return check(body, W.bom(body), stock, W.bom(stock), W.manifest())


def test_protocol_tasks_and_catalogue():
    assert isinstance(W, World)
    tasks = W.tasks()
    assert len(GROUPS) == len(ROUTES) == 10 and len({t.id for t in tasks}) == len(tasks)
    assert {t.group for t in tasks if t.split == "test"} == GROUPS and len(tasks) == 3 * len(GROUPS)
    stock = W.stock()
    assert set(stock.parts) == {p.slot for p in W.catalogue()} and W.check(stock) == []
    for slot, part in stock.parts.items():
        pt = CAT[part.type]
        assert pt.stock and pt.slot == slot
        assert all(pt.params[k].lo <= v <= pt.params[k].hi for k, v in part.params.items())
    for pt in W.catalogue():  # physical docs on every part and parameter; capabilities never name a task
        assert pt.doc and pt.provides and all(p.doc and p.lo < p.hi for p in pt.params.values())
        assert not any(g in pt.provides or g.replace("_", " ") in pt.provides for g in GROUPS)
    assert 5 <= len(W.principles()) <= 8
    brief = W.brief(stock, TRAIN)
    for fact in ("wide overall", "edges it can climb", "ground pressure", "Wh usable", "sensing in the dark"):
        assert fact in brief
    assert "swap" not in brief and "tracks" not in brief  # facts about this body, not advice


def test_edge_climbing_follows_the_quasi_static_rule():
    small = _spec(_body(Edit(slot="locomotion", params={"wheel_radius": 2.5})))
    big = _spec(_body(Edit(slot="locomotion", params={"wheel_radius": 4.5})))
    assert _step_max(big, 0.6) > _step_max(small, 0.6) > _step_max(small, 0.4)  # radius and traction both help
    alone = dataclasses.replace(small, f=1.0)  # nothing pushes from behind: the single-wheel result
    assert abs(_step_max(alone, 0.6) - 2.5 * (1 - 1 / np.sqrt(1 + 0.36))) < 1e-9
    assert _step_max(small, 0.6) > _step_max(alone, 0.6)  # all-wheel drive pushes the climbing wheel
    assert all(_step_max(_spec(_body(Edit(slot="locomotion", params={"wheel_radius": r}))), mu) <= r + 1e-9
               for r in (2.0, 3.0, 4.5) for mu in (0.5, 1.0, 3.0))  # never above the radius
    tracks = _spec(_body(*GOLD))
    assert _step_max(tracks, 0.6) == 0.9 * 6.6  # tracks: the edge must meet the sloped front run


def test_width_ground_pressure_trench_and_energy():
    assert W.solve(_body(Edit(slot="chassis", params={"width": 12.0})), TASK["bedroom"]).success
    out = W.solve(_body(Edit(slot="chassis", params={"width": 16.0})), TASK["bedroom"])
    assert not out.success and out.mechanism == "chassis_wider_than_gap" and "21.0 cm wide" in out.detail
    light, heavy = (_spec(_body(Edit(slot="battery", params={"capacity": c}))) for c in (2.4, 19.2))
    assert heavy.m > light.m and heavy.m / heavy.area > light.m / light.area  # more mass, more pressure
    assert _spec(_body(*GOLD)).m / _spec(_body(*GOLD)).area < light.m / light.area / 3  # tracks spread it
    short = _body(*GOLD[1:], Edit(slot="locomotion", type="tracks", params={"track_length": 18.0,
                                                                              "idler_height": 6.0}))
    out = W.solve(short, TASK["garden"])
    assert not out.success and out.mechanism == "gap_longer_than_support"  # 9 cm > 0.45 x 18 cm
    assert W.solve(_body(*GOLD), TASK["garden"]).success
    small, big = (W.solve(_body(Edit(slot="battery", params={"capacity": c})), TASK["lawn"]) for c in (4.8, 9.6))
    assert not small.success and small.mechanism == "battery_exhausted" and big.success


def test_stock_holds_the_floors_and_fails_the_rest_for_physical_reasons():
    held, outs = _held(W.stock())
    assert held == {"kitchen", "living_room", "under_sofa", "shed"}
    assert {g: o.mechanism for g, o in outs.items() if g not in held} == {
        "bedroom": "chassis_wider_than_gap", "hallway": "sensing_range_short", "porch": "step_higher_than_climbable",
        "garden": "gap_longer_than_support", "sandbox": "sinks_in_sand", "lawn": "battery_exhausted"}
    for g, o in outs.items():
        if g not in held:
            assert o.score < 1 and any(c.isdigit() for c in o.detail) and ("cm" in o.detail or " s " in o.detail
                                                                          or "Wh" in o.detail)
    # two of the failures yield to parameters alone, and keep every held group
    for edit, gain in ((Edit(slot="chassis", params={"width": 12.0}), "bedroom"),
                       (Edit(slot="battery", params={"capacity": 9.6}), "lawn")):
        h, _ = _held(_body(edit))
        assert h == held | {gain}


def test_no_parameter_setting_of_the_stock_parts_holds_everything():
    s = _spec(_body(Edit(slot="locomotion", params={"wheel_radius": 4.5})))
    assert _step_max(s, SURF["concrete"][0] * 1.1) < 5.5  # the largest wheel cannot meet the porch's edge
    loco = CAT["wheels4"].params["wheel_radius"]
    for r, w, ride, cap, gear in itertools.product(np.linspace(loco.lo, loco.hi, 4), (10.0, 18.0), (1.0, 4.0),
                                                   (2.4, 19.2), (50.0, 298.0)):
        body = _body(Edit(slot="locomotion", params={"wheel_radius": float(r)}),
                     Edit(slot="chassis", params={"width": w, "ride_height": min(ride, float(r) + 1.5)}),
                     Edit(slot="battery", params={"capacity": cap}), Edit(slot="drive", params={"gear_ratio": gear}))
        assert not any(W.solve(body, TASK[g]).success for g in ("porch", "garden", "hallway"))
    best = _body(Edit(slot="locomotion", params={"wheel_radius": 4.2}), Edit(slot="battery", params={"capacity": 9.6}),
                 Edit(slot="chassis", params={"width": 12.0, "ride_height": 4.0}))
    held, outs = _held(best)
    assert {"porch", "garden", "sandbox", "hallway"}.isdisjoint(held) and outs["sandbox"].mechanism == "sinks_in_sand"


def test_a_gold_body_from_the_catalogue_holds_every_group_within_budget():
    gold = _body(*GOLD)
    held, _ = _held(gold)
    assert held == GROUPS and W.check(gold) == []
    res = _cost(gold)
    assert res.ok and 0 < res.usd <= W.manifest()["budget"]["usd"] and res.fab_minutes > 0
    test = [t for t in W.tasks() if t.split == "test"]
    assert sum(W.solve(gold, t).success for t in test) >= len(test) - 2  # it generalises to the varied routes


def test_distractors_cost_held_groups_or_cannot_be_built():
    _, outs = _held(_body(Edit(slot="locomotion", type="mecanum4")))
    assert not outs["kitchen"].success and not outs["shed"].success  # rollers: low edge and slope traction
    rb = _body(Edit(slot="locomotion", type="rocker_bogie6", params={"wheel_radius": 6.0, "bogie_length": 14.0}),
               Edit(slot="battery", type="liion_pack"))
    assert W.check(rb) == [] and W.solve(rb, TASK["garden"]).success
    out = W.solve(rb, TASK["under_sofa"])
    assert not out.success and out.mechanism == "too_tall_for_clearance"
    assert W.solve(rb, TASK["porch"]).mechanism == "stall_or_bounce_at_step"  # geared too fast for its big wheels
    assert W.solve(_body(Edit(slot="sensor", type="ultrasonic")), TASK["hallway"]).success
    cam = W.solve(_body(Edit(slot="sensor", type="camera")), TASK["hallway"])
    assert not cam.success and cam.mechanism == "sensing_range_short"
    res = _cost(_body(Edit(slot="sensor", type="lidar")))
    assert not res.ok and "cannot be bought" in res.errors[0]


def test_bom_check_and_manifest():
    stock, man = W.stock(), W.manifest()
    bom = W.bom(stock)
    items = {b.item: b.qty for b in bom if not b.item.startswith("print:")}
    assert items == {"n20_gearmotor_100": 4, "driver_tb6612": 2, "nimh_aa": 2, "bump_switch": 2, "tire_oring": 4}
    assert {b.slot for b in bom if b.item == "print:PLA"} == {"chassis", "locomotion"}
    assert all(i in man["items"] for i in items) and set(man["budget"]) == {"usd", "lead_days", "fab_minutes"}
    assert _cost(stock).ok and _cost(stock).usd == 0
    rb = _body(Edit(slot="locomotion", type="rocker_bogie6"))
    assert any("supplies at most 4 A" in e for e in W.check(rb))  # six motors overload the NiMH pack
    bad = _body(Edit(slot="locomotion", type="tracks", params={"track_length": 30.0, "idler_height": 6.0}))
    assert any("track frames mount" in e for e in W.check(bad))  # a 30 cm track on a 20 cm tub
    tiny = _body(Edit(slot="drive", type="jga25_gearmotor"), Edit(slot="battery", type="liion_pack"),
                 Edit(slot="chassis", params={"width": 10.0, "length": 14.0}))
    assert any("floor cannot hold" in e for e in W.check(tiny))
    res = _cost(_body(Edit(slot="locomotion", type="rocker_bogie6"), Edit(slot="drive", type="jga25_gearmotor"),
                      Edit(slot="battery", type="liion_pack", params={"capacity": 28.2})))
    assert not res.ok and any("left" in e for e in res.errors)  # the expensive redesign is over budget


def test_mechanism_names_the_body_or_the_search():
    gold, stock = _spec(_body(*GOLD)), _spec(W.stock())
    bedroom = ROUTES["bedroom"][2]
    ok, _, mech, _ = _attempt(gold, bedroom, 60, {"speed": 0.3, "line": -6.0}, np.random.default_rng(0))
    assert not ok and mech == "clipped_gap_edge"  # the controls were wrong...
    assert _attempt(gold, bedroom, 60, PROBE, _Calm(), probe=True)[0]  # ...the body was fine
    ok, _, mech, _ = _attempt(stock, ROUTES["porch"][2], 40, PROBE, _Calm(), probe=True)
    assert not ok and mech == "step_higher_than_climbable"


def test_speed_and_the_loop(tmp_path):
    t0 = time.perf_counter()
    out = W.solve(W.stock(), TASK["porch"])
    assert time.perf_counter() - t0 < 0.5 and out.attempts >= 24 and out.sim_s < 0.5
    s = run(RunConfig(world="mobile", arm="random", iterations=2, seed=1), tmp_path / "random", world=MobileWorld(),
            log=lambda m: None)
    assert s["arm"] == "random" and len(s["curve"]) >= 2 and s["test"]["n"] == 20
    assert s["test"]["stock_held"] == ["kitchen", "living_room", "shed", "under_sofa"]
    s1 = run(RunConfig(world="mobile", arm="s1", iterations=2), tmp_path / "s1", world=MobileWorld(),
             log=lambda m: None)
    assert s1["final_held_train"] == ["kitchen", "living_room", "shed", "under_sofa"] and len(s1["curve"]) == 1
