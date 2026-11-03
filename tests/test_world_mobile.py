"""The mobile exploration world (sea.worlds.mobile): physics, the stock and gold bodies, resources, the loop."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from sea.evolve.core import Body, Edit, World, apply
from sea.evolve.loop import RunConfig, held_groups, run
from sea.evolve.resources import check
from sea.sim import mobile_model as MM
from sea.worlds.mobile import PROBE, ROUTES, MobileWorld, _attempt, _Calm, _climb, _drive, _robot, _spec

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
    assert len(GROUPS) == len(ROUTES) == 13 and len({t.id for t in tasks}) == len(tasks)
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
    for fact in ("wide overall", "edges it climbs at full duty", "ground pressure", "Wh usable", "sensing in the dark"):
        assert fact in brief
    assert "swap" not in brief and "tracks" not in brief  # facts about this body, not advice


def test_edges_are_driven_in_mujoco_and_climbed_as_a_vehicle_climbs_them():
    small = _spec(_body(Edit(slot="locomotion", params={"wheel_radius": 2.5})))
    big = _spec(_body(Edit(slot="locomotion", params={"wheel_radius": 4.5})))
    assert _climb(big, "concrete") > _climb(small, "concrete") > _climb(small, "wood")  # radius and traction help
    edge = {"k": "step", "s": "concrete", "h": 2.0}
    assert _drive(small, "step", edge, 0.75, 1.0).ok and not _drive(small, "step", edge | {"h": 6.0}, 0.75, 1.0).ok
    tracks, low = _spec(_body(*GOLD)), _spec(_body(*GOLD[1:], Edit(slot="locomotion", type="tracks",
                                                                     params={"track_length": 24.0, "idler_height": 3.0})))
    assert _climb(tracks, "concrete") > 5.5 > _climb(low, "concrete")  # the edge must meet the sloped front run
    ep = MM.episode(_robot(tracks, "concrete"), "step", {"h": 5.5}, 0.75, 1.0, trace=True)
    assert ep.ok and ep.qpos[-1][2] > 0.055  # it ends on top of the step, on its running gear
    slow = _drive(tracks, "step", {"k": "step", "s": "concrete", "h": 5.5}, 0.75, 0.2)
    assert not slow.ok and slow.why.startswith("stuck")  # at a low duty the drive cannot lift it over the edge


def test_a_ramp_is_climbed_on_its_surface_at_its_slope():
    """The ramp's euler angle went in as radians where MuJoCo reads degrees: a 22 degree ramp lay 0.38
    degrees off flat, 19 cm up, and robots drove under it (seen in a film)."""
    tracks = _spec(_body(*GOLD))
    ep = _drive(tracks, "ramp", {"k": "ramp", "s": "gravel", "deg": 22.0}, 0.6, 0.7, trace=True)
    x, z = ep.qpos[-1][0], ep.qpos[-1][2]
    assert ep.ok and x > 0.3 and abs(z - x * np.tan(np.radians(22.0)) - 0.05) < 0.02  # riding on the face


def test_width_ground_pressure_trench_and_energy():
    assert W.solve(_body(Edit(slot="chassis", params={"width": 12.0})), TASK["bedroom"]).success
    out = W.solve(_body(Edit(slot="chassis", params={"width": 16.0})), TASK["bedroom"])
    assert not out.success and out.mechanism == "chassis_wider_than_gap" and "21.0 cm wide" in out.detail
    light, heavy = (_spec(_body(Edit(slot="battery", params={"capacity": c}))) for c in (2.4, 19.2))
    assert heavy.m > light.m and heavy.m / heavy.area > light.m / light.area  # more mass, more pressure
    assert _spec(_body(*GOLD)).m / _spec(_body(*GOLD)).area < light.m / light.area / 3  # tracks spread it
    short = _body(*GOLD[1:], Edit(slot="locomotion", type="tracks", params={"track_length": 18.0,
                                                                              "idler_height": 6.0}))
    assert W.solve(short, TASK["garden"]).success  # a rigid run bridges a trench up to about half its length
    out = W.solve(W.stock(), TASK["garden"])
    assert not out.success and out.mechanism == "belly_hits_step_edge"  # 6 cm wheels drop into the 9 cm trench
    assert W.solve(_body(*GOLD), TASK["garden"]).success
    small, big = (W.solve(_body(Edit(slot="battery", params={"capacity": c})), TASK["lawn"]) for c in (4.8, 9.6))
    assert not small.success and small.mechanism == "battery_exhausted" and big.success


def test_stock_holds_the_floors_and_fails_the_rest_for_physical_reasons():
    held, outs = _held(W.stock())
    assert held == {"kitchen", "living_room", "under_sofa", "shed", "lawn", "study"}  # four cells (D-062) reach the lawn's end
    assert {g: o.mechanism for g, o in outs.items() if g not in held} == {
        "bedroom": "chassis_wider_than_gap", "hallway": "sensing_range_short", "porch": "belly_hits_step_edge",
        "garden": "belly_hits_step_edge", "sandbox": "sinks_in_sand", "patio": "belly_hits_step_edge",
        "greenhouse": "belly_hits_step_edge"}
    for g, o in outs.items():
        if g not in held:
            assert o.score < 1 and any(c.isdigit() for c in o.detail) and ("cm" in o.detail or " s " in o.detail
                                                                          or "Wh" in o.detail)
    # two of the failures yield to parameters alone, and keep every held group
    for edit, gain in ((Edit(slot="chassis", params={"width": 12.0}), "bedroom"),
                       (Edit(slot="battery", params={"capacity": 9.6}), "lawn")):
        h, _ = _held(_body(edit))
        assert h == held | {gain}


def test_stock_parts_reach_the_porch_garden_and_sandbox_by_parameters_but_never_the_dark_hallway():
    loco = CAT["wheels4"].params["wheel_radius"]
    for r, w, ride, cap, gear in itertools.product(np.linspace(loco.lo, loco.hi, 4), (10.0, 18.0), (1.0, 4.0),
                                                   (9.6, 14.4), (50.0, 298.0)):  # 4.8 and 7.2 V (D-062)
        body = _body(Edit(slot="locomotion", params={"wheel_radius": float(r)}),
                     Edit(slot="chassis", params={"width": w, "ride_height": min(ride, float(r) + 1.5)}),
                     Edit(slot="battery", params={"capacity": cap}), Edit(slot="drive", params={"gear_ratio": gear}))
        assert not W.solve(body, TASK["hallway"]).success  # a bump switch cannot see: a part must change
    # driven in MuJoCo (D-063), three routes the closed form called impossible yield to parameters:
    hop = _body(Edit(slot="locomotion", params={"wheel_radius": 3.67}), Edit(slot="drive", params={"gear_ratio": 50.0}),
                Edit(slot="chassis", params={"width": 10.0, "ride_height": 4.0}))
    assert W.check(hop) == [] and W.solve(hop, TASK["porch"]).success  # fast big wheels hop the 5.5 cm edge
    assert W.solve(hop, TASK["sandbox"]).success  # and pop a wheelie out of their own rut
    sled = _body(Edit(slot="locomotion", params={"wheel_radius": 2.83}),
                 Edit(slot="chassis", params={"width": 10.0, "ride_height": 1.0}))
    assert W.check(sled) == [] and W.solve(sled, TASK["garden"]).success  # the low tub slides over the trench
    assert not W.solve(W.stock(), TASK["porch"]).success


def test_a_gold_body_from_the_catalogue_holds_every_group_but_one_within_budget():
    """D-069: no catalogue body holds both the patio's 7.5 cm stairs and the 9 cm sofa.  Climbing the stairs takes
    an 8 cm idler, and the track stands 1.5 cm above it; under the sofa, 9 cm is the most."""
    gold = _body(*GOLD)
    held, _ = _held(gold)
    assert held == GROUPS - {"patio"} and W.check(gold) == []
    tall = _body(Edit(slot="locomotion", type="tracks", params={"track_length": 30.0, "idler_height": 8.0}),
                 Edit(slot="chassis", params={"width": 10.0, "length": 26.0, "ride_height": 3.0}), *GOLD[2:])
    assert _held(tall)[0] == GROUPS - {"under_sofa"} and W.check(tall) == []
    res = _cost(gold)
    assert res.ok and 0 < res.usd <= W.manifest()["budget"]["usd"] and res.fab_minutes > 0
    test = [t for t in W.tasks() if t.split == "test"]
    assert sum(W.solve(gold, t).success for t in test) >= len(test) - 2  # it generalises to the varied routes


def test_distractors_cost_held_groups_or_cannot_be_built():
    mec, _ = _held(_body(Edit(slot="locomotion", type="mecanum4")))
    assert mec <= _held(W.stock())[0]  # dearer wheels that gain nothing
    rb = _body(Edit(slot="locomotion", type="rocker_bogie6", params={"wheel_radius": 6.0, "bogie_length": 14.0}),
               Edit(slot="battery", type="liion_pack"))
    assert W.check(rb) == [] and W.solve(rb, TASK["garden"]).success
    out = W.solve(rb, TASK["under_sofa"])
    assert not out.success and out.mechanism == "too_tall_for_clearance"
    assert W.solve(rb, TASK["porch"]).success  # 6 cm wheels behind a pushing bogie take the 5.5 cm step
    assert W.solve(_body(Edit(slot="sensor", type="ultrasonic")), TASK["hallway"]).success
    cam = W.solve(_body(Edit(slot="sensor", type="camera")), TASK["hallway"])
    assert not cam.success and cam.mechanism == "sensing_range_short"
    res = _cost(_body(Edit(slot="sensor", type="lidar")))
    assert not res.ok and "cannot be bought" in res.errors[0]


def test_bom_check_and_manifest():
    stock, man = W.stock(), W.manifest()
    bom = W.bom(stock)
    items = {b.item: b.qty for b in bom if not b.item.startswith("print:")}
    assert items == {"n20_gearmotor_100": 4, "driver_tb6612": 2, "nimh_aa": 4, "bump_switch": 2, "tire_oring": 4}
    assert {b.slot for b in bom if b.item == "print:PLA"} == {"chassis", "locomotion"}
    assert all(i in man["items"] for i in items) and set(man["budget"]) == {"usd", "lead_days", "fab_minutes"}
    assert _cost(stock).ok and _cost(stock).usd == 0
    rb = _body(Edit(slot="locomotion", type="rocker_bogie6"), Edit(slot="battery", params={"capacity": 14.4}))
    assert any("supplies at most 4 A" in e for e in W.check(rb))  # six motors at 7.2 V: 5.4 A, over the NiMH 4 A
    bad = _body(Edit(slot="locomotion", type="tracks", params={"track_length": 30.0, "idler_height": 6.0}))
    assert any("track frames mount" in e for e in W.check(bad))  # a 30 cm track on a 20 cm tub
    tiny = _body(Edit(slot="drive", type="jga25_gearmotor"), Edit(slot="battery", type="liion_pack"),
                 Edit(slot="chassis", params={"width": 10.0, "length": 14.0}))
    assert any("floor cannot hold" in e for e in W.check(tiny))
    res = _cost(_body(Edit(slot="locomotion", type="rocker_bogie6"), Edit(slot="drive", type="jga25_gearmotor"),
                      Edit(slot="battery", type="liion_pack", params={"capacity": 28.2})))
    assert not res.ok and any("left" in e for e in res.errors)  # the expensive redesign is over budget


def test_the_pack_voltage_drives_the_motors_within_the_winding_range():
    stock = _spec(W.stock())
    two = _body(Edit(slot="battery", params={"capacity": 4.8}))  # two NiMH cells: 2.4 V on a 6 V winding
    assert any("2.4 V" in e and "3.0-9.0 V" in e for e in W.check(two))
    six = _spec(_body(Edit(slot="battery", params={"capacity": 14.4})))  # 7.2 V: 1.5 times the four cells' push
    assert six.f_stall == pytest.approx(1.5 * stock.f_stall) and six.v_nl == pytest.approx(1.5 * stock.v_nl)
    li = _body(Edit(slot="battery", type="liion_pack", params={"capacity": 28.2}))  # 12.6 V charged on 6 V
    assert any("12.6 V charged" in e for e in W.check(li))
    assert not any("V charged" in e for e in W.check(_body(Edit(slot="battery", type="liion_pack",
                                                                    params={"capacity": 18.8}))))  # 8.4 V: in range



def test_a_robots_parts_cannot_pass_into_each_other():
    """D-077: a large mecanum wheel's rollers reach into the tub; big wheels on a short bogie run into each other."""
    assert MM.self_clashes(_robot(_spec(W.stock()), "wood")) == []
    big = W.check(_body(Edit(slot="locomotion", type="mecanum4", params={"wheel_radius": 4.0})))
    assert len(big) == 4 and all("roller of wheel" in e and "tub" in e and "1.8 mm" in e for e in big)
    assert W.check(_body(Edit(slot="locomotion", type="mecanum4", params={"wheel_radius": 3.0}))) == []
    rb = W.check(_body(Edit(slot="locomotion", type="rocker_bogie6",
                            params={"wheel_radius": 6.0, "bogie_length": 10.0})))
    assert len(rb) == 4 and all("wheel w" in e and "20.0 mm" in e for e in rb)  # 12 cm wheels 10 cm apart
    assert W.check(_body(Edit(slot="locomotion", type="rocker_bogie6",
                              params={"wheel_radius": 4.0, "bogie_length": 12.0}))) == []

def test_mechanism_names_the_body_or_the_search():
    gold, stock = _spec(_body(*GOLD)), _spec(W.stock())
    bedroom = ROUTES["bedroom"][2]
    ok, _, mech, _ = _attempt(gold, bedroom, 60, {"speed": 0.3, "line": -6.0}, np.random.default_rng(0))
    assert not ok and mech == "clipped_gap_edge"  # the controls were wrong...
    assert _attempt(gold, bedroom, 60, PROBE, _Calm(), probe=True)[0]  # ...the body was fine
    ok, _, mech, _ = _attempt(stock, ROUTES["porch"][2], 40, PROBE, _Calm(), probe=True)
    assert not ok and mech == "belly_hits_step_edge"  # at full duty the wheels hop the edge and the tub grounds


def test_speed_and_the_loop(tmp_path):
    t0 = time.perf_counter()
    out = W.solve(W.stock(), TASK["porch"])
    assert time.perf_counter() - t0 < 10 and out.attempts >= 24 and out.sim_s < 10  # MuJoCo at the step (D-063)
    s = run(RunConfig(world="mobile", arm="random", iterations=2, seed=1), tmp_path / "random", world=MobileWorld(),
            log=lambda m: None)
    assert s["arm"] == "random" and len(s["curve"]) >= 2 and s["test"]["n"] == 26
    assert s["test"]["stock_held"] == ["kitchen", "lawn", "living_room", "shed", "study", "under_sofa"]
    s1 = run(RunConfig(world="mobile", arm="s1", iterations=2), tmp_path / "s1", world=MobileWorld(),
             log=lambda m: None)
    assert s1["final_held_train"] == ["kitchen", "lawn", "living_room", "shed", "study", "under_sofa"] and len(s1["curve"]) == 1


def test_stairs_a_cord_a_threshold_and_a_hose_are_driven_in_mujoco():
    """D-069: the stock crosses the study's cord and threshold but grounds its belly on the greenhouse hose and slips
    on the patio's 7.5 cm stairs; tracks with an 8 cm idler on 30 cm climb the stairs."""
    w = MobileWorld()
    tasks = {t.id: t for t in w.tasks()}
    stock = w.stock()
    assert w.solve(stock, tasks["study_train0"]).success
    hose = w.solve(stock, tasks["greenhouse_train0"])
    assert not hose.success and hose.mechanism == "belly_hits_step_edge" and "hose" in hose.detail
    assert not w.solve(stock, tasks["patio_train0"]).success
    cat = {p.name: p for p in w.catalogue()}
    tall = apply(stock, [Edit(slot="locomotion", type="tracks", params={"track_length": 30.0, "idler_height": 8.0}),
                         Edit(slot="chassis", type="printed_tub", params={"width": 10.0, "length": 24.0, "ride_height": 3.0}),
                         Edit(slot="sensor", type="ultrasonic", params={})], cat, "t")
    assert w.solve(tall, tasks["patio_train0"]).success
