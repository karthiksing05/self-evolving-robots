"""The mechanism verifier's measurements (sea.mechanism, scripts/audit_mechanism.py, D-051)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from sea.compose.library import Library
from sea.evolve.core import Body, Edit, Part, apply
from sea.mechanism.audit import Penetration
from sea.mechanism.geometry import Solid, clearance, crossings, extent, from_axis, grounded, overlap, winding
from sea.worlds.gripper import GripperWorld

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_mechanism  # noqa: E402


def _box(name: str, size: float = 10.0, body: str | None = None) -> Solid:
    return Solid(name, body or name, trimesh.creation.box(extents=(size, size, size)))


def _at(x: float) -> np.ndarray:
    t = np.eye(4)
    t[0, 3] = x
    return t


def test_overlapping_parts_cross_over_the_shared_region_and_apart_ones_keep_their_gap():
    a, b = _box("a"), _box("b")
    pts = overlap(a, _at(0), b, _at(8))
    assert len(pts) and extent(pts)[0] <= 2.0 + 1e-6  # the boxes share x in [3, 5]
    assert not len(overlap(a, _at(0), b, _at(12)))
    assert clearance(a, _at(0), b, _at(12)) == pytest.approx(2.0)


def test_a_part_sealed_inside_another_is_caught_though_no_surfaces_cross():
    big, small = _box("big", 20.0), _box("small", 4.0)
    assert not len(crossings(big, _at(0), small, _at(0)))
    assert len(overlap(big, _at(0), small, _at(0)))


def test_winding_number_tells_inside_from_outside():
    w = winding(trimesh.creation.icosphere(subdivisions=2, radius=5.0), np.array([[0.0, 0, 0], [9.0, 0, 0]]))
    assert w[0] == pytest.approx(1.0, abs=1e-6) and w[1] == pytest.approx(0.0, abs=1e-6)


def test_a_part_no_chain_of_touching_parts_reaches_is_floating():
    parts = [_box("base"), _box("arm"), _box("loose")]
    poses = {"base": _at(0), "arm": _at(10), "loose": _at(40)}
    assert grounded(parts, poses, "base") == ["loose"]


def test_distance_from_an_axis():
    d = from_axis(np.array([[3.0, 4.0, 7.0]]), np.zeros(3), np.array([0.0, 0.0, 2.0]))
    assert d[0] == pytest.approx(5.0)


def test_penetration_keeps_the_deepest_per_kind_of_pair_each_step():
    model = SimpleNamespace(geom_bodyid=np.array([0, 1, 2]))
    names = {0: "robot", 1: "object", 2: "table"}
    pen = Penetration(model, lambda b: names[b])
    contact = [SimpleNamespace(geom1=0, geom2=1, dist=-0.002), SimpleNamespace(geom1=1, geom2=0, dist=-0.005),
               SimpleNamespace(geom1=1, geom2=2, dist=0.001)]
    pen(SimpleNamespace(ncon=3, contact=contact))
    pen(SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=2, geom2=1, dist=-0.001)]))
    s = pen.summary()
    assert s["object-robot"] == {"peak_mm": 5.0, "p95_mm": 5.0, "steps": 1}
    assert s["object-table"]["peak_mm"] == 1.0  # the clear contact on step 1 is not penetration


def _jaw(world: GripperWorld, type_: str, **params: float):
    """A jaw part as System 2 would fit it, or past the catalogue's range (built directly) for an old body."""
    try:
        return apply(world.stock(), [Edit(slot="jaw", type=type_, params=params)], {p.name: p for p in world.catalogue()},
                     "t")
    except ValueError:
        return Body(parts={**world.stock().parts, "jaw": Part(type=type_, params=params)})


def test_the_stock_jaw_rides_its_horn_and_a_slide_rides_its_rods():
    """D-058: a 40-tooth slide (20 mm/rad) is a pinion on the horn and a carriage on two rods, sharing material
    with nothing but the servo's spline at any wrist roll.  b021's 56 mm/rad (gripper_s2v3_s0, a 112-tooth
    pinion) cannot be built on the catalogue's 100 mm rods: the carriage runs into the rods' clamp, the
    clamp's bracket has no way past a pinion 112 mm across to the frame, so the rods hang from nothing (D-076),
    and the pinion meets the wrist."""
    world = GripperWorld()
    stock = audit_mechanism.gripper_geometry(world.stock())["carried"]["moving_jaw_so101_v1"]
    assert stock["joint"] == "hinge" and stock["worst_gap_mm"] == 0.0
    geo = audit_mechanism.gripper_geometry(_jaw(world, "prismatic", pinion_teeth=40))
    assert geo["carried"]["jaw_carriage"]["joint"] == "slide" and geo["carried"]["jaw_carriage"]["worst_gap_mm"] <= 0.1
    assert all(c["from_axis_mm"][1] <= 3.0 for c in geo["clashes"].values())
    big = _jaw(world, "prismatic", pinion_teeth=112)
    b021 = audit_mechanism.gripper_geometry(big)
    assert any(k.startswith("jaw_clamp / jaw_carriage") for k in b021["clashes"])
    assert {"jaw_clamp", "jaw_clamp_bracket", "jaw_rod0", "jaw_rod1"} <= set(b021["loose"])
    assert any("wrist" in c["bodies"] and c.get("at_pose") for c in b021["clashes"].values())


def test_the_four_bar_is_built_from_parts_that_carry_each_other():
    """D-057: the parallelogram is links, pins and a coupler.  Every moving body is carried with no gap,
    and the only material shared across a joint is the servo's spline and rear pin, inside 3 mm of the axis.
    The kept bodies' 89 mm links outreach the finger, so a strut carries the blade from the pins."""
    geo = audit_mechanism.gripper_geometry(_jaw(GripperWorld(), "parallelogram", link_mm=89.136))
    assert set(geo["carried"]) == {"moving_jaw_so101_v1", "jaw_coupler", "jaw_follower"}
    assert all(c["material"] and c["worst_gap_mm"] <= 0.3 for c in geo["carried"].values())
    assert geo["clashes"] and all(c["from_axis_mm"][1] <= 3.0 and "gripper" in c["bodies"]
                                  for c in geo["clashes"].values())
    assert all(geo["sim_mass_g"][b] > 5.0 for b in geo["carried"])
    from sea.sim import jaws

    assert jaws.blade_gap(np.array([0.0, -0.050])) == 0.0 and jaws.blade_gap(np.array([0.0, -0.089])) > jaws.BOSS_R


def test_the_four_bar_loop_closes_and_keeps_the_finger_upright():
    from sea.contracts import Design, FingerSpec
    from sea.control.executor import HOME_Q
    from sea.sim import jaws
    from sea.sim.build_model import build_model

    sim = build_model(Design(id="p", fingers=FingerSpec(jaw="parallelogram", link_mm=70.0)))
    m, d = sim.model, sim.data
    f, c = m.body("jaw_follower").id, m.body("jaw_coupler").id
    turns = []
    for q in np.linspace(-0.17, 1.7, 12):
        qa = HOME_Q.copy()
        qa[-1] = q
        sim.set_arm_q(qa)
        at_f = d.xpos[f] + d.xmat[f].reshape(3, 3) @ m.eq_data[0][:3]
        at_c = d.xpos[c] + d.xmat[c].reshape(3, 3) @ m.eq_data[0][3:6]
        assert np.linalg.norm(at_f - at_c) < 1e-4  # the connect constraint is satisfied where the servo puts it
        turns.append(d.xmat[c].reshape(3, 3) @ d.xmat[m.body("gripper").id].reshape(3, 3).T)
        assert jaws.transmission_deg(q) >= 35.0
    assert all(np.allclose(t, turns[0], atol=1e-9) for t in turns)


def test_the_slide_follows_its_pinion_and_carries_the_finger_and_pads():
    from sea.contracts import Design, FingerSpec
    from sea.control.executor import HOME_Q
    from sea.design import fingers as F
    from sea.sim import jaws
    from sea.sim.build_model import build_model

    spec = FingerSpec(jaw="prismatic", drive_mm_per_rad=30.0, pad_tpu_mm=2.0, reach_mm=5.0)
    sim = build_model(Design(id="s", fingers=spec))
    m, d = sim.model, sim.data
    assert F.moving_body(spec) == "jaw_carriage" and {m.body(m.geom_bodyid[m.geom(f"jaw_moving_pad{k}").id]).name
                                                      for k in range(1)} == {"jaw_carriage"}
    assert jaws.module_for(0.030) == (60, pytest.approx(0.001))
    tip = np.array([F.MOVING_FACE_X, F.MOVING_TIP_Y, F.MOVING_MID_Z])
    g = m.body("gripper").id
    for q in (-0.17, 0.0, 0.8, 1.7):
        qa = HOME_Q.copy()
        qa[-1] = q
        sim.set_arm_q(qa)
        assert sim.arm_q()[-1] == pytest.approx(q)
        assert d.qpos[m.jnt_qposadr[m.joint("jaw_slide").id]] == pytest.approx(0.030 * q)
        c = m.body("jaw_carriage").id
        at = d.xmat[g].reshape(3, 3).T @ (d.xpos[c] + d.xmat[c].reshape(3, 3) @ tip - d.xpos[g])
        assert at == pytest.approx(F.moving_to_gripper(q, tip, spec), abs=1e-6)  # the facts' finger is the simulated one


def _whole_roll(monkeypatch, request):
    """The wrist's whole joint range, as before D-085; geometry swept over it is dropped when the test ends."""
    from sea.mechanism import gripper as MG
    from sea.worlds import gripper as W

    monkeypatch.setattr(W, "roll_range", lambda fingers: None)
    MG._CACHE.clear()
    request.addfinalizer(MG._CACHE.clear)


def test_the_mechanism_laws_refuse_what_cannot_be_assembled_and_pass_what_can(monkeypatch, request):
    """D-059: GripperWorld.check refuses a body whose jaw breaks a mechanism law, naming the parts and the
    numbers; each refusal has a twin that passes."""
    world = GripperWorld()
    assert world.check(_jaw(world, "prismatic", pinion_teeth=44)) == []  # D-085: clear over the roll grasps need
    assert world.check(world.stock()) == []
    assert world.check(_jaw(world, "parallelogram", link_mm=89.136)) == []
    assert world.check(_jaw(world, "prismatic", pinion_teeth=40)) == []
    _whole_roll(monkeypatch, request)
    wrist = world.check(_jaw(world, "prismatic", pinion_teeth=44))  # a pinion 46 mm across reaches the rolled wrist
    assert len(wrist) == 1 and "the wrist and the pinion share material" in wrist[0] and "wrist rolled" in wrist[0]
    b021 = "\n".join(world.check(_jaw(world, "prismatic", pinion_teeth=112)))
    assert "the clamp and the carriage share material" in b021
    assert "mechanism law (mounted): the clamp bracket touches nothing that reaches the gripper frame" in b021
    assert "mechanism law (carried): the carriage" in b021


def test_the_long_slide_passes_the_laws_up_to_the_wrist_and_opens_past_the_mustard_bottle(monkeypatch, request):
    """D-064: a gear sector clears what a whole pinion of its size hits.  At 47 mm it passes every law and
    opens 98 mm with the fingertip level; at 50 mm the sector meets the wrist rolled past -2.13 rad, which no grasp
    needs (D-085): it passes within the roll the grasps need, and is named with its numbers over the whole range."""
    from sea.design import fingers as F
    from sea.worlds.gripper import to_design

    world = GripperWorld()
    long47 = _jaw(world, "long_slide", sector_radius_mm=47)
    assert world.check(long47) == []
    f = to_design(long47).fingers
    assert f.jaw == "prismatic" and f.sector and f.drive_mm_per_rad == 47
    hi = F.Q_RANGES["prismatic"][1]
    assert F.fingertips(f, hi)[0] * 1000 == pytest.approx(98, abs=1.5)
    assert abs(F.fingertips(f, hi)[1] * 1000) < 3  # neither turns nor rises
    assert world.check(_jaw(world, "long_slide", sector_radius_mm=50)) == []
    _whole_roll(monkeypatch, request)
    wrist = world.check(_jaw(world, "long_slide", sector_radius_mm=50))
    assert len(wrist) == 1 and "the wrist and the pinion share material" in wrist[0] and "wrist rolled" in wrist[0]
    assert {"steel_rod_3x150", "lm3uu_bearing"} <= {b.item for b in world.bom(long47)}


def test_a_four_bar_near_toggle_breaks_the_transmission_law(monkeypatch):
    from sea.mechanism import gripper as G
    from sea.sim import jaws

    world = GripperWorld()
    body = _jaw(world, "parallelogram", link_mm=70.0)
    assert G.laws(body) == []
    # the follower's fixed pin moved onto the closing axis: opened wide, the links line up with the line through
    # their fixed pins (D-078: the angle is measured where the loop's links meet, in the simulator)
    monkeypatch.setattr(jaws, "FOLLOWER_DIR", np.array([1.0, 0.0, 0.0]))
    monkeypatch.setattr(G, "_CACHE", {})
    worst = min(jaws.transmission_deg(q) for q in np.linspace(*G.geometry(body)["jaw_range_servo_rad"], 50))
    assert worst < 15.0
    assert (f"mechanism law (transmission): the loop closing the follower links on the four-bar's coupler comes "
            f"within {worst:.0f} deg of toggle; its links must stay 30 deg away") in G.laws(body)


@pytest.mark.parametrize("fingers", [{"pad_tpu_mm": 2.0, "pad_angle_deg": 32.0, "width_mm": 12.0},
                                     {"jaw": "parallelogram", "link_mm": 70.0, "pad_tpu_mm": 2.0, "reach_mm": 3.0}])
def test_a_jaw_closing_on_nothing_stops_on_the_other_finger(fingers):
    """D-060: the gripper and its jaw are parent and child, so MuJoCo drops their contacts; without the
    finger pairs a padded jaw closed 12.7 mm through the fixed finger to its joint limit."""
    from sea.contracts import Design, FingerSpec
    from sea.control.executor import HOME_Q
    from sea.design import fingers as F
    from sea.sim.build_model import build_model

    spec = FingerSpec(**fingers)
    sim = build_model(Design(id="p", fingers=spec))
    q = HOME_Q.copy()
    q[-1] = 0.5
    sim.set_arm_q(q)
    q[-1] = -0.17
    for _ in range(500):
        sim.data.ctrl[:] = q
        sim.step(1)
    closed = float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint("gripper").id]])
    assert closed > -0.15 and F.gap(spec, closed) > -0.5e-3


def test_the_fingers_meeting_as_the_jaw_closes_is_its_stop_not_interference():
    """D-061, D-082: a moving finger part on a fixed one is the jaw's stop, wherever they meet: the simulator holds
    the fingers apart, so a trial measures where the jaw stops."""
    from sea.mechanism import gripper as G

    assert G._stop("jaw_fixed_pad0", "jaw_moving_pad0") and G._stop("wrist_roll_follower_so101_v1", "jaw_blade")
    assert G._stop("jaw_fixed_pad_shoe6", "jaw_moving_pad0")  # D-082: shoes past the stop are never reached
    assert not G._stop("jaw_fixed_pad0", "jaw_rack") and not G._stop("jaw_fixed_pad0", "jaw_fixed_pad_shoe1")
    world = GripperWorld()
    pads = Part(type="pads", params={"groove_mm": 0, "lip_mm": 0, "pad_angle_deg": 45, "pad_tpu_mm": 2, "reach_mm": 0,
                                     "width_mm": 12})
    assert world.check(Body(parts={**world.stock().parts, "pads": pads})) == []


def test_a_slides_rods_bend_and_a_long_overhang_skips_its_teeth_or_dents_its_bushings(monkeypatch):
    """D-076: a slide's carriage rides two 3 mm rods cantilevered from their clamp.  At the long slide's wide
    openings the squeeze's couple and the teeth's separating force bend them past what the parts allow: the
    brief's static numbers say so, and the simulator, loading the rods as cantilevers each step, stops the pick."""
    from sea.sim import jaws

    a, lift, bushing = jaws.rack_lift_mm(0.046, True, 1.5, 1.75, 1.62)
    assert a > 90 and lift > jaws.TOOTH_SKIP * 1.0 and bushing > jaws.BUSHING_RATED_N
    a, lift, _ = jaws.rack_lift_mm(0.046, True, 0.03, 1.75, 1.62)
    assert a < 30 and lift < jaws.TOOTH_SKIP * 1.0  # closed, the overhang is short and the rack stays in mesh
    world = GripperWorld()
    cube = next(t for t in world.tasks() if t.id == "reg_wood_block_cube")
    assert world.solve(world.stock(), cube).success
    slide = _jaw(world, "long_slide", sector_radius_mm=46.0)
    assert not world.solve(slide, cube).success  # however System 1 holds the cube (D-102)
    seed = Library.seed

    def plain(*a, **k):  # what the rods do to the grasp this was written about, the others reported apart
        lib = seed(*a, **k)
        lib.prims["grasp"] = lib.prims["grasp"].model_copy(update={"modes": []})
        return lib

    monkeypatch.setattr(Library, "seed", plain)
    out = world.solve(slide.model_copy(update={"id": "plain"}), cube)
    assert not out.success and out.mechanism == "unphysical" and ("bushing" in out.detail or "teeth skip" in out.detail)
