"""The mechanism verifier's measurements (sea.mechanism, scripts/audit_mechanism.py, D-051)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from sea.evolve.core import Edit, apply
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


def test_the_stock_jaw_rides_its_horn_and_the_slide_jaw_floats_free():
    """b021's slide (gripper_s2v3_s0): nothing carries the jaw once it leaves the servo."""
    world = GripperWorld()
    stock = audit_mechanism.gripper_geometry(world.stock())["carried"]["moving_jaw_so101_v1"]
    assert stock["joint"] == "hinge" and stock["worst_gap_mm"] == 0.0
    slide = apply(world.stock(), [Edit(slot="jaw", type="prismatic", params={"drive_mm_per_rad": 56.0})],
                  {p.name: p for p in world.catalogue()}, "t")
    carried = audit_mechanism.gripper_geometry(slide)["carried"]["moving_jaw_so101_v1"]
    assert carried["joint"] == "slide" and carried["worst_gap_mm"] > 50.0


def test_the_four_bar_is_built_from_parts_that_carry_each_other():
    """D-057: the parallelogram is links, pins and a coupler.  Every moving body is carried with no gap,
    and the only material shared across a joint is the servo's spline and rear pin, inside 3 mm of the axis.
    The kept bodies' 89 mm links outreach the finger, so a strut carries the blade from the pins."""
    world = GripperWorld()
    body = apply(world.stock(), [Edit(slot="jaw", type="parallelogram", params={"link_mm": 89.136})],
                 {p.name: p for p in world.catalogue()}, "t")
    geo = audit_mechanism.gripper_geometry(body)
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
