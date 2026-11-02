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
