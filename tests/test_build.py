"""Tool templates and the build pipeline (spec 8.8, Appendix B, 12.2 test_build)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from sea.contracts import ArmSpec, Design, FingerSpec, HardLimits, ToolSpec, Vec3
from sea.design import build as B
from sea.design.space import RACK_TEMPLATES, make_tool_spec, params_at
from sea.design.tools import ConvexPart, hook_dims, make_tool, points_in_part, points_in_parts, tip_transform

LIMITS = HardLimits(workspace_box=(Vec3(x=0.1, y=-0.2, z=-0.1), Vec3(x=0.4, y=0.2, z=0.25)))


def _design(*tools: ToolSpec, **arm) -> Design:
    return Design(id="d", tool_rack=list(tools), arm=ArmSpec(**arm))


@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("template", RACK_TEMPLATES)
def test_every_template_builds_a_valid_solid(template, frac):
    spec = make_tool_spec(template, **params_at(template, frac))
    built = B.build(_design(spec), LIMITS)
    assert built.checks["validity"] is True
    tool = built.tools[0]
    assert tool.valid and tool.volume_cm3 > 0 and tool.mass_g > 0 and tool.parts
    assert tool.contact == spec.contact


def test_tip_transforms_match_appendix_b():
    def z(spec):
        return tip_transform(spec)[:3, 3] * 1000.0

    assert z(make_tool_spec("pad", thickness=5)) == pytest.approx([0, 0, 17])
    assert z(make_tool_spec("hook", depth=9, length=40)) == pytest.approx([-9, 0, 40])
    assert z(make_tool_spec("knob_cup", depth=10)) == pytest.approx([0, 0, 22])
    assert z(make_tool_spec("soft_tip", radius=4)) == pytest.approx([0, 0, 19])
    assert z(make_tool_spec("slot_inserter", length=33)) == pytest.approx([0, 0, 33])
    assert z(make_tool_spec("probe", length=50)) == pytest.approx([0, 0, 50])
    stock = ToolSpec(template="stock", params={}, contact="edge")
    assert np.array_equal(tip_transform(stock), np.eye(4))
    for t in RACK_TEMPLATES:
        spec = make_tool_spec(t)
        assert np.allclose(make_tool(spec).tip, tip_transform(spec))
        assert np.allclose(tip_transform(spec)[:3, :3], np.eye(3))


def test_tip_is_on_the_working_surface():
    """The tip point lies on the solid's boundary (pad face, foot end, rim, apex, edge, probe end)."""
    for t in RACK_TEMPLATES:
        geom = make_tool(make_tool_spec(t))
        tip = geom.tip[:3, 3]
        near = points_in_parts(geom.parts, (tip - 1e-4 * np.array([0, 0, 1]))[None])[0]
        beyond = points_in_parts(geom.parts, (tip + 1e-3 * np.array([0, 0, 1]))[None])[0]
        if t == "knob_cup":  # rim centre is in the open mouth: the rim wall is at radius r_in
            assert not near and not beyond
        elif t == "hook":  # foot end corner: inside just behind the far face and the end face
            inside = points_in_parts(geom.parts, (tip + np.array([1e-4, 0, -1e-4]))[None])[0]
            assert inside and not beyond
        else:
            assert near and not beyond


def test_parts_are_an_exact_decomposition_of_the_solid():
    rng = np.random.default_rng(0)
    for t in RACK_TEMPLATES:
        geom = make_tool(make_tool_spec(t, **params_at(t, 0.7)))
        shape = geom.solid.val()
        bb = shape.BoundingBox()
        pts_mm = rng.uniform([bb.xmin, bb.ymin, bb.zmin], [bb.xmax, bb.ymax, bb.zmax], size=(150, 3))
        import cadquery as cq

        in_solid = np.array([shape.isInside(cq.Vector(*p), 1e-6) for p in pts_mm])
        in_parts = points_in_parts(geom.parts, pts_mm / 1000.0)
        assert np.mean(in_solid == in_parts) > 0.98, t


@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
def test_hook_throat_stays_open(frac):
    p = params_at("hook", frac)
    geom = make_tool(make_tool_spec("hook", **p))
    d = hook_dims(p)
    assert d["gap"] > 0 and d["barb"] > 0
    # centre of the pocket between shaft and barb, just above the foot
    x = (-p["depth"] + d["wt"] - d["wt"] / 2) / 2
    z = p["length"] - d["wt"] - d["barb"] / 2
    probe = np.array([[x, y, z] for y in (-8.0, 0.0, 8.0)]) / 1000.0
    assert not points_in_parts(geom.parts, probe).any()
    # the hull stub fills it (why analytic parts are the default)
    built = B.build_tool(make_tool_spec("hook", **p), LIMITS.min_wall_mm, decompose="hull")
    assert len(built.parts) == 1 and built.parts[0].kind == "mesh"
    assert points_in_parts(built.parts, probe).all()


def test_thin_wall_case_is_rejected():
    built = B.build(_design(make_tool_spec("slot_inserter", blade_thickness=0.6)), LIMITS)
    assert built.checks["min_wall"] is False and not built.ok
    assert any("wall" in e for e in built.errors)
    ok = B.build(_design(make_tool_spec("slot_inserter", blade_thickness=1.5)), LIMITS)
    assert ok.checks["min_wall"] is True


def test_over_mass_case_is_rejected():
    light = LIMITS.model_copy(update={"max_tool_mass_g": 2.0})
    built = B.build(_design(make_tool_spec("knob_cup", inner_diameter=40, depth=15)), light)
    assert built.checks["mass"] is False and not built.ok
    assert any("mass" in e for e in built.errors)


def test_print_estimate_monotone_in_volume():
    vols = np.linspace(0.0, 50.0, 26)
    for mat in ("pla", "tpu"):
        est = np.array([B.print_estimate(v, mat) for v in vols])
        assert np.all(np.diff(est[:, 0]) > 0) and np.all(np.diff(est[:, 1]) > 0)
    assert B.print_estimate(10.0, "pla") == pytest.approx((22.0, 10.0 * 1.24 * B.INFILL_FACTOR))
    assert B.print_estimate(10.0, "pla", a=2.0, b=0.0)[0] == pytest.approx(20.0)
    probes = [B.build_tool(make_tool_spec("probe", length=L), 1.2) for L in (20, 40, 60, 80)]
    assert np.all(np.diff([t.volume_cm3 for t in probes]) > 0)
    assert np.all(np.diff([t.print_minutes for t in probes]) > 0)


def test_design_estimate_and_added_mass():
    pad, hook = make_tool_spec("pad", w=25, h=25), make_tool_spec("hook")
    built = B.build(_design(pad, hook), LIMITS)
    est = built.design.est
    assert est.print_minutes == pytest.approx(sum(t.print_minutes for t in built.tools))
    assert est.grams_pla == pytest.approx(sum(t.grams_pla for t in built.tools))
    assert est.added_mass_g == pytest.approx(max(t.mass_g for t in built.tools))
    assert built.ok and built.checks["self_collision"] is None
    empty = B.build(Design(id="stock"), LIMITS)
    assert empty.design.est.print_minutes == 0 and empty.ok


def test_finger_estimate():
    assert B.finger_estimate(Design(id="s")) == (0.0, 0.0, 0.0, 0.0)
    pla = Design(id="f", fingers=FingerSpec(reach_mm=20, pad_angle_deg=30))
    tpu = Design(id="g", fingers=FingerSpec(reach_mm=20, pad_angle_deg=30, pad_tpu_mm=2))
    longer = Design(id="h", fingers=FingerSpec(reach_mm=30, pad_angle_deg=30))
    m1, p1, t1, a1 = B.finger_estimate(pla)
    m2, p2, t2, a2 = B.finger_estimate(tpu)
    assert m1 > 0 and t1 == 0 and t2 > 0 and p2 == pytest.approx(p1, rel=0.05)  # the skin adds TPU, not PLA
    assert B.finger_estimate(longer)[0] > m1 and a1 > 0
    est = B.build(tpu, LIMITS).design.est
    assert est.print_minutes == pytest.approx(m2) and est.grams_tpu == pytest.approx(t2)


def test_soft_tip_is_two_materials_and_softness_softens_contact():
    hard = B.build_tool(make_tool_spec("soft_tip", softness=0.0), 1.2)
    soft = B.build_tool(make_tool_spec("soft_tip", softness=1.0), 1.2)
    assert soft.grams_tpu > 0 and soft.grams_pla > 0 and soft.material == "tpu"
    assert soft.solref[0] > hard.solref[0]  # longer time constant = softer
    assert soft.solimp[0] < hard.solimp[0] and soft.solimp[2] > hard.solimp[2]
    assert soft.needs_support


def test_knob_cup_torsion_grows_with_ribs_and_diameter():
    def torsion(**p):
        return make_tool(make_tool_spec("knob_cup", **p)).friction[1]

    assert torsion(ribs=8) > torsion(ribs=0)
    assert torsion(inner_diameter=40) > torsion(inner_diameter=10)
    assert make_tool(make_tool_spec("pad", friction=1.1)).friction[0] == pytest.approx(1.1)


def test_wrist_torque_check_and_voltage():
    long_probe = make_tool_spec("probe", length=80)
    weak = B.build(_design(long_probe), LIMITS)
    assert weak.checks["wrist_torque"] is False and not weak.ok
    volts = {j: 7.4 for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "gripper")}
    strong = B.build(_design(long_probe, servo_voltage={**volts, "wrist_flex": 12.0}), LIMITS)
    assert strong.checks["wrist_torque"] is True
    assert B.wrist_torque(None) < B.wrist_torque(weak.tools[0])


def test_link_scale_check():
    built = B.build(_design(upper_scale=1.5), LIMITS)
    assert built.checks["link_scale"] is False
    assert any("link scale" in e for e in built.errors)


def test_collision_check_hook():
    spec = make_tool_spec("probe")
    seen = []

    def bad(bd):
        seen.append(bd.design.id)
        return ["tool hits upper_arm"]

    built = B.build(_design(spec), LIMITS, collision_check=bad)
    assert seen == ["d"]
    assert built.checks["self_collision"] is False and "tool hits upper_arm" in built.errors
    assert B.build(_design(spec), LIMITS, collision_check=lambda bd: []).checks["self_collision"] is True


def test_invalid_rack_entry_is_reported():
    bad = ToolSpec(template="probe", params={"diameter": 50.0, "length": 30.0}, contact="probe")
    built = B.build(_design(bad), LIMITS)
    assert not built.ok and built.checks["validity"] is False
    assert any(e.startswith("rack[0]") for e in built.errors)


def test_stock_tool_has_nothing_to_print():
    stock = B.build_tool(ToolSpec(template="stock", params={}, contact="edge"), 1.2)
    assert stock.parts == [] and stock.mass_g == 0 and stock.print_minutes == 0
    assert B.wrist_torque(stock) == pytest.approx(B.wrist_torque(None))


def test_cache_and_stl_export(tmp_path):
    spec = make_tool_spec("pad", w=17)
    a = B.build_tool(spec, 1.2)
    assert B.build_tool(spec, 1.2) is a
    with_stl = B.build_tool(spec, 1.2, cache_dir=tmp_path)
    assert with_stl.stl_path is not None
    mesh = trimesh.load(with_stl.stl_path)
    assert mesh.is_watertight and mesh.volume / 1000.0 == pytest.approx(a.volume_cm3, rel=0.02)
    assert B.spec_hash(spec) != B.spec_hash(make_tool_spec("pad", w=18))


def test_pad_needs_support_only_when_smaller_than_mount():
    assert make_tool(make_tool_spec("pad", w=10, h=25)).needs_support
    assert not make_tool(make_tool_spec("pad", w=25, h=25)).needs_support
    small = B.build(_design(make_tool_spec("pad", w=10, h=10)), LIMITS)
    assert small.tools[0].overhang_mm2 > 0 and small.checks["overhang"] is True


def test_overhang_detects_unsupported_ledge():
    stem = trimesh.creation.box(extents=(4, 4, 10))
    stem.apply_translation((0, 0, 5))
    cap = trimesh.creation.box(extents=(20, 20, 2))
    cap.apply_translation((0, 0, 11))
    tee = trimesh.util.concatenate([stem, cap])
    assert B.overhang_area(tee, np.eye(3)) == pytest.approx(400.0)  # the cap's underside
    flat = trimesh.creation.box(extents=(10, 10, 10))
    assert B.overhang_area(flat, np.eye(3)) == pytest.approx(0.0)
    # a 45 degree overhang (from vertical) is printable, a 60 degree one is not
    for deg, flagged in ((45.0, False), (60.0, True)):
        phi = math.radians(90.0 - deg)  # angle between the face normal and straight down
        n = np.array([math.sin(phi), 0.0, -math.cos(phi)])
        u = np.array([0.0, 1.0, 0.0])
        v = np.cross(n, u)  # u x v is parallel to n, so the winding gives normal n
        p0 = np.array([0.0, 0.0, 5.0])
        tri = trimesh.Trimesh(vertices=[p0, p0 + 3 * u, p0 + 3 * v], faces=[[0, 1, 2]], process=False)
        assert np.allclose(tri.face_normals[0], n)
        assert (B.overhang_area(tri, np.eye(3)) > 0) == flagged


def test_wall_thickness_of_a_plate():
    plate = trimesh.creation.box(extents=(30, 30, 1.0))
    t = B.wall_thickness(plate, n=300)
    assert np.percentile(t, 50) == pytest.approx(1.0, abs=1e-3)


def test_points_in_part_kinds():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    pts = np.array([[0, 0, 0], [0, 0, 0.009], [0.004, 0, 0], [0.02, 0, 0]], dtype=float)
    cap = ConvexPart("capsule", np.zeros(3), q, np.array([0.005, 0.005]))
    assert points_in_part(cap, pts).tolist() == [True, True, True, False]
    cyl = ConvexPart("cylinder", np.zeros(3), q, np.array([0.005, 0.005]))
    assert points_in_part(cyl, pts).tolist() == [True, False, True, False]
    sph = ConvexPart("sphere", np.zeros(3), q, np.array([0.005]))
    assert points_in_part(sph, pts).tolist() == [True, False, True, False]
    yaw = math.pi / 4
    box = ConvexPart("box", np.zeros(3), np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
                     np.array([0.005, 0.001, 0.005]))
    assert points_in_part(box, np.array([[0.002, 0.002, 0.0], [0.002, -0.002, 0.0]])).tolist() == [True, False]


@pytest.mark.slow
def test_coacd_decomposition_keeps_hook_throat_open():
    p = params_at("hook", 1.0)
    built = B.build_tool(make_tool_spec("hook", **p), 1.2, decompose="coacd")
    assert len(built.parts) >= 2 and all(part.kind == "mesh" for part in built.parts)
    d = hook_dims(p)
    x = (-p["depth"] + d["wt"] / 2) / 2
    z = p["length"] - d["wt"] - d["barb"] / 2
    assert not points_in_parts(built.parts, np.array([[x, 0.0, z]]) / 1000.0).any()
