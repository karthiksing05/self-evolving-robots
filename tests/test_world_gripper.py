"""The gripper world for the evolution core (D-042): parts, resources, brief, diagnosis."""

from __future__ import annotations

import pytest

from sea.evolve.core import Body, Edit, Part, World, apply
from sea.evolve.resources import check
from sea.worlds.gripper import GripperWorld, diagnose, to_design

W = GripperWorld()


def _body(**parts: Part) -> Body:
    return Body(parts={**W.stock().parts, **parts})


def test_tasks_and_protocol():
    assert isinstance(W, World)
    tasks = W.tasks()
    groups = {t.group for t in tasks}
    assert len(groups) == 21
    assert sum(t.split == "train" for t in tasks) == 42 and sum(t.split == "test" for t in tasks) == 42
    assert all("mm across the jaws" in t.facts for t in tasks)


def test_parts_map_to_the_design():
    cat = {p.name: p for p in W.catalogue()}
    b = apply(W.stock(), [Edit(slot="jaw", type="prismatic", params={"drive_mm_per_rad": 50}),
                          Edit(slot="pads", type="pads", params={"pad_tpu_mm": 2, "pad_angle_deg": 0, "reach_mm": 5,
                                                                 "groove_mm": 0, "lip_mm": 0, "width_mm": 12}),
                          Edit(slot="arm_servos", type="servos_12v")], cat, "x")
    d = to_design(b)
    assert d.fingers.jaw == "prismatic" and d.fingers.drive_mm_per_rad == 50 and d.fingers.pad_tpu_mm == 2
    assert d.arm.servo_voltage["shoulder_lift"] == 12.0 and d.arm.servo_voltage["gripper"] == 7.4
    assert to_design(W.stock()).fingers == to_design(W.stock()).fingers.__class__()
    assert W.check(b) == []


def test_resources_ground_each_redesign():
    m, stock = W.manifest(), W.stock()
    r = check(stock, W.bom(stock), stock, W.bom(stock), m)
    assert r.ok and r.usd == 0 and r.fab_minutes == 0
    slide = _body(jaw=Part(type="prismatic", params={"drive_mm_per_rad": 50}))
    r = check(slide, W.bom(slide), stock, W.bom(stock), m)
    assert r.ok and r.usd == pytest.approx(2 * 1.5 + 2 * 1.2) and r.lead_days == 3
    assert r.fab_minutes == pytest.approx((18 + 0.35 * 50) * 5.0)
    volts = _body(arm_servos=Part(type="servos_12v"))
    r = check(volts, W.bom(volts), stock, W.bom(stock), m)
    assert not r.ok and "$68.00" in r.errors[0]  # three 12 V servos and a supply cost more than the run has


def test_brief_states_the_jaw_kinematics():
    hinge = W.brief(W.stock(), W.tasks())
    assert "hinged moving jaw" in hinge and "85 mm: servo" in hinge and "off parallel" in hinge
    slide = W.brief(_body(jaw=Part(type="prismatic", params={"drive_mm_per_rad": 50})), W.tasks())
    assert "moving face parallel to the fixed face" in slide and "+1 mm vs the fixed one" in slide
    assert "swap" not in hinge.lower()  # facts, not advice


def _row(t, ctrl, z, fixed, moving):
    import numpy as np

    return {"t": t, "ctrl": ctrl, "q": 0.7, "z": z, "xy": np.zeros(2),
            "cons": {"jaw_fixed": fixed, "jaw_moving": moving, "arm": []}}


def test_diagnose_names_the_mechanism():
    f = to_design(W.stock()).fingers
    side, top, down = [(20.0, 0.0, 0.0, "jaw_fixed_0")], [(20.0, 15.0, -0.99, "jaw_moving_3")], [(20.0, 9.0, -0.5, "m")]
    descent = [_row(0.1 * k, 1.0, 0.015, [], []) for k in range(5)]
    on_top = descent + [_row(1 + 0.1 * k, -0.17, 0.015, side, top) for k in range(30)]
    assert diagnose(on_top, "", f, 0.03, 97)[0] == "moving_jaw_on_top_face"
    squeezed = descent + [_row(1 + 0.1 * k, -0.17, 0.015 + 0.0004 * k, side, down) for k in range(30)]
    mech, detail = diagnose(squeezed, "", f, 0.03, 97)
    assert mech == "squeezed_down_out" and "pushing 30 deg down" in detail
    missed = descent + [_row(1 + 0.1 * k, -0.17, 0.015, side, []) for k in range(30)]
    assert diagnose(missed, "", f, 0.03, 97)[0] == "moving_jaw_missed"
    assert diagnose(on_top, "load_limit", f, 0.03, 514)[0] == "moving_jaw_on_top_face"  # pressing, not payload
    lifted = descent + [_row(1 + 0.1 * k, -0.17, 0.015 + 0.002 * k, side, side) for k in range(30)]
    assert diagnose(lifted, "load_limit", f, 0.03, 514)[0] == "arm_stall"


def test_solve_reads_the_failure():
    tasks = {t.id: t for t in W.tasks()}
    assert W.solve(W.stock(), tasks["reg_banana"]).success
    o = W.solve(W.stock(), tasks["reg_gelatin_box"])
    assert not o.success and o.mechanism in {"squeezed_down_out", "moving_jaw_on_top_face"}
    assert "moving jaw" in o.detail and o.attempts > 1


def test_part_facts_show_each_joint_on_this_body():
    cat = {p.name: p for p in W.catalogue()}
    slide = W.part_facts(W.stock(), cat["prismatic"])
    assert len(slide) == 2 and "fingertips up to 51 mm apart" in slide[0] and "85 mm out of reach" in slide[0]
    assert "tip +1 mm" in slide[1]
    links = W.part_facts(W.stock(), cat["parallelogram"])[1]
    assert "85 mm: face -0 deg, tip +33 mm" in links
