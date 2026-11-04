"""Pads on any body of a mechanism (D-081): a levelling parallelogram on the stock jaw keeps its pads level."""

from __future__ import annotations

import copy

from sea.design import fingers as F
from sea.evolve.core import Idea, Mechanism, Part
from sea.sim import jaws as J
from sea.worlds.gripper import GripperWorld, _site, to_design


def levelled(L: float = 0.06) -> Idea:
    """A carrier pinned on the stock jaw and a link from a pin on the gripper, offset as the parallelogram's
    follower is: a parallelogram that keeps the carrier's orientation; the pads ride on the carrier."""
    g_pos = (J.HINGE + J.R0 @ J.follower_offset()).tolist()
    pin = {"armature": J.JOINT_ARMATURE, "damping": J.JOINT_DAMPING}
    link = {"kind": "link", "name": "level_link", "length": L, "z": list(J.HORN_SIDE), "half_width": J.PLATE_HALF_W,
            "boss_r": J.BOSS_R, "pin_head": list(J.PIN_HEAD), "outward": 1}
    patch = [
        {"op": "add", "path": "bodies/carrier", "value": {"name": "pad_carrier", "parent": J.JAW_MESH, "pos": [0, -L, 0],
         "joints": [{"name": "carrier_q", "type": "hinge", "axis": [0, 0, 1], **pin}],
         "parts": [{"kind": "cylinder", "name": "carrier_hub", "pos": [0, 0, float(J.MID_Z)], "radius": J.BOSS_R,
                    "half_length": 0.004}, {"kind": "pad_mount", "offset": [0, L, 0]}],
         "inertial": {"pos": [0, 0, 0], "kind": "lumped", "size": 4e-4}}},
        {"op": "add", "path": "bodies/link", "value": {"name": "level_link_body", "parent": "gripper", "pos": g_pos,
         "quat": list(_site().jaw_quat), "joints": [{"name": "level_q", "type": "hinge", "axis": [0, 0, 1], **pin}],
         "parts": [link], "inertial": {"pos": [0, -L / 2, 0], "kind": "slender", "size": L}}},
        {"op": "add", "path": "loops", "value": {"a": "level_link_body", "b": "pad_carrier", "point": [0, -L, 0]}},
        {"op": "add", "path": "excludes", "value": ["level_link_body", "pad_carrier"]},
        {"op": "add", "path": "excludes", "value": ["level_link_body", J.JAW_MESH]},
        {"op": "add", "path": "excludes", "value": [J.JAW_MESH, "pad_carrier"]},
    ]
    return Idea(name="levelled pad", description="fixture", slot="jaw", mechanism=Mechanism(module="revolute", patch=patch))


def _pads(w: GripperWorld) -> Part:
    pt = next(p for p in w.catalogue() if p.slot == "pads" and p.name == "pads")
    return Part(type="pads", params={k: (v.lo + v.hi) / 2 for k, v in pt.params.items()})


def test_the_pads_ride_a_level_carrier_while_the_jaw_turns():
    w = GripperWorld()
    b = w.build_idea(w.stock(), levelled(), "lv")
    f = to_design(b).fingers
    assert F.moving_body(f) == "pad_carrier"
    assert all(abs(F.finger_tilt_deg(f, q)) < 0.1 for q in (0.0, 0.5, 1.0, 1.5))
    assert F.q_range(f)[1] > 1.7  # the stock jaw turns through 100 deg and more beneath it
    padded = b.model_copy(update={"parts": {**b.parts, "pads": _pads(w)}})
    assert {g.body for g in F.pad_geoms(to_design(padded).fingers) if g.kind == "jaw_moving"} == {"pad_carrier"}
    errors = w.check(padded)  # drawn over the stock jaw, the carrier's hub and link pass through it: real faults
    assert errors and all(e.startswith("mechanism law") for e in errors)
    assert any("moving jaw and the carrier hub share material" in e for e in errors)


def test_a_mount_needs_pads_and_the_jaws_axes():
    w = GripperWorld()
    b = w.build_idea(w.stock(), levelled(), "lv")
    assert "mounts pads on a body but the body has none" in w.check(b)[0]
    turned = levelled()
    turned = turned.model_copy(update={"mechanism": turned.mechanism.model_copy(
        update={"patch": copy.deepcopy(turned.mechanism.patch)})})
    turned.mechanism.patch[0]["value"]["quat"] = [0.9659, 0.0, 0.0, 0.2588]  # 30 deg about the hinge
    padded = w.build_idea(w.stock(), turned, "t")
    padded = padded.model_copy(update={"parts": {**padded.parts, "pads": _pads(w)}})
    assert "turned 30.0 deg from the jaw's axes" in w.check(padded)[0]
