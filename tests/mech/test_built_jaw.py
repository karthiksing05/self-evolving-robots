"""A jaw System 2 composes: built from a module and a patch, closed, priced and judged like a catalogue jaw
(D-079)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.design import fingers as F
from sea.evolve.core import Idea, Mechanism
from sea.mech.graph import Graph
from sea.mech.kinematics import closes
from sea.mech.seed.gripper import BASE
from sea.sim import jaws as J
from sea.worlds.gripper import GripperWorld, _built_bom, to_design


def _four_bar(lf: float) -> Idea:
    """The parallelogram with its follower links shortened to ``lf``: a four-bar whose coupler turns."""
    L = 0.07
    pos = J.HINGE + J.R0 @ (J.follower_offset() + np.array([0, L - lf, 0]))
    return Idea(name="tilting four-bar", description="unequal links", slot="jaw", mechanism=Mechanism(
        module="parallelogram", params={"link_mm": 70}, patch=[
            {"op": "set", "path": "bodies/jaw_follower/pos", "value": pos.tolist()},
            {"op": "set", "path": "bodies/jaw_follower/parts/jaw_follower_horn/length", "value": lf},
            {"op": "set", "path": "bodies/jaw_follower/parts/jaw_follower_rear/length", "value": lf},
            {"op": "set", "path": "loops/jaw_follower/point", "value": [0, -lf, 0]},
            {"op": "set", "path": "linked", "value": []}]))


def test_a_four_bar_closes_tilts_its_finger_is_priced_and_judged():
    w = GripperWorld()
    b = w.build_idea(w.stock(), _four_bar(0.06), "t")
    f = to_design(b).fingers
    assert f.jaw == "built" and "tilting four-bar" in b.describe()
    assert closes(Graph.model_validate_json(f.mechanism), BASE)[0] < 1e-8
    tilts = [F.finger_tilt_deg(f, q) for q in (0.0, 0.5, 1.0, 1.5)]
    assert abs(tilts[0]) < 1e-6 and 0 < tilts[1] < tilts[2] < tilts[3]  # a parallelogram's would stay 0
    bom = {line.item: line.qty for line in _built_bom(f.mechanism)}
    assert bom["print:PLA"] > 0 and bom["mr63_bearing"] == 6 and bom["m3_screw"] == 2
    errors = w.check(b)  # moved without redesigning around it, the follower hits the wrist and nears toggle
    assert any("follower horn share material" in e for e in errors)
    assert any("mechanism law (transmission)" in e for e in errors)


def test_a_loop_that_cannot_close_is_refused_before_the_laws():
    w = GripperWorld()
    idea = _four_bar(0.06)
    idea.mechanism.patch[3]["value"] = [0, -0.02, 0]  # the pin lands 40 mm from where the coupler can meet it
    errors = w.check(w.build_idea(w.stock(), idea, "t"))
    assert len(errors) == 1 and "do not close" in errors[0]


@pytest.mark.parametrize("idea, msg", [
    (Idea(name="x", description="d", slot="pads", mechanism=Mechanism(module="revolute")), "is a jaw"),
    (Idea(name="x", description="d", slot="jaw", mechanism=Mechanism(module="scissor")), "no module 'scissor'"),
    (Idea(name="x", description="d", slot="jaw", mechanism=Mechanism(module="parallelogram",
                                                                     params={"link_mm": 200})), "outside"),
])
def test_build_idea_names_what_it_cannot_build(idea, msg):
    w = GripperWorld()
    with pytest.raises(ValueError, match=msg):
        w.build_idea(w.stock(), idea, "t")


def test_the_mechanisms_section_lists_every_module_and_part_kind():
    text = GripperWorld().modules_text()
    for name in ("revolute", "parallelogram", "prismatic", "long_slide", "- link:", "- guide:", "loops/<a>/<field>"):
        assert name in text


def test_the_brief_describes_a_built_jaw():
    """D-084: a kept built jaw is the body the next iteration's brief describes (all three C1 seeds crashed here)."""
    w = GripperWorld()
    b = w.build_idea(w.stock(), _four_bar(0.06), "t")
    tasks = [t for t in w.tasks() if t.split == "train"][:2]
    text = w.brief(b, tasks)
    assert "jaw: built mechanism 'tilting four-bar'" in text and "by object width" in text
