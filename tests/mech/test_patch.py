"""Patches edit a mechanism graph by names (D-079)."""

from __future__ import annotations

import pytest

from sea.mech.graph import Box, Graph
from sea.mech.patch import Op, apply_patch
from sea.worlds.gripper import _module


def test_patch_sets_adds_and_removes_by_name():
    g = _module("parallelogram", {"link_mm": 70})
    out = apply_patch(g, [
        Op(op="set", path="bodies/jaw_follower/pos", value=[0.0, 0.01, 0.0]),
        Op(op="set", path="bodies/jaw_follower/parts/jaw_follower_horn/length", value=0.06),
        Op(op="set", path="loops/jaw_follower/point", value=[0.0, -0.06, 0.0]),
        Op(op="add", path="bodies/jaw_follower/parts", value={"kind": "box", "name": "rib", "pos": [0, -0.03, 0],
                                                              "half": [0.002, 0.01, 0.002]}),
        Op(op="remove", path="bodies/jaw_follower/parts/jaw_follower_rear"),
        Op(op="set", path="linked", value=[]),
    ])
    b = next(b for b in out.bodies if b.name == "jaw_follower")
    assert b.pos == (0.0, 0.01, 0.0) and not out.linked
    assert {p.name for p in b.parts} >= {"jaw_follower_horn", "rib"} and "jaw_follower_rear" not in {p.name for p in b.parts}
    assert next(p for p in b.parts if p.name == "jaw_follower_horn").length == 0.06
    assert isinstance(next(p for p in b.parts if p.name == "rib"), Box)
    assert next(lp for lp in out.loops if lp.a == "jaw_follower").point == (0.0, -0.06, 0.0)
    assert g == _module("parallelogram", {"link_mm": 70})  # the module itself is unchanged
    assert isinstance(out, Graph)


@pytest.mark.parametrize("op, msg", [
    (Op(op="set", path="bodies/nobody/pos", value=[0, 0, 0]), "no body 'nobody'"),
    (Op(op="set", path="bodies/jaw_follower/parts/nopart/length", value=1), "no part on jaw_follower 'nopart'"),
    (Op(op="set", path="loops", value=[]), "only add to a list"),
    (Op(op="set", path="wheels/x", value=1), "unknown path"),
    (Op(op="set", path="bodies/jaw_follower/parts/jaw_follower_horn/length", value="long"), "not a valid graph"),
])
def test_patch_names_what_does_not_apply(op, msg):
    with pytest.raises(ValueError, match=msg):
        apply_patch(_module("parallelogram", {"link_mm": 70}), [op])
