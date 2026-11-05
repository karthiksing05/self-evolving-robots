"""A mechanism's joints solved from its own constraints, and the finger's motion measured from it (D-079)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.design import fingers as F
from sea.mech.kinematics import closes, closure_tables, follow
from sea.mech.seed.gripper import BASE
from sea.worlds.gripper import GripperWorld, _module, to_design


def test_closure_recovers_the_parallelograms_ratios():
    g = _module("parallelogram", {"link_mm": 70})
    tables = dict(closure_tables(g.model_copy(update={"linked": ()}), BASE))
    for joint, ratio in g.linked:  # the hand-given ratios, now solved from the loops
        for q in (-0.1, 0.5, 1.2):
            assert follow(tables[joint], q) == pytest.approx(ratio * q, abs=1e-6)
    assert closes(g.model_copy(update={"linked": ()}), BASE)[0] < 1e-8


@pytest.mark.parametrize("jaw, params", [("parallelogram", {"link_mm": 70}), ("prismatic", {"pinion_teeth": 30})])
def test_a_built_catalogue_jaw_moves_its_finger_as_the_catalogue_says(jaw, params):
    """The same module built from its graph (no patch) and taken from the catalogue: the finger's measured
    motion equals the per-jaw formula, to the table's interpolation between its servo angles (microns)."""
    from sea.evolve.core import Idea, Mechanism

    w = GripperWorld()
    built = w.build_idea(w.stock(), Idea(name=jaw, description="as is", slot="jaw",
                                         mechanism=Mechanism(module=jaw, params=params)), "b")
    cat = w.stock().model_copy(update={"parts": {**w.stock().parts, "jaw": built.parts["jaw"].model_copy(
        update={"type": jaw, "params": params, "mechanism": ""})}})
    fb, fc = to_design(built).fingers, to_design(cat).fingers
    for q in (0.0, 0.4, 0.9, 1.4):
        assert np.allclose(F.fingertips(fb, q), F.fingertips(fc, q), atol=1e-5)
        assert F.finger_tilt_deg(fb, q) == pytest.approx(F.finger_tilt_deg(fc, q), abs=1e-2)
