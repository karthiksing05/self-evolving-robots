"""The wrist's roll is limited to what the body's grasps need, from the IK (D-085)."""

from __future__ import annotations

import pytest

from sea.evolve.core import Part
from sea.mechanism import gripper as MG
from sea.sim.build_model import build_model
from sea.worlds.gripper import ROLL_MARGIN, GripperWorld, roll_range, to_design


def test_the_stock_grasps_need_the_roll_the_picks_measured():
    lo, hi = roll_range(to_design(GripperWorld().stock()).fingers)
    # the plain grasps' -1.93 rad, and the 0.2 rad a turned wrist adds (D-102)
    assert lo == pytest.approx(-1.93 - 0.2 - ROLL_MARGIN, abs=0.05) and hi == pytest.approx(ROLL_MARGIN, abs=1e-6)
    assert lo <= -1.95  # every stock pick, train and test, stayed within -1.95..0 rad (measured)


def test_the_simulator_holds_the_wrist_to_it():
    d = to_design(GripperWorld().stock())
    m = build_model(d).model
    assert tuple(m.jnt_range[m.joint("wrist_roll").id]) == pytest.approx(d.arm.wrist_roll_range)


@pytest.mark.parametrize("jaw, params", [("long_slide", {"sector_radius_mm": 50}), ("prismatic", {"pinion_teeth": 60})])
def test_parts_that_met_the_wrist_only_past_the_grasps_roll_pass(jaw, params):
    """Both met the wrist only at -2.74 rad (D-078's golden laws, swept over the joint's whole range)."""
    w = GripperWorld()
    b = w.stock().model_copy(update={"parts": {**w.stock().parts, "jaw": Part(type=jaw, params=params)}})
    MG._CACHE.clear()
    assert not [e for e in MG.laws(b) if "wrist" in e]
