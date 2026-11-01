"""Placo IK against the Appendix C fixture points (spec 8.1; test list 12.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sea.control import ik as ik_mod
from sea.control.frames import STOCK_URDF, approach_axis, nearest_bin
from sea.control.ik import IK
from sea.control.reach import REACH_DEG, REACH_MM

ROOT = Path(__file__).resolve().parents[1]
POINTS = yaml.safe_load((ROOT / "tests" / "fixtures" / "ik_points.yaml").read_text())["points"]


@pytest.fixture(scope="module")
def ik() -> IK:
    return IK(STOCK_URDF)


@pytest.mark.parametrize("pt", POINTS, ids=lambda p: f"{p['x']}_{p.get('y', 0)}_{p['z']}_{p['approach']}")
def test_appendix_c_points(ik, pt):
    pos = np.array([pt["x"], pt.get("y", 0.0), pt["z"]])
    axis = approach_axis(pt["approach"], pos)
    q, res, ang = ik.solve(pos, axis, tol_mm=REACH_MM, tol_deg=REACH_DEG)
    reachable = res < REACH_MM and ang < REACH_DEG
    assert reachable == (pt["expected"] == "reachable"), (res, ang)
    if reachable:
        assert res < 2.0
        t = ik.fk(q)
        assert np.linalg.norm(t[:3, 3] - pos) * 1000 < 2.0


def test_fixture_covers_appendix_c():
    assert len(POINTS) == 16
    assert sum(p["expected"] == "reachable" for p in POINTS) == 10  # 8 in the table + the 2 lateral checks


def test_relative_path_rejected():
    with pytest.raises(ValueError):
        IK(Path("assets/so101/so101_new_calib.urdf"))


def test_tool_tip_variant_solves_for_the_tip(ik):
    tip = np.eye(4)
    tip[:3, 3] = [0.0, 0.0, 0.03]  # e.g. a probe 30 mm past the jaw
    pos = np.array([0.22, 0.0, -0.03])
    q, res, ang = ik.solve(pos, np.array([0.0, 0.0, -1.0]), tool_tip=tip)
    assert res < 1.0 and ang < 3.0
    frame = ik.fk(q)  # the stock tool frame sits 30 mm back along the approach
    np.testing.assert_allclose(frame[:3, 3], pos + np.array([0.0, 0.0, 0.03]), atol=2e-3)
    variant = ik.for_tip(tip)
    assert variant is ik.for_tip(tip) and variant.frame == ik_mod.TIP_FRAME
    assert ik.for_tip(None) is ik
    np.testing.assert_allclose(variant.fk(q)[:3, 3], pos, atol=1e-3)
    assert variant.frame_fk(ik_mod.URDF_TOOL_FRAME, q).shape == (4, 4)


def test_pose_tip_and_rpy_round_trip():
    from sea.contracts import Pose, Vec3

    t = ik_mod.tip_matrix(Pose(pos=Vec3(x=0.0, y=0.0, z=0.01), quat_wxyz=(1.0, 0.0, 0.0, 0.0)))
    assert t[2, 3] == pytest.approx(0.01)
    r = ik_mod.tip_matrix(None)
    assert np.allclose(r, np.eye(4))
    gimbal = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    roll, pitch, yaw = ik_mod._rpy(gimbal)
    assert pitch == pytest.approx(np.pi / 2) and roll == 0.0


def test_track_follows_small_moves_and_roll(ik):
    q0, res, _ = ik.solve(np.array([0.22, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    assert res < 1.0
    q1, res1, ang1 = ik.track(np.array([0.222, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), q0)
    assert res1 < 0.5 and ang1 < 1.5
    assert q1[-1] == q0[-1]  # the gripper is not solved
    roll = np.array([0.0, 1.0, 0.0])
    q2, _, _ = ik.track(np.array([0.222, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), q1, roll_axis=roll)
    x = ik.fk(q2)[:3, 0]
    assert np.degrees(np.arccos(np.clip(abs(x @ roll), -1, 1))) < 3.0


def test_seeds_and_bins():
    seeds = ik_mod.load_seeds()
    assert len(seeds) == 4 and all(s.shape == (6,) for s in seeds)
    assert nearest_bin(np.array([0.0, 0.0, -1.0])) == "down"
    assert nearest_bin(approach_axis("45", np.array([0.2, 0.1, 0.0]))) == "45"
    assert nearest_bin(approach_axis("horizontal", np.array([0.0, 0.0, 0.0]))) == "horizontal"
