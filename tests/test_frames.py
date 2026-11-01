"""Frame helpers shared by IK, the executor and typecheck (spec 5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.control import frames as F


@pytest.mark.parametrize("axis, angle", [((0, 0, 1), 0.3), ((1, 0, 0), 2.9), ((0, 1, 0), -3.0), ((1, 1, 1), 3.1),
                                         ((0, 0, 1), np.pi), ((1, 0, 0), np.pi), ((0, 1, 0), np.pi)])
def test_quaternion_round_trip(axis, angle):
    r = F.rot_about(np.array(axis, dtype=float), angle)
    q = F.mat_to_quat_wxyz(r)
    assert q[0] >= 0 and np.isclose(np.linalg.norm(q), 1.0)
    np.testing.assert_allclose(F.quat_wxyz_to_mat(q), r, atol=1e-9)


def test_site_and_urdf_frames():
    r_site = F.rot_about(np.array([0.0, 1.0, 0.0]), 0.4)
    r_urdf = F.site_to_urdf_rot(r_site)
    np.testing.assert_allclose(r_urdf[:, 2], r_site[:, 0], atol=1e-12)  # URDF z = MJCF site x (approach)
    np.testing.assert_allclose(F.R_SITE_URDF @ F.R_URDF_SITE, np.eye(3), atol=1e-12)


def test_small_helpers():
    t = F.transform(np.array([1.0, 2.0, 3.0]), F.rot_about(np.array([0, 0, 1.0]), np.pi / 2))
    np.testing.assert_allclose(t @ np.array([1.0, 0, 0, 1]), [1.0, 3.0, 3.0, 1.0], atol=1e-12)
    assert np.allclose(F.transform(np.zeros(3))[:3, :3], np.eye(3))
    assert F.angle_deg(np.array([1.0, 0, 0]), np.array([0, 2.0, 0])) == pytest.approx(90.0)
    for a in (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0.3, -0.2, 0.9])):
        p = F.perpendicular(a)
        assert abs(p @ a) < 1e-12 and np.isclose(np.linalg.norm(p), 1.0)
    assert F.LINK_TO_BODY[F.BODY_TO_LINK["gripper"]] == "gripper"
    np.testing.assert_allclose(F.approach_axis("45"), F.APPROACH_DIRS["45"])
