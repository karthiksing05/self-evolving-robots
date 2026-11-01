"""MJCF vs URDF tool frames (spec 12.2): the sim and the IK must agree on the tool point."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sea.contracts import ARM_JOINTS, Design
from sea.control.frames import STOCK_URDF
from sea.control.ik import IK
from sea.sim.build_model import build_model


def _check(design: Design, urdf, n: int = 100, seed: int = 0) -> tuple[float, float]:
    sim = build_model(design)
    ik = IK(urdf)
    rng = np.random.default_rng(seed)
    jr = np.array([sim.model.jnt_range[sim.model.joint(j).id] for j in ARM_JOINTS])
    worst_mm, worst_deg = 0.0, 0.0
    for _ in range(n):
        q = rng.uniform(jr[:, 0], jr[:, 1])
        sim.set_arm_q(q)
        p = sim.data.site_xpos[sim.site_id]
        site_x = sim.data.site_xmat[sim.site_id].reshape(3, 3)[:, 0]
        t = ik.fk(q)
        worst_mm = max(worst_mm, float(np.linalg.norm(t[:3, 3] - p) * 1000.0))
        worst_deg = max(worst_deg, float(np.degrees(np.arccos(np.clip(t[:3, 2] @ site_x, -1.0, 1.0)))))
    return worst_mm, worst_deg


def test_stock_frames_agree():
    mm, deg = _check(Design(id="stock"), STOCK_URDF)
    assert mm < 1.0 and deg < 1.0


@pytest.mark.skipif(importlib.util.find_spec("sea.design.arm") is None,
                    reason="design/arm.py (L2 URDF/MJCF scaling, M9) is not implemented yet")
def test_l2_designs_agree():  # pragma: no cover - enabled with design/arm.py
    from sea.contracts import ArmSpec
    from sea.design import arm

    rng = np.random.default_rng(1)
    for i in range(5):
        spec = ArmSpec(upper_scale=float(rng.uniform(0.8, 1.3)), fore_scale=float(rng.uniform(0.8, 1.3)),
                       wrist_offset_mm=float(rng.uniform(0, 20)))
        design = Design(id=f"l2_{i}", arm=spec)
        urdf = arm.urdf_for(design)
        mm, deg = _check(design, urdf, seed=i)
        assert mm < 1.0 and deg < 1.0
