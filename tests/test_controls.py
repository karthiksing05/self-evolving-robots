"""Control dynamics on compiled models (spec 7.2, 12.2 ``test_controls``)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from sea.contracts import ControlSpec, Design, PanelSpec, Pose, Rules, Vec3
from sea.sim.build_model import build_model
from sea.sim.controls import PUSH_LATCH_DEPTH, PUSH_LATCH_OPEN_LIN, ControlDynamics, Faults, h
from sea.sim.panels import layout, load_config, panel_pose, success_check


def _spec(cls: str, **kw) -> ControlSpec:
    base = dict(id="c0", cls=cls, pose_panel=Pose(pos=Vec3(x=0.0, y=0.0, z=0.0)), detents=[], detent_peak=0.0,
                handle="none", recess_depth_mm=0.0, latch_threshold=None, friction=0.0)
    base.update(kw)
    base["success"] = success_check(cls, base["handle"], len(base["detents"]))
    return ControlSpec(**base)


def _sim(spec: ControlSpec):
    cfg = load_config()
    placed, _ = layout([spec], cfg)
    panel = PanelSpec(id="t", seed=0, split="reference", pose_world=panel_pose(0.22, 0.0, -0.03, 0.0), tilt_rad=0.0,
                      controls=placed, rules=Rules(force_cap_by_class={k: float(v) for k, v in cfg["force_caps"].items()}))
    sim = build_model(Design(id="d"), panel)
    return sim, sim.controls["c0"]


def _required_force(sim, handle, q: float) -> float:
    """Force the robot must apply along +q to hold the control at ``q`` (as the sim applies it)."""
    sim.data.qpos[handle.qadr] = q
    sim.data.qvel[handle.dadr] = 0.0
    sim.apply_control_forces()
    return -float(sim.data.qfrc_applied[handle.dadr])


@pytest.mark.parametrize("spec", [
    _spec("push_button", size_mm=(12.0,), stiffness=800.0, travel=0.003, detents=[0.0018], detent_peak=1.2),
    _spec("slider", size_mm=(50.0,), stiffness=100.0, travel=0.03, detents=[0.005, 0.015, 0.025], detent_peak=0.8),
], ids=["button", "slider"])
def test_detent_dip_at_each_detent(spec: ControlSpec):
    """Force-travel minus the spring line dips by the detent peak (within 10%) and snaps through
    zero at each declared detent (within 10% of the detent spacing)."""
    sim, handle = _sim(spec)
    k = handle.dyn.k
    qs = np.linspace(0.0, spec.travel * 1.2, 4001)
    resid = np.array([_required_force(sim, handle, q) - k * q for q in qs])
    spacing = min(np.diff([0.0] + spec.detents + [spec.travel]))
    for d in spec.detents:
        win = (qs > d - 0.5 * spacing) & (qs < d)
        dip = resid[win].min()
        assert abs(-dip - spec.detent_peak) <= 0.1 * spec.detent_peak
        near = (qs > d - 0.5 * spacing) & (qs < d + 0.5 * spacing)
        q_near, r_near = qs[near], resid[near]
        cross = q_near[np.argmax((r_near[:-1] < 0) & (r_near[1:] >= 0))]
        assert abs(cross - d) <= 0.1 * spacing


def test_detent_bump_shape():
    assert h(0.0) == 0.0
    assert h(1.0) == pytest.approx(1.0)
    assert h(-1.0) == pytest.approx(-1.0)


def _push(sim, handle, force: float, seconds: float) -> None:
    """Apply ``force`` along the control's +q direction (world wrench on its body) for ``seconds``."""
    m, d = sim.model, sim.data
    axis = d.xmat[handle.body_id].reshape(3, 3) @ m.jnt_axis[handle.joint_id]
    d.xfrc_applied[handle.body_id] = 0.0
    if m.jnt_type[handle.joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
        d.xfrc_applied[handle.body_id, 3:] = axis * force * handle.dyn.lever_r
    else:
        d.xfrc_applied[handle.body_id, :3] = axis * force
    sim.step(int(round(seconds / m.opt.timestep)))
    d.xfrc_applied[handle.body_id] = 0.0


@pytest.mark.parametrize("cls,kw", [
    ("drawer", dict(size_mm=(80.0, 40.0), stiffness=40.0, travel=0.04)),
    ("door", dict(size_mm=(80.0, 80.0), stiffness=0.05, travel=math.radians(70))),
])
def test_latch_holds_below_threshold_and_releases_above(cls, kw):
    spec = _spec(cls, handle="pull", latch_threshold=5.0, **kw)
    sim, handle = _sim(spec)
    assert handle.dyn.latched
    _push(sim, handle, 0.8 * 5.0, 0.4)
    assert handle.dyn.latched
    assert sim.control_q("c0") < 0.05 * spec.travel  # the latch is a (slightly soft) joint limit
    _push(sim, handle, 1.2 * 5.0, 0.4)
    assert not handle.dyn.latched
    assert sim.control_q("c0") > 0.2 * spec.travel


def test_latch_fault_scales_threshold():
    dyn = ControlDynamics(_spec("drawer", size_mm=(80.0, 40.0), stiffness=40.0, travel=0.04, handle="tab",
                                latch_threshold=5.0), faults=Faults(latch_scale=3.0))
    assert dyn.latch_threshold == pytest.approx(15.0)
    dyn.update(0.0, latch_force=12.0)
    assert dyn.latched
    dyn.update(0.0, latch_force=16.0)
    assert not dyn.latched


def test_push_latch_releases_at_two_mm():
    sim, handle = _sim(_spec("drawer", size_mm=(80.0, 40.0), stiffness=40.0, travel=0.015, handle="push_latch",
                             latch_threshold=4.0, friction=0.05))
    d = sim.data

    def press_to(q: float) -> None:
        d.qpos[handle.qadr] = q
        d.qvel[handle.dadr] = 0.0
        mujoco.mj_forward(sim.model, d)
        sim.update_controls()

    press_to(-0.75 * PUSH_LATCH_DEPTH)
    assert handle.dyn.latched
    press_to(-1.0 * PUSH_LATCH_DEPTH)
    assert not handle.dyn.latched
    press_to(0.0)
    sim.step(2500)  # released: the pop spring opens the drawer to its 15 mm rest
    assert sim.control_q("c0") == pytest.approx(PUSH_LATCH_OPEN_LIN, abs=2e-3)


def test_faults_reverse_and_jam_change_the_joint():
    spec = _spec("slider", size_mm=(40.0,), stiffness=100.0, travel=0.02)
    cfg = load_config()
    placed, _ = layout([spec], cfg)
    panel = PanelSpec(id="t", seed=0, split="reference", pose_world=panel_pose(0.22, 0.0, -0.03, 0.0), tilt_rad=0.0,
                      controls=placed, rules=Rules(force_cap_by_class={"slider": 10.0}))
    rev = build_model(Design(id="d"), panel, faults={"c0": Faults(reverse=True)})
    jam = build_model(Design(id="d"), panel, faults={"c0": Faults(jammed=True)})
    ok = build_model(Design(id="d"), panel)
    jid = ok.controls["c0"].joint_id
    assert np.allclose(rev.model.jnt_axis[jid], -ok.model.jnt_axis[jid])
    assert rev.model.jnt_range[jid][0] == pytest.approx(-ok.model.jnt_range[jid][1])
    assert np.ptp(jam.model.jnt_range[jid]) < 1e-3
