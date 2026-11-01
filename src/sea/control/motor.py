"""Motor primitives (spec 8.1 ``motor.py``): one generator per primitive.

Each yields joint targets at 50 Hz (``executor.Episode.tick`` steps the sim 10 times per target)
and returns True on success.  Guarded contact stops when tip force reaches the cap, the travel is
reached, or a joint hits its limit.  Approach standoff is 30 mm along the control normal; speed
never exceeds the hard limit and halves within 10 mm of contact.

Where the tool tip must be for each (control, tool) pair lives in ``engage_pose``; it is the only
place that knows tool geometry (hook throat, pinch point between the stock jaws, blade edge).
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from sea.control.executor import GRIP_CLOSED, GRIP_OPEN, GRIP_WIDE, TICK_DT, Abort, Episode
from sea.control.frames import rot_about

Gen = Generator[np.ndarray, None, bool]

APPROACH_OFFSET = 0.03
SLOW_ZONE = 0.010
FINAL_SPEED = 0.01  # m/s for the guarded last centimetre (<= half the hard-limit speed)
CONTACT_FORCE = 0.5
MM = 1e-3
STOCK_TIP_EXT = 6.5 * MM  # the fixed jaw reaches this far past gripper_frame_link along +z
# Tool-frame point that sits 2 mm above the pinched object's base when the jaw tips are just clear
# of it: centred in the jaw gap (x = -7 mm), 4.3 mm short of the jaw ends.
PINCH_TOOL = np.array([-7 * MM, 0.0, STOCK_TIP_EXT - 2.2 * MM])
# Pinch point height above the plate so the fixed jaw tip (2.2 mm past it) clears the plate by 2 mm.
PINCH_HEIGHT = STOCK_TIP_EXT - PINCH_TOOL[2] + 2 * MM
UNREACHABLE_MM = 5.0  # contact-waypoint IK residual above which approach refuses to move
ROT_GAIN = 0.02 * MM  # m of depth per N of normal-force error per tick (rotate)
TAB_POINT_H = 7 * MM  # handle point height above the plate for tabs (sim/controls.py)
STOCK_PATCH = np.array([4.25 * MM, 0.0, 0.0])  # centre of the fixed jaw's end face


# ---------------------------------------------------------------- tool knowledge
def _tool(ep: Episode):
    slot = ep.sim.mounted
    if slot is None:
        return "stock", {}
    spec = ep.sim.tool_specs.get(slot)
    return (spec.template, dict(spec.params)) if spec is not None else ("?", {})


def tip_extension(ep: Episode) -> float:
    """How far the physical tool end lies beyond the IK tool point along the approach axis."""
    return STOCK_TIP_EXT if ep.sim.mounted is None else 0.0


def tool_half_width(ep: Episode) -> float:
    tpl, p = _tool(ep)
    return {
        "stock": 7 * MM,
        "pad": max(p.get("w", 20), p.get("h", 20)) / 2 * MM,
        "probe": p.get("diameter", 5) / 2 * MM,
        "soft_tip": p.get("radius", 5) * MM,
        "knob_cup": (p.get("inner_diameter", 20) / 2 + 2) * MM,
        "hook": 10 * MM,
        "slot_inserter": max(p.get("width", 20), 20) / 2 * MM,
    }.get(tpl, 10 * MM)


def _hook_geometry(p: dict, handle: str = "pull") -> tuple[np.ndarray, np.ndarray, float]:
    """(engage point, tip point) in the tool frame and the sideways clearance for the descent.

    A pull bar sits in the throat (engage = throat centre).  A tab lip is caught by the barb
    (engage = barb centre, under the lip)."""
    from sea.design.tools import hook_dims

    d = hook_dims(p)
    depth, length, wt = p["depth"] * MM, p["length"] * MM, d["wt"] * MM
    tip = np.array([-depth, 0.0, length])
    if handle == "tab":  # barb centred in the pocket under the lip, foot 1 mm above the plate
        engage = np.array([-depth + wt / 2, 0.0, length - (TAB_POINT_H - 1 * MM)])
        return engage, tip, 9 * MM + wt
    x_mid = ((-wt / 2) + (-depth + wt)) / 2
    throat = np.array([x_mid, 0.0, length - wt - 1 * MM])
    return throat, tip, 5 * MM + depth / 2 + wt / 4


def _tool_rot(axis: np.ndarray, roll: np.ndarray | None) -> np.ndarray:
    z = axis / np.linalg.norm(axis)
    x = roll if roll is not None else np.array([1.0, 0.0, 0.0])
    x = x - z * (x @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _roll_for(ep: Episode, cid: str) -> np.ndarray | None:
    """Tool x axis the engagement needs (None = free).  The wrist cannot roll the tool x axis to
    point away from the base (needs |roll| > 157 deg), so asymmetric engagements are designed to
    face the base: hook feet and stock jaw patches extend along +tool x."""
    h = ep.sim.controls[cid]
    tpl, _ = _tool(ep)
    cls = h.spec.cls
    if tpl == "slot_inserter" and cls == "slot":
        return _closest_sign(ep, ep.ctl_axis(1, cid))  # blade thickness (tool x) across the gap
    if tpl in ("hook", "stock") and cls in ("door", "drawer", "slot") and h.spec.handle in ("tab", "pull"):
        return -ep.ctl_axis(0, cid)  # hook foot / jaw gap opens away from the hinge side
    if tpl == "stock" and cls == "toggle":
        return ep.ctl_axis(1, cid)  # fixed-jaw patch toward the raised end
    if tpl == "stock" and cls == "knob":
        return _closest_sign(ep, ep.ctl_axis(0, cid))
    if tpl == "stock":
        return _open_jaw_roll(ep, cid)
    return None


def _open_jaw_roll(ep: Episode, cid: str) -> np.ndarray:
    """Tool x for a stock press: the open moving jaw sweeps along -tool x (20-75 mm out), so
    point it where the other controls leave the most room.  Reachable sweeps: away from the base
    or to either side (the wrist cannot point it back toward the base)."""
    h = ep.sim.controls[cid]
    p0 = h.T_world_ctl[:3, 3]
    others = [o.T_world_ctl[:3, 3] for k, o in ep.sim.controls.items() if k != cid]
    best, best_clear = -ep.ctl_axis(0, cid), -1.0
    for x_tool in (-ep.ctl_axis(0, cid), -ep.ctl_axis(1, cid), ep.ctl_axis(1, cid)):
        u = -x_tool
        a, b = p0 + u * 0.02, p0 + u * 0.075
        clear = min((_seg_dist(c, a, b) for c in others), default=1.0)
        if clear > best_clear + 1e-6:
            best, best_clear = x_tool, clear
    return best


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _closest_sign(ep: Episode, axis: np.ndarray) -> np.ndarray:
    x_now = ep.ik.fk(ep.q_cmd)[:3, 0]
    return axis if float(axis @ x_now) >= 0 else -axis


def _handle_body(ep: Episode, cid: str) -> tuple[int, np.ndarray | None]:
    h = ep.sim.controls[cid]
    if h.flap is not None:
        return h.flap[0], h.geo.flap_handle_body
    return h.body_id, h.geo.handle_body


def handle_world(ep: Episode, cid: str) -> np.ndarray | None:
    bid, pt = _handle_body(ep, cid)
    if pt is None:
        return None
    pose = ep.sim.body_pose(bid)
    return pose[:3, :3] @ pt + pose[:3, 3] + ep.cfg.perception_offset


def engage_pose(ep: Episode, cid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(tip position, approach axis, roll axis) where actuation starts for this control and tool."""
    h = ep.sim.controls[cid]
    d = ep.approach_dir(cid)
    roll = _roll_for(ep, cid)
    tpl, p = _tool(ep)
    cls = h.spec.cls
    n = -d
    if cls == "slider":
        cb = h.geo.contact_body.copy()
        cb[2] = 2 * MM  # tool end 2 mm above the track, beside the knob face
        p_side = ep.world_point(cb, cid) - ep.actuation(cid) * (tool_half_width(ep) + 2 * MM) + n * tip_extension(ep)
        if tpl == "stock":
            p_side = p_side - _tool_rot(d, roll if roll is not None else ep.ik.fk(ep.q_cmd)[:3, 0]) @ STOCK_PATCH
        return p_side, d, roll
    if cls == "knob" and tpl == "stock":
        base = ep.world_point(np.array([0.0, 0.0, PINCH_HEIGHT]), cid)
        return base - _tool_rot(d, roll) @ PINCH_TOOL, d, roll
    hp = handle_world(ep, cid)
    if hp is not None and h.spec.handle in ("tab", "pull"):
        if tpl == "hook":
            throat, tip, clear = _hook_geometry(p, h.spec.handle)
            r = _tool_rot(d, roll)
            beside = hp - ep.ctl_axis(0, cid) * clear
            return beside + r @ (tip - throat), d, roll
        if tpl == "stock":
            r = _tool_rot(d, roll)
            return _fin_base(ep, cid, hp) - r @ PINCH_TOOL, d, roll
    if tpl == "stock" and cls == "toggle":  # jaw patch runs from the contact out past the end
        return ep.contact_point(cid), d, roll
    if tpl == "stock":  # centre the fixed-jaw patch (tool x in [0, 8.5] mm) on the target
        r = _tool_rot(d, roll if roll is not None else ep.ik.fk(ep.q_cmd)[:3, 0])
        return ep.contact_point(cid) - r @ STOCK_PATCH, d, roll
    return ep.contact_point(cid), d, roll


def _fin_base(ep: Episode, cid: str, hp: np.ndarray) -> np.ndarray:
    """Point on the tab fin (or bar) where the jaws pinch: jaw tips 2 mm above the plate."""
    return hp + ep.ctl_axis(0, cid) * 4 * MM - ep.normal(cid) * (TAB_POINT_H - PINCH_HEIGHT)


# ---------------------------------------------------------------- primitives
def approach(ep: Episode, cid: str, offset: float = APPROACH_OFFSET, speed: float | None = None) -> Gen:
    tpl, _ = _tool(ep)
    if tpl == "stock":
        yield from ep.set_gripper(GRIP_OPEN if ep.sim.controls[cid].spec.cls not in ("knob",) else GRIP_WIDE, 0.3)
    p, d, roll = engage_pose(ep, cid)
    if ep.sim.controls[cid].spec.handle in ("tab", "pull") and tpl == "stock":
        yield from ep.set_gripper(GRIP_WIDE, 0.2)
    if ep.contact_waypoint_residual(p, d) > UNREACHABLE_MM:
        # The verdict will be `unreachable` regardless (Sec 8.2 rule 2); moving on would only drive the
        # best-effort pose into the panel and turn a reach failure into a force violation (D-023).
        raise Abort("unreachable")
    standoff = p - d * offset
    pre = standoff - d * 0.04
    ep.roll_ref = roll  # type: ignore[attr-defined]
    ep.axis_ref = d  # type: ignore[attr-defined]
    yield from ep.move_to(pre, d, speed, roll)
    yield from ep.move_to(standoff, d, speed, roll)
    return True


def contact(ep: Episode, cid: str, force: float = CONTACT_FORCE, speed: float | None = None) -> Gen:
    """Advance along the approach axis until tip force exceeds ``force`` (half speed near contact).

    For engagements that end in free space (slider side, pinch and hook poses) it stops at the
    engage point instead.
    """
    p, d, roll = engage_pose(ep, cid)
    h = ep.sim.controls[cid]
    tpl, prm = _tool(ep)
    free = h.spec.cls == "slider" or (tpl in ("stock", "hook") and (h.spec.handle in ("tab", "pull")
                                                                      or (h.spec.cls == "knob" and tpl == "stock")))
    vmax = min(speed or ep.cfg.limits.max_tip_speed_mps, ep.cfg.limits.max_tip_speed_mps)
    slow = min(vmax / 2, FINAL_SPEED)
    ext = 0.0 if free else tip_extension(ep)
    to_go = float((p - ep.tip_cmd) @ d) - ext
    if to_go > SLOW_ZONE:
        ok = yield from ep.move_to(ep.tip_cmd + d * (to_go - SLOW_ZONE), d, vmax, roll, guard_force=force)
        if not ok:
            return True
    if free:
        ok = yield from ep.move_to(p, d, slow, roll, guard_force=max(force, 1.0))
        return True
    overshoot = 15 * MM + (prm.get("depth", 0.0) * MM if tpl == "knob_cup" else 0.0)
    ok = yield from ep.move_to(p + d * (overshoot - ext), d, slow, roll, guard_force=force)
    return not ok


def travel_reached(ep: Episode, cid: str, margin: float = 1.05) -> bool:
    """Spec 8.1 guarded stop: the control reached its travel (with a small overshoot margin)."""
    h = ep.sim.controls[cid]
    q = ep.sim.control_q(cid) if h.flap is None else float(ep.sim.data.qpos[h.flap[2]])
    need = h.spec.travel if h.flap is None else 1.4
    return q - (ep.q_rest0[cid] if h.flap is None else 0.0) >= need * margin


def _cap(ep: Episode, force: float | None) -> float:
    cap = ep.cfg.force_cap if force is None else min(float(force), ep.cfg.force_cap)
    return min(cap, ep.cfg.limits.max_force_n)


def push(ep: Episode, cid: str, depth: float | None = None, force: float | None = None, speed: float | None = None,
         offset_u: float = 0.0, offset_v: float = 0.0) -> Gen:
    h = ep.sim.controls[cid]
    cap = _cap(ep, force)
    if depth is None:
        span = h.geo.lever_arm * h.spec.travel if h.geo.rotary else h.spec.travel
        depth = span * 1.25 + 3 * MM
    v = min(speed or 0.015, ep.cfg.limits.max_tip_speed_mps)
    roll = getattr(ep, "roll_ref", None)
    axis = getattr(ep, "axis_ref", ep.approach_dir(cid))
    p = ep.tip_cmd + ep.ctl_axis(0, cid) * offset_u + ep.ctl_axis(1, cid) * offset_v
    if offset_u or offset_v:
        yield from ep.move_to(p, axis, v, roll)
    ep.push_phase = True
    ep.reset_guard()
    s = 0.0
    try:
        while s < depth:
            step = ep.guarded_step(v * TICK_DT, cap)
            a = ep.actuation(cid)
            p = ep.tip_cmd + a * step
            q = ep.solve_tip(p, axis, roll)
            ep.tip_cmd = p
            s += step
            yield q
            if travel_reached(ep, cid):
                return True
            if ep.last_force >= cap - 0.25:
                return False
            if ep.trace.joint_limit_hits and ep.trace.joint_limit_hits[-1] != "gripper":
                return False
    finally:
        ep.push_phase = False
    return True


def rotate(ep: Episode, cid: str, angle: float | None = None, force: float = 3.0, speed: float = 0.6) -> Gen:
    h = ep.sim.controls[cid]
    hold = float(force)
    tpl, _ = _tool(ep)
    cap = _cap(ep, None)
    if tpl == "stock":
        yield from ep.set_gripper(GRIP_CLOSED, 0.4, stop_force=3.0)
    k = ep.ctl_axis(2, cid)  # positive control rotation about the control normal
    moved = ep.sim.control_q(cid) - ep.q_rest0[cid]
    target = angle if angle is not None else max(h.spec.travel - moved + 0.08, 0.05)
    axis = getattr(ep, "axis_ref", ep.approach_dir(cid))
    x0 = ep.ik.fk(ep.q_cmd)[:3, 0]
    roll_idx = 4
    jr = ep.sim.model.jnt_range[ep.sim.model.joint("wrist_roll").id]
    theta = 0.0
    depth = 0.0
    ep.push_phase = True
    try:
        while theta < abs(target):
            theta += speed * TICK_DT
            roll = rot_about(k, np.sign(target) * theta) @ x0
            # proportional normal-force hold: small steps so a stiff contact cannot jump past the cap
            err = min(hold, 0.6 * cap) - ep.last_force
            depth = float(np.clip(depth + np.clip(ROT_GAIN * err, -0.1 * MM, 0.05 * MM), -2 * MM, 4 * MM))
            p = ep.tip_cmd + axis * (depth - getattr(ep, "_rot_depth", 0.0))
            ep._rot_depth = depth  # type: ignore[attr-defined]
            q = ep.ik.track(p, axis, ep.q_cmd, roll_axis=roll, roll_weight=1.0)[0]
            q[-1] = ep.q_cmd[-1]
            ep.tip_cmd = p
            yield q
            if ep.last_force >= cap - 0.25:
                return False
            if q[roll_idx] <= jr[0] + 0.05 or q[roll_idx] >= jr[1] - 0.05:
                return False
    finally:
        ep.push_phase = False
        ep._rot_depth = 0.0  # type: ignore[attr-defined]
    return True


def _tangent(ep: Episode, cid: str) -> np.ndarray:
    """Direction the handle moves for increasing joint position (world)."""
    h = ep.sim.controls[cid]
    m = ep.sim.model
    if h.flap is not None:
        bid, jid = h.flap[0], h.flap[1]
    else:
        bid, jid = h.body_id, h.joint_id
    axis_w = ep.sim.data.xmat[bid].reshape(3, 3) @ m.jnt_axis[jid]
    if m.jnt_type[jid] == 3:  # hinge
        anchor = ep.sim.data.xanchor[jid]
        hp = handle_world(ep, cid)
        if hp is None:
            return axis_w
        t = np.cross(axis_w, hp - anchor)
        n = np.linalg.norm(t)
        return t / n if n > 1e-9 else axis_w
    return axis_w / np.linalg.norm(axis_w)


def hook_pull(ep: Episode, cid: str, distance: float | None = None, force: float | None = None,
              speed: float | None = None) -> Gen:
    h = ep.sim.controls[cid]
    tpl, prm = _tool(ep)
    hp = handle_world(ep, cid)
    if hp is None or h.spec.handle not in ("tab", "pull") or tpl not in ("hook", "stock"):
        return False
    cap = _cap(ep, force)
    v = min(speed or 0.02, ep.cfg.limits.max_tip_speed_mps)
    d = ep.approach_dir(cid)
    roll = _roll_for(ep, cid)
    xc = ep.ctl_axis(0, cid)
    r = _tool_rot(d, roll)
    if tpl == "hook":
        throat, tip, clear = _hook_geometry(prm, h.spec.handle)
        off = r @ (tip - throat)
        above = hp - xc * clear - d * 0.03 + off
        yield from ep.move_to(above, d, None, roll)
        yield from ep.move_to(hp - xc * clear + off, d, v, roll, guard_force=3.0)
        yield from ep.move_to(hp + off, d, v, roll, guard_force=3.0)
    else:
        grip = _fin_base(ep, cid, hp) - r @ PINCH_TOOL
        yield from ep.set_gripper(GRIP_WIDE, 0.2)
        yield from ep.move_to(grip - d * 0.03, d, None, roll)
        yield from ep.move_to(grip, d, v, roll, guard_force=3.0)
        yield from ep.set_gripper(GRIP_CLOSED, 0.5, stop_force=4.0)
    span = (h.geo.lever_arm if h.geo.rotary and h.flap is None else 0.03 if h.flap is not None else 1.0)
    budget = distance if distance is not None else span * (h.spec.travel if h.flap is None else 1.6) * 2.0 + 0.02
    s = 0.0
    ep.push_phase = True
    ep.reset_guard()
    try:
        while s < budget:
            step = ep.guarded_step(v * TICK_DT, cap)
            p = ep.tip_cmd + _tangent(ep, cid) * step
            q = ep.solve_tip(p, d, roll)
            ep.tip_cmd = p
            s += step
            yield q
            if distance is None and travel_reached(ep, cid):
                return True
            if ep.last_force >= cap - 0.25:
                return False
    finally:
        ep.push_phase = False
    return True


def slide(ep: Episode, cid: str, distance: float | None = None, normal_force: float = 1.0,
          speed: float | None = None) -> Gen:
    h = ep.sim.controls[cid]
    cap = _cap(ep, None)
    dist = distance if distance is not None else h.spec.travel * 1.15 + 2 * MM + 2 * MM + 3 * MM
    v = min(speed or 0.02, ep.cfg.limits.max_tip_speed_mps)
    roll = getattr(ep, "roll_ref", None)
    axis = getattr(ep, "axis_ref", ep.approach_dir(cid))
    a = ep.actuation(cid)
    ep.push_phase = True
    try:
        ok = yield from ep.move_to(ep.tip_cmd + a * dist, axis, v, roll, adaptive_cap=cap)
    finally:
        ep.push_phase = False
    return ok


def insert(ep: Episode, cid: str, depth: float | None = None, force: float | None = None,
           speed: float | None = None) -> Gen:
    h = ep.sim.controls[cid]
    cap = _cap(ep, force)
    dist = depth if depth is not None else h.spec.travel + 3 * MM
    v = min(speed or 0.01, ep.cfg.limits.max_tip_speed_mps)
    roll = getattr(ep, "roll_ref", None)
    axis = getattr(ep, "axis_ref", ep.approach_dir(cid))
    gap_dir = ep.ctl_axis(1, cid)
    start = ep.tip_cmd.copy()
    ep.push_phase = True
    try:
        for k, shift in enumerate((0.0, 0.5 * MM, -0.5 * MM, 1.0 * MM, -1.0 * MM)):
            if k:
                yield from ep.move_to(start - axis * 3 * MM + gap_dir * shift, axis, v, roll)
            ok = yield from ep.move_to(start + gap_dir * shift + axis * dist, axis, v, roll, adaptive_cap=cap)
            moved = ep.sim.control_q(cid) - ep.q_rest0[cid]
            if ok or moved > 2 * MM:
                return ok
    finally:
        ep.push_phase = False
    return False


def release(ep: Episode, cid: str) -> Gen:
    tpl, _ = _tool(ep)
    if tpl == "stock":
        yield from ep.set_gripper(GRIP_WIDE if ep.sim.controls[cid].spec.cls == "knob" else GRIP_OPEN, 0.3)
    axis = getattr(ep, "axis_ref", ep.approach_dir(cid))
    roll0 = getattr(ep, "roll_ref", None)
    back = ep.tip_cmd - axis * 5 * MM
    yield from ep.move_to(back, axis, None, None)
    if ep.sim.controls[cid].spec.cls == "knob":
        # unwind the wrist roll while clear of the knob
        q = ep.q_cmd.copy()
        start = q[4]
        _, _, _ = ep.ik.track(ep.tip_cmd, axis, q, roll_axis=roll0) if roll0 is not None else (q, 0, 0)
        steps = max(1, int(abs(start) / (1.2 * TICK_DT)))
        for i in range(1, steps + 1):
            qq = ep.q_cmd.copy()
            qq[4] = start * (1 - i / steps)
            yield qq
        ep.q_cmd[4] = 0.0
        ep.tip_cmd = ep.ik.fk(ep.q_cmd)[:3, 3]
    return True


def retreat(ep: Episode, cid: str, distance: float = 0.05) -> Gen:
    tpl, _ = _tool(ep)
    if tpl == "stock" and ep.q_cmd[-1] < GRIP_OPEN - 0.1:
        yield from ep.set_gripper(GRIP_OPEN, 0.3)
    axis = getattr(ep, "axis_ref", ep.approach_dir(cid))
    roll = getattr(ep, "roll_ref", None)
    yield from ep.unload()
    if ep.last_force > CONTACT_FORCE and distance > SLOW_ZONE / 2:  # leave a dragging contact slowly
        yield from ep.move_to(ep.tip_cmd - axis * (SLOW_ZONE / 2), axis, FINAL_SPEED, roll)
        distance -= SLOW_ZONE / 2
    yield from ep.move_to(ep.tip_cmd - axis * distance, axis, None, roll)
    return True


def finish(ep: Episode) -> Gen:
    """End-of-program back-off (not a library primitive): release, retreat, let springs settle."""
    axis = getattr(ep, "axis_ref", None)
    if axis is not None:
        dist = float(np.linalg.norm(ep.tip_cmd - ep.contact_point(ep.cid)))
        if dist < APPROACH_OFFSET:  # still at the control (no retreat in the program)
            if ep.q_cmd[-1] < GRIP_OPEN - 0.1 and ep.sim.mounted is None:
                yield from ep.set_gripper(GRIP_OPEN, 0.3)
            yield from ep.unload()
            try:
                yield from ep.move_to(ep.tip_cmd - axis * 0.05, axis, None, getattr(ep, "roll_ref", None))
            except Exception:  # pragma: no cover - workspace edge; settling still happens below
                pass
    yield from ep.hold(0.4)
    return True


PICK_OPEN = 1.0  # gripper joint while descending beside an object (the V1 probe's setting)
PICK_SIDE = 2 * MM  # fixed jaw's inner face to the object on the way down
PICK_TABLE_CLEAR = 3 * MM  # jaw tips never go closer than this to the table


def pick_depth(spec, past_equator: float = 5 * MM) -> float:
    """Default depth of the tool point below an object's top: the V1 probe's grip (sim/grasp_probe.py)."""
    w, hgt = spec.size_mm[0] * MM, spec.size_mm[1] * MM
    if spec.shape == "sphere":
        return w / 2 + past_equator - STOCK_TIP_EXT
    return min(17 * MM, hgt - STOCK_TIP_EXT - PICK_TABLE_CLEAR)


def grasp(ep: Episode, cid: str, depth: float | None = None, side: float = PICK_SIDE, close: float = GRIP_CLOSED,
          close_s: float = 0.6, lift: float = 0.05, hold_s: float = 1.2, speed: float = 0.03) -> Gen:
    """Pick a free object from above with the stock jaws: open, descend beside it with the fixed jaw
    ``side`` clear, close toward ``close`` (the servo stalls on the object first), lift by ``lift`` and
    hold for ``hold_s``.  ``depth`` is how far below the object's top the tool point goes.  Returns
    False if the descent lands on something (the opening does not fit) or a tool is mounted."""
    h = ep.sim.controls[cid]
    if h.spec.cls != "grasp" or ep.sim.mounted is not None:
        return False
    w, hgt = h.spec.size_mm[0] * MM, h.spec.size_mm[1] * MM
    depth = pick_depth(h.spec) if depth is None else depth
    depth = min(depth, hgt - STOCK_TIP_EXT - PICK_TABLE_CLEAR)
    down = ep.approach_dir(cid)
    roll = ep.sim.ctl_dir_world(cid, h.geo.roll_ctl)
    target = ep.contact_point(cid) + roll * (w / 2 + side) + down * depth
    v = min(speed, ep.cfg.limits.max_tip_speed_mps)
    ep.axis_ref, ep.roll_ref = down, roll  # type: ignore[attr-defined]
    yield from ep.set_gripper(PICK_OPEN, 0.3)
    yield from ep.move_to(target - down * 0.04, down, None, roll)
    if not (yield from ep.move_to(target, down, v, roll, guard_force=3.0)):
        return False
    ep.squeezing = True
    yield from ep.set_gripper(close, close_s)
    yield from ep.hold(0.3)
    yield from ep.move_to(ep.tip_cmd - down * lift, down, v, roll)
    yield from ep.hold(hold_s)
    return True


MOTOR = {
    "approach": approach,
    "contact": contact,
    "push": push,
    "rotate": rotate,
    "hook_pull": hook_pull,
    "slide": slide,
    "insert": insert,
    "release": release,
    "retreat": retreat,
    "grasp": grasp,
}
