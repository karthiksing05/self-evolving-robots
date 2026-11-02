"""Program interpreter and episode recorder (spec 8.1 ``executor.py``).

``run_program(sim, program, control_id, ...)`` interprets a program tree (``seq``, ``guard``,
``repeat``, ``branch``, ``with``, ``bind`` and primitive calls) on one control and returns the
``StepTrace``.  Motor primitives are generators that yield joint targets at 50 Hz; the episode
steps the sim 10 times per target, classifies contacts, enforces hard limits (force, tip speed,
workspace), runs the servo thermal model, and evaluates the control's success check every tick.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field

import mujoco
import numpy as np

from sea.compose import checks
from sea.contracts import ARM_JOINTS, GRASP_CLASS, HardLimits, Node, PrimitiveDef, Program, StepTrace
from sea.control.frames import STOCK_URDF, rot_about
from sea.control.ik import IK
from sea.sim.build_model import Sim
from sea.sim.sensors import (
    LOAD_ABORT_S,
    N_ARM,
    PAD_GRIP_N,
    PAD_LOST_N,
    STALL_LAG_RAD,
    STALL_LOAD,
    SensorFrame,
    SensorLog,
    Sensors,
    self_model_for,
)

TICK_STEPS = 10  # sim steps per joint target (50 Hz targets at 500 Hz physics)
TICK_DT = 0.002 * TICK_STEPS
PINCH_WINDOW_S = 0.05
TOOL_SWAP_S = 5.0
CRUISE_FRACTION = 0.85  # commanded cruise speed / hard limit: margin for servo lag and catch-up (D-025)
SWAP_TIP = np.array([0.17, 0.0, 0.14])  # stock tool point during a swap: well above any panel
HOME_Q = np.array([0.0, -0.386, 0.369, 1.588, 0.0, 0.8])
# The grasp world's home: the arm's lowest point 112 mm above the table over the object area, clear of
# the tallest object (85 mm).  From HOME_Q the fixed jaw sat 73 mm up, inside the potted meat can (D-040).
GRASP_HOME_Q = np.array([0.0, -0.45, 0.15, 1.6, 0.0, 0.8])
GRIP_OPEN = 0.6  # moving jaw tip just behind the fixed jaw: presses lead with the fixed jaw
GRIP_WIDE = 1.2
GRIP_CLOSED = -0.17
MAX_REPEAT = 6
GUARD_STOP_N = 0.25  # primitives stop when force >= cap - this
CONTACT_N = 0.2  # tip force that counts as touching for the guard
PROBE_STEP_M = 2e-5  # first step into an unmeasured contact
MIN_GUARD_STEP_M = 2e-6
GUARD_K_DECAY = 0.9  # per tick, on the remembered contact stiffness
GUARD_GROWTH = 1.5  # max step growth per tick while in contact
ROLL_RATE = 2.0  # rad/s: wrist-roll alignment rate inside moves
ACCEL = 1.0  # m/s^2: tip acceleration at the start and end of a move
UNLOAD_MIN_M = 2e-5  # commanded-vs-measured tip gap worth unloading before a retreat
TRACK_TOL_MM = 0.05  # per-tick IK tolerance: looser tolerances alternate and jitter the tip speed

Gen = Generator[np.ndarray, None, bool]

_IK_CACHE: dict[str, IK] = {}


def stock_ik() -> IK:
    key = str(STOCK_URDF)
    if key not in _IK_CACHE:
        _IK_CACHE[key] = IK(STOCK_URDF)
    return _IK_CACHE[key]


class Abort(Exception):
    """Raised inside an episode to stop the program (hard limit, time limit)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class EpisodeConfig:
    limits: HardLimits
    force_cap: float  # effective guard cap for this control (posture fraction x class cap, <= hard limit)
    class_cap: float
    banned: frozenset[str]
    time_limit_s: float = 20.0
    perception_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    library: dict[str, PrimitiveDef] = field(default_factory=dict)
    param_overrides: dict[str, dict[str, float]] = field(default_factory=dict)  # prim -> arg -> value
    thermal: object | None = None  # spec.ledger.ServoThermal
    sense: bool = True  # read the robot's sensors every tick and run the reflexes on them (D-036); physics unchanged
    sensor_seed: int | None = 0


class SimCheckEnv:
    """``CheckEnv`` over live sim state (ground truth for the verifier's success checks)."""

    def __init__(self, ep: Episode):
        self.ep = ep

    def _cid(self, ref: str) -> str:
        return self.ep.cid if ref == checks.TARGET else ref

    def moved(self, ref: str) -> float:
        cid = self._cid(ref)
        return self.ep.sim.control_q(cid) - self.ep.q_rest0[cid]

    def travel(self, ref: str) -> float:
        return self.ep.sim.controls[self._cid(ref)].spec.travel

    def latched(self, ref: str) -> bool:
        return self.ep.sim.controls[self._cid(ref)].dyn.latched

    def detents(self, ref: str) -> int:
        return self.ep.sim.controls[self._cid(ref)].dyn.crossings

    def force(self) -> float:
        return self.ep.last_force

    def banned_contact(self) -> bool:
        return self.ep.banned_seen

    def held_s(self, ref: str) -> float:
        """Seconds the episode's target has stayed at least its travel from rest (0 for other refs)."""
        if self._cid(ref) != self.ep.cid or self.ep.held_since is None:
            return 0.0
        return self.ep.t - self.ep.held_since


class Episode:
    """One attempt on one control: commanded joints, recording, contacts, limits."""

    def __init__(self, sim: Sim, cid: str, cfg: EpisodeConfig, trace: StepTrace):
        self.sim = sim
        self.cid = cid
        self.cfg = cfg
        self.trace = trace
        self.t = 0.0
        self.t_offset = 0.0  # tool swaps advance the clock without physics
        self.q_cmd = sim.arm_q()
        self.last_force = 0.0
        # The force on the target the class cap reads is the one held for a whole control tick (D-053).
        # A shorter pulse is an impact: the rotor's momentum meeting the drivetrain's compliance, which the
        # servo does not command and which the gear train's unpublished stiffness decides.
        self.peak_ctl_force = 0.0
        self.peak_ctl_impact = 0.0  # the largest single-step force
        self._ctl_window: deque[float] = deque(maxlen=TICK_STEPS)
        self.banned_seen = False
        self.contact_types: set[str] = set()
        self.jaw_touch: dict[tuple[str, str], float] = {}
        self.push_phase = False
        self.held_since: float | None = None  # when the target last reached its travel (held_for)
        self.squeezing = False  # set by ``grasp`` once the jaws close on its target
        self.carrying = False  # set by ``grasp`` once it has lifted; cleared when the program ends
        self.success = checks.parse(sim.controls[cid].spec.success)
        self.q_rest0 = {c: sim.control_q(c) for c in sim.controls}
        self.env = SimCheckEnv(self)
        self.ik_base = stock_ik()
        self.ik = self.ik_base.for_tip(sim.tip_transform())
        self.tip_cmd = self.ik.fk(self.q_cmd)[:3, 3]
        self.axis_cmd = self.ik.fk(self.q_cmd)[:3, 2]
        self.contact_res_mm: float | None = None
        self.sensors: Sensors | None = None
        self.sense_log: SensorLog | None = None
        self.frame: SensorFrame | None = None  # the latest reading: what System 1's guards may use
        self._grip_prev = float(self.q_cmd[-1])
        self._losing_ticks = 0  # consecutive ticks with a stalled arm joint behind its command
        self._pads_gripped = False
        if cfg.sense:
            self.sensors = Sensors(sim, seed=cfg.sensor_seed, self_model=self_model_for(sim.design))
            self.sense_log = SensorLog(self.sensors.mounted)
        self.max_ticks = int(np.ceil(cfg.time_limit_s / TICK_DT))
        self.ticks = 0
        self._last_contact_log: dict[tuple[str, str], float] = {}

    # ---------------------------------------------------------------- geometry (perceived)
    @property
    def handle(self):
        return self.sim.controls[self.cid]

    def remount(self, slot: int | None) -> None:
        """Swap tools at the raised swap pose (the swap's 5 s covers the trip; no physics runs)."""
        q = self.ik_base.solve(SWAP_TIP, np.array([0.0, 0.0, -1.0]))[0]
        q[-1] = GRIP_CLOSED if slot is not None else self.q_cmd[-1]  # jaws close behind the mount block
        self.sim.mount(slot)
        self.sim.set_arm_q(q)
        self.sim.data.ctrl[: len(q)] = q
        self.q_cmd = q.copy()
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.ik = self.ik_base.for_tip(self.sim.tip_transform())
        t = self.ik.fk(self.q_cmd)
        self.tip_cmd = t[:3, 3]
        self.axis_cmd = t[:3, 2]

    def world_point(self, p_body: np.ndarray, cid: str | None = None) -> np.ndarray:
        cid = cid or self.cid
        return self.sim.control_point_world(cid, p_body) + self.cfg.perception_offset

    def approach_dir(self, cid: str | None = None) -> np.ndarray:
        h = self.sim.controls[cid or self.cid]
        a = h.T_world_ctl[:3, :3] @ h.geo.approach_ctl
        return a / np.linalg.norm(a)

    def normal(self, cid: str | None = None) -> np.ndarray:
        h = self.sim.controls[cid or self.cid]
        return h.T_world_ctl[:3, 2].copy()

    def ctl_axis(self, i: int, cid: str | None = None) -> np.ndarray:
        h = self.sim.controls[cid or self.cid]
        return h.T_world_ctl[:3, i].copy()

    def actuation(self, cid: str | None = None) -> np.ndarray:
        h = self.sim.controls[cid or self.cid]
        a = self.sim.control_dir_world(cid or self.cid, h.geo.actuation_body)
        return a / np.linalg.norm(a)

    def contact_point(self, cid: str | None = None) -> np.ndarray:
        h = self.sim.controls[cid or self.cid]
        return self.world_point(h.geo.contact_body, cid)

    def tip_actual(self) -> np.ndarray:
        return self.sim.tool_frame_world() @ self.sim.tip_transform()

    # ---------------------------------------------------------------- commanding
    def solve_tip(self, pos: np.ndarray, axis: np.ndarray, roll: np.ndarray | None = None) -> np.ndarray:
        """Joint target that puts the tool tip at ``pos`` with approach ``axis``."""
        lo, hi = self.cfg.limits.workspace_box
        box_lo = np.array(lo.as_list())
        box_hi = np.array(hi.as_list())
        if np.any(pos < box_lo - 1e-9) or np.any(pos > box_hi + 1e-9):
            raise Abort("workspace")
        q, _, _ = self.ik.track(pos, axis, self.q_cmd, roll_axis=roll, tol_mm=TRACK_TOL_MM)
        q[-1] = self.q_cmd[-1]
        return q

    def contact_waypoint_residual(self, pos: np.ndarray, axis: np.ndarray) -> float:
        """Full multi-seed IK residual at the contact waypoint (feeds ``unreachable``)."""
        _, res, ang = self.ik.solve(pos, axis, tol_mm=2.0, tol_deg=5.0)
        if ang > 20.0:  # the 5-DOF arm cannot point the tool this way here
            res = max(res, 10.0 + ang)
        self.contact_res_mm = max(self.contact_res_mm or 0.0, res)
        self.trace.ik_residual_mm = self.contact_res_mm
        return res

    # ---------------------------------------------------------------- stepping
    def tick(self, q_target: np.ndarray) -> None:
        if self.ticks >= self.max_ticks:
            raise Abort("time_limit")
        sim = self.sim
        q_target = np.asarray(q_target, dtype=float)
        jr = sim.servo_ranges()
        clipped = np.clip(q_target, jr[:, 0], jr[:, 1])
        for j, (a, b) in enumerate(zip(q_target, clipped, strict=True)):
            if abs(a - b) > 1e-6 and ARM_JOINTS[j] not in self.trace.joint_limit_hits:
                self.trace.joint_limit_hits.append(ARM_JOINTS[j])
        self.q_cmd = clipped
        sim.data.ctrl[:] = clipped
        # MuJoCo resets the state (and counts a warning) when the integration blows up.
        bad = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQPOS)
        if self.ticks == 0:
            self._bad0 = sum(int(sim.data.warning[w].number) for w in bad)
        fmax = 0.0
        paused = False
        for _ in range(TICK_STEPS):
            sim.step(1)
            f, fc = self._process_contacts()
            fmax = max(fmax, f)
            self.peak_ctl_impact = max(self.peak_ctl_impact, fc)
            self._ctl_window.append(fc)
            if len(self._ctl_window) == TICK_STEPS:
                self.peak_ctl_force = max(self.peak_ctl_force, min(self._ctl_window))
            if self.cfg.thermal is not None:
                paused = self.cfg.thermal.step(np.abs(sim.data.actuator_force), 0.002) or paused  # type: ignore[attr-defined]
        unstable = sum(int(sim.data.warning[w].number) for w in bad) > getattr(self, "_bad0", 0)
        if unstable or not (np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()):
            raise Abort("sim_unstable")  # the only abort that leaves a trace incomplete (verdict 0)
        self.last_force = fmax
        self.t += TICK_DT
        self.ticks += 1
        holding = False
        if self.sensors is not None and self.sense_log is not None:
            self.frame = self.sensors.read()
            holding = self.q_cmd[-1] < GRIP_OPEN - 0.1 and not self.q_cmd[-1] > self._grip_prev + 1e-6
            self.sense_log.add(self.t, self.frame, grip_closed=holding, carrying=self.carrying, tick_s=TICK_DT)
        self._grip_prev = float(self.q_cmd[-1])
        self._record(fmax)
        if fmax > self.cfg.limits.max_force_n:
            raise Abort("force_limit")
        if self.frame is not None and self.squeezing:
            self._holding_reflexes(self.frame, holding)
        while paused and self.cfg.thermal is not None:  # servo cool-down: time passes, no motion
            self.t_offset += 1.0
            paused = self.cfg.thermal.step(np.zeros(len(ARM_JOINTS)), 1.0)  # type: ignore[attr-defined]

    def _holding_reflexes(self, f: SensorFrame, holding: bool) -> None:
        """System 1's reflexes while the jaws hold a grasp target, on what the robot feels (D-036).

        ``load_limit``: an arm joint at stall falls behind its command for ``LOAD_ABORT_S``; the arm
        cannot carry the payload.  ``slip``: the pads had a grip and lost it with the jaws still
        commanded shut; the object left the jaws.  Slip needs the jaw pads mounted."""
        losing = (np.abs(f.load[:N_ARM]) >= STALL_LOAD) & (np.abs(self.q_cmd[:N_ARM] - f.q[:N_ARM]) > STALL_LAG_RAD)
        self._losing_ticks = self._losing_ticks + 1 if losing.any() else 0
        if self._losing_ticks * TICK_DT >= LOAD_ABORT_S:
            raise Abort("load_limit")
        if f.jaw_force_n is not None and holding:
            pads = float(f.jaw_force_n.sum())
            self._pads_gripped = self._pads_gripped or pads >= PAD_GRIP_N
            if self._pads_gripped and pads < PAD_LOST_N:
                raise Abort("slip")

    def _process_contacts(self) -> tuple[float, float]:
        """The step's robot-vs-world normal force (summed), and the force the robot applies to its target: per
        robot body, the magnitude of the vector sum of its contact forces on the target, at the largest body.
        Summing magnitudes instead would count a squeeze once per jaw and once per contact point, so an object
        resting in several convex pieces of one jaw would read several times the force that jaw applies."""
        sim, d, m = self.sim, self.sim.data, self.sim.model
        total = 0.0
        on_target: dict[int, np.ndarray] = {}
        f6 = np.zeros(6)
        now = self.t
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            kind1, kind2 = sim.geom_kind.get(g1), sim.geom_kind.get(g2)
            if (kind1 is None) == (kind2 is None):
                continue
            rg, og = (g1, g2) if kind1 is not None else (g2, g1)
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            cid = sim.body_to_control.get(int(m.geom_bodyid[og]))
            kind = sim.geom_kind[rg]
            # The jaws' squeeze on a grasp target is the task, bounded by the servo and the grasp class
            # cap; the hard force limit bounds every other contact (configs/limits_grasp.yaml).
            if not (self.squeezing and cid == self.cid and kind in ("jaw_fixed", "jaw_moving")):
                total += fn
            if cid is None:
                continue
            ctype: str
            if kind == "tool":
                ctype = sim.geom_contact[rg]
            elif kind in ("jaw_fixed", "jaw_moving"):
                self.jaw_touch[(cid, kind)] = now
                other = "jaw_moving" if kind == "jaw_fixed" else "jaw_fixed"
                t_other = self.jaw_touch.get((cid, other))
                ctype = "pinch" if t_other is not None and now - t_other <= PINCH_WINDOW_S else "edge"
            else:
                ctype = "edge"
            if cid == self.cid:
                f_world = c.frame.reshape(3, 3).T @ f6[:3]  # on geom2, from geom1 (MuJoCo's contact frame)
                body = int(m.geom_bodyid[rg])
                on_target[body] = on_target.get(body, np.zeros(3)) + (f_world if rg == g1 else -f_world)
                self.contact_types.add(ctype)
                if ctype in self.cfg.banned:
                    self.banned_seen = True
            self._log_contact(ctype, cid)
        return total, max((float(np.linalg.norm(v)) for v in on_target.values()), default=0.0)

    def _log_contact(self, ctype: str, cid: str) -> None:
        key = (ctype, cid)
        last = self._last_contact_log.get(key)
        if last is None or self.t - last >= 0.1:
            self._last_contact_log[key] = self.t
            self.trace.contacts.append((round(self.t, 4), ctype, cid))  # type: ignore[arg-type]

    def _record(self, fmax: float) -> None:
        tr = self.trace
        tr.t.append(round(self.t + self.t_offset, 4))
        tr.q.append([round(float(v), 5) for v in self.sim.arm_q()])
        tr.tip_force.append(round(fmax, 4))
        tr.control_q.append(round(self.sim.control_q(self.cid), 6))
        up = self.sim.control_q(self.cid) - self.q_rest0[self.cid] >= self.handle.spec.travel - 1e-9
        self.held_since = (self.held_since if self.held_since is not None else self.t) if up else None
        touching = any(c[2] == self.cid and abs(c[0] - self.t) < 1e-9 for c in tr.contacts[-6:])
        tr.in_contact.append(touching or self._touching())
        tr.push_phase.append(self.push_phase)
        if tr.success_time_s is None and checks.evaluate(self.success, self.env):
            tr.success_time_s = round(self.t + self.t_offset, 4)

    def _touching(self) -> bool:
        return any(k[0] == self.cid and abs(v - self.t) < TICK_DT + 1e-9 for k, v in self.jaw_touch.items()) or (
            self.peak_ctl_impact > 0 and self.last_ctl_contact())

    def last_ctl_contact(self) -> bool:
        d, sim = self.sim.data, self.sim
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if g in sim.geom_kind and sim.body_to_control.get(int(sim.model.geom_bodyid[o])) == self.cid:
                    return True
        return False

    def guarded_step(self, nominal: float, cap: float) -> float:
        """Step length (m) for a force-guarded move.

        Shrinks as force nears the cap.  In contact it plans with the contact stiffness measured
        over earlier steps (dF/dx, remembered with a slow decay so a bounce does not erase it):
        the next step may add at most half the remaining margin.  The first step into an
        unmeasured contact is a short probe and steps then grow at most 1.5x per tick, so a rigid
        control (or one hidden by stiction) cannot jump past the cap.
        """
        f = self.last_force
        margin = cap - GUARD_STOP_N - f
        frac = float(np.clip((cap - f) / max(0.5 * cap, 1e-6), 0.02, 1.0))
        step = nominal * frac
        prev_f, prev_step = getattr(self, "_g_prev", (None, None))
        k = getattr(self, "_g_k", None)
        if prev_f is not None and prev_step and f > prev_f + 1e-6:
            k_new = (f - prev_f) / prev_step
            k = k_new if k is None else max(k_new, GUARD_K_DECAY * k)
        elif k is not None:
            k *= GUARD_K_DECAY
        self._g_k = k
        if f > CONTACT_N:
            if k is None:
                step = min(step, PROBE_STEP_M)
            else:
                step = min(step, max(0.5 * margin / k, 0.0))
            if prev_step:  # ramp, never jump: stiction can hide the stiffness for a few ticks
                step = min(step, GUARD_GROWTH * prev_step)
            step = max(step, MIN_GUARD_STEP_M) if margin > 0 else 0.0
        self._g_prev = (f, step)
        return step

    def reset_guard(self) -> None:
        self._g_prev = (None, None)
        self._g_k = None

    # ---------------------------------------------------------------- motion helpers (sub-generators)
    def unload(self, seconds: float = 0.1) -> Iterator[np.ndarray]:
        """Release a contact preload gently before moving away.

        Ramps the commanded tip back to the measured tip.  Without this the servos spring a
        held-back tip off the control at the start of a retreat.
        """
        start = self.tip_cmd.copy()
        gap = self.tip_actual()[:3, 3] - start
        if np.linalg.norm(gap) < UNLOAD_MIN_M or self.last_force < CONTACT_N:
            return
        n = max(1, int(round(seconds / TICK_DT)))
        for i in range(1, n + 1):
            p = start + gap * (i / n)
            q = self.solve_tip(p, self.axis_cmd, None)
            self.tip_cmd = p
            yield q

    def hold(self, seconds: float) -> Iterator[np.ndarray]:
        for _ in range(max(1, int(round(seconds / TICK_DT)))):
            yield self.q_cmd.copy()

    def set_gripper(self, target: float, seconds: float = 0.4, stop_force: float | None = None) -> Iterator[np.ndarray]:
        start = float(self.q_cmd[-1])
        n = max(1, int(round(seconds / TICK_DT)))
        for k in range(1, n + 1):
            q = self.q_cmd.copy()
            q[-1] = start + (target - start) * k / n
            yield q
            if stop_force is not None and self.last_force > stop_force:
                break

    def move_to(self, pos: np.ndarray, axis: np.ndarray, speed: float | None = None, roll: np.ndarray | None = None,
                guard_force: float | None = None, adaptive_cap: float | None = None) -> Generator[np.ndarray, None, bool]:
        """Straight tip line from the current commanded tip to ``pos``.

        Returns False if ``guard_force`` (or ``adaptive_cap``) stopped the motion early.
        Commanded speed never exceeds the hard limit; with ``adaptive_cap`` the step shrinks as
        force approaches the cap so the stop lands within a fraction of a newton of it.
        """
        vmax = self.cfg.limits.max_tip_speed_mps
        v = min(speed or vmax, vmax * CRUISE_FRACTION)
        start = self.tip_cmd.copy()
        delta = np.asarray(pos, dtype=float) - start
        dist = float(np.linalg.norm(delta))
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        roll_cmd = None if roll is None else self.ik.fk(self.q_cmd)[:3, 0]
        roll_done = roll is None or _turn_toward(roll_cmd, roll, axis, 0.0)[1]
        if dist < 1e-6 and roll_done:
            return True
        u = delta / dist if dist >= 1e-6 else np.zeros(3)
        axis0 = self.axis_cmd.copy()
        s = 0.0
        self.reset_guard()
        v_now = 0.0
        while s < dist - 1e-9 or not roll_done:
            # trapezoidal speed: moves start and end at rest (a step in velocity makes the servos
            # overshoot the commanded tip speed)
            v_now = min(v, v_now + ACCEL * TICK_DT, float(np.sqrt(2.0 * ACCEL * max(dist - s, 0.0))) + ACCEL * TICK_DT)
            step = min(v_now * TICK_DT, dist - s) if s < dist - 1e-9 else 0.0
            if adaptive_cap is not None and step > 0.0:
                step = self.guarded_step(step, adaptive_cap)
            s = min(dist, s + step)
            p = start + u * s
            w = min(1.0, s / max(min(dist, 0.05), 1e-6)) if dist >= 1e-6 else 1.0
            ax = (1 - w) * axis0 + w * axis
            ax = ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-9 else axis
            if roll is not None:  # turn the wrist at a bounded rate; a jump whips the tool point
                roll_cmd, roll_done = _turn_toward(roll_cmd, roll, ax, ROLL_RATE * TICK_DT)
            q = self.solve_tip(p, ax, roll_cmd)
            self.tip_cmd = p
            self.axis_cmd = ax
            yield q
            if guard_force is not None and self.last_force >= guard_force:
                return False
            if adaptive_cap is not None and self.last_force >= adaptive_cap - 0.25:
                return False
        return True


def _turn_toward(cur: np.ndarray, target: np.ndarray, axis: np.ndarray, max_angle: float) -> tuple[np.ndarray, bool]:
    """Rotate ``cur`` about ``axis`` toward ``target`` (both projected off the axis) by at most
    ``max_angle``; returns the new direction and whether it reached the target (within 2 deg)."""
    a = axis / np.linalg.norm(axis)
    p0 = cur - a * float(cur @ a)
    p1 = target - a * float(target @ a)
    n0, n1 = float(np.linalg.norm(p0)), float(np.linalg.norm(p1))
    if n0 < 1e-6 or n1 < 1e-6:
        return target, True
    p0, p1 = p0 / n0, p1 / n1
    ang = float(np.arctan2(a @ np.cross(p0, p1), p0 @ p1))
    if abs(ang) <= max(max_angle, np.radians(2.0)):
        return p1, abs(ang) <= np.radians(2.0) or abs(ang) <= max_angle
    return rot_about(a, np.sign(ang) * max_angle) @ p0, False


# ---------------------------------------------------------------- interpretation
def _prim_params(name: str, library: dict[str, PrimitiveDef]) -> list[str]:
    d = library.get(name)
    return list(d.params) if d is not None else []


def resolve_args(node: Node, library: dict[str, PrimitiveDef], scope: dict[str, float | str]) -> tuple[str, str | None, dict]:
    """Map positional args onto the primitive's declared params; returns (name, target, kwargs)."""
    assert node.prim is not None
    name = node.prim.name
    params = _prim_params(name, library)
    args = dict(node.bind_args) if node.op == "bind" else {}
    args.update(node.prim.args)
    target: str | None = None
    kwargs: dict[str, float | str] = {}
    pos_keys = sorted((k for k in args if k.startswith("_") and k[1:].isdigit()), key=lambda k: int(k[1:]))
    pi = 0
    for k in pos_keys:
        v = args[k]
        if isinstance(v, str) and target is None:
            target = v
            continue
        if pi < len(params):
            kwargs[params[pi]] = v
        pi += 1
    for k, v in args.items():
        if k in pos_keys:
            continue
        if k == "target" and isinstance(v, str):
            target = v
            continue
        kwargs[k] = v
    for k, v in list(kwargs.items()):  # composite placeholders: "$name" -> scope value
        if isinstance(v, str) and v.startswith("$") and v != checks.TARGET:
            kwargs[k] = scope.get(v[1:], v)
    return name, target, kwargs


def exec_node(ep: Episode, node: Node, scope: dict | None = None) -> Gen:
    from sea.control import motor

    scope = scope or {}
    op = node.op
    if op == "seq":
        for child in node.children:
            ok = yield from exec_node(ep, child, scope)
            if not ok:
                return False
        return True
    if op == "guard":
        until = checks.parse(node.check or "")
        cap = checks.parse(node.cap or "")
        child = exec_node(ep, node.children[0], scope)
        prev = ep.push_phase
        ep.push_phase = True
        try:
            q = next(child)
            while True:
                yield q
                if checks.evaluate(until, ep.env):
                    child.close()
                    return True
                if not checks.evaluate(cap, ep.env):
                    child.close()
                    _mark_failed(ep, getattr(ep, "active_prim", ""))
                    return False
                q = child.send(None)
        except StopIteration:
            return checks.evaluate(until, ep.env)
        finally:
            ep.push_phase = prev
    if op == "repeat":
        until = checks.parse(node.check) if node.check else None
        n = node.n if node.n is not None else MAX_REPEAT
        for _ in range(n):
            ok = yield from exec_node(ep, node.children[0], scope)
            if until is not None and checks.evaluate(until, ep.env):
                return True
            if not ok:
                return False
        return until is None
    if op == "branch":
        cond = checks.evaluate(checks.parse(node.check or ""), ep.env)
        return (yield from exec_node(ep, node.children[0 if cond else 1], scope))
    if op == "with":
        slot = int(node.tool_index or 0)
        if slot not in ep.sim.tool_geoms:
            return False
        prev = ep.sim.mounted
        ep.remount(slot)
        ep.t_offset += TOOL_SWAP_S
        ep.trace.tool_swaps += 1
        try:
            ok = yield from exec_node(ep, node.children[0], scope)
        finally:
            ep.remount(prev)
        return ok
    # prim / bind
    name, target, kwargs = resolve_args(node, ep.cfg.library, scope)
    cid = ep.cid if target in (None, checks.TARGET) else target
    if cid not in ep.sim.controls:
        return False
    lib = ep.cfg.library.get(name)
    if lib is not None and lib.kind == "composite" and lib.body is not None:
        inner = {k: float(np.mean(v)) for k, v in lib.params.items()}
        inner.update({k: v for k, v in kwargs.items() if not isinstance(v, str)})
        return (yield from exec_node(ep, lib.body.root, inner))
    fn = motor.MOTOR.get(name)
    if fn is None:
        return False
    kwargs = {**kwargs, **ep.cfg.param_overrides.get(name, {})}
    prev_cid = ep.cid
    ep.cid_target = cid  # type: ignore[attr-defined]
    ep.active_prim = name  # type: ignore[attr-defined]
    try:
        ok = yield from fn(ep, cid, **kwargs)
    finally:
        ep.cid_target = prev_cid  # type: ignore[attr-defined]
    if not ok:
        _mark_failed(ep, name)
    return ok


def _mark_failed(ep: Episode, name: str) -> None:
    """Remember the first motor primitive that failed (System 1 searches its parameters)."""
    if name and not ep.trace.failed_primitive:
        ep.trace.failed_primitive = name


def drive(ep: Episode, gen: Gen) -> tuple[bool, str]:
    """Run a primitive/program generator to completion, one tick per yielded joint target."""
    try:
        q = next(gen)
        while True:
            ep.tick(q)
            q = gen.send(None)
    except StopIteration as stop:
        return bool(stop.value), ""
    except Abort as a:
        gen.close()
        _mark_failed(ep, getattr(ep, "active_prim", ""))
        return False, a.reason


def home_q(sim: Sim) -> np.ndarray:
    """The pose an episode starts from: ``GRASP_HOME_Q`` over a grasp table, ``HOME_Q`` elsewhere."""
    grasp = sim.panel is not None and any(c.cls == GRASP_CLASS for c in sim.panel.controls)
    return GRASP_HOME_Q if grasp else HOME_Q


def reset_episode(sim: Sim, q_home: np.ndarray | None = None) -> None:
    """Arm at home, every control at rest, zero velocities."""
    q_home = home_q(sim) if q_home is None else q_home
    mujoco.mj_resetData(sim.model, sim.data)
    for h in sim.controls.values():
        h.dyn.reset()
        sim.model.jnt_range[h.joint_id][1] = h.upper
    sim.set_arm_q(q_home)


def run_program(sim: Sim, program: Program, control_id: str, cfg: EpisodeConfig, *, panel_id: str = "",
                attempt: int = 0, finish: bool = True) -> StepTrace:
    """Execute ``program`` on ``control_id`` from the home pose and return the recorded trace."""
    from sea.control import motor

    trace = StepTrace(panel_id=panel_id, control_id=control_id, program_id=program.id, attempt=attempt)
    ep = Episode(sim, control_id, cfg, trace)
    trace.expected_samples = 0
    ok, reason = drive(ep, exec_node(ep, program.root))
    ep.carrying = False
    if finish and reason not in ("force_limit", "sim_unstable"):
        drive(ep, motor.finish(ep))  # always back off so springs and push-latches settle
    trace.aborted = reason == "sim_unstable"
    if reason == "time_limit":
        trace.note = "time_limit"
    elif reason:
        trace.note = reason
    trace.expected_samples = len(trace.t)
    dyn = sim.controls[control_id].dyn
    trace.detents_crossed = dyn.crossings
    trace.latched_final = dyn.latched
    trace.sim_seconds = round(ep.t + ep.t_offset, 4)
    trace.force_cap = cfg.force_cap
    trace.peak_control_force = round(ep.peak_ctl_force, 4)
    trace.contact_types = sorted(ep.contact_types)  # type: ignore[assignment]
    trace.q_rest = ep.q_rest0[control_id]
    trace.program_ok = ok
    if ep.sense_log is not None:
        trace.sensors = ep.sense_log.out
    if cfg.thermal is not None:
        trace.servo_heat_j = round(float(cfg.thermal.total_j), 4)  # type: ignore[attr-defined]
    return trace
