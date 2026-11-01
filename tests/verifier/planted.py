"""Planted-fault panels for the verifier acceptance suite (spec 12.3).

Each ``Planted`` case is a single-control panel whose physics (``Faults``), placement, rules, or
perception differ from what the robot expects, plus its declared verdict.  ``twin()`` is the
matching fault-free case, which must pass.  Two runners produce traces for ``judge``:

* ``synthetic_trace``: a 1-D quasi-static actuator against the real ``ControlDynamics`` (fast,
  M2 acceptance: "the verifier suite passes on synthetic traces");
* ``sim_trace``: the seed library on the compiled MuJoCo model (slow, M3 acceptance).

Choices are recorded in docs/decisions.md (V-004, V-005).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cache

import numpy as np

from sea.contracts import ControlSpec, HardLimits, PanelSpec, Pose, Rules, StepTrace, Vec3
from sea.sim.controls import ControlDynamics, Faults, build_geometry
from sea.sim.panels import layout, load_config, panel_pose, success_check

MM = 1e-3
LIMITS = HardLimits(workspace_box=(Vec3(x=0.10, y=-0.20, z=-0.10), Vec3(x=0.40, y=0.20, z=0.25)))
FORCE_CAP_FRACTION = 0.8  # derive(Posture()) at beta = 0
REFERENCE_POSE = (0.22, 0.0, -0.03)
HOOK = ("hook", {"depth": 14.0, "throat": 6.0, "length": 45.0})
PAD = ("pad", {"w": 16.0, "h": 16.0, "thickness": 5.0, "friction": 0.9})
CUP = ("knob_cup", {"inner_diameter": 22.0, "depth": 10.0, "ribs": 4.0})
TOOL_CONTACT = {"stock": "edge", "hook": "hook", "pad": "pad", "knob_cup": "cup"}


def control(cls: str, **kw) -> ControlSpec:
    base = dict(id="c0", cls=cls, pose_panel=Pose(pos=Vec3(x=0.0, y=0.0, z=0.0)), detents=[], detent_peak=0.0,
                handle="none", recess_depth_mm=0.0, latch_threshold=None)
    base.update(kw)
    base["success"] = success_check(cls, base["handle"], len(base["detents"]))
    return ControlSpec(**base)


@dataclass(frozen=True)
class Planted:
    name: str
    spec: ControlSpec
    expected: str
    faults: Faults = field(default_factory=Faults)
    tool: tuple[str, dict] | None = None  # rack tool (template, params); None = stock gripper
    pose: tuple[float, float, float] = REFERENCE_POSE
    standoff_mm: float = 0.0  # perceived surface this much closer to the robot than the real one
    contact: str | None = None  # synthetic only: contact type the stock gripper makes (pinch for tabs)
    obstacle_mm: float | None = None  # synthetic only: rigid rim this far before the control surface
    twin_changes: dict = field(default_factory=dict)
    sim_xfail: str | None = None  # reason the real-sim run cannot meet the declared verdict yet

    @property
    def contact_type(self) -> str:
        return self.contact or TOOL_CONTACT[self.tool[0] if self.tool else "stock"]

    def twin(self) -> Planted:
        """The matching fault-free case (declared ``pass``)."""
        changes = {"faults": Faults(), "pose": REFERENCE_POSE, "standoff_mm": 0.0, "expected": "pass",
                   "name": f"{self.name}_twin", "obstacle_mm": None, **self.twin_changes}
        return replace(self, **changes)

    def panel(self) -> PanelSpec:
        cfg = load_config()
        placed, _ = layout([self.spec], cfg)
        rules = Rules(force_cap_by_class={k: float(v) for k, v in cfg["force_caps"].items()})
        return PanelSpec(id=self.name, seed=0, split="reference", pose_world=panel_pose(*self.pose, 0.0),
                         tilt_rad=0.0, controls=placed, rules=rules)

    def force_cap(self, rules: Rules) -> float:
        return min(FORCE_CAP_FRACTION * rules.force_cap_by_class[self.spec.cls], LIMITS.max_force_n)


_BUTTON = dict(size_mm=(14.0,), stiffness=1800.0, travel=0.0033, friction=0.05)
_DOOR = dict(size_mm=(80.0, 80.0), stiffness=0.05, latch_threshold=6.0, friction=0.003)

CASES: tuple[Planted, ...] = (
    Planted("stiff_button", control("push_button", **_BUTTON), "stuck", faults=Faults(stiffness_scale=5.0)),
    Planted("no_detent", control("push_button", size_mm=(14.0,), stiffness=800.0, travel=0.003, detents=[0.0018],
                                 detent_peak=1.0, friction=0.05), "wrong_behavior", faults=Faults(no_detent=True)),
    Planted("latch_above_cap", control("door", handle="pull", travel=math.radians(60), **_DOOR), "stuck",
            faults=Faults(latch_scale=2.5), tool=HOOK),
    Planted("deep_recess", control("recessed_button", size_mm=(10.0,), stiffness=800.0, travel=0.002, friction=0.05,
                                   recess_depth_mm=22.0), "rule_violation", obstacle_mm=6.0,
            twin_changes={"spec": control("recessed_button", size_mm=(10.0,), stiffness=800.0, travel=0.002,
                                          friction=0.05, recess_depth_mm=5.0)}),
    Planted("tab_door_no_pinch", control("door", handle="tab", travel=math.radians(62), **_DOOR), "rule_violation",
            contact="pinch", twin_changes={"tool": HOOK, "contact": None}),
    Planted("knob_over_torque", control("knob", size_mm=(20.0,), stiffness=0.01, travel=math.radians(60),
                                        friction=0.002), "stuck", faults=Faults(stiffness_scale=200.0), tool=CUP,
            sim_xfail="rotate presses the cup into the knob above the 10 N class cap (V-005)"),
    Planted("reversed_slider", control("slider", size_mm=(40.0,), stiffness=100.0, travel=0.02, friction=0.05),
            "wrong_behavior", faults=Faults(reverse=True)),
    Planted("jammed_lever", control("lever", size_mm=(50.0,), stiffness=0.1, travel=math.radians(35), friction=0.002),
            "stuck", faults=Faults(jammed=True), tool=PAD),
    Planted("out_of_reach", control("push_button", size_mm=(14.0,), stiffness=800.0, travel=0.0025, friction=0.05),
            "unreachable", pose=(0.42, 0.0, -0.03)),
    Planted("mislocalized", control("push_button", size_mm=(14.0,), stiffness=800.0, travel=0.0025, friction=0.05),
            "no_actuation", standoff_mm=15.0),
)


# ---------------------------------------------------------------- synthetic runner
DT = 0.02
TIP_SPEED = 0.01  # m/s, the final contact speed of the motor primitives
CONTACT_K = 5e3  # N/m, tool-control contact compliance
RIGID_K = 1e5  # N/m, a rigid rim (no compliance): one tick of motion overshoots any cap
APPROACH_GAP = 0.01


def synthetic_trace(case: Planted, seed: int | None = None, horizon_s: float = 12.0) -> StepTrace:
    """A guarded 1-D push/pull against the control's real dynamics (Faults applied).

    The actuator advances at ``TIP_SPEED`` along the described actuation direction, stops when tip
    force reaches the guard cap or the wrist stalls, when travel is reached, or when its planned
    depth (travel x 1.3 + 1 mm past the perceived surface) runs out.  Rotary controls are handled in
    generalized units with the geometry's lever arm.
    """
    panel = case.panel()
    rules = panel.rules
    spec = panel.controls[0]
    rng = np.random.default_rng(seed)
    fr, st = (float(rng.uniform(0.8, 1.2)), float(rng.uniform(0.9, 1.1))) if seed is not None else (1.0, 1.0)
    dyn = ControlDynamics(spec, stiffness_scale=st, friction_scale=fr, faults=case.faults)
    geo = build_geometry(spec)
    r = geo.lever_arm if geo.rotary else 1.0
    dyn.set_lever_arm(r)
    cap = case.force_cap(rules)
    trace = StepTrace(panel_id=panel.id, control_id=spec.id, program_id="synthetic", attempt=0, force_cap=cap)
    x = panel.pose_world.pos.x
    if not LIMITS.workspace_box[0].x <= x <= LIMITS.workspace_box[1].x:
        trace.note = "workspace"  # the executor refuses targets outside the box before moving
        trace.ik_residual_mm = 99.0
        trace.expected_samples = 0
        return trace
    trace.ik_residual_mm = 0.6
    jam = case.faults.jammed
    sign = -1.0 if case.faults.reverse else 1.0
    s = -APPROACH_GAP  # actuator position along the actuation direction, linear (m at the tip)
    depth = spec.travel * r * 1.3 + 1 * MM - case.standoff_mm * MM
    q = 0.0  # control displacement along the actuation direction (generalized)
    t, peak_ctl = 0.0, 0.0
    banned = False
    f_tip = 0.0
    while t < horizon_s:
        # Guarded step (as Episode.guarded_step): shrink the step as force nears the cap.
        s_next = s + TIP_SPEED * DT * min(1.0, max(0.02, (cap - f_tip) / (0.5 * cap)))
        f_tip = 0.0
        touching = False
        if case.obstacle_mm is not None and s_next >= -case.obstacle_mm * MM:
            f_tip = RIGID_K * (s_next + case.obstacle_mm * MM)
        elif s_next >= q * r:
            touching = True
            q_lim = 0.0 if (jam or dyn.latch_upper() is not None) else math.inf
            q_try = min(s_next / r, q_lim)
            gen = max(0.0, -dyn.force(sign * q_try, 0.0) * sign + dyn.friction)
            f_tip = gen / r + CONTACT_K * max(0.0, s_next - q_try * r)
            q = q_try
            dyn.update(sign * q, latch_force=f_tip * r if q >= 0.0 and q_lim == 0.0 else 0.0)
            peak_ctl = max(peak_ctl, f_tip)
            banned = banned or case.contact_type in rules.banned_contacts
        s = s_next
        t += DT
        trace.t.append(round(t, 4))
        trace.tip_force.append(round(f_tip, 4))
        trace.control_q.append(round(sign * q, 6))
        trace.push_phase.append(s >= -1 * MM)
        trace.in_contact.append(touching)
        if trace.success_time_s is None and _success(dyn, spec, sign * q):
            trace.success_time_s = round(t, 4)
        if f_tip >= cap - 0.25 or q >= spec.travel * 1.05 or s >= depth:
            break
    trace.expected_samples = len(trace.t)
    trace.peak_control_force = round(peak_ctl, 4)
    trace.detents_crossed = dyn.crossings
    trace.latched_final = dyn.latched
    trace.sim_seconds = round(t, 4)
    trace.contact_types = [case.contact_type] if peak_ctl > 0 or banned else []  # type: ignore[list-item]
    return trace


def _success(dyn: ControlDynamics, spec: ControlSpec, moved: float) -> bool:
    ok = moved >= spec.travel
    if spec.cls in ("door", "drawer"):
        ok = ok and not dyn.latched
    if "detent_crossed" in spec.success:
        ok = ok and dyn.crossings >= 1
    return ok


# ---------------------------------------------------------------- real-sim runner
@cache
def _built(tool: tuple | None):
    from sea.contracts import Design
    from sea.design.build import build
    from sea.design.space import make_tool_spec

    rack = [make_tool_spec(tool[0], **dict(tool[1]))] if tool else []
    return build(Design(id="planted", tool_rack=rack), LIMITS)


def sim_trace(case: Planted, seed: int | None = None) -> StepTrace:
    """Run the first seed-library program for the control on the compiled model."""
    from sea.compose import dsl
    from sea.compose.library import Library
    from sea.contracts import Program
    from sea.control.executor import EpisodeConfig, reset_episode, run_program
    from sea.control.frames import quat_wxyz_to_mat
    from sea.sim.build_model import DomainRandomization, build_model

    panel = case.panel()
    spec = panel.controls[0]
    built = _built(None if case.tool is None else (case.tool[0], tuple(sorted(case.tool[1].items()))))
    sim = build_model(built.design, panel, built.tools, faults={spec.id: case.faults}, dr=DomainRandomization(seed))
    reset_episode(sim)
    lib = Library.seed()
    defn = lib.programs_for(spec)[0]
    root = defn.body.root if case.tool is None else dsl.with_tool(0, defn.body.root)
    normal = quat_wxyz_to_mat(panel.pose_world.quat_wxyz)[:, 2]
    cap = case.force_cap(panel.rules)
    cfg = EpisodeConfig(limits=LIMITS, force_cap=cap, class_cap=panel.rules.force_cap_by_class[spec.cls],
                        banned=frozenset(panel.rules.banned_contacts), library=lib.prims,
                        perception_offset=normal * case.standoff_mm * MM)
    prog = Program(id=defn.name, target_cls=spec.cls, root=root, source="seed")
    return run_program(sim, prog, spec.id, cfg, panel_id=panel.id)
