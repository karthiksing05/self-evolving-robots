"""The gripper world for the evolution core (D-042): the SO-101 picking 21 real objects off a table.

Tasks are the YCB objects of D-038 (configs/ycb_objects.yaml): pick one up, lift it 30 mm, hold it for a
second.  Each object is a task group with a centred table and one seeded placement to train on, and
two more placements to test on.  ``solve`` is the established System 1: ``sim.env.run_panel`` with its
CMA-ES search over the grasp parameters and confirmation re-runs (D-016), so verdicts here are the
verdicts the panel loop gets.

The body has three slots, each a part with a bill of materials:

- ``jaw``: the moving jaw's joint (``design/fingers.py``): the stock hinge, a parallelogram linkage or
  a rack-and-pinion slide;
- ``pads``: none, or the printed pad family of D-038 (reach, tilt, groove, lip, TPU skin, width);
- ``arm_servos``: the kit's 7.4 V STS3215s, or 12 V ones on the three lifting joints (L3).

When a pick fails, the failed attempt is replayed with its final parameters and the simulator's
contacts are read (a twin's analysis, not the robot's senses): where each jaw touches the object
relative to its centre and top, which way the contact pushes, how far the moving fingertip has risen,
and what happened on the lift.  That is the ``mechanism`` System 2 reads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from sea.contracts import ARM_JOINTS, ArmSpec, Design, FingerSpec, HardLimits, PanelSpec, SensorSuite
from sea.design import fingers as F
from sea.evolve.core import Body, BomLine, Outcome, Param, Part, PartType, Task

ROOT = Path(__file__).resolve().parents[3]
LIFT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
GRIP_TORQUE_NM = 1.62  # the gripper servo at 7.4 V (sim/build_model.py STALL_TORQUE)
PAD_DOCS = {
    "reach_mm": "how far both pads extend past the stock fingertips",
    "pad_angle_deg": "the moving pad is tilted back by this angle on its jaw (on the hinge: its face is parallel "
                     "to the fixed one when the jaw has opened by this angle)",
    "groove_mm": "depth of a V-groove running up each pad face, which centres round objects across the pad",
    "lip_mm": "an inward hook at the bottom of each pad",
    "pad_tpu_mm": "a TPU skin on each face: friction 0.9 instead of bare PLA's 0.45",
    "width_mm": "pad width across the jaw (default 12 mm)",
}


@lru_cache(maxsize=1)
def _limits() -> HardLimits:
    return HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits_grasp.yaml").read_text()))


def to_design(body: Body) -> Design:
    jaw, pads, servos = body.parts["jaw"], body.parts["pads"], body.parts["arm_servos"]
    kw: dict[str, Any] = {"jaw": jaw.type}
    if jaw.type == "parallelogram":
        kw["link_mm"] = jaw.params["link_mm"]
    if jaw.type == "prismatic":
        kw["drive_mm_per_rad"] = jaw.params["drive_mm_per_rad"]
    if pads.type == "pads":
        kw.update({k: float(v) for k, v in pads.params.items()})
    volts = {j: (12.0 if servos.type == "servos_12v" and j in LIFT_JOINTS else 7.4) for j in ARM_JOINTS}
    return Design(id=body.key(), fingers=FingerSpec(**kw), arm=ArmSpec(servo_voltage=volts),  # type: ignore[arg-type]
                  sensors=SensorSuite(jaw_pads=True))


# ---------------------------------------------------------------- mechanism diagnosis
def _replay(sim, attempt) -> list[dict[str, Any]]:
    """Run ``attempt()`` on ``sim`` with its step instrumented: jaw-object contacts every 10 sim steps."""
    m, d = sim.model, sim.data
    obody = next(h.body_id for h in sim.controls.values())
    ogeoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == obody}
    gact = m.actuator("gripper").id
    rows: list[dict[str, Any]] = []
    n = [0]
    orig = sim.step
    fr = np.zeros(6)

    def step(k: int = 1) -> None:
        orig(k)
        n[0] += 1
        if n[0] % 10:
            return
        com = d.xipos[obody].copy()
        cons: dict[str, list[tuple[float, float, float, str]]] = {"jaw_fixed": [], "jaw_moving": [], "arm": []}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in ogeoms:
                other, sign = g2, -1.0
            elif g2 in ogeoms:
                other, sign = g1, 1.0
            else:
                continue
            kind = sim.geom_kind.get(other)
            if kind not in cons:
                continue
            mujoco.mj_contactForce(m, d, i, fr)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            cons[kind].append((float(fr[0]), float(c.pos[2] - com[2]) * 1000, float(c.frame[2] * sign), name))
        rows.append({"t": n[0] * 0.002, "ctrl": float(d.ctrl[gact]), "q": float(sim.arm_q()[-1]),
                     "z": float(com[2]), "xy": com[:2].copy(), "cons": cons})

    sim.step = step
    try:
        attempt()
    finally:
        sim.step = orig
    return rows


def _jaw_summary(cs: list[tuple[float, float, float, str]]) -> tuple[float, float, float, str]:
    f = sum(c[0] for c in cs)
    if f < 0.2:
        return 0.0, 0.0, 0.0, ""
    zc = sum(c[0] * c[1] for c in cs) / f
    nz = sum(c[0] * c[2] for c in cs) / f
    parts = "pad" if any("pad" in c[3] for c in cs) else "stock jaw"
    if any("pad" in c[3] for c in cs) and any("pad" not in c[3] for c in cs):
        parts = "pad and stock jaw"
    return f, zc, nz, parts


def diagnose(rows: list[dict[str, Any]], note: str, fing: FingerSpec, height_m: float, mass_g: float) -> tuple[str, str]:
    """(mechanism, detail) for one failed pick, from the replayed contacts."""
    if note == "load_limit":
        return "arm_stall", f"an arm joint stalled carrying {mass_g:g} g (load_limit reflex)"
    if not rows:
        return "no_trace", note
    z0, xy0 = rows[0]["z"], rows[0]["xy"]
    start = [r for r in rows[:5] if sum(c[0] for k in ("jaw_fixed", "jaw_moving", "arm") for c in r["cons"][k]) > 50]
    if start:
        return "start_collision", "the robot touches the object at the start of the episode"
    if note == "force_limit":
        r = max(rows, key=lambda r: sum(c[0] for k in r["cons"] for c in r["cons"][k]))
        return "force_limit", f"a contact exceeded the force limit at t={r['t']:.2f} s (squeezing or pressing down)"
    ctrl = np.array([r["ctrl"] for r in rows])
    peak = int(np.argmax(ctrl))  # the descent opening; closing starts when the command falls from it
    after = np.nonzero(ctrl[peak:] < ctrl[peak] - 0.05)[0]
    i_close = peak + int(after[0]) if len(after) else len(rows) - 1
    knocked = float(np.linalg.norm(rows[max(i_close - 1, 0)]["xy"] - xy0)) * 1000
    squeeze = rows[min(len(rows) - 1, i_close + 25):]
    if not squeeze:
        return "never_closed", "the jaws never closed on the object"
    snap = max(squeeze[:40], key=lambda r: sum(c[0] for c in r["cons"]["jaw_fixed"] + r["cons"]["jaw_moving"]))
    fx, zfx, nfx, pfx = _jaw_summary(snap["cons"]["jaw_fixed"])
    fm, zm, nm, pm = _jaw_summary(snap["cons"]["jaw_moving"])
    q = snap["q"]
    gap, rise = (v * 1000 for v in F.fingertips(fing, q))
    half = height_m / 2 * 1000
    lifted = max(r["z"] for r in rows) - z0
    end = rows[-1]["z"] - z0

    def jaw(name: str, f: float, zc: float, nz: float, part: str) -> str:
        if f < 0.2:
            return f"{name} jaw: no contact"
        tilt = math.degrees(math.asin(max(-1.0, min(1.0, nz))))
        return (f"{name} jaw: {f:.0f} N on the {part} at {zc:+.0f} mm from the centre (top at +{half:.0f}), "
                f"pushing {abs(tilt):.0f} deg {'down' if tilt < 0 else 'up'}")

    detail = (f"closed to {q:.2f} rad (fingertips {gap:.0f} mm apart), moving fingertip {rise:+.0f} mm vs the fixed one; "
              f"{jaw('fixed', fx, zfx, nfx, pfx)}; {jaw('moving', fm, zm, nm, pm)}; "
              f"rose {lifted * 1000:.0f} mm, ended {end * 1000:+.0f} mm; {mass_g:g} g")
    if knocked > 8:
        return "knocked_on_descent", f"the descending jaws pushed the object {knocked:.0f} mm before closing; " + detail
    if fx < 0.2 or fm < 0.2:
        which = "fixed" if fx < 0.2 else "moving"
        return f"{which}_jaw_missed", detail
    if min(nfx, nm) < -0.8:
        which = "moving" if nm < nfx else "fixed"
        return f"{which}_jaw_on_top_face", detail
    if (nfx + nm) / 2 < -0.12 or min(nfx, nm) < -0.3:
        return "squeezed_down_out", detail
    if (nfx + nm) / 2 > 0.12:
        return "squeezed_up_out", detail
    if lifted > 0.01:
        return "slipped_under_load", detail
    return "never_lifted", detail


# ---------------------------------------------------------------- the world
class GripperWorld:
    name = "gripper"
    description = ("An SO-101 arm (5 joints plus a gripper, STS3215 servos) picks real objects (YCB scans with "
                   "published masses) off a table: pick it up, lift it 30 mm, hold it 1 s.  It grasps from above "
                   "with the jaws closing along the object's narrow side.  The fixed jaw's face is a vertical plane; "
                   "the moving jaw is driven by the gripper servo (1.62 N*m at 7.4 V).  Each object is one task "
                   "group: it is held when every placement passes.  System 1 tunes the grasp (side clearance, close "
                   "target and speed, lift, hold, speed) by CMA-ES with confirmation re-runs.")

    def __init__(self, train_offsets: int = 1, test_offsets: int = 2):
        self.train_offsets, self.test_offsets = train_offsets, test_offsets
        self._built: dict[str, Any] = {}

    # tasks
    def tasks(self) -> list[Task]:
        from sea.sim.grasp_world import real_panels, regression_panels
        from sea.sim.objects import catalog

        out = []
        for name, o in catalog().items():
            facts = (f"{o['name']}: {o['width_mm']:.0f} mm across the jaws, {o['depth_mm']:.0f} deep, "
                     f"{o['height_mm']:.0f} tall, {o['mass_g']} g, {o['material']} (friction {o['friction']}), "
                     f"{o['pose']}{', rolls' if o['round'] else ''}")
            panels = [("train", p) for p in regression_panels([name]) + real_panels("val", self.train_offsets, [name], seed=7)]
            panels += [("test", p) for p in real_panels("test", self.test_offsets, [name], seed=11)]
            for split, p in panels:
                out.append(Task(id=p.id, split=split, group=name, facts=facts,  # type: ignore[arg-type]
                                data={"panel": p.model_dump(mode="json"), "height_mm": o["height_mm"],
                                      "mass_g": o["mass_g"], "width_mm": o["width_mm"]}))
        return out

    # parts
    def catalogue(self) -> list[PartType]:
        pad_params = {k: Param(lo=lo, hi=hi, unit="deg" if k.endswith("deg") else "mm", doc=PAD_DOCS[k])
                      for k, (lo, hi) in F.RANGES.items()}
        return [
            PartType(name="revolute", slot="jaw", stock=True,
                     doc="the stock moving jaw on its hinge: the servo turns it about an axis 81 mm above its fingertip",
                     provides="a wide swing from a compact part"),
            PartType(name="parallelogram", slot="jaw",
                     doc="two printed links from the servo axis carry the finger without turning it (a four-bar); the "
                         "finger still swings on a circle of radius link_mm",
                     params={"link_mm": Param(lo=50, hi=90, unit="mm", doc="link length: the radius of the finger's "
                                              "circular path; longer opens wider per radian and squeezes less")},
                     provides="a finger face that keeps its orientation at every opening"),
            PartType(name="prismatic", slot="jaw",
                     doc="a printed rack and pinion on two steel rods slides the finger straight along the closing axis",
                     params={"drive_mm_per_rad": Param(lo=20, hi=60, unit="mm/rad", doc="finger travel per servo "
                                                       "radian (pinion pitch radius): the opening grows with it, the "
                                                       "squeeze falls with it (servo torque / radius)")},
                     provides="a finger that neither turns nor rises as it opens"),
            PartType(name="none", slot="pads", stock=True, doc="the bare stock jaw faces (PLA)"),
            PartType(name="pads", slot="pads", doc="printed pads over the bottom of each jaw face (design/fingers.py)",
                     params=pad_params, provides="contact shape, extension and friction at the fingertips"),
            PartType(name="servos_7v4", slot="arm_servos", stock=True, doc="the kit's 7.4 V STS3215 servos: 1.62 N*m stall"),
            PartType(name="servos_12v", slot="arm_servos",
                     doc="12 V STS3215 servos on shoulder lift, elbow and wrist flex: 2.94 N*m stall; needs a 12 V supply",
                     provides="more torque on the joints that lift"),
        ]

    def stock(self) -> Body:
        return Body(parts={"jaw": Part(type="revolute"), "pads": Part(type="none"), "arm_servos": Part(type="servos_7v4")})

    def bom(self, body: Body) -> list[BomLine]:
        from sea.design.build import finger_estimate

        out = []
        jaw = body.parts["jaw"]
        if jaw.type == "parallelogram":
            out += [BomLine(item="print:PLA", qty=8 + 0.12 * jaw.params["link_mm"], slot="jaw", note="links, coupler"),
                    BomLine(item="m3_screw", qty=4, slot="jaw"), BomLine(item="mr63_bearing", qty=4, slot="jaw")]
        if jaw.type == "prismatic":
            out += [BomLine(item="print:PLA", qty=18 + 0.35 * jaw.params["drive_mm_per_rad"], slot="jaw",
                            note="carriage, rack, pinion, rod clamps"),
                    BomLine(item="steel_rod_3x100", qty=2, slot="jaw"), BomLine(item="lm3uu_bearing", qty=2, slot="jaw"),
                    BomLine(item="m3_screw", qty=6, slot="jaw")]
        if body.parts["pads"].type == "pads":
            _, pla, tpu, _ = finger_estimate(to_design(body))
            out.append(BomLine(item="print:PLA", qty=round(pla, 2), slot="pads"))
            if tpu > 0:
                out.append(BomLine(item="print:TPU", qty=round(tpu, 2), slot="pads"))
            out.append(BomLine(item="m2_screw", qty=4, slot="pads"))
        if body.parts["arm_servos"].type == "servos_12v":
            out += [BomLine(item="sts3215_12v", qty=3, slot="arm_servos"), BomLine(item="sts3215_7v4", qty=3, slot="arm_servos"),
                    BomLine(item="psu_12v", qty=1, slot="arm_servos")]
        else:
            out.append(BomLine(item="sts3215_7v4", qty=6, slot="arm_servos"))
        return out

    def manifest(self) -> dict[str, Any]:
        return {
            "budget": {"usd": 40.0, "lead_days": 7.0, "fab_minutes": 600.0},
            "fabrication": {"print:PLA": {"minutes_per_unit": 5.0, "unit": "g", "doc": "FDM printer, 0.4 mm nozzle"},
                            "print:TPU": {"minutes_per_unit": 8.0, "unit": "g", "doc": "same printer, slower for TPU"}},
            "items": {
                "sts3215_7v4": {"on_hand": 6, "usd": 16.0, "lead_days": 4, "doc": "the SO-101 kit's servos"},
                "sts3215_12v": {"on_hand": 0, "usd": 18.0, "lead_days": 5},
                "psu_12v": {"on_hand": 0, "usd": 14.0, "lead_days": 3, "doc": "12 V 5 A supply"},
                "m2_screw": {"on_hand": 50}, "m3_screw": {"on_hand": 50},
                "steel_rod_3x100": {"on_hand": 0, "usd": 1.5, "lead_days": 2, "doc": "3 mm ground steel rod"},
                "lm3uu_bearing": {"on_hand": 0, "usd": 1.2, "lead_days": 3, "doc": "3 mm linear bushing"},
                "mr63_bearing": {"on_hand": 0, "usd": 0.9, "lead_days": 3, "doc": "3 mm pivot bearing"},
                "mgn7_rail": {"on_hand": 0, "doc": "miniature linear rail (none on the bench)"},
            },
        }

    def check(self, body: Body) -> list[str]:
        from sea.design.build import build

        try:
            built = build(to_design(body), _limits())
        except (ValueError, ZeroDivisionError) as exc:
            return [f"cannot build: {exc}"]
        return list(built.errors)

    # facts and physics
    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        f = to_design(body).fingers
        lo, hi = F.q_range(f)
        rows = []
        for w in (17, 25, 35, 45, 55, 65, 75, 85, 95):
            q = F.tip_opening(f, w / 1000)
            if q is None:
                rows.append(f"  {w} mm: beyond the widest opening")
                continue
            tilt = math.degrees(q) - f.pad_angle_deg if f.jaw == "revolute" else -f.pad_angle_deg
            rise = F.fingertips(f, q)[1] * 1000
            face = (f"{abs(tilt):.0f} deg off parallel, bottom swung {'out' if tilt > 0 else 'in'}" if abs(tilt) > 0.5
                    else "parallel to the fixed face")
            rows.append(f"  {w} mm: servo {q:.2f} rad; moving face {face}; "
                        f"moving fingertip {rise:+.0f} mm vs the fixed one; squeeze {F.squeeze_n(f, q, GRIP_TORQUE_NM):.0f} N")
        jaw = {"revolute": "hinged moving jaw (the stock part)",
               "parallelogram": f"parallelogram linkage, link {f.link_mm:g} mm",
               "prismatic": f"rack-and-pinion slide, {f.drive_mm_per_rad:g} mm per servo radian"}[f.jaw]
        widest = F.fingertips(f, hi)[0] * 1000
        pads = ("bare stock faces, PLA friction 0.45" if not F.has_pads(f) else
                f"pads {F.thickness(f) * 1000:.0f} mm thick per face, friction {0.9 if f.pad_tpu_mm > 0 else 0.45}, "
                f"tilted {f.pad_angle_deg:g} deg, groove {f.groove_mm:g} mm, reach {f.reach_mm:g} mm, lip {f.lip_mm:g} mm")
        volts = 12.0 if body.parts["arm_servos"].type == "servos_12v" else 7.4
        heavy = max((t for t in tasks), key=lambda t: t.data.get("mass_g", 0), default=None)
        load = _shoulder_load()
        arm = (f"lifting joints stall at {2.94 if volts == 12.0 else 1.62} N*m ({volts:g} V); holding the arm itself over "
               f"the table centre takes about {load[0]:.2f} N*m at the shoulder lift, plus {load[1]:.2f} N*m per 100 g held")
        if heavy is not None:
            arm += f" (heaviest object {heavy.data['mass_g']} g)"
        return "\n".join([f"jaw: {jaw}; servo range {lo:.2f} to {hi:.2f} rad; fingertips at most {widest:.0f} mm apart",
                          f"faces: {pads}", "by object width (the fingertips that far apart):", *rows, f"arm: {arm}"])

    def principles(self) -> list[str]:
        return [
            "Two contacts hold an object when their normals are horizontal and opposed (antipodal).  A contact whose "
            "normal is tilted by a pushes the object up or down with N sin(a); friction mu N must hold it while the "
            "servo keeps closing, so a tilted contact squeezes the object out.",
            "A face that turns with its joint is parallel to the fixed face at one opening only; a fingertip on a "
            "swinging joint rises as the jaw opens (by about R (1 - cos q) for radius R and angle q).",
            "A finger can only hold an object's side if, at that object's opening, the fingertip reaches below the "
            "object's top with clearance above the table; otherwise it presses on the top face.",
            "A round object is held at its equator: contacts above it push it down, contacts below it push it up.  "
            "Friction alone does not trap a sphere squeezed by a closing jaw; a concave or grooved contact centres it.",
            "Squeeze is servo torque over the lever from the joint to the contact (over the pinion radius on a "
            "slide).  Friction holds weight m g when 2 mu N exceeds it with margin for the lift's acceleration.",
            "The arm must carry what the hand holds: the lifting joints' gravity torque grows with mass and reach, "
            "and a joint at its stall torque sags and stalls.",
            "Every added part adds mass at the wrist, machine time and cost, and can change what the hand already "
            "does well: a change that helps wide objects can hurt small ones.",
        ]

    # System 1
    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        from sea.compose.library import Library
        from sea.design.build import build
        from sea.sim.env import PanelRunner, SuiteConfig, attempts, candidate_programs, run_panel

        key = body.key()
        if key not in self._built:
            self._built[key] = build(to_design(body), _limits())
        built = self._built[key]
        cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=_limits(), seed=seed)
        panel = PanelSpec.model_validate(task.data["panel"])
        run = run_panel(panel, cfg)[0]
        trace = run.trace
        verdict = run.result.verdict
        n = attempts(run)
        sim_s = (trace.sim_seconds * n) if trace is not None else 0.0
        if verdict == "pass":
            return Outcome(task=task.id, success=True, score=1.0, mechanism="held", detail="held",
                           controls=_flat(run.params), attempts=n, sim_s=round(sim_s, 2))
        score = {"wrong_behavior": 0.5, "stuck": 0.25}.get(verdict, 0.0)
        mech, detail = verdict, trace.note if trace is not None else ""
        cands = dict(candidate_programs(panel.controls[0], panel, cfg))
        prog = cands.get(run.program)
        if prog is not None:
            runner = PanelRunner(panel, cfg, 0)
            box: dict[str, Any] = {}
            rows = _replay(runner.sim, lambda: box.setdefault("t", runner.attempt(panel.controls[0], prog, run.params)))
            note = box["t"].note if "t" in box else ""
            mech, detail = diagnose(rows, note, built.design.fingers, task.data["height_mm"] / 1000, task.data["mass_g"])
        return Outcome(task=task.id, success=False, score=score, mechanism=mech, detail=f"verdict {verdict}; {detail}",
                       controls=_flat(run.params), attempts=n, sim_s=round(sim_s, 2))


def _flat(params: dict[str, dict[str, float]]) -> dict[str, float]:
    return {f"{p}.{k}": float(v) for p, kv in params.items() for k, v in kv.items()}


@lru_cache(maxsize=1)
def _shoulder_load() -> tuple[float, float]:
    """Static shoulder-lift torque (N*m) with the tool over the table centre: the arm alone, and per 100 g held."""
    from sea.control.executor import stock_ik
    from sea.sim.build_model import build_model

    sim = build_model(Design(id="load"), None, table_z=0.0)
    q, _, _ = stock_ik().solve(np.array([0.22, 0.0, 0.03]), np.array([0.0, 0.0, -1.0]))
    sim.set_arm_q(q)
    m, d = sim.model, sim.data
    mujoco.mj_forward(m, d)
    dof = m.jnt_dofadr[m.joint("shoulder_lift").id]
    base = abs(float(d.qfrc_bias[dof]))
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, sim.site_id)
    per100 = abs(float(jac[2, dof])) * 0.1 * 9.81
    return base, per100
