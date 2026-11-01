"""The gripper's measuring instrument (docs/gripper-experiment.md, V1).

``pick`` is one scripted top-down pick with fixed settings: open, descend beside the object, close
to the servo's position target, lift, hold.  It is an instrument, not a skill: nothing in it is
tuned, so a change in what it measures is a change in the gripper or the physics, never in a
learned grasp.  Each pick reports ground truth (did the object stay up, how hard the jaws squeeze)
next to what the robot's own sensors read during the hold, which is the usage profile phase A
records.  ``scripts/measure_grasp.py`` runs the fixed case list into ``tests/golden/stock_grasp.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from sea.contracts import Design
from sea.control.executor import HOME_Q, TICK_STEPS, stock_ik
from sea.control.motor import STOCK_TIP_EXT
from sea.sim.build_model import SceneObject, Sim, build_model
from sea.sim.sensors import JAW_KINDS, Sensors

MM = 1e-3
TICK_S = 0.02  # one 50 Hz joint target
Shape = Literal["box", "cylinder", "sphere"]


@dataclass(frozen=True)
class Probe:
    """Fixed instrument settings.  Changing one re-measures the table; it is not a tuning knob."""

    x: float = 0.22  # object centre on the table, straight ahead of the base (m)
    q_open: float = 1.0  # gripper joint while descending (rad)
    q_close: float = -0.17  # position target when closing; the servo stalls on the object first
    side_mm: float = 2.0  # clearance between the fixed jaw's inner face and the object on descent
    site_depth_mm: float = 17.0  # tool site below the object top for prisms
    sphere_past_mm: float = 5.0  # jaw tips this far below a sphere's equator
    lift_mm: float = 50.0
    hold_s: float = 1.0
    held_mm: float = 30.0  # phase A task: still at least this high after the hold


PROBE = Probe()
HEIGHT_MM = {"cylinder": 60.0, "box": 50.0}
BOX_DEPTH_MM = 30.0  # box extent along the approach (x); width is along the closing axis (y)


def scene_object(shape: Shape, width_mm: float, mass_g: float, friction: float = 0.8, probe: Probe = PROBE,
                 name: str = "obj") -> SceneObject:
    """An object standing on the table with ``width_mm`` across the jaws' closing axis (world y)."""
    w = width_mm * MM
    if shape == "sphere":
        size: tuple[float, ...] = (w / 2,)
        h = w
    elif shape == "cylinder":
        h = HEIGHT_MM[shape] * MM
        size = (w / 2, h / 2)
    else:
        h = HEIGHT_MM[shape] * MM
        size = (BOX_DEPTH_MM * MM / 2, w / 2, h / 2)
    return SceneObject(name, shape, size, (probe.x, 0.0, h / 2), mass_g / 1000, friction)


def object_top(obj: SceneObject) -> float:
    return 2 * obj.size[0] if obj.shape == "sphere" else 2 * obj.size[-1]


def object_width(obj: SceneObject) -> float:
    return 2 * obj.size[1] if obj.shape == "box" else 2 * obj.size[0]


def site_depth(obj: SceneObject, probe: Probe = PROBE, below: float = STOCK_TIP_EXT) -> float:
    """How far below the object top the tool site goes.  Spheres: jaw tips ``sphere_past_mm`` below
    the equator; prisms: ``site_depth_mm``, but never so deep that the tips come within 3 mm of the table.
    ``below``: how far the jaw tips (or the finger pads, D-038) reach below the tool site."""
    top = object_top(obj)
    if obj.shape == "sphere":
        return top / 2 + probe.sphere_past_mm * MM - below
    return min(probe.site_depth_mm * MM, top - below - 3 * MM)


@dataclass(frozen=True)
class Jaws:
    """The probe's settings that follow from the fingers (D-038); for the stock jaws they are V1's."""

    q_open: float
    face: float  # fixed pad face to the stock jaw face (m)
    below: float  # lowest jaw or pad point below the tool site with the jaws on the object (m)


def jaws(design: Design, width: float, probe: Probe = PROBE) -> Jaws:
    from sea.design import fingers as F

    f = design.fingers
    if F.is_stock(f):
        return Jaws(probe.q_open, 0.0, STOCK_TIP_EXT)
    q_open = F.opening_for(f, width + 2 * probe.side_mm * MM + 10 * MM)
    q_obj = F.opening_for(f, width)
    return Jaws(max(probe.q_open, q_open if q_open is not None else 1.7), F.thickness(f),
                F.below_tool(f, q_obj if q_obj is not None else probe.q_open))


@dataclass
class PickResult:
    shape: str
    width_mm: float
    mass_g: float
    friction: float
    held: bool  # ground truth: still ``held_mm`` up after the hold
    lift_mm: float  # object rise at the end of the lift
    hold_mm: float  # object rise at the end of the hold
    knocked_mm: float  # object moved sideways before the jaws closed (the descent hit it)
    jaw_q: float  # gripper joint angle once closed on the object (rad)
    squeeze_n: float  # ground truth: total jaw normal force on the object after closing
    pads_n: list[float] | None  # sensor: (fixed, moving) jaw pads at the end of the hold
    grip_current_a: float  # sensor: gripper servo current at the end of the hold
    payload_g: float | None  # sensor: payload estimate at the end of the hold (None if the arm moved)
    sim: Sim | None = field(default=None, repr=False, compare=False)

    def row(self) -> dict[str, Any]:
        out = asdict(self)
        del out["sim"]
        return out


def jaw_squeeze(sim: Sim, geom: int) -> float:
    """Ground truth: total normal force between the two jaws and one scene geom."""
    m, d = sim.model, sim.data
    total, fr = 0.0, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if geom not in (c.geom1, c.geom2):
            continue
        other = int(c.geom2 if c.geom1 == geom else c.geom1)
        if sim.geom_kind.get(other) in JAW_KINDS:
            mujoco.mj_contactForce(m, d, i, fr)
            total += abs(float(fr[0]))
    return total


def run_pick(sim: Sim, obj: SceneObject, probe: Probe = PROBE, jw: Jaws | None = None) -> dict[str, float]:
    """Drive ``sim`` through the scripted pick of ``obj`` (already in the scene).

    Returns the ground-truth marks taken between phases: sideways knock before closing, jaw angle and
    squeeze once closed, and the object's rise after the lift and after the hold (m, rad, N)."""
    ik = stock_ik()
    jw = jw or jaws(sim.design, object_width(obj), probe)
    body, geom = sim.objects[obj.name], sim.object_geoms[obj.name]
    q = HOME_Q.copy()
    sim.set_arm_q(q)
    down, roll = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])  # jaws close along world y
    top = np.array([obj.pos[0], obj.pos[1], object_top(obj)])
    beside = np.array([0.0, object_width(obj) / 2 + probe.side_mm * MM + jw.face, 0.0])  # fixed jaw on +y
    grasp = top + beside - [0.0, 0.0, site_depth(obj, probe, jw.below)]

    def tick(target: np.ndarray, n: int = 1) -> None:
        for _ in range(n):
            sim.data.ctrl[:] = target
            sim.step(TICK_STEPS)

    def move(goal: np.ndarray, seconds: float, grip: float) -> None:
        nonlocal q
        start = ik.fk(q)[:3, 3]
        n = int(seconds / TICK_S)
        for k in range(1, n + 1):
            q, _, _ = ik.track(start + (goal - start) * k / n, down, q, roll_axis=roll)
            q[-1] = grip
            tick(q)

    tick(q, 10)  # let the object settle on the table
    xy0, z0 = sim.data.xpos[body][:2].copy(), float(sim.data.xpos[body][2])
    move(grasp + [0.0, 0.0, 0.04], 2.0, jw.q_open)
    move(grasp, 1.5, jw.q_open)
    knocked = float(np.linalg.norm(sim.data.xpos[body][:2] - xy0))
    for k in range(1, 31):
        q = q.copy()
        q[-1] = jw.q_open + (probe.q_close - jw.q_open) * k / 30
        tick(q)
    tick(q, 15)
    jaw_q = float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint("gripper").id]])
    squeeze = jaw_squeeze(sim, geom)
    move(ik.fk(q)[:3, 3] + [0.0, 0.0, probe.lift_mm * MM], 1.5, q[-1])
    lift = float(sim.data.xpos[body][2]) - z0
    tick(q, int(probe.hold_s / TICK_S))
    hold = float(sim.data.xpos[body][2]) - z0
    return {"knocked": knocked, "jaw_q": jaw_q, "squeeze": squeeze, "lift": lift, "hold": hold}


def pick(obj: SceneObject, design: Design | None = None, probe: Probe = PROBE, seed: int = 0) -> PickResult:
    """Build a table scene with ``obj``, run the scripted pick, and read the sensors during the hold."""
    sim = build_model(design or Design(id="stock"), objects=[obj], table_z=0.0)
    marks = run_pick(sim, obj, probe)
    frame = Sensors(sim, seed=seed).read()
    pads = frame.jaw_force_n
    return PickResult(
        shape=obj.shape, width_mm=round(object_width(obj) / MM, 2), mass_g=round(obj.mass_kg * 1000, 1),
        friction=obj.friction, held=marks["hold"] >= probe.held_mm * MM,
        lift_mm=round(marks["lift"] / MM, 1), hold_mm=round(marks["hold"] / MM, 1),
        knocked_mm=round(marks["knocked"] / MM, 1), jaw_q=round(marks["jaw_q"], 3),
        squeeze_n=round(marks["squeeze"], 2),
        pads_n=None if pads is None else [round(float(f), 2) for f in pads],
        grip_current_a=round(float(frame.current_a[-1]), 3),
        payload_g=round(frame.payload_g, 1) if frame.static else None, sim=sim,
    )
