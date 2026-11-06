"""The house and garden from a kit (D-111): the same routes, terrain and motors, and no mechanism anyone has designed.

A body is one part, ``robot``: a graph on a free joint.  It starts as a plate on four wheels, and System 2 composes
everything else.  Routes are driven by ``worlds.mobile``'s walk over them, with each obstacle's MuJoCo episode run on the
graph.  What the old world reads from its catalogue is measured here from the graph: mass, reach, wheels, contact.

The world holds the brief, the physics and what can be bought, and no advice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np

from sea.evolve.core import Body, BomLine, Part, PartType, Task
from sea.mech import graph as G
from sea.mech.build import printed_lines
from sea.mech.laws import floating, interference
from sea.mech.world import SLOT, GraphWorld, graph_of
from sea.sim import mobile_model as MM
from sea.worlds import mobile as W

Y_AXIS = (0.7071068, 0.7071068, 0.0, 0.0)  # a wheel's axis along y
DRIVER_EVERY = 2  # a dual H-bridge board per two motors
CM = 0.01
KIT_RULES = [
    "Parts and prices are in the manifest.  A gearmotor is what can be bought: its stall torque and no-load speed at "
    "the joint are the kind's numbers at its ratio, scaled by the pack's voltage over its winding.  A Bought cell gives "
    "the pack, in series, and states its table mass.  A RangeFinder is a sensor that can be bought, by its range.",
]


CORNERS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
CASE_MM = {"n20_gearmotor": (34.0, 12.0, 10.0), "n20_encoder_gearmotor": (40.0, 12.0, 10.0), "jga25_gearmotor": (60.0, 25.0, 25.0)}


def _motor_rating(kind: str, ratio: int, volts: float) -> tuple[float, float]:
    tau, w0, geta = W.MOTOR[kind][:3]
    k = volts / W.WINDING[kind]
    return k * tau * ratio * geta, k * w0 / ratio


def _motor_mount(k: int, x: float, sy: int, w: float, r: float, ride: float, hw: float, kind: str) -> tuple[Any, ...]:
    """A gearmotor as what it is (D-115): its case inside the wheel on the wheel's axis, a post from the case to the
    floor, and the shaft from the case to the wheel."""
    (cl, cw, ch), gap = CASE_MM[kind], hw + 0.001
    half = (cl / 2000, cw / 2000, ch / 2000)
    y = sy * (w / 2 - half[0])
    return (G.Box(name=f"motor_case{k}", pos=(x, y, r), half=(half[1], half[0], half[2]), material="stock", kg=W.MOTOR[kind][3] / 1000),
            G.Box(name=f"axle_post{k}", pos=(x, y, (r + ride) / 2), half=(0.003, 0.003, max(abs(ride - r) / 2, 0.001))),
            G.Cylinder(name=f"axle{k}", pos=(x, sy * (w / 2 + gap / 2), r), radius=0.002, half_length=gap / 2, quat=Y_AXIS))


def _chassis(p: dict[str, float]) -> G.Graph:
    """A plate on four wheels, each on its own gearmotor, carrying a pack of cells."""
    w, ln, ride, r = (p[k] * CM for k in ("width_cm", "length_cm", "ride_cm", "wheel_radius_cm"))
    kind, cells = "n20_gearmotor", max(1, round(p["cells"]))
    ratio = min(W.RATIOS[kind], key=lambda x: abs(x - p["gear_ratio"]))
    stall, w0 = _motor_rating(kind, ratio, cells * W.VOLTS["nimh_pack"][0])
    hw, cell = 0.0125, W.CELL["nimh_pack"]
    arm = W.ROTOR[kind] * ratio**2
    wheels, motors = [], []
    for k, (sx, sy) in enumerate(CORNERS):
        wheels.append(G.Body(name=f"wheel{k}", parent="robot", pos=(sx * (ln / 2 - r / 2), sy * (w / 2 + hw + 0.001), r),
                             joints=(G.Joint(name=f"wheel{k}_q", type="hinge", axis=(0.0, 1.0, 0.0), armature=arm),),
                             parts=(G.Wheel(name=f"tyre{k}", pos=(0.0, 0.0, 0.0), radius=r, half_width=hw, quat=Y_AXIS, fill=0.15),)))
        motors.append(G.Gearmotor(name=f"motor{k}", joint=f"wheel{k}_q", stall_nm=stall, no_load_rad_s=w0))
    tub = (G.Box(name="tub_floor", pos=(0.0, 0.0, ride + 0.0015), half=(ln / 2, w / 2, 0.0015)),
           G.Box(name="tub", pos=(0.0, 0.0, ride + W.TUB_H * CM / 2), half=(ln / 2, w / 2, W.TUB_H * CM / 2), massive=False),
           *(x for k, (sx, sy) in enumerate(CORNERS) for x in _motor_mount(k, sx * (ln / 2 - r / 2), sy, w, r, ride, hw, kind)),
           *(G.Bought(item=cell[0], kg=cell[2] / 1000) for _ in range(cells)))
    return G.Graph(name="chassis", bodies=(G.Body(name="robot", parent="world", joints=(G.Joint(name="root", type="free"),),
                                                  parts=tub), *wheels), actuators=tuple(motors))


def _parts(g: G.Graph) -> list[Any]:
    return [p for b in g.bodies for p in b.parts]


def _pack(g: G.Graph) -> tuple[str, int]:
    """The pack the graph carries: its kind and how many cells in series."""
    for kind, c in W.CELL.items():
        n = sum(isinstance(p, G.Bought) and p.item == c[0] for p in _parts(g))
        if n:
            return kind, n
    return "", 0


def _motors(g: G.Graph) -> tuple[list[tuple[G.Gearmotor, str, int]], list[str]]:
    """Each gearmotor with the kind and ratio it is, at the pack's voltage, and what cannot be bought."""
    kind, n = _pack(g)
    volts = n * W.VOLTS[kind][0] if kind else 0.0
    out, bad = [], []
    for a in g.actuators:
        if not isinstance(a, G.Gearmotor):
            continue
        hit = next(((k, r) for k, rs in W.RATIOS.items() for r in rs
                    if volts and all(abs(x - y) <= 0.01 * y for x, y in zip((a.stall_nm, a.no_load_rad_s), _motor_rating(k, r, volts), strict=True))),
                   None)
        if hit is None:
            bad.append(f"{a.name} at {a.stall_nm:g} N m and {a.no_load_rad_s:g} rad/s is no gearmotor that can be bought, "
                       f"at the pack's {volts:g} V")
        else:
            out.append((a, *hit))
    return out, bad


def _wheels(g: G.Graph) -> dict[str, G.Wheel]:
    """Each wheel by the joint it turns on."""
    return {b.joints[0].name: p for b in g.bodies for p in b.parts if isinstance(p, G.Wheel) and b.joints}


def _sensor(g: G.Graph) -> str:
    """The sensor the graph carries: the longest range it has."""
    ranges = [s.max_m for s in g.sensors if isinstance(s, G.RangeFinder)]
    if ranges:
        return next((k for k, v in W.SENSOR.items() if abs(v[0] - max(ranges)) < 1e-6), "")
    return "bump_switch"


def kit_mass(g: G.Graph) -> float:
    """What the kit adds in kg, without a solid of its own: electronics, drivers and sensor."""
    motors, _ = _motors(g)
    grams = W.ELEC_G + sum(W.DRIVER[k][1] / DRIVER_EVERY for _, k, _ in motors) + W.SENSOR.get(_sensor(g), W.SENSOR["bump_switch"])[2]
    return grams / 1000


@lru_cache(maxsize=256)
def _measured(graph_json: str) -> dict[str, float]:
    """What the graph measures at rest on flat ground: cm, and kg with the kit."""
    g = G.Graph.model_validate_json(graph_json)
    m = MM.graph_model(g, "floor", {}, 0.6, kit_mass(g))[0]
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    ids = np.array([i for i in range(m.ngeom) if m.geom_contype[i] == 2])
    lo, hi = _extent(m, d, ids)
    wheel = {f"{p.name}" for p in _wheels(g).values()}
    rest = [i for i in ids if m.geom(i).name not in wheel]
    belly = float(min((_extent(m, d, np.array([i]))[0][2] for i in rest), default=lo[2]))
    return {"width": (hi[1] - lo[1]) / CM, "length": (hi[0] - lo[0]) / CM, "height": hi[2] / CM, "belly": belly / CM,
            "mass": float(m.body_subtreemass[m.body("robot").id])}


def _extent(m: mujoco.MjModel, d: mujoco.MjData, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in ids:
        c, h = d.geom_xpos[i], np.abs(d.geom_xmat[i].reshape(3, 3)) @ _half(m, i)
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def _half(m: mujoco.MjModel, i: int) -> np.ndarray:
    s = m.geom_size[i]
    t = int(m.geom_type[i])
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        return s.copy()
    if t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        extra = s[0] if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0
        return np.array([s[0], s[0], s[1] + extra])
    return np.full(3, m.geom_rbound[i])


class MobileKit(GraphWorld, W.MobileWorld):
    name = "mobile_kit"
    description = ("A small ground robot explores a house and its garden, built from parts rather than a catalogue.  Each "
                   "task is a route from its dock to an exploration target, over floors, edges, openings, trenches, ramps, "
                   "sand, low furniture and dark corridors.  System 1 drives each route (cruise duty, the line through an "
                   "opening, the approach duty at an edge, the speed in the dark).")
    frame = ("A robot is a graph (lengths in metres; a part's coordinates in its body's frame; x forward, z up from the "
             "floor).  Its first body is 'robot' on a free joint 'root'.  A gearmotor on a wheel's joint drives it forward.")
    modules = {"chassis": (_chassis, {"width_cm": (8.0, 30.0), "length_cm": (8.0, 35.0), "ride_cm": (0.5, 6.0),
                                      "wheel_radius_cm": (1.5, 8.0), "gear_ratio": (50.0, 298.0), "cells": (3.0, 8.0)})}

    def catalogue(self) -> list[PartType]:
        return []

    def stock(self) -> Body:
        return Body(parts={SLOT: Part(type="built", mechanism=self.module("chassis", {
            "width_cm": 18.0, "length_cm": 20.0, "ride_cm": 2.0, "wheel_radius_cm": 3.0, "gear_ratio": 100.0,
            "cells": 4.0}).model_dump_json())})

    def spec_of(self, body: Body) -> W.Spec:
        return _spec(body.parts[SLOT].mechanism)

    def brief_rules(self) -> list[str]:
        return list(KIT_RULES)

    def principles(self) -> list[str]:
        return []

    def bom(self, body: Body) -> list[BomLine]:
        g = graph_of(body)
        motors, _ = _motors(g)
        kind, cells = _pack(g)
        out = printed_lines(g, SLOT, skip=lambda p: isinstance(p, G.Bought))
        if _wheels(g):
            out.append(BomLine(item="tire_oring", qty=len(_wheels(g)), slot=SLOT))
        counts: dict[str, int] = {}
        for _, k, r in motors:
            counts[f"{k}_{r}"] = counts.get(f"{k}_{r}", 0) + 1
        out += [BomLine(item=i, qty=n, slot=SLOT) for i, n in counts.items()]
        drivers: dict[str, int] = {}
        for _, k, _ in motors:
            drivers[W.DRIVER[k][0]] = drivers.get(W.DRIVER[k][0], 0) + 1
        out += [BomLine(item=i, qty=math.ceil(n / DRIVER_EVERY), slot=SLOT) for i, n in drivers.items()]
        if kind:
            out.append(BomLine(item=W.CELL[kind][0], qty=cells, slot=SLOT))
        if kind == "liion_pack":
            out.append(BomLine(item="bms_board", qty=1, slot=SLOT))
        sensor = W.SENSOR.get(_sensor(g))
        if sensor is not None:
            out.append(BomLine(item=sensor[4], qty=sensor[5], slot=SLOT))
        return out

    def check(self, body: Body) -> list[str]:
        return list(_laws(body.parts[SLOT].mechanism))

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        g = graph_of(body)
        joints = [f"{j.name} ({j.type}{f', {j.range[0]:.3g}..{j.range[1]:.3g}' if j.range else ''})" for b in g.bodies
                  for j in b.joints if j.type != "free"]
        return super().brief(body, tasks) + "\njoints: " + (", ".join(joints) or "none")


def _caseless(g: G.Graph, motors: list[tuple[G.Gearmotor, str, int]]) -> list[str]:
    """A gearmotor is a solid on what carries its wheel, on the wheel's axis, the size and mass it is (D-115)."""
    m = MM.graph_model(g, "floor", {}, 0.6, 0.0)[0]
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    boxes = {p.name: p for p in _parts(g) if isinstance(p, G.Box) and p.material == "stock"}
    out, used = [], set()
    for a, kind, _ in motors:
        j = m.joint(a.joint).id
        size = np.sort(np.array(CASE_MM[kind]) / 2000)
        grams = W.MOTOR[kind][3] / 1000
        hit = None
        for name, p in boxes.items():
            i = m.geom(name).id
            off = d.geom_xpos[i] - d.xanchor[j]
            if (name not in used and m.geom_bodyid[i] == m.body_parentid[m.jnt_bodyid[j]] and np.allclose(np.sort(p.half), size, atol=1e-4)
                    and abs((p.kg or 0.0) - grams) < 1e-6 and np.linalg.norm(np.cross(off, d.xaxis[j])) < 0.002):
                hit = name
                break
        if hit is None:
            out.append(f"{a.name} has no case on {a.joint}'s axis on what carries it: a stock Box of {CASE_MM[kind][0]:g} x "
                       f"{CASE_MM[kind][1]:g} x {CASE_MM[kind][2]:g} mm and {W.MOTOR[kind][3]:g} g")
        else:
            used.add(hit)
    return out


@lru_cache(maxsize=256)
def _laws(graph_json: str) -> tuple[str, ...]:
    g = G.Graph.model_validate_json(graph_json)
    errs: list[str] = []
    if not g.bodies or g.bodies[0].name != "robot" or not any(j.type == "free" for j in g.bodies[0].joints):
        return ("the graph's first body must be 'robot' on a free joint named 'root'",)
    kind, cells = _pack(g)
    if not kind:
        errs.append(f"it carries no pack: a Bought cell, one of {sorted(c[0] for c in W.CELL.values())}")
    for p in _parts(g):
        if isinstance(p, G.Bought):
            cell = next((c for c in W.CELL.values() if c[0] == p.item), None)
            if cell is None:
                errs.append(f"{p.item} is not a cell: the kit's mass is added from what the graph uses")
            elif abs(p.kg - cell[2] / 1000) > 1e-6:
                errs.append(f"a {p.item} is {cell[2]:g} g, not {p.kg * 1000:g}")
    motors, bad = _motors(g)
    errs += bad
    errs += [f"{a.name} is on {a.joint}, which turns no wheel" for a, _, _ in motors if a.joint not in _wheels(g)]
    errs += _caseless(g, motors)
    if not any(a.joint in _wheels(g) for a, _, _ in motors):
        errs.append("nothing drives a wheel")
    if kind:
        volts_full = cells * W.VOLTS[kind][1]
        amps = sum(W.MOTOR[k][4] * cells * W.VOLTS[kind][0] / W.WINDING[k] for _, k, _ in motors)
        for k in {k for _, k, _ in motors}:
            lo, hi = (f * W.WINDING[k] for f in W.V_RANGE)
            if cells * W.VOLTS[kind][0] < lo or volts_full > hi:
                errs.append(f"{cells} {kind} cells give {cells * W.VOLTS[kind][0]:.1f} V ({volts_full:.1f} V charged); a "
                            f"{W.WINDING[k]:.0f} V {k} winding takes {lo:.1f}-{hi:.1f} V")
        if amps > W.CELL[kind][4]:
            errs.append(f"its motors draw up to {amps:.1f} A at stall; the {kind} supplies at most {W.CELL[kind][4]:.0f} A")
    if not _sensor(g):
        errs.append("a RangeFinder's range is no sensor that can be bought: " + ", ".join(f"{k} {v[0]:g} m" for k, v in W.SENSOR.items()))
    if errs:
        return tuple(errs)
    try:
        m = MM.graph_model(g, "floor", {}, 0.6, kit_mass(g))[0]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return (f"it cannot be built: {str(exc)[:300]}",)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return (*floating(g, m, d), *interference(g, m, (), (CM, "cm"), 0.002))


@lru_cache(maxsize=256)
def _spec(graph_json: str) -> W.Spec:
    """The numbers the routes read, measured from the graph."""
    g = G.Graph.model_validate_json(graph_json)
    motors, _ = _motors(g)
    wheels = _wheels(g)
    driving = [(a, k) for a, k, _ in motors if a.joint in wheels]
    kind, cells = _pack(g)
    me = _measured(graph_json)
    f_stall = sum(a.stall_nm / wheels[a.joint].radius for a, _ in driving) or 1e-6
    v_nl = min((a.no_load_rad_s * wheels[a.joint].radius for a, _ in driving), default=0.0)
    r = min((w.radius for w in wheels.values()), default=0.03) / CM
    area = sum(2 * w.half_width / CM * 0.5 * w.radius / CM for w in wheels.values())
    first = driving[0][1] if driving else "n20_gearmotor"
    sensor = W.SENSOR.get(_sensor(g), W.SENSOR["bump_switch"])
    return W.Spec(loco="built", m=me["mass"], width=me["width"], height=me["height"], length=me["length"], belly=me["belly"], r=r,
                  n=len(driving), f_stall=f_stall, v_nl=v_nl, eta=0.7 * W.MOTOR[first][2], drift=W.MOTOR[first][5], sense=sensor[0],
                  wh=cells * W.CELL[kind][1] if kind else 0.0, p_w=W.ELEC_W + sensor[3], area=area, base=me["length"],
                  volts=cells * W.VOLTS[kind][0] if kind else 0.0, graph=graph_json, kit_kg=kit_mass(g))
