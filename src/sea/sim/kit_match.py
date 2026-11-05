"""A robot written as a graph, in ME 2110's arena (D-094): the same fortress, events and scoring as any match.

``build`` compiles the graph into the arena on one track, placing the items it carries, and ``KitMatch`` plays its
program there.  Nothing here names a mechanism: what a robot can do comes from its graph and its controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sea.mech import build as B
from sea.mech.base import WORLD
from sea.mech.build import printed_members
from sea.mech.graph import Gearmotor, Graph, Piston, Servo, TorsionSpring
from sea.mech.members import PrintedMembers
from sea.mech.program import Program
from sea.mech.runtime import Machine
from sea.sim import me2110_model as MM

IN = MM.IN
START_GAP = 0.25  # in: the robot's front edge behind the start line
START_JITTER = 0.1  # in: how far from its mark a robot is set down
KIT_KG = {"motor": 0.200, "tank": 0.350, "valve": 0.080, "trap": 0.030, "controller": 0.300}  # the kit's own masses
ROOT = "robot"


def kit_mass(g: Graph) -> float:
    """What the kit parts a graph uses add: its motors, traps, valves, the tank if there is a piston, the controller."""
    n = {t: sum(isinstance(a, t) for a in g.actuators) for t in (Gearmotor, Servo, Piston, TorsionSpring)}
    return (KIT_KG["controller"] + KIT_KG["motor"] * (n[Gearmotor] + n[Servo]) + KIT_KG["trap"] * n[TorsionSpring]
            + (KIT_KG["tank"] + KIT_KG["valve"] * n[Piston] if n[Piston] else 0.0))


def _boxes(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = d.geom_xmat[geoms].reshape(-1, 3, 3)
    c = d.geom_xpos[geoms] + np.einsum("nij,nj->ni", R, m.geom_aabb[geoms, :3])
    return c, np.einsum("nij,nj->ni", np.abs(R), m.geom_aabb[geoms, 3:])


def _item(spec: mujoco.MjSpec, name: str, pos: np.ndarray, quat: np.ndarray, mount: str) -> None:
    """An item of the arena at this pose: a ball sits free; the flag is held to its mount by a clip."""
    if name.startswith("ball"):
        b = spec.worldbody.add_body(name=name, pos=list(pos))
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[MM.BALL_R * IN, 0.0, 0.0], mass=MM.M_BALL,
                   friction=[0.3, 0.005, 0.0001], contype=1, conaffinity=3, solref=[0.004, 1.0],
                   solimp=[0.95, 0.99, 0.001, 0.5, 2.0], rgba=[0.05, 0.05, 0.05, 1.0])
        return
    if name != "flag":
        raise ValueError(f"no item {name!r} to carry: the items are the balls ball0..ball3 and the flag")
    b = spec.worldbody.add_body(name="flag", pos=list(pos), quat=list(quat))
    b.add_freejoint()
    b.add_geom(name="flag_plate", type=mujoco.mjtGeom.mjGEOM_BOX, mass=0.010, pos=[MM.FLAG_T / 2 * IN, 0.0, 0.0],
               size=[MM.FLAG_T / 2 * IN, MM.FLAG_W / 2 * IN, MM.FLAG_H / 2 * IN], contype=4, conaffinity=1,
               solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0], friction=[0.5, 0.005, 0.0001])
    for nm, other, active in (("clip", mount, True), ("magnet", "fortress", False)):
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=nm, name1="flag", name2=other, active=active,
                          objtype=mujoco.mjtObj.mjOBJ_BODY, data=[0.0] * 10 + [1.0, 0.0], solref=[0.004, 1.0])


@dataclass
class Compiled:
    model: mujoco.MjModel
    extent_in: tuple[float, float, float, float, float, float]  # x lo, x hi, y lo, y hi, z lo, z hi: at the start pose


def _adopt(spec: mujoco.MjSpec, g: Graph) -> None:
    """The robot's solids take the arena's contact properties: they meet the arena, not each other."""
    names = {b.name for b in g.bodies}
    for x in spec.geoms:
        if x.parent.name in names:
            x.contype, x.conaffinity = 2, 1
            x.solref, x.solimp = [0.004, 1.0], [0.95, 0.99, 0.001, 0.5, 2.0]
            if x.friction[0] == 1.0:
                x.friction = [0.5, 0.005, 0.0001]
    for j in spec.joints:
        if j.parent.name in names:
            j.solref_limit, j.solimp_limit = [0.004, 1.0], [0.99, 0.999, 0.0005, 0.5, 2.0]


def compile_robot(g: Graph, v: MM.Variant, y0: float = 0.0, add_kit_mass: bool = True, x0: float | None = None) -> Compiled:
    """The arena on track ``v`` with ``g`` in the dock: its front ``START_GAP`` behind the line, at ``y0`` (in)."""
    if not g.bodies or g.bodies[0].name != ROOT:
        raise ValueError(f"the graph's first body must be {ROOT!r}, on a free joint named 'root'")
    spec = mujoco.MjSpec.from_string(MM.arena_xml(v))
    B.compile(spec, g, WORLD)
    _adopt(spec, g)
    if add_kit_mass and kit_mass(g) > 0:
        spec.body(ROOT).add_geom(name="kit_parts", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0.0, 0.0],
                                 pos=[0.0, 0.0, 0.05], mass=kit_mass(g), contype=0, conaffinity=0, group=4)
    probe = spec.compile()
    d = mujoco.MjData(probe)
    mujoco.mj_kinematics(probe, d)
    root = probe.body(ROOT).id
    geoms = np.array([i for i in range(probe.ngeom) if probe.geom_contype[i] == 2])
    c, h = _boxes(probe, d, geoms)
    home = d.xpos[root].copy()
    ext = [float(v_) / IN for pair in zip(np.min(c - h, 0) - home, np.max(c + h, 0) - home, strict=True) for v_ in pair]
    at = np.array([(-(ext[1] + START_GAP) if x0 is None else x0) * IN, y0 * IN, 0.0])
    for it in g.carried:
        bid = probe.body(it.body).id
        R = d.xmat[bid].reshape(3, 3)
        pos = d.xpos[bid] + R @ np.asarray(it.pos) - home + at
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, d.xquat[bid], np.asarray(it.quat))
        _item(spec, it.item, pos, q, it.body)
    spec.body(ROOT).pos = at.tolist()
    return Compiled(spec.compile(), tuple(ext))  # type: ignore[arg-type]


class KitMatch(MM.Match):
    """One match of a graph robot under its program."""

    def __init__(self, g: Graph, v: MM.Variant, ctl: dict[str, float], seed: int = 0, horizon: float = 40.0,
                 triggers: dict[str, str] | None = None, add_kit_mass: bool = True, x0: float | None = None,
                 jitter: bool = True) -> None:
        y0 = ctl.get("y0", 0.0)
        if jitter:  # where a team sets a robot down: no two matches start in quite the same place
            y0 += float(np.random.default_rng(seed).normal(0.0, START_JITTER))
        cp = compile_robot(g, v, y0, add_kit_mass, x0)
        bd = MM.Build()
        bd.balls = [it.item for it in g.carried if it.item.startswith("ball")]
        bd.flag = any(it.item == "flag" for it in g.carried)
        area = math.pi * (MM.BORE * IN / 2) ** 2
        bd.pistons = {a.name: {"a_ext": area, "a_ret": area - math.pi * (MM.ROD * IN / 2) ** 2, "rod": a.joint}
                      for a in g.actuators if isinstance(a, Piston)}
        rb = MM.Robot(L=cp.extent_in[1] - cp.extent_in[0], W=cp.extent_in[3] - cp.extent_in[2], H=cp.extent_in[5],
                      drive="kit")
        self._setup(rb, v, ctl, seed, horizon, None, False, cp.model, "", bd)
        self.g = g
        self.gate_t = 0.0  # the projectile rule watches every ball from the start: any mechanism may let one go
        self.machine = Machine(self.m, self.d, g)
        members = printed_members(g, self.m)
        self.members = PrintedMembers(self.m, members) if members else None
        self.member_broke = False
        self.program = Program(g, ctl, 40.0, triggers)
        self.pistons = [a for a in g.actuators if isinstance(a, Piston)]
        self.commanded = [a.name for a in g.actuators if isinstance(a, Servo | Gearmotor | Piston)]
        self.held = {lt.name for lt in g.latches}
        mujoco.mj_forward(self.m, self.d)

    def control(self) -> None:
        d = self.d
        d.ctrl[self.aid["spin"]] = -self.v.rpm * 2 * math.pi / 60
        if self.members is not None and not self.member_broke:  # printed members, judged as on any robot (D-109)
            self.members.read(d)
            if broke := self.members.broken():
                self.member_broke = True
                self.laws.append(broke)
        self.program.step(self.machine, d.time)
        for name in list(self.held):
            if not self.machine.held(name):
                self.held.discard(name)
                self.released.add(name)
                self.log(f"released {name}")
        if self.phase == "idle" and (self.released or any(d.ctrl[self.aid[n]] != 0.0 for n in self.commanded)):
            self.phase = "active"
        for a in self.pistons:  # one tank: each stroke draws it down, and every piston then pushes less
            u = float(d.ctrl[self.aid[a.name]])
            if u != 0.0:
                self.stroke(a.name, u > 0)
            g = self.bd.pistons[a.name]
            force = u * self.p_tank * (g["a_ext"] if u >= 0 else g["a_ret"])  # N
            jid, adr = self.jid[a.joint], self.m.jnt_qposadr[self.jid[a.joint]]
            lo, hi = self.m.jnt_range[jid]
            q = float(d.qpos[adr])
            if self.m.jnt_limited[jid] and ((u > 0 and q > hi - 0.001) or (u < 0 and q < lo + 0.001)):
                force = math.copysign(a.end_hold_n, u)  # at an end of its stroke the cylinder's end cap takes the rest
            d.ctrl[self.aid[a.name]] = force / (a.pressure_pa * g["a_ext"])

    def _ir(self) -> None:  # the old catalogue's sensors: a graph's own sensors read the simulator
        return

    def _face(self) -> None:
        return
