"""The cast tube as a thing the robot presses on (D-122): silicone wall segments in the simulator, and servos that have
to move them.

Along the tube, sections of RING wall segments, each free to move across the axis, each sprung to its place by the ring's
bending, and tied to its neighbours by the wall's shear.  The stiffnesses come from the cast's modulus and wall (a ring
pinched: F/d = E I / (0.149 R^3), Roark; shear: G t w / L).  Segments meet each other and every solid of the graph, so a
closure happens when one wall reaches the other, not when something comes near the axis.

A posture is a command: the servos slew to it at their own speed from the rest posture, with their own torque, and the
area function is read from where the walls came to rest.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sea.mech import graph as G
from sea.mech.base import WORLD
from sea.mech.build import compile as compile_graph
from sea.mech.laws import _close

CM = 0.01
NT = 24
BORE, RING_R = 1.2, 1.45  # cm
E = 7.0e4  # Pa: a soft platinum silicone, Shore 00-30
WALL = 0.002  # m
RHO_SILICONE = 1070.0  # kg/m3
NU = 0.5
RING = 8
SECTIONS = 12
SLEW = np.radians(60.0) / 0.1  # rad/s: an MG90S at 4.8 V
DT = 0.001
SETTLE = 0.15  # s after the servos arrive: formants converge to 0.2%
STALL = 0.216  # N m
MOTOR_LINE = STALL / SLEW  # N m s/rad: the torque-speed line's slope, the gearbox's own drag
ROTOR = 0.001  # kg m2 seen through the gears: stall torque covers 60 degrees from rest in the datasheet's 0.1 s
MEMORY = 2048  # postures remembered: one search asks for one posture many times
RAD = WALL / 2
ZETA = 0.5  # the wall's damping ratio
HINGE_CLEAR = 0.01  # m


def _at_rest(r_cm: float) -> float:
    """What the polygon of a ring's segments reads at rest, cm2: its round tube's area, from this."""
    r = r_cm * CM + RAD
    side = 2 * r * np.sin(np.pi / RING)
    return (0.5 * RING * r * side * np.cos(np.pi / RING) - RAD * RING * side + np.pi * RAD**2) / CM**2


def wall(radius_m: float, length_m: float) -> tuple[float, float, float, float]:
    """A wall segment's mass, its spring to its place, its shear tie along the tube and around the ring."""
    circ = 2 * np.pi * radius_m / RING
    mass = RHO_SILICONE * circ * length_m * WALL
    i = length_m * WALL**3 / 12
    k = E * i / (0.149 * radius_m**3)
    g = E / (2 * (1 + NU))
    return mass, k, g * WALL * circ / length_m, 2 * k


class SoftTract:
    def __init__(self, g: G.Graph):
        self.g = g
        spec = WORLD.model()
        spec.option.timestep = DT
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        compile_graph(spec, g, WORLD)
        names = {b.name for b in g.bodies}
        for x in spec.geoms:
            if x.parent.name in names and (x.contype or x.conaffinity):
                x.contype, x.conaffinity = 2, 1
            elif x.name == "floor":  # the world's floor lies on the tube's axis: this robot stands on its own plate
                x.contype, x.conaffinity, x.group = 0, 0, 5
        # the tube's length from the graph at rest
        m0 = spec.copy().compile()
        probe = mujoco.MjData(m0)
        mujoco.mj_kinematics(m0, probe)
        ln, lip = probe.body("glottis").xpos[0], max(-probe.body("lip_end").xpos[0], 0.001)
        self.xs = np.r_[(np.arange(SECTIONS) + 0.5) / SECTIONS * ln, -lip / 2]  # sections, teeth to glottis then the lips
        self.ds = np.r_[np.full(SECTIONS, ln / SECTIONS), lip]
        self.radii = np.r_[np.full(SECTIONS, BORE * CM + RAD), RING_R * CM + RAD]
        angle = 2 * np.pi * np.arange(RING) / RING
        self.dirs = np.c_[np.cos(angle), np.sin(angle)]
        js = {}
        for s, (x, ds, r) in enumerate(zip(self.xs, self.ds, self.radii, strict=True)):
            mass, k, kl, kh = wall(r - RAD, ds)
            for a in range(RING):
                y, z = r * self.dirs[a]
                b = spec.worldbody.add_body(name=f"w{s}_{a}", pos=[x, y, z], gravcomp=1.0)
                c = 2 * ZETA * np.sqrt(mass * k)
                for ax, nm in (((0, 1, 0), "y"), ((0, 0, 1), "z")):
                    j = b.add_joint(name=f"w{s}_{a}{nm}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax, stiffness=k, damping=c)
                    js[(s, a, nm)] = j
                b.add_geom(name=f"wg{s}_{a}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[RAD, ds / 2, 0.0], quat=[0.7071068, 0.0, 0.7071068, 0.0],
                           mass=mass, contype=1, conaffinity=2, condim=1, group=3, priority=2, solref=[2 * DT, 1.0], solimp=[0.95, 0.99, 0.0005, 0.5, 2.0])
        def tie(n1, n2, k, mass):
            t = spec.add_tendon(name=f"t_{n1}_{n2}", stiffness=k, damping=2 * ZETA * np.sqrt(mass * k))
            t.wrap_joint(n1, 1.0)
            t.wrap_joint(n2, -1.0)
        for s in range(SECTIONS):
            mass, _, kl, kh = wall(BORE * CM, self.ds[s])
            for a in range(RING):
                for nm in "yz":
                    if s + 1 < SECTIONS:
                        tie(f"w{s}_{a}{nm}", f"w{s + 1}_{a}{nm}", kl, mass)
                    tie(f"w{s}_{a}{nm}", f"w{s}_{(a + 1) % RING}{nm}", kh, mass)
        for s in range(SECTIONS + 1):  # a wall meets itself
            for a in range(RING):
                for b in range(a + 2, RING):
                    if (b - a) % RING not in (1, RING - 1):
                        spec.add_pair(name=f"p{s}_{a}_{b}", geomname1=f"wg{s}_{a}", geomname2=f"wg{s}_{b}", condim=1, solref=[2 * DT, 1.0])
        self._solid_pairs(spec, g)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.dof, self.kp, self.kd, self.memory = {}, {}, {}, {}
        for a in g.actuators:
            if isinstance(a, G.Servo):
                dof = m.jnt_dofadr[m.joint(a.joint).id]
                m.dof_damping[dof], m.dof_armature[dof] = MOTOR_LINE, ROTOR
                self.dof[a.name], self.kp[a.name] = dof, a.kp
                self.kd[a.name] = 2 * np.sqrt(a.kp * ROTOR)  # its position loop, critically damped on the rotor
        self.servos = [a for a in g.actuators if isinstance(a, G.Servo)]
        self.act = {a.name: m.actuator(a.name).id for a in self.servos}
        self.adr = {a.name: m.jnt_qposadr[m.joint(a.joint).id] for a in self.servos}
        self.rng = {a.name: tuple(m.jnt_range[m.joint(a.joint).id]) for a in self.servos}
        self.wall_y = np.array([[m.jnt_qposadr[m.joint(f"w{s}_{a}y").id] for a in range(RING)] for s in range(SECTIONS + 1)])
        self.wall_z = self.wall_y + 1
        self.rest_y = self.radii[:, None] * self.dirs[:, 0][None]
        self.rest_z = self.radii[:, None] * self.dirs[:, 1][None]
        self.glottis, self.lip_end = m.body("glottis").id, m.body("lip_end").id
        moved = {a.joint for a in g.actuators}
        self.free = sorted({m.joint(j.name).id for lp in g.loops for b in g.bodies if b.name in (lp.a, lp.b) for j in b.joints if j.name not in moved})
        ang = 2 * np.pi * np.arange(2 * RING) / (2 * RING)
        self.ray = np.stack([np.zeros_like(ang), np.cos(ang), np.sin(ang)], 1)
        self.groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        self.rest_q = None
        self._rest()

    @staticmethod
    def _solid_pairs(spec: mujoco.MjSpec, g: G.Graph) -> None:
        """Solids meet: any two on different bodies, except a part and what carries it within HINGE_CLEAR of their hinge,
        where a clevis or a bearing may share space."""
        pos = {b.name: b for b in g.bodies}
        have = {x.name for x in spec.geoms}
        parent = {b.name: b.parent for b in g.bodies}
        solids = [(b.name, p) for b in g.bodies for p in b.parts if p.name in have]
        n = 0
        for i, (a, p) in enumerate(solids):
            for b, q in solids[i + 1:]:
                if a == b or (not any(x.joints for x in (pos[a], pos[b]))):
                    continue
                if parent[a] == b or parent[b] == a:
                    child, part = (a, p) if parent[a] == b else (b, q)
                    if np.linalg.norm(part.pos) < HINGE_CLEAR + max(getattr(part, "half", (0, 0, 0))):
                        continue
                n += 1
                spec.add_pair(name=f"s{n}", geomname1=p.name, geomname2=q.name, condim=1, solref=[2 * DT, 1.0])

    def _run(self, target: dict[str, float], settle: float = SETTLE, slew: float = SLEW, record=None) -> float:
        m, d = self.m, self.d
        start = {k: float(d.qpos[self.adr[k]]) for k in target}
        far = max((abs(target[k] - start[k]) for k in target), default=0.0)
        t_move = far / slew
        n = int((t_move + settle) / DT)
        peak = 0.0
        for i in range(n):
            f = min((i + 1) * DT / t_move, 1.0) if t_move > 0 else 1.0
            for k in target:
                rate = (target[k] - start[k]) / t_move if f < 1.0 else 0.0
                d.ctrl[self.act[k]] = start[k] + f * (target[k] - start[k]) + (self.kd[k] / self.kp[k]) * (rate - d.qvel[self.dof[k]])
            mujoco.mj_step(m, d)
            if record is not None:
                record(i)
            if i >= n - 5:  # what it holds with, once it has come to rest
                peak = max(peak, max((abs(d.actuator_force[self.act[k]]) for k in target), default=0.0))
        return peak

    def _intrusion(self) -> float:
        """How far solids stand inside the bore now, summed over the walls: rays out from the axis."""
        m, d = self.m, self.d
        total = 0.0
        gid = np.array([-1], np.int32)
        for s in range(SECTIONS + 1):
            for a in range(RING):
                hit = mujoco.mj_ray(m, d, np.array([self.xs[s], 0.0, 0.0]), np.array([0.0, *self.dirs[a]]), self.groups, 1, -1, gid)
                if 0 <= hit < self.radii[s]:
                    total += self.radii[s] - hit
        return total

    def _rest(self) -> None:
        """The servo posture with least of the robot inside the bore; a wall still in the way lies on the air's side of
        the solid, which is where the cast is."""
        d, m = self.d, self.m
        names = list(self.rng)
        corners = np.array(np.meshgrid(*[[lo, 0.5 * (lo + hi), hi] for lo, hi in self.rng.values()])).reshape(len(names), -1).T
        if len(corners) > 81:
            corners = corners[np.random.default_rng(0).choice(len(corners), 81, replace=False)]
        best, best_q = np.inf, None
        for q in corners:
            mujoco.mj_resetData(m, d)
            for k, v in zip(names, q, strict=True):
                d.qpos[self.adr[k]] = v
            _close(m, d, self.free)
            mujoco.mj_kinematics(m, d)
            e = self._intrusion()
            if e < best:
                best, best_q = e, d.qpos.copy()
        d.qpos[:] = best_q
        mujoco.mj_kinematics(m, d)
        gid = np.array([-1], np.int32)
        for s in range(SECTIONS + 1):
            for a in range(RING):
                hit = mujoco.mj_ray(m, d, np.array([self.xs[s], 0.0, 0.0]), np.array([0.0, *self.dirs[a]]), self.groups, 1, -1, gid)
                if 0 <= hit < self.radii[s] + RAD:
                    move = hit - RAD - self.radii[s]
                    d.qpos[self.wall_y[s, a]], d.qpos[self.wall_z[s, a]] = move * self.dirs[a]
        mujoco.mj_forward(m, d)
        hold = {k: float(d.qpos[self.adr[k]]) for k in names}
        for k, v in hold.items():
            d.ctrl[self.act[k]] = v
        self._run(hold, settle=0.2)
        self.rest_q = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())

    def reset(self) -> None:
        d = self.d
        d.qpos[:], d.qvel[:], d.ctrl[:] = self.rest_q
        d.act[:] = 0
        d.time = 0
        mujoco.mj_forward(self.m, d)

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Areas, lengths, distances from where the walls and solids are now."""
        m, d = self.m, self.d
        length, lip = d.xpos[self.glottis][0] / CM, max(-d.xpos[self.lip_end][0] / CM, 0.1)
        at = np.r_[(1.0 - (np.arange(NT) + 0.5) / NT) * length, -np.array([lip / 4, 3 * lip / 4])]
        y = self.rest_y + d.qpos[self.wall_y]
        z = self.rest_z + d.qpos[self.wall_z]
        centres = self.xs / CM
        order = np.argsort(centres)
        areas = np.zeros(at.size)
        gid = np.array([-1], np.int32)
        for i, x in enumerate(at):
            # between the two nearest wall sections
            c = centres[order]
            j = int(np.clip(np.searchsorted(c, x), 1, c.size - 1))
            w = np.clip((x - c[j - 1]) / (c[j] - c[j - 1]), 0.0, 1.0)
            a, b = order[j - 1], order[j]
            py = (1 - w) * y[a] + w * y[b]
            pz = (1 - w) * z[a] + w * z[b]
            cy, cz = py.mean(), pz.mean()
            # the line through the wall's segments around, with a point between each pair
            yy, zz = np.empty(2 * RING), np.empty(2 * RING)
            yy[0::2], zz[0::2] = py - cy, pz - cz
            yy[1::2], zz[1::2] = 0.5 * (yy[0::2] + np.roll(yy[0::2], -1)), 0.5 * (zz[0::2] + np.roll(zz[0::2], -1))
            # what stands in the way, by rays from the centre: a solid inside the wall holds the wall's line back from it
            for q in range(2 * RING):
                rq = float(np.hypot(yy[q], zz[q]))
                if rq < 1e-6:
                    continue
                uy, uz = yy[q] / rq, zz[q] / rq
                hit = mujoco.mj_ray(m, d, np.array([x * CM, cy, cz]), np.array([0.0, uy, uz]), self.groups, 1, -1, gid)
                if 0 <= hit < rq - RAD:
                    yy[q], zz[q] = uy * (hit + RAD), uz * (hit + RAD)
            poly = 0.5 * abs((yy * np.roll(zz, -1) - zz * np.roll(yy, -1)).sum())
            perim = np.hypot(yy - np.roll(yy, -1), zz - np.roll(zz, -1)).sum()
            area = max(poly - RAD * perim + np.pi * RAD**2, 0.0) / CM**2  # the air inside the wall's inner surface
            area *= np.pi * (BORE if x >= 0 else RING_R) ** 2 / _at_rest(BORE if x >= 0 else RING_R)
            areas[i] = area
        lens = np.r_[np.full(NT, length / NT), [lip / 2, lip / 2]]
        return areas, lens, at

    def __call__(self, ctl: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A batch of postures, each commanded from rest: areas, lengths and distances, glottis first.  ``load`` and
        ``short``: what the hardest-pressed servo holds with (N m), and how far short of its command any servo stands (rad)."""
        n = next(iter(ctl.values())).shape[0] if ctl else 1
        keys = [k for k in self.act if k in ctl]
        got = []
        for i in range(n):
            target = {k: float(ctl[k][i]) for k in keys}
            key = tuple(np.round(list(target.values()), 6))
            if key not in self.memory:
                self.reset()
                if len(self.memory) >= MEMORY:
                    self.memory.pop(next(iter(self.memory)))
                load = self._run(target) if target else 0.0
                short = max((abs(target[k] - float(self.d.qpos[self.adr[k]])) for k in target), default=0.0)
                self.memory[key] = (*self.read(), load, short)
            got.append(self.memory[key])
        self.load = np.array([g[3] for g in got])
        self.short = np.array([g[4] for g in got])
        return tuple(np.stack([g[j] for g in got]) for j in range(3))

    def transition(self, first: dict[str, float], then: dict[str, float], hold: float = 0.02, after: float = 0.15, frame: float = 0.005
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Held at ``first``, then commanded to ``then``: the area function (F, N) every ``frame`` s, and its lengths."""
        self.reset()
        self._run(first)
        out, lens = [], None
        step = max(int(frame / DT), 1)
        start = {k: float(self.d.qpos[self.adr[k]]) for k in first}
        far = max((abs(then[k] - start[k]) for k in then), default=0.0)
        move = far / SLEW
        steps = int((hold + after) / DT)
        for i in range(steps):
            t = i * DT - hold
            f = min(max(t, 0.0) / move, 1.0) if move > 0 else 1.0
            for k in then:
                rate = (then[k] - start[k]) / move if 0.0 <= t and f < 1.0 else 0.0
                self.d.ctrl[self.act[k]] = start[k] + f * (then[k] - start[k]) + (self.kd[k] / self.kp[k]) * (rate - self.d.qvel[self.dof[k]])
            mujoco.mj_step(self.m, self.d)
            if i % step == 0:
                a, lens, _ = self.read()
                out.append(a)
        return np.stack(out), lens
