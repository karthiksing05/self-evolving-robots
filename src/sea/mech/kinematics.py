"""Where a mechanism's joints are at each servo angle, solved from its own constraints (D-079).

A gear mesh or a parallelogram moves its joints in proportion to the servo, and a graph may say so
(``Graph.linked``).  A general linkage does not: a four-bar with unequal links turns its coupler by an
angle that is no multiple of the crank's.  ``closure`` solves the mechanism's joints at each servo angle so
that every loop and mesh constraint holds (least squares on the simulator's own equality residuals, stepping
out from the closed jaw), and ``Follow`` carries the result as a table.  A mechanism that cannot close
somewhere in the servo's range jams there: ``closure`` reports where.

``finger_motion`` is the finger's pose through the range, which the grasp skills read in place of a
per-jaw-type formula (``design/fingers.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from sea.mech.base import GRIPPER, base
from sea.mech.graph import Blade, Graph, PadMount

GUIDE_JOINTS = ("jaw_rod_dy", "jaw_rod_dz", "jaw_rod_yaw", "jaw_rod_pitch")  # a guide's bending: zero when posed
STEPS = 61  # servo angles over the range at which the closure is solved
CLOSE_TOL = 1e-5  # m (or rad): a constraint left open by more than this does not close


@dataclass(frozen=True)
class Follow:
    """A joint that follows the servo by a table: its position at servo angles ``qs``."""

    qs: tuple[float, ...]
    vals: tuple[float, ...]

    def __call__(self, q: float) -> float:
        return float(np.interp(q, self.qs, self.vals))


def follow(r, q: float) -> float:
    """A linked joint's position at servo angle ``q``: ``r`` is a ratio or a ``Follow``."""
    return r(q) if callable(r) else q * r


def grid(lo: float, hi: float, steps: int = STEPS) -> np.ndarray:
    """Servo angles over the range, with the closed jaw (q = 0, where a graph is drawn assembled) among them."""
    return np.unique(np.r_[np.linspace(lo, hi, steps), 0.0])


def _free_joints(model: mujoco.MjModel, g: Graph) -> list[int]:
    names = [j.name for b in g.bodies if not b.stock for j in b.joints if j.name not in GUIDE_JOINTS]
    return [model.joint(n).id for n in names]


def _residual(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    mujoco.mj_forward(model, data)
    rows = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    return data.efc_pos[: data.nefc][rows].copy()


def closure(model: mujoco.MjModel, g: Graph, steps: int = STEPS,
            driver: str = "gripper") -> tuple[list[tuple[str, Follow]], float, float]:
    """Each of the mechanism's own joints as a ``Follow`` of the driving joint, and the worst constraint left open
    over its range with the angle where it is.  Joints already given in ``g.linked`` keep their ratios."""
    from scipy.optimize import least_squares

    given = {n for n, _ in g.linked}
    jids = [j for j in _free_joints(model, g) if model.joint(j).name not in given]
    if not jids:
        return [], 0.0, 0.0
    d = mujoco.MjData(model)
    servo = model.jnt_qposadr[model.joint(driver).id]
    linked = [(int(model.jnt_qposadr[model.joint(n).id]), r) for n, r in g.linked]
    adr = np.array([model.jnt_qposadr[j] for j in jids])
    lo, hi = model.jnt_range[model.joint(driver).id]
    qs = grid(lo, hi, steps)
    sol = np.zeros((len(qs), len(jids)))
    worst, at = 0.0, 0.0
    start = int(np.argmin(np.abs(qs)))  # the graph is drawn assembled at q = 0: step out from there both ways
    for order in (range(start, len(qs)), range(start, -1, -1)):
        x = np.zeros(len(jids))
        for i in order:
            d.qpos[:] = model.qpos0
            d.qpos[servo] = qs[i]
            for a, r in linked:
                d.qpos[a] = follow(r, qs[i])

            def f(v, d=d):
                d.qpos[adr] = v
                return _residual(model, d)

            res = least_squares(f, x, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
            x = sol[i] = res.x
            err = float(np.max(np.abs(res.fun))) if len(res.fun) else 0.0
            if err > worst:
                worst, at = err, float(qs[i])
    out = [(model.joint(j).name, Follow(tuple(qs.tolist()), tuple(sol[:, k].tolist()))) for k, j in enumerate(jids)]
    return out, worst, at


@lru_cache(maxsize=64)
def _tables(graph_json: str, base_name: str = GRIPPER.name) -> tuple[tuple[tuple[str, Follow], ...], float, float]:
    from sea.mech import build

    bs = base(base_name)
    g = Graph.model_validate_json(graph_json)
    spec = bs.model()
    build.compile(spec, g, bs)
    tables, worst, at = closure(spec.compile(), g, driver=bs.driver)
    return tuple(tables), worst, at


def closure_tables(g: Graph, base_name: str = GRIPPER.name) -> list[tuple[str, Follow]]:
    """``closure`` on the base robot with the mechanism, cached: the joints the graph does not link by ratio."""
    if not g.new_bodies():
        return []
    return list(_tables(graph_key(g), base_name)[0])


def closes(g: Graph, base_name: str = GRIPPER.name) -> tuple[float, float]:
    """The worst constraint left open over the driving joint's range, and where (m or rad, rad)."""
    if not g.new_bodies():
        return 0.0, 0.0
    return _tables(graph_key(g), base_name)[1:]


def finger_body(g: Graph, base_name: str = GRIPPER.name) -> tuple[str, np.ndarray]:
    """The finger: the body the pads ride on (a ``PadMount``'s, D-081, else the blade's), and the offset in it
    (where the jaw's frame at q = 0 sits in the finger body's frame)."""
    for b in g.bodies:
        for p in b.parts:
            if isinstance(p, PadMount):
                return b.name, np.asarray(p.offset)
    for b in g.bodies:
        for p in b.parts:
            if isinstance(p, Blade):
                return b.name, np.asarray(p.offset)
    return base(base_name).finger, np.zeros(3)


@lru_cache(maxsize=64)
def finger_motion(graph_json: str, base_name: str = GRIPPER.name) -> tuple[np.ndarray, np.ndarray, np.ndarray, str,
                                                                          np.ndarray]:
    """The finger through the driving joint's range: its angles, and at each the rotation and translation that
    carry a point fixed on the finger from where it is at q = 0 to where it is (root frame); the finger's body
    and the blade's offset in it."""
    from sea.mech import build

    bs = base(base_name)
    g = Graph.model_validate_json(graph_json)
    spec = bs.model()
    build.compile(spec, g, bs)
    model = spec.compile()
    tables, _, _ = closure(model, g, driver=bs.driver)
    linked = [(int(model.jnt_qposadr[model.joint(n).id]), r) for n, r in [*g.linked, *tables]]
    d = mujoco.MjData(model)
    servo = model.jnt_qposadr[model.joint(bs.driver).id]
    body, offset = finger_body(g, base_name)
    bid, gid = model.body(body).id, model.body(bs.root).id
    lo, hi = model.jnt_range[model.joint(bs.driver).id]
    qs = grid(lo, hi)
    rots, trans = [], []
    poses = []
    for q in qs:
        d.qpos[:] = model.qpos0
        d.qpos[servo] = q
        for a, r in linked:
            d.qpos[a] = follow(r, q)
        mujoco.mj_kinematics(model, d)
        rg = d.xmat[gid].reshape(3, 3)
        poses.append((rg.T @ d.xmat[bid].reshape(3, 3), rg.T @ (d.xpos[bid] - d.xpos[gid])))
    r0, t0 = poses[int(np.argmin(np.abs(qs)))]
    for r, t in poses:  # p(q) = R(q) p0 + T(q): the motion from the finger's pose at q = 0
        rel = r @ r0.T
        rots.append(rel)
        trans.append(t - rel @ t0)
    return qs, np.asarray(rots), np.asarray(trans), body, offset


def graph_key(g: Graph) -> str:
    return json.dumps(g.model_dump(), sort_keys=True)
