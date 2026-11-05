"""The laws any graph robot answers to before a trial (D-110): what is rigid must touch, what moves must be near what it
moves on, and parts may not pass through each other.  A robot's solids are the geoms the world gives contype 2.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from sea.mech import graph as G

TOUCH = 0.002  # m: two parts of one body closer than this are fastened
PIN = 0.01  # m: a jointed body this far from what it moves on can still be pinned to it
SWEEP = 5  # poses over each driven joint's range


def floating(g: G.Graph, m: mujoco.MjModel, d: mujoco.MjData) -> list[str]:
    """What is rigid must touch: each solid of a body, and of the bodies bolted to it without a joint, is fastened to
    another or it is floating.  What moves must be near what it moves on: a joint needs a pin to hold it (D-096)."""
    by_name = {b.name: b for b in g.bodies}

    def rigid(b: G.Body) -> str:  # the body this one is bolted to, through every body without a joint
        while not b.joints and b.parent in by_name:
            b = by_name[b.parent]
        return b.name

    groups: dict[str, list[int]] = {}
    for x in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[x])
        if m.geom_contype[x] == 2 and name in by_name:
            groups.setdefault(rigid(by_name[name]), []).append(x)
    out = []
    for name, ids in groups.items():
        reached, todo = {ids[0]}, [ids[0]]
        while todo:
            a = todo.pop()
            for c in ids:
                if c not in reached and mujoco.mj_geomDistance(m, d, a, c, TOUCH * 2, None) <= TOUCH:
                    reached.add(c)
                    todo.append(c)
        loose = sorted(m.geom(x).name for x in ids if x not in reached)
        if loose:
            out.append(f"on body {name!r}, {', '.join(loose)} touch(es) nothing fastened to {m.geom(ids[0]).name}: floating")
    for b in g.bodies:
        if not b.joints or b.parent not in by_name or b.name not in groups:
            continue
        on = rigid(by_name[b.parent])
        if on in groups and min(mujoco.mj_geomDistance(m, d, x, y, PIN * 2, None) for x in groups[b.name] for y in groups[on]) > PIN:
            out.append(f"body {b.name!r} comes nowhere near {on!r}, which its joint is on: nothing holds the joint")
    out += _unheld_hinges(g, m, d, rigid, groups)
    out += _unsupported(g, m, d)
    return out


def _unsupported(g: G.Graph, m: mujoco.MjModel, d: mujoco.MjData) -> list[str]:
    """A bought part given a place must sit on a solid of its body (D-115)."""
    out = []
    for b in g.bodies:
        bid = m.body(b.name).id
        solids = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and m.geom_group[i] != 4]
        for p in b.parts:
            if isinstance(p, G.Bought) and p.pos is not None:
                at = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.array(p.pos)
                if min((_distance_to_geom(m, d, i, at) for i in solids), default=np.inf) > PIN:
                    out.append(f"{p.item} on {b.name} is placed where no solid of it is")
    return out


def unjoined_solids(g: G.Graph, m: mujoco.MjModel) -> list[tuple[int, int]]:
    """Solids of moving bodies neither carrying the other: what ``interference``, one joint at a time, can't judge
    when two joints move together, and what ``excludes`` may hide from it (D-116)."""
    root = g.bodies[0].name
    by_body = {b.name: [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(b.name).id and m.geom_contype[i] == 2]
               for b in g.bodies if b.joints and b.name != root}
    parent = {b.name: b.parent for b in g.bodies}
    names = sorted(by_body)
    return [(x, y) for i, a in enumerate(names) for b in names[i + 1:] if parent[a] != b and parent[b] != a
            for x in by_body[a] for y in by_body[b]]


def overlapping(m: mujoco.MjModel, d: mujoco.MjData, pairs: Sequence[tuple[int, int]], tol: float = 0.0005) -> bool:
    """Do any of these solids pass through each other at this pose?  Such a pose does not exist in a real mechanism."""
    return any(mujoco.mj_geomDistance(m, d, x, y, tol + 1e-3, None) < -tol for x, y in pairs)


def _distance_to_geom(m: mujoco.MjModel, d: mujoco.MjData, g: int, p: np.ndarray) -> float:
    """From a point to a solid (m): in its frame, by its shape.  0 inside."""
    q = d.geom_xmat[g].reshape(3, 3).T @ (p - d.geom_xpos[g])
    s, t = m.geom_size[g], int(m.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(np.linalg.norm(np.maximum(np.abs(q) - s[:3], 0.0)))
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return max(float(np.linalg.norm(q)) - s[0], 0.0)
    if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return max(float(np.linalg.norm([q[0], q[1], max(abs(q[2]) - s[1], 0.0)])) - s[0], 0.0)
    if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return float(np.hypot(max(np.hypot(q[0], q[1]) - s[0], 0.0), max(abs(q[2]) - s[1], 0.0)))
    return max(float(np.linalg.norm(q)) - m.geom_rbound[g], 0.0)


def _unheld_hinges(g: G.Graph, m: mujoco.MjModel, d: mujoco.MjData, rigid, groups: dict[str, list[int]]) -> list[str]:
    """A hinge needs something to turn on: its axis must pass through what carries it, or a bearing there, not merely
    beside it (D-113).  A body on a free joint carries itself."""
    by_name = {b.name: b for b in g.bodies}
    out = []
    for b in g.bodies:
        if b.parent not in by_name:
            continue
        on = rigid(by_name[b.parent])
        for j in b.joints:
            if j.type != "hinge" or on not in groups:
                continue
            at = d.xanchor[m.joint(j.name).id]
            far = min(_distance_to_geom(m, d, x, at) for x in groups[on])
            if far > PIN:
                out.append(f"joint {j.name!r} turns {far * 1000:.0f} mm from anything on {on!r}: no axle or bearing holds it")
    return out


def _driven(g: G.Graph) -> list[str]:
    names = [a.joint for a in g.actuators if not isinstance(a, G.Band)] + [lt.joint for lt in g.latches]
    return list(dict.fromkeys(names))


def _close(m: mujoco.MjModel, d: mujoco.MjData, free: list[int]) -> None:
    """Move the joints of closed loops until the loops close, by least squares from where they are."""
    from scipy.optimize import least_squares

    adrs = [m.jnt_qposadr[j] for j in free]

    def res(x: np.ndarray) -> np.ndarray:
        d.qpos[adrs] = x
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_makeConstraint(m, d)
        rows = d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        return d.efc_pos[: d.nefc][rows].copy()

    if free and res(d.qpos[adrs].copy()).size:
        d.qpos[adrs] = least_squares(res, d.qpos[adrs].copy(), max_nfev=40).x


def interference(g: G.Graph, m: mujoco.MjModel, keep: Sequence[str] = (), unit: tuple[float, str] = (1.0, "m"),
                 tol: float = 0.0025) -> list[str]:
    """Parts may not pass through each other: each driven joint swept over its range, with closed loops solved at each
    pose.  Parts of bodies joined to each other may overlap as they are fastened, not further as they move.  ``keep``: the
    world's own constraints that stay on; ``unit`` is what its refusals measure in, ``tol`` how far parts may overlap (m)."""
    d = mujoco.MjData(m)
    names = {b.name for b in g.bodies}
    ids = [x for x in range(m.ngeom) if m.geom_contype[x] == 2 and m.body(m.geom_bodyid[x]).name in names]
    excl = {frozenset(e) for e in g.excludes}
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if ba != bb and frozenset((m.body(ba).name, m.body(bb).name)) not in excl:
                pairs.append((a, b, m.body_parentid[ba] == bb or m.body_parentid[bb] == ba))
    keep = [i for i in range(m.neq) if m.eq(i).name in {lt.name for lt in g.latches} or m.eq(i).name in keep]
    looped = [m.joint(j.name).id for lp in g.loops for b in g.bodies if b.name in (lp.a, lp.b) for j in b.joints
              if j.type != "free"]
    over: dict[tuple[int, int], list[float]] = {}
    worst_pose: dict[tuple[int, int], str] = {}
    for jn in _driven(g) or [""]:
        jid = m.joint(jn).id if jn else -1
        lo, hi = (m.jnt_range[jid] if jn and m.jnt_limited[jid] else (0.0, 0.0))
        for q in np.linspace(lo, hi, SWEEP if hi > lo else 1):
            mujoco.mj_resetData(m, d)
            d.eq_active[keep] = 0
            if jn:
                d.qpos[m.jnt_qposadr[jid]] = q
            _close(m, d, sorted({j for j in looped if j != jid}))
            mujoco.mj_kinematics(m, d)
            for a, b, _ in pairs:
                r = m.geom_rbound[a] + m.geom_rbound[b]
                gap = 0.0 if np.sum((d.geom_xpos[a] - d.geom_xpos[b]) ** 2) > r * r else \
                    min(0.0, mujoco.mj_geomDistance(m, d, a, b, 0.01, None))
                over.setdefault((a, b), []).append(-gap)
                if -gap >= max(over[(a, b)]):
                    worst_pose[(a, b)] = f"{jn or 'at rest'} at {q:.2f}" if jn else "at rest"
    rows = []
    for a, b, joined in pairs:
        ov = over.get((a, b), [0.0])
        bad = max(ov) - (min(ov) if joined else 0.0)
        if bad > tol:
            rows.append((bad, f"{m.geom(a).name} and {m.geom(b).name} pass {bad / unit[0]:.2f} {unit[1]} into each other "
                              f"({worst_pose[(a, b)]})"))
    return [t for _, t in sorted(rows, reverse=True)[:6]]
