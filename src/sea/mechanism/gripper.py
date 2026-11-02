"""The gripper's mechanism laws (D-059): what a jaw part must satisfy before any trial.

The simulator builds each jaw as its parts (``sim/jaws.py``, docs/harness.md invariant 11).  The laws read
that build on the real meshes, through the jaw's range and the wrist's roll range:

- **carried:** every moving body bears on the body that carries it: at most ``CARRY_MM`` from it at every
  jaw angle (a pin's or a bushing's clearance);
- **no interference:** no two parts share material, except at a bearing seat: the servo's spline and rear
  pin, inside ``SEAT_MM`` of the hinge axis, where the hinge body meets the servo;
- **transmission:** a four-bar's links stay at least ``MIN_TRANSMISSION_DEG`` from toggle.

Each violation names the law, the parts and the numbers, for System 2's "refused before trial" channel.
Tooth bending is not a law: no measured allowable stress exists for printed PLA teeth (D-059).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

HAND = ("gripper", "moving_jaw_so101_v1", "jaw_coupler", "jaw_follower", "jaw_carriage")
OBSTACLES = ("wrist",)  # what the jaw mechanism must clear beyond the gripper (a pinion can reach it, D-058)
SAMPLES = 9  # jaw positions, ends included
ROLLS = 13  # wrist roll positions, ends included
SERVO = "sts3215_03a_no_horn_v1"
CARRY_MM = 0.5
TOUCH_MM = 0.1  # a shared region thinner than this is two faces in contact, not material in both parts
SEAT_MM = 10.0  # the stock jaw's back plate bears on the servo's rear boss within 10 mm of the axis (D-056)
MIN_TRANSMISSION_DEG = 30.0
_CACHE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------- geometry
def _near(a, pa: np.ndarray, b, pb: np.ndarray) -> bool:
    """Whether two parts' bounding spheres meet: a cheap test before the mesh one."""
    ca = pa[:3, :3] @ a.mesh.bounding_sphere.primitive.center + pa[:3, 3]
    cb = pb[:3, :3] @ b.mesh.bounding_sphere.primitive.center + pb[:3, 3]
    return float(np.linalg.norm(ca - cb)) <= a.mesh.bounding_sphere.primitive.radius + b.mesh.bounding_sphere.primitive.radius


def _hand(model):
    import mujoco
    import trimesh

    from sea.mechanism.geometry import Solid

    out = []
    for i in range(model.ngeom):
        body = model.body(model.geom_bodyid[i]).name
        if model.geom_group[i] != 2 or body not in HAND + OBSTACLES:
            continue
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            k = model.geom_dataid[i]
            v0, nv, f0, nf = model.mesh_vertadr[k], model.mesh_vertnum[k], model.mesh_faceadr[k], model.mesh_facenum[k]
            mesh = trimesh.Trimesh(model.mesh_vert[v0:v0 + nv].copy(), model.mesh_face[f0:f0 + nf].copy(), process=False)
            name = model.mesh(k).name
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * model.geom_size[i])
            name = model.geom(i).name
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=model.geom_size[i][0], height=2 * model.geom_size[i][1], sections=48)
            name = model.geom(i).name
        else:
            continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[i])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rot.reshape(3, 3), model.geom_pos[i]
        mesh.apply_transform(pose)
        mesh.apply_scale(1000.0)
        out.append(Solid(f"{body}:{name}" if body in OBSTACLES else name, body, mesh))  # the wrist has a servo too
    return out


def geometry(body) -> dict[str, Any]:
    """The jaw mechanism's parts posed through the jaw's range (and the wrist's roll, for the wrist): where
    parts on different bodies share material, and how far each moving body is from the body that carries it.
    Cached per body (its content key): a sweep takes about 20 s."""
    key = body.key()
    if key not in _CACHE:
        _CACHE[key] = _geometry(body)
    return _CACHE[key]


def _geometry(body) -> dict[str, Any]:
    import mujoco

    from sea.contracts import ARM_JOINTS
    from sea.control.executor import HOME_Q
    from sea.mechanism.geometry import clearance, extent, from_axis, overlap
    from sea.sim.build_model import build_model
    from sea.worlds.gripper import to_design

    sim = build_model(to_design(body))
    m, d = sim.model, sim.data
    solids = _hand(m)
    jid = m.joint("gripper").id
    lo, hi = m.jnt_range[jid]
    bodies = {s.body for s in solids}
    clashes: dict[str, dict] = {}
    names = {m.body(i).name for i in range(m.nbody)}
    support: dict[str, list[float]] = {b: [] for b in HAND[1:] if b in names}

    def pose(b: str) -> np.ndarray:  # in the gripper's frame, mm
        rot, at = d.xmat[m.body("gripper").id].reshape(3, 3), d.xpos[m.body("gripper").id]
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rot.T @ d.xmat[m.body(b).id].reshape(3, 3), 1000 * rot.T @ (d.xpos[m.body(b).id] - at)
        return t

    roll = ARM_JOINTS.index("wrist_roll")
    wrists = []  # the wrist seen from the gripper over the wrist's roll range: a jaw part must clear it at every roll
    for r in np.linspace(*m.jnt_range[m.joint("wrist_roll").id], ROLLS):
        qa = HOME_Q.copy()
        qa[roll] = r
        sim.set_arm_q(qa)
        wrists.append((round(float(r), 2), pose("wrist")))
    for q in np.linspace(lo, hi, SAMPLES):
        qa = HOME_Q.copy()
        qa[-1] = q
        sim.set_arm_q(qa)  # the jaw mechanism's linked joints follow (Sim.linked)
        poses = {b: pose(b) for b in (bodies | set(support)) - set(OBSTACLES)}
        rot = d.xmat[m.body("gripper").id].reshape(3, 3)
        axis_o = rot.T @ (1000 * (d.xanchor[jid] - d.xpos[m.body("gripper").id]))
        axis_u = rot.T @ d.xaxis[jid]
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                if a.body == b.body or {a.body, b.body} == {"wrist", "gripper"}:  # the wrist carries the gripper
                    continue
                if a.body in OBSTACLES or b.body in OBSTACLES:
                    tries = [(rl, {**poses, "wrist": w}) for rl, w in wrists]
                else:
                    tries = [(None, poses)]
                for rl, ps in tries:
                    if not _near(a, ps[a.body], b, ps[b.body]):
                        continue
                    pts = overlap(a, ps[a.body], b, ps[b.body])
                    if not len(pts):
                        continue
                    r = from_axis(pts, axis_o, axis_u)
                    c = clashes.setdefault(f"{a.name} / {b.name}", {"bodies": [a.body, b.body], "at_q": [], "crossings": [],
                                                                    "region_mm": [0.0, 0.0, 0.0], "from_axis_mm": [1e9, 0.0]})
                    if rl is not None:
                        c.setdefault("at_roll", []).append(rl)
                    c["at_q"].append(round(float(q), 3))
                    c["crossings"].append(int(len(pts)))
                    c["region_mm"] = [max(x, y) for x, y in zip(c["region_mm"], extent(pts), strict=True)]
                    c["from_axis_mm"] = [round(min(c["from_axis_mm"][0], float(r.min())), 1),
                                         round(max(c["from_axis_mm"][1], float(r.max())), 1)]
        for child in support:
            parent = m.body(m.body_parentid[m.body(child).id]).name
            mine = [s for s in solids if s.body == child]
            theirs = [s for s in solids if s.body == parent]
            gap = min((clearance(a, poses[child], b, poses[parent]) for a in mine for b in theirs), default=float("nan"))
            support[child].append(round(float(gap), 1))
    carried = {}
    for child, gaps in support.items():
        parent = m.body(m.body_parentid[m.body(child).id]).name
        no_material = not any(s.body == child for s in solids)
        own = m.jnt_type[m.body_jntadr[m.body(child).id]]
        carried[child] = {"parent": parent, "joint": "slide" if own == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                          "material": not no_material,
                          "worst_gap_mm": None if no_material else max(gaps), "gap_by_q_mm": None if no_material else gaps}
    mass = {b: round(1000 * float(m.body_mass[m.body(b).id]), 1) for b in [*support, "gripper"]}
    return {"jaw_range_servo_rad": [round(float(lo), 3), round(float(hi), 3)], "clashes": clashes, "carried": carried,
            "sim_mass_g": mass}


LABELS = {"sts3215_horn": "the servo horn", "wrist_roll_pitch_so101_v2": "the wrist",
          "wrist_roll_follower_so101_v1": "the gripper frame", "moving_jaw_so101_v1": "the moving jaw",
          "jaw_carriage": "the carriage", "jaw_coupler": "the four-bar's coupler", "jaw_follower": "the follower links",
          "gripper": "the gripper"}


def _label(name: str, body: str) -> str:
    name = name.split(":")[-1]
    if name == SERVO:  # the wrist carries an STS3215 of its own
        return "the gripper servo" if body == "gripper" else f"the {body} servo"
    if name in LABELS:
        return LABELS[name]
    part = re.sub(r"^jaw_", "", name)
    part = re.sub(r"^(carriage|coupler|driver_horn|driver_rear|follower_horn|follower_rear)_?.*", r"\1", part)
    return "the " + part.rstrip("0123456789").replace("_", " ").strip()


def _seat(c: dict[str, Any], a: str, b: str) -> bool:
    """A bearing seat: the hinge body on the servo, close around the hinge axis."""
    return set(c["bodies"]) == {"gripper", "moving_jaw_so101_v1"} and SERVO in (a, b) and c["from_axis_mm"][1] <= SEAT_MM


def laws(body) -> list[str]:
    """The mechanism laws this body's jaw breaks, each with its parts and numbers (empty: none)."""
    from sea.sim import jaws
    from sea.worlds.gripper import to_design

    geo = geometry(body)
    out = []
    shared: dict[tuple[str, str], dict[str, Any]] = {}  # per pair of named parts: the worst region, where
    for pair, c in geo["clashes"].items():
        a, b = pair.split(" / ")
        if _seat(c, a, b) or min(c["region_mm"]) < TOUCH_MM:
            continue
        g = shared.setdefault((_label(a, c["bodies"][0]), _label(b, c["bodies"][1])),
                              {"region": [0.0, 0.0, 0.0], "q": set(), "roll": set()})
        g["region"] = max(g["region"], c["region_mm"], key=lambda r: float(np.prod(r)))
        g["q"] |= set(c["at_q"])
        g["roll"] |= set(c.get("at_roll", []))
    for (la, lb), g in shared.items():
        qs = sorted(g["q"])
        where = f"jaw at {qs[0]:.2f}..{qs[-1]:.2f} rad" if len(qs) > 1 else f"jaw at {qs[0]:.2f} rad"
        if g["roll"]:
            where += f", wrist rolled {min(g['roll']):+.2f}..{max(g['roll']):+.2f} rad"
        out.append(f"mechanism law (no interference): {la} and {lb} share material, a region up to "
                   f"{' x '.join(f'{x:.1f}' for x in g['region'])} mm ({where}); parts may share material only at "
                   f"a bearing seat, within {SEAT_MM:g} mm of the hinge axis")
    for child, c in geo["carried"].items():
        if not c["material"] or c["worst_gap_mm"] <= CARRY_MM:
            continue
        lo, hi = geo["jaw_range_servo_rad"]
        q = np.linspace(lo, hi, SAMPLES)[int(np.argmax(c["gap_by_q_mm"]))]
        out.append(f"mechanism law (carried): {_label(child, child)} is {c['worst_gap_mm']:.1f} mm from "
                   f"{_label(c['parent'], c['parent'])} "
                   f"that should carry it (jaw at {q:.2f} rad); a {c['joint']} bears on its parts within {CARRY_MM:g} mm")
    f = to_design(body).fingers
    if f.jaw == "parallelogram":
        worst = min(jaws.transmission_deg(q) for q in np.linspace(*geo["jaw_range_servo_rad"], 50))
        if worst < MIN_TRANSMISSION_DEG:
            out.append(f"mechanism law (transmission): the four-bar's links come within {worst:.0f} deg of toggle; "
                       f"they must stay {MIN_TRANSMISSION_DEG:g} deg away")
    return out
