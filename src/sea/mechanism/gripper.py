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

import math
import re
from typing import Any

import numpy as np

HAND = ("gripper", "moving_jaw_so101_v1", "jaw_coupler", "jaw_follower", "jaw_carriage")
# What the jaw mechanism must clear beyond the gripper (D-058): the wrist, which carries the gripper, so the
# simulator never collides the two.  The lower arm is no parent of the jaw's bodies: the simulator's contacts
# stop a jaw that meets it, in the poses the arm takes (D-076)
OBSTACLES = ("wrist",)
SAMPLES = 9  # jaw positions, ends included, before any step is split
MAX_SPLITS = 4  # a step halves at most this often: 1/16 of a step at the finest
ROLLS = 13  # wrist roll positions, ends included
SERVO = "sts3215_03a_no_horn_v1"
CARRY_MM = 0.5
TOUCH_MM = 0.1  # a shared region thinner than this is two faces in contact, not material in both parts
SEAT_MM = 10.0  # the stock jaw's back plate bears on the servo's rear boss within 10 mm of the axis (D-056)
MIN_TRANSMISSION_DEG = 30.0
FRAME = "wrist_roll_follower_so101_v1"  # the gripper's frame: every fixed part must reach it (D-076)
STOCK_PARTS = {SERVO, FRAME}  # the SO-101's own: the servo sits in the frame's pocket
FIT_MM = 1.0  # a printed part screwed to another: the meshes may share this thin a layer at the mating face
PAD_SEAT_MM = 2.0  # a pad's back and shoes on the finger they wrap (TPU stretched over it)
# the fingers' faces: the moving one meeting the fixed one as the jaw closes is the jaw's stop (D-061)
FIXED_FINGER = ("wrist_roll_follower_so101_v1", "jaw_fixed_pad")
MOVING_FINGER = ("moving_jaw_so101_v1", "jaw_blade", "jaw_moving_pad")
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
    from sea.mechanism.geometry import clearance, extent, from_axis, grounded, overlap
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
    support: dict[str, dict[float, float]] = {b: {} for b in HAND[1:] if b in names}

    def pose(b: str) -> np.ndarray:  # in the gripper's frame, mm
        rot, at = d.xmat[m.body("gripper").id].reshape(3, 3), d.xpos[m.body("gripper").id]
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rot.T @ d.xmat[m.body(b).id].reshape(3, 3), 1000 * rot.T @ (d.xpos[m.body(b).id] - at)
        return t

    # the wrist seen from the gripper over the wrist's roll range: a jaw part must clear it at every roll
    roll = ARM_JOINTS.index("wrist_roll")
    seen: dict[str, list[tuple[str, np.ndarray]]] = {"wrist": []}
    for r in np.linspace(*m.jnt_range[m.joint("wrist_roll").id], ROLLS):
        qa = HOME_Q.copy()
        qa[roll] = r
        sim.set_arm_q(qa)
        seen["wrist"].append((f"wrist rolled {r:+.2f} rad", pose("wrist")))

    def added(s) -> bool:  # a part the jaw mechanism brings, or one that moves: what an obstacle must clear
        return s.body != "gripper" or s.name.startswith("jaw_")

    tracks = []  # (a, b, obstacle body, obstacle pose label, obstacle pose): each checked through the jaw's range
    for i, a in enumerate(solids):
        for b in solids[i + 1:]:
            if a.body == b.body or {a.body, b.body} == {"wrist", "gripper"}:
                continue  # same body (the fixed parts' own law, below); the arm's links carry the gripper
            ob = a.body if a.body in OBSTACLES else b.body if b.body in OBSTACLES else None
            if ob is None:
                tracks.append((a, b, None, "", None))
            elif added(b if ob == a.body else a):
                tracks += [(a, b, ob, lab, ps) for lab, ps in seen[ob]]

    cache: dict[float, dict[str, Any]] = {}

    def at(q: float) -> dict[str, Any]:
        """Every track's state at one jaw position: the overlap's points, or the clearance if the parts are near."""
        if q in cache:
            return cache[q]
        qa = HOME_Q.copy()
        qa[-1] = q
        sim.set_arm_q(qa)  # the jaw mechanism's linked joints follow (Sim.linked)
        poses = {b: pose(b) for b in (bodies | set(support)) - set(OBSTACLES)}
        rot = d.xmat[m.body("gripper").id].reshape(3, 3)
        axis = (rot.T @ (1000 * (d.xanchor[jid] - d.xpos[m.body("gripper").id])), rot.T @ d.xaxis[jid])
        out: dict[str, Any] = {"poses": poses, "axis": axis, "tracks": []}
        for a, b, ob, _, ops in tracks:
            ps = {**poses, ob: ops} if ob else poses
            if not _near(a, ps[a.body], b, ps[b.body]):
                out["tracks"].append((None, math.inf))
                continue
            pts = overlap(a, ps[a.body], b, ps[b.body])
            out["tracks"].append((pts, 0.0) if len(pts) else (None, clearance(a, ps[a.body], b, ps[b.body])))
        for child in support:
            parent = m.body(m.body_parentid[m.body(child).id]).name
            mine = [s for s in solids if s.body == child]
            theirs = [s for s in solids if s.body == parent]
            support[child][q] = round(float(min((clearance(a, poses[child], b, poses[parent]) for a in mine
                                                 for b in theirs), default=float("nan"))), 1)
        cache[q] = out
        return out

    def moved(s, p0: np.ndarray, p1: np.ndarray) -> float:
        """How far any of a part's points moves between two poses (mm)."""
        v = np.asarray(s.mesh.vertices)[:: max(1, len(s.mesh.vertices) // 200)]
        return float(np.max(np.linalg.norm((v @ p0[:3, :3].T + p0[:3, 3]) - (v @ p1[:3, :3].T + p1[:3, 3]), axis=1)))

    # the jaw's range in SAMPLES steps, each split while some pair is near enough that it could meet between two
    # samples: their clearances at both ends no more than the distance a point moves between them (D-076)
    qs = [float(q) for q in np.linspace(lo, hi, SAMPLES)]
    todo = list(zip(qs, qs[1:], strict=False))
    depth = {iv: 0 for iv in todo}
    while todo:
        q0, q1 = todo.pop()
        s0, s1 = at(q0), at(q1)
        split = False
        for k, (a, b, ob, _, _) in enumerate(tracks):
            (p0, c0), (p1, c1) = s0["tracks"][k], s1["tracks"][k]
            if p0 is not None or p1 is not None or c0 + c1 == math.inf or max(c0, c1) <= TOUCH_MM:
                continue  # crossing, far apart, or in contact at both ends (a seat): a finer step shows nothing new
            rel = max(moved(a, s0["poses"][a.body], s1["poses"][a.body]) if a.body != ob else 0.0,
                      moved(b, s0["poses"][b.body], s1["poses"][b.body]) if b.body != ob else 0.0)
            if c0 + c1 <= rel:
                split = True
                break
        if split and depth[(q0, q1)] < MAX_SPLITS:
            mid = (q0 + q1) / 2
            for iv in ((q0, mid), (mid, q1)):
                depth[iv] = depth[(q0, q1)] + 1
                todo.append(iv)
    samples = sorted(cache)
    for q in samples:
        st = cache[q]
        for k, (a, b, ob, lab, _) in enumerate(tracks):
            pts, _ = st["tracks"][k]
            if pts is None:
                continue
            r = from_axis(pts, *st["axis"])
            c = clashes.setdefault(f"{a.name} / {b.name}", {"bodies": [a.body, b.body], "at_q": [], "crossings": [],
                                                            "region_mm": [0.0, 0.0, 0.0], "from_axis_mm": [1e9, 0.0]})
            if ob:
                c.setdefault("at_pose", []).append(lab)
            c["at_q"].append(round(q, 3))
            c["crossings"].append(int(len(pts)))
            c["region_mm"] = [max(x, y) for x, y in zip(c["region_mm"], extent(pts), strict=True)]
            c["from_axis_mm"] = [round(min(c["from_axis_mm"][0], float(r.min())), 1),
                                 round(max(c["from_axis_mm"][1], float(r.max())), 1)]
    carried = {}
    for child, gaps in support.items():
        parent = m.body(m.body_parentid[m.body(child).id]).name
        no_material = not any(s.body == child for s in solids)
        own = m.jnt_type[m.body_jntadr[m.body(child).id]]
        by_q = [gaps[q] for q in samples]
        carried[child] = {"parent": parent, "joint": "slide" if own == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                          "material": not no_material, "worst_gap_mm": None if no_material else max(by_q),
                          "worst_q": None if no_material else samples[int(np.argmax(by_q))],
                          "gap_by_q_mm": None if no_material else by_q}
    # the gripper's own fixed parts: what they share, and whether each reaches the frame through parts it touches
    eye = np.eye(4)
    fixed = [s for s in solids if s.body == "gripper"]
    fastened = {}
    for i, a in enumerate(fixed):
        for b in fixed[i + 1:]:
            if _assembly(a.name) == _assembly(b.name) or {a.name, b.name} <= STOCK_PARTS:
                continue
            pts = overlap(a, eye, b, eye)
            if len(pts):
                fastened[f"{a.name} / {b.name}"] = extent(pts)
    poses0 = {"gripper": eye}
    loose = grounded(fixed, poses0, FRAME, touch=CARRY_MM)
    mass = {b: round(1000 * float(m.body_mass[m.body(b).id]), 1) for b in [*support, "gripper"]}
    return {"jaw_range_servo_rad": [round(float(lo), 3), round(float(hi), 3)], "samples": [round(q, 3) for q in samples],
            "clashes": clashes, "carried": carried, "fastened": fastened, "loose": loose, "sim_mass_g": mass}


def _assembly(name: str) -> str:
    """The piece a fixed part belongs to: a pad and its shoes and screws are one, a clamp and its bracket one."""
    for prefix, piece in (("jaw_fixed_pad", "pad"), ("jaw_clamp", "clamp"), ("jaw_bracket", "bracket")):
        if name.startswith(prefix):
            return piece
    return name


def _allowed_mm(a: str, b: str) -> float:
    """How thick a region two fixed parts may share: a rod in its clamp's bore, any; a pad seated on the finger it
    wraps, ``PAD_SEAT_MM``; a printed part screwed to another, ``FIT_MM``; otherwise faces in contact."""
    pa, pb = {_assembly(a), _assembly(b)}, {a, b}
    if "clamp" in pa and any(n.startswith("jaw_rod") for n in pb):
        return math.inf
    if "pad" in pa and FRAME in pb:
        return PAD_SEAT_MM
    if pa & {"clamp", "bracket"} and pb & STOCK_PARTS:
        return FIT_MM
    return TOUCH_MM


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


def _stop(c: dict[str, Any], a: str, b: str, qs: list[float]) -> bool:
    """The jaw closing on its other finger: a moving finger part meets a fixed one only over the closed end of
    the range (the first jaw positions, without a gap).  The jaw stops there; the simulator holds the fingers
    apart with contact pairs (D-060)."""
    fingers = {a.startswith(FIXED_FINGER), b.startswith(FIXED_FINGER)}, {a.startswith(MOVING_FINGER),
                                                                           b.startswith(MOVING_FINGER)}
    if fingers != ({True, False}, {True, False}) or a.startswith(FIXED_FINGER) == b.startswith(FIXED_FINGER):
        return False
    at = sorted(set(c["at_q"]))
    return at == qs[:len(at)]


def _seat(c: dict[str, Any], a: str, b: str) -> bool:
    """A bearing seat: the hinge body on the servo, close around the hinge axis."""
    return set(c["bodies"]) == {"gripper", "moving_jaw_so101_v1"} and SERVO in (a, b) and c["from_axis_mm"][1] <= SEAT_MM


def laws(body) -> list[str]:
    """The mechanism laws this body's jaw breaks, each with its parts and numbers (empty: none)."""
    from sea.sim import jaws
    from sea.worlds.gripper import to_design

    geo = geometry(body)
    qs = geo["samples"]
    out = []
    shared: dict[tuple[str, str], dict[str, Any]] = {}  # per pair of named parts: the worst region, where
    for pair, c in geo["clashes"].items():
        a, b = pair.split(" / ")
        if _seat(c, a, b) or _stop(c, a, b, qs) or min(c["region_mm"]) < TOUCH_MM:
            continue
        g = shared.setdefault((_label(a, c["bodies"][0]), _label(b, c["bodies"][1])),
                              {"region": [0.0, 0.0, 0.0], "q": set(), "pose": []})
        g["region"] = max(g["region"], c["region_mm"], key=lambda r: float(np.prod(r)))
        g["q"] |= set(c["at_q"])
        g["pose"] += c.get("at_pose", [])
    for (la, lb), g in shared.items():
        at_q = sorted(g["q"])
        where = f"jaw at {at_q[0]:.2f}..{at_q[-1]:.2f} rad" if len(at_q) > 1 else f"jaw at {at_q[0]:.2f} rad"
        if g["pose"]:
            where += f", {g['pose'][0]}" + (f" and {len(set(g['pose'])) - 1} other arm poses" if len(set(g["pose"])) > 1 else "")
        out.append(f"mechanism law (no interference): {la} and {lb} share material, a region up to "
                   f"{' x '.join(f'{x:.1f}' for x in g['region'])} mm ({where}); parts may share material only at "
                   f"a bearing seat, within {SEAT_MM:g} mm of the hinge axis")
    for child, c in geo["carried"].items():
        if not c["material"] or c["worst_gap_mm"] <= CARRY_MM:
            continue
        q = c["worst_q"]
        out.append(f"mechanism law (carried): {_label(child, child)} is {c['worst_gap_mm']:.1f} mm from "
                   f"{_label(c['parent'], c['parent'])} "
                   f"that should carry it (jaw at {q:.2f} rad); a {c['joint']} bears on its parts within {CARRY_MM:g} mm")
    for pair, region in geo["fastened"].items():  # the gripper's fixed parts, on one another
        a, b = pair.split(" / ")
        allowed = _allowed_mm(a, b)
        if min(region) > allowed:
            out.append(f"mechanism law (no interference): {_label(a, 'gripper')} and {_label(b, 'gripper')} share "
                       f"material, a region up to {' x '.join(f'{x:.1f}' for x in region)} mm, both fixed to the "
                       f"gripper; two fixed parts may share at most {allowed:g} mm where one is fastened to the other")
    for part in dict.fromkeys(_label(p, "gripper") for p in geo["loose"]):
        out.append(f"mechanism law (mounted): {part} touches nothing that reaches the gripper "
                   f"frame (within {CARRY_MM:g} mm); a fixed part must be fastened to the frame or to a part that is")
    f = to_design(body).fingers
    if f.jaw == "parallelogram":
        worst = min(jaws.transmission_deg(q) for q in np.linspace(*geo["jaw_range_servo_rad"], 50))
        if worst < MIN_TRANSMISSION_DEG:
            out.append(f"mechanism law (transmission): the four-bar's links come within {worst:.0f} deg of toggle; "
                       f"they must stay {MIN_TRANSMISSION_DEG:g} deg away")
    return out
