"""The moving jaw's mechanisms, built from their parts (D-057; docs/harness.md, real mechanisms).

A jaw type exists in the simulator only as the parts that make it work.  The stock hinge is the SO-101's
own: the finger is screwed to the servo horn (D-056).  The other jaws carry the stock finger's blade, the
part of the stock moving jaw below the servo (``BLADE_CUT_Z``), on a mechanism of printed and bought parts.

``four_bar``: two driver plates, one on the servo horn and one on the servo's rear hub, and two follower
plates on a pin held by a bracket on the servo's end.  The finger is the coupler, pinned to both pairs; a
connect constraint closes the loop at the follower's lower pin, so the finger keeps its orientation
because the links are parallel, not because a constraint holds its angle.  The follower pin sits
``FOLLOWER_MM`` from the hinge at 45 degrees up from the closing axis, so the transmission angle stays
above 35 degrees over the jaw's range (it reaches toggle only at 135 degrees).

Every part has mass from its volume (PLA ``PLA_KG_M3``); each solid is both drawn and in contact.  Joints
the mechanism moves with the servo are returned as ``(joint, ratio)`` for ``Sim.set_arm_q``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mujoco
import numpy as np

MM = 1e-3
PLA_KG_M3 = 1240.0
HINGE = np.array([20.2, 18.8, -23.4]) * MM  # the gripper joint's axis point, gripper frame
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # moving jaw axes in the gripper frame
JAW_MESH = "moving_jaw_so101_v1"
JAW_MESH_Z = 0.0189  # the jaw mesh's offset along the hinge axis in its body (SO-101 MJCF)
BLADE_CUT_Z = -55.0 * MM  # gripper frame, jaw closed: below it the stock finger is a blade under 13 mm thick
# Link plates, in the jaw's frame (z along the hinge, gripper y = HINGE y - z): in the stock finger's own
# wall planes.  The horn-side plates lie on the horn's face (z 37.0 mm), the rear plates on the rear hub
# (the stock wall's outer face, z -5.1 mm); the servo case lies between them.
PLATE_T = 4.0 * MM
HORN_SIDE = (37.0 * MM, 37.0 * MM + PLATE_T)
REAR_SIDE = (-5.1 * MM, -5.1 * MM + PLATE_T)
MID_Z = (REAR_SIDE[1] + HORN_SIDE[0]) / 2
PLATE_HALF_W = 4.5 * MM
BOSS_R = 6.0 * MM  # round ends of plates and hubs around each pin
PIN_HEAD = (3.5 * MM, 0.75 * MM)  # each pin's head on the plates' outer faces: radius, half thickness
FOLLOWER_MM = 22.0
FOLLOWER_DIR = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4)])  # gripper frame (x, z)
JOINT_ARMATURE, JOINT_DAMPING = 1e-4, 0.01  # pinned joints: printed bores on steel pins


def follower_offset() -> np.ndarray:
    """The follower pin's offset from the hinge, in the jaw's frame at q = 0."""
    return R0.T @ (FOLLOWER_MM * MM * FOLLOWER_DIR)


def transmission_deg(q: float) -> float:
    """Angle between the links and the line joining their fixed pins; 0 is toggle."""
    link = math.degrees(q) - 90.0  # the links point down at q = 0 and turn toward +x
    mu = abs(link - math.degrees(math.atan2(FOLLOWER_DIR[2], FOLLOWER_DIR[0]))) % 180.0
    return min(mu, 180.0 - mu)


# ---------------------------------------------------------------- the blade
def _clip_convex(verts: np.ndarray, y_cut: float) -> np.ndarray | None:
    """A convex part cut by the plane y = ``y_cut`` (jaw frame), keeping y < y_cut: still convex."""
    from scipy.spatial import ConvexHull

    keep = verts[verts[:, 1] < y_cut]
    if len(keep) < 4:
        return None
    hull = ConvexHull(verts)
    edges = {tuple(sorted((a, b))) for s in hull.simplices for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2]))}
    cut = []
    for a, b in edges:
        ya, yb = verts[a, 1] - y_cut, verts[b, 1] - y_cut
        if ya * yb < 0:
            cut.append(verts[a] + (verts[b] - verts[a]) * ya / (ya - yb))
    pts = np.vstack([keep, *cut]) if cut else keep
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:  # a sliver: nothing worth a contact
        return None


@lru_cache(maxsize=1)
def blade_parts() -> tuple[np.ndarray, ...]:
    """The stock finger's convex contact parts below the cut, in the jaw's frame."""
    from sea.sim.build_model import _jaw_parts

    y_cut = BLADE_CUT_Z - HINGE[2]
    out = []
    for v in _jaw_parts(JAW_MESH):
        c = _clip_convex(np.asarray(v) + [0.0, 0.0, JAW_MESH_Z], y_cut)
        if c is not None and np.ptp(c, axis=0).min() > 0.3 * MM:
            out.append(c)
    return tuple(out)


def _below(verts: np.ndarray, faces: np.ndarray, y_cut: float) -> tuple[np.ndarray, np.ndarray]:
    """The part of a triangle mesh with y < ``y_cut``: each crossing triangle clipped to the plane and fanned."""
    out_v: list[np.ndarray] = []
    out_f: list[list[int]] = []
    for tri in verts[faces]:
        poly = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            da, db = a[1] - y_cut, b[1] - y_cut
            if da < 0:
                poly.append(a)
            if da * db < 0:
                poly.append(a + (b - a) * da / (da - db))
        if len(poly) < 3:
            continue
        k = len(out_v)
        out_v += poly
        out_f += [[k, k + i, k + i + 1] for i in range(1, len(poly) - 1)]
    return np.asarray(out_v), np.asarray(out_f)


@lru_cache(maxsize=1)
def blade_mesh() -> tuple[np.ndarray, np.ndarray]:
    """The stock finger's drawn surface below the cut (jaw frame), closed on top by a printed cap."""
    import trimesh

    from sea.control.frames import STOCK_MJCF

    m = trimesh.load(STOCK_MJCF.parent / "assets" / f"{JAW_MESH}.stl")
    y_cut = BLADE_CUT_Z - HINGE[2]
    v, f = _below(np.asarray(m.vertices) + [0.0, 0.0, JAW_MESH_Z], np.asarray(m.faces), y_cut)
    lo, hi = v.min(axis=0), v.max(axis=0)
    cap = trimesh.creation.box(extents=(hi[0] - lo[0], 2.0 * MM, hi[2] - lo[2]))
    cap.apply_translation([(lo[0] + hi[0]) / 2, y_cut - 1.0 * MM, (lo[2] + hi[2]) / 2])
    both = trimesh.util.concatenate([trimesh.Trimesh(v, f, process=True), cap])
    return np.asarray(both.vertices), np.asarray(both.faces)


def blade_mass() -> float:
    from scipy.spatial import ConvexHull

    return PLA_KG_M3 * sum(ConvexHull(p).volume for p in blade_parts())


def blade_gap(at: np.ndarray) -> float:
    """How far a point in the jaw's plane (x, y) lies from the blade's outline: 0 inside it."""
    from scipy.spatial import ConvexHull

    best = math.inf
    for part in blade_parts():
        xy = part[:, :2]
        ring = xy[ConvexHull(xy).vertices]  # counter-clockwise
        edges = list(zip(ring, np.roll(ring, -1, axis=0), strict=True))
        inside = all((b - a)[0] * (at - a)[1] - (b - a)[1] * (at - a)[0] >= 0 for a, b in edges)
        d = [np.linalg.norm(at - (a + np.clip(np.dot(at - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0) * (b - a)))
             for a, b in edges]
        best = min(best, 0.0 if inside else min(d))
    return best


def blade_tip() -> np.ndarray:
    """The middle of the blade's lowest contact part, (x, y) in the jaw's frame."""
    low = min(blade_parts(), key=lambda p: p[:, 1].min())
    return (low.min(axis=0)[:2] + low.max(axis=0)[:2]) / 2


# ---------------------------------------------------------------- parts
def _plate(body, name: str, length: float, side: tuple[float, float], rgba: list[float]) -> float:
    """A link plate from the body's origin to (0, -length, 0) in one side plane, round at both pins, with
    the lower pin's head on its outer face (the pin ties the body's two plates together through the hub)."""
    zc, hz = (side[0] + side[1]) / 2, (side[1] - side[0]) / 2
    vol = 0.0
    body.add_geom(name=f"{name}_web", type=mujoco.mjtGeom.mjGEOM_BOX, size=[PLATE_HALF_W, length / 2, hz],
                  pos=[0.0, -length / 2, zc], rgba=rgba, group=2, mass=0.0)
    vol += 2 * PLATE_HALF_W * length * 2 * hz
    for k, y in enumerate((0.0, -length)):
        body.add_geom(name=f"{name}_boss{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BOSS_R, hz],
                      pos=[0.0, y, zc], rgba=rgba, group=2, mass=0.0)
        vol += math.pi * BOSS_R**2 * 2 * hz / 2  # half of each round end lies outside the web
    out = side[1] + PIN_HEAD[1] if zc > MID_Z else side[0] - PIN_HEAD[1]
    body.add_geom(name=f"{name}_pin", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=list(PIN_HEAD),
                  pos=[0.0, -length, out], rgba=STEEL, group=2, mass=0.0)
    return vol


def _hub(body, name: str, at: np.ndarray, rgba: list[float], r: float = BOSS_R) -> float:
    """A hub on a pin between the two side planes."""
    lo, hi = REAR_SIDE[1] + 0.3 * MM, HORN_SIDE[0] - 0.3 * MM
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, (hi - lo) / 2],
                  pos=[at[0], at[1], (lo + hi) / 2], rgba=rgba, group=2, mass=0.0)
    return math.pi * r**2 * (hi - lo)


def _web(body, name: str, a: np.ndarray, b: np.ndarray) -> float:
    """A printed web joining two points of the body's mid-plane; returns its volume."""
    length = float(np.linalg.norm(b - a))
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2, 3.5 * MM, 6.0 * MM],
                  pos=[*((a + b) / 2), MID_Z], quat=_quat_z(math.atan2(*(b - a)[::-1])), rgba=PRINTED, group=2,
                  mass=0.0)
    return length * 7.0 * MM * 12.0 * MM


def _box_mass(*vols: float) -> float:
    return PLA_KG_M3 * sum(vols)


# Each jaw type's bodies beyond the servo's own hinge body
BODIES = {"revolute": (), "parallelogram": ("jaw_coupler", "jaw_follower"), "prismatic": ("jaw_carriage",)}
PRINTED = [0.93, 0.91, 0.86, 1.0]
STEEL = [0.55, 0.57, 0.6, 1.0]


# ---------------------------------------------------------------- the four-bar
def four_bar(spec: mujoco.MjSpec, link_mm: float) -> list[tuple[str, float]]:
    """Rebuild the moving jaw as a parallelogram four-bar of ``link_mm`` links (D-057)."""
    L = link_mm * MM
    jaw = spec.body(JAW_MESH)
    for g in [g for g in spec.geoms if g.parent.name == JAW_MESH and g.meshname == JAW_MESH]:
        spec.delete(g)  # the stock finger's hub is replaced by the driver plates; its blade rides the coupler
    vol = _plate(jaw, "jaw_driver_horn", L, HORN_SIDE, PRINTED) + _plate(jaw, "jaw_driver_rear", L, REAR_SIDE, PRINTED)
    jaw.mass, jaw.explicitinertial = _box_mass(vol), True
    jaw.ipos, jaw.iquat, jaw.fullinertia = [0.0, -L / 2, 0.0], [1.0, 0.0, 0.0, 0.0], [math.nan] * 6
    jaw.inertia = [jaw.mass * L**2 / 12 + 1e-7] * 2 + [1e-7]

    e = follower_offset()
    coupler = jaw.add_body(name="jaw_coupler", pos=[0.0, -L, 0.0])
    coupler.add_joint(name="jaw_coupler_q", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 0.0, 1.0],
                      armature=JOINT_ARMATURE, damping=[JOINT_DAMPING, 0.0, 0.0])
    verts, faces = blade_mesh()
    spec.add_mesh(name="jaw_blade", uservert=(verts + [0.0, L, 0.0]).ravel().tolist(), userface=faces.ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    coupler.add_geom(name="jaw_blade_drawn", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="jaw_blade", contype=0,
                     conaffinity=0, group=2, mass=0.0, rgba=PRINTED)
    for k, part in enumerate(blade_parts()):  # the finger's contact parts, named as the verifier maps them
        spec.add_mesh(name=f"jaw_blade_cvx{k}", uservert=(part + [0.0, L, 0.0]).ravel().tolist())
        coupler.add_geom(name=f"jaw_moving_{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"jaw_blade_cvx{k}",
                         condim=4, group=3, mass=0.0)
    hubs = _hub(coupler, "jaw_coupler_hub0", np.zeros(3), PRINTED) + _hub(coupler, "jaw_coupler_hub1", e, PRINTED)
    webs = _web(coupler, "jaw_coupler_web", np.zeros(2), e[:2])
    if blade_gap(np.array([0.0, -L])) > BOSS_R - 1.0 * MM:
        # links longer than the finger: the pins hang below the blade, and a strut carries it from them
        webs += _web(coupler, "jaw_coupler_strut", np.zeros(2), blade_tip() + [0.0, L])
    coupler.mass = blade_mass() + _box_mass(hubs, webs)
    coupler.explicitinertial, coupler.ipos = True, [0.016, L - 0.057, 0.0]  # the blade's middle
    coupler.inertia = [coupler.mass * 4e-4] * 3

    grip = spec.body("gripper")
    follower = grip.add_body(name="jaw_follower", pos=(HINGE + R0 @ e).tolist(), quat=list(jaw.quat))
    follower.add_joint(name="jaw_follower_q", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 0.0, 1.0],
                       armature=JOINT_ARMATURE, damping=[JOINT_DAMPING, 0.0, 0.0])
    fvol = _plate(follower, "jaw_follower_horn", L, HORN_SIDE, PRINTED) + _plate(follower, "jaw_follower_rear", L,
                                                                                  REAR_SIDE, PRINTED)
    follower.mass, follower.explicitinertial = _box_mass(fvol), True
    follower.ipos, follower.inertia = [0.0, -L / 2, 0.0], [follower.mass * L**2 / 12 + 1e-7] * 2 + [1e-7]
    # the bracket on the servo's end that holds the follower pin
    f_g = HINGE + R0 @ e
    bvol = _hub_gripper(grip, "jaw_bracket_hub", f_g)
    grip.add_geom(name="jaw_bracket", type=mujoco.mjtGeom.mjGEOM_BOX, size=[2.8 * MM, 15.0 * MM, 9.0 * MM],
                  pos=[30.4 * MM + 2.8 * MM, HINGE[1] - MID_Z, f_g[2] - 6.0 * MM], rgba=PRINTED, group=2, mass=0.0)
    _carry(grip, _box_mass(bvol, 5.6 * MM * 30.0 * MM * 18.0 * MM))

    spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name1="jaw_follower", name2="jaw_coupler",
                      objtype=mujoco.mjtObj.mjOBJ_BODY, data=[0.0, -L, 0.0] + [0.0] * 9,
                      solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])
    for a, b in (("gripper", "jaw_coupler"), ("jaw_follower", JAW_MESH), ("jaw_follower", "jaw_coupler")):
        spec.add_exclude(bodyname1=a, bodyname2=b)  # pinned to each other; clearance elsewhere is a checked law
    return [("jaw_coupler_q", -1.0), ("jaw_follower_q", 1.0)]


def _hub_gripper(body, name: str, at_g: np.ndarray) -> float:
    """The follower's fixed pin hub, on the gripper body (axis along the gripper's y); returns its volume."""
    lo, hi = HINGE[1] - (HORN_SIDE[0] - 0.3 * MM), HINGE[1] - (REAR_SIDE[1] + 0.3 * MM)  # gripper y = HINGE y - jaw z
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[5.0 * MM, (hi - lo) / 2],
                  pos=[at_g[0], (lo + hi) / 2, at_g[2]], quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
                  rgba=PRINTED, group=2, mass=0.0)
    return math.pi * (5.0 * MM) ** 2 * (hi - lo)


def _carry(body, kg: float) -> None:
    """Parts fixed to a body with its own inertial (the SO-101's gripper): their mass joins it."""
    body.mass = float(body.mass) + kg


def _quat_z(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


# ---------------------------------------------------------------- the slide
# Laid out in the gripper frame (x closes the jaws, z up).  The pinion and rack run in a plane outboard of
# the gripper's frame on the horn side (the frame reaches y = -28 mm); the rods, their clamp and the
# carriage's bushings in a second plane beyond it.  The carriage's own frame is the jaw's at q = 0, so the
# blade and the pads sit on it as they sit on the stock jaw.
MODULE = 1.0 * MM  # printed spur gear and rack, nominal: the pitch radius sets the exact module
BACKLASH = 0.6 * MM  # flank to flank in mesh, both gears together
GEAR_Y = (-34.5 * MM, -28.5 * MM)  # the pinion's and rack's face
HUB_R = 8.0 * MM  # the pinion's hub on the horn, through the frame's cut-out around it (clear to 12 mm)
ROD_Y, ROD_R, ROD_LEN = -40.0 * MM, 1.5 * MM, 100.0 * MM  # the two 3 x 100 mm steel rods of the bill of materials
ROD_DROP = (10.0 * MM, 22.0 * MM)  # the rods' heights below the rack's pitch line
CARRIAGE_X = (8.0 * MM, 28.0 * MM)  # over the blade
CARRIAGE_Y = (-45.0 * MM, GEAR_Y[1])
BORE = 1.6 * MM  # half the square bore each rod runs in: the LM3UU bushing's 3 mm bore with 0.1 mm clearance
BUSHING_KG = 0.0014  # LM3UU, each
STEEL_KG_M3 = 7850.0


def _to_jaw(p: np.ndarray) -> np.ndarray:
    """Gripper-frame points into the jaw's frame at q = 0 (rows)."""
    return (np.asarray(p) - HINGE) @ R0


def _box(body, name: str, lo, hi, rgba: list[float], jaw_frame: bool) -> float:
    """An axis-aligned box given by its gripper-frame corners; returns its volume."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if jaw_frame:
        lo, hi = np.minimum(_to_jaw(lo), _to_jaw(hi)), np.maximum(_to_jaw(lo), _to_jaw(hi))
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=((hi - lo) / 2).tolist(), pos=((lo + hi) / 2).tolist(),
                  rgba=rgba, group=2, mass=0.0)
    return float(np.prod(hi - lo))


def _prisms(outlines: list[np.ndarray], width: float) -> tuple[np.ndarray, np.ndarray]:
    """Convex outlines in a plane (x, y; counter-clockwise) extruded along z by ``width``, centred."""
    verts: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for ring in outlines:
        n, k = len(ring), len(verts)
        verts += [(x, y, -width / 2) for x, y in ring] + [(x, y, width / 2) for x, y in ring]
        faces += [[k, k + i + 1, k + i] for i in range(1, n - 1)] + [[k + n, k + n + i, k + n + i + 1] for i in range(1, n - 1)]
        for a in range(n):
            b = (a + 1) % n
            faces += [[k + a, k + b, k + n + b], [k + a, k + n + b, k + n + a]]
    return np.asarray(verts), np.asarray(faces)


def _tooth(c: float, along: np.ndarray, out: np.ndarray, root: np.ndarray, m: float) -> np.ndarray:
    """One tooth (trapezoid, 20 degree flanks) from the root line outward, centred ``c`` along it, thinned
    for printed backlash (``BACKLASH`` between flanks in mesh: straight flanks only approximate involutes)."""
    p = math.pi * m
    hb, ht = 0.32 * p - BACKLASH / 4, 0.18 * p - BACKLASH / 4  # half widths at the root and the tip
    return np.array([root + (c - hb) * along, root + (c + hb) * along, root + (c + ht) * along + 2.25 * m * out,
                     root + (c - ht) * along + 2.25 * m * out])


def module_for(r_pitch: float) -> tuple[int, float]:
    """Whole teeth for a pitch radius: the count nearest ``MODULE``, and the module that makes it exact."""
    n = max(round(2 * r_pitch / MODULE), 8)
    return n, 2 * r_pitch / n


def gear(r_pitch: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """A spur gear on z: a root disc and whole teeth, a gap facing -y (where it meets the rack at q = 0)."""
    n, m = module_for(r_pitch)
    rf = r_pitch - 1.25 * m
    a = np.linspace(0.0, 2 * math.pi, 4 * n, endpoint=False)
    disc = np.c_[rf * np.cos(a), rf * np.sin(a)]
    teeth = []
    for k in range(n):
        t = -math.pi / 2 + 2 * math.pi * (k + 0.5) / n
        out, along = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
        teeth.append(_tooth(0.0, along, out, (rf - 0.2 * m) * out, m))
    return _prisms([disc, *teeth], width)


def rack_teeth(x0: float, x1: float, x_mesh: float, z_pitch: float, width: float, m: float) -> tuple[np.ndarray, np.ndarray]:
    """A rack's teeth along x up from its root line, one centred on ``x_mesh`` (gripper x-z, extruded along y)."""
    p = math.pi * m
    root = np.array([0.0, z_pitch - 1.25 * m])
    first = x_mesh - math.floor((x_mesh - x0) / p) * p
    rings = [_tooth(c, np.array([1.0, 0.0]), np.array([0.0, 1.0]), root, m) for c in np.arange(first, x1, p)]
    v, f = _prisms(rings, width)
    return v[:, [0, 2, 1]] * [1.0, -1.0, 1.0], f[:, ::-1]  # x-z outline, extruded along y (mirrored: flip faces)


def slide(spec: mujoco.MjSpec, r_pitch: float) -> list[tuple[str, float]]:
    """Rebuild the moving jaw as a rack and pinion on two rods (D-058).  The servo's hinge turns a pinion of
    pitch radius ``r_pitch`` on its horn; the rack's carriage runs on the rods ``r_pitch`` per radian and
    carries the stock finger's blade.  A stiff joint equality is the teeth: tooth contact itself is too
    fine for a 2 ms step."""
    jaw = spec.body(JAW_MESH)
    for g in [g for g in spec.geoms if g.parent.name == JAW_MESH and g.meshname == JAW_MESH]:
        spec.delete(g)
    lo, hi = spec.joint("gripper").range
    s_lo, s_hi = lo * r_pitch, hi * r_pitch
    width = GEAR_Y[1] - GEAR_Y[0]
    # the pinion, in the jaw's frame (z along the axis: gripper y = HINGE y - z)
    hub = (HORN_SIDE[0], HINGE[1] - GEAR_Y[1])
    jaw.add_geom(name="jaw_pinion_hub", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[HUB_R, (hub[1] - hub[0]) / 2],
                 pos=[0.0, 0.0, sum(hub) / 2], rgba=PRINTED, group=2, mass=0.0)
    gv, gf = gear(r_pitch, width)
    spec.add_mesh(name="jaw_pinion", uservert=gv.ravel().tolist(), userface=gf.ravel().tolist())
    jaw.add_geom(name="jaw_pinion_gear", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="jaw_pinion",
                 pos=[0.0, 0.0, HINGE[1] - sum(GEAR_Y) / 2], rgba=PRINTED, group=2, mass=0.0)
    jaw.mass, jaw.explicitinertial = _box_mass(math.pi * (HUB_R**2 * (hub[1] - hub[0]) + r_pitch**2 * width)), True
    jaw.ipos, jaw.iquat, jaw.fullinertia = [0.0, 0.0, HINGE[1] - sum(GEAR_Y) / 2], [1.0, 0.0, 0.0, 0.0], [math.nan] * 6
    jaw.inertia = [jaw.mass * r_pitch**2 / 4 + 1e-7] * 2 + [jaw.mass * r_pitch**2 / 2 + 1e-7]

    # the carriage: its frame is the jaw's at q = 0, its slide the gripper's x (the jaw's x)
    grip = spec.body("gripper")
    car = grip.add_body(name="jaw_carriage", pos=HINGE.tolist(), quat=list(jaw.quat))
    car.add_joint(name="jaw_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1.0, 0.0, 0.0], range=[s_lo, s_hi],
                  armature=JOINT_ARMATURE, damping=[JOINT_DAMPING, 0.0, 0.0])
    z_pitch = HINGE[2] - r_pitch
    _, m = module_for(r_pitch)
    root = z_pitch - 1.25 * m
    rack_x = (HINGE[0] - s_hi - 6 * MM, HINGE[0] - s_lo + 6 * MM)  # meshed under the pinion at every opening
    vol = _box(car, "jaw_rack", [rack_x[0], GEAR_Y[0], root - 5 * MM], [rack_x[1], GEAR_Y[1], root], PRINTED, True)
    tv, tf = rack_teeth(*rack_x, HINGE[0], z_pitch, width, m)
    spec.add_mesh(name="jaw_rack_teeth", uservert=_to_jaw(tv + [0.0, sum(GEAR_Y) / 2, 0.0]).ravel().tolist(),
                  userface=tf.ravel().tolist())
    car.add_geom(name="jaw_rack_teeth", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="jaw_rack_teeth", contype=0,
                 conaffinity=0, rgba=PRINTED, group=2, mass=0.0)
    # the carriage block under the rack, with a square bore for each rod's bushing
    rods = [z_pitch - d for d in ROD_DROP]
    top, bottom = root - 5 * MM, rods[1] - 6 * MM
    x0, x1 = CARRIAGE_X
    for k, (z0, z1) in enumerate([(rods[0] + BORE, top), (rods[1] + BORE, rods[0] - BORE), (bottom, rods[1] - BORE)]):
        vol += _box(car, f"jaw_carriage{k}", [x0, CARRIAGE_Y[0], z0], [x1, CARRIAGE_Y[1], z1], PRINTED, True)
    for k, zr in enumerate(rods):
        for side, (y0, y1) in (("out", (CARRIAGE_Y[0], ROD_Y - BORE)), ("in", (ROD_Y + BORE, CARRIAGE_Y[1]))):
            vol += _box(car, f"jaw_carriage_bore{k}_{side}", [x0, y0, zr - BORE], [x1, y1, zr + BORE], PRINTED, True)
    # the arm across to the blade, at a height where both are
    arm_z = min(max(z_pitch - 15 * MM, -95 * MM), -62 * MM)
    arm_x1 = 19 * MM if arm_z < -88 * MM else 26 * MM
    vol += _box(car, "jaw_carriage_arm", [12 * MM, CARRIAGE_Y[1], arm_z - 4 * MM], [arm_x1, -6 * MM, arm_z + 4 * MM],
                PRINTED, True)
    verts, faces = blade_mesh()
    spec.add_mesh(name="jaw_blade", uservert=verts.ravel().tolist(), userface=faces.ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    car.add_geom(name="jaw_blade_drawn", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="jaw_blade", contype=0,
                 conaffinity=0, group=2, mass=0.0, rgba=PRINTED)
    for k, part in enumerate(blade_parts()):
        spec.add_mesh(name=f"jaw_blade_cvx{k}", uservert=part.ravel().tolist())
        car.add_geom(name=f"jaw_moving_{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"jaw_blade_cvx{k}",
                     condim=4, group=3, mass=0.0)
    car.mass = blade_mass() + _box_mass(vol) + 2 * BUSHING_KG
    car.explicitinertial, car.ipos = True, _to_jaw(np.array([sum(CARRIAGE_X) / 2, -20 * MM, (top + bottom) / 2]))
    car.inertia = [car.mass * 4e-4] * 3

    # the rods, their clamp and its bracket to the frame, on the gripper; the rods end just past the
    # carriage fully open, so a long travel leaves the carriage off their back end when closed
    front = CARRIAGE_X[1] + s_hi + 2 * MM
    for k, zr in enumerate(rods):
        grip.add_geom(name=f"jaw_rod{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ROD_R, ROD_LEN / 2],
                      pos=[front - ROD_LEN / 2, ROD_Y, zr], quat=[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
                      rgba=STEEL, group=2, mass=0.0)
    back = front - ROD_LEN
    clamp_top = -31 * MM  # the bracket's top meets the frame's horn side (y = -28 mm, down to z = -37 mm)
    cvol = _box(grip, "jaw_clamp", [back, CARRIAGE_Y[0], rods[1] - 5 * MM], [back + 8 * MM, ROD_Y + 5 * MM, clamp_top],
                PRINTED, False)
    # the bracket crosses the gear plane behind the pinion to reach the frame (x >= -19.5 mm there): a
    # pinion too large leaves it no room, and the clamp hangs from nothing
    reach = min(max(back + 8 * MM, -15 * MM), HINGE[0] - r_pitch - 2 * m - 1 * MM)
    cvol += _box(grip, "jaw_clamp_bracket", [back, CARRIAGE_Y[0], clamp_top], [max(reach, back + 1 * MM), -28.0 * MM,
                                                                             clamp_top + 6 * MM], PRINTED, False)
    _carry(grip, _box_mass(cvol) + 2 * STEEL_KG_M3 * math.pi * ROD_R**2 * ROD_LEN)

    # the teeth: as stiff as printed teeth in mesh (about 1.5 N/um on a 6 mm face: 0.07 mm at 110 N here)
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="jaw_slide", name2="gripper",
                      data=[0.0, r_pitch, 0.0, 0.0, 0.0] + [0.0] * 7, solref=[0.004, 1.0],
                      solimp=[0.999, 0.9999, 0.001, 0.5, 2.0])
    spec.add_exclude(bodyname1=JAW_MESH, bodyname2="jaw_carriage")  # the pinion meshes with the rack
    return [("jaw_slide", r_pitch)]
