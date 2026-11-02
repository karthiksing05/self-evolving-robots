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
    _hub_gripper(grip, "jaw_bracket_hub", f_g)
    grip.add_geom(name="jaw_bracket", type=mujoco.mjtGeom.mjGEOM_BOX, size=[2.8 * MM, 15.0 * MM, 9.0 * MM],
                  pos=[30.4 * MM + 2.8 * MM, HINGE[1] - MID_Z, f_g[2] - 6.0 * MM], rgba=PRINTED, group=2, mass=0.0)

    spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name1="jaw_follower", name2="jaw_coupler",
                      objtype=mujoco.mjtObj.mjOBJ_BODY, data=[0.0, -L, 0.0] + [0.0] * 9,
                      solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])
    for a, b in (("gripper", "jaw_coupler"), ("jaw_follower", JAW_MESH), ("jaw_follower", "jaw_coupler")):
        spec.add_exclude(bodyname1=a, bodyname2=b)  # pinned to each other; clearance elsewhere is a checked law
    return [("jaw_coupler_q", -1.0), ("jaw_follower_q", 1.0)]


def _hub_gripper(body, name: str, at_g: np.ndarray) -> None:
    """The follower's fixed pin hub, on the gripper body (axis along the gripper's y)."""
    lo, hi = HINGE[1] - (HORN_SIDE[0] - 0.3 * MM), HINGE[1] - (REAR_SIDE[1] + 0.3 * MM)  # gripper y = HINGE y - jaw z
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[5.0 * MM, (hi - lo) / 2],
                  pos=[at_g[0], (lo + hi) / 2, at_g[2]], quat=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
                  rgba=PRINTED, group=2, mass=0.0)


def _quat_z(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]
