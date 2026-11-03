"""The SO-101 gripper's measurements and the physics of its jaw mechanisms (D-057, D-076).

The mechanisms themselves are graphs (``sea.mech``; the catalogue's jaws are ``mech/seed/gripper.py``) that
``mech.build`` compiles.  This module holds what they are measured against: the hinge and the jaw's side
planes, the stock finger's blade below its cut (``BLADE_CUT_Z``), the gripper frame's mesh that brackets seat
on, gear and rack teeth, and the rods' bending (``RodSprings``, ``rack_lift_mm``).

The four-bar's follower pin sits ``FOLLOWER_MM`` from the hinge at 45 degrees up from the closing axis, so the
transmission angle stays above 35 degrees over the jaw's range (it reaches toggle only at 135 degrees).
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
LONG_ROD = 150.0 * MM  # the long slide's rods (D-064)
SECTOR_TEETH = 2  # a sector's teeth beyond the arc in mesh, each end
# The long slide's rack hangs a sector radius below the hinge, near the fingertips: its rods run outboard of a
# 102 mm deep object (y = -56 mm) and above the rack's pitch line, the carriage block beside the gear plane,
# so the lowest moving part is the arm under the rack.  The clamp sits beside the fixed finger, its bracket
# crossing the gear plane between the rack's teeth and the sector's sweep to the finger's outer face (y about
# -8 mm there, from the frame mesh): behind the hinge the sector sweeps where the short slide's bracket ran.
LONG_ROD_Y, LONG_ROD_RISE = -56.0 * MM, (4.0 * MM, 16.0 * MM)  # the rods' heights above the rack's pitch line
CLAMP_X = (-19.0 * MM, -12.0 * MM)
FRAME_MESH = "wrist_roll_follower_so101_v1"
FRAME_POSE = ((8.32667e-17, -0.000218214, 0.000949706), (0.0, 1.0, 0.0, 0.0))  # its geom in the gripper (SO-101 MJCF)
FIT = 0.3 * MM  # a bracket screwed to the frame seats this far into the frame mesh's face


@lru_cache(maxsize=1)
def _frame():
    """The gripper frame's mesh in the gripper's frame (m)."""
    import trimesh

    from sea.control.frames import STOCK_MJCF

    m = trimesh.load(STOCK_MJCF.parent / "assets" / f"{FRAME_MESH}.stl")
    pos, quat = FRAME_POSE
    t = np.eye(4)
    t[:3, :3] = trimesh.transformations.quaternion_matrix(quat)[:3, :3]
    t[:3, 3] = pos
    m.apply_transform(t)
    return m


def frame_face_y(x: tuple[float, float], z: tuple[float, float], y_from: float) -> float | None:
    """Where a bracket coming from ``y_from`` toward +y first meets the frame over its footprint (x, z): the
    nearest the frame's material comes, so the bracket ends on the frame's face and nowhere enters it.  None if
    no part of the footprint has frame behind it (D-076)."""
    tri = np.asarray(_frame().triangles)  # (n, 3, xyz)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) - (c[:, 0] - a[:, 0]) * (b[:, 2] - a[:, 2])
    ok = np.abs(det) > 1e-14
    a, b, c, det = a[ok], b[ok], c[ok], det[ok]
    best = None
    for px in np.linspace(*x, 7):
        for pz in np.linspace(*z, 5):  # a ray along +y through (px, pz): the triangles whose x-z shadow holds it
            u = ((px - a[:, 0]) * (c[:, 2] - a[:, 2]) - (c[:, 0] - a[:, 0]) * (pz - a[:, 2])) / det
            v = ((b[:, 0] - a[:, 0]) * (pz - a[:, 2]) - (px - a[:, 0]) * (b[:, 2] - a[:, 2])) / det
            inside = (u >= 0) & (v >= 0) & (u + v <= 1)
            y = (a[:, 1] + u * (b[:, 1] - a[:, 1]) + v * (c[:, 1] - a[:, 1]))[inside]
            y = y[y > y_from]
            if len(y):
                best = float(y.min()) if best is None else min(best, float(y.min()))
    return best
ROD_DROP = (10.0 * MM, 22.0 * MM)  # the rods' heights below the rack's pitch line
CARRIAGE_X = (8.0 * MM, 28.0 * MM)  # over the blade
CARRIAGE_Y = (-45.0 * MM, GEAR_Y[1])
BORE = 1.6 * MM  # half the square bore each rod runs in: the LM3UU bushing's 3 mm bore with 0.1 mm clearance
BUSHING_KG = 0.0014  # LM3UU, each
STEEL_KG_M3 = 7850.0
# The rods bend (D-076).  Each is a cantilever from the clamp, loaded at its free end by the carriage's bushing
# (LM3UU: 10 mm long).  Ground steel, E = 200 GPa, 3 mm across: EI = 0.795 N m^2.
ROD_EI = 200e9 * math.pi * ROD_R**4 / 4
BUSHING_LEN = 10.0 * MM
# The bushing's rated load: no LM3UU datasheet gives one; the 3 mm miniature LMS3UU is rated 37.4-52.9 N static
# and 48.9-56.1 N dynamic (motion.com, sku 08943031).  Above its static rating the balls dent the rod.
BUSHING_RATED_N = 52.9
PRESSURE_ANGLE = math.radians(20.0)  # the teeth's: the mesh pushes the rack off the gear with tan(20) of its load
TOOTH_SKIP = 0.5  # of a module: a rack pushed this far off the gear has lost most of its working depth (2 m)
HOLD_STEPS = 10  # a lift or load counts once it lasts 20 ms, as the force limit's does: not an impact's spike


def rod_stiffness(a: float) -> tuple[float, float]:
    """The two rods' stiffness at overhang ``a`` (m), each mode held still: across them (N/m) and turning (N m/rad)."""
    return 2 * 12 * ROD_EI / a**3, 2 * 4 * ROD_EI / a


def rack_lift_mm(r_pitch: float, sector: bool, q: float, q_hi: float, torque: float) -> tuple[float, float, float]:
    """A slide at jaw angle ``q`` with the servo at ``torque``, statically, as ``RodSprings`` loads it in the
    simulator (D-076): the rods' overhang (mm), how far the rack lifts off the gear (mm) under the teeth's
    separating force and the squeeze's couple (the pad grips below and beside the rods, worst sign), and the
    load on each bushing (N)."""
    centre = sum(CARRIAGE_X) / 2
    z_pitch = HINGE[2] - r_pitch
    if sector:
        clamp, rod_y, rod_z = CLAMP_X[1], LONG_ROD_Y, z_pitch + sum(LONG_ROD_RISE) / 2
    else:
        clamp = CARRIAGE_X[1] + q_hi * r_pitch + 2 * MM - ROD_LEN + 8 * MM
        rod_y, rod_z = ROD_Y, z_pitch - sum(ROD_DROP) / 2
    s = q * r_pitch
    a = max(centre + s - BUSHING_LEN / 2 - clamp, BUSHING_LEN / 2)
    squeeze = torque / r_pitch
    sep = squeeze * math.tan(PRESSURE_ANGLE)
    dx = HINGE[0] - (centre + s)
    tip = HINGE + R0 @ np.array([*blade_tip(), 0.0])  # where the pad meets the object, gripper frame
    pitch, yaw = squeeze * (tip[2] - rod_z), squeeze * (tip[1] - rod_y)  # the squeeze's couples about y and z
    k = 2 * ROD_EI / a**3 * np.array([[12.0, -6 * a], [-6 * a, 4 * a * a]])
    lift = 0.0
    for sign in (-1.0, 1.0):
        v, t = np.linalg.solve(k, [-sep, -sep * dx + sign * pitch])
        lift = max(lift, -(v + t * dx))
    bushing = math.hypot(sep / 2 + abs(pitch) / 2 / BUSHING_LEN, abs(yaw) / 2 / BUSHING_LEN)
    return a * 1000, lift * 1000, bushing


class RodSprings:
    """The rods' bending as forces on the carriage's four rod joints (D-076).  Per step, at the carriage's
    overhang a from the clamp, each plane's deflection v and slope t are loaded by the two cantilevers'
    stiffness 2 EI / a^3 [[12, -6a], [-6a, 4a^2]], and the teeth push the rack off the gear with tan(20) of the
    force they carry.  ``broken`` names what a real slide could not do: a rack lifted ``TOOTH_SKIP`` of a module
    off the gear skips its teeth; a bushing past its rated load dents the rods."""

    JOINTS = ("jaw_rod_dy", "jaw_rod_yaw", "jaw_rod_dz", "jaw_rod_pitch")

    def __init__(self, model: mujoco.MjModel):
        self.clamp, self.centre, self.module, self.mesh_x = (float(x) for x in model.numeric("jaw_rods").data)
        self.q = [int(model.jnt_qposadr[model.joint(n).id]) for n in self.JOINTS]
        self.v = [int(model.jnt_dofadr[model.joint(n).id]) for n in self.JOINTS]
        self.slide = int(model.jnt_qposadr[model.joint("jaw_slide").id])
        slide_j = model.joint("jaw_slide").id
        self.eq = next(i for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
                       and model.eq_obj1id[i] == slide_j)
        self.reset()

    def reset(self) -> None:
        self.lift_mm = 0.0
        self.bushing_n = 0.0
        self.at_mm = 0.0
        self._recent: list[tuple[float, float, float]] = []  # (lift mm, bushing N, overhang mm) per step

    def overhang(self, s: float) -> float:
        return max(self.centre + s - BUSHING_LEN / 2 - self.clamp, BUSHING_LEN / 2)

    def apply(self, d: mujoco.MjData) -> None:
        s = float(d.qpos[self.slide])
        a = self.overhang(s)
        c = 2 * ROD_EI / a**3
        load = []
        for k in (0, 2):  # (dy, yaw), then (dz, pitch)
            v, t = float(d.qpos[self.q[k]]), float(d.qpos[self.q[k + 1]])
            f, m = c * (12 * v - 6 * a * t), c * (-6 * a * v + 4 * a * a * t)
            d.qfrc_applied[self.v[k]], d.qfrc_applied[self.v[k + 1]] = -f, -m
            load.append(abs(f) / 2 + abs(m) / 2 / BUSHING_LEN)  # per rod: its share, and the couple across a bushing
        rows = (d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[: d.nefc] == self.eq)
        f_t = float(np.abs(d.efc_force[: d.nefc][rows]).sum()) if d.nefc else 0.0
        dx = self.mesh_x - (self.centre + s)  # the mesh, from the carriage's centre (pitch turns about it)
        sep = f_t * math.tan(PRESSURE_ANGLE)
        d.qfrc_applied[self.v[2]] -= sep  # down: the rack hangs below the gear
        d.qfrc_applied[self.v[3]] -= sep * dx
        lift = -(float(d.qpos[self.q[2]]) + float(d.qpos[self.q[3]]) * dx)
        self._recent = [*self._recent[-(HOLD_STEPS - 1):], (lift * 1000, math.hypot(*load), a * 1000)]
        if len(self._recent) == HOLD_STEPS:  # what held over the last HOLD_STEPS steps
            held_lift = min(r[0] for r in self._recent)
            if held_lift > self.lift_mm:
                self.lift_mm, self.at_mm = held_lift, self._recent[-1][2]
            self.bushing_n = max(self.bushing_n, min(r[1] for r in self._recent))

    def broken(self) -> str:
        if self.lift_mm > TOOTH_SKIP * self.module * 1000:
            return (f"unphysical: the rods let the rack lift {self.lift_mm:.1f} mm off the gear ({self.at_mm:.0f} mm "
                    f"out from the clamp); past {TOOTH_SKIP * self.module * 1000:.1f} mm the teeth skip")
        if self.bushing_n > BUSHING_RATED_N:
            return (f"unphysical: a bushing carried {self.bushing_n:.0f} N; it is rated {BUSHING_RATED_N:g} N, past "
                    "which it dents the rod")
        return ""


def _to_jaw(p: np.ndarray) -> np.ndarray:
    """Gripper-frame points into the jaw's frame at q = 0 (rows)."""
    return (np.asarray(p) - HINGE) @ R0


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


def gear(r_pitch: float, width: float, arc: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A spur gear on z: a root disc and whole teeth, a gap facing -y (where it meets the rack at q = 0).  With
    ``arc`` (gear-frame angles, at most half a turn) a sector: a wedge from the axis and the teeth inside it."""
    n, m = module_for(r_pitch)
    rf = r_pitch - 1.25 * m
    if arc is None:
        a = np.linspace(0.0, 2 * math.pi, 4 * n, endpoint=False)
        disc = np.c_[rf * np.cos(a), rf * np.sin(a)]
    else:
        a = np.linspace(*arc, max(int(4 * n * (arc[1] - arc[0]) / (2 * math.pi)), 8))
        disc = np.r_[[[0.0, 0.0]], np.c_[rf * np.cos(a), rf * np.sin(a)]]
    teeth = []
    for k in range(n):
        t = -math.pi / 2 + 2 * math.pi * (k + 0.5) / n
        if arc is not None and (t - arc[0]) % (2 * math.pi) > arc[1] - arc[0]:
            continue
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
