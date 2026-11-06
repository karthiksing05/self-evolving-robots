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
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from sea.mech.members import (  # noqa: F401
    HOLD_STEPS,
    PLA_ALLOW_PA,
    PLA_E_PA,
    PLA_FLEX_PA,
    PRINT_FACTOR,
    Member,
    PrintedMembers,
)
from sea.sim import solid_fe

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


@lru_cache(maxsize=512)
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


@dataclass(frozen=True)
class Bushing:
    """A linear ball bushing and the steel rod it runs on (D-098).  Above its static rating the balls dent the rod."""

    item: str
    rod_r: float
    od: float
    length: float
    kg: float
    rated_n: float

    @property
    def ei(self) -> float:
        return 200e9 * math.pi * self.rod_r**4 / 4


# Static ratings, the lowest listed where sources differ: LM4 and LM5 from THK's LM table (catalogue excerpt; RS lists
# 210 N for LM4, so 127 N is the safe one), LM6 from THK, BKL and HC (all 265 N), LM8 from Misumi C-LM8UU (Samick 400,
# BKL 392, THK 402).  Sizes and masses are the same tables' (bore x outside x length).
BUSHINGS: dict[str, Bushing] = {b.item: b for b in (
    Bushing("lm3uu", ROD_R, 7.0 * MM, BUSHING_LEN, BUSHING_KG, BUSHING_RATED_N),
    Bushing("lm4uu", 2.0 * MM, 8.0 * MM, 12.0 * MM, 0.0019, 127.0),
    Bushing("lm5uu", 2.5 * MM, 10.0 * MM, 15.0 * MM, 0.004, 206.0),
    Bushing("lm6uu", 3.0 * MM, 12.0 * MM, 19.0 * MM, 0.0085, 265.0),
    Bushing("lm8uu", 4.0 * MM, 15.0 * MM, 24.0 * MM, 0.017, 380.0))}
LM3 = BUSHINGS["lm3uu"]


def bushing_for(rod_mm: float) -> Bushing:
    """The bushing of a rod this thick; a size the catalogue lacks is refused, naming those it has."""
    for b in BUSHINGS.values():
        if abs(2000 * b.rod_r - rod_mm) < 1e-6:
            return b
    raise ValueError(f"no bushing for a {rod_mm:g} mm rod; the sizes are {[round(2000 * b.rod_r) for b in BUSHINGS.values()]} mm")


PRESSURE_ANGLE = math.radians(20.0)  # the teeth's: the mesh pushes the rack off the gear with tan(20) of its load
TOOTH_SKIP = 0.5  # of a module: a rack pushed this far off the gear has lost most of its working depth (2 m)


# Lewis form factors for 20 degree teeth (Shigley, table 14-2).
_LEWIS = ((12, 0.245), (14, 0.277), (17, 0.302), (20, 0.322), (24, 0.337), (30, 0.359), (40, 0.389), (50, 0.408),
          (75, 0.435), (100, 0.447), (150, 0.460), (300, 0.472))
LEWIS_RACK = 0.485


def lewis_y(teeth: float) -> float:
    return float(np.interp(teeth, [n for n, _ in _LEWIS], [y for _, y in _LEWIS]))


@dataclass(frozen=True)
class RackBeam:
    """A printed rack as a beam, with what its teeth can carry (D-100).  ``lo``..``hi`` is where the carriage holds it
    (gripper x at zero travel); ``ei_sep`` and ``c_sep`` its bending against the teeth's separating force, ``ei_min`` its
    weakest bending, ``tooth`` its stress per newton along the teeth (Pa/N), ``backed`` a roller behind the mesh."""

    lo: float
    hi: float
    ei_sep: float
    c_sep: float
    ei_min: float
    tooth: float
    backed: bool
    back_l: float = 0.0  # the roller's arm, from its support to the mesh, and its bending stiffness and half depth (D-103)
    back_ei: float = math.inf
    back_c: float = 0.0

    def data(self) -> list[float]:
        return [self.lo, self.hi, self.ei_sep, self.c_sep, self.ei_min, self.tooth, float(self.backed), self.back_l,
                self.back_ei, self.back_c]

    def overhang(self, x_mesh: float) -> float:
        return max(self.lo - x_mesh, x_mesh - self.hi, 0.0)

    def area_hint(self, depth: float) -> float:
        """The section's area, as the rectangle of its separating bending: its width by its depth."""
        return 12 * (self.ei_sep / PLA_E_PA) / depth**2

    def loads(self, x_mesh: float, teeth_n: float) -> dict[str, float]:
        """At the point on the rack under the gear (``x_mesh``, zero-travel gripper x) with this force on its teeth: the
        bend off the gear (m), the bending and tooth stress (Pa) and the load it buckles at (N, inf when in tension)."""
        n = self.overhang(x_mesh)
        sep = teeth_n * math.tan(PRESSURE_ANGLE)
        def beam(length: float, ei: float, c: float) -> tuple[float, float]:  # solved on the part, as a rectangle of this section (D-124)
            depth = 2 * c
            return solid_fe.bend_near(length, 12 * (ei / PLA_E_PA) / depth**3, depth)

        if self.backed:  # the roller holds the rack at the mesh, as far as its arm does
            per = beam(self.back_l, self.back_ei, self.back_c) if self.back_l > 0 and math.isfinite(self.back_ei) else (0.0, 0.0)
        else:
            per = beam(n, self.ei_sep, self.c_sep)
        stress, bend = sep * per[0], sep * per[1]
        k = 1.0 if self.backed else 2.0
        # closing pulls the rack rearward: behind its support it is in tension, ahead of it in compression
        if x_mesh > self.hi and n > 0:
            depth = 2 * self.c_sep
            area = self.area_hint(depth)
            weak = math.sqrt(12 * (self.ei_min / PLA_E_PA) / area)
            buckle = solid_fe.buckle_near(k * n, area / weak, weak)
        else:
            buckle = math.inf
        return {"overhang": n, "bend": bend, "stress": stress, "tooth": teeth_n * self.tooth, "buckle": buckle}


def rod_stiffness(a: float, ei: float = ROD_EI) -> tuple[float, float]:
    """The two rods' stiffness at overhang ``a`` (m), each mode held still: across them (N/m) and turning (N m/rad)."""
    return 2 * 12 * ei / a**3, 2 * 4 * ei / a


def _beam(n: float, ei: float = ROD_EI) -> np.ndarray:
    """A rod segment of length ``n`` as a beam element: (deflection, slope) at both of its ends."""
    return ei / n**3 * np.array([[12, 6 * n, -12, 6 * n], [6 * n, 4 * n * n, -6 * n, 2 * n * n],
                                     [-12, -6 * n, 12, -6 * n], [6 * n, 2 * n * n, -6 * n, 4 * n * n]])


def guide_beam(xs: tuple[float, ...], bs: tuple[float, ...], span: float | None,
               ei: float = ROD_EI) -> tuple[np.ndarray, np.ndarray]:
    """Two rods clamped at x = 0 (and at ``span`` if held there too), each sleeved by bushings at ``xs`` from the
    clamp (sorted, apart), the bushings ``bs`` along the rods from the carriage's middle (D-080).  The rigid
    carriage sets every bushing's deflection and slope from its own (v, t) at its middle.  Returns the
    carriage's stiffness (2 x 2, both rods) and the map from (v, t) to each bushing's force and couple on one rod
    (2n x 2).  One bushing on cantilevers gives D-076's 2 EI / a^3 [[12, -6a], [-6a, 4a^2]]."""
    pts = [0.0, *xs, *([span] if span is not None else [])]
    n = len(pts)
    k = np.zeros((2 * n, 2 * n))
    for i in range(n - 1):
        idx = np.arange(2 * i, 2 * i + 4)
        k[np.ix_(idx, idx)] += _beam(pts[i + 1] - pts[i], ei)
    free = slice(2, 2 * (len(xs) + 1))
    k = k[free, free]
    b = np.zeros((2 * len(xs), 2))
    for i, off in enumerate(bs):
        b[2 * i], b[2 * i + 1] = (1.0, off), (0.0, 1.0)
    return 2 * b.T @ k @ b, k @ b


def bushing_overhangs(centre: float, s: float, clamp: float, bs: tuple[float, ...],
                      far: float | None, length: float = BUSHING_LEN) -> tuple[float, ...]:
    """Each bushing's distance from the clamp at slide travel ``s``, kept a half bushing clear of each clamp and a
    bushing apart (where a carriage would meet them, the interference law sees it)."""
    out: list[float] = []
    for off in sorted(bs):
        a = max(centre + s + off - length / 2 - clamp, length / 2, (out[-1] + length) if out else 0.0)
        out.append(a)
    if far is not None:
        span = far - clamp
        out = [min(a, span - length / 2 - (len(out) - 1 - i) * length) for i, a in enumerate(out)]
    return tuple(out)


def guide_static(clamp: float, centre: float, bs: tuple[float, ...], far: float | None, s: float, sep: float,
                 dx: float, pitch: float, yaw: float, hw: Bushing = LM3) -> tuple[float, float, float]:
    """A guide loaded statically at travel ``s``: the teeth's separating force ``sep`` at ``dx`` from the
    carriage's middle and the squeeze's couples about y (``pitch``) and z (``yaw``), worst sign.  Returns the
    nearest bushing's overhang (m), the rack's lift off the gear (m) and the most loaded bushing's load (N)."""
    xs = bushing_overhangs(centre, s, clamp, bs, far, hw.length)
    kc, kb = guide_beam(xs, tuple(sorted(bs)), None if far is None else far - clamp, hw.ei)
    lift, worst = 0.0, 0.0
    for sign in (-1.0, 1.0):
        u = np.linalg.solve(kc, [-sep, -sep * dx + sign * pitch])
        w = np.linalg.solve(kc, [0.0, yaw])
        lift = max(lift, -(u[0] + u[1] * dx))
        fz, fy = kb @ u, kb @ w
        for i in range(len(xs)):
            worst = max(worst, math.hypot(abs(fz[2 * i]) + abs(fz[2 * i + 1]) / hw.length,
                                          abs(fy[2 * i]) + abs(fy[2 * i + 1]) / hw.length))
    return xs[0], lift, worst


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
        # a built guide's rods held at both ends or several bushings (D-080): far end (nan: cantilevered), the
        # rods' free end, then each bushing's offset from the carriage's middle
        self.beam = None
        if any(model.numeric(i).name == "jaw_guide" for i in range(model.nnumeric)):
            far, end, *bs = (float(x) for x in model.numeric("jaw_guide").data)
            self.beam = (None if math.isnan(far) else far, end, tuple(sorted(bs)))
        self.ei, self.length, self.rated = ROD_EI, BUSHING_LEN, BUSHING_RATED_N  # a guide on other parts says so (D-098)
        if any(model.numeric(i).name == "jaw_bushing" for i in range(model.nnumeric)):
            self.ei, self.length, self.rated = (float(x) for x in model.numeric("jaw_bushing").data)
        self.rack = None  # the printed rack as a beam (D-100)
        if any(model.numeric(i).name == "jaw_rack" for i in range(model.nnumeric)):
            lo, hi, ei_sep, c, ei_min, tooth, backed, *back = (float(x) for x in model.numeric("jaw_rack").data)
            self.rack = RackBeam(lo, hi, ei_sep, c, ei_min, tooth, bool(backed), *back)
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
        self.off_end_mm = 0.0  # how far a bushing ran past the rods' free end
        self._recent: list[tuple[float, float, float]] = []  # (lift mm, bushing N, overhang mm) per step
        self.bend_mm = 0.0  # the printed rack's own bend, and its stresses and buckling, each as held 20 ms (D-100)
        self.bend_mpa = 0.0
        self.tooth_mpa = 0.0
        self.buckled = (0.0, math.inf)
        self._rack: list[tuple[float, float, float, float, float]] = []

    def overhang(self, s: float) -> float:
        return max(self.centre + s - self.length / 2 - self.clamp, self.length / 2)

    def apply(self, d: mujoco.MjData) -> None:
        s = float(d.qpos[self.slide])
        a = self.overhang(s)
        load = []
        if self.beam is None:
            c = 2 * self.ei / a**3
            for k in (0, 2):  # (dy, yaw), then (dz, pitch)
                v, t = float(d.qpos[self.q[k]]), float(d.qpos[self.q[k + 1]])
                f, m = c * (12 * v - 6 * a * t), c * (-6 * a * v + 4 * a * a * t)
                d.qfrc_applied[self.v[k]], d.qfrc_applied[self.v[k + 1]] = -f, -m
                load.append(abs(f) / 2 + abs(m) / 2 / self.length)  # per rod: its share, and the couple across a bushing
        else:  # the rods as beams through every bushing (D-080)
            far, end, bs = self.beam
            xs = bushing_overhangs(self.centre, s, self.clamp, bs, far, self.length)
            a = xs[0]
            kc, kb = guide_beam(xs, bs, None if far is None else far - self.clamp, self.ei)
            per = []
            for k in (0, 2):
                u = np.array([float(d.qpos[self.q[k]]), float(d.qpos[self.q[k + 1]])])
                f, m = kc @ u
                d.qfrc_applied[self.v[k]], d.qfrc_applied[self.v[k + 1]] = -f, -m
                per.append(kb @ u)
            load = [max(math.hypot(abs(per[0][2 * i]) + abs(per[0][2 * i + 1]) / self.length,
                                   abs(per[1][2 * i]) + abs(per[1][2 * i + 1]) / self.length) for i in range(len(xs))), 0.0]
            self.off_end_mm = max(self.off_end_mm, (self.centre + s + bs[-1] + self.length / 2 - end) * 1000)
        rows = (d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[: d.nefc] == self.eq)
        f_t = float(np.abs(d.efc_force[: d.nefc][rows]).sum()) if d.nefc else 0.0
        dx = self.mesh_x - (self.centre + s)  # the mesh, from the carriage's centre (pitch turns about it)
        sep = f_t * math.tan(PRESSURE_ANGLE)
        d.qfrc_applied[self.v[2]] -= sep  # down: the rack hangs below the gear
        d.qfrc_applied[self.v[3]] -= sep * dx
        lift = -(float(d.qpos[self.q[2]]) + float(d.qpos[self.q[3]]) * dx)
        if self.rack is not None:  # the point on the rack under the gear: it moved with the carriage
            r = self.rack.loads(self.mesh_x - s, f_t)
            lift += r["bend"]
            self._rack = [*self._rack[-(HOLD_STEPS - 1):], (r["bend"] * 1000, r["stress"] / 1e6, r["tooth"] / 1e6, f_t, r["buckle"])]
            if len(self._rack) == HOLD_STEPS:
                self.bend_mm = max(self.bend_mm, min(x[0] for x in self._rack))
                self.bend_mpa = max(self.bend_mpa, min(x[1] for x in self._rack))
                self.tooth_mpa = max(self.tooth_mpa, min(x[2] for x in self._rack))
                held, at = min(x[3] for x in self._rack), self._rack[-1][4]
                if held / at > self.buckled[0] / self.buckled[1]:
                    self.buckled = (held, at)
        self._recent = [*self._recent[-(HOLD_STEPS - 1):], (lift * 1000, math.hypot(*load), a * 1000)]
        if len(self._recent) == HOLD_STEPS:  # what held over the last HOLD_STEPS steps
            held_lift = min(r[0] for r in self._recent)
            if held_lift > self.lift_mm:
                self.lift_mm, self.at_mm = held_lift, self._recent[-1][2]
            self.bushing_n = max(self.bushing_n, min(r[1] for r in self._recent))

    def broken(self) -> str:
        if self.off_end_mm > 0:
            return (f"unphysical: a bushing ran {self.off_end_mm:.0f} mm past the rods' free end; the carriage "
                    "leaves its rods")
        if self.lift_mm > TOOTH_SKIP * self.module * 1000:
            flex = f", {self.bend_mm:.1f} mm of it the rack's own flex" if self.bend_mm > 0.05 else ""
            return (f"unphysical: the rods let the rack lift {self.lift_mm:.1f} mm off the gear ({self.at_mm:.0f} mm "
                    f"out from the clamp{flex}); past {TOOTH_SKIP * self.module * 1000:.1f} mm the teeth skip")
        allow = PLA_ALLOW_PA / 1e6
        if self.tooth_mpa > allow:
            return f"unphysical: the printed teeth carried {self.tooth_mpa:.0f} MPa at their root; printed PLA allows {allow:.0f} MPa"
        if self.bend_mpa > allow:
            return f"unphysical: the printed rack bent to {self.bend_mpa:.0f} MPa beyond the carriage; printed PLA allows {allow:.0f} MPa"
        if self.buckled[0] > self.buckled[1] / PRINT_FACTOR:
            return (f"unphysical: the rack carried {self.buckled[0]:.0f} N along it; at that length it buckles at "
                    f"{self.buckled[1]:.0f} N, so may carry {self.buckled[1] / PRINT_FACTOR:.0f}")
        if self.bushing_n > self.rated:
            return (f"unphysical: a bushing carried {self.bushing_n:.0f} N; it is rated {self.rated:g} N, past "
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
