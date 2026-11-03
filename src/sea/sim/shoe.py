"""The printed pad shoe as the simulator builds it (D-066, docs/design-language.md).

A pad is not only its land: a PLA shoe slides over the stock jaw's tapered tip and carries the land in front.  Its
socket is the jaw's own geometry (the convex hull of the tip, grown by the slip fit), its walls wrap the tip at
least ``WALL`` thick, and two M2 cap screws hold it from behind.  ``design/pad_cad.py`` draws it for pictures from
the sections here; the simulator adds:

- **the shoe's outside**, as convex pieces between successive sections, behind the land's face by ``FRONT_MM``
  (the land boxes, ``design/fingers.pad_geoms``, stay the faces an object touches, with the skin's friction);
- **the screws' heads** behind the jaw, as cylinders;
- **the masses**: the shoe (its loft less the socket), the skin and both screws, on the jaw body that carries them.

Everything is in the pad's frame (mm): x out of the land, y up the jaw from the pad's bottom, z across.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import trimesh

from sea.contracts import FingerSpec
from sea.design import fingers as F

WALL = 1.6  # mm, four perimeters around the host
ARC = 8  # segments per round
SCREW = (1.0, 1.9, 2.0, 3.0)  # M2 cap screw: shank radius, head radius, head height, thread engaged in the jaw
SLIP = 0.2  # mm, the socket's slip fit (the design language's CLEARANCE)
EDGE = 1.0  # mm, a printed edge's round (the design language's EDGE_R["printed"])
FRONT_MM = 0.3  # the shoe's collision pieces stop this far behind the land's face
SCREW_AT = (0.3, 0.72)  # the screws' heights on the covered stock face, as fractions of COVER above the stock tip
DENSITY = {"pla": 1.24, "tpu": 1.21, "steel": 7.9}  # g/cm^3, solid: a shoe this thin prints at 100 %


@dataclass(frozen=True)
class PadFrame:
    """A pad's own frame in its jaw body's frame (mm): x out of the land, y up the jaw from the pad's bottom, z
    across.  The simulated boxes are ``fingers._face_boxes`` in it."""

    kind: str  # "jaw_fixed" | "jaw_moving"
    body: str
    rot: np.ndarray  # pad axes (columns) in the body frame
    origin: np.ndarray  # mm
    length: float  # mm the pad reaches past the stock tip


def frames(spec: FingerSpec) -> list[PadFrame]:
    """The two pad frames, as ``fingers.pad_geoms`` places the boxes."""
    rf = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    fixed = PadFrame("jaw_fixed", "gripper", rf, 1000 * np.array([F.FIXED_FACE_X, 0.0, F.FIXED_TIP_Z - spec.reach_mm * F.MM]),
                     spec.reach_mm)
    tilt = F._rz(-math.radians(spec.pad_angle_deg))
    rm = tilt @ np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    lm = F.moving_reach(spec)
    corner = np.array([F.MOVING_FACE_X, F.MOVING_TIP_Y, F.MOVING_MID_Z])
    shift = F.finger_shift(spec)
    moving = PadFrame("jaw_moving", F.moving_body(spec), rm, 1000 * (corner + tilt @ np.array([0.0, -lm, 0.0]) + shift),
                      1000 * lm)
    return [fixed, moving]


def boxes(spec: FingerSpec, fr: PadFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The simulated land boxes in the pad frame (mm): centre, half sizes, rotation."""
    return [(1000 * c, 1000 * h, r) for c, h, r in F._face_boxes(spec, fr.length / 1000)]


# ---------------------------------------------------------------- sections
def host_extent(host: trimesh.Trimesh, ys: list[float]) -> tuple[float, float] | None:
    """Over the planes y = each of ``ys``, the host's rearmost x and widest |z| near the pad."""
    pts = []
    for y in ys:
        sec = host.section(plane_origin=[0.0, y, 0.0], plane_normal=[0.0, 1.0, 0.0])
        if sec is not None:
            p = np.asarray(sec.vertices)
            pts.append(p[(np.abs(p[:, 0]) < 40) & (np.abs(p[:, 2]) < 40)])
    pts = [p for p in pts if len(p)]
    if not pts:
        return None
    p = np.vstack(pts)
    return float(p[:, 0].min()), float(np.abs(p[:, 2]).max())


def profile(t: float, w: float, g: float, r: float, xb: float) -> list[tuple[float, float]]:
    """A section of the shoe in (x, z): the land in front (flat, or a V of depth ``g``), rounds of radius ``r`` that
    start where the land ends and turn back to the side walls, and a back at ``xb`` with rounded corners."""
    half = w / 2
    d = np.array([g, half]) / math.hypot(g, half) if g > 0 else np.array([0.0, 1.0])  # the land's direction at its edge
    left = np.array([-d[1], d[0]])
    c = np.array([t, half]) + r * left
    a0 = math.atan2(-left[1], -left[0])
    side = [tuple(c + r * np.array([math.cos(a), math.sin(a)])) for a in np.linspace(a0, math.pi / 2, ARC + 1)]
    zs = c[1] + r
    rb = max(0.3, min(3.0, 0.45 * (c[0] - xb), 0.45 * zs))
    back = [(xb + rb + rb * math.cos(a), zs - rb + rb * math.sin(a)) for a in np.linspace(math.pi / 2, math.pi, ARC + 1)]
    plus = side + back
    minus = [(x, -z) for x, z in reversed(plus)]
    return plus + minus + ([(t - g, 0.0)] if g > 0 else [])


def socket_hull(host: trimesh.Trimesh, top: float, grid: float = 0.25) -> trimesh.Trimesh:
    """The host below ``top`` as a convex mesh grown by the slip fit.  The tip's points are snapped to ``grid`` (the
    STL is dense where it is round) and the hull is grown by that error too, so it still holds the whole tip."""
    v = np.asarray(host.vertices)
    near = (v[:, 1] <= top) & (np.abs(v[:, 0]) < 40) & (np.abs(v[:, 2]) < 40)
    pts = [v[near]]
    sec = host.section(plane_origin=[0.0, top, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if sec is not None:
        pts.append(np.asarray(sec.vertices))
    hull = trimesh.convex.convex_hull(np.unique(np.round(np.vstack(pts) / grid) * grid, axis=0))
    grow = (SLIP + grid * math.sqrt(3) / 2) / 0.85  # vertex normals lean up to ~30 deg off their faces
    return trimesh.convex.convex_hull(hull.vertices + grow * hull.vertex_normals)


@dataclass(frozen=True)
class Sections:
    ys: list[float]  # mm, up the pad from its bottom
    xb: list[float]  # each section's back
    r: list[float]  # each section's side round
    t: float  # land thickness, mm
    w: float  # land width
    g: float  # groove depth
    height: float  # the shoe's top: the reach plus the covered stock face
    socket: trimesh.Trimesh


def sections(spec: FingerSpec, fr: PadFrame, host: trimesh.Trimesh) -> Sections:
    """The shoe's sections: every 2 mm up the covered face, each wrapping the socket by ``WALL``, never behind the
    land boxes' rearmost point."""
    t, w, g = 1000 * F.thickness(spec), 1000 * F.width(spec), spec.groove_mm
    length, h = fr.length, fr.length + 1000 * F.COVER
    corners = np.vstack([c + (np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * hf) @ r.T
                         for c, hf, r in boxes(spec, fr)])
    xb0 = min(0.0, float(corners[:, 0].min()))
    ys = [0.0] + ([length] if length > 0.5 else []) + [y for y in np.arange(length + 2.0, h - 0.9, 2.0)] + [h]
    half_g = g / math.hypot(g, w / 2) if g > 0 else 0.0
    sock = socket_hull(host, h + 3.0)
    xbs, rs = [], []
    for y in ys:
        ext = host_extent(sock, [min(max(y + dy, 0.0), h) for dy in (-1.0, 0.0, 1.0)])
        xb, r = xb0, EDGE
        if ext is not None:
            xb = min(xb0, ext[0] - WALL)
            r = max(EDGE, (ext[1] + WALL - w / 2) / (1 + half_g))
        xbs.append(xb)
        rs.append(r)
    return Sections([float(y) for y in ys], xbs, rs, t, w, g, h, sock)


def screw_poses(sec: Sections, length: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two screws: where each head sits on the shoe's back, and the back's outward normal there."""
    yb, xb = np.array(sec.ys), np.array(sec.xb)
    out = []
    for f in SCREW_AT:
        y = length + f * F.COVER * 1000
        k = int(np.clip(np.searchsorted(yb, y) - 1, 0, len(yb) - 2))
        d = np.array([xb[k + 1] - xb[k], yb[k + 1] - yb[k]])
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])  # out of the back (toward -x: the sections climb in y)
        out.append((np.array([float(np.interp(y, yb, xb)), y, 0.0]), np.array([n[0], n[1], 0.0])))
    return out


# ---------------------------------------------------------------- what the simulator adds
@dataclass(frozen=True)
class Shoe:
    pieces: tuple[np.ndarray, ...]  # convex vertex sets, the jaw body's frame (m)
    screws: tuple[tuple[np.ndarray, np.ndarray], ...]  # head centre and axis, the body's frame (m)
    grams: float  # shoe, skin and both screws
    pla_cm3: float  # the shoe as printed, lands included
    tpu_cm3: float  # the skin


def _piece(pa: list[tuple[float, float]], ya: float, pb: list[tuple[float, float]], yb: float, xmax: float) -> np.ndarray:
    """The convex piece between two sections' profiles, cut off at ``xmax``."""
    pts = np.array([(min(x, xmax), ya, z) for x, z in pa] + [(min(x, xmax), yb, z) for x, z in pb])
    return trimesh.convex.convex_hull(pts).vertices


def _between(hull: trimesh.Trimesh, lo: float, hi: float) -> float:
    """The volume (mm^3) of a convex mesh between the planes y = ``lo`` and y = ``hi``."""
    v = np.asarray(hull.vertices)
    pts = [v[(v[:, 1] >= lo) & (v[:, 1] <= hi)]]
    for a, b in hull.edges_unique:
        for y in (lo, hi):
            da, db = v[a, 1] - y, v[b, 1] - y
            if da * db < 0:
                pts.append((v[a] + (v[b] - v[a]) * da / (da - db))[None])
    p = np.vstack(pts)
    return float(trimesh.convex.convex_hull(p).volume) if len(p) >= 4 else 0.0


def build(spec: FingerSpec, fr: PadFrame, host: trimesh.Trimesh) -> Shoe:
    """The shoe on one pad.  ``host`` is the jaw's mesh in the pad frame (mm)."""
    sec = sections(spec, fr, host)
    profs = [profile(sec.t, sec.w, sec.g, r, xb) for xb, r in zip(sec.xb, sec.r, strict=True)]
    halves = ((lambda z: z >= -1e-9), (lambda z: z <= 1e-9)) if sec.g > 0 else ((lambda z: True),)
    pieces, loft = [], 0.0
    for k in range(len(sec.ys) - 1):
        for keep in halves:  # a V groove is concave across: one convex piece per side of it
            pa = [p for p in profs[k] if keep(p[1])]
            pb = [p for p in profs[k + 1] if keep(p[1])]
            pieces.append(_piece(pa, sec.ys[k], pb, sec.ys[k + 1], sec.t - FRONT_MM))
            loft += trimesh.convex.convex_hull(np.array([(x, sec.ys[k], z) for x, z in pa]
                                                         + [(x, sec.ys[k + 1], z) for x, z in pb])).volume
    socket_cm3 = _between(sec.socket, 0.0, sec.ys[-1]) / 1000
    skin_cm3 = spec.pad_tpu_mm * sec.w * sec.height / 1000 if spec.pad_tpu_mm > 0 else 0.0
    shoe_cm3 = max(loft / 1000 - socket_cm3 - skin_cm3, 0.0)
    r, rh, hh, bite = SCREW
    screw_cm3 = 2 * math.pi * (rh**2 * hh + r**2 * (WALL + SLIP + bite)) / 1000
    grams = shoe_cm3 * DENSITY["pla"] + skin_cm3 * DENSITY["tpu"] + screw_cm3 * DENSITY["steel"]
    to_body = lambda p: (fr.rot @ np.asarray(p).T).T / 1000 + fr.origin / 1000  # noqa: E731
    screws = tuple((to_body(c + n * hh / 2), fr.rot @ n) for c, n in screw_poses(sec, fr.length))
    return Shoe(tuple(to_body(p) for p in pieces), screws, grams, shoe_cm3, skin_cm3)


Host = tuple[str, bytes, bytes]  # a jaw body, its finger's vertices in the body's frame (m) and faces, as bytes


@lru_cache(maxsize=64)
def shoes(spec: FingerSpec, hosts: tuple[Host, ...]) -> tuple[tuple[PadFrame, Shoe], ...]:
    """Both pads' shoes, on the fingers ``hosts`` gives for each jaw body (bytes, so the result caches on them)."""
    meshes = {b: (np.frombuffer(v).reshape(-1, 3), np.frombuffer(f, dtype=np.int64).reshape(-1, 3)) for b, v, f in hosts}
    out = []
    for fr in frames(spec):
        if fr.body not in meshes:
            continue
        v, f = meshes[fr.body]
        host = trimesh.Trimesh((fr.rot.T @ (v * 1000 - fr.origin).T).T, f, process=False)
        out.append((fr, build(spec, fr, host)))
    return tuple(out)
