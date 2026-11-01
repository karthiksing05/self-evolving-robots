"""Parametric tool templates (spec 8.8, Appendix B).

Every template is built from simple convex pieces, so its collision decomposition is known by
construction ("analytic" parts): the CadQuery solid is exactly the union of ``parts``.  CadQuery
works in millimetres; everything a ``ToolGeometry`` exposes (parts, tip) is in meters, in the tool
frame = URDF ``gripper_frame_link``: origin at the stock jaw tip, +z = approach direction.

Mount convention (all templates): a 20 x 20 x 12 mm dovetail block occupies z in [0, 12] mm,
centred on the z axis.  Working geometry starts at its top face (z = 12 mm); ``length``
parameters (hook, slot_inserter, probe) are measured from the jaw tip, so the tip lands where
Appendix B puts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cadquery as cq
import numpy as np

from sea.contracts import ContactType, ToolSpec

MOUNT_W = 20.0  # mm, dovetail block footprint (x and y)
MOUNT_H = 12.0  # mm, dovetail block height along +z
RING_SEGMENTS = 16  # knob_cup wall polygon

PartKind = Literal["box", "cylinder", "capsule", "sphere", "mesh"]
Material = Literal["pla", "tpu"]

DEFAULT_SOLREF: tuple[float, float] = (0.02, 1.0)
DEFAULT_SOLIMP: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass
class ConvexPart:
    """One convex collision piece, tool frame, meters, MuJoCo geom-size convention."""

    kind: PartKind
    pos: np.ndarray
    quat_wxyz: np.ndarray
    size: np.ndarray
    vertices: np.ndarray | None = None  # (n, 3) for kind == "mesh"


@dataclass
class PrintPiece:
    """A separately printed body (one material) of a tool; the unit of the print estimate."""

    solid: cq.Workplane
    material: Material


@dataclass
class ToolGeometry:
    spec: ToolSpec
    solid: cq.Workplane | None  # None for the stock gripper
    parts: list[ConvexPart]
    tip: np.ndarray  # 4x4, tip in the tool frame (meters)
    contact: ContactType
    needs_support: bool = False
    material: Material = "pla"  # material of the contacting surface
    friction: tuple[float, float, float] = (0.6, 0.005, 0.0001)  # slide, torsion, roll
    solref: tuple[float, float] = DEFAULT_SOLREF
    solimp: tuple[float, float, float, float, float] = DEFAULT_SOLIMP
    pieces: list[PrintPiece] = field(default_factory=list)
    print_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # tool frame -> print frame


# ---------------------------------------------------------------- small helpers
def _translation(x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = np.array([x_mm, y_mm, z_mm]) / 1000.0
    return t


_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_z(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[cq.Workplane, ConvexPart]:
    """Axis-aligned box from mm corners -> (solid, part)."""
    lo_a, hi_a = np.array(lo, float), np.array(hi, float)
    ext, ctr = hi_a - lo_a, (hi_a + lo_a) / 2
    solid = cq.Workplane("XY").box(*ext).translate(tuple(ctr))
    return solid, ConvexPart("box", ctr / 1000.0, _IDENTITY_Q.copy(), ext / 2000.0)


def _rot_box(center: tuple[float, float, float], ext: tuple[float, float, float], yaw: float) -> tuple[cq.Workplane, ConvexPart]:
    solid = (
        cq.Workplane("XY").box(*ext).rotate((0, 0, 0), (0, 0, 1), math.degrees(yaw)).translate(center)
    )
    return solid, ConvexPart("box", np.array(center) / 1000.0, _quat_z(yaw), np.array(ext) / 2000.0)


def _cyl(radius: float, z0: float, z1: float) -> tuple[cq.Workplane, ConvexPart]:
    h = z1 - z0
    solid = cq.Workplane("XY").cylinder(h, radius).translate((0, 0, (z0 + z1) / 2))
    part = ConvexPart("cylinder", np.array([0.0, 0.0, (z0 + z1) / 2000.0]), _IDENTITY_Q.copy(),
                      np.array([radius / 1000.0, h / 2000.0]))
    return solid, part


def _sphere(radius: float, z: float) -> tuple[cq.Workplane, ConvexPart]:
    solid = cq.Workplane("XY").sphere(radius).translate((0, 0, z))
    return solid, ConvexPart("sphere", np.array([0.0, 0.0, z / 1000.0]), _IDENTITY_Q.copy(),
                             np.array([radius / 1000.0]))


def _prism(poly_xy: list[tuple[float, float]], z0: float, z1: float) -> tuple[cq.Workplane, ConvexPart]:
    """Convex polygon (mm, CCW) extruded from z0 to z1."""
    solid = cq.Workplane("XY").workplane(offset=z0).polyline(poly_xy).close().extrude(z1 - z0)
    pts = np.array([[x, y, z] for z in (z0, z1) for x, y in poly_xy]) / 1000.0
    return solid, ConvexPart("mesh", np.zeros(3), _IDENTITY_Q.copy(), np.zeros(3), vertices=pts)


def _union(solids: list[cq.Workplane]) -> cq.Workplane:
    out = solids[0]
    for s in solids[1:]:
        out = out.union(s)
    return out


def _mount() -> tuple[cq.Workplane, ConvexPart]:
    h = MOUNT_W / 2
    return _box((-h, -h, 0.0), (h, h, MOUNT_H))


# ---------------------------------------------------------------- templates
def _pad(p: dict[str, float]) -> ToolGeometry:
    w, h, t = p["w"], p["h"], p["thickness"]
    mount_s, mount_p = _mount()
    plate_s, plate_p = _box((-w / 2, -h / 2, MOUNT_H), (w / 2, h / 2, MOUNT_H + t))
    solid = _union([mount_s, plate_s])
    spec = make_spec("pad", p)
    # Printed pad-face down: the mount ledge overhangs when the pad is smaller than the mount.
    return ToolGeometry(
        spec, solid, [mount_p, plate_p], tip_transform(spec), "pad",
        needs_support=bool(w < MOUNT_W or h < MOUNT_W),
        friction=(float(p["friction"]), 0.005, 0.0001),
        pieces=[PrintPiece(solid, "pla")],
        print_rotation=np.diag([1.0, -1.0, -1.0]),
    )


def hook_dims(p: dict[str, float]) -> dict[str, float]:
    """Derived hook dimensions (mm): material thickness and barb height.

    Profile (x-z plane, extruded 20 mm along y): a shaft on the z axis from the mount to
    ``length``, a foot at the far end running to x = -depth, and a barb at the foot's end
    rising back toward the mount.  The open throat is the gap between shaft and barb.
    """
    depth, length = p["depth"], p["length"]
    wt = float(np.clip(0.3 * depth, 1.5, 3.0))
    barb = float(min(p["throat"], max(0.0, length - wt - MOUNT_H - 2.0)))
    return {"wt": wt, "barb": barb, "gap": depth - 1.5 * wt, "width": MOUNT_W}


def _hook(p: dict[str, float]) -> ToolGeometry:
    depth, length = p["depth"], p["length"]
    d = hook_dims(p)
    wt, barb, hw = d["wt"], d["barb"], d["width"] / 2
    mount_s, mount_p = _mount()
    shaft_s, shaft_p = _box((-wt / 2, -hw, MOUNT_H), (wt / 2, hw, length))
    foot_s, foot_p = _box((-depth, -hw, length - wt), (wt / 2, hw, length))
    barb_s, barb_p = _box((-depth, -hw, length - wt - barb), (-depth + wt, hw, length - wt))
    solid = _union([mount_s, shaft_s, foot_s, barb_s])
    spec = make_spec("hook", p)
    # Printed on its side (tool y -> print -z): the whole profile is an extrusion, no overhang.
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return ToolGeometry(
        spec, solid, [mount_p, shaft_p, foot_p, barb_p], tip_transform(spec), "hook",
        friction=(0.6, 0.005, 0.0001), pieces=[PrintPiece(solid, "pla")], print_rotation=rot,
    )


def knob_cup_dims(p: dict[str, float]) -> dict[str, float]:
    wall = 2.0
    r_in = p["inner_diameter"] / 2
    c = math.cos(math.pi / RING_SEGMENTS)
    return {"wall": wall, "r_in": r_in, "r_in_vertex": r_in / c, "r_out_vertex": (r_in + wall) / c,
            "ribs": float(round(p["ribs"])), "rib": 1.5}


def _knob_cup(p: dict[str, float]) -> ToolGeometry:
    depth = p["depth"]
    d = knob_cup_dims(p)
    z0, z1 = MOUNT_H, MOUNT_H + depth
    mount_s, mount_p = _mount()
    base_s, base_p = _cyl(d["r_out_vertex"], 0.0, MOUNT_H)
    solids, parts = [mount_s, base_s], [mount_p, base_p]
    n = RING_SEGMENTS

    def ring_pt(r: float, i: int) -> tuple[float, float]:
        a = 2 * math.pi * i / n
        return (r * math.cos(a), r * math.sin(a))

    # One cut for the solid (fast); the parts tile the same polygon annulus as n convex prisms.
    outer = cq.Workplane("XY").workplane(offset=z0).polyline([ring_pt(d["r_out_vertex"], i) for i in range(n)])
    inner = cq.Workplane("XY").workplane(offset=z0).polyline([ring_pt(d["r_in_vertex"], i) for i in range(n)])
    solids.append(outer.close().extrude(depth).cut(inner.close().extrude(depth)))
    for i in range(n):
        quad = [ring_pt(d["r_in_vertex"], i), ring_pt(d["r_out_vertex"], i),
                ring_pt(d["r_out_vertex"], i + 1), ring_pt(d["r_in_vertex"], i + 1)]
        parts.append(_prism(quad, z0, z1)[1])
    n_ribs = int(d["ribs"])
    for k in range(n_ribs):
        yaw = 2 * math.pi * (k + 0.5) / n_ribs
        rc = d["r_in"] - d["rib"] / 2 + 0.3  # overlaps the wall by 0.3 mm
        ctr = (rc * math.cos(yaw), rc * math.sin(yaw), (z0 + z1) / 2)
        s, part = _rot_box(ctr, (d["rib"], d["rib"], depth), yaw)
        solids.append(s)
        parts.append(part)
    solid = _union(solids)
    spec = make_spec("knob_cup", p)
    # A cup pressed on a knob turns it through torsional friction (condim=4).  The coefficient
    # (units of length) stands in for rib/knurl engagement, which a smooth MuJoCo cylinder cannot
    # form: 0.02 m bare plus 0.01 m per rib at a 20 mm bore, scaling with the bore like a contact
    # radius (D-015).  A bare disc model, mu * (2/3) * r, gives ~5 mm and cannot reach the
    # 0.05-0.25 N*m the knob range needs at a few newtons of press.
    mu = 0.9
    torsion = (0.02 + 0.01 * n_ribs) * p["inner_diameter"] / 20.0
    return ToolGeometry(
        spec, solid, parts, tip_transform(spec), "cup",
        friction=(mu, torsion, 0.0001), pieces=[PrintPiece(solid, "pla")],
    )


def _soft_tip(p: dict[str, float]) -> ToolGeometry:
    r, s = p["radius"], p["softness"]
    mount_s, mount_p = _mount()
    stalk_r = min(3.0, 0.8 * r)
    stalk_s, stalk_p = _cyl(stalk_r, MOUNT_H - 1.0, 15.0)
    ball_s, ball_p = _sphere(r, 15.0)
    tip_solid = stalk_s.union(ball_s)
    spec = make_spec("soft_tip", p)
    solref = (0.005 + 0.045 * s, 1.0)
    solimp = (0.9 - 0.4 * s, 0.95 - 0.2 * s, 0.001 + 0.004 * s, 0.5, 2.0)
    return ToolGeometry(
        spec, mount_s.union(tip_solid), [mount_p, stalk_p, ball_p], tip_transform(spec), "pad",
        needs_support=True, material="tpu", friction=(1.0, 0.01, 0.0001),
        solref=solref, solimp=solimp,
        pieces=[PrintPiece(mount_s, "pla"), PrintPiece(tip_solid, "tpu")],
    )


def _slot_inserter(p: dict[str, float]) -> ToolGeometry:
    t, w, length = p["blade_thickness"], p["width"], p["length"]
    base_w = max(MOUNT_W, w)
    base_s, base_p = _box((-MOUNT_W / 2, -base_w / 2, 0.0), (MOUNT_W / 2, base_w / 2, MOUNT_H))
    blade_s, blade_p = _box((-t / 2, -w / 2, MOUNT_H), (t / 2, w / 2, length))
    solid = base_s.union(blade_s)
    spec = make_spec("slot_inserter", p)
    return ToolGeometry(
        spec, solid, [base_p, blade_p], tip_transform(spec), "edge",
        friction=(0.3, 0.003, 0.0001), pieces=[PrintPiece(solid, "pla")],
    )


def _probe(p: dict[str, float]) -> ToolGeometry:
    mount_s, mount_p = _mount()
    rod_s, rod_p = _cyl(p["diameter"] / 2, MOUNT_H, p["length"])
    solid = mount_s.union(rod_s)
    spec = make_spec("probe", p)
    return ToolGeometry(
        spec, solid, [mount_p, rod_p], tip_transform(spec), "probe",
        friction=(0.5, 0.003, 0.0001), pieces=[PrintPiece(solid, "pla")],
    )


def _stock(p: dict[str, float]) -> ToolGeometry:
    spec = make_spec("stock", p)
    return ToolGeometry(spec, None, [], np.eye(4), "edge")


_BUILDERS = {
    "stock": _stock,
    "pad": _pad,
    "hook": _hook,
    "knob_cup": _knob_cup,
    "soft_tip": _soft_tip,
    "slot_inserter": _slot_inserter,
    "probe": _probe,
}


def make_spec(template: str, params: dict[str, float]) -> ToolSpec:
    from sea.design.space import TEMPLATE_CONTACT

    return ToolSpec(template=template, params=dict(params), contact=TEMPLATE_CONTACT[template])  # type: ignore[arg-type]


def tip_transform(spec: ToolSpec) -> np.ndarray:
    """Tip pose in the tool frame (4x4, meters), exactly as Appendix B."""
    p = spec.params
    match spec.template:
        case "pad":
            return _translation(0.0, 0.0, p["thickness"] + 12.0)
        case "hook":
            return _translation(-p["depth"], 0.0, p["length"])
        case "knob_cup":
            return _translation(0.0, 0.0, p["depth"] + 12.0)
        case "soft_tip":
            return _translation(0.0, 0.0, p["radius"] + 15.0)
        case "slot_inserter" | "probe":
            return _translation(0.0, 0.0, p["length"])
        case _:
            return np.eye(4)


def make_tool(spec: ToolSpec) -> ToolGeometry:
    """Build the template geometry for a validated ``ToolSpec``."""
    from sea.design.space import tool_errors

    errs = tool_errors(spec)
    if errs:
        raise ValueError("; ".join(errs))
    return _BUILDERS[spec.template](dict(spec.params))


# ---------------------------------------------------------------- containment (tests, checks)
def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def points_in_part(part: ConvexPart, pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of points (n, 3, meters, tool frame) inside a convex part."""
    local = (pts - part.pos) @ _quat_to_mat(part.quat_wxyz)
    s = part.size
    if part.kind == "box":
        return np.all(np.abs(local) <= s + tol, axis=1)
    if part.kind == "cylinder":
        return (np.hypot(local[:, 0], local[:, 1]) <= s[0] + tol) & (np.abs(local[:, 2]) <= s[1] + tol)
    if part.kind == "capsule":
        zc = np.clip(local[:, 2], -s[1], s[1])
        return np.linalg.norm(local - np.c_[np.zeros_like(zc), np.zeros_like(zc), zc], axis=1) <= s[0] + tol
    if part.kind == "sphere":
        return np.linalg.norm(local, axis=1) <= s[0] + tol
    from scipy.spatial import ConvexHull

    assert part.vertices is not None
    hull = ConvexHull(part.vertices)
    return np.all(local @ hull.equations[:, :3].T + hull.equations[:, 3] <= tol, axis=1)


def points_in_parts(parts: list[ConvexPart], pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(pts), dtype=bool)
    for part in parts:
        mask |= points_in_part(part, pts)
    return mask
