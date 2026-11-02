"""The printed finger pads as they would be built (docs/design-language.md), drawn for pictures and never simulated.

The simulator's pads are the boxes ``design/fingers.py`` builds.  This module draws the part those boxes stand for,
and keeps every surface an object can touch where the simulator has it:

- **Lands.** Each face box's front (both halves of a V groove) and the lip's top and front are coplanar with their
  simulated faces and end where those end, outside the jaw.  The underside lies in the boxes' bottom plane, so the
  pad reaches exactly as low as simulated.  ``width_mm`` is the land's width; the rounds that soften its sides lie
  outside it.
- **Shoe.** A PLA shoe slides over the stock jaw's tapered tip.  Its socket is the jaw's own CAD (the convex hull of
  the tip, grown by the slip-fit clearance), its walls wrap the tip, and two M2 cap screws (two of the bill of
  materials' four) hold it from behind.  Below the tip it tapers to the land; it is printed lying on its back.
- **Skin.** With ``pad_tpu_mm`` the land's outer layer and the lip are TPU, printed onto the shoe.

Where a tilted pad's simulated face runs into the jaw (the moving pad past about atan(thickness / 10 mm)), the land
stops at the jaw: the buried part has no surface to draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cadquery as cq
import numpy as np
import trimesh

from sea.contracts import FingerSpec
from sea.design import fingers as F
from sea.design import language as L
from sea.design.cad import CadPart
from sea.design.language import tessellate as to_mesh

V = cq.Vector
R_EDGE = L.EDGE_R["printed"]
WALL = 1.6  # mm, four perimeters around the host
ARC = 8  # segments per round
SCREW = (1.0, 1.9, 2.0, 3.0)  # M2 cap screw: shank radius, head radius, head height, thread engaged in the jaw


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
    shift = -F.link_vec(spec) if spec.jaw == "parallelogram" else np.zeros(3)
    moving = PadFrame("jaw_moving", F.moving_body(spec), rm, 1000 * (corner + tilt @ np.array([0.0, -lm, 0.0]) + shift),
                      1000 * lm)
    return [fixed, moving]


def boxes(spec: FingerSpec, fr: PadFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The simulated boxes in the pad frame (mm): centre, half sizes, rotation."""
    return [(1000 * c, 1000 * h, r) for c, h, r in F._face_boxes(spec, fr.length / 1000)]


# ---------------------------------------------------------------- the shoe's sections
def _host_extent(host: trimesh.Trimesh, ys: list[float]) -> tuple[float, float] | None:
    """Over the planes y = each of ``ys`` (pad frame), the host's rearmost x and widest |z| near the pad."""
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


def _profile(t: float, w: float, g: float, r: float, xb: float) -> list[tuple[float, float]]:
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


def _hull(host: trimesh.Trimesh, top: float, grid: float = 0.25) -> trimesh.Trimesh:
    """The host below ``top`` (pad frame) as a convex mesh grown by the slip-fit clearance.  The tip's points are
    snapped to ``grid`` (the STL is dense where it is round) and the hull is grown by that error too, so it still
    holds the whole tip."""
    v = np.asarray(host.vertices)
    near = (v[:, 1] <= top) & (np.abs(v[:, 0]) < 40) & (np.abs(v[:, 2]) < 40)
    pts = [v[near]]
    sec = host.section(plane_origin=[0.0, top, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if sec is not None:
        pts.append(np.asarray(sec.vertices))
    hull = trimesh.convex.convex_hull(np.unique(np.round(np.vstack(pts) / grid) * grid, axis=0))
    grow = (L.CLEARANCE + grid * math.sqrt(3) / 2) / 0.85  # vertex normals lean up to ~30 deg off their faces
    return trimesh.convex.convex_hull(hull.vertices + grow * hull.vertex_normals)


def socket(host: trimesh.Trimesh, top: float, grid: float = 0.25) -> cq.Solid:
    """``_hull`` as a solid, to cut from the shoe."""
    faces = [cq.Face.makeFromWires(cq.Wire.makePolygon([V(*p) for p in tri], close=True))
             for tri in _hull(host, top, grid).triangles]
    return cq.Solid.makeSolid(cq.Shell.makeShell(faces))


def _slab(t: float, w: float, g: float, depth: float, h: float) -> cq.Solid:
    """Everything within ``depth`` behind the land (and in front of it), over the pad's height."""
    if g <= 0:
        return cq.Solid.makeBox(depth + 20, h + 2, 200, V(t - depth, -1, -100))
    half = w / 2
    out = None
    for s in (1.0, -1.0):
        a, b = np.array([t, s * half]), np.array([t - g, 0.0])
        u = (a - b) / np.linalg.norm(a - b)
        n = np.array([u[1], -u[0]])
        n = n if n[0] > 0 else -n  # out of the land
        quad = [tuple(p) for p in (b - n * depth - u * 4, a - n * depth + u * 20, a + n * 20 + u * 20, b + n * 20 - u * 4)]
        prism = cq.Workplane("XZ").polyline(quad).close().extrude(-(h + 2)).translate(V(0, -1, 0)).val()
        piece = prism.intersect(cq.Solid.makeBox(100, h + 4, 100, V(-50, -2, 0 if s > 0 else -100)))
        out = piece if out is None else out.fuse(piece)
    return out


def _screws(sections: list[tuple[float, float]], ys: list[float]) -> tuple[cq.Shape, cq.Shape]:
    """Two M2 cap screws through the shoe's back into the jaw, square to the back: the screws, and their holes."""
    yb, xb = np.array([s[0] for s in sections]), np.array([s[1] for s in sections])
    r, rh, hh, bite = SCREW
    screws, holes = [], []
    for y in ys:
        k = int(np.clip(np.searchsorted(yb, y) - 1, 0, len(yb) - 2))
        d = np.array([xb[k + 1] - xb[k], yb[k + 1] - yb[k]])
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])  # out of the back (toward -x: the sections climb in y)
        x = float(np.interp(y, yb, xb))
        p, nn = V(x, y, 0.0), V(n[0], n[1], 0.0)
        head = cq.Solid.makeCylinder(rh, hh, p, nn)
        key = cq.Workplane(cq.Plane(origin=p + nn * hh, xDir=V(0, 0, 1), normal=nn)).polygon(6, 1.7).extrude(-1.2).val()
        shank = cq.Solid.makeCylinder(r, WALL + L.CLEARANCE + bite, p, -nn)
        screws.append(head.cut(key).fuse(shank))
        holes.append(cq.Solid.makeCylinder(r + 0.2, WALL + L.CLEARANCE + 2, p + nn, -nn))
    return cq.Compound.makeCompound(screws), cq.Compound.makeCompound(holes)


def pad(spec: FingerSpec, fr: PadFrame, host: trimesh.Trimesh) -> list[CadPart]:
    """One pad in its own frame (mm): the shoe (PLA), the skin (TPU, if any) and its screws.  ``host`` is the jaw's
    mesh in the pad frame."""
    t, tpu, w, g = 1000 * F.thickness(spec), spec.pad_tpu_mm, 1000 * F.width(spec), spec.groove_mm
    length, h = fr.length, fr.length + 1000 * F.COVER
    bx = boxes(spec, fr)
    corners = np.vstack([c + (np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * hf) @ r.T
                         for c, hf, r in bx])
    xb0 = min(0.0, float(corners[:, 0].min()))  # the boxes' rearmost point: the underside covers theirs
    ys = [0.0] + ([length] if length > 0.5 else []) + [y for y in np.arange(length + 2.0, h - 0.9, 2.0)] + [h]
    sections, wires = [], []
    half_g = g / math.hypot(g, w / 2) if g > 0 else 0.0
    hull = _hull(host, h + 3.0)  # the socket: walls are measured from it, so none is thinner than WALL
    for y in ys:
        ext = _host_extent(hull, [min(max(y + dy, 0.0), h) for dy in (-1.0, 0.0, 1.0)])
        xb, r = xb0, R_EDGE
        if ext is not None:
            xb = min(xb0, ext[0] - WALL)
            r = max(R_EDGE, (ext[1] + WALL - w / 2) / (1 + half_g))
        sections.append((y, xb))
        wires.append(cq.Wire.makePolygon([V(x, y, z) for x, z in _profile(t, w, g, r, xb)], close=True))
    shoe = cq.Solid.makeLoft(wires, True)
    if spec.lip_mm > 0:
        lip = spec.lip_mm
        shoe = shoe.fuse(cq.Solid.makeBox(t + lip, F.LIP_THICK * 1000, w, V(t / 2, 0, -w / 2)))
    shoe = shoe.cut(socket(host, h + 3.0))
    screws, holes = _screws(sections, [length + 0.3 * F.COVER * 1000, length + 0.72 * F.COVER * 1000])
    shoe = shoe.cut(holes)
    side = "fixed" if fr.kind == "jaw_fixed" else "moving"
    out = []
    if tpu > 0:
        slab = _slab(t, w, g, tpu, h)
        if spec.lip_mm > 0:
            slab = slab.fuse(cq.Solid.makeBox(t + spec.lip_mm + 5, F.LIP_THICK * 1000 + 1, w + 2,
                                              V(t - tpu, -1, -w / 2 - 1)))
        out.append(CadPart(f"{side} pad shoe", "pads", to_mesh(shoe.cut(slab)), "printed"))
        out.append(CadPart(f"{side} pad skin", "pads", to_mesh(shoe.intersect(slab)), "tpu"))
    else:
        out.append(CadPart(f"{side} pad shoe", "pads", to_mesh(shoe), "printed"))
    out.append(CadPart(f"{side} pad screws", "pads", to_mesh(screws), "metal"))
    return out


def _to(mesh: trimesh.Trimesh, rot: np.ndarray, origin: np.ndarray) -> trimesh.Trimesh:
    return mesh.copy().apply_transform(_pose(rot, origin))


def draw(spec: FingerSpec, hosts: dict[str, trimesh.Trimesh]) -> list[tuple[str, CadPart]]:
    """Both pads in their jaw bodies' frames (mm), as (body, part).  ``hosts`` maps a body to the mesh of the jaw
    its pad sits on, in that body's frame (mm)."""
    if not F.has_pads(spec):
        return []
    out = []
    for fr in frames(spec):
        host = _to(hosts[fr.body], fr.rot.T, -fr.rot.T @ fr.origin)
        for p in pad(spec, fr, host):
            out.append((fr.body, CadPart(p.name, p.slot, _to(p.mesh, fr.rot, fr.origin), p.material)))
    return out


def _fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The rigid transform (4 x 4) that carries points ``src`` onto ``dst`` (Kabsch)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    s = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rot = vt.T @ s @ u.T
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, cd - rot @ cs
    return out


def jaws(parts: list[CadPart], spec: FingerSpec) -> dict[str, tuple[np.ndarray, trimesh.Trimesh]]:
    """For each jaw body of a posed drawing with the simulated pad boxes (``worlds/cad.gripper``): the body's pose
    (4 x 4, mm), read off its own pad box, and the mesh of the jaw that box sits on, in the body's frame (mm)."""
    from sea.control.frames import quat_wxyz_to_mat

    geoms = {g.name: g for g in F.pad_geoms(spec)}
    drawn = {p.name: p for p in parts if p.slot == "pads" and p.name in geoms}
    rest = [p for p in parts if p.slot != "pads"]
    out = {}
    for fr in frames(spec):
        name = next((n for n in drawn if geoms[n].body == fr.body and geoms[n].kind == fr.kind), None)
        if name is None:
            continue
        g, box = geoms[name], drawn[name]
        local = trimesh.creation.box(extents=2 * np.asarray(g.half))
        local.apply_transform(_pose(quat_wxyz_to_mat(g.quat), np.asarray(g.pos)))
        pose = _fit(1000 * local.vertices, box.mesh.vertices)
        out[fr.body] = (pose, _host(box.mesh, rest).mesh.copy().apply_transform(np.linalg.inv(pose)))
    return out


def dress(parts: list[CadPart], spec: FingerSpec) -> list[CadPart]:
    """A posed drawing of the jaws (``worlds/cad.gripper``) with its simulated pad boxes replaced by the drawn pads."""
    fit = jaws(parts, spec)
    if len(fit) < 2:
        return parts
    out = [p for p in parts if p.slot != "pads"]
    for body, p in draw(spec, {b: host for b, (_, host) in fit.items()}):
        out.append(CadPart(p.name, p.slot, p.mesh.copy().apply_transform(fit[body][0]), p.material))
    return out


def _pose(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, pos
    return out


def _host(mesh: trimesh.Trimesh, parts: list[CadPart]) -> CadPart:
    """The part a pad sits on: the one nearest it."""
    from sea.mechanism.geometry import Solid, clearance

    eye, near = np.eye(4), Solid("pad", "pad", mesh)
    return min(parts, key=lambda p: clearance(near, eye, Solid(p.name, "", p.mesh), eye))


def assemblies(parts: list[CadPart]) -> list[L.Assembly]:
    """Each drawn pad with the jaw it sits on, for ``language.audit``: the shoe is a slip fit over the jaw's tip, the
    skin is printed onto the shoe, and the screws pass the shoe into holes drilled in the jaw."""
    rest = [p for p in parts if p.slot != "pads"]
    out = []
    for side in ("fixed", "moving"):
        mine = {p.name.removeprefix(f"{side} pad "): p for p in parts if p.name.startswith(f"{side} pad ")}
        if not mine:
            continue
        host = _host(mine["shoe"].mesh, rest)
        asm = L.Assembly(base=host.name)
        asm.add(L.Member(host.name, host, "the stock jaw, from its maker's CAD", stock=True))
        for p in mine.values():
            asm.add(L.Member(p.name, p))
        shoe, screws = f"{side} pad shoe", f"{side} pad screws"
        asm.join(shoe, host.name)
        asm.join(screws, shoe, "joint")
        asm.join(screws, host.name, "joint")
        if "skin" in mine:
            asm.join(f"{side} pad skin", shoe, "joint")
        out.append(asm)
    return out


def refusals(spec: FingerSpec) -> list[str]:
    """Why a pad parameter set cannot be built to the design language; empty when it can.  These are the rules a
    proposal would be refused by (docs/design-language.md); nothing calls them in the judge yet."""
    if not F.has_pads(spec):
        return []
    out = []
    tpu, back = spec.pad_tpu_mm, 1000 * F.THICK - spec.groove_mm - L.CLEARANCE  # PLA from the groove to the socket
    if 0 < tpu < L.WALL["tpu"]:
        out.append(f"a {tpu:g} mm TPU skin is thinner than TPU's {L.WALL['tpu']:g} mm wall")
    if back < L.WALL["printed"]:
        out.append(f"a {spec.groove_mm:g} mm groove leaves {max(back, 0.0):.1f} mm of PLA between its floor and the "
                   f"jaw's socket (at least {L.WALL['printed']:g})")
    return out
