"""The design language for printed and cast parts (docs/design-language.md): the numbers every generator builds
to, and the laws a drawn assembly is measured by.

Generators (``design/pad_cad.py``, ``worlds/rig_cad.py``) read the constants and follow the rules by construction.
``audit`` measures an ``Assembly``: parts that no chain of contacts or joints carries to the base, parts that occupy
the same space without a declared joint, and per part the length of sharp edge, the overhang in its print
orientation and the thinnest wall.  Nothing here decides a verdict: like ``sea.mechanism`` it is tooling until its
laws join the judge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh

from sea.design.cad import CadModel, CadPart, View

# ---------------------------------------------------------------- the numbers generators build to
# FDM with a 0.4 mm nozzle: a wall is whole perimeters (3 for PLA, 2 for TPU); a cast wall must survive demolding.
WALL = {"printed": 1.2, "tpu": 0.8, "silicone": 2.0}  # mm, the thinnest wall of each material
FEATURE = {"printed": 0.8, "tpu": 1.0, "silicone": 1.5}  # mm, the smallest pin, rib or slot
EDGE_R = {"printed": 1.0, "tpu": 1.0, "silicone": 1.5}  # mm, the least radius on an exposed edge that is not a land
CLEARANCE = 0.2  # mm, a printed socket over its host: a slip fit
OVERHANG_DEG = 45.0  # from vertical, unsupported, in the stated print orientation
GRID = 5.0  # mm, where fixed parts stand on a base plate
SHARP_DEG = 60.0  # a convex edge whose faces turn by more than this is sharp (a tessellated 1 mm round turns by ~20)
TOUCH = 0.5  # mm, parts this close are in contact

JointKind = Literal["joint", "contact", "guide"]
# joint: the two share material by design (a pin, a screw in its thread, a horn on its spline, a bonded insert);
# contact: they touch (a foot on the plate, a follower on its cam); guide: a rod passes through a hole with clearance.


def tessellate(shape, tol: float = 0.1, angle: float = 0.3) -> trimesh.Trimesh:
    """A CadQuery shape as a closed mesh (mm), read straight from OCC's triangulation (CadQuery's own loop builds a
    Python object per node, which dominates a bench of filleted parts)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(shape.wrapped, tol, False, angle, True)
    verts, faces, off = [], [], 0
    for f in shape.Faces():
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(f.wrapped, loc)
        if poly is None:
            continue
        trsf = loc.Transformation()
        n = poly.NbNodes()
        pts = (poly.Node(i).Transformed(trsf) for i in range(1, n + 1))
        verts.append(np.array([(p.X(), p.Y(), p.Z()) for p in pts], dtype=float))
        tri = poly.Triangles()
        t = np.array([(tri.Value(i).Value(1), tri.Value(i).Value(2), tri.Value(i).Value(3))
                      for i in range(1, poly.NbTriangles() + 1)], dtype=np.int64) - 1 + off
        faces.append(t[:, [0, 2, 1]] if f.wrapped.Orientation() == TopAbs_REVERSED else t)
        off += n
    mesh = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    mesh.merge_vertices(digits_vertex=6)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


@dataclass
class Member:
    key: str  # unique in the assembly
    part: CadPart  # name (shown in the legend), slot, mesh (mm), material
    role: str = ""  # what it does, in a few words: the legend's line
    print_up: tuple[float, float, float] | None = None  # printed parts: the build direction in the model frame
    section: trimesh.Trimesh | None = None  # drawn instead of the whole in a section view (empty: not drawn)
    stock: bool = False  # drawn from its maker's CAD (a stock jaw): the generators' form rules do not apply


@dataclass
class Assembly:
    members: list[Member] = field(default_factory=list)
    joints: list[tuple[str, str, JointKind]] = field(default_factory=list)
    base: str = "base plate"

    def add(self, m: Member) -> str:
        keys = {x.key for x in self.members}
        k, n = m.key, 1
        while k in keys:
            n += 1
            k = f"{m.key} {n}"
        m.key = k
        self.members.append(m)
        return k

    def join(self, a: str, b: str, kind: JointKind = "contact") -> None:
        self.joints.append((a, b, kind))

    def model(self, views: list[View], tile: tuple[int, int], section: bool = True) -> CadModel:
        """The picture: every member whole, or in a section view its ``section`` mesh where it has one."""
        parts = []
        for m in self.members:
            p = m.part
            if section and m.section is not None:
                if not len(m.section.faces):
                    continue
                p = CadPart(p.name, p.slot, m.section, p.material)
            parts.append(p)
        return CadModel(parts, views, tile)


# ---------------------------------------------------------------- the laws, measured
def floating(asm: Assembly, touch: float = TOUCH) -> list[str]:
    """Members that no chain of parts touching within ``touch`` mm carries to the base (``mechanism.geometry
    .grounded``).  A declared joint carries nothing by being declared: its parts must touch (``interference``
    reports those that do not)."""
    from sea.mechanism.geometry import Solid, grounded

    solids = [Solid(m.key, "drawn", m.part.mesh) for m in asm.members]
    return grounded(solids, {"drawn": np.eye(4)}, asm.base, touch)


def _near(a, b, touch: float) -> bool:
    lo = np.maximum(a.mesh.bounds[0], b.mesh.bounds[0])
    hi = np.minimum(a.mesh.bounds[1], b.mesh.bounds[1])
    return bool(np.all(lo <= hi + touch))


def _deep(a: trimesh.Trimesh, b: trimesh.Trimesh, depth: float, per_mm2: float = 4.0) -> bool:
    """Whether two parts share volume more than ``depth`` deep: a point of either's surface, moved ``depth`` into it,
    lies inside the other, clear of its surface by more than the tessellation's 0.1 mm.  Surfaces that only touch (a
    part pressed against another, two halves meeting at a seam) do not."""
    from sea.mechanism.geometry import winding

    probe = 0.15 * np.vstack([np.eye(3), -np.eye(3)])
    for p, q in ((a, b), (b, a)):
        n = int(np.clip(p.area * per_mm2, 500, 40000))
        pts, idx = trimesh.sample.sample_surface(p, n, seed=0)
        inner = pts - depth * p.face_normals[idx]
        lo, hi = q.bounds
        inner = inner[np.all((inner >= lo) & (inner <= hi), axis=1)]
        if not len(inner):
            continue
        inner = inner[winding(q, inner) > 0.5]
        if len(inner) and (winding(q, (inner[:, None, :] + probe).reshape(-1, 3)).reshape(-1, 6) > 0.5).all(1).any():
            return True
    return False


def interference(asm: Assembly, touch: float = TOUCH) -> dict[str, list]:
    """``undeclared``: pairs that share volume more than ``touch`` deep with no declared joint or contact between
    them (a guided rod that cuts into its guide counts; parts that only touch do not); ``loose``: declared joints
    and contacts whose parts do not touch."""
    from sea.mechanism.geometry import Solid, clearance, extent, overlap

    solids = {m.key: Solid(m.key, m.key, m.part.mesh) for m in asm.members}
    eye = np.eye(4)
    allowed = {frozenset((a, b)) for a, b, kind in asm.joints if kind in ("joint", "contact")}
    keys = list(solids)
    bad = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if frozenset((a, b)) in allowed or not _near(solids[a], solids[b], 0.0):
                continue
            pts = overlap(solids[a], eye, solids[b], eye)
            if len(pts) and _deep(solids[a].mesh, solids[b].mesh, touch):
                bad.append([a, b, extent(pts)])
    loose = [[a, b, round(clearance(solids[a], eye, solids[b], eye), 2)] for a, b, kind in asm.joints
             if kind in ("joint", "contact") and a in solids and b in solids
             and clearance(solids[a], eye, solids[b], eye) > touch]
    return {"undeclared": bad, "loose": loose}


def sharp_edge_mm(mesh: trimesh.Trimesh, deg: float = SHARP_DEG,
                  up: tuple[float, float, float] | None = None) -> float:
    """Length (mm) of convex edges whose two faces turn by more than ``deg``: edges no fillet or chamfer softens.
    With a print direction ``up``, the edges of the first layer, on the bed, do not count."""
    angles = mesh.face_adjacency_angles
    convex = mesh.face_adjacency_convex
    edges = mesh.vertices[mesh.face_adjacency_edges]
    length = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
    sharp = convex & (angles > math.radians(deg))
    if up is not None:
        u = np.asarray(up, dtype=float) / np.linalg.norm(up)
        sharp &= ~np.all(edges @ u < float((mesh.vertices @ u).min()) + 0.05, axis=1)
    return float(length[sharp].sum())


def overhang_fraction(mesh: trimesh.Trimesh, up: tuple[float, float, float], deg: float = OVERHANG_DEG) -> float:
    """Share of the surface that faces down more steeply than ``deg`` from vertical when printed with ``up`` as the
    build direction, the faces on the bed excepted."""
    from sea.design.build import overhang_area

    u = np.asarray(up, dtype=float) / np.linalg.norm(up)
    rot = trimesh.geometry.align_vectors(u, [0.0, 0.0, 1.0])[:3, :3]
    return overhang_area(mesh, rot, 90.0 - deg) / max(float(mesh.area), 1e-9)


def thinnest_wall(mesh: trimesh.Trimesh, n: int = 400) -> float:
    """1st percentile of the local wall thickness (mm), by inward rays from the surface."""
    from sea.design.build import wall_thickness

    t = wall_thickness(mesh, n=n)
    t = t[np.isfinite(t)]
    return float(np.percentile(t, 1)) if len(t) else float("nan")


def form(m: Member) -> dict[str, str | float | None]:
    """The form laws of one member: sharp edge (mm), overhang share in its print orientation, thinnest wall (mm)."""
    mesh = m.part.mesh
    out: dict[str, str | float | None] = {"material": m.part.material,
                                          "sharp_mm": round(sharp_edge_mm(mesh, up=m.print_up), 1),
                                          "overhang": None, "wall_p01_mm": None}
    if m.print_up is not None:
        out["overhang"] = round(overhang_fraction(mesh, m.print_up), 3)
    if m.part.material in WALL:
        out["wall_p01_mm"] = round(thinnest_wall(mesh), 2)
    return out


def audit(asm: Assembly, forms: bool = True) -> dict:
    """Every law on one assembly: what floats, what interferes, and the form of each member."""
    out: dict = {"members": len(asm.members), "joints": len(asm.joints), "floating": floating(asm)}
    out |= interference(asm)
    if forms:
        out["form"] = {m.key: form(m) for m in asm.members if not m.stock}
    return out
