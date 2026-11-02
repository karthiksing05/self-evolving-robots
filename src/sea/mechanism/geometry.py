"""Geometry on real part meshes: where two parts overlap, how far apart they are, and which parts touch.

Meshes are trimesh in millimetres, each in its own rigid body's frame; a pose (4 x 4) places a body.  The
bounding volume hierarchies come from coal (hpp-fcl) and are built once per mesh, so a sweep through a
joint's range only moves transforms.  Surface crossings miss a part sealed inside another, so ``overlap``
also tests vertices for containment by their generalized winding number (Jacobson et al. 2013), which needs
no spatial index and tolerates the small holes of exported CAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import coal
import numpy as np
import trimesh


@dataclass
class Solid:
    """One part's mesh in its body's frame, with its collision structure."""

    name: str
    body: str
    mesh: trimesh.Trimesh
    bvh: coal.BVHModelOBBRSS = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.bvh = _bvh(self.mesh)


def _bvh(mesh: trimesh.Trimesh) -> coal.BVHModelOBBRSS:
    verts = coal.StdVec_Vec3s()
    for v in np.asarray(mesh.vertices, dtype=float):
        verts.append(v)
    tris = coal.StdVec_Triangle()
    for a, b, c in np.asarray(mesh.faces, dtype=int):
        tris.append(coal.Triangle(int(a), int(b), int(c)))
    out = coal.BVHModelOBBRSS()
    out.beginModel(len(mesh.faces), len(mesh.vertices))
    out.addSubModel(verts, tris)
    out.endModel()
    return out


def _tf(pose: np.ndarray) -> coal.Transform3s:
    return coal.Transform3s(np.asarray(pose[:3, :3], dtype=float), np.asarray(pose[:3, 3], dtype=float))


def clearance(a: Solid, pa: np.ndarray, b: Solid, pb: np.ndarray) -> float:
    """Shortest distance between the two surfaces (mm); 0 where they touch or cross."""
    res = coal.DistanceResult()
    return max(0.0, float(coal.distance(a.bvh, _tf(pa), b.bvh, _tf(pb), coal.DistanceRequest(), res)))


def crossings(a: Solid, pa: np.ndarray, b: Solid, pb: np.ndarray, cap: int = 4000) -> np.ndarray:
    """Points (k x 3, world frame, mm) where the two surfaces cross; empty if they do not."""
    req = coal.CollisionRequest()
    req.num_max_contacts = cap
    req.enable_contact = True
    res = coal.CollisionResult()
    if not coal.collide(a.bvh, _tf(pa), b.bvh, _tf(pb), req, res):
        return np.zeros((0, 3))
    return np.array([c.pos for c in res.getContacts()], dtype=float)


def winding(mesh: trimesh.Trimesh, pts: np.ndarray, chunk: int = 8) -> np.ndarray:
    """Generalized winding number of each point (mesh frame): about 1 inside a closed mesh, 0 outside."""
    tri = np.asarray(mesh.triangles, dtype=float)
    out = np.zeros(len(pts))
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk, None, None, :]
        a, b, c = (tri[None, :, k, :] - p[:, :, 0, :] for k in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("pti,pti->pt", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("pti,pti->pt", a, b) * lc + np.einsum("pti,pti->pt", b, c) * la
               + np.einsum("pti,pti->pt", c, a) * lb)
        out[s:s + chunk] = 2 * np.arctan2(det, den).sum(axis=1) / (4 * np.pi)
    return np.abs(out)


def _inside(a: Solid, pa: np.ndarray, b: Solid, pb: np.ndarray, n: int) -> np.ndarray:
    """Up to ``n`` of a's vertices (world frame) that lie inside b."""
    va = np.asarray(a.mesh.vertices, dtype=float)
    if len(va) > n:
        va = va[np.linspace(0, len(va) - 1, n).astype(int)]
    world = va @ pa[:3, :3].T + pa[:3, 3]
    local = (world - pb[:3, 3]) @ pb[:3, :3]
    lo, hi = b.mesh.bounds
    near = np.all((local >= lo) & (local <= hi), axis=1)
    if not near.any():
        return np.zeros((0, 3))
    hit = np.zeros(len(local), dtype=bool)
    hit[near] = winding(b.mesh, local[near]) > 0.5
    return world[hit]


def overlap(a: Solid, pa: np.ndarray, b: Solid, pb: np.ndarray, samples: int = 400) -> np.ndarray:
    """Points marking where two parts occupy the same space: their surface crossings, or, if a part is
    sealed inside the other, its vertices that lie within."""
    pts = crossings(a, pa, b, pb)
    if len(pts):
        return pts
    return np.vstack([_inside(a, pa, b, pb, samples), _inside(b, pb, a, pa, samples)])


def extent(pts: np.ndarray) -> list[float]:
    """The size (mm) of the box around a set of points, per axis."""
    return [round(float(x), 1) for x in (np.ptp(pts, axis=0) if len(pts) else np.zeros(3))]


def from_axis(pts: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Each point's distance (mm) from a line."""
    u = axis / np.linalg.norm(axis)
    d = pts - origin
    return np.linalg.norm(d - np.outer(d @ u, u), axis=1)


def grounded(solids: list[Solid], poses: dict[str, np.ndarray], base: str, touch: float = 0.5) -> list[str]:
    """Parts that no chain of touching parts (within ``touch`` mm) joins to the ``base`` part."""
    names = [s.name for s in solids]
    reached, todo = {base}, [base]
    by = {s.name: s for s in solids}
    while todo:
        cur = by[todo.pop()]
        for s in solids:
            if s.name not in reached and clearance(cur, poses[cur.body], s, poses[s.body]) <= touch:
                reached.add(s.name)
                todo.append(s.name)
    return [n for n in names if n not in reached]
