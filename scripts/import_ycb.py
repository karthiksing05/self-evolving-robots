"""Import YCB benchmark objects as collision-ready assets for the tabletop grasp world.

    uv run python scripts/import_ycb.py                                 # every object in OBJECTS
    uv run python scripts/import_ycb.py --only tennis_ball,golf_ball    # just these (yaml entries merge)

Each object's Google 16k scan (YCB object and model set, Calli et al. 2015, CC BY 4.0) is
downloaded once into ``.cache/ycb/`` (the Berkeley Poisson mesh if there is no Google scan; the
object is dropped if neither exists) and put in its resting pose on the table.  The jaws come
straight down and close along world y, so the object is then turned about z until the short side
of its footprint's minimum-area rectangle lies along y: the y extent is the grasp width.  Object
frame: bottom at z = 0, footprint bounding-box centre at the origin.

``assets/ycb/<name>.npz`` holds ``part_<i>`` (convex collision pieces: the hull alone when the
scan fills 96% of it, else CoACD as in ``decompose_jaws.py``; vertices only, since MuJoCo collides
each mesh by its hull, at most 256 each), ``visual_v``/``visual_f`` (at most 3000 faces, by vertex
clustering: no decimation library is a dependency), ``rgba`` (the texture's area-weighted mean),
``com`` and ``inertia`` (about the COM, object axes) of a uniform solid of the scanned volume at
the published mass, and ``extents_mm`` (width y, depth x, height z).  ``configs/ycb_objects.yaml``
gets one entry per object and ``assets/ycb/LICENSE.md`` the attribution.  Every object is then
dropped 1 mm onto a plane in MuJoCo for 1 s and the summary table is printed.
"""

from __future__ import annotations

import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coacd
import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "ycb"
OUT = ROOT / "assets" / "ycb"
CONFIG = ROOT / "configs" / "ycb_objects.yaml"
S3 = "https://ycb-benchmarks.s3.amazonaws.com/data"

MAX_HEIGHT_MM = 85.0  # a straight-down SO-101 tool point tops out near z = 0.095 m: taller cannot be lifted 30 mm
MAX_WIDTH_MM = 90.0  # wider than this is flagged "too wide" in the yaml
MAX_PART_VERTS = 256
MAX_VISUAL_FACES = 3000
DIMS_TOL = 0.10  # measured vs published dimension, relative

CONVEX_FILL = 0.96  # a scan filling this much of its hull is one convex piece; the rest is surface texture
DEFAULT = (0.05, 16)  # CoACD (concavity threshold, max hulls)
CAN = (0.10, 4)  # cans are convex but for the rims; a handful of pieces
SHELL = (0.03, 32)  # an open cup: enough wall pieces to keep the cavity empty


@dataclass(frozen=True)
class Obj:
    """One row of Calli et al. 2015, Table II, plus how it rests and what it is made of.

    ``pose``: ``stable`` is the most probable resting pose of the convex hull; ``upright`` the
    stable pose nearest the scan's own +z (the scanner's turntable pose, base down); ``lying`` the
    most probable stable pose with the scan's +z horizontal; ``low_narrow`` the stable pose no
    taller than MAX_HEIGHT_MM with the smallest grasp width.  ``dims`` are as published: three
    edges, (diameter, length), or one diameter/edge.
    """

    short: str
    ycb_id: str
    name: str
    mass_g: float
    dims: tuple[float, ...]
    pose: str
    material: str
    friction: float
    round: bool = False
    decomposition: tuple[float, int] = DEFAULT
    remark: str = ""


OBJECTS = [
    Obj("gelatin_box", "009_gelatin_box", "gelatin box", 97, (28, 85, 73), "stable", "cardboard", 0.5),
    Obj("pudding_box", "008_pudding_box", "pudding box", 187, (35, 110, 89), "low_narrow", "cardboard", 0.5),
    Obj("sugar_box", "004_sugar_box", "sugar box", 514, (38, 89, 175), "lying", "cardboard", 0.5),
    Obj("potted_meat_can", "010_potted_meat_can", "potted meat can", 370, (50, 97, 82), "upright", "metal", 0.35,
        decomposition=CAN),
    Obj("mustard_bottle", "006_mustard_bottle", "mustard bottle", 431, (50, 85, 175), "lying", "plastic", 0.4),
    Obj("foam_brick", "061_foam_brick", "foam brick", 28, (50, 75, 50), "stable", "foam", 0.8),
    Obj("wood_block_cube", "070-b_colored_wood_blocks", "wood block (cube)", 10.8, (26,), "stable", "wood", 0.5,
        remark="070-a's scan is the whole block set (142x141x165 mm); 070-b, a single cube, is used"),
    Obj("strawberry", "012_strawberry", "plastic strawberry", 18, (43.8, 55), "stable", "plastic", 0.5, round=True),
    Obj("banana", "011_banana", "plastic banana", 66, (36, 190), "stable", "plastic", 0.5),
    Obj("tomato_soup_can", "005_tomato_soup_can", "tomato soup can", 349, (66, 101), "lying", "metal", 0.35,
        round=True, decomposition=CAN),
    Obj("tuna_fish_can", "007_tuna_fish_can", "tuna fish can", 171, (85, 33), "upright", "metal", 0.35,
        decomposition=CAN),
    Obj("tennis_ball", "056_tennis_ball", "tennis ball", 58, (64.7,), "stable", "felt", 0.7, round=True),
    Obj("racquetball", "057_racquetball", "racquetball", 41, (55.3,), "stable", "rubber", 0.9, round=True),
    Obj("golf_ball", "058_golf_ball", "golf ball", 46, (42.7,), "stable", "plastic", 0.4, round=True),
    Obj("lemon", "014_lemon", "plastic lemon", 29, (54, 68), "stable", "plastic", 0.5, round=True),
    Obj("plum", "018_plum", "plastic plum", 25, (52,), "stable", "plastic", 0.5, round=True),
    Obj("apple", "013_apple", "plastic apple", 68, (75,), "stable", "plastic", 0.5, round=True),
    Obj("large_marker", "040_large_marker", "large marker", 16, (19, 119), "stable", "plastic", 0.4),
    Obj("flat_screwdriver", "044_flat_screwdriver", "flat screwdriver", 70, (23, 198), "stable", "plastic", 0.4),
    Obj("cup", "065-a_cups", "stacking cup (smallest)", 13, (55, 60), "upright", "plastic", 0.4,
        decomposition=SHELL),
    Obj("dice", "062_dice", "dice", 5.2, (16.2,), "stable", "plastic", 0.4),
]

FALLBACK_RGB = {"cardboard": (0.76, 0.62, 0.42), "metal": (0.7, 0.7, 0.72), "plastic": (0.8, 0.3, 0.2),
                "foam": (0.85, 0.3, 0.25), "wood": (0.75, 0.55, 0.3), "felt": (0.8, 0.9, 0.2),
                "rubber": (0.2, 0.35, 0.8)}

YAML_HEADER = f"""\
# YCB objects for the tabletop grasp world.  Written by scripts/import_ycb.py; re-run it, do not hand-edit.
# Masses and published_dims_mm: Calli et al. 2015, "Benchmarking in Manipulation Research: The YCB Object
# and Model Set and Benchmarking Protocols", arXiv:1502.03143, Table II.
# Meshes: the YCB Google 16k scans (ycb-benchmarks.s3.amazonaws.com/data/google/), licensed CC BY 4.0 per
# the YCB data page (ycb-benchmarks.s3.amazonaws.com/index.html, checked 2026-11-01); see
# assets/ycb/LICENSE.md.  The assets are processed derivatives (convex pieces, decimated visuals).
#
# Object frame: bottom at z = 0, footprint bounding-box centre at the origin, the footprint's narrow side
# along y (the jaws close along y).  width_mm = y extent (the grasp width), depth_mm = x, height_mm = z,
# all measured on the scan, not copied from the paper; note lists differences over {DIMS_TOL:.0%}.
# height_ok: height_mm <= {MAX_HEIGHT_MM:g} (a straight-down SO-101 tool point tops out near z = 0.095 m, so a
# taller object cannot be lifted 30 mm from above).  note "too wide": width_mm > {MAX_WIDTH_MM:g}.
# Mass properties (in the npz) are a uniform solid of the scanned volume at the published mass: close for
# filled cans and boxes, wrong for hollow objects (the tennis ball and racquetball are shells, true
# inertia about 5/3 of this; the cup's scan is its wall, so its distribution is roughly right).
# round: rolls when nudged (balls, round fruit, the soup can on its side).
"""

LICENSE = """\
# YCB object assets: attribution and licence

The files in this directory are derived from the YCB Object and Model Set:

> B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel, A. M. Dollar. "Benchmarking in
> Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols."
> arXiv:1502.03143 (2015); IEEE Robotics & Automation Magazine 22 (2015) 36-52,
> doi:10.1109/MRA.2015.2448951.

Source data: the Google 16k scans at https://ycb-benchmarks.s3.amazonaws.com/data/google/
(project site https://www.ycbbenchmarks.com/).  Masses are from Table II of the paper above.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/.  The YCB data page
(https://ycb-benchmarks.s3.amazonaws.com/index.html, checked 2026-11-01) states "License for the
data set: Creative Commons Attribution 4.0 International (CC BY 4.0)" and "License for the code:
MIT license".  ycbbenchmarks.com itself returned HTTP 500 that day.

Changes: these are processed derivatives, not the original scans, produced by
`scripts/import_ycb.py`.  Each scan was re-posed (resting on the table, narrow side along y,
bottom at z = 0), reduced to convex pieces (`part_<i>`: the convex hull for nearly convex objects,
else a CoACD decomposition; hull vertices, thinned to at most 256),
decimated to at most 3000 faces by vertex clustering (`visual_v`, `visual_f`), reduced to one
average texture colour (`rgba`), and given the mass properties of a uniform solid at the published
mass (`com`, `inertia`).  The original meshes and textures are not redistributed here.
"""


# ---------------------------------------------------------------------------------------------- source


def fetch(obj: Obj) -> tuple[Path, Path | None] | None:
    """(geometry mesh, textured obj or None) in the cache, downloading once; None if neither source has it."""
    sources = [(f"{S3}/google/{obj.ycb_id}_google_16k.tgz", "google_16k"),
               (f"{S3}/berkeley/{obj.ycb_id}/{obj.ycb_id}_berkeley_meshes.tgz", "poisson")]
    for url, sub in sources:
        folder = CACHE / obj.ycb_id / sub
        if not (folder / "nontextured.ply").exists():
            archive = CACHE / url.rsplit("/", 1)[1]
            if not archive.exists():
                try:
                    CACHE.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(url, archive.with_suffix(".part"))
                except urllib.error.HTTPError:
                    continue
                archive.with_suffix(".part").rename(archive)
            with tarfile.open(archive) as tar:
                tar.extractall(CACHE, filter="data")
        if (folder / "nontextured.ply").exists():
            textured = folder / "textured.obj"
            return folder / "nontextured.ply", textured if textured.exists() else None
    return None


def load_mesh(path: Path, **kwargs: Any) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", **kwargs)
    assert isinstance(mesh, trimesh.Trimesh), path
    return mesh


def mean_rgb(textured: Path | None, material: str) -> np.ndarray:
    """Area-weighted mean of the texture sampled at the vertex UVs; a material colour without one."""
    if textured is not None:
        mesh = load_mesh(textured, process=False)  # no merge: 007 has a "vn nan nan nan"
        vis = mesh.visual
        material_ = getattr(vis, "material", None)
        image = getattr(material_, "image", None)
        if image is None:
            image = getattr(material_, "baseColorTexture", None)
        uv = getattr(vis, "uv", None)
        if uv is not None and image is not None:
            col = trimesh.visual.color.uv_to_color(uv, image)[:, :3] / 255.0
            face = col[mesh.faces].mean(axis=1)
            return (face * mesh.area_faces[:, None]).sum(axis=0) / mesh.area
    return np.asarray(FALLBACK_RGB[material])


# ---------------------------------------------------------------------------------------------- pose


def footprint_yaw(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Turn about z that axis-aligns the footprint's minimum-area rectangle with its short side on y.

    Rotating calipers: the minimum-area rectangle has a side on a hull edge, so try each edge.
    Returns (yaw, (x extent, y extent)) of the turned footprint.
    """
    h = xy[ConvexHull(xy).vertices]
    edge = np.roll(h, -1, axis=0) - h
    yaw = -np.arctan2(edge[:, 1], edge[:, 0])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x, y = c * h[:, 0] - s * h[:, 1], s * h[:, 0] + c * h[:, 1]
    ext = np.stack([np.ptp(x, axis=1), np.ptp(y, axis=1)], axis=1)
    i = int(np.argmin(ext[:, 0] * ext[:, 1]))
    if ext[i, 1] > ext[i, 0]:
        return float(yaw[i] + np.pi / 2), ext[i, ::-1]
    return float(yaw[i]), ext[i]


def rest_transform(mesh: trimesh.Trimesh, com: np.ndarray, pose: str) -> tuple[np.ndarray, float]:
    """Scan frame -> object frame (resting, narrow side on y, bottom at z = 0, footprint centred), and its probability."""
    tfs, probs = trimesh.poses.compute_stable_poses(mesh, center_mass=com, sigma=0.0, n_samples=1, seed=0)
    up = tfs[:, 2, :3]  # the scan-frame direction each pose turns to world +z
    hull = mesh.convex_hull.vertices
    if pose == "stable":
        i = 0
    elif pose == "upright":
        i = int(np.argmax(up[:, 2]))
        assert up[i, 2] > np.cos(np.radians(10)), f"no stable pose within 10 deg of the scan's +z ({up[i]})"
    elif pose == "lying":
        lying = np.flatnonzero(np.abs(up[:, 2]) < np.sin(np.radians(15)))
        assert len(lying), "no stable pose with the scan's +z horizontal"
        i = int(lying[0])  # poses come sorted by probability
    elif pose == "low_narrow":
        rows = []
        for k, tf in enumerate(tfs):
            v = hull @ tf[:3, :3].T
            if np.ptp(v[:, 2]) * 1000 <= MAX_HEIGHT_MM:
                rows.append((footprint_yaw(v[:, :2])[1][1], k))
        assert rows, f"no stable pose at most {MAX_HEIGHT_MM} mm tall"
        i = min(rows)[1]
    else:
        raise ValueError(pose)
    tf = tfs[i].copy()
    yaw, _ = footprint_yaw((hull @ tf[:3, :3].T)[:, :2])
    tf = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]) @ tf
    v = hull @ tf[:3, :3].T + tf[:3, 3]
    lo, hi = v.min(axis=0), v.max(axis=0)
    tf[:3, 3] -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    return tf, float(probs[i])


# ---------------------------------------------------------------------------------------------- geometry


def cap_vertices(points: np.ndarray, n: int = MAX_PART_VERTS) -> np.ndarray:
    """Hull vertices of ``points``, thinned by farthest-point sampling to at most ``n`` (all stay extreme)."""
    v = points[ConvexHull(points).vertices]
    if len(v) <= n:
        return v
    pick = [int(np.argmax(np.linalg.norm(v - v.mean(axis=0), axis=1)))]
    d = np.linalg.norm(v - v[pick[0]], axis=1)
    for _ in range(n - 1):
        pick.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(v - v[pick[-1]], axis=1))
    return v[np.sort(pick)]


def collision_parts(mesh: trimesh.Trimesh, decomposition: tuple[float, int]) -> list[np.ndarray]:
    """One hull if the scan fills CONVEX_FILL of it (balls, fruit, dice, boxes), else CoACD pieces."""
    if mesh.volume >= CONVEX_FILL * mesh.convex_hull.volume:
        return [cap_vertices(mesh.vertices)]
    threshold, max_hulls = decomposition
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold,
                            max_convex_hull=max_hulls, seed=0)
    return [cap_vertices(np.asarray(v)) for v, _ in parts]


def decimate(mesh: trimesh.Trimesh, max_faces: int = MAX_VISUAL_FACES) -> trimesh.Trimesh:
    """Vertex clustering on a uniform grid, each cell's vertex at its faces' quadric-error minimiser.

    The cell starts where a flat sheet would keep ``max_faces`` faces (two per cell) and grows 5%
    until the clustered mesh fits.  Placing each vertex at the point nearest its faces' planes
    (Lindstrom 2000), not at the cell mean, keeps flat faces flat and box edges sharp; keying the
    cells by the octant of the vertex normal too keeps the two sides of a thin wall (the cup) apart.
    """
    v, f = mesh.vertices, mesh.faces
    plane = np.c_[mesh.face_normals, -np.einsum("ij,ij->i", mesh.face_normals, v[f[:, 0]])]
    kf = mesh.area_faces[:, None, None] * plane[:, :, None] * plane[:, None, :]
    kv = np.zeros((len(v), 4, 4))
    for j in range(3):
        np.add.at(kv, f[:, j], kf)
    octant = (mesh.vertex_normals > 0).astype(np.int64)
    cell = np.sqrt(2 * mesh.area / max_faces)
    while True:
        key = np.c_[np.floor((v - v.min(axis=0)) / cell).astype(np.int64), octant]
        _, cluster = np.unique(key, axis=0, return_inverse=True)
        cluster = cluster.ravel()
        g = cluster[f]
        g = g[(g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2])]
        _, first = np.unique(np.sort(g, axis=1), axis=0, return_index=True)
        g = g[np.sort(first)]
        if len(g) <= max_faces:
            break
        cell *= 1.05
    k = int(cluster.max()) + 1
    q = np.zeros((k, 4, 4))
    np.add.at(q, cluster, kv)
    mean = np.zeros((k, 3))
    np.add.at(mean, cluster, v)
    mean /= np.bincount(cluster, minlength=k)[:, None]
    a, b = q[:, :3, :3], q[:, :3, 3]
    u, s, vt = np.linalg.svd(a)
    inv = np.where(s > 1e-3 * s[:, :1], 1.0 / np.maximum(s, 1e-300), 0.0)  # drop flat directions: stay near the mean
    r = -b - np.einsum("kij,kj->ki", a, mean)
    x = mean + np.einsum("kji,kj->ki", vt, inv * np.einsum("kji,kj->ki", u, r))
    far = np.linalg.norm(x - mean, axis=1) > cell
    x[far] = mean[far]
    out = trimesh.Trimesh(x, g, process=False)
    out.remove_unreferenced_vertices()
    return out


def mass_properties(mesh: trimesh.Trimesh, mass_kg: float) -> tuple[np.ndarray, np.ndarray]:
    """COM and inertia about it (object axes) of a uniform solid at ``mass_kg``: the mesh if watertight, else its hull."""
    solid = mesh if mesh.is_watertight else mesh.convex_hull
    return np.asarray(solid.center_mass), np.asarray(solid.moment_inertia) * mass_kg / solid.mass


def dims_differences(extents_mm: np.ndarray, published: tuple[float, ...]) -> list[str]:
    """Published dims more than DIMS_TOL off the measured extents (both sorted; (d, L) uses min and max)."""
    s, p = sorted(extents_mm), sorted(published)
    pairs = list(zip(s, p, strict=True)) if len(p) == 3 else [(s[0], p[0]), (s[2], p[1])] if len(p) == 2 else [(x, p[0]) for x in s]
    return [f"{m:.1f} vs {q:g} mm ({(m - q) / q:+.0%})" for m, q in pairs if abs(m - q) > DIMS_TOL * q]


# ---------------------------------------------------------------------------------------------- check


def settle(parts: list[np.ndarray], com: np.ndarray, inertia: np.ndarray, mass_kg: float, friction: float,
           seconds: float = 1.0, drop: float = 0.001) -> dict[str, float]:
    """Drop the object 1 mm onto a plane and let it rest; how far its COM moved, how much it tilted and sank."""
    spec = mujoco.MjSpec()
    fr = [friction, 0.005, 0.0001]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], friction=fr)
    body = spec.worldbody.add_body(name="obj", pos=[0, 0, drop])
    body.add_freejoint()
    body.explicitinertial = True
    body.mass = mass_kg
    body.ipos = com.tolist()
    body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    for k, part in enumerate(parts):
        spec.add_mesh(name=f"part_{k}", uservert=np.asarray(part, dtype=float).ravel().tolist())
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"part_{k}", friction=fr)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q0, com0 = data.qpos[3:7].copy(), data.xipos[1].copy()
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    world = np.concatenate(parts) @ rot.reshape(3, 3).T + data.qpos[:3]
    dq = np.zeros(3)
    mujoco.mju_subQuat(dq, data.qpos[3:7], q0)
    return {"finite": bool(np.all(np.isfinite(data.qpos))), "drift_mm": float(np.linalg.norm(data.xipos[1, :2] - com0[:2]) * 1000),
            "tilt_deg": float(np.degrees(np.linalg.norm(dq))), "bottom_mm": float(world[:, 2].min() * 1000),
            "top_mm": float(world[:, 2].max() * 1000), "speed_mm_s": float(np.linalg.norm(data.qvel[:3]) * 1000)}


def verdict(chk: dict[str, float], height_mm: float, round_: bool) -> str:
    bad = []
    if not chk["finite"]:
        return "FAIL non-finite"
    if chk["bottom_mm"] < -1.0:
        bad.append(f"sank {-chk['bottom_mm']:.1f} mm")
    if abs(chk["top_mm"] - height_mm) > 2.0:
        bad.append(f"top {chk['top_mm']:.1f} vs {height_mm:.1f} mm")
    if not round_ and (chk["drift_mm"] > 3.0 or chk["tilt_deg"] > 5.0):
        bad.append(f"moved {chk['drift_mm']:.1f} mm / {chk['tilt_deg']:.1f} deg")
    if not round_ and chk["speed_mm_s"] > 5.0:
        bad.append(f"still moving {chk['speed_mm_s']:.0f} mm/s")
    if bad:
        return "FAIL " + ", ".join(bad)
    return f"ok, rolled {chk['drift_mm']:.1f} mm" if chk["drift_mm"] > 3.0 else "ok"


# ---------------------------------------------------------------------------------------------- main


def process(obj: Obj) -> dict[str, Any] | None:
    found = fetch(obj)
    if found is None:
        return None
    geometry, textured = found
    mesh = load_mesh(geometry)
    scan_com = mesh.center_mass if mesh.is_watertight else mesh.convex_hull.center_mass
    tf, prob = rest_transform(mesh, scan_com, obj.pose)
    mesh.apply_transform(tf)
    extents_mm = np.ptp(mesh.vertices, axis=0)[[1, 0, 2]] * 1000  # width y, depth x, height z
    parts = collision_parts(mesh, obj.decomposition)
    visual = decimate(mesh)
    rgba = np.r_[mean_rgb(textured, obj.material), 1.0]
    com, inertia = mass_properties(mesh, obj.mass_g / 1000)
    arrays: dict[str, np.ndarray] = {f"part_{i}": p.astype(np.float32) for i, p in enumerate(parts)}
    arrays |= {"visual_v": visual.vertices.astype(np.float32), "visual_f": visual.faces.astype(np.int32),
               "com": com, "inertia": inertia, "rgba": rgba, "extents_mm": extents_mm}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{obj.short}.npz", **arrays)

    saved = np.load(OUT / f"{obj.short}.npz")  # check what was written, at its stored precision
    stored = [saved[f"part_{i}"].astype(float) for i in range(len(parts))]
    chk = settle(stored, saved["com"], saved["inertia"], obj.mass_g / 1000, obj.friction)
    width, depth, height = (float(x) for x in extents_mm)
    notes = [f"too wide: {width:.1f} mm > {MAX_WIDTH_MM:g} mm"] if width > MAX_WIDTH_MM else []
    diffs = dims_differences(extents_mm, obj.dims)
    if diffs:
        notes.append("scan differs from published dims: " + ", ".join(diffs))
    if obj.remark:
        notes.append(obj.remark)
    entry: dict[str, Any] = {
        "ycb_id": obj.ycb_id, "name": obj.name, "mass_g": obj.mass_g, "published_dims_mm": list(obj.dims),
        "width_mm": round(width, 1), "depth_mm": round(depth, 1), "height_mm": round(height, 1),
        "height_ok": height <= MAX_HEIGHT_MM, "pose": obj.pose, "material": obj.material,
        "friction": obj.friction, "round": obj.round, "parts": len(parts), "asset": f"assets/ycb/{obj.short}.npz"}
    if notes:
        entry["note"] = "; ".join(notes)
    return {"entry": entry, "check": chk, "verdict": verdict(chk, height, obj.round), "pose_p": prob,
            "verts": sum(len(p) for p in parts), "faces": len(visual.faces), "source": geometry.parent.name}


def write_config(entries: dict[str, dict[str, Any]]) -> None:
    old = yaml.safe_load(CONFIG.read_text()).get("objects", {}) if CONFIG.exists() else {}
    merged = {o.short: entries.get(o.short, old.get(o.short)) for o in OBJECTS}
    merged = {k: v for k, v in merged.items() if v is not None}
    body = yaml.safe_dump({"objects": merged}, sort_keys=False, default_flow_style=None, width=120)
    CONFIG.write_text(YAML_HEADER + "\n" + body)


def main(argv: list[str]) -> int:
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    chosen = [o for o in OBJECTS if only is None or o.short in only]
    if only and len(chosen) != len(only):
        print("unknown:", ", ".join(sorted(only - {o.short for o in chosen})))
        return 2
    coacd.set_log_level("error")
    t = time.time()
    results, dropped = {}, []
    for obj in chosen:
        t0 = time.time()
        r = process(obj)
        if r is None:
            dropped.append(obj.short)
            print(f"{obj.short}: no google_16k or berkeley mesh, dropped")
            continue
        results[obj.short] = r
        print(f"{obj.short}: {r['entry']['parts']} parts, pose p={r['pose_p']:.2f} ({r['source']}), "
              f"{time.time() - t0:.0f}s")
    write_config({k: r["entry"] for k, r in results.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENSE.md").write_text(LICENSE)

    print(f"\n{'object':17s} {'w x d x h mm':>19s}  {'published':>12s} {'mass g':>6s} {'parts':>5s} {'verts':>5s} "
          f"{'faces':>5s}  {'settled top':>11s}  {'drift':>6s} {'tilt':>5s}  check")
    for short, r in results.items():
        e, c = r["entry"], r["check"]
        dims = f"{e['width_mm']:.0f} x {e['depth_mm']:.0f} x {e['height_mm']:.0f}"
        pub = "x".join(f"{d:g}" for d in e["published_dims_mm"])
        print(f"{short:17s} {dims:>19s}  {pub:>12s} {e['mass_g']:6g} {e['parts']:5d} {r['verts']:5d} {r['faces']:5d}  "
              f"{c['top_mm']:5.1f}/{e['height_mm']:5.1f}  {c['drift_mm']:5.1f}mm {c['tilt_deg']:4.1f}d  {r['verdict']}")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"\n{len(results)} objects in {time.time() - t:.0f}s; dropped: {', '.join(dropped) or 'none'}; "
          f"{OUT.relative_to(ROOT)} is {size / 1e6:.1f} MB")
    return 1 if any(r["verdict"].startswith("FAIL") for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
