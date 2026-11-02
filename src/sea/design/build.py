"""Build pipeline: CadQuery solids -> STL -> convex parts, with build checks and print estimates (spec 8.8).

Checks recorded in ``BuiltDesign.checks`` (``None`` = skipped):
``validity``, ``min_wall``, ``overhang``, ``mass``, ``link_scale``, ``wrist_torque``, ``self_collision``.
Hard limits come only from ``HardLimits`` (Section 9), never from proposals.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from sea.contracts import BuildEstimate, Design, HardLimits, ToolSpec
from sea.design.tools import ConvexPart, Material, ToolGeometry, make_tool

Decompose = Literal["analytic", "hull", "coacd"]

DENSITY_G_CM3: dict[str, float] = {"pla": 1.24, "tpu": 1.21}
INFILL_FACTOR = 0.5  # effective solid fraction of small FDM parts (walls dominate at this size)
SUPPORT_FACTOR = 1.25  # supported prints: +25% printed volume (time and material)
PRINT_A, PRINT_B = 1.6, 6.0  # print_minutes = a * volume_cm3 + b, per printed piece

# Wrist torque check (stock SO-101, measured with MuJoCo from assets/so101/so101_new_calib.xml at
# zero pose): distance from the wrist_flex axis to the gripperframe site, and the stock wrist
# subtree (wrist + gripper + moving jaw) mass and centre-of-mass lever arm.
L_FRAME_M = 0.1594
STOCK_WRIST_KG = 0.178
STOCK_WRIST_COM_M = 0.0670
NOMINAL_TIP_LOAD_N = 5.0  # half the 10 N class cap, applied at the tool tip
STALL_NM: dict[float, float] = {7.4: 1.62, 12.0: 2.94}
G = 9.81

WALL_SAMPLES = 2000
WALL_FAIL_FRACTION = 0.005  # more than 0.5% of samples thinner than min_wall -> reject
OVERHANG_MAX_DEG = 50.0
OVERHANG_TOL_MM2 = 1.0


@dataclass
class BuiltTool:
    spec: ToolSpec
    contact: str
    tip: np.ndarray
    parts: list[ConvexPart]
    mass_g: float
    volume_cm3: float
    material: Material
    friction: tuple[float, float, float]
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    stl_path: str | None
    needs_support: bool
    com_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    print_minutes: float = 0.0
    grams_pla: float = 0.0
    grams_tpu: float = 0.0
    valid: bool = True
    min_wall_ok: bool = True
    wall_p01_mm: float = 0.0
    overhang_mm2: float = 0.0


@dataclass
class BuiltDesign:
    design: Design
    tools: list[BuiltTool]
    checks: dict[str, bool | None]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors and all(v is not False for v in self.checks.values())


# ---------------------------------------------------------------- estimates
def print_estimate(volume_cm3: float, material: str, a: float = PRINT_A, b: float = PRINT_B) -> tuple[float, float]:
    """(print_minutes, grams) for one printed piece."""
    minutes = volume_cm3 * a + b
    grams = volume_cm3 * DENSITY_G_CM3[material] * INFILL_FACTOR
    return minutes, grams


# ---------------------------------------------------------------- meshes
def to_mesh(solid) -> trimesh.Trimesh:
    """Tessellate a CadQuery workplane/shape into a trimesh (millimetres)."""
    shape = solid.val() if hasattr(solid, "val") else solid
    verts, tris = shape.tessellate(0.05, 0.2)
    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float)
    mesh = trimesh.Trimesh(vertices=v, faces=np.array(tris, dtype=np.int64), process=True)
    # OCC leaves duplicated seam/pole vertices on curved faces; weld them so the mesh is closed.
    mesh.merge_vertices(digits_vertex=6)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def _ray_hits(origins: np.ndarray, dirs: np.ndarray, tris: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Distance to the nearest triangle hit along each ray (inf if none); Moller-Trumbore."""
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    out = np.full(len(origins), np.inf)
    for s in range(0, len(origins), chunk):
        o, d = origins[s : s + chunk, None, :], dirs[s : s + chunk, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum("rtk,tk->rt", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[None]
        u = np.einsum("rtk,rtk->rt", tv, p) * inv
        q = np.cross(tv, e1[None])
        v = np.einsum("rtk,rtk->rt", np.broadcast_to(d, q.shape), q) * inv
        t = np.einsum("rtk,tk->rt", q, e2) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        out[s : s + chunk] = np.where(hit, t, np.inf).min(axis=1)
    return out


def wall_thickness(mesh: trimesh.Trimesh, n: int = WALL_SAMPLES, seed: int = 0) -> np.ndarray:
    """Local wall thickness (mm) at ``n`` area-weighted surface samples, by inward ray cast."""
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[face_idx]
    origins = pts - normals * 1e-4
    return _ray_hits(origins, -normals, mesh.triangles)


def overhang_area(mesh: trimesh.Trimesh, rotation: np.ndarray, max_deg: float = OVERHANG_MAX_DEG) -> float:
    """Area (mm^2) of down-facing faces steeper than ``max_deg`` from vertical, off the bed."""
    verts = mesh.vertices @ rotation.T
    normals = mesh.face_normals @ rotation.T
    zmin = verts[:, 2].min()
    on_bed = np.all(verts[mesh.faces][:, :, 2] < zmin + 1e-3, axis=1)
    steep = -normals[:, 2] > math.sin(math.radians(max_deg))
    return float(mesh.area_faces[steep & ~on_bed].sum())


def _decompose(geom: ToolGeometry, mesh: trimesh.Trimesh, method: Decompose) -> list[ConvexPart]:
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    if method == "analytic":
        return geom.parts
    if method == "hull":
        hull = mesh.convex_hull
        return [ConvexPart("mesh", np.zeros(3), ident, np.zeros(3), vertices=np.asarray(hull.vertices) / 1000.0)]
    import coacd

    coacd.set_log_level("error")
    pieces = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.05, seed=0)
    return [ConvexPart("mesh", np.zeros(3), ident, np.zeros(3), vertices=np.asarray(v) / 1000.0) for v, _ in pieces]


# ---------------------------------------------------------------- tools
_TOOL_CACHE: dict[str, BuiltTool] = {}


def spec_hash(spec: ToolSpec, decompose: str = "analytic") -> str:
    payload = json.dumps({"t": spec.template, "p": sorted(spec.params.items()), "d": decompose})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_tool(spec: ToolSpec, min_wall_mm: float, decompose: Decompose = "analytic",
               cache_dir: Path | None = None) -> BuiltTool:
    """Build one tool (cached by spec hash).  Raises ValueError on an invalid spec."""
    key = f"{spec_hash(spec, decompose)}:{min_wall_mm:g}"
    cached = _TOOL_CACHE.get(key)
    if cached is not None and (cache_dir is None or cached.stl_path is not None):
        return cached
    geom = make_tool(spec)
    if geom.solid is None:  # stock gripper: nothing to print or attach
        bt = BuiltTool(spec, geom.contact, geom.tip, [], 0.0, 0.0, geom.material, geom.friction,
                       geom.solref, geom.solimp, None, False)
        _TOOL_CACHE[key] = bt
        return bt
    shape = geom.solid.val()
    mesh = to_mesh(geom.solid)
    valid = bool(shape.isValid()) and bool(mesh.is_watertight)
    minutes = grams_pla = grams_tpu = mass = volume = 0.0
    com = np.zeros(3)
    for piece in geom.pieces:
        pshape = piece.solid.val()
        vol = float(pshape.Volume()) / 1000.0
        factor = SUPPORT_FACTOR if geom.needs_support else 1.0
        m_min, grams = print_estimate(vol * factor, piece.material)
        minutes += m_min
        if piece.material == "pla":
            grams_pla += grams
        else:
            grams_tpu += grams
        g_part = vol * DENSITY_G_CM3[piece.material] * INFILL_FACTOR
        c = pshape.Center()
        com += g_part * np.array([c.x, c.y, c.z])
        mass += g_part
        volume += vol
    com = com / max(mass, 1e-12) / 1000.0
    thick = wall_thickness(mesh)
    thin = float(np.mean(thick < min_wall_mm - 1e-3))
    stl_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stl_path = str(cache_dir / f"tool_{spec_hash(spec, decompose)}.stl")
        mesh.export(stl_path)
    bt = BuiltTool(
        spec=spec, contact=geom.contact, tip=geom.tip, parts=_decompose(geom, mesh, decompose),
        mass_g=mass, volume_cm3=volume, material=geom.material, friction=geom.friction,
        solref=geom.solref, solimp=geom.solimp, stl_path=stl_path, needs_support=geom.needs_support,
        com_m=com, print_minutes=minutes, grams_pla=grams_pla, grams_tpu=grams_tpu, valid=valid,
        min_wall_ok=thin <= WALL_FAIL_FRACTION, wall_p01_mm=float(np.percentile(thick, 1)),
        overhang_mm2=overhang_area(mesh, geom.print_rotation),
    )
    _TOOL_CACHE[key] = bt
    return bt


# ---------------------------------------------------------------- design-level checks
def wrist_torque(tool: BuiltTool | None, tip_load_n: float = NOMINAL_TIP_LOAD_N) -> float:
    """Worst-case wrist_flex torque (N*m): arm extended horizontally, tool mounted, tip loaded."""
    tau = G * STOCK_WRIST_KG * STOCK_WRIST_COM_M + tip_load_n * L_FRAME_M
    if tool is not None and tool.parts:
        tau += G * tool.mass_g / 1000.0 * (L_FRAME_M + float(tool.com_m[2]))
        tau += tip_load_n * float(tool.tip[2, 3])
    return tau


def finger_estimate(design: Design) -> tuple[float, float, float, float]:
    """(print_minutes, grams_pla, grams_tpu, added_mass_g) for the two finger pads (design/fingers.py).

    Volume is the sum of the pad boxes the simulator collides with; the TPU skin is its share of it.
    """
    from sea.design.fingers import has_pads, pad_geoms, thickness

    f = design.fingers
    if not has_pads(f):  # the stock pads; a jaw joint change is priced by the world that offers it (D-042)
        return 0.0, 0.0, 0.0, 0.0
    vol = sum(8 * g.half[0] * g.half[1] * g.half[2] for g in pad_geoms(f)) * 1e6  # cm^3, both pads
    tpu = vol * (f.pad_tpu_mm * 1e-3) / thickness(f)
    pla = vol - tpu
    m_pla, g_pla = print_estimate(pla, "pla")
    m_tpu, g_tpu = print_estimate(tpu, "tpu") if tpu > 0 else (0.0, 0.0)
    mass = (pla * DENSITY_G_CM3["pla"] + tpu * DENSITY_G_CM3["tpu"]) * INFILL_FACTOR
    return m_pla + m_tpu, g_pla, g_tpu, mass


def finger_print(old: Design, new: Design) -> tuple[float, float, float]:
    """(volume_cm3, grams_pla, grams_tpu) to print when ``new`` changes ``old``'s fingers: both pads,
    whole, since a changed pad is a new part.  Zero when the fingers are unchanged or go back to stock."""
    from sea.design.fingers import pad_geoms

    if new.fingers == old.fingers:
        return 0.0, 0.0, 0.0
    vol = sum(8 * g.half[0] * g.half[1] * g.half[2] for g in pad_geoms(new.fingers)) * 1e6
    _, pla, tpu, _ = finger_estimate(new)
    return vol, pla, tpu


def build(design: Design, limits: HardLimits, *, decompose: Decompose = "analytic",
          collision_check: Callable[[BuiltDesign], list[str]] | None = None,
          cache_dir: Path | None = None) -> BuiltDesign:
    """Build every rack tool, run the Section 8.8 checks, and fill ``design.est``."""
    errors: list[str] = []
    tools: list[BuiltTool] = []
    for i, spec in enumerate(design.tool_rack):
        try:
            tools.append(build_tool(spec, limits.min_wall_mm, decompose, cache_dir))
        except ValueError as exc:
            errors.append(f"rack[{i}]: {exc}")
    checks: dict[str, bool | None] = {}
    checks["validity"] = all(t.valid for t in tools) and not errors
    checks["min_wall"] = all(t.min_wall_ok for t in tools)
    checks["overhang"] = all(t.needs_support or t.overhang_mm2 <= OVERHANG_TOL_MM2 for t in tools)
    checks["mass"] = all(t.mass_g <= limits.max_tool_mass_g for t in tools)
    arm = design.arm
    checks["link_scale"] = max(arm.upper_scale, arm.fore_scale) <= limits.max_link_scale + 1e-9 and \
        min(arm.upper_scale, arm.fore_scale) > 0
    stall = STALL_NM[float(arm.servo_voltage.get("wrist_flex", 7.4))]
    worst = max([wrist_torque(t) for t in tools] + [wrist_torque(None)])
    checks["wrist_torque"] = worst <= 0.8 * stall
    for i, t in enumerate(tools):
        if not t.valid:
            errors.append(f"rack[{i}]: invalid solid")
        if not t.min_wall_ok:
            errors.append(f"rack[{i}]: wall thinner than {limits.min_wall_mm} mm (p01 {t.wall_p01_mm:.2f} mm)")
        if not (t.needs_support or t.overhang_mm2 <= OVERHANG_TOL_MM2):
            errors.append(f"rack[{i}]: unsupported overhang {t.overhang_mm2:.1f} mm^2")
        if t.mass_g > limits.max_tool_mass_g:
            errors.append(f"rack[{i}]: mass {t.mass_g:.1f} g > {limits.max_tool_mass_g} g")
    if not checks["link_scale"]:
        errors.append(f"link scale above {limits.max_link_scale}")
    if not checks["wrist_torque"]:
        errors.append(f"wrist torque {worst:.2f} N*m > 0.8 x {stall} N*m")
    f_min, f_pla, f_tpu, f_mass = finger_estimate(design)
    est = BuildEstimate(
        print_minutes=sum(t.print_minutes for t in tools) + f_min,
        grams_pla=sum(t.grams_pla for t in tools) + f_pla,
        grams_tpu=sum(t.grams_tpu for t in tools) + f_tpu,
        added_mass_g=max([t.mass_g for t in tools] + [0.0]) + f_mass,
    )
    built = BuiltDesign(design.model_copy(update={"est": est}), tools, checks, errors)
    if collision_check is None:
        checks["self_collision"] = None
    else:
        coll = collision_check(built)
        checks["self_collision"] = not coll
        errors.extend(coll)
    return built
