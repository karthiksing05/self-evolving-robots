"""Compose arm + rack tools + panel into one MuJoCo model with ``MjSpec`` (spec 7.1).

Timestep 2 ms, ``implicitfast``, elliptic cones, condim 4 on robot and control geoms.  Collision
bit layout: robot geoms contype 1 / conaffinity 1; panel geoms contype 2 / conaffinity 1; control
geoms contype 4 / conaffinity 1.  So the robot touches panels and controls, but controls never
collide with their own panel or each other (their motion is defined by ``controls.py``).
Scene objects (free rigid bodies for the tabletop world, or fixed fixture probes) are contype 8 /
conaffinity 10: they rest on panels and the table (contype 2) and touch each other.  When a scene has
objects, non-jaw robot geoms add bit 8 to their conaffinity, and every jaw-object geom pair is an
explicit contact pair whose friction combines both surfaces (``pair_friction``, D-032); MuJoCo's
default rule would take the larger of the two and hide any surface slicker than the jaw.  Grasp-world
controls (class ``grasp``) are built the same way: a free body with object bits on the panel, which is
the table.

Rack tools are all attached to the gripper at build time; only the mounted one collides
(``Sim.mount``), so a ``With`` swap needs no recompile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from sea.contracts import ARM_JOINTS, ContactType, ControlSpec, Design, PanelSpec
from sea.control.frames import MJCF_TOOL_SITE, R_SITE_URDF, STOCK_MJCF, mat_to_quat_wxyz, quat_wxyz_to_mat
from sea.design.fingers import pad_geoms
from sea.sim.controls import (
    HINGE_ARMATURE,
    SLIDE_ARMATURE,
    ControlDynamics,
    ControlGeometry,
    Faults,
    GeomSpec,
    MovingPart,
    build_geometry,
)
from sea.sim.jaws import BODIES, four_bar, slide

TIMESTEP = 0.002
STALL_TORQUE = {7.4: 1.62, 12.0: 2.94}
TABLE_HALF = 0.3  # m; the table spans x in [-0.05, 0.55], y in [-0.3, 0.3]
FLOOR_DROP = 0.4  # m from a grasp-world table top down to the floor
ROBOT_BITS = (1, 1)
PANEL_BITS = (2, 1)
CONTROL_BITS = (4, 1)
OBJECT_BIT = 8
OBJECT_BITS = (OBJECT_BIT, PANEL_BITS[0] | OBJECT_BIT)
JAW_FRICTION = 0.45  # bare printed PLA jaw face (D-032)
# Frictional-to-normal impedance for elliptic cones in scenes where the jaws hold a free object (D-048).  At
# MuJoCo's default of 1, soft friction lets a pinched object creep out under a steady load: a 50 g plum
# squeezed at 30 N slid out.  Values above 1 stiffen friction without raising mu ("preventing slip", MuJoCo
# XML reference, option/impratio); MuJoCo Menagerie's gripper scenes use 10.
GRASP_IMPRATIO = 10.0
# Contact stiffness in the same scenes (D-054).  MuJoCo's contact is a spring per unit of effective mass that
# relaxes over the time constant ``solref[0]``.  At the default 20 ms a held object sank into the jaws by an
# amount set by its mass, not its material (a 5 g die 8 mm, a 500 g box 0.3 mm), and an object set down on the
# table sank 5 to 10 mm before the contact caught it; part of every grip was held by overlap (D-051).  Two
# timesteps is the stiffest time constant MuJoCo integrates stably: every object then sinks under 1 mm, light or
# heavy, held or set down.  The compliance a grip does have is the finger's, on its own joint (``_finger_flex``).
GRASP_SOLREF = (2 * TIMESTEP, 1.0)
GRASP_SOLIMP = (0.97, 0.99, 0.001, 0.5, 2.0)


def pair_friction(a: float, b: float) -> float:
    """Friction of a contact between two surfaces: the geometric mean, as Box2D mixes it (D-032)."""
    return float(np.sqrt(a * b))
GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}


JAW_MESHES = {"wrist_roll_follower_so101_v1": "jaw_fixed", "moving_jaw_so101_v1": "jaw_moving"}
COLLISION_DIR = STOCK_MJCF.parent / "collision"
_JAW_CACHE: dict[str, list[np.ndarray]] = {}


def _jaw_parts(mesh: str) -> list[np.ndarray]:
    if mesh not in _JAW_CACHE:
        data = np.load(COLLISION_DIR / f"{mesh}.npz")
        _JAW_CACHE[mesh] = [data[k] for k in sorted(data.files, key=lambda s: int(s.split("_")[1]))]
    return _JAW_CACHE[mesh]


@dataclass
class DomainRandomization:
    """Seeded per-episode variation (spec 8.9): friction x U(0.8,1.2), stiffness x U(0.9,1.1),
    perceived target pose noise N(0, 1 mm)."""

    seed: int | None = None
    friction: tuple[float, float] = (0.8, 1.2)
    stiffness: tuple[float, float] = (0.9, 1.1)
    pose_sigma_m: float = 0.001

    def draws(self, control_ids: list[str]) -> dict[str, tuple[float, float, np.ndarray]]:
        if self.seed is None:
            return {c: (1.0, 1.0, np.zeros(3)) for c in control_ids}
        rng = np.random.default_rng(self.seed)
        out = {}
        for c in control_ids:
            out[c] = (float(rng.uniform(*self.friction)), float(rng.uniform(*self.stiffness)),
                      rng.normal(0.0, self.pose_sigma_m, 3))
        return out


@dataclass
class ControlHandle:
    spec: ControlSpec
    geo: ControlGeometry
    dyn: ControlDynamics
    body_id: int
    joint_id: int
    qadr: int
    dadr: int
    geom_ids: set[int]
    T_world_ctl: np.ndarray  # control frame at rest (world)
    noise: np.ndarray  # perceived position offset (m)
    upper: float  # nominal upper joint limit
    flap: tuple[int, int, int, int] | None = None  # body, joint, qadr, dadr

    @property
    def q(self) -> float:  # set by Sim.refresh
        return self._q

    _q: float = 0.0


@dataclass(frozen=True)
class SceneObject:
    """A rigid body in the scene: free (a tabletop object) or ``fixed`` to the world (a probe).

    ``size`` follows MuJoCo half-sizes: box (hx, hy, hz), cylinder (radius, half-height), sphere (radius,).
    """

    name: str
    shape: Literal["box", "cylinder", "sphere"]
    size: tuple[float, ...]
    pos: tuple[float, float, float]
    mass_kg: float
    friction: float = 0.8
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fixed: bool = False


@dataclass
class Sim:
    model: Any
    data: Any
    panel: PanelSpec | None
    design: Design
    controls: dict[str, ControlHandle]
    arm_qadr: np.ndarray
    arm_dadr: np.ndarray
    site_id: int
    geom_kind: dict[int, str]  # robot geom id -> "arm" | "jaw_fixed" | "jaw_moving" | "tool"
    geom_contact: dict[int, ContactType]  # tool geom id -> its contact type
    tool_geoms: dict[int, list[int]] = field(default_factory=dict)
    tool_bodies: dict[int, int] = field(default_factory=dict)
    tool_tips: dict[int, np.ndarray] = field(default_factory=dict)
    tool_contact: dict[int, ContactType] = field(default_factory=dict)
    tool_template: dict[int, str] = field(default_factory=dict)
    tool_specs: dict[int, Any] = field(default_factory=dict)
    mounted: int | None = None
    body_to_control: dict[int, str] = field(default_factory=dict)
    panel_body: int = -1
    objects: dict[str, int] = field(default_factory=dict)  # scene object name -> body id
    object_geoms: dict[str, int] = field(default_factory=dict)  # scene object name -> geom id
    linked: list[tuple[int, float]] = field(default_factory=list)  # (qpos address, units per servo radian)
    _limit_type: int = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)

    # ---------------------------------------------------------------- tools
    def mount(self, slot: int | None) -> None:
        """Enable collisions (and mass) for one rack tool; ``None`` = stock gripper only."""
        for s, gids in self.tool_geoms.items():
            on = s == slot
            for g in gids:
                self.model.geom_contype[g] = ROBOT_BITS[0] if on else 0
                self.model.geom_conaffinity[g] = (ROBOT_BITS[1] | (OBJECT_BIT if self.objects else 0)) if on else 0
                self.model.geom_rgba[g][3] = 1.0 if on else 0.15
        self.mounted = slot

    def tip_transform(self) -> np.ndarray:
        return self.tool_tips.get(self.mounted, np.eye(4)) if self.mounted is not None else np.eye(4)

    def mounted_contact(self) -> ContactType | None:
        return None if self.mounted is None else self.tool_contact[self.mounted]

    # ---------------------------------------------------------------- state
    def arm_q(self) -> np.ndarray:
        """Joint angles; the gripper joint is the servo's output on every jaw (D-058)."""
        return self.data.qpos[self.arm_qadr].copy()

    def servo_ranges(self) -> np.ndarray:
        """(lo, hi) per arm joint in servo units (radians)."""
        return np.array([self.model.jnt_range[self.model.joint(j).id] for j in ARM_JOINTS], dtype=float)

    def arm_qd(self) -> np.ndarray:
        return self.data.qvel[self.arm_dadr].copy()

    def set_arm_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qadr] = q
        for adr, ratio in self.linked:  # the jaw mechanism's other joints, where the servo angle puts them
            self.data.qpos[adr] = q[-1] * ratio
        self.data.qvel[self.arm_dadr] = 0.0
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def tool_frame_world(self) -> np.ndarray:
        """World transform of the URDF-convention tool frame (z = approach) at the jaw tip."""
        t = np.eye(4)
        t[:3, 3] = self.data.site_xpos[self.site_id]
        t[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3) @ R_SITE_URDF
        return t

    def tip_world(self) -> np.ndarray:
        return self.tool_frame_world() @ self.tip_transform()

    def control_q(self, cid: str) -> float:
        h = self.controls[cid]
        return float(self.data.qpos[h.qadr])

    def body_pose(self, body_id: int) -> np.ndarray:
        t = np.eye(4)
        t[:3, 3] = self.data.xpos[body_id]
        t[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        return t

    def control_point_world(self, cid: str, point_body: np.ndarray) -> np.ndarray:
        h = self.controls[cid]
        return self.body_pose(h.body_id)[:3, :3] @ point_body + self.data.xpos[h.body_id]

    def control_dir_world(self, cid: str, vec_body: np.ndarray) -> np.ndarray:
        h = self.controls[cid]
        return self.data.xmat[h.body_id].reshape(3, 3) @ vec_body

    def ctl_dir_world(self, cid: str, vec_ctl: np.ndarray) -> np.ndarray:
        return self.controls[cid].T_world_ctl[:3, :3] @ vec_ctl

    # ---------------------------------------------------------------- stepping
    def apply_control_forces(self) -> None:
        d = self.data
        for h in self.controls.values():
            q = float(d.qpos[h.qadr])
            d.qfrc_applied[h.dadr] = h.dyn.force(q, float(d.qvel[h.dadr]))
            up = h.dyn.latch_upper()
            self.model.jnt_range[h.joint_id][1] = h.upper if up is None else up

    def _limit_force(self, joint_id: int) -> float:
        d = self.data
        if d.nefc == 0:
            return 0.0
        mask = (d.efc_type[: d.nefc] == self._limit_type) & (d.efc_id[: d.nefc] == joint_id)
        return float(np.abs(d.efc_force[: d.nefc][mask]).sum()) if mask.any() else 0.0

    def update_controls(self) -> None:
        for h in self.controls.values():
            lf = self._limit_force(h.joint_id) if h.dyn.latched else 0.0
            h.dyn.update(float(self.data.qpos[h.qadr]), lf)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.apply_control_forces()
            mujoco.mj_step(self.model, self.data)
            self.update_controls()


# ---------------------------------------------------------------- building
def _add_geom(body, g: GeomSpec, name: str, bits: tuple[int, int], condim: int = 4, mass: float | None = None,
              spec: mujoco.MjSpec | None = None):
    kw = dict(name=name, type=GEOM_TYPES[g.type], size=list(g.size) + [0.0] * (3 - len(g.size)), pos=list(g.pos),
              quat=list(g.quat), rgba=list(g.rgba), contype=bits[0], conaffinity=bits[1], condim=condim)
    if mass is not None:
        kw["mass"] = mass
    if g.type == "mesh":
        assert spec is not None and g.vertices is not None
        mesh = spec.add_mesh(name=f"{name}_mesh", uservert=np.asarray(g.vertices, dtype=float).ravel().tolist())
        if g.faces is not None:
            mesh.userface = np.asarray(g.faces, dtype=int).ravel().tolist()
        kw.update(meshname=f"{name}_mesh", group=3)
    if g.visual:
        kw.update(contype=0, conaffinity=0, group=2, mass=0.0)
    return body.add_geom(**kw)


def _add_moving(parent, part: MovingPart, prefix: str, ctl_pos: np.ndarray, ctl_rot: np.ndarray, faults: Faults,
                friction: float = 1.0, spec: mujoco.MjSpec | None = None):
    pos = ctl_pos + ctl_rot @ np.asarray(part.pos)
    rot = ctl_rot @ quat_wxyz_to_mat(part.quat)
    if part.joint == "free":  # a grasp object: gravity on, object bits, its own surface friction
        body = parent.add_body(name=prefix, pos=pos.tolist(), quat=mat_to_quat_wxyz(rot).tolist())
        body.add_freejoint(name=f"{prefix}_q")
        solid = [g for g in part.geoms if not g.visual]
        for k, g in enumerate(part.geoms):
            each = 0.0 if part.inertia is not None or g.visual else part.mass / max(len(solid), 1)
            geom = _add_geom(body, g, f"{prefix}_{k}_{g.name}", OBJECT_BITS, mass=each, spec=spec)
            geom.friction = [friction, 0.005, 0.0001]
        if part.inertia is not None:  # a real object: the scan's inertia at its published mass
            com, tensor = part.inertia
            evals, evecs = np.linalg.eigh(np.asarray(tensor, dtype=float))
            if np.linalg.det(evecs) < 0:
                evecs[:, 0] = -evecs[:, 0]
            body.explicitinertial = True
            body.mass = part.mass
            body.ipos = np.asarray(com, dtype=float).tolist()
            body.iquat = mat_to_quat_wxyz(evecs).tolist()
            body.inertia = np.maximum(evals, 1e-9).tolist()
        return body, part.range[1]
    body = parent.add_body(name=prefix, pos=pos.tolist(), quat=mat_to_quat_wxyz(rot).tolist(), gravcomp=1.0)
    axis = np.asarray(part.axis, dtype=float)
    lo, hi = part.range
    if faults.reverse:
        axis, (lo, hi) = -axis, (-hi, -lo)
    if faults.jammed:
        lo, hi = -1e-4, 1e-4
    jtype = mujoco.mjtJoint.mjJNT_SLIDE if part.joint == "slide" else mujoco.mjtJoint.mjJNT_HINGE
    body.add_joint(name=f"{prefix}_q", type=jtype, axis=axis.tolist(), range=[lo, hi], limited=True,
                   armature=SLIDE_ARMATURE if part.joint == "slide" else HINGE_ARMATURE, damping=0.0)
    per = part.mass / max(len(part.geoms), 1)
    for k, g in enumerate(part.geoms):
        _add_geom(body, g, f"{prefix}_{k}_{g.name}", CONTROL_BITS, mass=per)
    return body, hi


def _tool_body(gripper, site_pos, site_quat, slot: int, tool, gidx: list[int], spec: mujoco.MjSpec):
    r_site = quat_wxyz_to_mat(site_quat)
    quat = mat_to_quat_wxyz(r_site @ R_SITE_URDF)
    body = gripper.add_body(name=f"tool_slot{slot}", pos=list(site_pos), quat=quat.tolist())
    mass_each = max(tool.mass_g, 0.1) / 1000.0 / max(len(tool.parts), 1)
    names = []
    for k, part in enumerate(tool.parts):
        name = f"tool_{tool.contact}_{gidx[0]}"
        gidx[0] += 1
        # Compiled with collisions ON so the body-level broad-phase filter includes the tool body;
        # Sim.mount() then disables unmounted tools per geom (D-014).
        kw: dict[str, Any] = dict(name=name, type=GEOM_TYPES[part.kind], pos=np.asarray(part.pos).tolist(),
                                  quat=np.asarray(part.quat_wxyz).tolist(), contype=ROBOT_BITS[0],
                                  conaffinity=ROBOT_BITS[1], condim=4,
                                  mass=mass_each, rgba=[0.2, 0.75, 0.9, 0.15])
        if part.kind == "mesh":
            mname = f"toolmesh_s{slot}_{k}"
            spec.add_mesh(name=mname, uservert=np.asarray(part.vertices, dtype=float).ravel().tolist())
            kw["meshname"] = mname
        else:
            size = list(np.asarray(part.size, dtype=float))
            kw["size"] = size + [0.0] * (3 - len(size))
        friction = getattr(tool, "friction", None)
        if friction is not None:
            kw["friction"] = list(friction)
        solref = getattr(tool, "solref", None)
        if solref is not None:
            kw["solref"] = list(solref)
        solimp = getattr(tool, "solimp", None)
        if solimp is not None:
            kw["solimp"] = list(solimp)
        body.add_geom(**kw)
        names.append(name)
    return body, names


MM = 1e-3
# The moving finger bends (D-054).  Its bending stiffness about the hinge, integrated over the stock jaw mesh's
# sections in the closing direction with solid PLA (E 3.5 GPa), loaded 70 mm out: 1.1e3 N m/rad (2.2e5 N/m at the
# load).  The servo's gear train has no published stiffness and is rigid here, so the finger is the only spring
# between the rotor and the object: the stiff bound on an impact.  The rebound is taken as lost (damping ratio 1)
# in the gear train's tooth friction and backlash, which the servo's damping, identified in free motion, leaves out.
FINGER_STIFFNESS_NM_PER_RAD = 1100.0
FINGER_LEVER_M = 0.07
FINGER_DAMPING_RATIO = 1.0


def _finger_flex(spec: mujoco.MjSpec) -> None:
    """A spring joint in series with the joint that carries the finger, on the same axis: the rotor drives
    the gripper joint (and through the pinion a slide's carriage) and the finger deflects on this one.  Its
    inertia keeps the flex mode at one radian per timestep, the fastest the integrator follows."""
    j = spec.joint("jaw_slide") or spec.joint("gripper")
    k = FINGER_STIFFNESS_NM_PER_RAD if j.type == mujoco.mjtJoint.mjJNT_HINGE else FINGER_STIFFNESS_NM_PER_RAD / FINGER_LEVER_M**2
    inertia = k * TIMESTEP**2
    j.parent.add_joint(name="jaw_flex", type=j.type, axis=list(j.axis), pos=list(j.pos), stiffness=[k, 0.0, 0.0],
                       damping=[2 * FINGER_DAMPING_RATIO * float(np.sqrt(k * inertia)), 0.0, 0.0], armature=inertia)


HORN_MESH, HORNLESS_MESH = "sts3215_03a_v1", "sts3215_03a_no_horn_v1"


@lru_cache(maxsize=1)
def _horn_mesh() -> tuple[np.ndarray, np.ndarray]:
    """The STS3215's horn, in the servo mesh's frame (m): the faces of the servo mesh that its horn-less twin lacks."""
    import trimesh
    from scipy.spatial import cKDTree

    whole = trimesh.load(STOCK_MJCF.parent / "assets" / f"{HORN_MESH}.stl")
    bare = trimesh.load(STOCK_MJCF.parent / "assets" / f"{HORNLESS_MESH}.stl")
    new = cKDTree(bare.vertices).query(whole.vertices)[0] > 1e-6
    horn = trimesh.Trimesh(whole.vertices, whole.faces[new[whole.faces].any(axis=1)], process=True)
    return np.asarray(horn.vertices), np.asarray(horn.faces)


def _horn(spec: mujoco.MjSpec, f) -> None:
    """The gripper servo's horn turns with its output (D-056).  The SO-101 model draws the horn on the servo, so the
    hinged jaw, screwed to the horn, cut through it at every angle; the horn-less servo stays on the gripper and
    the horn rides the jaw (on a slide, the pinion's body)."""
    servo = [g for g in spec.geoms if g.parent.name == "gripper" and g.meshname == HORN_MESH]
    for g in servo:
        g.meshname = HORNLESS_MESH
    g = next(g for g in servo if g.contype == 0 and g.conaffinity == 0)  # the drawn one
    jaw = spec.body("moving_jaw_so101_v1")
    r_s, r_j = quat_wxyz_to_mat(np.asarray(g.quat)), quat_wxyz_to_mat(np.asarray(jaw.quat))
    verts, faces = _horn_mesh()
    local = (np.asarray(g.pos) + verts @ r_s.T - np.asarray(jaw.pos)) @ r_j  # servo frame -> jaw frame at q = 0
    spec.add_mesh(name="sts3215_horn", uservert=local.ravel().tolist(), userface=faces.ravel().tolist(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    jaw.add_geom(name="gripper_horn", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sts3215_horn", contype=0,
                 conaffinity=0, group=2, mass=0.0, material=g.material)


def _jaw_joint(spec: mujoco.MjSpec, f) -> list[tuple[str, float]]:
    """Rebuild the moving jaw as its part's mechanism (D-042, ``sim/jaws.py``); returns the mechanism's other
    joints with their units per servo radian.  The gripper joint stays the servo's output, a hinge in
    radians: the slide's pinion turns on it (D-058), and the parallelogram's driving links (D-057)."""
    if f.jaw == "prismatic":
        return slide(spec, f.drive_mm_per_rad * MM)
    if f.jaw == "parallelogram":
        return four_bar(spec, f.link_mm)
    return []


def build_model(design: Design, panel: PanelSpec | None = None, tools: list | None = None,
                faults: dict[str, Faults] | None = None, dr: DomainRandomization | None = None,
                mjcf_path: Path = STOCK_MJCF, objects: Sequence[SceneObject] = (),
                table_z: float | None = None) -> Sim:
    """Compile one episode model.  ``tools`` are ``design.build.BuiltTool``-like objects with
    ``contact, parts, mass_g, tip`` (and optional ``friction, solref, solimp``), one per rack slot.
    ``table_z`` adds a table whose top is at that height under the workspace."""
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 0
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    for act in spec.actuators:
        volts = design.arm.servo_voltage.get(act.name, 7.4)
        tau = STALL_TORQUE[float(volts)]
        act.forcerange = [-tau, tau]
    linked = _jaw_joint(spec, design.fingers)
    _horn(spec, design.fingers)
    # Stock jaw geoms get the names the verifier maps (Appendix B).  The jaw meshes collide through
    # their CoACD parts instead of one convex hull (D-013); other arm meshes keep their hulls.
    counters = {"jaw_fixed": 0, "jaw_moving": 0}
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            continue
        g.condim = 4
        prefix = JAW_MESHES.get(g.meshname)
        if prefix is None:
            if g.parent.name == "gripper" and g.meshname in (HORN_MESH, HORNLESS_MESH):
                g.name = "gripper_servo"
            continue
        g.contype = 0
        g.conaffinity = 0
        g.group = 4
        parent, pos = g.parent, np.asarray(g.pos)
        for k, verts in enumerate(_jaw_parts(g.meshname)):
            mname = f"{g.meshname}_cvx{k}"
            spec.add_mesh(name=mname, uservert=verts.ravel().tolist())
            parent.add_geom(name=f"{prefix}_{counters[prefix]}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                            pos=list(pos), quat=list(g.quat), contype=ROBOT_BITS[0], conaffinity=ROBOT_BITS[1],
                            condim=4, group=3, mass=0.0)
            counters[prefix] += 1
    # Finger pads (design/fingers.py, D-038) are jaw geoms too: they pinch, and pair with objects.
    jaw_mu: dict[str, float] = {}
    for pg in pad_geoms(design.fingers):
        spec.body(pg.body).add_geom(name=pg.name, type=GEOM_TYPES["box"], size=list(pg.half), pos=list(pg.pos),
                                    quat=list(pg.quat), contype=ROBOT_BITS[0], conaffinity=ROBOT_BITS[1], condim=4,
                                    group=2, mass=0.0, rgba=[0.95, 0.55, 0.15, 1.0])
        jaw_mu[pg.name] = pg.friction
    spec.worldbody.add_light(pos=[0.2, 0.0, 1.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    site = next(s for s in spec.sites if s.name == MJCF_TOOL_SITE)
    gripper = spec.body("gripper")
    tool_names: dict[int, list[str]] = {}
    gidx = [0]
    for slot, tool in enumerate(tools or []):
        if tool is None or tool.contact is None or not tool.parts:
            continue
        _, names = _tool_body(gripper, np.asarray(site.pos), np.asarray(site.quat), slot, tool, gidx, spec)
        tool_names[slot] = names
        spec.add_exclude(bodyname1="moving_jaw_so101_v1", bodyname2=f"tool_slot{slot}")
        for b in BODIES[design.fingers.jaw]:
            spec.add_exclude(bodyname1=b, bodyname2=f"tool_slot{slot}")

    faults = faults or {}
    handles_pre: dict[str, tuple] = {}
    draws = (dr or DomainRandomization()).draws([c.id for c in panel.controls] if panel else [])
    surfaces: list[tuple[str, float]] = [(f"obj_{ob.name}_geom", ob.friction) for ob in objects]  # jaw pairs
    if panel is not None:
        from sea.sim.panels import board_half_extents

        p_pos = np.array(panel.pose_world.pos.as_list())
        p_rot = quat_wxyz_to_mat(panel.pose_world.quat_wxyz)
        pbody = spec.worldbody.add_body(name="panel", pos=p_pos.tolist(), quat=list(panel.pose_world.quat_wxyz))
        from sea.sim.grasp_world import table_half

        grasp_only = bool(panel.controls) and all(c.cls == "grasp" for c in panel.controls)
        hx, hy = table_half() if grasp_only else board_half_extents(panel)
        if grasp_only:  # a floor under the table, so an object squeezed off the edge comes to rest
            spec.worldbody.add_geom(name="floor", type=GEOM_TYPES["box"], size=[1.0, 1.0, 0.01],
                                    pos=[0.0, 0.0, float(panel.pose_world.pos.z) - FLOOR_DROP - 0.01],
                                    rgba=[0.5, 0.5, 0.5, 1.0], contype=PANEL_BITS[0], conaffinity=PANEL_BITS[1])
        pbody.add_geom(name="panel_board", type=GEOM_TYPES["box"], size=[hx, hy, 0.005], pos=[0, 0, -0.005],
                       rgba=[0.9, 0.9, 0.85, 1.0], contype=PANEL_BITS[0], conaffinity=PANEL_BITS[1], condim=4)
        pbody.add_camera(name="panel_cam", pos=[0, 0, 0.35], quat=[1, 0, 0, 0], fovy=45)
        for c in panel.controls:
            geo = build_geometry(c)
            f = faults.get(c.id, Faults())
            cpos = np.array(c.pose_panel.pos.as_list())
            crot = quat_wxyz_to_mat(c.pose_panel.quat_wxyz)
            for k, g in enumerate(geo.static):
                gg = GeomSpec(g.name, g.type, g.size, tuple(cpos + crot @ np.asarray(g.pos)),
                              tuple(mat_to_quat_wxyz(crot @ quat_wxyz_to_mat(g.quat))), g.rgba)
                _add_geom(pbody, gg, f"panel_{c.id}_{k}_{g.name}", PANEL_BITS)
            mu = c.friction * draws[c.id][0]
            if geo.moving.joint == "free":  # MuJoCo allows free joints on top-level bodies only
                _, hi = _add_moving(spec.worldbody, geo.moving, f"ctl_{c.id}", p_pos + p_rot @ cpos, p_rot @ crot, f,
                                    friction=mu, spec=spec)
                surfaces += [(f"ctl_{c.id}_{k}_{g.name}", mu) for k, g in enumerate(geo.moving.geoms) if not g.visual]
            else:
                _, hi = _add_moving(pbody, geo.moving, f"ctl_{c.id}", cpos, crot, f)
            flap_hi = None
            if geo.flap is not None:
                _, flap_hi = _add_moving(pbody, geo.flap, f"ctl_{c.id}_flap", cpos, crot, Faults())
            T = np.eye(4)
            T[:3, :3] = p_rot @ crot
            T[:3, 3] = p_pos + p_rot @ cpos
            handles_pre[c.id] = (c, geo, f, T, hi, flap_hi)

    if table_z is not None:
        spec.worldbody.add_geom(name="table", type=GEOM_TYPES["box"], size=[TABLE_HALF, TABLE_HALF, 0.01],
                                pos=[TABLE_HALF - 0.05, 0.0, table_z - 0.01], rgba=[0.75, 0.7, 0.6, 1.0],
                                contype=PANEL_BITS[0], conaffinity=PANEL_BITS[1], condim=4)
    for ob in objects:
        obody = spec.worldbody.add_body(name=f"obj_{ob.name}", pos=list(ob.pos), quat=list(ob.quat_wxyz))
        if not ob.fixed:
            obody.add_freejoint(name=f"obj_{ob.name}_free")
        obody.add_geom(name=f"obj_{ob.name}_geom", type=GEOM_TYPES[ob.shape], size=list(ob.size) + [0.0] * (3 - len(ob.size)),
                       mass=ob.mass_kg, friction=[ob.friction, 0.005, 0.0001], rgba=[0.3, 0.5, 0.8, 1.0],
                       contype=OBJECT_BITS[0], conaffinity=OBJECT_BITS[1], condim=4)
    if surfaces:
        spec.option.impratio = GRASP_IMPRATIO
        _finger_flex(spec)
        for g in spec.geoms:
            g.solref, g.solimp = list(GRASP_SOLREF), list(GRASP_SOLIMP)
        jaws = [g.name for g in spec.geoms if g.name.startswith(("jaw_fixed", "jaw_moving"))]
        for g in spec.geoms:
            if g.contype == ROBOT_BITS[0] and not g.name.startswith(("jaw_fixed", "jaw_moving")):
                g.conaffinity |= OBJECT_BIT
        for name, friction in surfaces:
            for j in jaws:
                mu = pair_friction(jaw_mu.get(j, JAW_FRICTION), friction)
                spec.add_pair(geomname1=j, geomname2=name, condim=4, friction=[mu, mu, 0.005, 0.0001, 0.0001],
                              solref=list(GRASP_SOLREF), solimp=list(GRASP_SOLIMP))

    model = spec.compile()
    data = mujoco.MjData(model)
    arm_qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
    arm_dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
    geom_kind: dict[int, str] = {}
    geom_contact: dict[int, ContactType] = {}
    robot_bodies = {model.body(n).id for n in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                                                "moving_jaw_so101_v1", *BODIES[design.fingers.jaw])}
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith("jaw_fixed"):
            geom_kind[g] = "jaw_fixed"
        elif name.startswith("jaw_moving"):
            geom_kind[g] = "jaw_moving"
        elif name.startswith("tool_"):
            geom_kind[g] = "tool"
            geom_contact[g] = name.split("_")[1]  # type: ignore[assignment]
        elif model.geom_bodyid[g] in robot_bodies and (model.geom_contype[g] or model.geom_conaffinity[g]):
            geom_kind[g] = "arm"

    controls: dict[str, ControlHandle] = {}
    body_to_control: dict[int, str] = {}
    for cid, (c, geo, f, T, _hi, _flap_hi) in handles_pre.items():
        fr, st, noise = draws[cid]
        dyn = ControlDynamics(c, stiffness_scale=st, friction_scale=fr, faults=f)
        dyn.set_lever_arm(geo.lever_arm if geo.rotary else 1.0)
        bid = model.body(f"ctl_{cid}").id
        jid = model.joint(f"ctl_{cid}_q").id
        gids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
        body_to_control[bid] = cid
        flap = None
        if geo.flap is not None:
            fb = model.body(f"ctl_{cid}_flap").id
            fj = model.joint(f"ctl_{cid}_flap_q").id
            flap = (fb, fj, int(model.jnt_qposadr[fj]), int(model.jnt_dofadr[fj]))
            gids |= {g for g in range(model.ngeom) if model.geom_bodyid[g] == fb}
            body_to_control[fb] = cid
        off = 2 if geo.moving.joint == "free" else 0  # a free object's coordinate is its height (z)
        controls[cid] = ControlHandle(c, geo, dyn, bid, jid, int(model.jnt_qposadr[jid]) + off,
                                      int(model.jnt_dofadr[jid]) + off,
                                      gids, T, noise, float(model.jnt_range[jid][1]), flap)
    sim = Sim(model, data, panel, design, controls, arm_qadr, arm_dadr, model.site(MJCF_TOOL_SITE).id,
              geom_kind, geom_contact, body_to_control=body_to_control,
              panel_body=model.body("panel").id if panel is not None else -1,
              objects={ob.name: model.body(f"obj_{ob.name}").id for ob in objects},
              object_geoms={ob.name: model.geom(f"obj_{ob.name}_geom").id for ob in objects},
              linked=[(int(model.jnt_qposadr[model.joint(n).id]), r) for n, r in linked])
    for slot, names in tool_names.items():
        sim.tool_geoms[slot] = [model.geom(n).id for n in names]
        sim.tool_bodies[slot] = model.body(f"tool_slot{slot}").id
        sim.tool_tips[slot] = np.asarray(tools[slot].tip, dtype=float)  # type: ignore[index]
        sim.tool_contact[slot] = tools[slot].contact  # type: ignore[index]
        spec_t = getattr(tools[slot], "spec", None)  # type: ignore[index]
        sim.tool_template[slot] = spec_t.template if spec_t is not None else "?"
        if spec_t is not None:
            sim.tool_specs[slot] = spec_t
    sim.mount(None)
    mujoco.mj_forward(model, data)
    return sim
