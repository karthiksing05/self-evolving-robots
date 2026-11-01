"""Compose arm + rack tools + panel into one MuJoCo model with ``MjSpec`` (spec 7.1).

Timestep 2 ms, ``implicitfast``, elliptic cones, condim 4 on robot and control geoms.  Collision
bit layout: robot geoms contype 1 / conaffinity 1; panel geoms contype 2 / conaffinity 1; control
geoms contype 4 / conaffinity 1.  So the robot touches panels and controls, but controls never
collide with their own panel or each other (their motion is defined by ``controls.py``).
Scene objects (free rigid bodies for the tabletop world, or fixed fixture probes) are contype 4 /
conaffinity 3, so they also rest on panels and the table (contype 2).

Rack tools are all attached to the gripper at build time; only the mounted one collides
(``Sim.mount``), so a ``With`` swap needs no recompile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from sea.contracts import ARM_JOINTS, ContactType, ControlSpec, Design, PanelSpec
from sea.control.frames import MJCF_TOOL_SITE, R_SITE_URDF, STOCK_MJCF, mat_to_quat_wxyz, quat_wxyz_to_mat
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

TIMESTEP = 0.002
STALL_TORQUE = {7.4: 1.62, 12.0: 2.94}
TABLE_HALF = 0.3  # m; the table spans x in [-0.05, 0.55], y in [-0.3, 0.3]
ROBOT_BITS = (1, 1)
PANEL_BITS = (2, 1)
CONTROL_BITS = (4, 1)
OBJECT_BITS = (4, 3)
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
    _limit_type: int = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)

    # ---------------------------------------------------------------- tools
    def mount(self, slot: int | None) -> None:
        """Enable collisions (and mass) for one rack tool; ``None`` = stock gripper only."""
        for s, gids in self.tool_geoms.items():
            on = s == slot
            for g in gids:
                self.model.geom_contype[g] = ROBOT_BITS[0] if on else 0
                self.model.geom_conaffinity[g] = ROBOT_BITS[1] if on else 0
                self.model.geom_rgba[g][3] = 1.0 if on else 0.15
        self.mounted = slot

    def tip_transform(self) -> np.ndarray:
        return self.tool_tips.get(self.mounted, np.eye(4)) if self.mounted is not None else np.eye(4)

    def mounted_contact(self) -> ContactType | None:
        return None if self.mounted is None else self.tool_contact[self.mounted]

    # ---------------------------------------------------------------- state
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def set_arm_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qadr] = q
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
def _add_geom(body, g: GeomSpec, name: str, bits: tuple[int, int], condim: int = 4, mass: float | None = None):
    kw = dict(name=name, type=GEOM_TYPES[g.type], size=list(g.size) + [0.0] * (3 - len(g.size)), pos=list(g.pos),
              quat=list(g.quat), rgba=list(g.rgba), contype=bits[0], conaffinity=bits[1], condim=condim)
    if mass is not None:
        kw["mass"] = mass
    return body.add_geom(**kw)


def _add_moving(parent, part: MovingPart, prefix: str, ctl_pos: np.ndarray, ctl_rot: np.ndarray, faults: Faults):
    pos = ctl_pos + ctl_rot @ np.asarray(part.pos)
    rot = ctl_rot @ quat_wxyz_to_mat(part.quat)
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
    # Stock jaw geoms get the names the verifier maps (Appendix B).  The jaw meshes collide through
    # their CoACD parts instead of one convex hull (D-013); other arm meshes keep their hulls.
    counters = {"jaw_fixed": 0, "jaw_moving": 0}
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            continue
        g.condim = 4
        prefix = JAW_MESHES.get(g.meshname)
        if prefix is None:
            if g.parent.name == "gripper":
                g.name = "gripper_servo"
            continue
        g.contype = 0
        g.conaffinity = 0
        g.group = 4
        for k, verts in enumerate(_jaw_parts(g.meshname)):
            mname = f"{g.meshname}_cvx{k}"
            spec.add_mesh(name=mname, uservert=verts.ravel().tolist())
            g.parent.add_geom(name=f"{prefix}_{counters[prefix]}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                              pos=list(g.pos), quat=list(g.quat), contype=ROBOT_BITS[0], conaffinity=ROBOT_BITS[1],
                              condim=4, group=3, mass=0.0)
            counters[prefix] += 1
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

    faults = faults or {}
    handles_pre: dict[str, tuple] = {}
    if panel is not None:
        from sea.sim.panels import board_half_extents

        p_pos = np.array(panel.pose_world.pos.as_list())
        p_rot = quat_wxyz_to_mat(panel.pose_world.quat_wxyz)
        pbody = spec.worldbody.add_body(name="panel", pos=p_pos.tolist(), quat=list(panel.pose_world.quat_wxyz))
        hx, hy = board_half_extents(panel)
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

    model = spec.compile()
    data = mujoco.MjData(model)
    arm_qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
    arm_dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
    geom_kind: dict[int, str] = {}
    geom_contact: dict[int, ContactType] = {}
    robot_bodies = {model.body(n).id for n in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                                                "moving_jaw_so101_v1")}
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

    draws = (dr or DomainRandomization()).draws([c.id for c in panel.controls] if panel else [])
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
        controls[cid] = ControlHandle(c, geo, dyn, bid, jid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]),
                                      gids, T, noise, float(model.jnt_range[jid][1]), flap)
    sim = Sim(model, data, panel, design, controls, arm_qadr, arm_dadr, model.site(MJCF_TOOL_SITE).id,
              geom_kind, geom_contact, body_to_control=body_to_control,
              panel_body=model.body("panel").id if panel is not None else -1,
              objects={ob.name: model.body(f"obj_{ob.name}").id for ob in objects},
              object_geoms={ob.name: model.geom(f"obj_{ob.name}_geom").id for ob in objects})
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
