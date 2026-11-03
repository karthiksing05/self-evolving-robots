"""Compile a mechanism graph into the simulator's model (D-078).

``compile`` adds a ``Graph``'s bodies, joints, solids, constraints and pairs that may touch to an ``MjSpec``.
Every rule here is general: a part's mass is its volume times its material's density (bought parts weigh
what their datasheet says); a body's inertia follows its ``Inertial``; a bracket ends where it meets the
frame; a loop is a pin constraint; teeth in mesh are a joint equality as stiff as printed teeth; a guide
gives its carriage the four ways two clamped rods let it move (``sim/jaws.RodSprings`` loads them).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from sea.mech.graph import (
    DENSITY,
    Blade,
    Body,
    Bought,
    Box,
    Bracket,
    Cylinder,
    Gear,
    Graph,
    Link,
    RackTeeth,
    Web,
)

PRINTED = [0.93, 0.91, 0.86, 1.0]
STEEL = [0.55, 0.57, 0.6, 1.0]
STIFF = {"solref": [0.004, 1.0]}  # pins and teeth: two timesteps (D-054)
LOOP_IMP = [0.99, 0.999, 0.001, 0.5, 2.0]
TEETH_IMP = [0.999, 0.9999, 0.001, 0.5, 2.0]  # printed teeth in mesh: about 1.5 N/um on a 6 mm face
_JOINT = {"hinge": mujoco.mjtJoint.mjJNT_HINGE, "slide": mujoco.mjtJoint.mjJNT_SLIDE}
_GEOM = mujoco.mjtGeom


def _rgba(material: str) -> list[float]:
    return STEEL if material == "steel" else PRINTED


def _quat_z(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def _add_part(spec: mujoco.MjSpec, body, p, pose: tuple[np.ndarray, np.ndarray]) -> float:
    """Add one part's solids to ``body``; returns the mass it brings (kg)."""
    from sea.sim import jaws

    if isinstance(p, Box):
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_BOX, size=list(p.half), pos=list(p.pos), quat=list(p.quat),
                      rgba=_rgba(p.material), group=2, mass=0.0)
        return DENSITY[p.material] * 8 * float(np.prod(p.half)) if p.massive else 0.0
    if isinstance(p, Cylinder):
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_CYLINDER, size=[p.radius, p.half_length], pos=list(p.pos),
                      quat=list(p.quat), rgba=_rgba(p.material), group=2, mass=0.0)
        return DENSITY[p.material] * math.pi * p.radius**2 * 2 * p.half_length if p.massive else 0.0
    if isinstance(p, Link):
        zc, hz = (p.z[0] + p.z[1]) / 2, (p.z[1] - p.z[0]) / 2
        body.add_geom(name=f"{p.name}_web", type=_GEOM.mjGEOM_BOX, size=[p.half_width, p.length / 2, hz],
                      pos=[0.0, -p.length / 2, zc], rgba=PRINTED, group=2, mass=0.0)
        vol = 2 * p.half_width * p.length * 2 * hz
        for k, y in enumerate((0.0, -p.length)):
            body.add_geom(name=f"{p.name}_boss{k}", type=_GEOM.mjGEOM_CYLINDER, size=[p.boss_r, hz], pos=[0.0, y, zc],
                          rgba=PRINTED, group=2, mass=0.0)
            vol += math.pi * p.boss_r**2 * 2 * hz / 2  # half of each round end lies outside the web
        out = p.z[1] + p.pin_head[1] if p.outward > 0 else p.z[0] - p.pin_head[1]
        body.add_geom(name=f"{p.name}_pin", type=_GEOM.mjGEOM_CYLINDER, size=list(p.pin_head),
                      pos=[0.0, -p.length, out], rgba=STEEL, group=2, mass=0.0)
        return DENSITY["pla"] * vol
    if isinstance(p, Web):
        a, b = np.asarray(p.a), np.asarray(p.b)
        length = float(np.linalg.norm(b - a))
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_BOX, size=[length / 2, *p.half], pos=[*((a + b) / 2), p.z],
                      quat=_quat_z(math.atan2(*(b - a)[::-1])), rgba=PRINTED, group=2, mass=0.0)
        return DENSITY["pla"] * length * 4 * p.half[0] * p.half[1]
    if isinstance(p, Gear):
        gv, gf = jaws.gear(p.r_pitch, p.width, p.arc)
        spec.add_mesh(name=p.name, uservert=gv.ravel().tolist(), userface=gf.ravel().tolist())
        body.add_geom(name=f"{p.name}_gear", type=_GEOM.mjGEOM_MESH, meshname=p.name, pos=[0.0, 0.0, p.z],
                      rgba=PRINTED, group=2, mass=0.0)
        share = 1.0 if p.arc is None else (p.arc[1] - p.arc[0]) / (2 * math.pi)
        return DENSITY["pla"] * math.pi * share * p.r_pitch**2 * p.width
    if isinstance(p, RackTeeth):
        tv, tf = jaws.rack_teeth(*p.x, p.x_mesh, p.z_pitch, p.width, p.module)
        pos, rot = pose  # the body in the gripper frame
        local = (tv + [0.0, p.y_mid, 0.0] - pos) @ rot
        spec.add_mesh(name=p.name, uservert=local.ravel().tolist(), userface=tf.ravel().tolist())
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_MESH, meshname=p.name, contype=0, conaffinity=0, rgba=PRINTED,
                      group=2, mass=0.0)
        return 0.0  # drawn: the mesh constraint is the teeth
    if isinstance(p, Blade):
        verts, faces = jaws.blade_mesh()
        spec.add_mesh(name="jaw_blade", uservert=(verts + p.offset).ravel().tolist(), userface=faces.ravel().tolist(),
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        body.add_geom(name="jaw_blade_drawn", type=_GEOM.mjGEOM_MESH, meshname="jaw_blade", contype=0, conaffinity=0,
                      group=2, mass=0.0, rgba=PRINTED)
        for k, part in enumerate(jaws.blade_parts()):  # the finger's contact parts, named as the verifier maps them
            spec.add_mesh(name=f"jaw_blade_cvx{k}", uservert=(part + p.offset).ravel().tolist())
            body.add_geom(name=f"jaw_moving_{k}", type=_GEOM.mjGEOM_MESH, meshname=f"jaw_blade_cvx{k}", condim=4,
                          group=3, mass=0.0)
        return jaws.blade_mass()
    if isinstance(p, Bracket):
        face = jaws.frame_face_y(p.x, p.z, p.y_from)
        lo = np.array([p.x[0], p.y_from, p.z[0]])
        hi = np.array([p.x[1], p.fallback_y if face is None else face + jaws.FIT, p.z[1]])
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_BOX, size=((hi - lo) / 2).tolist(), pos=((lo + hi) / 2).tolist(),
                      rgba=PRINTED, group=2, mass=0.0)
        return DENSITY["pla"] * float(np.prod(hi - lo))
    if isinstance(p, Bought):
        return p.kg
    raise TypeError(p)


def solid_names(p) -> tuple[str, ...]:
    """The simulator's names for a part's solids."""
    if isinstance(p, Link):
        return (f"{p.name}_web", f"{p.name}_boss0", f"{p.name}_boss1", f"{p.name}_pin")
    if isinstance(p, Gear):
        return (f"{p.name}_gear",)
    if isinstance(p, Blade):
        from sea.sim import jaws

        return ("jaw_blade_drawn", *(f"jaw_moving_{k}" for k in range(len(jaws.blade_parts()))))
    if isinstance(p, Bought):
        return ()
    return (p.name,)


def solids(g: Graph) -> dict[str, tuple[str, str, str]]:
    """Every solid the mechanism adds: name -> (body, role, piece).  A part with no piece is a piece of its own."""
    out = {}
    for b in g.bodies:
        for p in b.parts:
            role = getattr(p, "role", "finger" if isinstance(p, Blade) else "structure")
            for n in solid_names(p):
                out[n] = (b.name, role, getattr(p, "piece", "") or getattr(p, "name", n))
    return out


def _inertial(sb, b: Body, mass: float) -> None:
    i = b.inertial
    sb.mass, sb.explicitinertial = mass, True
    sb.ipos, sb.iquat, sb.fullinertia = list(i.pos), [1.0, 0.0, 0.0, 0.0], [math.nan] * 6
    if i.kind == "slender":
        sb.inertia = [mass * i.size**2 / 12 + 1e-7] * 2 + [1e-7]
    elif i.kind == "disc":
        sb.inertia = [mass * i.size**2 / 4 + 1e-7] * 2 + [mass * i.size**2 / 2 + 1e-7]
    else:
        sb.inertia = [mass * i.size] * 3


def _pose(spec: mujoco.MjSpec, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A body's position and rotation in the gripper's frame (its parent chain up to the gripper)."""
    pos, rot = np.zeros(3), np.eye(3)
    b = spec.body(name)
    while b.name != "gripper":
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, np.asarray(b.quat, float))
        r = r.reshape(3, 3)
        pos, rot = np.asarray(b.pos) + r @ pos, r @ rot
        b = b.parent
    return pos, rot


def compile(spec: mujoco.MjSpec, g: Graph) -> list[tuple[str, float]]:
    """Add the mechanism ``g`` to ``spec``; returns the joints that move with the driving joint, with their units
    per radian of it (``Sim.set_arm_q``)."""
    from sea.sim import jaws
    from sea.sim.build_model import TIMESTEP

    for name in g.remove_stock_mesh:
        for geom in [x for x in spec.geoms if x.parent.name == name and x.meshname == name]:
            spec.delete(geom)
    for b in g.bodies:
        if b.stock:
            sb = spec.body(b.name)
        else:
            sb = spec.body(b.parent).add_body(name=b.name, pos=list(b.pos), quat=list(b.quat))
        for j in b.joints:
            kw = {"range": list(j.range)} if j.range is not None else {}
            sb.add_joint(name=j.name, type=_JOINT[j.type], axis=list(j.axis), pos=list(j.pos), armature=j.armature,
                         damping=[j.damping, 0.0, 0.0], **kw)
        pose = _pose(spec, b.name) if any(isinstance(p, RackTeeth) for p in b.parts) else (np.zeros(3), np.eye(3))
        kg = sum(_add_part(spec, sb, p, pose) for p in b.parts)
        if b.inertial is not None:
            _inertial(sb, b, kg)
        elif b.stock:
            sb.mass = float(sb.mass) + kg  # parts fixed to a body with its own inertial: their mass joins it
    for lp in g.loops:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name1=lp.a, name2=lp.b, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          data=list(lp.point) + [0.0] * 9, solimp=LOOP_IMP, **STIFF)
    for gd in g.guides:
        if gd.bushings != (0.0,):
            raise NotImplementedError("one bushing per rod, at the carriage's middle (D-076)")
        car = spec.body(gd.carriage)
        k_v, k_t = jaws.rod_stiffness(jaws.BUSHING_LEN)  # the stiffest the rods get: the armature keeps each mode stable
        for name, kind, axis, k in (("jaw_rod_dy", "slide", [0.0, 0.0, -1.0], k_v), ("jaw_rod_dz", "slide", [0.0, 1.0, 0.0], k_v),
                                    ("jaw_rod_yaw", "hinge", [0.0, 1.0, 0.0], k_t), ("jaw_rod_pitch", "hinge", [0.0, 0.0, 1.0], k_t)):
            car.add_joint(name=name, type=_JOINT[kind], axis=axis, pos=list(gd.at), armature=k * TIMESTEP**2,
                          damping=[2 * k * TIMESTEP, 0.0, 0.0])
        spec.add_numeric(name="jaw_rods", data=[gd.clamp_end, gd.centre, gd.module, gd.mesh_x])
    for ms in g.meshes:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=ms.follower, name2=ms.driver,
                          data=[0.0, ms.ratio, 0.0, 0.0, 0.0] + [0.0] * 7, solimp=TEETH_IMP, **STIFF)
    for a, b in g.excludes:
        spec.add_exclude(bodyname1=a, bodyname2=b)
    return list(g.linked)
