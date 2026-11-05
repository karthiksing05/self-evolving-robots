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

from sea.mech.base import Base
from sea.mech.base import base as _base
from sea.mech.graph import (
    DENSITY,
    Band,
    Blade,
    Body,
    Bought,
    Box,
    Bracket,
    Capsule,
    Cylinder,
    Encoder,
    Gear,
    Gearmotor,
    Graph,
    LimitSwitch,
    Link,
    PadMount,
    Piston,
    RackTeeth,
    Servo,
    Sphere,
    TorsionSpring,
    Web,
    Wheel,
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


def bracket_box(p: Bracket) -> tuple[np.ndarray, np.ndarray]:
    """A bracket's corners: from ``y_from`` to the frame's face over its footprint, seated into it."""
    from sea.sim import jaws

    face = jaws.frame_face_y(tuple(p.x), tuple(p.z), p.y_from)
    return (np.array([p.x[0], p.y_from, p.z[0]]),
            np.array([p.x[1], p.fallback_y if face is None else face + jaws.FIT, p.z[1]]))


def part_kg(p) -> float:
    """A part's mass: its volume times its material's density, or a bought part's own."""
    from sea.sim import jaws

    if getattr(p, "kg", None) is not None and not isinstance(p, Bought):
        return p.kg
    if isinstance(p, Box):
        return DENSITY[p.material] * 8 * float(np.prod(p.half)) if p.massive and p.material != "stock" else 0.0
    if isinstance(p, Cylinder):
        return (DENSITY[p.material] * math.pi * p.radius**2 * 2 * p.half_length
                if p.massive and p.material != "stock" else 0.0)
    if isinstance(p, Capsule):
        return (DENSITY[p.material] * math.pi * p.radius**2 * (2 * p.half_length + 4 * p.radius / 3)
                if p.massive and p.material != "stock" else 0.0)
    if isinstance(p, Sphere):
        return DENSITY[p.material] * 4 * math.pi * p.radius**3 / 3 if p.massive and p.material != "stock" else 0.0
    if isinstance(p, Link):
        hz = (p.z[1] - p.z[0]) / 2
        vol = 2 * p.half_width * p.length * 2 * hz + 2 * math.pi * p.boss_r**2 * 2 * hz / 2  # half of each round end
        return DENSITY["pla"] * vol
    if isinstance(p, Web):
        return DENSITY["pla"] * float(np.linalg.norm(np.subtract(p.b, p.a))) * 4 * p.half[0] * p.half[1]
    if isinstance(p, Gear):
        share = 1.0 if p.arc is None else (p.arc[1] - p.arc[0]) / (2 * math.pi)
        return DENSITY["pla"] * math.pi * share * p.r_pitch**2 * p.width
    if isinstance(p, Blade):
        return jaws.blade_mass()
    if isinstance(p, Bracket):
        lo, hi = bracket_box(p)
        return DENSITY["pla"] * float(np.prod(hi - lo))
    if isinstance(p, Bought):
        return p.kg
    if isinstance(p, Wheel):
        return DENSITY[p.material] * math.pi * p.radius**2 * 2 * p.half_width if p.massive else 0.0
    return 0.0  # a rack's teeth: drawn


def _add_part(spec: mujoco.MjSpec, body, p, pose: tuple[np.ndarray, np.ndarray]) -> float:
    """Add one part's solids to ``body``; returns the mass it brings (kg)."""
    from sea.sim import jaws

    if isinstance(p, Box):
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_BOX, size=list(p.half), pos=list(p.pos), quat=list(p.quat),
                      rgba=_rgba(p.material), group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Cylinder):
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_CYLINDER, size=[p.radius, p.half_length], pos=list(p.pos),
                      quat=list(p.quat), rgba=_rgba(p.material), group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Capsule):
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_CAPSULE, size=[p.radius, p.half_length], pos=list(p.pos),
                      quat=list(p.quat), rgba=_rgba(p.material), group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Sphere):
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_SPHERE, size=[p.radius, 0.0, 0.0], pos=list(p.pos),
                      rgba=_rgba(p.material), group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Link):
        zc, hz = (p.z[0] + p.z[1]) / 2, (p.z[1] - p.z[0]) / 2
        body.add_geom(name=f"{p.name}_web", type=_GEOM.mjGEOM_BOX, size=[p.half_width, p.length / 2, hz],
                      pos=[0.0, -p.length / 2, zc], rgba=PRINTED, group=2, mass=0.0)
        for k, y in enumerate((0.0, -p.length)):
            body.add_geom(name=f"{p.name}_boss{k}", type=_GEOM.mjGEOM_CYLINDER, size=[p.boss_r, hz], pos=[0.0, y, zc],
                          rgba=PRINTED, group=2, mass=0.0)
        out = p.z[1] + p.pin_head[1] if p.outward > 0 else p.z[0] - p.pin_head[1]
        body.add_geom(name=f"{p.name}_pin", type=_GEOM.mjGEOM_CYLINDER, size=list(p.pin_head),
                      pos=[0.0, -p.length, out], rgba=STEEL, group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Web):
        a, b = np.asarray(p.a), np.asarray(p.b)
        length = float(np.linalg.norm(b - a))
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_BOX, size=[length / 2, *p.half], pos=[*((a + b) / 2), p.z],
                      quat=_quat_z(math.atan2(*(b - a)[::-1])), rgba=PRINTED, group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Gear):
        gv, gf = jaws.gear(p.r_pitch, p.width, p.arc)
        spec.add_mesh(name=p.name, uservert=gv.ravel().tolist(), userface=gf.ravel().tolist())
        body.add_geom(name=f"{p.name}_gear", type=_GEOM.mjGEOM_MESH, meshname=p.name, pos=[0.0, 0.0, p.z],
                      rgba=PRINTED, group=2, mass=0.0)
        return part_kg(p)
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
        return part_kg(p)
    if isinstance(p, Bracket):
        lo, hi = bracket_box(p)
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_BOX, size=((hi - lo) / 2).tolist(), pos=((lo + hi) / 2).tolist(),
                      rgba=PRINTED, group=2, mass=0.0)
        return part_kg(p)
    if isinstance(p, Wheel):  # it rolls: torsional and rolling friction are small (D-087)
        body.add_geom(name=p.name, type=_GEOM.mjGEOM_CYLINDER, size=[p.radius, p.half_width], pos=list(p.pos),
                      quat=list(p.quat), rgba=_rgba(p.material), group=2, mass=0.0, condim=p.condim,
                      friction=[p.mu, 0.005, 0.0001])
        return part_kg(p)
    if isinstance(p, Bought | PadMount):
        return part_kg(p)
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
    if isinstance(p, Bought | PadMount):
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


def _pose(spec: mujoco.MjSpec, name: str, root: str) -> tuple[np.ndarray, np.ndarray]:
    """A body's position and rotation in the base's root frame (its parent chain up to the root)."""
    pos, rot = np.zeros(3), np.eye(3)
    b = spec.body(name)
    while b.name != root:
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, np.asarray(b.quat, float))
        r = r.reshape(3, 3)
        pos, rot = np.asarray(b.pos) + r @ pos, r @ rot
        b = b.parent
    return pos, rot


def rack_beam(g: Graph, gd, root: str) -> jaws.RackBeam | None:  # noqa: F821
    """A guide's printed rack as a beam (D-100), from its parts: the box longest along the teeth is the rack, boxes
    touching it that run along most of it join its section, the others hold it, and a ``roller`` on the root behind
    the rack at the mesh takes the teeth's separating force.  None without printed teeth."""
    from sea.sim import jaws

    car = next((b for b in g.bodies if b.name == gd.carriage), None)
    teeth = next((p for p in (car.parts if car else ()) if isinstance(p, RackTeeth)), None)
    boxes = [p for p in (car.parts if car else ()) if isinstance(p, Box)]
    if teeth is None or not boxes:
        return None

    def span(p):  # gripper-frame extents: the carriage's frame is the jaw's at zero travel
        c, h = jaws.HINGE + jaws.R0 @ np.array(p.pos), np.abs(jaws.R0) @ np.array(p.half)
        return c - h, c + h

    def touches(a, b, gap=0.5e-3):
        return bool(np.all(a[0] <= b[1] + gap) and np.all(b[0] <= a[1] + gap))

    rack = max(boxes, key=lambda p: span(p)[1][0] - span(p)[0][0])
    lo, hi = span(rack)
    section, holders = [rack], []
    for p in boxes:
        if p is not rack and touches(span(p), (lo, hi)):
            (section if span(p)[1][0] - span(p)[0][0] >= 0.9 * (hi[0] - lo[0]) else holders).append(p)
    hold = [span(p) for p in holders] or [(lo, hi)]
    # width (y), height (z) and their middles, with the section's bending found by parallel axes
    a = [(s[1][1] - s[0][1], s[1][2] - s[0][2], (s[1][1] + s[0][1]) / 2, (s[1][2] + s[0][2]) / 2) for s in map(span, section)]
    area = sum(w * h for w, h, _, _ in a)
    yc, zc = sum(w * h * y for w, h, y, _ in a) / area, sum(w * h * z for w, h, _, z in a) / area
    i_sep = sum(w * h**3 / 12 + w * h * (z - zc) ** 2 for w, h, _, z in a)
    i_lat = sum(h * w**3 / 12 + w * h * (y - yc) ** 2 for w, h, y, _ in a)
    c = max(max(abs(z + h / 2 - zc), abs(z - h / 2 - zc)) for _, h, _, z in a)
    gears = [p for b in g.bodies for p in b.parts if isinstance(p, Gear)]
    width = min([teeth.width, *(p.width for p in gears)])
    y = min([jaws.LEWIS_RACK, *(jaws.lewis_y(2 * p.r_pitch / gd.module) for p in gears)])
    rollers = [p for b in g.bodies if b.name == root for p in b.parts if isinstance(p, Cylinder) and p.role == "roller"]
    under = [p for p in rollers if abs(p.pos[0] - gd.mesh_x) <= p.radius + 5e-3 and abs(p.pos[2] + p.radius - lo[2]) <= 0.5e-3
             and lo[1] - 1e-3 <= p.pos[1] <= hi[1] + 1e-3]
    return jaws.RackBeam(float(min(h[0][0] for h in hold)), float(max(h[1][0] for h in hold)), jaws.PLA_E_PA * float(i_sep),
                         float(c), jaws.PLA_E_PA * float(min(i_sep, i_lat)), 1.0 / (width * gd.module * y), bool(under),
                         *(_roller_arm(g, under[0], gd.mesh_x, root) if under else ()))


def _roller_arm(g: Graph, roller: Cylinder, mesh_x: float, root: str) -> tuple[float, float, float]:
    """A roller's printed arm as a cantilever (D-103): the root's longest box touching the roller, from the nearest
    edge of a box holding it to the mesh; its bending stiffness and half depth against the teeth's separating force."""
    from sea.sim import jaws

    boxes = [p for b in g.bodies if b.name == root for p in b.parts if isinstance(p, Box)]

    def touches(p: Box, q: Box) -> bool:
        return all(abs(p.pos[i] - q.pos[i]) <= p.half[i] + q.half[i] + 0.5e-3 for i in range(3))

    near = [p for p in boxes if all(abs(roller.pos[i] - p.pos[i]) <= p.half[i] + roller.radius + 1e-3 for i in range(3))]
    if not near:  # nothing printed reaches it: the mounted law's to refuse
        return 0.0, math.inf, 0.0
    arm = max(near, key=lambda p: p.half[0])
    edges = [e for p in boxes if p is not arm and touches(p, arm) for e in (p.pos[0] - p.half[0], p.pos[0] + p.half[0])]
    length = min((abs(mesh_x - e) for e in edges), default=abs(mesh_x - arm.pos[0]) + arm.half[0])
    return length, jaws.PLA_E_PA * (2 * arm.half[1]) * (2 * arm.half[2]) ** 3 / 12, arm.half[2]


def printed_members(g: Graph, model: mujoco.MjModel) -> list:
    """The printed members a loop's pin force passes through (D-103), from the graph's parts and the assembled pose: at
    each loop, what on ``a`` spans its origin to the pin (a strut), and what reaches from a hub to the same pin on ``b``
    or to ``a``'s own pin on its parent (a crank)."""
    from sea.sim import jaws

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    bodies = {b.name: b for b in g.bodies}
    out = []

    def web(p: Web, kind: str, loop: int, body: str, pin, plates: int, root: float) -> None:
        w, h = 2 * p.half[0], 2 * p.half[1]
        out.append(jaws.Member(p.name, kind, loop, body, tuple(float(x) for x in pin), plates, root, w * h, h * w**3 / 12, w / 2,
                               min(h * w**3, w * h**3) / 12))

    def crank(loop: int, body: str, xy: np.ndarray) -> None:
        for p in getattr(bodies.get(body), "parts", ()):
            if isinstance(p, Web) and np.linalg.norm(p.a) < 1e-3 and np.linalg.norm(np.array(p.b) - xy) < 3e-3:
                hubs = [c.radius for c in bodies[body].parts if isinstance(c, Cylinder) and np.linalg.norm(c.pos[:2]) < 1e-3]
                web(p, "crank", loop, body, (*p.b, 0.0), 1, max(hubs, default=0.0))

    for k, lp in enumerate(g.loops):
        point = np.array(lp.point)
        parts = getattr(bodies.get(lp.a), "parts", ())
        plates = [p for p in parts if isinstance(p, Link) and abs(p.length - np.linalg.norm(point)) < 2e-3]
        webs = [p for p in parts if isinstance(p, Web) and np.linalg.norm(p.a) < 1e-3 and np.linalg.norm(np.array(p.b) - point[:2]) < 2e-3]
        for p in plates:
            t, w = p.z[1] - p.z[0], 2 * p.half_width
            out.append(jaws.Member(p.name, "strut", k, lp.a, tuple(float(x) for x in point), len(plates), 0.0, t * w, t * w**3 / 12,
                                   w / 2, min(t * w**3, w * t**3) / 12))
        for p in webs:
            web(p, "strut", k, lp.a, point, len(webs), 0.0)
        ia, ib = model.body(lp.a).id, model.body(lp.b).id
        world = d.xpos[ia] + d.xmat[ia].reshape(3, 3) @ point
        crank(k, lp.b, (d.xmat[ib].reshape(3, 3).T @ (world - d.xpos[ib]))[:2])
        parent = model.body_parentid[ia]
        crank(k, model.body(parent).name, model.body_pos[ia][:2])
    return out


def rod_end(g: Graph, root: str) -> float:
    """Where the guide's rods end, toward the carriage's travel (root x): the nearest free end of the rods on
    the root."""
    ends = []
    for b in g.bodies:
        if b.name != root:
            continue
        for p in b.parts:
            if isinstance(p, Cylinder) and p.role == "rod":
                axis = np.array([2 * (p.quat[1] * p.quat[3] + p.quat[0] * p.quat[2]),
                                 2 * (p.quat[2] * p.quat[3] - p.quat[0] * p.quat[1]),
                                 1 - 2 * (p.quat[1] ** 2 + p.quat[2] ** 2)])  # the quat's z axis
                ends.append(p.pos[0] + abs(axis[0]) * p.half_length)
    return min(ends) if ends else math.inf


def compile(spec: mujoco.MjSpec, g: Graph, base: str | Base) -> list[tuple[str, float]]:
    """Add the mechanism ``g`` to ``spec`` (the robot ``base`` names); returns the joints
    that move with the driving joint, with their units per radian of it (``Sim.set_arm_q``)."""
    from sea.sim import jaws
    from sea.sim.build_model import TIMESTEP

    bs = _base(base)
    root = bs.root
    whole = bs.driver == ""  # a robot built whole from the graph: every body's mass comes from its parts

    for name in g.remove_stock_mesh:
        for geom in [x for x in spec.geoms if x.parent.name == name and x.meshname == name]:
            spec.delete(geom)
    for b in g.bodies:
        if b.stock:
            sb = spec.body(b.name)
        else:
            sb = spec.body(b.parent).add_body(name=b.name, pos=list(b.pos), quat=list(b.quat))
        for j in b.joints:
            if j.type == "free":
                sb.add_freejoint(name=j.name)
                continue
            kw = {"range": list(j.range)} if j.range is not None else {}
            sb.add_joint(name=j.name, type=_JOINT[j.type], axis=list(j.axis), pos=list(j.pos), armature=j.armature,
                         damping=[j.damping, 0.0, 0.0], stiffness=[j.stiffness, 0.0, 0.0], springref=j.springref,
                         frictionloss=j.frictionloss, **kw)
        pose = _pose(spec, b.name, root) if any(isinstance(p, RackTeeth) for p in b.parts) else (np.zeros(3), np.eye(3))
        if (b.joints or whole) and b.inertial is None and not b.stock:  # its inertia from its solids (D-087)
            for p in b.parts:
                n0 = len(sb.geoms)
                kg = _add_part(spec, sb, p, pose)
                new = [x for x in sb.geoms[n0:] if x.contype or x.conaffinity] or list(sb.geoms[n0:])
                for x in new:
                    x.mass = kg / len(new)
                    if getattr(p, "mu", None) is not None and not isinstance(p, Wheel):
                        x.friction = [p.mu, 0.005, 0.0001]
            continue
        kg = sum(_add_part(spec, sb, p, pose) for p in b.parts)
        if b.inertial is not None:
            _inertial(sb, b, kg)
        elif b.stock:
            sb.mass = float(sb.mass) + kg  # parts fixed to a body with its own inertial: their mass joins it
    for lp in g.loops:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name1=lp.a, name2=lp.b, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          data=list(lp.point) + [0.0] * 9, solimp=LOOP_IMP, **STIFF)
    for gd in g.guides:
        car = spec.body(gd.carriage)
        hw = jaws.BUSHINGS.get(gd.bushing)
        if hw is None:
            raise ValueError(f"no bushing {gd.bushing!r}; there are {sorted(jaws.BUSHINGS)}")
        k_v, k_t = jaws.rod_stiffness(hw.length, hw.ei)  # the stiffest the rods get: the armature keeps each mode stable
        for name, kind, axis, k in (("jaw_rod_dy", "slide", [0.0, 0.0, -1.0], k_v), ("jaw_rod_dz", "slide", [0.0, 1.0, 0.0], k_v),
                                    ("jaw_rod_yaw", "hinge", [0.0, 1.0, 0.0], k_t), ("jaw_rod_pitch", "hinge", [0.0, 0.0, 1.0], k_t)):
            car.add_joint(name=name, type=_JOINT[kind], axis=axis, pos=list(gd.at), armature=k * TIMESTEP**2,
                          damping=[2 * k * TIMESTEP, 0.0, 0.0])
        spec.add_numeric(name="jaw_rods", data=[gd.clamp_end, gd.centre, gd.module, gd.mesh_x])
        if hw is not jaws.LM3:
            spec.add_numeric(name="jaw_bushing", data=[hw.ei, hw.length, hw.rated_n])
        beam = rack_beam(g, gd, root)
        if beam is not None:
            spec.add_numeric(name="jaw_rack", data=beam.data())
        if gd.bushings != (0.0,) or gd.far_end is not None:  # the rods as beams through every bushing (D-080)
            spec.add_numeric(name="jaw_guide", data=[math.nan if gd.far_end is None else gd.far_end, rod_end(g, root),
                                                     *sorted(gd.bushings)])
    for ms in g.meshes:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=ms.follower, name2=ms.driver,
                          data=[0.0, ms.ratio, 0.0, 0.0, 0.0] + [0.0] * 7, solimp=TEETH_IMP, **STIFF)
    for a, b in g.excludes:
        spec.add_exclude(bodyname1=a, bodyname2=b)
    for a, b in g.contacts:
        spec.add_pair(geomname1=a, geomname2=b, condim=3, friction=[0.3, 0.3, 0.005, 0.0001, 0.0001],
                      solref=[0.004, 1.0])  # hard parts in a named pair: the default lets a light part's load sink through a tooth
    _elements(spec, g)
    return list(g.linked)


def _elements(spec: mujoco.MjSpec, g: Graph) -> None:
    """The graph's actuators, latches and sensors as the simulator's own elements (D-087).  A force that
    depends on the joint's angle is an affine actuator (force = gain * command + bias0 + bias1 * q); one that
    depends on its speed (a motor's back-EMF, a needle valve) is the joint's damping, which the simulator
    integrates implicitly."""
    trn = mujoco.mjtTrn.mjTRN_JOINT
    for a in g.actuators:
        if isinstance(a, Band):
            for end, (body, point) in (("a", (a.a, a.a_point)), ("b", (a.b, a.b_point))):
                spec.body(body).add_site(name=f"{a.name}_{end}", pos=list(point), size=[0.002, 0.0, 0.0])
            t = spec.add_tendon(name=a.name, stiffness=a.k_n_m, springlength=[0.0, a.rest_m], width=0.002)
            t.wrap_site(f"{a.name}_a")
            t.wrap_site(f"{a.name}_b")
            continue
        act = spec.add_actuator(name=a.name, target=a.joint, trntype=trn)
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        gain, bias = 0.0, [0.0, 0.0, 0.0]
        if isinstance(a, Servo):  # its gearbox's damping and reflected inertia sit on the joint
            gain, bias = a.kp, [0.0, -a.kp, 0.0]
            act.forcelimited, act.forcerange = True, [-a.stall_nm, a.stall_nm]
            j = spec.joint(a.joint)
            j.damping[0] += a.kv
            j.armature += a.armature
        elif isinstance(a, Gearmotor):  # the DC motor's line: stall torque at rest, none at no-load speed; its
            gain = a.stall_nm             # back-EMF is a damper on the joint (implicit: stable on a light wheel)
            spec.joint(a.joint).damping[0] += a.stall_nm / a.no_load_rad_s
            act.ctrllimited, act.ctrlrange = True, [-1.0, 1.0]
        elif isinstance(a, Piston):  # P A on the valve's command; the needle valve as a damper P A / v
            pa = a.pressure_pa * math.pi * (a.bore_m / 2) ** 2
            gain = pa
            spec.joint(a.joint).damping[0] += pa / a.valve_m_s
            act.ctrllimited, act.ctrlrange = True, [-1.0, 1.0]
        elif isinstance(a, TorsionSpring):  # torque linear in the joint's angle over the span
            slope = (a.torque_nm[1] - a.torque_nm[0]) / a.span_rad
            bias = [a.sign * a.torque_nm[0] - slope * a.set_rad, slope, 0.0]
        act.gainprm = [gain] + [0.0] * 9
        act.biasprm = bias + [0.0] * 7
    for lt in g.latches:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name=lt.name, name1=lt.joint,
                          data=[lt.at, 0.0, 0.0, 0.0, 0.0] + [0.0] * 7, **STIFF)
    for sn in g.sensors:
        if isinstance(sn, Encoder):
            spec.add_sensor(name=sn.name, type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                            objname=sn.joint)
            continue
        box = isinstance(sn, LimitSwitch)
        spec.body(sn.body).add_site(name=sn.name, pos=list(sn.pos), quat=[1.0, 0.0, 0.0, 0.0] if box else list(sn.quat),
                                    type=_GEOM.mjGEOM_BOX if box else _GEOM.mjGEOM_SPHERE,
                                    size=list(sn.half) if box else [0.003, 0.0, 0.0])
        spec.add_sensor(name=sn.name, type=mujoco.mjtSensor.mjSENS_TOUCH if box else mujoco.mjtSensor.mjSENS_RANGEFINDER,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=sn.name, cutoff=0.0 if box else sn.max_m,
                        intprm=[0, 0, 0] if box else [1, 0, 0])  # a range finder reports its distance
