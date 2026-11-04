"""The old ME 2110 catalogue's robots, as graphs (D-094): a test fixture, never a seed.

Each robot is read from its compiled simulator model, so the fixture shows that the graph language can say what the
catalogue said without anyone re-drawing a mechanism.  ``timeline`` turns what the old controller did into a program.
"""

from __future__ import annotations

import math

import mujoco

from sea.mech.graph import (
    Body,
    Box,
    Capsule,
    Carry,
    Cylinder,
    Gearmotor,
    Graph,
    Joint,
    Latch,
    Loop,
    Piston,
    Sphere,
    TorsionSpring,
    Wheel,
)
from sea.sim import me2110_model as MM

_G = mujoco.mjtGeom


def _kg(g) -> float:
    if g.mass is not None and not math.isnan(g.mass) and g.mass >= 0:
        return float(g.mass)
    s = g.size
    vol = {_G.mjGEOM_BOX: 8 * s[0] * s[1] * s[2], _G.mjGEOM_SPHERE: 4 * math.pi * s[0] ** 3 / 3,
           _G.mjGEOM_CYLINDER: math.pi * s[0] ** 2 * 2 * s[1],
           _G.mjGEOM_CAPSULE: math.pi * s[0] ** 2 * (2 * s[1] + 4 * s[0] / 3)}[g.type]
    return float(g.density) * vol


def _subtree(m: mujoco.MjModel, root: int) -> list[int]:
    return [i for i in range(m.nbody) if m.body_rootid[i] == root]


def graph_of(rb: MM.Robot, v: MM.Variant = MM.Variant()) -> Graph:
    xml, bd = MM.scene(rb, v, 0.0)
    m = mujoco.MjModel.from_xml_string(xml)
    spec = mujoco.MjSpec.from_string(xml)
    spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    root = m.body("robot").id
    ids = _subtree(m, root)
    name = {i: m.body(i).name or f"body{i}" for i in ids}
    masses: dict[int, list[float]] = {}  # the model keeps only each body's total: the spec keeps each geom's

    def walk(sb, it) -> None:
        masses[next(it)] = [_kg(x) for x in sb.geoms]
        for child in sb.bodies:
            walk(child, it)

    walk(spec.body("robot"), iter(ids))
    rods = {g["rod"] for g in bd.pistons.values()}
    bodies = []
    counters: dict[int, int] = {}
    for i in ids:
        joints = []
        for j in range(m.body_jntnum[i]):
            jid = m.body_jntadr[i] + j
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                joints.append(Joint(name="root", type="free"))
                continue
            dof, jn = m.jnt_dofadr[jid], m.joint(jid).name
            joints.append(Joint(
                name=jn, type="hinge" if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                axis=tuple(float(x) for x in m.jnt_axis[jid]), pos=tuple(float(x) for x in m.jnt_pos[jid]),
                range=(float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])) if m.jnt_limited[jid] else None,
                armature=float(m.dof_armature[dof]), damping=0.0 if jn in rods else float(m.dof_damping[dof]),
                stiffness=float(m.jnt_stiffness[jid]), springref=float(m.qpos_spring[m.jnt_qposadr[jid]]),
                frictionloss=float(m.dof_frictionloss[dof])))
        parts = []
        spec_kg = masses[i]
        for k in range(m.body_geomnum[i]):
            g = m.body_geomadr[i] + k
            nm, kind = m.geom(g).name or f"g{g}", m.geom_type[g]
            pos, quat = tuple(float(x) for x in m.geom_pos[g]), tuple(float(x) for x in m.geom_quat[g])
            s = m.geom_size[g]
            kg = spec_kg[k] if k < len(spec_kg) else None
            mu = float(m.geom_friction[g][0])
            if kind == _G.mjGEOM_BOX:
                parts.append(Box(name=nm, pos=pos, half=tuple(float(x) for x in s), quat=quat, kg=kg, mu=mu))
            elif kind == _G.mjGEOM_CAPSULE:
                parts.append(Capsule(name=nm, pos=pos, radius=float(s[0]), half_length=float(s[1]), quat=quat, kg=kg, mu=mu))
            elif kind == _G.mjGEOM_SPHERE:
                parts.append(Sphere(name=nm, pos=pos, radius=float(s[0]), kg=kg, mu=mu))
            elif kind == _G.mjGEOM_CYLINDER and name[i] in bd.wheels:
                parts.append(Wheel(name=nm, pos=pos, radius=float(s[0]), half_width=float(s[1]), quat=quat, kg=kg, mu=mu, condim=int(m.geom_condim[g])))
            elif kind == _G.mjGEOM_CYLINDER:
                parts.append(Cylinder(name=nm, pos=pos, radius=float(s[0]), half_length=float(s[1]), quat=quat, kg=kg, mu=mu))
            else:
                raise TypeError(f"no part kind for geom {nm} of type {kind}")
        counters[i] = len(parts)
        bodies.append(Body(name=name[i], parent="world" if i == root else name[m.body_parentid[i]],
                           pos=tuple(float(x) for x in m.body_pos[i]), quat=tuple(float(x) for x in m.body_quat[i]),
                           joints=tuple(joints), parts=tuple(parts)))
    loops, latches = [], []
    latched = {e for es in bd.latches.values() for e in es}
    for e in range(m.neq):
        nm = m.eq(e).name
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT:
            loops.append(Loop(a=m.body(m.eq_obj1id[e]).name, b=m.body(m.eq_obj2id[e]).name,
                              point=tuple(float(x) for x in m.eq_data[e][:3])))
        elif nm in latched:
            latches.append(Latch(name=nm, joint=m.joint(m.eq_obj1id[e]).name))
    acts: list = []
    if rb.drive == "wheels2":
        acts += [Gearmotor(name=f"m_{w}", joint=w, stall_nm=MM.MOTOR_TS, no_load_rad_s=MM.MOTOR_W0) for w in bd.wheels]
    for joints in bd.traps.values():
        acts += [TorsionSpring(name=f"m_{j}", joint=j, torque_nm=(MM.TRAP_T0, MM.TRAP_T1), span_rad=math.pi) for j in joints]
    acts += [Piston(name=mech, joint=g["rod"], bore_m=MM.BORE * MM.IN, pressure_pa=MM.P_TANK0, valve_m_s=MM.PISTON_V,
                    end_hold_n=MM.END_HOLD) for mech, g in bd.pistons.items()]
    carried = []
    x0 = m.body_pos[root][0]
    for nm in bd.balls:
        p = m.body_pos[m.body(nm).id]
        carried.append(Carry(item=nm, body="robot", pos=(float(p[0] - x0), float(p[1]), float(p[2]))))
    if bd.flag:
        carried.append(Carry(item="flag", body="flagmount"))
    return Graph(name="catalogue robot", bodies=tuple(bodies), loops=tuple(loops), latches=tuple(latches),
                 actuators=tuple(acts), carried=tuple(carried))


def timeline(rb: MM.Robot, g: Graph, events: list[tuple[float, str]], ctl: dict[str, float]) -> dict[str, float]:
    """What the old controller did in one match, as a program for ``g``: the events it logged, with their times."""
    bd = MM.build_robot(rb)
    out: dict[str, float] = {}
    first = {}
    for t, e in events:
        first.setdefault(e.split(" (")[0].split(" at x")[0], t)
    if rb.drive == "wheels2":
        stop = first.get("drive stops", 40.0)
        back = first.get("drives back")
        home = first.get("back in the dock")
        for w in bd.wheels:
            out.update({f"m_{w}.at": 0.0, f"m_{w}.run": stop, f"m_{w}.duty": ctl.get("duty", 0.6)})
            if back is not None:
                out.update({f"m_{w}.back_at": back, f"m_{w}.back_run": (home or 40.0) - back})
    for mech, latches in bd.latches.items():
        t = first.get(f"released {mech}", 40.0)
        out.update({f"{lt}.at": t for lt in latches})
    for mech in bd.pistons:
        ext = first.get(f"{mech} piston extends")
        if ext is None:
            out.update({f"{mech}.at": 40.0, f"{mech}.hold": 0.0})
        else:
            out.update({f"{mech}.at": ext, f"{mech}.hold": first.get(f"{mech} piston retracts", 40.0) - ext})
    return out
