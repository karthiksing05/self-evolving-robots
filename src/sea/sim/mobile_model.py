"""The mobile robot in MuJoCo at one obstacle (D-063): a step, a trench, a ramp or an opening.

The robot is built from the world's numbers (``sea.worlds.mobile._spec``) with the CAD's layout
(``sea.worlds.cad.mobile``), in metres:
- **Wheels** (four, mecanum or six): a hinge per wheel, each driven by its own gearmotor.
- **Rocker-bogie:** a rocker per side on a pivot in the tub, a bogie on each rocker, and the two rockers
  tied by a differential (a joint equality, q_left = -q_right), as the real bar does.
- **Tracks:** the belt is drawn only.  What touches the ground is a dense row of rollers along the bottom
  run and up the sloped front run to the idler, every one coupled to the drive sprocket by a joint
  equality at equal surface speed, so the whole run pulls as a belt does and meets an edge on its slope.
- **Motors:** a general actuator per driven joint on the DC motor's torque-speed line at the pack's
  voltage, tau = T_s (duty - omega / omega_0), clamped at stall, through the gearbox's efficiency.
- **Rolling resistance:** the world's C_rr as Coulomb friction at each wheel joint, C_rr N r.

Contacts are stiff (a 4 ms time constant on a 1 ms step) with elliptic friction cones, the terrain's
friction taking priority over the robot's.  Mecanum rollers are not modelled one by one: their wheels
take the world's reduced friction.  Sand stays analytic (a Winkler soil): MuJoCo's soft contact is not sinkage.

An episode starts the robot just short of the obstacle, at the steady speed its motors reach on the flat
at that duty, and runs until it is past, stuck for 0.8 s, tipped over, or out of time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import mujoco
import numpy as np

DT = 0.001
SOLREF, SOLIMP = "0.004 1", "0.95 0.99 0.001"
ROLLER_GAP = 0.012  # m between track rollers: under half the lowest edge they meet
SIDE = {"tracks": 0.030, "rocker_bogie6": 0.035, "mecanum4": 0.030, "wheels4": 0.025}
SPROCKET, IDLER = 0.018, 0.015  # m, as the CAD draws them


@dataclass(frozen=True)
class Robot:
    """What the model needs from a body, in SI units (built by ``sea.worlds.mobile._robot``)."""

    loco: str
    m: float  # kg, the whole robot
    wheel_kg: float  # kg, each wheel (or each track's running gear, per side)
    width: float  # inner tub width, m
    length: float  # tub length, m
    tub_h: float  # m
    belly: float  # m, belly clearance
    r: float  # m, wheel or sprocket radius
    torque: float  # N m at each driven joint at stall and full duty (after the gearbox)
    omega0: float  # rad/s, each driven joint's no-load speed at full duty
    crr: float  # rolling-resistance coefficient on this surface
    rotor: float = 0.0  # kg m^2, the motor's rotor inertia seen at the driven joint (J ratio^2)
    track: float = 0.0  # m, track contact length
    idler: float = 0.0  # m, idler height
    bogie: float = 0.0  # m, bogie length


@dataclass
class Episode:
    ok: bool
    progress: float  # 0..1, how far past the obstacle it got
    why: str  # "", or what stopped it: stuck_slipping, stuck_stalled, belly, wall, tipped, bounced, slow
    detail: str
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # trace times (s)
    qpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # trace of every joint, every 20 ms
    model: Any = None


def _cyl_y(name: str, pos: tuple[float, float, float], r: float, half: float, rgba: str, mass: float) -> str:
    return (f'<geom name="{name}" type="cylinder" size="{r:.5f} {half:.5f}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
            f'zaxis="0 1 0" mass="{mass:.5f}" rgba="{rgba}" class="robot"/>')


def _robot_xml(rb: Robot) -> tuple[str, str, str, list[str]]:
    """(the robot's body tree, its equalities, its actuators, the driven joints)."""
    side = SIDE[rb.loco]
    ys = (rb.width / 2 + side / 2, -rb.width / 2 - side / 2)
    zc = rb.belly + rb.tub_h / 2  # the tub's centre above the ground
    tub = (f'<geom name="tub" type="box" size="{rb.length / 2:.5f} {rb.width / 2:.5f} {rb.tub_h / 2:.5f}" '
           f'mass="{{tub_kg:.5f}}" rgba="0.85 0.84 0.8 1" class="robot"/>')
    bodies, eqs, acts, driven = [], [], [], []
    wheel_rgba = "0.12 0.12 0.13 1"

    def joint(name: str, nload: float, r: float, driven: bool = True) -> str:
        return (f'<joint name="{name}" type="hinge" axis="0 1 0" frictionloss="{rb.crr * nload * r:.6f}" '
                f'armature="{rb.rotor if driven else 1e-7:.3g}"/>')

    if rb.loco in ("wheels4", "mecanum4"):
        n_load = rb.m * 9.81 / 4
        for i, x in enumerate((rb.length / 2 - 0.01, -rb.length / 2 + 0.01)):
            for j, y in enumerate(ys):
                nm = f"w{i}{j}"
                bodies.append(f'<body name="{nm}" pos="{x:.5f} {y:.5f} {rb.r - zc:.5f}">{joint(nm, n_load, rb.r)}'
                              + _cyl_y(nm, (0, 0, 0), rb.r, side / 2, wheel_rgba, rb.wheel_kg) + "</body>")
                driven.append(nm)
        tub_kg = rb.m - 4 * rb.wheel_kg
    elif rb.loco == "rocker_bogie6":
        r, bl = rb.r, rb.bogie
        xs = [r, r + bl, r + 2 * bl]  # from the rear, as the CAD lays them out
        pb = ((xs[0] + xs[1]) / 2, r + 0.020)
        pr = ((pb[0] + xs[2]) / 2, r + 0.035)
        cx = pr[0]  # the tub is centred on the rocker pivots
        n_load = rb.m * 9.81 / 6
        for j, y in enumerate(ys):
            rk, bg = f"rocker{j}", f"bogie{j}"
            w0, w1, w2 = (f"w{k}{j}" for k in range(3))
            link = f'<geom type="capsule" size="0.004" fromto="0 0 0 {xs[2] - pr[0]:.5f} 0 {r - pr[1]:.5f}" mass="0.01" rgba="0.8 0.8 0.78 1" class="robot"/>'
            bog = (f'<body name="{bg}" pos="{pb[0] - pr[0]:.5f} 0 {pb[1] - pr[1]:.5f}">'
                   f'<joint name="{bg}" type="hinge" axis="0 1 0" range="-0.7 0.7" limited="true"/>'
                   f'<geom type="capsule" size="0.004" fromto="{xs[0] - pb[0]:.5f} 0 {r - pb[1]:.5f} {xs[1] - pb[0]:.5f} 0 {r - pb[1]:.5f}" mass="0.01" rgba="0.8 0.8 0.78 1" class="robot"/>'
                   + "".join(f'<body name="{w}" pos="{x - pb[0]:.5f} 0 {r - pb[1]:.5f}">{joint(w, n_load, r)}'
                             + _cyl_y(w, (0, 0, 0), r, side / 2, wheel_rgba, rb.wheel_kg) + "</body>"
                             for w, x in ((w0, xs[0]), (w1, xs[1]))) + "</body>")
            front = (f'<body name="{w2}" pos="{xs[2] - pr[0]:.5f} 0 {r - pr[1]:.5f}">{joint(w2, n_load, r)}'
                     + _cyl_y(w2, (0, 0, 0), r, side / 2, wheel_rgba, rb.wheel_kg) + "</body>")
            bodies.append(f'<body name="{rk}" pos="0 {y:.5f} {pr[1] - zc:.5f}">'
                          f'<joint name="{rk}" type="hinge" axis="0 1 0" range="-0.6 0.6" limited="true"/>'
                          + link + bog + front + "</body>")
            driven += [w0, w1, w2]
        eqs.append('<joint joint1="rocker0" joint2="rocker1" polycoef="0 -1 0 0 0"/>')  # the differential bar
        tub_kg = rb.m - 6 * rb.wheel_kg - 0.04
        del cx
    else:  # tracks: rollers along the bottom run and up the sloped front run, coupled to the rear sprocket
        rear, front = SPROCKET, SPROCKET + rb.track
        ix, iz = front + 0.8 * (rb.idler - SPROCKET), rb.idler
        cx = (rear + front) / 2
        centres = [(x, SPROCKET, SPROCKET) for x in np.arange(rear, front + 1e-9, ROLLER_GAP)]
        run = math.hypot(ix - front, iz - SPROCKET)
        for u in np.arange(ROLLER_GAP, run + 1e-9, ROLLER_GAP) / max(run, 1e-9):
            centres.append((front + u * (ix - front), SPROCKET + u * (iz - SPROCKET), SPROCKET + u * (IDLER - SPROCKET)))
        n_load = rb.m * 9.81 / (2 * len(centres))
        per = rb.wheel_kg / len(centres)
        for j, y in enumerate(ys):
            for k, (x, z, r) in enumerate(centres):
                nm = f"t{j}_{k}"
                bodies.append(f'<body name="{nm}" pos="{x - cx:.5f} {y:.5f} {z - zc:.5f}">{joint(nm, n_load, r, k == 0)}'
                              + _cyl_y(nm, (0, 0, 0), r, side / 2, wheel_rgba, per) + "</body>")
                if k:
                    eqs.append(f'<joint joint1="{nm}" joint2="t{j}_0" polycoef="0 {SPROCKET / r:.6f} 0 0 0"/>')
            driven.append(f"t{j}_0")
        tub_kg = rb.m - 2 * rb.wheel_kg
    tree = (f'<body name="robot" pos="0 0 {zc:.5f}"><freejoint name="root"/>' + tub.format(tub_kg=max(tub_kg, 0.05))
            + "".join(bodies) + "</body>")
    for nm in driven:
        t = rb.torque
        acts.append(f'<general name="m_{nm}" joint="{nm}" gaintype="fixed" gainprm="{t:.6f}" biastype="affine" '
                    f'biasprm="0 0 {-t / rb.omega0:.6f}" ctrlrange="-1 1" ctrllimited="true" '
                    f'forcerange="{-t:.6f} {t:.6f}" forcelimited="true"/>')
    return tree, "".join(eqs), "".join(acts), driven


NOSE = 0.003  # m: an edge's rounding (a sharp box edge gives contact normals from one face, not the corner)


def _block(name: str, x0: float, x1: float, bottom: float, top: float, g: str, edge: int) -> str:
    """A slab from ``x0`` to ``x1`` whose top edge at ``x0`` (``edge`` = +1) or ``x1`` (-1) is rounded by NOSE: two
    boxes whose corners sit tangent to a capsule along the edge, so nothing a wheel meets there is a sharp corner."""
    rgba = 'rgba="0.7 0.68 0.64 1"'
    lo_z, hi_x0, hi_x1 = top - NOSE, x0 + NOSE * (edge > 0), x1 - NOSE * (edge < 0)
    cx = x0 + NOSE if edge > 0 else x1 - NOSE

    def box(a: float, b: float, z0: float, z1: float, nm: str = "") -> str:
        named = f'name="{nm}" ' if nm else ""
        return (f'<geom {named}type="box" size="{(b - a) / 2:.5f} 0.6 {(z1 - z0) / 2:.5f}" '
                f'pos="{(a + b) / 2:.5f} 0 {(z0 + z1) / 2:.5f}" {rgba} {g}/>')
    return (box(x0, x1, bottom, lo_z, name) + box(hi_x0, hi_x1, lo_z, top)
            + f'<geom type="capsule" size="{NOSE} 0.6" pos="{cx:.5f} 0 {lo_z:.5f}" zaxis="0 1 0" {rgba} {g}/>')


def terrain_xml(kind: str, f: dict[str, float], mu: float) -> tuple[str, float]:
    """Terrain for one obstacle at x = 0, and the x the robot must get its rear past (m).  Every edge is
    rounded by NOSE, as a real one is."""
    g = f'friction="{mu:.4f} 0.005 0.0001" priority="1" class="ground"'
    floor = '<geom name="floor" type="plane" size="3 1 0.1" rgba="0.55 0.53 0.5 1" {g}/>'
    if kind == "floor":  # flat ground: a film's still of a feature MuJoCo does not drive
        return floor.format(g=g), 0.0
    if kind == "step":
        h = f["h"] / 100
        return floor.format(g=g) + _block("obstacle", 0.0, 2.0, 0.0, h, g, +1), 0.0
    if kind == "trench":
        w, d = f["w"] / 100, f["depth"] / 100
        return (f'<geom name="floor" type="plane" size="3 1 0.1" pos="0 0 {-d:.5f}" rgba="0.4 0.38 0.35 1" {g}/>'
                + _block("near", -2.0, 0.0, -d, 0.0, g, -1) + _block("obstacle", w, w + 2.0, -d, 0.0, g, +1)), w
    if kind == "ramp":
        th = math.radians(f["deg"])
        ln = 1.0
        return floor.format(g=g) + (
            f'<geom name="obstacle" type="box" size="{ln / 2:.5f} 0.6 0.01" euler="0 {-f['deg']:.4f} 0" '
            f'pos="{ln / 2 * math.cos(th) + 0.01 * math.sin(th):.5f} 0 {ln / 2 * math.sin(th) - 0.01 * math.cos(th):.5f}" '
            f'rgba="0.7 0.68 0.64 1" {g}/>'), 0.5 * math.cos(th)
    if kind == "gap":  # two jambs, the opening centred ``off`` off the robot's line
        w, off = f["w"] / 100, f["off"] / 100
        jamb = '<geom name="{n}" type="box" size="0.02 0.25 0.15" pos="0.02 {y:.5f} 0.15" rgba="0.55 0.42 0.3 1" {g}/>'
        return floor.format(g=g) + jamb.format(n="obstacle", y=off + w / 2 + 0.25, g=g) + \
            jamb.format(n="jamb_r", y=off - w / 2 - 0.25, g=g), 0.04
    raise ValueError(kind)


def build(rb: Robot, kind: str, f: dict[str, float], mu: float) -> tuple[mujoco.MjModel, list[str], float]:
    tree, eqs, acts, driven = _robot_xml(rb)
    ground, past = terrain_xml(kind, f, mu)
    xml = f"""<mujoco model="mobile_{kind}">
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>
  <default>
    <default class="robot"><geom contype="2" conaffinity="1" friction="{0.5 * mu:.4f} 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}" condim="3"/></default>
    <default class="ground"><geom contype="1" conaffinity="2" solref="{SOLREF}" solimp="{SOLIMP}" condim="3"/></default>
    <equality solref="{SOLREF}" solimp="{SOLIMP}"/>
  </default>
  <worldbody><light pos="0 -1 2" dir="0 0.5 -1" directional="true"/>{ground}{tree}</worldbody>
  <equality>{eqs}</equality>
  <actuator>{acts}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), driven, past


def episode(rb: Robot, kind: str, f: dict[str, float], mu: float, duty: float, y0: float = 0.0,
            trace: bool = False, standing: bool = False) -> Episode:
    """Drive at ``duty`` into one obstacle, starting ``y0`` m off the robot's line; ``standing``, from rest
    with its leading edge 2 mm short, as a robot climbing out of its own rut starts.  It has 2 s plus two and a
    half times what the distance takes at its flat-ground speed, at most 15 s."""
    m, driven, past = build(rb, kind, f, mu)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mine = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0]
    front = max(d.geom_xpos[g][0] + m.geom_rbound[g] for g in mine)  # the robot's extent about its centre
    rear = min(d.geom_xpos[g][0] - m.geom_rbound[g] for g in mine)
    x0 = -front - (0.002 if standing else 0.02)  # its leading edge 2 cm (or 2 mm) short of the obstacle
    d.qpos[0:3] = (x0, y0, d.qpos[2])
    v_ss = 0.0 if standing else max(0.0, rb.omega0 * rb.r * (duty - rb.crr * rb.m * 9.81 * rb.r
                                                             / (len(driven) * rb.torque)))
    d.qvel[0] = v_ss
    jid = [m.joint(n).id for n in driven]
    radii = [m.geom(n).size[0] for n in driven]
    for j in range(1, m.njnt):  # every wheel and roller already turning at the robot's speed
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name[0] in "wt":
            d.qvel[m.jnt_dofadr[j]] = v_ss / m.geom(m.joint(j).name).size[0]
    for k in range(m.nu):
        d.ctrl[k] = duty
    mujoco.mj_forward(m, d)
    tub = m.geom("tub").id
    goal = past - rear + 0.01  # the robot's centre once its rear is past
    t_max = min(15.0, 2.0 + 2.5 * (goal - x0) / max(v_ss, 0.1 * rb.omega0 * rb.r, 0.03))
    ts, qs = [], []
    best_x, hist = d.qpos[0], []
    belly = bounced = walled = False
    every = int(0.02 / DT)
    for i in range(int(t_max / DT)):
        mujoco.mj_step(m, d)
        if trace and i % every == 0:
            ts.append(d.time)
            qs.append(d.qpos.copy())
        x = d.qpos[0]
        best_x = max(best_x, x)
        if best_x - x > 0.01 and kind in ("step", "trench"):
            bounced = True
        for c in d.contact[:d.ncon]:
            if tub in (c.geom1, c.geom2) and m.geom_bodyid[c.geom1 if c.geom2 == tub else c.geom2] == 0:
                belly = True
            if kind == "gap" and (m.geom(c.geom1).name.startswith(("obstacle", "jamb"))
                                  or m.geom(c.geom2).name.startswith(("obstacle", "jamb"))):
                walled = True
        up = d.xmat[1].reshape(3, 3)[2, 2]
        if up < math.cos(math.radians(70)):
            return _done(False, x, x0, goal, "tipped", "it rolled or pitched past 70 degrees", ts, qs, m)
        if x >= goal:
            return _done(True, x, x0, goal, "", f"past in {d.time:.2f} s", ts, qs, m)
        if i % every == 0:
            hist.append(x)
            if len(hist) > 40 and hist[-1] - hist[-41] < 0.005:  # under 5 mm in the last 0.8 s
                spin = np.mean([abs(d.qvel[m.jnt_dofadr[j]]) * r for j, r in zip(jid, radii, strict=True)])
                if kind == "gap":
                    why = "wall"
                elif belly and kind in ("step", "trench"):
                    why = "belly"
                elif bounced and kind in ("step", "trench"):
                    why = "bounced"
                else:
                    why = "stuck_slipping" if spin > 0.03 else "stuck_stalled"
                return _done(False, x, x0, goal, why, f"stuck {100 * (x - x0):.1f} cm along, its running gear "
                             f"turning at {spin:.2f} m/s", ts, qs, m)
    return _done(False, d.qpos[0], x0, goal, "wall" if walled else "slow", f"not past after {t_max:.1f} s",
                 ts, qs, m)


def _done(ok: bool, x: float, x0: float, goal: float, why: str, detail: str, ts: list, qs: list, m) -> Episode:
    return Episode(ok, float(np.clip((x - x0) / (goal - x0), 0, 1)), why, detail, np.array(ts),
                   np.array(qs) if qs else np.zeros((0, 0)), m)


@cache
def cached(rb: Robot, kind: str, items: tuple[tuple[str, float], ...], mu: float, duty: float, y0: float,
           standing: bool = False) -> Episode:
    """``episode`` for search: its arguments arrive quantised, so attempts that differ by less share physics."""
    return episode(rb, kind, dict(items), mu, duty, y0, standing=standing)
