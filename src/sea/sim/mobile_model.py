"""The mobile robot in MuJoCo at one obstacle (D-063, D-069): a step, a flight of stairs, a cable or hose, a
threshold strip, a trench, a ramp or an opening.

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
friction taking priority over the robot's.  A mecanum wheel is a driven hub carrying ``ROLLERS`` free rollers on
45 degree axes (D-070), so it passes force only across each roller's axis.  Sand stays analytic (a Winkler soil): MuJoCo's soft contact is not sinkage.

An episode starts the robot just short of the obstacle, at the steady speed its motors reach on the flat
at that duty, and runs until it is past, stuck for 0.8 s, tipped over, or out of time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any

import mujoco
import numpy as np

DT = 0.001
# A wheeled robot's parts meet each other (D-076): a wheel, a rocker or a bogie that swings into the tub or into
# another wheel is stopped there, as the real one is (bit 4 on both sides).  A track's rollers overlap by design,
# standing for one belt, so a tracked robot's parts keep to the ground only.
SELF = (2 | 4, 1 | 4)
SOLREF, SOLIMP = "0.004 1", "0.95 0.99 0.001"
ROLLER_GAP = 0.012  # m between track rollers: under half the lowest edge they meet
SIDE = {"tracks": 0.030, "rocker_bogie6": 0.035, "mecanum4": 0.030, "wheels4": 0.025}
SPROCKET, IDLER = 0.018, 0.015  # m, as the CAD draws them
ROLLERS, ROLLER_AT, ROLLER_R = 10, 0.82, 0.18  # a mecanum wheel's rollers: count, centre and radius over the wheel's


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
    v: float = 0.0  # m/s, its speed over the last moments
    power: float = 0.0  # W, what its motors delivered over the last moments
    sink: float = 0.0  # m, how deep it runs in soft ground
    elapsed: float = 0.0  # s it took, as simulated
    work: float = 0.0  # J its motors delivered, as simulated


def _cyl_y(name: str, pos: tuple[float, float, float], r: float, half: float, rgba: str, mass: float) -> str:
    return (f'<geom name="{name}" type="cylinder" size="{r:.5f} {half:.5f}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
            f'zaxis="0 1 0" mass="{mass:.5f}" rgba="{rgba}" class="robot"/>')


def roller_axes(r: float, lean: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """A mecanum wheel's rollers in the hub's frame (axle along y): each centre and its axis, 45 degrees across the
    rim toward ``lean`` (+1 or -1) along the axle."""
    out = []
    for k in range(ROLLERS):
        a = 2 * math.pi * k / ROLLERS
        rim = np.array([-math.sin(a), 0.0, math.cos(a)])  # the rim's direction at the roller
        out.append((ROLLER_AT * r * np.array([math.cos(a), 0.0, math.sin(a)]),
                    (rim + lean * np.array([0.0, 1.0, 0.0])) / math.sqrt(2)))
    return out


def _mecanum(nm: str, r: float, side: float, lean: float, kg: float) -> str:
    """A mecanum wheel's hub and its free rollers, each a capsule on a hinge along its own axis (D-070)."""
    hub = (f'<geom name="{nm}" type="cylinder" size="{0.62 * r:.5f} {side / 2:.5f}" zaxis="0 1 0" mass="{0.6 * kg:.5f}" '
           f'rgba="0.55 0.56 0.6 1" class="robot"/>')
    half = 0.45 * side  # half a roller's length, as the CAD draws it
    rollers = "".join(
        f'<body name="{nm}_r{k}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}">'
        f'<joint name="{nm}_r{k}" type="hinge" axis="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f}" damping="1e-6" armature="1e-8"/>'
        f'<geom name="{nm}_roller{k}" type="capsule" size="{ROLLER_R * r:.5f}" fromto="{-half * a[0]:.5f} {-half * a[1]:.5f} {-half * a[2]:.5f} '
        f'{half * a[0]:.5f} {half * a[1]:.5f} {half * a[2]:.5f}" mass="{0.4 * kg / ROLLERS:.6f}" rgba="0.12 0.12 0.13 1" '
        f'class="robot"/></body>' for k, (c, a) in enumerate(roller_axes(r, lean)))
    return hub + rollers


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
                if rb.loco == "mecanum4":  # the X pattern: front-left and rear-right lean one way, the others the other
                    wheel = _mecanum(nm, rb.r, side, 1.0 if i == j else -1.0, rb.wheel_kg)
                else:
                    wheel = _cyl_y(nm, (0, 0, 0), rb.r, side / 2, wheel_rgba, rb.wheel_kg)
                bodies.append(f'<body name="{nm}" pos="{x:.5f} {y:.5f} {rb.r - zc:.5f}">{joint(nm, n_load, rb.r)}'
                              + wheel + "</body>")
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
            ly = (1 if y > 0 else -1) * (side / 2 + 0.006)  # the links run outboard of the wheels, on stub axles
            link = f'<geom name="rocker_link{j}" type="capsule" size="0.004" fromto="0 {ly:.5f} 0 {xs[2] - pr[0]:.5f} {ly:.5f} {r - pr[1]:.5f}" mass="0.01" rgba="0.8 0.8 0.78 1" class="robot"/>'
            bog = (f'<body name="{bg}" pos="{pb[0] - pr[0]:.5f} 0 {pb[1] - pr[1]:.5f}">'
                   f'<joint name="{bg}" type="hinge" axis="0 1 0" range="-40 40" limited="true"/>'
                   f'<geom name="bogie_link{j}" type="capsule" size="0.004" fromto="{xs[0] - pb[0]:.5f} {ly:.5f} {r - pb[1]:.5f} {xs[1] - pb[0]:.5f} {ly:.5f} {r - pb[1]:.5f}" mass="0.01" rgba="0.8 0.8 0.78 1" class="robot"/>'
                   + "".join(f'<body name="{w}" pos="{x - pb[0]:.5f} 0 {r - pb[1]:.5f}">{joint(w, n_load, r)}'
                             + _cyl_y(w, (0, 0, 0), r, side / 2, wheel_rgba, rb.wheel_kg) + "</body>"
                             for w, x in ((w0, xs[0]), (w1, xs[1]))) + "</body>")
            front = (f'<body name="{w2}" pos="{xs[2] - pr[0]:.5f} 0 {r - pr[1]:.5f}">{joint(w2, n_load, r)}'
                     + _cyl_y(w2, (0, 0, 0), r, side / 2, wheel_rgba, rb.wheel_kg) + "</body>")
            bodies.append(f'<body name="{rk}" pos="0 {y:.5f} {pr[1] - zc:.5f}">'
                          f'<joint name="{rk}" type="hinge" axis="0 1 0" range="-34 34" limited="true"/>'
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


def self_clashes(rb: Robot, tol: float = 0.0005) -> list[tuple[str, str, float, str]]:
    """Parts of the robot that pass into each other (more than ``tol`` m), at rest and with each rocker and bogie
    at the ends and middle of its swing: (part, part, depth m, pose).  A track's rollers overlap by design."""
    if rb.loco == "tracks":
        return []
    m, _, _ = build(rb, "floor", {}, 0.5)
    d = mujoco.MjData(m)
    swing = [m.joint(n).id for n in ("rocker0", "bogie0", "bogie1")] if rb.loco == "rocker_bogie6" else []
    poses = [{}] + [{j: v} for j in swing for v in m.jnt_range[j]]
    worst: dict[tuple[str, str], tuple[float, str]] = {}
    for pose in poses:
        mujoco.mj_resetData(m, d)
        d.qpos[2] += 0.2  # off the ground: only the robot's own parts meet
        for j, v in pose.items():
            d.qpos[m.jnt_qposadr[j]] = v
            if m.joint(j).name == "rocker0":  # the differential turns the other rocker the other way
                d.qpos[m.jnt_qposadr[m.joint("rocker1").id]] = -v
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            if c.dist < -tol:
                key = tuple(sorted((m.geom(c.geom1).name, m.geom(c.geom2).name)))
                what = ", ".join(f"{m.joint(j).name} {v:+.2f} rad" for j, v in pose.items()) or "at rest"
                if key not in worst or -c.dist > worst[key][0]:
                    worst[key] = (-float(c.dist), what)
    return [(a, b, dz, what) for (a, b), (dz, what) in sorted(worst.items(), key=lambda kv: -kv[1][0])]


OBST = (0.05, 0.08, 0.06)  # m: a corridor obstacle's half length and half width, and how far off the line it stands
SAND_DEPTH = 0.2  # m of soil above the hard floor
BUMP_V, REACT_S, BACKOFF_S = 0.25, 0.1, 4.0  # a bumper's rated impact speed (m/s), the time to react to a reading, to back off and turn (s)
CORRIDOR_S = 300.0  # s, a corridor given up on
PLASTIC_MU = 0.35  # dry plastic on wood or concrete, from engineering friction tables (about 0.3 to 0.4)
SWERVE = 0.35  # rad off the line at most, going round something: what's ahead stays in the sensor's cone
BEAM = np.radians([0.0, -7.5, 7.5, -15.0, 15.0])  # an ultrasonic ranger's cone, as an HC-SR04's datasheet gives it
EDGES = ("step", "trench", "stairs", "bump", "threshold")  # obstacles met at an approach duty, that can throw it back
BEVEL = 30.0  # degrees: a threshold strip's two bevels
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
    if kind == "floor":  # flat ground, as long as the feature is
        return floor.format(g=g), f.get("len", 0.0)
    if kind == "step":
        h = f["h"] / 100
        return floor.format(g=g) + _block("obstacle", 0.0, 2.0, 0.0, h, g, +1), 0.0
    if kind == "stairs":  # n rises of h on treads of ``run``: the rear must be past the last nosing
        h, run, n = f["h"] / 100, f["run"] / 100, int(round(f["n"]))
        end = (n - 1) * run + 2.0
        return floor.format(g=g) + "".join(_block("obstacle" if i == 0 else f"stair{i}", i * run, end, 0.0, (i + 1) * h,
                                                  g, +1) for i in range(n)), (n - 1) * run
    if kind == "bump":  # a cable or hose of diameter d lying across the path
        r = f["d"] / 200
        return floor.format(g=g) + (f'<geom name="obstacle" type="capsule" size="{r:.5f} 0.6" pos="{r:.5f} 0 {r:.5f}" '
                                    f'zaxis="0 1 0" rgba="0.2 0.22 0.2 1" {g}/>'), 2 * r
    if kind == "threshold":  # a strip h high and w wide: a BEVEL ramp up, a flat top, a BEVEL ramp down
        h, w, th, t = f["h"] / 100, f["w"] / 100, math.radians(BEVEL), 0.01
        b = h / math.tan(th)
        top = max(w - 2 * b, 0.002)
        rgba = 'rgba="0.62 0.5 0.36 1"'

        def bevel(nm: str, x0: float, up: bool) -> str:
            n = np.array([-math.sin(th), math.cos(th)]) if up else np.array([math.sin(th), math.cos(th)])
            c = np.array([x0 + b / 2, h / 2]) - n * t / 2
            return (f'<geom name="{nm}" type="box" size="{math.hypot(b, h) / 2:.5f} 0.6 {t / 2:.5f}" '
                    f'euler="0 {-BEVEL if up else BEVEL:.4f} 0" pos="{c[0]:.5f} 0 {c[1]:.5f}" {rgba} {g}/>')
        return (floor.format(g=g) + bevel("obstacle", 0.0, True)
                + f'<geom name="strip_top" type="box" size="{top / 2:.5f} 0.6 {h / 2:.5f}" pos="{b + top / 2:.5f} 0 '
                  f'{h / 2:.5f}" {rgba} {g}/>' + bevel("strip_down", b + top, False)), 2 * b + top
    if kind == "trench":
        w, d = f["w"] / 100, f["depth"] / 100
        return (f'<geom name="floor" type="plane" size="3 1 0.1" pos="0 0 {-d:.5f}" rgba="0.4 0.38 0.35 1" {g}/>'
                + _block("near", -2.0, 0.0, -d, 0.0, g, -1) + _block("obstacle", w, w + 2.0, -d, 0.0, g, +1)), w
    if kind == "ramp":
        th = math.radians(f["deg"])
        ln = max(f.get("len", 0.0) + 0.5, 1.0)  # past its top, as long as the feature; 1 m for a robot from the catalogue
        return floor.format(g=g) + (
            f'<geom name="obstacle" type="box" size="{ln / 2:.5f} 0.6 0.01" euler="0 {-f['deg']:.4f} 0" '
            f'pos="{ln / 2 * math.cos(th) + 0.01 * math.sin(th):.5f} 0 {ln / 2 * math.sin(th) - 0.01 * math.cos(th):.5f}" '
            f'rgba="0.7 0.68 0.64 1" {g}/>'), (ln - 0.5) * math.cos(th)
    if kind == "low":  # a ceiling over the way, ``clear`` above the floor
        ln, clear = f.get("len", 0.8), f["clear"] / 100
        return floor.format(g=g) + (f'<geom name="obstacle" type="box" size="{ln / 2:.5f} 0.6 0.05" pos="{ln / 2:.5f} 0 {clear + 0.05:.5f}" '
                                    f'rgba="0.55 0.42 0.3 1" {g}/>'), ln
    if kind == "dark":  # a corridor's obstacles, spaced along it, either side of the line by turns
        ln, n = f.get("len", 2.0), int(round(f.get("n", 1)))
        return floor.format(g=g) + "".join(
            f'<geom name="{"obstacle" if i == 0 else f"obstacle{i}"}" type="box" size="{OBST[0]} {OBST[1]} 0.15" '
            f'pos="{(i + 0.5) * ln / n:.5f} {OBST[2] * (1 if i % 2 else -1):.5f} 0.15" rgba="0.55 0.42 0.3 1" {g}/>' for i in range(n)), ln
    if kind == "sand":  # a deep bed: its surface is drawn at z = 0, the soil's forces are ``_Soil``'s, and the hard floor is far below
        return (f'<geom name="floor" type="plane" size="3 1 0.1" pos="0 0 {-SAND_DEPTH:.5f}" rgba="0.4 0.38 0.35 1" {g}/>'
                '<geom name="sand_top" type="plane" size="3 1 0.1" rgba="0.8 0.72 0.52 1" contype="0" conaffinity="0"/>'), f.get("len", 0.6)
    if kind == "gap":  # two jambs, the opening centred ``off`` off the robot's line
        w, off = f["w"] / 100, f["off"] / 100
        jamb = '<geom name="{n}" type="box" size="0.02 0.25 0.15" pos="0.02 {y:.5f} 0.15" rgba="0.55 0.42 0.3 1" {g}/>'
        return floor.format(g=g) + jamb.format(n="obstacle", y=off + w / 2 + 0.25, g=g) + \
            jamb.format(n="jamb_r", y=off - w / 2 - 0.25, g=g), 0.04
    raise ValueError(kind)


class _Soil:
    """Soft ground under a graph robot (D-119): a Winkler soil, p = k z, in forces applied where each solid meets it.  A
    wheel bears on the contact patch the brief's area reports, half its radius long, and its traction is mu N limited
    by slip.  Compacting the soil resists its travel by k z^2 b / 2.  Anything else that reaches the soil bears on its
    own footprint and is dragged along by mu N, and it bulldozes the same way."""

    def __init__(self, m: mujoco.MjModel, mine: list[int], wheels: set[int], k_kpa_cm: float, mu: float):
        self.k, self.mu = k_kpa_cm * 1e5, mu  # Pa/m
        self.wheels, self.rest = [i for i in mine if i in wheels], [i for i in mine if i not in wheels]
        self.m, self.res, self.sink = m, np.zeros(6), 0.0

    def _apply(self, d: mujoco.MjData, i: int, z: float, b: float, ln: float, at: np.ndarray, slip: float, vel: np.ndarray) -> None:
        n = max(self.k * b * ln * (z - 0.05 * vel[2]), 0.0)  # soil doesn't bounce
        fx = -self.mu * n * math.tanh(slip / 0.03) - 0.5 * self.k * z**2 * b * math.tanh(vel[0] / 0.02)
        fy = -self.mu * n * math.tanh(vel[1] / 0.03)
        mujoco.mj_applyFT(self.m, d, np.array([fx, fy, n]), np.zeros(3), at, int(self.m.geom_bodyid[i]), d.qfrc_applied)

    def __call__(self, d: mujoco.MjData) -> None:
        m = self.m
        d.qfrc_applied[:] = 0.0
        deep = max((m.geom_size[i][0] - d.geom_xpos[i][2] for i in self.wheels), default=0.0)
        self.sink += (DT / 0.3) * (max(deep, 0.0) - self.sink)  # how deep it runs, not how it first dropped in
        for i in self.wheels:
            r, hw = m.geom_size[i][0], m.geom_size[i][1]
            c = d.geom_xpos[i]
            z = r - c[2]
            if z > 0:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i]), self.res, 0)
                vel = self.res[3:]
                self._apply(d, i, z, 2 * hw, 0.5 * r, np.array([c[0], c[1], -z]), vel[0] - self.res[1] * r, vel)
        for i in self.rest:
            c, rot = d.geom_xpos[i], d.geom_xmat[i].reshape(3, 3)
            s, t = m.geom_size[i], int(m.geom_type[i])
            local = s[:3] if t == int(mujoco.mjtGeom.mjGEOM_BOX) else np.array([s[0], s[0], s[1] + (s[0] if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0)])
            half = np.abs(rot) @ local
            z = half[2] - c[2]
            if z > 0:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, i, self.res, 0)
                vel = self.res[3:]
                self._apply(d, i, z, 2 * half[1], 2 * half[0], np.array([c[0], c[1], -z]), vel[0], vel)


def build(rb: Robot, kind: str, f: dict[str, float], mu: float) -> tuple[mujoco.MjModel, list[str], float]:
    tree, eqs, acts, driven = _robot_xml(rb)
    ground, past = terrain_xml(kind, f, mu)
    xml = f"""<mujoco model="mobile_{kind}">
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>
  <default>
    <default class="robot"><geom contype="{SELF[0] if rb.loco != 'tracks' else 2}" conaffinity="{SELF[1] if rb.loco != 'tracks' else 1}" friction="{0.5 * mu:.4f} 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}" condim="3"/></default>
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
    radii = [rb.r if rb.loco == "mecanum4" else m.geom(n).size[0] for n in driven]  # a mecanum hub rolls on rollers
    for j in range(1, m.njnt):  # every wheel and roller already turning at the robot's speed
        nm = m.joint(j).name
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and nm[0] in "wt" and "_r" not in nm:  # not a mecanum roller
            d.qvel[m.jnt_dofadr[j]] = v_ss / (rb.r if rb.loco == "mecanum4" else m.geom(nm).size[0])
    for k in range(m.nu):
        d.ctrl[k] = duty
    mujoco.mj_forward(m, d)
    goal = past - rear + 0.01  # the robot's centre once its rear is past
    t_max = min(15.0, 2.0 + 2.5 * (goal - x0) / max(v_ss, 0.1 * rb.omega0 * rb.r, 0.03))
    return _roll(m, d, kind, x0, goal, t_max, jid, radii, {m.geom("tub").id}, trace)


def _roll(m: mujoco.MjModel, d: mujoco.MjData, kind: str, x0: float, goal: float, t_max: float, jid: list[int],
          radii: list[float], belly_geoms: set[int], trace: bool, members: Any = None, soil: _Soil | None = None) -> Episode:
    """Run a robot set going at an obstacle until it is past, or what stopped it."""
    ts, qs = [], []
    run = {"v": float(d.qvel[0]), "power": 0.0, "work": 0.0}
    done = _done

    def _done_measured(*args: Any) -> Episode:
        return replace(done(*args), v=run["v"], power=run["power"], sink=soil.sink if soil else 0.0, elapsed=float(d.time), work=run["work"])

    best_x, hist = d.qpos[0], []
    belly = bounced = walled = False
    every = int(0.02 / DT)
    dofs = [m.jnt_dofadr[j] for j in jid]
    for i in range(int(t_max / DT)):
        if soil is not None:
            soil(d)
        mujoco.mj_step(m, d)
        a = DT / 0.3  # what it is doing now, not how it got going
        run["v"] += a * (float(d.qvel[0]) - run["v"])
        now = sum(abs(float((d.qfrc_actuator[k] + d.qfrc_passive[k]) * d.qvel[k])) for k in dofs)
        run["power"] += a * (now - run["power"])
        run["work"] += now * DT
        if members is not None:  # printed members, judged as on any robot (D-109)
            members.read(d)
            if broke := members.broken():
                return _done_measured(False, d.qpos[0], x0, goal, "broke", broke, ts, qs, m)
        if trace and i % every == 0:
            ts.append(d.time)
            qs.append(d.qpos.copy())
        x = d.qpos[0]
        best_x = max(best_x, x)
        if best_x - x > 0.01 and kind in EDGES:
            bounced = True
        for c in d.contact[:d.ncon]:
            hit = c.geom1 if c.geom1 in belly_geoms else c.geom2 if c.geom2 in belly_geoms else -1
            if hit >= 0 and m.geom_bodyid[c.geom1 if c.geom2 == hit else c.geom2] == 0:
                belly = True
            if kind in ("gap", "low") and (m.geom(c.geom1).name.startswith(("obstacle", "jamb"))
                                  or m.geom(c.geom2).name.startswith(("obstacle", "jamb"))):
                walled = True
        up = d.xmat[1].reshape(3, 3)[2, 2]
        if up < math.cos(math.radians(70)):
            return _done_measured(False, x, x0, goal, "tipped", "it rolled or pitched past 70 degrees", ts, qs, m)
        if x >= goal:
            return _done_measured(True, x, x0, goal, "", f"past in {d.time:.2f} s", ts, qs, m)
        if i % every == 0:
            hist.append(x)
            if len(hist) > 40 and hist[-1] - hist[-41] < 0.005:  # under 5 mm in the last 0.8 s
                spin = np.mean([abs(d.qvel[m.jnt_dofadr[j]]) * r for j, r in zip(jid, radii, strict=True)])
                if kind == "gap" or (kind == "low" and walled):
                    why = "wall"
                elif belly and kind in EDGES:
                    why = "belly"
                elif bounced and kind in EDGES:
                    why = "bounced"
                else:
                    why = "stuck_slipping" if spin > 0.03 else "stuck_stalled"
                return _done_measured(False, x, x0, goal, why, f"stuck {100 * (x - x0):.1f} cm along, its running gear "
                             f"turning at {spin:.2f} m/s", ts, qs, m)
    return _done_measured(False, d.qpos[0], x0, goal, "wall" if walled else "slow", f"not past after {t_max:.1f} s",
                 ts, qs, m)


def _done(ok: bool, x: float, x0: float, goal: float, why: str, detail: str, ts: list, qs: list, m) -> Episode:
    return Episode(ok, float(np.clip((x - x0) / (goal - x0), 0, 1)), why, detail, np.array(ts),
                   np.array(qs) if qs else np.zeros((0, 0)), m)


@cache
def cached(rb: Robot, kind: str, items: tuple[tuple[str, float], ...], mu: float, duty: float, y0: float,
           standing: bool = False) -> Episode:
    """``episode`` for search: its arguments arrive quantised, so attempts that differ by less share physics."""
    return episode(rb, kind, dict(items), mu, duty, y0, standing=standing)


def graph_model(g: Any, kind: str, f: dict[str, float], mu: float, kit_kg: float) -> tuple[mujoco.MjModel, float]:
    """An obstacle's terrain with the graph robot ``g`` on it, whose solids meet the terrain and not each other (D-111);
    and the x its rear must get past."""
    from sea.mech.base import WORLD
    from sea.mech.build import compile as compile_graph
    from sea.mech.build import solids_centre

    ground, past = terrain_xml(kind, f, mu)
    spec = mujoco.MjSpec.from_string(f"""<mujoco model="mobile_{kind}">
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default><default class="ground"><geom contype="1" conaffinity="2" solref="{SOLREF}" solimp="{SOLIMP}" condim="3"/></default></default>
  <worldbody>{ground}</worldbody>
</mujoco>""")
    compile_graph(spec, g, WORLD)
    names = {b.name for b in g.bodies}
    solref, solimp = [float(x) for x in SOLREF.split()], [*(float(x) for x in SOLIMP.split()), 0.5, 2.0]
    for x in spec.geoms:
        if x.parent.name in names and (x.contype or x.conaffinity):
            x.contype, x.conaffinity, x.solref, x.solimp = 2, 1, solref, solimp
    if kit_kg > 0:
        spec.body("robot").add_geom(name="kit_parts", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0.0, 0.0], pos=list(solids_centre(g.bodies[0].parts)),
                                    mass=kit_kg, contype=0, conaffinity=0, group=4)
    return spec.compile(), past


def graph_episode(graph_json: str, kind: str, f: dict[str, float], mu: float, duty: float, y0: float, crr: float, kit_kg: float,
                  standing: bool = False, trace: bool = False) -> Episode:
    """``episode`` for a robot built from a graph: its wheels' motors at ``duty``, each signed to drive it forward."""
    from sea.mech.build import printed_members
    from sea.mech.graph import Gearmotor, Graph, Wheel
    from sea.mech.members import PrintedMembers

    g = Graph.model_validate_json(graph_json)
    m, past = graph_model(g, kind, f, mu, kit_kg)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    wheels = {b.joints[0].name: p for b in g.bodies for p in b.parts if isinstance(p, Wheel) and b.joints}
    mine = [i for i in range(m.ngeom) if m.geom_contype[i] == 2]
    front = max(d.geom_xpos[i][0] + m.geom_rbound[i] for i in mine)
    rear = min(d.geom_xpos[i][0] - m.geom_rbound[i] for i in mine)
    x0 = -front - (0.002 if standing else 0.02)
    d.qpos[0:3] = (x0, y0, d.qpos[2])
    drive = [a for a in g.actuators if isinstance(a, Gearmotor) and a.joint in wheels]
    stall = sum(a.stall_nm / wheels[a.joint].radius for a in drive) or 1e-6
    v0 = min((a.no_load_rad_s * wheels[a.joint].radius for a in drive), default=0.03)
    mass = float(m.body_subtreemass[m.body("robot").id])
    v_ss = 0.0 if standing else max(0.0, v0 * (duty - crr * mass * 9.81 / stall))
    jid = [m.joint(a.joint).id for a in drive]
    radii = [wheels[a.joint].radius for a in drive]
    for a, j in zip(drive, jid, strict=True):
        axis = d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]
        sign = 1.0 if axis[1] >= 0 else -1.0  # a wheel on this axis rolls toward +x when it turns about +y
        d.ctrl[m.actuator(a.name).id] = duty * sign
        d.qvel[m.jnt_dofadr[j]] = sign * v_ss / wheels[a.joint].radius
    d.qvel[0] = v_ss
    mujoco.mj_forward(m, d)
    goal = past - rear + 0.01
    t_max = min(15.0, 2.0 + 2.5 * (goal - x0) / max(v_ss, 0.1 * v0, 0.03))
    wheel_geoms = {p.name for p in wheels.values()}
    for i in mine:  # only a wheel has a tyre: anything else meets the ground as what it is (D-123)
        if m.geom(i).name not in wheel_geoms:
            m.geom_priority[i], m.geom_friction[i, 0] = 2, min(mu, PLASTIC_MU)
    dofs = [m.jnt_dofadr[j] for j in jid]
    if kind == "floor" and f.get("len", 0.0) > 0:
        _friction(m, dofs, radii, crr, mass)
        return _cruise(m, d, x0, goal, max(t_max, 3.0 * goal / max(v_ss, 0.05) + 10.0), dofs, f.get("cap", 0.0), trace)
    if kind == "dark":
        _friction(m, dofs, radii, crr, mass)
        sides = [(m.actuator(a.name).id, float(np.sign(d.ctrl[m.actuator(a.name).id]) or 1.0), 1 if d.xpos[m.jnt_bodyid[m.joint(a.joint).id]][1] > 0 else -1) for a in drive]
        lo, hi = min((d.geom_xpos[i][1] - m.geom_rbound[i] for i in mine), default=0.0), max((d.geom_xpos[i][1] + m.geom_rbound[i] for i in mine), default=0.0)
        return _corridor(m, d, g, x0, goal, duty, sides, dofs, front - rear, (hi - lo) / 2, trace)
    soil = _Soil(m, mine, {i for i in mine if m.geom(i).name in wheel_geoms}, f["kmod"], mu) if kind == "sand" else None
    if soil is None:  # soil brings its own
        _friction(m, dofs, radii, crr, mass)
    return _roll(m, d, kind, x0, goal, t_max, jid, radii, {i for i in mine if m.geom(i).name not in wheel_geoms}, trace,
                 PrintedMembers(m, found) if (found := printed_members(g, m)) else None, soil)


def _friction(m: mujoco.MjModel, dofs: list[int], radii: list[float], crr: float, mass: float) -> None:
    """Rolling resistance, on the wheels' own joints."""
    for k, r in zip(dofs, radii, strict=True):
        m.dof_frictionloss[k] = crr * mass * 9.81 * r / max(len(dofs), 1)


def _cruise(m: mujoco.MjModel, d: mujoco.MjData, x0: float, goal: float, t_max: float, dofs: list[int], cap: float, trace: bool) -> Episode:
    """Plain ground, driven its whole length: a long way, so stepped in C, and read every 0.1 s.  ``cap``: the J its pack
    could ever deliver, past which there's no use going on."""
    ts, qs, work, best, last = [], [], 0.0, d.qpos[0], 0.0
    power = 0.0
    m.opt.timestep = 2 * DT  # as coarse as the contact's time constant allows: there is nothing here to climb
    chunk = 50
    n = int(t_max / (chunk * m.opt.timestep))
    for _ in range(n):
        mujoco.mj_step(m, d, nstep=chunk)
        power = sum(abs(float((d.qfrc_actuator[k] + d.qfrc_passive[k]) * d.qvel[k])) for k in dofs)
        work += power * chunk * m.opt.timestep
        x = float(d.qpos[0])
        if trace and len(ts) < 1000:
            ts.append(d.time)
            qs.append(d.qpos.copy())
        if x >= goal or work >= cap > 0:
            return replace(_done(x >= goal, x, x0, goal, "" if x >= goal else "drained", f"past in {d.time:.1f} s", ts, qs, m),
                           v=float(d.qvel[0]), power=power, elapsed=float(d.time), work=work)
        if x - best > 0.005:
            best, last = x, d.time
        elif d.time - last > 1.0:
            return replace(_done(False, x, x0, goal, "stuck_stalled", f"stuck {100 * (x - x0):.1f} cm along", ts, qs, m),
                           v=float(d.qvel[0]), power=power, elapsed=float(d.time), work=work)
    return replace(_done(False, float(d.qpos[0]), x0, goal, "slow", f"not past after {t_max:.0f} s", ts, qs, m),
                   v=float(d.qvel[0]), power=power, elapsed=float(d.time), work=work)


def _corridor(m: mujoco.MjModel, d: mujoco.MjData, g: Any, x0: float, goal: float, duty: float, drive: list[tuple[int, float, int]],
              dofs: list[int], length: float, half_w: float, trace: bool) -> Episode:
    """A dark corridor driven whole (D-119).  Its sensors read what is ahead by rays from where they are mounted and aimed.
    Seeing something with room to go round, it steers past on the nearer side by driving its wheels unequally; too close for
    that, a reaction time later it brakes and backs up until there is room; with nothing seen, it learns of an obstacle by
    touching it.  A touch above the
    bumper's rating fails, and so does getting nowhere."""
    from sea.mech.graph import RangeFinder

    rays = []
    for s in g.sensors:
        if isinstance(s, RangeFinder):
            q = np.zeros(9)
            mujoco.mju_quat2Mat(q, np.array(s.quat))
            rays.append((m.body(s.body).id, np.array(s.pos), q.reshape(3, 3)[:, 2], s.max_m))
    names = {i: m.geom(i).name for i in range(m.ngeom)}
    obstacles = {i for i, n in names.items() if n.startswith("obstacle")}
    mine = {i for i in range(m.ngeom) if m.geom_contype[i] == 2}
    ts, qs, work, power, geom = [], [], 0.0, 0.0, np.zeros(1, np.int32)
    mask = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # the terrain's geom group, not the robot's own solids
    chunk, dt = 10, 10 * DT
    mode, until, target_y, passing = "cruise", 0.0, 0.0, None
    best, last, tries = float(d.qpos[0]), 0.0, 0
    before, back_to = d.qvel[:3].copy(), float(d.qpos[0])

    def done(ok: bool, why: str, detail: str) -> Episode:
        return replace(_done(ok, float(d.qpos[0]), x0, goal, why, detail, ts, qs, m), v=abs(float(d.qvel[0])), power=power,
                       elapsed=float(d.time), work=work)

    def wheels(left: float, right: float) -> None:
        for i, sign, side in drive:
            d.ctrl[i] = sign * float(np.clip(left if side > 0 else right, -1.0, 1.0))

    while d.time < CORRIDOR_S:
        x, y, v = float(d.qpos[0]), float(d.qpos[1]), float(d.qvel[0])
        w, _, _, z = d.qpos[3:7]
        yaw = 2.0 * math.atan2(z, w)
        hit = next((c for c in d.contact[:d.ncon]
                    if (c.geom1 in obstacles and c.geom2 in mine) or (c.geom2 in obstacles and c.geom1 in mine)), None)
        touching = None if hit is None else (hit.geom1 if hit.geom1 in obstacles else hit.geom2)
        if touching is not None and mode != "back":
            into = abs(float(before @ hit.frame[:3]))  # into the obstacle, as it arrived: brushing past at speed is no impact
            if into > BUMP_V:
                return done(False, "wall", f"met an obstacle at {into:.2f} m/s; a bumper is rated for {BUMP_V:.2f} m/s")
            passing, tries = touching, tries + 1
            if tries > 6 * len(obstacles):
                return done(False, "wall", f"kept meeting obstacles, {tries} times")
            target_y = _clear_line(m, d, touching, half_w)
            mode, until = "back", d.time + BACKOFF_S
            back_to = x - abs(target_y - y) / math.tan(SWERVE) - 0.1
        elif mode == "cruise":
            for bid, pos, axis, reach in rays:
                rot = d.xmat[bid].reshape(3, 3)
                aim = rot @ axis
                for off in BEAM:  # a sensor sees a cone, not a line: across it, in the plane of the floor
                    c, s_ = math.cos(off), math.sin(off)
                    r = mujoco.mj_ray(m, d, d.xpos[bid] + rot @ pos, np.array([c * aim[0] - s_ * aim[1], s_ * aim[0] + c * aim[1], aim[2]]),
                                      mask, 1, -1, geom)
                    seen = int(geom[0])
                    if not (0 <= r <= reach and seen in obstacles and seen != passing):
                        continue
                    stop = v**2 / (2 * 0.5 * 9.81) + v * REACT_S + 0.05 + half_w * abs(math.sin(yaw))  # to stop, its corner included
                    aside = _clear_line(m, d, seen, half_w)
                    swerve = abs(aside - y) / math.tan(SWERVE) + stop
                    if r >= swerve and (passing is None or x > d.geom_xpos[passing][0] + OBST[0] + length):  # room to go round
                        passing, target_y = seen, aside
                    elif r < swerve:  # too late to go round: stop, and back up until there is room
                        passing, target_y, mode, until = seen, aside, "brake", d.time + REACT_S
                        back_to = float(d.geom_xpos[seen][0]) - OBST[0] - length - abs(aside - y) / math.tan(SWERVE) - 0.1
                        break
                if mode == "brake":
                    break
        if mode == "brake" and d.time >= until:
            wheels(0.0, 0.0)
            if abs(v) < 0.02:
                mode, until = "back", d.time + BACKOFF_S
        elif mode == "back":
            wheels(-0.5, -0.5)
            if x <= back_to or d.time >= until:
                mode = "cruise"
        elif mode == "cruise":
            if passing is not None and x > d.geom_xpos[passing][0] + OBST[0] + length:
                passing, target_y = None, 0.0
            want = float(np.clip(-4.0 * (y - target_y), -SWERVE, SWERVE))
            steer = float(np.clip(1.5 * (want - yaw), -0.6, 0.6))
            wheels(duty - steer, duty + steer)
        before = d.qvel[:3].copy()
        mujoco.mj_step(m, d, nstep=chunk)
        power = sum(abs(float((d.qfrc_actuator[k] + d.qfrc_passive[k]) * d.qvel[k])) for k in dofs)
        work += power * dt
        if trace and int(d.time / dt) % 2 == 0:
            ts.append(d.time)
            qs.append(d.qpos.copy())
        if d.xmat[1].reshape(3, 3)[2, 2] < math.cos(math.radians(70)):
            return done(False, "tipped", "it rolled or pitched past 70 degrees")
        if d.qpos[0] >= goal:
            return done(True, "", f"through in {d.time:.1f} s, touching {tries} times")
        if d.qpos[0] - best > 0.01:
            best, last = float(d.qpos[0]), d.time
        elif d.time - last > 20.0:
            return done(False, "stuck_stalled", f"got nowhere for 20 s at {100 * (d.qpos[0] - x0):.0f} cm along")
    return done(False, "slow", f"not through after {CORRIDOR_S:.0f} s")


def _clear_line(m: mujoco.MjModel, d: mujoco.MjData, obstacle: int, half_w: float) -> float:
    """The line past an obstacle on whichever side of it is nearer."""
    c, y = float(d.geom_xpos[obstacle][1]), float(d.qpos[1])
    return c + (1.0 if y >= c else -1.0) * (OBST[1] + half_w + 0.03)


@cache
def graph_cached(graph_json: str, kind: str, items: tuple[tuple[str, float], ...], mu: float, duty: float, y0: float, crr: float,
                 kit_kg: float, standing: bool = False) -> Episode:
    return graph_episode(graph_json, kind, dict(items), mu, duty, y0, crr, kit_kg, standing)
