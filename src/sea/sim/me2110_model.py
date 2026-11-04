"""The ME 2110 match in MuJoCo (D-073): the arena from the brief's figures, the robot from kit actuators.

Units: the brief is in inches, so geometry here is written in inches and converted (``IN``); MuJoCo runs in SI.
The start line is x = 0 and the robot faces +x toward the fortress; +y is the robot's left.

**The arena** (``ME2110_Fall_2026_Final_Project_v1.3.pdf``):
- the fortress, 24 in across and 13 in tall, centred 39.6 in out (chest 19.9 in out, 4.2 in deep, 3.5 in short of
  the fortress), turning clockwise seen from above; its wall has four 4 in cannon ports, each a barrier at its
  colour's tab height with a receptacle behind it; the 4 in square flagpole stands to 72 in on its top, inside
  the tentacles (14 in across, to 24 in);
- 10 coins (2.5 x 0.5 in) and 5 jewels (1.5 in) on the fortress top's rim;
- the mermaid (3.5 in base) and merman (3.7 in), 7 in tall, 20 in out and 20 in apart, each on a 4 in mark;
- the chest (4.7 x 3.2 in, 2.4 in body under a 1.2 in lid), lid propped open, the heart (1.35 x 2.5 in) inside
  0.5 in below the lip; the prop gives way when the chest is jostled (moved 0.4 in or tilted 6 degrees);
- the gold mound (left) and jewel mound (right): 3.5 x 4 in tops, 1.5 in tall, a 1 in scoring border.

**The robot** is real mechanisms on the kit's actuators:
- gearmotors on a DC torque-speed line, tau = T_s (duty - omega / omega_0), clamped at stall; stopped, a motor is
  shorted (its back-EMF brakes it, with no power);
- a piston is a cylinder on a pin, its rod on a slide, the rod's end pinned to the lever it drives (a closed loop:
  a connect equality).  It pushes P A at the tank's pressure through a 5/2 valve (unpowered: retracted), flow-
  limited by the needle valve, and every stroke spends the tank's air (isothermal);
- a mousetrap is its spring's torque-angle line on the arm it swings; a rubber band is a spring;
- a latch is a joint equality released at a time; a magnet and a clip are welds that let go above their
  holding force; the chest's prop is an equality that gives way when the chest is jostled.

``run`` plays a match (or the first ``horizon`` seconds of one) under a controller built from System 1's controls,
and returns each task's points, what happened, and the laws the run broke.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

IN = 0.0254
DT = 0.002
SOLREF, SOLIMP = "0.004 1", "0.95 0.99 0.001"
G = 9.81

# ---------------------------------------------------------------- the arena (inches)
XF, RF, HF = 39.625, 12.0, 13.0  # fortress centre x, radius, height
TOP_T = 1.0  # the top disc's thickness
CORE_R = 8.0  # the receptacles lie between the core and the wall
PORT_W = 4.0  # a cannon port's width
POLE_HALF, POLE_TOP = 2.0, 72.0
TENT_R, TENT_TOP = 6.3, 24.0
COIN_R, COIN_T, JEWEL_R, JEWEL_T = 1.25, 0.5, 0.75, 1.0  # a jewel: a cut stone on its flat table
COIN_RING, JEWEL_RING = 10.6, 10.9
MERFOLK = {"mermaid": (20.0, 10.0, 1.75), "merman": (20.0, -10.0, 1.85)}  # x, y, base radius
MARK_R = 2.0
CHEST_X, CHEST_D, CHEST_W, CHEST_H, LID_H, WALL = 19.925, 3.2, 4.7, 2.4, 1.2, 0.25
LID_OPEN = math.radians(80.0)
HEART_R, HEART_L = 0.675, 2.5
MOUNDS = {"gold": (14.0, 15.0), "jewel": (14.0, -15.0)}  # centres; tops 3.5 (x) by 4 (y), 1.5 tall
MOUND_X, MOUND_Y, MOUND_H, BORDER = 3.5, 4.0, 1.5, 1.0
BALL_R = 0.020 / IN  # a 40 mm ping-pong ball
DOCK = 30.0

# masses (kg), assumed: the brief gives none
M_COIN, M_JEWEL, M_BALL, M_HEART = 0.025, 0.012, 0.0027, 0.020
# the treasure chute is lined with acrylic sheet and the coins and jewels are moulded plastic: about 0.3 between them
# (at 0.5, printed PLA, no sliding route from the fortress top to a mound fits in its 11.5 in of drop)
SLICK_MU = PIECE_MU = 0.3
ACRYLIC = "0.75 0.85 0.95 0.6"
M_FOLK = 0.20  # a printed figure on its base
M_CHEST, M_LID = 0.15, 0.04

# ---------------------------------------------------------------- the kit (assumed: the brief gives no datasheets)
MOTOR_TS, MOTOR_W0, MOTOR_A = 1.2, 16.8, 2.5  # N m stall at the output, rad/s no-load (160 rpm), A stall, at 12 V
BORE, ROD, STROKE = 0.75, 0.3125, 4.0  # in
P_TANK0, TANK_L = 6.9e5, 1.0  # Pa gauge (100 psi), litres
PISTON_V = 0.3  # m/s, the needle valve's limit on a stroke's speed
CYL_CLOSED = 5.5  # in, a cylinder's pin-to-pin length retracted
TRAP_T0, TRAP_T1, TRAP_ARM = 0.45, 0.20, 1.8  # N m at the set and sprung ends of 180 degrees; its own arm (in)
BAND_K, BAND_L0 = 40.0, 3.5  # N/m per band, free length (in); stretched to at most 3 times
HOOK_FOLD_DEG = 95.0  # the heart's hook boom stands just past upright when retracted
MIN_TRANSMISSION_DEG = 20.0  # the least angle between a piston's rod and the boom it swings
VALVE_C = P_TANK0 * math.pi * (BORE * IN / 2) ** 2 / PISTON_V  # N s/m: the needle valve, as a damper on the rod
END_HOLD = 5.0  # N: what keeps a rod seated at an end of its stroke (the end cap reacts the rest of P A)
RAMP_S = 0.4  # s: the drive's duty ramp
IR_RANGE = 10.0  # in: the active IR sensor's reach
FLAG_SENSOR_Z, FLAG_IR_RANGE = 14.0, 60.0  # in: the sensed boom's long-range IR sensor, its height and reach
FACE_SQUARE_DEG = 2.0  # how near square a pole face must turn for the sensor's range reading to bottom out
MAGNET_N, CLIP_N = 3.0, 1.0  # N: the flag's magnet on the pole, the clip that holds it to the robot
HOLD_TICKS = 3  # event ticks (20 ms each) a clip or magnet must be overloaded before it lets go
PROP_MOVE, PROP_TILT = 0.4, math.radians(6.0)

# points (the brief's Table 1)
MAX_POINTS = {"activate": 11, "treasure": 175, "flag": 40, "cannon": 36, "mermaids": 24, "heart": 61}


def _f(v: float) -> str:
    return f"{v:.5f}"


def _v(*xs: float) -> str:
    return " ".join(_f(x) for x in xs)


@dataclass(frozen=True)
class Variant:
    """One track: our port's barrier height (its colour), the fortress's speed, and the plywood's friction."""

    barrier: float = 7.0  # in
    rpm: float = 3.0
    mu: float = 0.5
    phase: float = 0.0  # rad: our port's angle at t = 0, from the -x direction (facing us), clockwise
    others: tuple[float, float, float] = (6.0, 8.0, 8.75)


@dataclass(frozen=True)
class Robot:
    """A body resolved to geometry, in inches (``sea.worlds.me2110._robot``).  ``parts`` maps a slot to (type, params)."""

    L: float  # frame length along x
    W: float  # frame width along y
    H: float  # deck height above the floor
    drive: str
    wheel_r: float = 1.5
    axle_r: float = 0.25  # mousetrap car: the string's axle radius
    lever: float = 4.0  # mousetrap car: the lever arm the string runs from
    mermaid: tuple[str, dict[str, float]] = ("push_bar", {"bar_z": 4.0, "bar_w": 12.0})
    cannon: tuple[str, dict[str, float]] = ("no_cannon", {})
    flag: tuple[str, dict[str, float]] = ("no_flag", {})
    heart: tuple[str, dict[str, float]] = ("no_heart", {})
    treasure: tuple[str, dict[str, float]] = ("no_treasure", {})
    release: str = "drive_trip"
    frame_kg: float = 1.0  # the frame, electronics and anything not modelled as a moving part


BELLY = 1.0  # in: the frame's base plate above the floor


# ---------------------------------------------------------------- XML helpers
def _box(name: str, half: tuple[float, float, float], pos: tuple[float, float, float], rgba: str, cls: str,
         mass: float | None = None, euler: tuple[float, float, float] | None = None, extra: str = "") -> str:
    nm = f'name="{name}" ' if name else ""
    e = f'euler="{_v(*euler)}" ' if euler else ""
    m = f'mass="{mass:.5f}" ' if mass is not None else ""
    return (f'<geom {nm}type="box" size="{_v(*(h * IN for h in half))}" pos="{_v(*(p * IN for p in pos))}" {e}{m}'
            f'rgba="{rgba}" class="{cls}" {extra}/>')


def _cap(name: str, a: tuple[float, float, float], b: tuple[float, float, float], r: float, rgba: str, cls: str,
         mass: float | None = None) -> str:
    nm = f'name="{name}" ' if name else ""
    m = f'mass="{mass:.5f}" ' if mass is not None else ""
    return (f'<geom {nm}type="capsule" size="{_f(r * IN)}" fromto="{_v(*(x * IN for x in (*a, *b)))}" {m}'
            f'rgba="{rgba}" class="{cls}"/>')


WOOD, PLY, PRINT, METAL, RUBBER = "0.72 0.56 0.36 1", "0.83 0.74 0.58 1", "0.30 0.45 0.70 1", "0.62 0.63 0.66 1", \
    "0.12 0.12 0.13 1"


# ---------------------------------------------------------------- the arena
def _fortress(v: Variant) -> str:
    """The fortress body: core, wall with four ports (barrier + receptacle), top disc, pole and tentacles."""
    g = [f'<geom name="f_core" type="cylinder" size="{_f(CORE_R * IN)} {_f((HF - TOP_T) / 2 * IN)}" '
         f'pos="0 0 {_f((HF - TOP_T) / 2 * IN)}" mass="20" rgba="0.15 0.15 0.16 1" class="arena"/>',
         f'<geom name="f_top" type="cylinder" size="{_f(RF * IN)} {_f(TOP_T / 2 * IN)}" '
         f'pos="0 0 {_f((HF - TOP_T / 2) * IN)}" mass="3" rgba="0.18 0.18 0.2 1" class="arena" friction="0.4 0.005 0.0001"/>',
         f'<geom name="f_base" type="cylinder" size="{_f(RF * IN)} {_f(0.25 * IN)}" pos="0 0 {_f(0.25 * IN)}" mass="3" '
         f'rgba="0.15 0.15 0.16 1" class="arena"/>']
    half = PORT_W / 2 / RF  # a port's angular half-width
    ports = [math.pi + k * math.pi / 2 for k in range(4)]  # in the fortress frame; port 0 faces -x at phase 0
    n = 48
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        if min(abs((a - p + math.pi) % (2 * math.pi) - math.pi) for p in ports) < half + math.pi / n:
            continue
        g.append(_box("", (0.25, RF * math.pi / n + 0.05, (HF - TOP_T - 0.5) / 2),
                      ((RF - 0.25) * math.cos(a), (RF - 0.25) * math.sin(a), 0.5 + (HF - TOP_T - 0.5) / 2),
                      "0.15 0.15 0.16 1", "arena", 0.2, (0, 0, a)))
    for k, p in enumerate(ports):
        h = v.barrier if k == 0 else v.others[k - 1]
        g.append(_box(f"barrier{k}", (0.25, PORT_W / 2 + 0.3, h / 2), ((RF - 0.25) * math.cos(p), (RF - 0.25) * math.sin(p),
                                                                        h / 2), "0.85 0.2 0.2 1" if k == 0 else "0.5 0.5 0.5 1",
                      "arena", 0.1, (0, 0, p)))
        for s in (-1, 1):  # the receptacle's radial dividers
            a = p + s * (half + 0.08)
            r = (CORE_R + RF) / 2
            g.append(_box("", ((RF - CORE_R) / 2, 0.12, (HF - TOP_T) / 2), (r * math.cos(a), r * math.sin(a), (HF - TOP_T) / 2),
                          "0.15 0.15 0.16 1", "arena", 0.05, (0, 0, a)))
        g.append(_box(f"tab{k}", (0.3, 0.5, 0.5), ((RF + 0.3) * math.cos(p), (RF + 0.3) * math.sin(p), HF - 1.5),
                      "0.95 0.85 0.1 1" if k == 0 else "0.4 0.4 0.4 1", "arena", 0.01, (0, 0, p), 'contype="0" conaffinity="0"'))
    g.append(_box("pole", (POLE_HALF, POLE_HALF, (POLE_TOP - HF) / 2), (0, 0, (POLE_TOP + HF) / 2), "0.35 0.35 0.38 1",
                  "arena", 2.0))
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        g.append(_cap("", (TENT_R * math.cos(a), TENT_R * math.sin(a), HF), (TENT_R * math.cos(a), TENT_R * math.sin(a),
                                                                            TENT_TOP - 0.7), 0.7, "0.45 0.2 0.6 1", "arena", 0.1))
    return (f'<body name="fortress" pos="{_f(XF * IN)} 0 0" euler="0 0 {_f(-v.phase)}">'
            f'<joint name="spin" type="hinge" axis="0 0 1" damping="0"/>' + "".join(g) + "</body>")


def _figures() -> str:
    """Coins and jewels on the top's rim, every 24 degrees, a jewel every third place."""
    out = []
    for k in range(15):
        a = 2 * math.pi * k / 15 + math.pi / 15
        if k % 3 == 1:
            x, y, z = XF + JEWEL_RING * math.cos(a), JEWEL_RING * math.sin(a), HF + JEWEL_T / 2
            out.append(f'<body name="jewel{k // 3}" pos="{_v(x * IN, y * IN, z * IN)}"><freejoint/>'
                       f'<geom type="cylinder" size="{_f(JEWEL_R * IN)} {_f(JEWEL_T / 2 * IN)}" mass="{M_JEWEL}" '
                       f'rgba="0.2 0.85 0.9 1" class="item" friction="{PIECE_MU} 0.005 0.0001"/></body>')
        else:
            x, y, z = XF + COIN_RING * math.cos(a), COIN_RING * math.sin(a), HF + COIN_T / 2
            out.append(f'<body name="coin{k - k // 3 - (1 if k % 3 == 2 else 0)}" pos="{_v(x * IN, y * IN, z * IN)}"><freejoint/>'
                       f'<geom type="cylinder" size="{_f(COIN_R * IN)} {_f(COIN_T / 2 * IN)}" mass="{M_COIN}" '
                       f'rgba="0.95 0.78 0.15 1" class="item" friction="{PIECE_MU} 0.005 0.0001"/></body>')
    return "".join(out)


def _merfolk() -> str:
    out = []
    for nm, (x, y, r) in MERFOLK.items():
        out.append(f'<body name="{nm}" pos="{_v(x * IN, y * IN, 0)}"><freejoint/>'
                   f'<geom type="cylinder" size="{_f(r * IN)} {_f(0.25 * IN)}" pos="0 0 {_f(0.25 * IN)}" mass="{0.45 * M_FOLK}" '
                   f'rgba="0.3 0.6 0.4 1" class="item"/>'
                   + _cap("", (0, 0, 1.2), (0, 0, 6.3), 0.7, "0.25 0.7 0.55 1", "item", 0.55 * M_FOLK) + "</body>")
    return "".join(out)


def _chest() -> str:
    d, w, h, t = CHEST_D, CHEST_W, CHEST_H, WALL
    c = "0.62 0.45 0.25 1"
    walls = [_box("chest_floor", (d / 2, w / 2, t / 2), (0, 0, t / 2), c, "item", 0.5 * M_CHEST),
             _box("", (t / 2, w / 2, h / 2), (-d / 2 + t / 2, 0, h / 2), c, "item", 0.12 * M_CHEST),
             _box("", (t / 2, w / 2, h / 2), (d / 2 - t / 2, 0, h / 2), c, "item", 0.12 * M_CHEST),
             _box("", (d / 2, t / 2, h / 2), (0, w / 2 - t / 2, h / 2), c, "item", 0.12 * M_CHEST),
             _box("", (d / 2, t / 2, h / 2), (0, -w / 2 + t / 2, h / 2), c, "item", 0.12 * M_CHEST),
             _box("cushion", (d / 2 - t, w / 2 - t, 0.15), (0, 0, t + 0.15), "0.5 0.1 0.1 1", "item", 0.01)]
    lid = (f'<body name="lid" pos="{_v(d / 2 * IN, 0, h * IN)}">'
           f'<joint name="lid" type="hinge" axis="0 -1 0" range="0 {_f(LID_OPEN + 0.05)}" limited="true" damping="0.002"/>'
           + _box("", (d / 2, w / 2, t / 2), (-d / 2, 0, t / 2), c, "item", 0.6 * M_LID)
           + _box("", (d / 2, t / 2, LID_H / 2), (-d / 2, w / 2 - t / 2, LID_H / 2), c, "item", 0.2 * M_LID)
           + _box("", (d / 2, t / 2, LID_H / 2), (-d / 2, -w / 2 + t / 2, LID_H / 2), c, "item", 0.2 * M_LID) + "</body>")
    cz = h - 0.5 - HEART_R  # the heart's top 0.5 in below the lip
    heart = (f'<body name="heart" pos="{_v((CHEST_X + d / 2) * IN, 0, cz * IN)}"><freejoint/>'
             + _cap("heart", (0, -(HEART_L / 2 - HEART_R), 0), (0, HEART_L / 2 - HEART_R, 0), HEART_R, "0.75 0.05 0.12 1",
                    "item", M_HEART) + "</body>")
    return (f'<body name="chest" pos="{_v((CHEST_X + d / 2) * IN, 0, 0)}"><freejoint/>' + "".join(walls) + lid + "</body>"
            + heart)


ARENA = 84.0  # in: the square, centred on the fortress


def _walls() -> str:
    """The 2 x 4 lumber border (1.5 in tall, 3.5 in wide), open at the dock's mouth, and the dock's walls."""
    h, t, half = 1.5, 3.5, ARENA / 2
    x0, x1 = XF - half, XF + half
    c = "0.6 0.45 0.3 1"
    out = [_box("", (half + t, t / 2, h / 2), (XF, s * (half + t / 2), h / 2), c, "arena") for s in (-1, 1)]
    out.append(_box("", (t / 2, half, h / 2), (x1 + t / 2, 0, h / 2), c, "arena"))
    seg = (half - DOCK / 2) / 2
    out += [_box("", (t / 2, seg, h / 2), (x0 - t / 2, s * (DOCK / 2 + seg), h / 2), c, "arena") for s in (-1, 1)]
    out += [_box("", (DOCK / 2, t / 2, h / 2), (x0 - DOCK / 2, s * (DOCK / 2 + t / 2), h / 2), c, "arena") for s in (-1, 1)]
    out.append(_box("", (t / 2, DOCK / 2 + t, h / 2), (x0 - DOCK - t / 2, 0, h / 2), c, "arena"))
    return "".join(out)


def _mounds() -> str:
    out = []
    for nm, (x, y) in MOUNDS.items():
        out.append(_box(f"mound_{nm}", (MOUND_X / 2, MOUND_Y / 2, MOUND_H / 2), (x, y, MOUND_H / 2),
                        "0.95 0.8 0.2 1" if nm == "gold" else "0.85 0.2 0.8 1", "arena"))
        bx, by = MOUND_X / 2 + BORDER, MOUND_Y / 2 + BORDER
        out.append(_box("", (bx, by, 0.01), (x, y, 0.01), "0.95 0.9 0.6 1" if nm == "gold" else "0.95 0.75 0.95 1",
                        "arena", None, None, 'contype="0" conaffinity="0"'))
    return "".join(out)


# ---------------------------------------------------------------- the robot's mechanisms
@dataclass
class Build:
    """What the controller needs to know about the robot it built."""

    xml: str = ""
    eqs: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    latches: dict[str, list[str]] = field(default_factory=dict)  # mechanism -> its latch equalities
    pistons: dict[str, dict[str, Any]] = field(default_factory=dict)  # mechanism -> geometry, its rod joint and travel
    traps: dict[str, list[str]] = field(default_factory=dict)  # mechanism -> the joints its mousetraps drive
    wheels: list[str] = field(default_factory=list)
    balls: list[str] = field(default_factory=list)
    flag: bool = False
    flag_sensed: bool = False
    part_of: dict[str, str] = field(default_factory=dict)  # robot geom name -> the mechanism it belongs to
    moves: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)  # mechanism -> (joint, stowed, deployed)


def boom_pin_angle(theta0: float, theta1: float) -> float:
    """Where the cylinder's pin sits about the boom's pivot: 90 degrees beyond the middle of the swing, so that the
    rod, the pivot and the rod's pin come into line (toggle) only outside any swing of less than 180 degrees."""
    return (theta0 + theta1) / 2 + math.pi / 2


def boom_geometry(theta0: float, theta1: float) -> tuple[float, float]:
    """A piston driving a boom about its pivot from ``theta0`` (retracted) to ``theta1`` (extended, ``theta1 <
    theta0``), angles from +x in the x-z plane.  The cylinder's pin sits a distance c from the pivot at
    ``boom_pin_angle``, the rod's pin a distance b along the boom: rod length^2 = b^2 + c^2 - 2 b c cos(theta - phi),
    which grows monotonically as theta falls from theta0 to theta1.  Returns (b, c) in inches so that the retracted
    length is CYL_CLOSED and the stroke is STROKE."""
    phi = boom_pin_angle(theta0, theta1)
    c0, c1 = math.cos(theta0 - phi), math.cos(theta1 - phi)
    lo, hi = 0.05, 20.0
    for _ in range(80):  # bisection on b: l(theta1) - l(theta0) - STROKE
        b = (lo + hi) / 2
        c = _c_for(b, c0)
        l1 = math.sqrt(max(b * b + c * c - 2 * b * c * c1, 0.0))
        lo, hi = (b, hi) if l1 - CYL_CLOSED < STROKE else (lo, b)
    b = (lo + hi) / 2
    return b, _c_for(b, c0)


def boom_pins(theta0: float, theta1: float) -> tuple[float, float, float]:
    """(b, c, phi): the rod's pin along the boom, the cylinder's pin from the pivot (inches) and its angle."""
    return (*boom_geometry(theta0, theta1), boom_pin_angle(theta0, theta1))


def boom_transmission(theta0: float, theta1: float, n: int = 64) -> float:
    """The least angle (degrees) between the rod and the boom over the swing: the piston's push across the boom goes
    as its sine, so near 0 the rod pushes along the boom (toggle) and cannot steer it."""
    b, c, phi = boom_pins(theta0, theta1)
    worst = 90.0
    for k in range(n + 1):
        th = theta0 + (theta1 - theta0) * k / n
        e = np.array([math.cos(th), math.sin(th)])
        u = b * e - c * np.array([math.cos(phi), math.sin(phi)])  # the rod pin less the cylinder pin
        worst = min(worst, math.degrees(math.asin(min(1.0, abs(float(e[0] * u[1] - e[1] * u[0])) / float(np.linalg.norm(u))))))
    return worst


def _c_for(b: float, c0: float) -> float:
    """The cylinder pin's distance c from the pivot so that the rod length at cos(theta0 - phi) = c0 is CYL_CLOSED."""
    # c^2 - 2 b c0 c + b^2 - CYL_CLOSED^2 = 0, the positive root
    disc = (b * c0) ** 2 - (b * b - CYL_CLOSED ** 2)
    return b * c0 + math.sqrt(max(disc, 0.0))


def _piston_boom(bd: Build, name: str, mech: str, pivot: tuple[float, float, float], length: float, theta0: float,
                 theta1: float, boom_geoms: str) -> str:
    """A boom on a hinge (axis y) at ``pivot`` (frame coordinates, inches), driven by a piston through the closed
    loop cylinder-rod-boom.  The boom's own frame has +x along the boom, so ``boom_geoms`` are drawn at theta = 0."""
    b, c, phi = boom_pins(theta0, theta1)
    px, py, pz = pivot
    cx, cz = px + c * math.cos(phi), pz + c * math.sin(phi)  # the cylinder's pin, on a post from the frame
    bx, bz = px + b * math.cos(theta0), pz + b * math.sin(theta0)  # the rod's pin on the boom, retracted
    ang = math.atan2(bz - cz, bx - cx)
    rod_d = ROD / 2
    cyl = (f'<body name="{name}_cyl" pos="{_v(cx * IN, py * IN, cz * IN)}" euler="0 {_f(-ang)} 0">'
           f'<joint name="{name}_cyl" type="hinge" axis="0 1 0" damping="0.001"/>'
           + _cap(f"{name}_cylinder", (0, 0, 0), (CYL_CLOSED - 0.8, 0, 0), BORE / 2 + 0.1, METAL, "robot", 0.08)
           + f'<body name="{name}_rod" pos="0 0 0"><joint name="{name}_rod" type="slide" axis="1 0 0" range="0 {_f(STROKE * IN)}" '
           f'limited="true" armature="0.05" damping="{_f(0.01 * VALVE_C)}"/>'
           + _cap(f"{name}_rodbar", (1.0, 0, 0), (CYL_CLOSED, 0, 0), rod_d, "0.8 0.8 0.82 1", "robot", 0.02)
           + "</body></body>")
    # a hinge about +y turns the boom's +x from theta0 to theta0 - q, so its stroke is q in [0, theta0 - theta1]
    boom = (f'<body name="{name}" pos="{_v(px * IN, py * IN, pz * IN)}" euler="0 {_f(-theta0)} 0">'
            f'<joint name="{name}" type="hinge" axis="0 1 0" range="-0.05 {_f(theta0 - theta1 + 0.05)}" limited="true" '
            f'damping="0.01"/>' + boom_geoms + "</body>")
    bd.eqs.append(f'<connect name="{name}_loop" body1="{name}_rod" body2="{name}" anchor="{_v(CYL_CLOSED * IN, 0, 0)}"/>')
    area_ext, area_ret = math.pi * (BORE * IN / 2) ** 2, math.pi * ((BORE * IN / 2) ** 2 - (ROD * IN / 2) ** 2)
    bd.acts.append(f'<motor name="{mech}_p" joint="{name}_rod" gear="1" ctrlrange="-400 400" ctrllimited="true"/>')
    bd.pistons[mech] = {"a_ext": area_ext, "a_ret": area_ret, "b": b, "c": c, "phi": phi, "hinge": name, "theta0": theta0, "theta1": theta1,
                        "length": length, "rod": f"{name}_rod", "travel": STROKE * IN}  # type: ignore[dict-item]
    bd.moves[mech] = [(name, 0.0, theta0 - theta1)]
    return cyl + boom


def build_robot(rb: Robot) -> Build:
    """The robot's body tree, its equalities and actuators.  The robot's frame origin is its centre on the floor."""
    bd = Build()
    L, W, H = rb.L, rb.W, rb.H
    body: list[str] = []
    marks: list[tuple[int, str]] = [(0, "frame")]
    plate = 0.25
    body.append(_box("frame_base", (L / 2, W / 2, plate / 2), (0, 0, BELLY + plate / 2), PLY, "robot", 0.4 * rb.frame_kg))
    if rb.treasure[0] == "peel_chute":  # the deck has a slot over its front half where the inner trough passes
        sw = rb.treasure[1]["width"] / 2 + 0.3
        body.append(_box("frame_deck", (L / 4, W / 2, plate / 2), (-L / 4, 0, H - plate / 2), PLY, "robot",
                         0.125 * rb.frame_kg))
        for sy in (-1, 1):
            body.append(_box(f"frame_deck_{'lr'[sy > 0]}", (L / 4, (W / 2 - sw) / 2, plate / 2), (L / 4, sy * (sw + W / 2) / 2, H - plate / 2), PLY,
                             "robot", 0.0625 * rb.frame_kg))
    else:
        body.append(_box("frame_deck", (L / 2, W / 2, plate / 2), (0, 0, H - plate / 2), PLY, "robot", 0.25 * rb.frame_kg))
    for sx in (-1, 1):
        for sy in (-1, 1):
            body.append(_box("", (0.375, 0.375, (H - BELLY) / 2), (sx * (L / 2 - 0.375), sy * (W / 2 - 0.375),
                                                                   (H + BELLY) / 2), PLY, "robot", 0.05 * rb.frame_kg))
    body.append(_box("electronics", (min(2.5, L / 2 - 1), min(2.0, W / 2 - 1), 0.75), (-L / 4, 0, BELLY + plate + 0.75),
                     "0.2 0.5 0.3 1", "robot", 0.15 * rb.frame_kg))
    marks.append((len(body), "drive"))
    # ---- drive
    if rb.drive in ("wheels2", "mousetrap_car"):
        r = rb.wheel_r
        for j, sy in enumerate((1, -1)):
            nm = f"wheel{j}"
            body.append(f'<body name="{nm}" pos="{_v(-L / 4 * IN, sy * (W / 2 + 0.6) * IN, r * IN)}">'
                        f'<joint name="{nm}" type="hinge" axis="0 1 0" damping="0.0005" armature="0.0005"/>'
                        f'<geom type="cylinder" size="{_f(r * IN)} {_f(0.375 * IN)}" zaxis="0 1 0" mass="0.06" rgba="{RUBBER}" '
                        f'class="robot" friction="0.9 0.005 0.0001" condim="6"/></body>')
            bd.wheels.append(nm)
            if rb.drive == "wheels2":
                bd.acts.append(f'<general name="m_{nm}" joint="{nm}" gaintype="fixed" gainprm="{MOTOR_TS}" biastype="affine" '
                               f'biasprm="0 0 {-MOTOR_TS / MOTOR_W0:.6f}" ctrlrange="-1 1" ctrllimited="true" '
                               f'forcerange="{-MOTOR_TS} {MOTOR_TS}" forcelimited="true"/>')
            else:
                bd.acts.append(f'<motor name="m_{nm}" joint="{nm}" gear="1" ctrlrange="-5 5" ctrllimited="true"/>')
        # a ball caster up front: a sphere that slides (the ball's own rolling, as low friction)
        body.append(f'<geom name="caster" type="sphere" size="{_f(0.5 * IN)}" pos="{_v((L / 2 - 0.75) * IN, 0, 0.5 * IN)}" '
                    f'mass="0.02" rgba="{METAL}" class="robot" friction="0.02 0.001 0.0001"/>')
        body.append(_box("", (0.4, 0.4, (BELLY - 0.9) / 2 + 0.05), (L / 2 - 0.75, 0, 0.95 + (BELLY - 0.9) / 2), METAL,
                         "robot", 0.005))
        if rb.drive == "mousetrap_car":
            bd.traps["drive"] = ["wheel0", "wheel1"]
            bd.latches["drive"] = ["drive_latch0", "drive_latch1"]
            bd.eqs += [f'<joint name="drive_latch{j}" joint1="wheel{j}" polycoef="0 0 0 0 0"/>' for j in range(2)]
    else:  # a static base: rubber feet
        for sx in (-1, 1):
            for sy in (-1, 1):
                body.append(_box("", (0.5, 0.5, BELLY / 2), (sx * (L / 2 - 0.5), sy * (W / 2 - 0.5), BELLY / 2), RUBBER,
                                 "robot", 0.01, None, 'friction="0.8 0.005 0.0001"'))
    marks.append((len(body), "mermaid"))
    front = L / 2
    # ---- mermaid
    kind, p = rb.mermaid
    if kind == "push_bar":
        w = p["bar_w"]
        body.append(_box("push_bar", (0.375, w / 2, 0.75), (front + 0.375, 0, p["bar_z"]), PRINT, "robot",
                         0.0124 * 1.5 * w))  # PLA, about 12 g per inch of a 1.5 x 0.75 in bar at 30 % infill
        body.append(_box("", (0.375, 0.75, max(0.1, (p["bar_z"] - BELLY) / 2)),
                         (front + 0.375, 0, BELLY + max(0.1, (p["bar_z"] - BELLY) / 2)), PRINT, "robot", 0.01))
    elif kind == "trap_arms":
        a, z = p["arm_len"], p["arm_z"]
        for j, sy in enumerate((1, -1)):
            nm = f"arm{j}"
            # folded along the frame's side pointing back; the trap swings it out and forward through 180 degrees
            body.append(f'<body name="{nm}" pos="{_v(front * IN, sy * (W / 2) * IN, z * IN)}" euler="0 0 {_f(math.pi)}">'
                        f'<joint name="{nm}" type="hinge" axis="0 0 {-sy}" range="0 {_f(math.pi)}" limited="true" damping="0.002"/>'
                        + _cap("", (0, 0, 0), (a, 0, 0), 0.25, PRINT, "robot", 0.004 * a + 0.01) + "</body>")
            # the trap's base, screwed to the frame under the arm's hinge; only its bail (the arm) swings
            body.append(_box("", (0.6, 0.6, 0.4), (front + 0.3, sy * (W / 2), z - 0.9), "0.7 0.55 0.3 1", "robot", 0.03))
            bd.eqs.append(f'<joint name="arm_latch{j}" joint1="{nm}" polycoef="0 0 0 0 0"/>')
            bd.acts.append(f'<motor name="m_{nm}" joint="{nm}" gear="1" ctrlrange="0 1" ctrllimited="true"/>')
        bd.traps["mermaid"] = ["arm0", "arm1"]
        bd.moves["mermaid"] = [("arm0", 0.0, math.pi), ("arm1", 0.0, math.pi)]
        bd.latches["mermaid"] = ["arm_latch0", "arm_latch1"]
    marks.append((len(body), "cannon"))
    # ---- cannon: a chute on the deck that tilts down to its exit, a gate at the exit on a piston
    kind, p = rb.cannon
    if kind == "gate_chute":
        ez, reach = p["exit_z"], p["reach"]
        x1, x0 = front + reach, front + reach - chute_len(ez, reach, rb)
        z1 = ez - BALL_R - 0.05
        z0 = z1 + (x1 - x0) * math.tan(math.radians(CHUTE_DEG))
        ang = math.atan2(z1 - z0, x1 - x0)
        mid, half = ((x0 + x1) / 2, 2.5, (z0 + z1) / 2), math.hypot(x1 - x0, z1 - z0) / 2
        body.append(_box("chute", (half, 0.9, 0.06), mid, PRINT, "robot", 0.05, (0, -ang, 0)))
        for sy in (1, -1):
            body.append(_box("", (half, 0.06, 0.9), (mid[0], 2.5 + sy * 0.95, mid[2] + 0.8), PRINT, "robot", 0.01, (0, -ang, 0)))
        if x0 <= front:  # a post from the deck (or the base) up to the chute's back end
            zb = H if z0 > H else BELLY
            body.append(_box("", (0.3, 0.9, abs(z0 - zb) / 2), (x0, 2.5, (z0 + zb) / 2), PRINT, "robot", 0.02))
        else:  # a cantilever from the deck's front edge out to the chute's back end
            ln, an = math.hypot(x0 - front, z0 - H), math.atan2(z0 - H, x0 - front)
            body.append(_box("", (ln / 2 + 0.3, 0.6, 0.25), ((front + x0) / 2, 2.5, (H + z0) / 2), PRINT, "robot", 0.03,
                             (0, -an, 0)))
        body.append(_box("", (0.08, 0.9, 0.7), (x0 + 0.1, 2.5, z0 + 0.7), PRINT, "robot", 0.005))  # the back wall
        # the gate: a plate across the exit on a vertical slide, lifted by the piston (modelled as its rod)
        body.append(f'<body name="gate" pos="{_v(x1 * IN, 2.5 * IN, z1 * IN)}">'
                    f'<joint name="gate" type="slide" axis="0 0 1" range="0 {_f(2.0 * IN)}" limited="true" armature="0.05" '
                    f'damping="{_f(VALVE_C)}"/>'
                    + _box("gate_plate", (0.08, 0.9, 1.0), (0.12, 0, 1.0), METAL, "robot", 0.02) + "</body>")
        bd.acts.append('<motor name="gate_p" joint="gate" gear="1" ctrlrange="-200 200" ctrllimited="true"/>')
        bd.moves["cannon"] = [("gate", 0.0, 2.0 * IN)]
        bd.pistons["cannon"] = {"a_ext": math.pi * (BORE * IN / 2) ** 2,
                                "a_ret": math.pi * ((BORE * IN / 2) ** 2 - (ROD * IN / 2) ** 2), "rod": "gate",  # type: ignore[dict-item]
                                "travel": 2.0 * IN}
        for k in range(min(4, int((x1 - x0 - 0.4) / (2 * BALL_R)))):
            u = (k + 0.6) * 2 * BALL_R
            bx, bz = x1 - 0.3 - u * math.cos(ang), z1 + 0.1 + BALL_R + u * -math.sin(ang)
            bd.balls.append(f"ball{k}")
            bd.xml += (f'<body name="ball{k}" pos="{{X}}{_f(bx * IN)} {{Y}}{_f(2.5 * IN)} {_f(bz * IN)}"><freejoint/>'
                       f'<geom type="sphere" size="{_f(BALL_R * IN)}" mass="{M_BALL}" rgba="0.05 0.05 0.05 1" class="item" '
                       f'friction="0.3 0.005 0.0001"/></body>')
    marks.append((len(body), "flag"))
    # ---- flag
    kind, p = rb.flag
    if kind in ("piston_boom", "sensed_boom"):
        Lb, zp, raise_ = p["boom_len"], p["pivot_z"], math.radians(p["raise_deg"])
        th0 = math.radians(p["fold_deg"])  # folded back over the deck
        # the flag's mount at the tip, turned so that the flag stands upright, facing +x, when the boom is raised
        geoms = (_cap("", (0, 0, 0), (Lb, 0, 0), 0.3, WOOD, "robot", 0.012 * Lb)
                 + f'<body name="flagmount" pos="{_v((Lb + 0.4) * IN, 0, 0)}" euler="0 {_f(raise_)} 0"></body>')
        body.append(_piston_boom(bd, "flagboom", "flag", (front - 0.5, -3.0, zp), Lb, th0, raise_, geoms))
        bd.flag = True
        if kind == "sensed_boom":  # the sensor on its post beside the boom's pivot, looking along +x
            body.append(_box("flag_sensor_post", (0.375, 0.375, (FLAG_SENSOR_Z - H) / 2), (front - 0.5, -4.5,
                             (FLAG_SENSOR_Z + H) / 2), PLY, "robot", 0.03))
            body.append(_box("flag_sensor", (0.4, 0.8, 0.4), (front - 0.3, -4.5, FLAG_SENSOR_Z), "0.15 0.15 0.15 1",
                             "robot", 0.01))
            bd.flag_sensed = True
    elif kind == "band_mast":
        n, sl, lean = int(round(p["stages"])), p["stage_len"], math.radians(p["lean_deg"])
        bands = int(round(p["bands"]))
        base_z = H
        inner = (f'<body name="mast0" pos="{_v((front - 1.0) * IN, -3.0 * IN, base_z * IN)}" euler="0 {_f(lean)} 0">'
                 + _box("", (0.6, 0.6, sl / 2), (0, 0, sl / 2), WOOD, "robot", 0.004 * sl * 6))
        close = "</body>"
        k_each = bands * BAND_K / max(n - 1, 1)
        for s in range(1, n):
            travel = sl - 1.5
            # a band pulls each stage out: its stiffness, with its rest length beyond full travel (always pulling)
            inner += (f'<body name="mast{s}" pos="0 0 {_f(0.75 * IN)}">'
                      f'<joint name="mast{s}" type="slide" axis="0 0 1" range="0 {_f(travel * IN)}" limited="true" '
                      f'stiffness="{_f(k_each)}" springref="{_f((travel + 0.5 * BAND_L0) * IN)}" damping="8" frictionloss="0.8"/>'
                      + _box("", (0.6 - 0.12 * s, 0.6 - 0.12 * s, sl / 2), (0, 0, sl / 2), WOOD, "robot",
                             0.004 * sl * (6 - s)))
            close += "</body>"
            bd.eqs.append(f'<joint name="mast_latch{s}" joint1="mast{s}" polycoef="0 0 0 0 0"/>')
            bd.moves.setdefault("flag", []).append((f"mast{s}", 0.0, travel * IN))
        inner += f'<body name="flagmount" pos="{_f(0.7 * IN)} 0 {_f((sl - 1.6) * IN)}" euler="0 {_f(-lean)} 0"></body>'
        body.append(inner + close)
        bd.latches["flag"] = [f"mast_latch{s}" for s in range(1, n)]
        bd.flag = True
    marks.append((len(body), "heart"))
    # ---- heart: a hook boom, pushed down into the chest by its piston and pulled back up by it
    kind, p = rb.heart
    if kind == "hook_boom":
        Lb, depth, drop = p["boom_len"], p["hook_depth"], math.radians(p["drop_deg"])
        th0 = math.radians(HOOK_FOLD_DEG)
        geoms = (_cap("", (0, 0, 0), (Lb, 0, 0), 0.3, WOOD, "robot", 0.012 * Lb)
                 # the hook hangs from the tip: a bar straight down (at the extended angle) and a toe pointing back
                 + f'<body name="hook" pos="{_v(Lb * IN, 0, 0)}" euler="0 {_f(math.pi / 2 - drop)} 0">'
                 + _cap("hook_bar", (0, 0, 0), (depth, 0, 0), 0.15, METAL, "robot", 0.01)
                 + _cap("hook_toe", (depth, 0, 0), (depth, 0, -1.2), 0.15, METAL, "robot", 0.004) + "</body>")
        body.append(_piston_boom(bd, "hookboom", "heart", (front - 0.5, 0.0, H), Lb, th0, -drop, geoms))
        body.append(_box("tray", (1.8, 1.8, 0.1), (front - 2.5, 0, H + 0.1), PRINT, "robot", 0.02))
    marks.append((len(body), "treasure"))
    # ---- treasure: the peel blade on a fold-out trough, a size sieve, and two side troughs to the mounds
    kind, p = rb.treasure
    if kind == "peel_chute":
        body.append(_treasure(bd, rb, p))
    marks.append((len(body), ""))
    tree = "".join(_tag(bd, "".join(body[i:j]), mech) for (i, mech), (j, _) in zip(marks, marks[1:], strict=False))
    bd.xml = (tree, bd.xml)  # type: ignore[assignment]
    return bd


def _tag(bd: Build, xml: str, mech: str) -> str:
    """Name every geom of one mechanism's XML that has no name, and note which mechanism each geom belongs to."""
    def one(mt: re.Match[str]) -> str:
        nm = re.match(r'<geom name="([^"]+)"', mt.group(0))
        if nm:
            bd.part_of[nm.group(1)] = mech
            return mt.group(0)
        name = f"{mech}_part{sum(1 for v in bd.part_of.values() if v == mech)}"
        bd.part_of[name] = mech
        return f'<geom name="{name}" ' + mt.group(0)[len("<geom "):]
    return re.sub(r"<geom [^>]*>", one, xml)


CHUTE_LEN, CHUTE_DEG = 7.0, 8.0  # in, deg: the cannon chute holds four balls on a gentle slope to its exit


def chute_len(exit_z: float, reach: float, rb: Robot) -> float:
    """The cannon chute's length: 7 in, unless its back end would pass through the deck; then it starts at the
    frame's front (a chute too short to hold four balls is refused before trial)."""
    rear_z = exit_z - BALL_R - 0.05 + CHUTE_LEN * math.tan(math.radians(CHUTE_DEG))
    return CHUTE_LEN if reach >= CHUTE_LEN or rear_z - 0.1 > rb.H else max(reach, 0.5)


def _trough(name: str, length: float, width: float, rgba: str = PRINT, mass: float = 0.05, cls: str = "robot") -> str:
    """A U-trough along +x from its own origin: floor and two walls (inches)."""
    return (_box(f"{name}_floor" if name else "", (length / 2, width / 2, 0.06), (length / 2, 0, 0), rgba, cls, 0.6 * mass)
            + _box("", (length / 2, 0.06, 0.6), (length / 2, width / 2 + 0.06, 0.55), rgba, cls, 0.2 * mass)
            + _box("", (length / 2, 0.06, 0.6), (length / 2, -width / 2 - 0.06, 0.55), rgba, cls, 0.2 * mass))


def peel_layout(rb: Robot, p: dict[str, float]) -> dict[str, float]:
    """Where the peel chute's pieces sit, in the frame's coordinates (inches), deployed.  Coins leave the inner
    trough's low end onto a cross trough that runs out to the left (+y) side trough; jewels drop through the sieve
    slot onto a funnel that runs out to the right (-y) one; both fall at the side troughs' slope."""
    s = math.radians(p["slope_deg"])
    jz, reach, tw = p["junction_z"], p["reach"], p["width"]
    front = rb.L / 2
    z_front = jz + front * math.tan(s)  # the fixed inner trough rises from the junction (frame centre) to the front
    z_tip = z_front + reach * math.tan(s)
    side = math.radians(p["side_deg"])
    run = rb.W / 2 + tw / 2 + 0.2 + SIDE_OUT  # from the trough's far wall out to the side troughs' hinges
    z_coin0 = jz - 0.4  # the cross trough's floor at the trough's wall
    z_jewel0 = jz + SLOT0 * math.sin(s) - JEWEL_T - 0.4  # the funnel's floor at the trough's wall, a jewel's height below
    z_l, z_r = z_coin0 - run * math.tan(side), z_jewel0 - run * math.tan(side)  # the side troughs' hinges
    drop = p["side_len"] * math.sin(side)
    return {"slope": s, "z_front": z_front, "z_tip": z_tip, "x_tip": front + reach, "blade_bottom": z_tip + 1.0,
            "side_angle": side, "run": run, "z_coin0": z_coin0, "z_jewel0": z_jewel0, "z_left": z_l, "z_right": z_r,
            "side_exit_z": min(z_l, z_r) - drop, "coin_exit_z": z_l - drop, "jewel_exit_z": z_r - drop,
            "side_exit_y": rb.W / 2 + SIDE_OUT + p["side_len"] * math.cos(side)}


SIDE_OUT = 1.2  # in: the side troughs hinge this far outboard of the frame, so that folded up their walls (1.15 in
# tall, facing the frame) clear the deck's edge
SIDE_WALL = 1.15  # in: a trough's floor and wall height
SIDE_FOLD_DEG = 85.0  # the side troughs stand folded 5 degrees past vertical, outward
SLOT0 = 1.0  # in: the sieve's start, along the inner trough from its low end
OUTER_FOLD_DEG = 75.0  # the outer trough stands folded 15 degrees forward of vertical, so it always falls forward
SIEVE_LEN, CENTRE_TILT = 3.0, 10.0  # in, deg: the sieve rails' length; the V floor's tilt toward the middle


def _treasure(bd: Build, rb: Robot, p: dict[str, float]) -> str:
    lay = peel_layout(rb, p)
    s, jz, reach, gap, side_len = lay["slope"], p["junction_z"], p["reach"], p["sieve_gap"], p["side_len"]
    front, tw = rb.L / 2, p["width"]  # the troughs' inner width
    side, run, W2 = lay["side_angle"], lay["run"], rb.W / 2
    out = []
    # the inner trough: fixed, from the junction at the frame centre up to the front, with the sieve slot near its
    # low end; a coin bridges the slot, a jewel drops through it
    run_in = front / math.cos(s)
    # the sieve: SIEVE_LEN of the floor is two flat rails along the trough with a sieve_gap between them; above it
    # the floor is a shallow V (each half tilted CENTRE_TILT toward the middle) that centres every piece, so a coin
    # arrives centred and bridges the rails flat while a jewel drops between them
    rw = (tw - gap) / 2
    tilt = math.radians(CENTRE_TILT)
    dz = -tw / 2 * math.sin(tilt)  # the V's outer edges meet the outer trough's floor; its middle is dz lower
    rails = "".join(_box("", (SIEVE_LEN / 2, rw / 2, 0.06), (SLOT0 + SIEVE_LEN / 2, sy * (gap / 2 + rw / 2), dz), PRINT,
                         "slick", 0.01) for sy in (-1, 1))
    up0, up = SLOT0 + SIEVE_LEN, run_in - SLOT0 - SIEVE_LEN
    vee = "".join(_box("trough_in_floor" if sy > 0 else "", (up / 2, tw / 4, 0.06), (up0 + up / 2, sy * tw / 4,
                       dz + tw / 4 * math.sin(tilt)), PRINT, "slick", 0.02, (sy * tilt, 0, 0)) for sy in (-1, 1))
    out.append(f'<body name="trough_in" pos="{_v(0, 0, jz * IN)}" euler="0 {_f(-s)} 0">' + vee
               + rails + _box("", (SLOT0 / 2, tw / 2, 0.06), (SLOT0 / 2, 0, dz), PRINT, "slick", 0.01)
               + _box("", (run_in / 2, 0.06, 0.6), (run_in / 2, tw / 2 + 0.06, 0.55), PRINT, "slick", 0.01)
               + _box("", (run_in / 2, 0.06, 0.6), (run_in / 2, -tw / 2 - 0.06, 0.55), PRINT, "slick", 0.01) + "</body>")
    # the coin cross trough under the low end: from the right wall out to the left side, falling at the side slope,
    # with a back wall and an end wall at its high end
    xc0, xc1 = -3.2, 0.3
    yc0 = -tw / 2 - 0.2
    ln = run / math.cos(side)
    zc = lay["z_coin0"] - run * math.tan(side) / 2
    out.append(_box("coin_cross", ((xc1 - xc0) / 2, ln / 2, 0.06), ((xc0 + xc1) / 2, yc0 + run / 2, zc), PRINT, "slick",
                    0.02, (-side, 0, 0)))
    out.append(_box("", (0.06, ln / 2, 0.7), (xc0 - 0.06, yc0 + run / 2, zc + 0.6), PRINT, "slick", 0.01, (-side, 0, 0)))
    out.append(_box("", ((xc1 - xc0) / 2, 0.06, 0.7), ((xc0 + xc1) / 2, yc0 - 0.06, lay["z_coin0"] + 0.6), PRINT,
                    "slick", 0.005))
    # the jewel funnel under the slot: from the left wall out to the right side, falling at the side slope, with
    # walls front and back
    xs0, xs1 = SLOT0 * math.cos(s) - 0.4, (SLOT0 + SIEVE_LEN) * math.cos(s) + 0.4
    zj = lay["z_jewel0"] - run * math.tan(side) / 2
    out.append(_box("jewel_funnel", ((xs1 - xs0) / 2, ln / 2, 0.06), ((xs0 + xs1) / 2, -(yc0 + run / 2), zj), PRINT,
                    "slick", 0.02, (side, 0, 0)))
    for xw in (xs0 - 0.06, xs1 + 0.06):
        out.append(_box("", (0.06, ln / 2, 0.6), (xw, -(yc0 + run / 2), zj + 0.5), PRINT, "slick", 0.005, (side, 0, 0)))
    # the outer trough, folded up and latched; it falls forward onto its stop at the trough's slope
    fold = math.radians(OUTER_FOLD_DEG)
    blade = (f'<body name="blade" pos="{_v(reach / math.cos(s) * IN, 0, 0)}" euler="0 {_f(s)} 0">'
             # a bracket up over the rim, then the blade inward over the top at 40 degrees in plan: the top turns
             # clockwise, carrying the pieces at its near rim toward +y, so the blade's inner end leads at -y and a
             # piece slides along it outward, toward the robot, and drops into the trough
             + _box("", (0.1, 0.5, 0.55), (-0.8, 2.0, 0.5), METAL, "slick", 0.01)
             # 5.5 in long, crossing the rim 2 in to the +y side of the trough's centreline: a piece rides the
             # blade's upstream (-y) face, its centre about a radius off it, so it crosses the rim over the trough
             + _box("blade_plate", (2.75, 0.06, 0.55), (-0.8 + 2.75 * math.cos(math.radians(40)),
                    2.0 - 2.75 * math.sin(math.radians(40)), 1.0 + 0.55), METAL, "slick", 0.025,
                    (0, 0, math.radians(-40)))
             + "</body>")
    out.append(f'<body name="trough_out" pos="{_v(front * IN, 0, lay["z_front"] * IN)}" euler="0 {_f(-fold)} 0">'
               f'<joint name="trough_out" type="hinge" axis="0 1 0" range="{_f(-0.02)} {_f(fold - s + 0.02)}" limited="true" '
               f'damping="0.02"/>' + _trough("trough_out", reach / math.cos(s), tw, ACRYLIC, mass=0.012 * reach, cls="slick") + blade + "</body>")
    bd.eqs.append('<joint name="chute_latch0" joint1="trough_out" polycoef="0 0 0 0 0"/>')
    latches = ["chute_latch0"]
    bd.moves["treasure"] = [("trough_out", 0.0, fold - s)]
    # the side troughs, hinged along x at the deck's sides at the junction's height, folded up (5 degrees past
    # vertical, outward) and latched; released, each falls outward onto its stop at SIDE_DEG below level
    fold_s = math.radians(SIDE_FOLD_DEG)
    for j, sy in enumerate((1, -1)):
        nm = f"side{j}"
        zf, xm = (lay["z_left"], (xc0 + xc1) / 2) if j == 0 else (lay["z_right"], (xs0 + xs1) / 2)
        out.append(f'<body name="{nm}" pos="{_v(xm * IN, sy * (W2 + SIDE_OUT) * IN, zf * IN)}" euler="{_f(sy * fold_s)} 0 0">'
                   f'<joint name="{nm}" type="hinge" axis="{sy} 0 0" range="{_f(-(fold_s + lay["side_angle"]) - 0.02)} 0.02" '
                   f'limited="true" damping="0.02"/>'
                   f'<body euler="0 0 {_f(sy * math.pi / 2)}">' + _trough(f"{nm}t", side_len, max(tw, 3.4), ACRYLIC, mass=0.01 * side_len, cls="slick")
                   + "</body></body>")
        bd.eqs.append(f'<joint name="chute_latch{j + 1}" joint1="{nm}" polycoef="0 0 0 0 0"/>')
        latches.append(f"chute_latch{j + 1}")
        bd.moves["treasure"].append((nm, 0.0, -(fold_s + side)))
    bd.latches["treasure"] = latches
    return "".join(out)


# ---------------------------------------------------------------- the scene
def arena_xml(v: Variant, robot: str = "", items: str = "", eqs: str = "", acts: str = "") -> str:
    """The match's MJCF: the arena on this track, with whatever robot and items are given."""
    return f"""<mujoco model="me2110">
<compiler angle="radian" autolimits="true"/>
<option timestep="{DT}" cone="elliptic" integrator="implicitfast" noslip_iterations="0"/>
<size memory="64M"/>
<visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
<default>
  <geom solref="{SOLREF}" solimp="{SOLIMP}" friction="0.5 0.005 0.0001"/>
  <default class="arena"><geom contype="1" conaffinity="3"/></default>
  <default class="item"><geom contype="1" conaffinity="3"/></default>
  <default class="robot"><geom contype="2" conaffinity="1"/></default>
  <default class="slick"><geom contype="2" conaffinity="1" friction="{SLICK_MU} 0.005 0.0001"/></default>
  <default class="flag"><geom contype="4" conaffinity="1"/></default>
  <equality solref="0.004 1"/>
  <joint solreflimit="0.004 1" solimplimit="0.99 0.999 0.0005"/>
</default>
<worldbody>
  <light pos="1 0 3" dir="0 0 -1" directional="true"/>
  <geom name="floor" type="plane" size="3 3 0.1" rgba="0.86 0.84 0.8 1" class="arena" friction="{v.mu} 0.005 0.0001"/>
  {_fortress(v)}{_figures()}{_merfolk()}{_chest()}{_mounds()}{_walls()}
  {robot}
  {items}
</worldbody>
<contact><exclude body1="world" body2="fortress"/></contact>
<equality><joint name="prop" joint1="lid" polycoef="{_f(LID_OPEN)} 0 0 0 0"/>{eqs}</equality>
<actuator><velocity name="spin" joint="spin" kv="400" ctrlrange="-2 2" ctrllimited="true"/>{acts}</actuator>
</mujoco>"""


def scene(rb: Robot, v: Variant, y0: float) -> tuple[str, Build]:
    """The match's MJCF: arena plus robot, the robot's centre at (-L/2 - 0.25, y0) in the dock."""
    bd = build_robot(rb)
    tree, extra = bd.xml  # type: ignore[misc]
    x0 = -(rb.L / 2 + 0.25)
    extra = extra.replace("{X}", "").replace("{Y}", "")
    # items that start on the robot are placed in world coordinates: shift them by the robot's start
    items = []
    for chunk in extra.split('<body name="ball')[1:]:
        head, rest = chunk.split('pos="', 1)
        xyz, tail = rest.split('"', 1)
        x, y, z = (float(t) for t in xyz.split())
        items.append(f'<body name="ball{head}pos="{_v(x + x0 * IN, y + y0 * IN, z)}"{tail}')
    xml = arena_xml(v, f'<body name="robot" pos="{_v(x0 * IN, y0 * IN, 0)}"><freejoint name="root"/>{tree}</body>',
                    "".join(items), "".join(bd.eqs), "".join(bd.acts))
    if bd.flag:
        xml = _add_flag(xml)
    return xml, bd


FLAG_W, FLAG_H, FLAG_T = 5.0, 3.0, 0.08  # in


def _add_flag(xml: str) -> str:
    """The flag: a free plate at its mount's pose, held to the mount by the clip (a weld) and, once it meets a pole
    face squarely, to the pole by its magnet (a weld made at that moment)."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b = m.body("flagmount").id
    pos, quat = d.xpos[b], d.xquat[b]
    flag = (f'<body name="flag" pos="{_v(*pos)}" quat="{_v(*quat)}"><freejoint/>'
            + _box("flag_plate", (FLAG_T / 2, FLAG_W / 2, FLAG_H / 2), (FLAG_T / 2, 0, 0), "0.1 0.1 0.1 1", "flag", 0.010)
            + _box("", (0.01, FLAG_W / 2 - 0.2, FLAG_H / 2 - 0.2), (FLAG_T + 0.005, 0, 0), "0.9 0.9 0.9 1", "item", 0.0005,
                   None, 'contype="0" conaffinity="0"') + "</body>")
    xml = xml.replace("</worldbody>", flag + "</worldbody>", 1)
    return xml.replace("</equality>", '<weld name="clip" body1="flag" body2="flagmount"/>'
                       '<weld name="magnet" body1="flag" body2="fortress" active="false"/></equality>', 1)


# ---------------------------------------------------------------- the law that parts stay clear of each other
MOUNT_TOL = 0.1  # in: two mechanisms' parts may overlap this much where one is fastened to the other
SWEEP_N = 12  # poses along each mechanism's motion, stowed to deployed


def _pose(m: mujoco.MjModel, d: mujoco.MjData, bd: Build, mech: str, u: float) -> None:
    """Put one mechanism a fraction u of the way from stowed to deployed, closing its piston's loop."""
    for jn, q0, q1 in bd.moves.get(mech, []):
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q0 + u * (q1 - q0)
    pist = bd.pistons.get(mech)
    if pist and "hinge" in pist:  # a boom: the cylinder turns and its rod slides to keep the rod's pin on the boom
        b, c, phi, th0 = pist["b"], pist["c"], pist["phi"], pist["theta0"]

        def rod(th: float) -> tuple[float, float]:
            dx, dz = b * math.cos(th) - c * math.cos(phi), b * math.sin(th) - c * math.sin(phi)
            return math.atan2(dz, dx), math.hypot(dx, dz)
        q = d.qpos[m.jnt_qposadr[m.joint(pist["hinge"]).id]]
        a0, _ = rod(th0)
        a, ln = rod(th0 - q)
        d.qpos[m.jnt_qposadr[m.joint(f"{pist['hinge']}_cyl").id]] = a0 - a
        d.qpos[m.jnt_qposadr[m.joint(pist["rod"]).id]] = (ln - CYL_CLOSED) * IN


def interference(rb: Robot, n: int = SWEEP_N, tol: float = MOUNT_TOL) -> list[tuple[float, str, str, str]]:
    """Every pair of parts that pass into each other by more than ``tol`` (inches): each mechanism swept from stowed
    to deployed with the others all stowed, then all deployed (the flag rides its mount).  Two mechanisms' parts may
    not overlap anywhere.  A mechanism is fastened to the frame, so against the frame what counts is how much further
    in it goes as it moves than where it is mounted.  Returns (overlap in, one part, the other, the pose) per pair of
    mechanisms, worst first."""
    xml, bd = scene(rb, Variant(), 0.0)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mech_of = {m.geom(nm).id: mech for nm, mech in bd.part_of.items()}
    if bd.flag:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == m.body("flag").id:
                mech_of[g] = "flag"
    geoms = sorted(mech_of)

    def slotted(g1: int, g2: int) -> bool:  # a piston's cylinder or rod passes through a slot cut in a plate
        n1, n2 = sorted((m.geom(g1).name, m.geom(g2).name), key=lambda x: not x.startswith("frame_"))
        return n1.startswith(("frame_base", "frame_deck")) and n2.endswith(("_cylinder", "_rodbar"))
    pairs = [(g1, g2) for i, g1 in enumerate(geoms) for g2 in geoms[i + 1:]
             if mech_of[g1] != mech_of[g2] and not slotted(g1, g2)]
    flag_adr = m.jnt_qposadr[m.body("flag").jntadr[0]] if bd.flag else None
    mount = m.body("flagmount").id if bd.flag else None
    worst: dict[tuple[str, ...], tuple[float, str, str, str]] = {}
    mechs = sorted(bd.moves)
    for others in (0.0, 1.0):
        for mv in mechs or [""]:
            over: dict[tuple[int, int], list[float]] = {}
            for k in range(n + 1 if mechs else 1):
                mujoco.mj_resetData(m, d)
                for o in mechs:
                    _pose(m, d, bd, o, k / n if o == mv else others)
                mujoco.mj_kinematics(m, d)
                if flag_adr is not None:
                    d.qpos[flag_adr:flag_adr + 3] = d.xpos[mount]
                    d.qpos[flag_adr + 3:flag_adr + 7] = d.xquat[mount]
                    mujoco.mj_kinematics(m, d)
                for g1, g2 in pairs:
                    r = m.geom_rbound[g1] + m.geom_rbound[g2]
                    if np.sum((d.geom_xpos[g1] - d.geom_xpos[g2]) ** 2) > r * r:
                        dist = 0.0
                    else:
                        dist = min(0.0, mujoco.mj_geomDistance(m, d, g1, g2, 0.01, None) / IN)
                    over.setdefault((g1, g2), []).append(-dist)
            for (g1, g2), ov in over.items():
                frame = "frame" in (mech_of[g1], mech_of[g2])
                k = int(np.argmax(ov))
                bad = ov[k] - (min(ov) if frame else 0.0)
                if bad > tol:
                    key = tuple(sorted((mech_of[g1], mech_of[g2])))
                    if key not in worst or bad > worst[key][0]:
                        pose = (f"the {mv} {round(100 * k / n)} % deployed, the rest "
                                f"{'deployed' if others else 'stowed'}") if mechs else "at rest"
                        worst[key] = (bad, f"the {mech_of[g1]}'s {m.geom(g1).name}", f"the {mech_of[g2]}'s "
                                      f"{m.geom(g2).name}", pose)
    return sorted(worst.values(), reverse=True)


# ---------------------------------------------------------------- the match
@dataclass
class Result:
    points: dict[str, float]
    notes: dict[str, str]  # task -> what happened, in numbers
    laws: list[str]  # rules the run broke (each disqualifies it)
    t_end: float
    events: list[tuple[float, str]]
    trace: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, float] = field(default_factory=dict)  # task -> 0..1, how close it came (guides search only)


def trap_torque(theta: float) -> float:
    """A Victor mousetrap's spring: torque (N m) falling linearly over its 180 degree swing."""
    u = min(max(theta / math.pi, 0.0), 1.0)
    return TRAP_T0 + (TRAP_T1 - TRAP_T0) * u


class Match:
    """One match under a controller.  ``ctl`` holds System 1's controls (inches, seconds, duty)."""

    def __init__(self, rb: Robot, v: Variant, ctl: dict[str, float], seed: int = 0, horizon: float = 40.0,
                 tasks: frozenset[str] | None = None, trace: bool = False) -> None:
        xml, bd = scene(rb, v, ctl.get("y0", 0.0))
        self._setup(rb, v, ctl, seed, horizon, tasks, trace, mujoco.MjModel.from_xml_string(xml), xml, bd)

    def _setup(self, rb: Robot, v: Variant, ctl: dict[str, float], seed: int, horizon: float, tasks: frozenset[str] | None,
               trace: bool, m: mujoco.MjModel, xml: str, bd: Build) -> None:
        self.rb, self.v, self.ctl, self.horizon, self.want_trace = rb, v, ctl, horizon, trace
        self.rng = np.random.default_rng(seed)
        self.tasks = tasks
        self.bd, self.m, self.xml = bd, m, xml
        self.d = mujoco.MjData(self.m)
        self.jid = {m.joint(i).name: i for i in range(m.njnt)}
        self.bid = {m.body(i).name: i for i in range(m.nbody)}
        self.eid = {m.eq(i).name: i for i in range(m.neq)}
        self.aid = {m.actuator(i).name: i for i in range(m.nu)}
        self.robot_geoms = np.array([i for i in range(m.ngeom) if m.geom_contype[i] == 2])
        self.events: list[tuple[float, str]] = []
        self.p_tank = P_TANK0
        self.valve: dict[str, bool] = {k: False for k in self.bd.pistons}
        self.released: set[str] = set()
        self.arrived_t: float | None = None
        self.phase = "out" if rb.drive == "wheels2" else "idle"
        self.phase_t = 0.0
        self.left_dock = False
        self.prop = True
        self.magnet = False
        self.clip = self.bd.flag
        self.clip_over = self.magnet_over = 0
        self.flag_best = 0.0  # how close the flag ever came to sitting square on a pole face, high enough (shaping)
        self.gate_t: float | None = None
        self.face_t: float | None = None  # when the sensed boom's sensor saw a pole face square
        self.ball_flight: dict[int, tuple[float, float, float, float, bool] | None] = {}
        self.ball_from_robot: dict[int, bool] = {}
        self.laws: list[str] = []
        self.out_of_quadrant = False
        x0 = (CHEST_X + CHEST_D / 2) * IN
        self.chest0 = np.array([x0, 0.0, 0.0])
        self.encoder_err = self.rng.normal(0.0, 0.25) * IN  # the drive's stopping error, from the encoder's count
        if self.bd.flag:
            self._make_flag()

    def _make_flag(self) -> None:
        self.flag_b, self.pole_g = self.bid["flag"], self.m.geom("pole").id

    def log(self, what: str) -> None:
        self.events.append((round(self.d.time, 2), what))

    # -- helpers
    def body_pos(self, name: str) -> np.ndarray:
        return self.d.xpos[self.bid[name]]

    def frame_x(self) -> float:
        return float(self.d.qpos[self.m.jnt_qposadr[self.jid["root"]]]) / IN

    def robot_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Each robot geom's world-frame bounding box: (centres, half extents), metres."""
        g = self.robot_geoms
        R = self.d.geom_xmat[g].reshape(-1, 3, 3)
        c = self.d.geom_xpos[g] + np.einsum("nij,nj->ni", R, self.m.geom_aabb[g, :3])
        h = np.einsum("nij,nj->ni", np.abs(R), self.m.geom_aabb[g, 3:])
        return c, h

    def robot_extent(self) -> tuple[float, float]:
        """(min x, how far the robot reaches past its quadrant's diagonals), inches."""
        c, h = self.robot_box()
        minx = float(np.min(c[:, 0] - h[:, 0])) / IN
        outside = np.hypot(c[:, 0] - XF * IN, c[:, 1]) > (RF + 1.0) * IN  # over the fortress is common ground
        over = np.abs(c[:, 1]) + h[:, 1] - (XF * IN - (c[:, 0] - h[:, 0]))
        return minx, float(np.max(over[outside])) / IN if outside.any() else -1.0

    def release(self, mech: str) -> None:
        if mech in self.released or mech not in self.bd.latches:
            return
        self.released.add(mech)
        for e in self.bd.latches[mech]:
            self.d.eq_active[self.eid[e]] = 0
        self.log(f"released {mech}")

    def stroke(self, mech: str, extend: bool) -> None:
        if self.valve.get(mech) == extend:
            return
        g = self.bd.pistons[mech]
        vol = (g["a_ext"] if extend else g["a_ret"]) * STROKE * IN
        atm = 1.013e5  # the tank's air expands into the cylinder at constant temperature (absolute pressures)
        self.p_tank = (self.p_tank + atm) * TANK_L * 1e-3 / (TANK_L * 1e-3 + vol) - atm
        self.valve[mech] = extend
        self.log(f"{mech} piston {'extends' if extend else 'retracts'} ({self.p_tank / 1e5:.2f} bar left)")

    # -- the controller, every step
    def control(self) -> None:
        d, m, c, t = self.d, self.m, self.ctl, self.d.time
        d.ctrl[self.aid["spin"]] = -self.v.rpm * 2 * math.pi / 60
        rb = self.rb
        # drive
        if rb.drive == "wheels2":
            x = self.frame_x() * IN + self.encoder_err
            duty = c.get("duty", 0.6)
            ramp = min(1.0, (t - self.phase_t) / RAMP_S)  # the controller ramps the duty: a step would pop a wheelie
            if self.phase == "out":
                u = duty * ramp
                if x >= c.get("park_x", 10.0) * IN:
                    self.phase, self.phase_t, u = "park", t, 0.0
                    self.log(f"drive stops at x = {self.frame_x():.1f} in")
            elif self.phase == "park":
                u = 0.0
                if self.arrived_t is None and abs(d.qvel[m.jnt_dofadr[self.jid["root"]]]) < 0.01:
                    self.arrived_t = t
                    self.log("parked")
                if self.arrived_t is not None and t >= self.arrived_t + c.get("return_at", 99.0):
                    self.phase, self.phase_t = "back", t
                    self.log("drives back")
            elif self.phase == "back":
                u = -duty * ramp
                if self.frame_x() <= -(rb.L / 2 + 1.0):
                    self.phase, u = "home", 0.0
                    self.log(f"back in the dock at x = {self.frame_x():.1f} in")
            else:
                u = 0.0
            for w in self.bd.wheels:
                d.ctrl[self.aid[f"m_{w}"]] = u
        elif rb.drive == "mousetrap_car":
            if "drive" in self.released:
                pulled = abs(d.qpos[m.jnt_qposadr[self.jid["wheel0"]]]) * rb.axle_r  # string wound off the axle (in)
                if pulled < 2 * rb.lever:
                    tension = trap_torque(math.pi * pulled / (2 * rb.lever)) / (rb.lever * IN)
                    for w in self.bd.wheels:
                        d.ctrl[self.aid[f"m_{w}"]] = tension * rb.axle_r * IN / 2
                else:
                    for w in self.bd.wheels:
                        d.ctrl[self.aid[f"m_{w}"]] = 0.0
                    if self.arrived_t is None and abs(d.qvel[m.jnt_dofadr[self.jid["root"]]]) < 0.01:
                        self.arrived_t = t
                        self.log(f"the car coasts to rest at x = {self.frame_x():.1f} in")
        elif self.arrived_t is None:
            self.arrived_t = 0.0
        # releases
        for mech in self.bd.latches:
            if rb.release == "cam_timer":
                if t >= c.get(f"t_{mech}", 1.0):
                    self.release(mech)
            elif rb.release == "drive_trip" and self.arrived_t is not None and t >= self.arrived_t + 0.2 \
                    and mech != "drive":
                self.release(mech)
        # mousetraps
        for mech, joints in self.bd.traps.items():
            if mech == "drive" or mech not in self.released:
                continue
            for j in joints:
                q = d.qpos[m.jnt_qposadr[self.jid[j]]]
                d.ctrl[self.aid[f"m_{j}"]] = trap_torque(q) if q < math.pi - 0.02 else 0.0
        # pistons
        t0 = self.arrived_t
        for g in self.bd.pistons.values():
            if "hinge" in g:  # a boom's needle valve, reflected onto its hinge: c (dl/dq)^2, the same power c v^2,
                # so the integrator takes it implicitly (on the light rod, through the loop, it rings at 2 ms steps)
                dof = m.jnt_dofadr[self.jid[g["hinge"]]]
                th = g["theta0"] - d.qpos[m.jnt_qposadr[self.jid[g["hinge"]]]]
                b, cc = g["b"] * IN, g["c"] * IN
                ln = math.sqrt(max(b * b + cc * cc - 2 * b * cc * math.cos(th - g["phi"]), 1e-12))
                m.dof_damping[dof] = 0.01 + VALVE_C * (b * cc * math.sin(th - g["phi"]) / ln) ** 2
        for mech in self.bd.pistons:
            if mech == "flag" and self.bd.flag_sensed:  # fired a delay after the sensor sees a face square
                at, hold = c.get("flag_delay", 0.2), c.get("flag_hold", 1.0)
                ext = self.face_t is not None and self.face_t + at <= t < self.face_t + at + hold
            elif mech in ("flag", "heart"):  # timed from the robot's arrival (its encoder says it has stopped)
                at, hold = c.get(f"{mech}_at", 99.0), c.get(f"{mech}_hold", 1.0)
                ext = t0 is not None and t0 + at <= t < t0 + at + hold
            else:  # the cannon's gate: open for gate_open s, gate_delay s after the IR sees our port's tab
                ext = self.gate_t is not None and self.gate_t + c.get("gate_delay", 0.0) <= t < \
                    self.gate_t + c.get("gate_delay", 0.0) + c.get("gate_open", 0.5)
            self.stroke(mech, ext)
            g = self.bd.pistons[mech]
            f = self.p_tank * (g["a_ext"] if ext else -g["a_ret"])
            q = d.qpos[m.jnt_qposadr[self.jid[g["rod"]]]]
            top = g["travel"]
            if (ext and q > top - 0.001) or (not ext and q < 0.001):
                f = math.copysign(END_HOLD, f)  # at an end of its stroke the cylinder's end cap takes the rest
            d.ctrl[self.aid["gate_p" if mech == "cannon" else f"{mech}_p"]] = f
        if "cannon" in self.bd.pistons and self.gate_t is None and self.arrived_t is not None:
            self._ir()
        if self.bd.flag_sensed and self.face_t is None and self.arrived_t is not None and t > self.arrived_t + 0.3:
            self._face()

    def _face(self) -> None:
        """The sensed boom's range sensor looks along the robot's +x over the fortress wall: when its beam lands on
        the pole within range, the reading bottoms out as a face turns square to it."""
        d, m = self.d, self.m
        s = d.geom_xpos[m.geom("flag_sensor").id]
        fx = d.xmat[self.bid["robot"]].reshape(3, 3)[:, 0]
        rel = d.geom_xpos[self.pole_g] - s
        along = float(rel[:2] @ fx[:2])
        across = abs(float(rel[0] * fx[1] - rel[1] * fx[0]))
        if not (0.0 < along < FLAG_IR_RANGE * IN and across < POLE_HALF * IN) or s[2] < HF * IN:
            return
        px = d.geom_xmat[self.pole_g].reshape(3, 3)[:, 0]
        ang = math.degrees(math.atan2(px[1], px[0]) - math.atan2(fx[1], fx[0])) % 90.0
        if min(ang, 90.0 - ang) < FACE_SQUARE_DEG:
            self.face_t = d.time
            self.log("the flag's sensor sees a pole face square")

    def _ir(self) -> None:
        """The IR sensor at the gate looks along +x: it sees our port's yellow tab when the tab faces it."""
        tab = self.d.geom_xpos[self.m.geom("tab0").id]
        gate = self.body_pos("gate")
        if 0.0 < tab[0] - gate[0] < IR_RANGE * IN and abs(tab[1] - gate[1]) < 0.25 * IN and self.d.time > self.arrived_t + 0.3:
            self.gate_t = self.d.time
            self.log("the IR sensor sees our port's tab")

    # -- physical events, every few steps
    def physics_events(self) -> None:
        d = self.d
        # the chest's prop
        if self.prop:
            q = d.qpos[self.m.jnt_qposadr[self.bid_joint("chest")]: self.m.jnt_qposadr[self.bid_joint("chest")] + 7]
            moved = np.linalg.norm(q[:2] - self.chest0[:2]) / IN
            tilt = 2 * math.acos(min(1.0, abs(q[3])))
            if moved > PROP_MOVE or tilt > PROP_TILT:
                self.prop = False
                d.eq_active[self.eid["prop"]] = 0
                self.log(f"the chest's prop gives way (moved {moved:.2f} in, tilted {math.degrees(tilt):.1f} deg); the lid shuts")
        minx, over = self.robot_extent()
        if minx > 0.0 and not self.left_dock:
            self.left_dock = True
            self.log("the robot has fully left the dock")
        if over > 0.5 and not self.out_of_quadrant:
            self.out_of_quadrant = True
            self.log(f"the robot crosses its quadrant's edge by {over:.1f} in")
        self._balls()
        if self.bd.flag:
            self._flag_events()

    def bid_joint(self, body: str) -> int:
        return int(self.m.body_jntadr[self.bid[body]])

    def _balls(self) -> None:
        """The projectile rule: a ball that leaves the robot flies unsupported (touching nothing) at most 12 in
        horizontally, and never rises above where it left.  Bounces after it lands are not launches."""
        d, m = self.d, self.m
        if self.gate_t is None:
            return
        robot = set(int(b) for b in m.geom_bodyid[self.robot_geoms])
        support: dict[int, set[int]] = {}
        for i in range(d.ncon):
            con = d.contact[i]
            a, b = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            support.setdefault(a, set()).add(b)
            support.setdefault(b, set()).add(a)
        for k, nm in enumerate(self.bd.balls):
            b = self.bid[nm]
            pos = d.xpos[b]
            fl = self.ball_flight.get(k)
            on = support.get(b, set())
            if on:
                if fl is not None and fl[4]:
                    dist = math.hypot(pos[0] - fl[0], pos[1] - fl[1]) / IN
                    rise = (fl[3] - fl[2]) / IN
                    if dist > 12.0:
                        self.laws.append(f"ball {k} flew {dist:.1f} in unsupported from the robot (12 in allowed)")
                    if rise > 0.25:
                        self.laws.append(f"ball {k} rose {rise:.2f} in after leaving the robot (launches must be level "
                                         "or down)")
                self.ball_flight[k] = None
                self.ball_from_robot[k] = bool(on & robot)
            elif fl is None:
                self.ball_flight[k] = (float(pos[0]), float(pos[1]), float(pos[2]), float(pos[2]),
                                       self.ball_from_robot.get(k, True))
            else:
                self.ball_flight[k] = (*fl[:3], max(fl[3], float(pos[2])), fl[4])

    def eq_force(self, name: str) -> float:
        d, e = self.d, self.eid[name]
        rows = (d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[: d.nefc] == e)
        f = d.efc_force[: d.nefc][rows]
        return float(np.linalg.norm(f[:3])) if f.size else 0.0

    def hold_load(self, name: str, press: float) -> float:
        """The load (N) a weld holding the flag carries as a clip or magnet: shear plus pull, without the push that
        presses the flag flat onto what holds it (a face takes that by contact).  ``press`` is +1 when that push
        is along the flag's face normal (its mount behind it), -1 when against it (the pole in front)."""
        d, e = self.d, self.eid[name]
        rows = np.where((d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[: d.nefc] == e))[0]
        if rows.size < 3:
            return 0.0
        f = d.efc_force[rows[:3]]  # the weld's force on the flag, world frame
        n = d.xmat[self.flag_b].reshape(3, 3)[:, 0]
        fn = float(f @ n)
        shear = f - fn * n
        pull = max(0.0, -press * fn)
        return float(math.sqrt(float(shear @ shear) + pull * pull))

    def _flag_events(self) -> None:
        """The clip lets go above CLIP_N; the magnet catches a flag pressed flat on a pole face, lets go above MAGNET_N."""
        d = self.d
        self.clip_over = self.clip_over + 1 if self.clip and self.hold_load("clip", 1.0) > CLIP_N else 0
        if self.clip and self.clip_over >= HOLD_TICKS:
            self.clip = False
            d.eq_active[self.eid["clip"]] = 0
            self.log("the flag's clip lets go of the robot")
        if self.magnet:
            self.magnet_over = self.magnet_over + 1 if self.hold_load("magnet", -1.0) > MAGNET_N else 0
            if self.magnet_over >= HOLD_TICKS:
                self.magnet = False
                d.eq_active[self.eid["magnet"]] = 0
                self.log("the flag is pulled off the pole")
            return
        R = d.geom_xmat[self.pole_g].reshape(3, 3)
        local = R.T @ (d.xpos[self.flag_b] - d.geom_xpos[self.pole_g])
        nrm = d.xmat[self.flag_b].reshape(3, 3)[:, 0]  # the flag's face normal, world
        nl = R.T @ nrm
        off = max(0.0, max(abs(local[0]), abs(local[1])) / IN - POLE_HALF - FLAG_T)
        low = max(0.0, 24.0 - FLAG_H / 2 - d.xpos[self.flag_b][2] / IN)
        skew = 1.0 - max(abs(nl[0]), abs(nl[1]))
        self.flag_best = max(self.flag_best, max(0.0, 1.0 - (off + low + 10.0 * skew) / 30.0))
        for ax in (0, 1):
            for sgn in (-1, 1):
                gap = sgn * local[ax] / IN - POLE_HALF - FLAG_T
                lat = abs(local[1 - ax]) / IN
                if -0.1 < gap < 0.12 and lat < POLE_HALF + FLAG_W / 2 - 0.5 and -sgn * nl[ax] > math.cos(math.radians(25)):
                    self.magnet = True
                    self._weld_now("magnet", "fortress")
                    self.log(f"the flag's magnet catches the pole, its top at {d.xpos[self.flag_b][2] / IN + FLAG_H / 2:.1f} in")
                    return

    def _weld_now(self, eq: str, other: str) -> None:
        """Make the weld ``eq`` (body1 = flag, body2 = ``other``) hold the two bodies as they are now."""
        d, m, e = self.d, self.m, self.eid[eq]
        b1, b2 = self.flag_b, self.bid[other]
        p1, q1, p2, q2 = d.xpos[b1], d.xquat[b1], d.xpos[b2], d.xquat[b2]
        qi = np.zeros(4)
        mujoco.mju_negQuat(qi, q1)
        rel = np.zeros(3)
        mujoco.mju_rotVecQuat(rel, p2 - p1, qi)  # body2's position in body1's frame
        rq = np.zeros(4)
        mujoco.mju_mulQuat(rq, qi, q2)
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = rel
        m.eq_data[e, 6:10] = rq
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1

    def run(self) -> Result:
        d, m = self.d, self.m
        trace: dict[str, list[Any]] = {"t": [], "qpos": []}
        n_ev = max(1, int(0.02 / DT))
        step = 0
        while d.time < self.horizon - 1e-9:
            self.control()
            mujoco.mj_step(m, d)
            step += 1
            if step % n_ev == 0:
                self.physics_events()
                if self.want_trace and step % (2 * n_ev) == 0:
                    trace["t"].append(d.time)
                    trace["qpos"].append(d.qpos.copy())
            if not np.isfinite(d.qacc).all():
                self.laws.append("the simulation diverged")
                break
        return self.score(trace)

    def progress(self) -> dict[str, float]:
        """How close each task came, 0..1, from the end state: a slope for System 1's search, never a verdict."""
        d, m = self.d, self.m
        out: dict[str, float] = {"activate": min(1.0, max(0.0, (self.robot_extent()[0] + self.rb.L) / self.rb.L))}
        mp = 0.0
        for nm, (x, y, _) in MERFOLK.items():
            b = self.bid[nm]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[b].reshape(3, 3)[2, 2]))))
            disp = math.hypot(d.xpos[b][0] / IN - x, d.xpos[b][1] / IN - y)
            gap = min(float(np.min(np.hypot(self.robot_box()[0][:, 0] / IN - x, self.robot_box()[0][:, 1] / IN - y))), 30.0)
            mp += 0.5 * max(min(1.0, tilt / 45.0), min(0.4, disp / 5.0), 0.3 * (1 - gap / 30.0))
        out["mermaids"] = mp
        a = math.pi + d.qpos[m.jnt_qposadr[self.jid["spin"]]] - self.v.phase
        port = np.array([XF + (RF - 1.0) * math.cos(a), (RF - 1.0) * math.sin(a), (self.v.barrier + HF - TOP_T) / 2])
        balls = [d.xpos[self.bid[nm]] / IN for nm in self.bd.balls]
        out["cannon"] = float(np.mean([max(0.0, 1 - np.linalg.norm(p - port) / 30.0) for p in balls])) if balls else 0.0
        if self.bd.flag:
            out["flag"] = 1.0 if self.magnet else 0.8 * self.flag_best
        else:
            out["flag"] = 0.0
        hb = self.bid["heart"]
        h0 = np.array([CHEST_X + CHEST_D / 2, 0.0, CHEST_H - 0.5 - HEART_R])
        moved = float(np.linalg.norm(d.xpos[hb] / IN - h0))
        out["heart"] = min(1.0, 0.5 * min(1.0, moved / 4.0) + 0.5 * max(0.0, 1 - max(d.xpos[hb][0] / IN, 0.0) / 22.0))
        tp = 0.0
        for kind, n in (("coin", 10), ("jewel", 5)):
            mx, my = MOUNDS["gold" if kind == "coin" else "jewel"]
            for k in range(n):
                p = d.xpos[self.bid[f"{kind}{k}"]] / IN
                off = math.hypot(p[0] - XF, p[1]) > RF or p[2] < HF - 0.5
                tp += (0.3 if off else 0.0) + 0.7 * max(0.0, 1 - math.hypot(p[0] - mx, p[1] - my) / 30.0) * off
        out["treasure"] = tp / 15
        return out

    # -- scoring at the end of the round
    def score(self, trace: dict[str, list[Any]]) -> Result:
        d, m = self.d, self.m
        pts: dict[str, float] = {}
        notes: dict[str, str] = {}
        # contacts, as a graph of bodies
        touch: dict[int, set[int]] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            touch.setdefault(a, set()).add(b)
            touch.setdefault(b, set()).add(a)
            touch.setdefault(c.geom1 + 100000, set()).add(b)
            touch.setdefault(c.geom2 + 100000, set()).add(a)
        robot_root = self.bid["robot"]
        robot_bodies = {i for i in range(m.nbody) if m.body_rootid[i] == robot_root}
        moved = self.arrived_t is not None or bool(self.released) or self.phase != "idle"
        pts["activate"] = (1 if moved else 0) + (10 if self.left_dock else 0)
        notes["activate"] = ("left the dock fully" if self.left_dock else "never fully left the dock") + \
            f"; the frame ended at x = {self.frame_x():.1f} in"
        # treasure
        tp, tn = 0.0, []
        for kind, n, top, ring in (("coin", 10, 10, 4), ("jewel", 5, 15, 9)):
            mound = m.geom(f"mound_{'gold' if kind == 'coin' else 'jewel'}").id
            mx, my = MOUNDS["gold" if kind == "coin" else "jewel"]
            on_top = set(touch.get(mound + 100000, set()))
            grow = True
            while grow:  # a piece resting on a piece on the mound counts too
                grow = False
                for k in range(n):
                    b = self.bid[f"{kind}{k}"]
                    if b not in on_top and touch.get(b, set()) & on_top:
                        on_top.add(b)
                        grow = True
            got_top = got_ring = 0
            for k in range(n):
                b = self.bid[f"{kind}{k}"]
                x, y = d.xpos[b][0] / IN, d.xpos[b][1] / IN
                if b in on_top:
                    got_top += 1
                elif abs(x - mx) <= MOUND_X / 2 + BORDER + (COIN_R if kind == "coin" else JEWEL_R) and \
                        abs(y - my) <= MOUND_Y / 2 + BORDER + (COIN_R if kind == "coin" else JEWEL_R) and \
                        d.xpos[b][2] / IN < MOUND_H + 2.0:
                    got_ring += 1
            tp += got_top * top + got_ring * ring
            off = sum(1 for k in range(n) if d.xpos[self.bid[f"{kind}{k}"]][2] / IN > HF - 0.2 and
                      math.hypot(d.xpos[self.bid[f"{kind}{k}"]][0] / IN - XF, d.xpos[self.bid[f"{kind}{k}"]][1] / IN) < RF)
            tn.append(f"{got_top} {kind}s on the mound, {got_ring} in its border, {off} still on the fortress")
        pts["treasure"], notes["treasure"] = tp, "; ".join(tn)
        # flag
        if self.bd.flag:
            fz = (d.xpos[self.bid["flag"]][2]) / IN + 1.5
            ok = self.magnet and fz >= 24.0
            pts["flag"] = 40.0 if ok else 0.0
            notes["flag"] = (f"affixed by its magnet, top at {fz:.1f} in" if self.magnet else
                             f"not affixed (its top at {fz:.1f} in)") + ("" if fz >= 24.0 or not self.magnet else
                                                                       "; below the 24 in minimum")
        else:
            pts["flag"], notes["flag"] = 0.0, "no flag mechanism"
        # cannon
        inside = 0
        ang_f = d.qpos[m.jnt_qposadr[self.jid["spin"]]] - self.v.phase
        for nm in self.bd.balls:
            p = d.xpos[self.bid[nm]] / IN
            r = math.hypot(p[0] - XF, p[1])
            a = math.atan2(p[1], p[0] - XF)
            port = math.pi + ang_f
            da = abs((a - port + math.pi) % (2 * math.pi) - math.pi)
            if CORE_R - 0.5 < r < RF and da < PORT_W / 2 / RF + 0.1 and p[2] < HF - TOP_T:
                inside += 1
        gx = d.geom_xpos[self.robot_geoms] / IN
        in_fort = bool(np.any(np.hypot(gx[:, 0] - XF, gx[:, 1]) < RF - 0.2))
        pts["cannon"] = 0.0 if in_fort else 9.0 * inside
        notes["cannon"] = f"{inside} of {len(self.bd.balls)} balls in our port's receptacle" + (
            "; void: part of the robot is inside the fortress's cylinder" if in_fort and inside else "") if self.bd.balls \
            else "no cannon mechanism"
        # mermaids
        mp, mn = 0.0, []
        for nm, (x, y, r) in MERFOLK.items():
            b = self.bid[nm]
            zx = d.xmat[b].reshape(3, 3)[:, 2]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, zx[2]))))
            disp = math.hypot(d.xpos[b][0] / IN - x, d.xpos[b][1] / IN - y)
            if tilt >= 45.0:
                mp += 12
                mn.append(f"the {nm} toppled ({tilt:.0f} deg)")
            elif disp + r > MARK_R:
                mp += 5
                mn.append(f"the {nm} moved {disp:.2f} in, standing")
            else:
                mn.append(f"the {nm} untouched (moved {disp:.2f} in, tilted {tilt:.0f} deg)")
        pts["mermaids"], notes["mermaids"] = mp, "; ".join(mn)
        # heart
        hb, cb = self.bid["heart"], self.bid["chest"]
        hx = d.xpos[hb][0] / IN
        in_chest = abs(hx - (CHEST_X + CHEST_D / 2)) < CHEST_D / 2 and abs(d.xpos[hb][1] / IN) < CHEST_W / 2 and \
            d.xpos[hb][2] / IN < CHEST_H + 0.3 and abs(d.xpos[cb][0] / IN - hx) < CHEST_D / 2 + 0.5
        on_robot = bool(touch.get(hb, set()) & robot_bodies)
        behind = hx + HEART_L / 2 < 0.0 if not in_chest else False
        chest_behind = d.xpos[cb][0] / IN + CHEST_D / 2 < 0.0
        if behind:
            pts["heart"], notes["heart"] = 61.0, "the heart is back in the dock"
        elif not in_chest and on_robot:
            pts["heart"], notes["heart"] = 39.0, f"the heart is on the robot at x = {hx:.1f} in"
        elif in_chest and chest_behind:
            pts["heart"], notes["heart"] = 32.0, "the chest, heart inside, is back in the dock"
        elif in_chest and bool(touch.get(cb, set()) & robot_bodies) and d.xpos[cb][2] / IN > 0.3:
            pts["heart"], notes["heart"] = 11.0, "the chest is on the robot"
        else:
            pts["heart"] = 0.0
            notes["heart"] = ("the heart is still in the chest" if in_chest else f"the heart lies at "
                              f"({hx:.1f}, {d.xpos[hb][1] / IN:.1f}) in, not on the robot") + \
                ("" if self.prop else "; the lid shut")
        # end-of-round laws (a match always runs the full 40 s)
        laws = list(self.laws)
        if self.horizon >= 40.0 - 1e-6:
            v = np.abs(d.cvel[list(robot_bodies)]).max() if robot_bodies else 0.0
            if v > 0.02:
                laws.append(f"the robot is still moving at 40 s ({v:.3f})")
            ext = [k for k, on in self.valve.items() if on]
            if ext:
                laws.append(f"pistons not retracted at 40 s: {', '.join(ext)}")
        if self.out_of_quadrant:
            laws.append("the robot left its quadrant")
        prog = self.progress()
        return Result(points=pts, notes=notes, laws=laws, t_end=d.time, events=self.events, progress=prog,
                      trace={k: np.array(v) for k, v in trace.items()} | {"xml": self.xml} if self.want_trace else {})
