"""The mobile world: a small ground robot that explores a house and garden (docs/evolve.md, D-041).

Each task is one route from the robot's dock to an exploration target: a sequence of terrain features.  A
body is five parts: locomotion (four wheels, mecanum wheels, a rocker-bogie or tracks), a printed chassis
tub, drive gearmotors, a battery pack and an obstacle sensor.  System 1 drives the route with a cruise duty,
the line through an opening, the approach duty at an edge and the speed in the dark, under seeded noise
(terrain dimensions ±2 %, traction ±3 %, steering error), and must repeat a success on fresh seeds.

The simulator walks the route feature by feature with quasi-static vehicle mechanics (plain Python, tens of
microseconds per attempt).  The pack's cells are in series and its voltage drives the motors: stall force
and no-load speed scale with volts / the winding's rated volts, which must lie in 0.5-1.5 times it (D-062).
Speed follows a linear DC-motor curve, v = v0 (duty - F / F_stall); energy is
F d / eta plus electronics power times time, against 85 % of the pack.

- step (height, edge mu): a driven wheel climbs while h <= r (1 - cos a), a = atan(mu_e) + atan(mu (1-f) / f),
  mu_e the traction at the edge and f the share of weight on the climbing wheels (with nothing pushing from
  behind, f = 1, this is the single-wheel result r (1 - 1/sqrt(1+mu^2))), never above r.  A rigid tub must
  clear the edge as its rear wheels arrive.  Tracks climb an edge that meets the sloped front run, about 0.9
  of the front idler height.  At first contact a0 = acos((r - h) / r), the climbing wheels need traction
  mu_e N and the wheels behind a push f m g tan(a0 - atan mu_e): less duty stalls, while a wheel that hits
  the edge so fast that it hops (v sin a0)^2 / 2g above half its radius is thrown off.
- trench (width, depth): tracks bridge it while it is under 0.45 of their contact length (the centre of mass
  stays supported); a wheel spans less than its diameter, dips r - sqrt(r^2 - w^2/4) or drops to the bottom,
  and must climb out of that depth as a step.
- gap (width, offset): the overall width plus the steering error (sigma grows with speed) must fit.
- ramp (slope, surface): tan(theta) + C_rr <= mu of the driven contacts, and m g (sin + C_rr cos) < F_stall.
- sand (modulus k): ground pressure p = m g / A sinks the robot z = p / k (a Winkler soil); deeper than the
  step it can climb out of its own rut, or than its belly clearance, and it is stuck.
- low (clearance): the overall height must pass under an overhang.
- dark (length, obstacles): the stopping distance v t + v^2 / 2a must fit inside the sensing range; with less
  range than a body length the robot stops and backs off at every obstacle, at a bumper's 0.25 m/s.

A mechanism label names the limit of the body that stopped the robot.  When System 1's best attempt failed
for a reason of its own (a bad line, too fast in the dark), a noise-free probe drives the route at the best
controls: if the body hits a limit further on, that limit is the label; if not, it is ``not_found_by_search``.
"""

from __future__ import annotations

import functools
import math
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sea.evolve.core import Body, BomLine, Outcome, Param, Part, PartType, Task
from sea.evolve.search import local_search
from sea.sim import mobile_model as MM

G = 9.81
TUB_H = 4.0  # cm: depth of the chassis tub
ELEC_W, ELEC_G = 2.5, 60.0  # onboard computer, IMU and wiring: power (W) and mass (g)
BUMP_V, BACKOFF_S, REACT_S = 0.25, 4.0, 0.1  # bumper's rated impact speed, time to back off and turn, reaction
SURF = {  # rubber-on-surface friction, rolling-resistance coefficient of a 3 cm wheel
    "tile": (0.60, 0.012), "wood": (0.60, 0.014), "carpet": (0.70, 0.045), "concrete": (0.75, 0.015),
    "gravel": (0.55, 0.060), "grass": (0.55, 0.120), "sand": (0.50, 0.150)}
LOOSE = {"gravel", "grass", "sand"}
RATIOS = {"n20_gearmotor": (50, 100, 150, 298), "n20_encoder_gearmotor": (50, 100, 150, 298),
          "jga25_gearmotor": (21, 34, 46, 90)}
MOTOR = {  # motor-shaft stall torque (N m), no-load speed (rad/s), gearbox efficiency, g, stall A, steering error
    "n20_gearmotor": (2.0e-3, 3140.0, 0.60, 11.0, 0.75, 1.0),
    "n20_encoder_gearmotor": (2.0e-3, 3140.0, 0.60, 13.0, 0.75, 0.6),
    "jga25_gearmotor": (13.0e-3, 620.0, 0.75, 95.0, 2.2, 1.0)}
DRIVER = {"n20_gearmotor": ("driver_tb6612", 8.0), "n20_encoder_gearmotor": ("driver_tb6612", 8.0),
          "jga25_gearmotor": ("driver_l298n", 30.0)}  # a dual H-bridge board per two motors, and its mass (g)
CELL = {"nimh_pack": ("nimh_aa", 2.4, 27.0, 7.3, 4.0), "liion_pack": ("cell_18650", 9.4, 46.0, 12.0, 10.0)}
# cell item, Wh per cell, g per cell, footprint cm^2, the pack's maximum current (A)
VOLTS = {"nimh_pack": (1.2, 1.4), "liion_pack": (3.7, 4.2)}  # a cell's nominal and full-charge volts; cells in series
WINDING = {"n20_gearmotor": 6.0, "n20_encoder_gearmotor": 6.0, "jga25_gearmotor": 12.0}  # V: MOTOR's numbers hold here
ROTOR = {"n20_gearmotor": 5e-9, "n20_encoder_gearmotor": 5e-9, "jga25_gearmotor": 3.7e-7}  # kg m^2; unpublished, about
V_RANGE = (0.5, 1.5)  # the supply a winding takes, as fractions of its rated voltage (full charge at the top)
SENSOR = {  # range in the dark (m), height above the tub (cm), g, W, item, count
    "bump_switch": (0.0, 0.0, 5.0, 0.0, "bump_switch", 2), "ultrasonic": (2.0, 0.0, 9.0, 0.08, "hc_sr04", 1),
    "camera": (0.12, 2.5, 15.0, 0.7, "camera_module", 1), "lidar": (6.0, 4.0, 45.0, 1.5, "lidar_ld06", 1)}
ITEMS: dict[str, tuple[float, float | None, float, str]] = {  # spares beyond the stock body, $, lead days
    "tire_oring": (8, 0.4, 2, "rubber tyre for a printed hub"), "bearing_608": (4, 0.6, 2, "skate bearing"),
    "mecanum_wheel": (0, 6.5, 5, "one mecanum wheel"),
    "rubber_track_kit": (0, 14.0, 4, "two rubber tracks with sprockets and idlers"),
    "driver_tb6612": (2, 3.0, 2, "dual 1.2 A H-bridge board"), "driver_l298n": (0, 3.5, 2, "dual 2 A H-bridge board"),
    "nimh_aa": (4, 2.5, 1, "AA NiMH cell, 2.4 Wh"), "cell_18650": (0, 4.5, 3, "18650 Li-ion cell, 9.4 Wh"),
    "bms_board": (0, 1.5, 3, "Li-ion protection board"), "bump_switch": (2, 0.5, 1, "microswitch"),
    "hc_sr04": (0, 2.5, 2, "ultrasonic range sensor"), "camera_module": (0, 11.0, 3, "camera with IR LED"),
    "lidar_ld06": (0, None, 0, "2D lidar: none in the lab, and none can be ordered in time")}
CONTROL_LEVEL = {"stalled_at_step", "bounced_off_step", "clipped_gap_edge", "collided_in_dark", "stalled_at_low_duty",
                 "too_slow_at_chosen_speed", "drained_at_chosen_speed"}  # a failure of the controls, not of the body


def _f(kind: str, surf: str = "wood", **kw: float) -> dict[str, Any]:
    return {"k": kind, "s": surf, **kw}


ROUTES: dict[str, tuple[str, float, list[dict[str, Any]]]] = {  # target: (where, seconds allowed, features)
    "kitchen": ("the kitchen", 40, [_f("floor", len=2.0), _f("step", "tile", h=0.8), _f("floor", "tile", len=3.0)]),
    "living_room": ("the living room", 40, [_f("floor", len=2.0), _f("step", "carpet", h=1.5),
                                            _f("floor", "carpet", len=3.0)]),
    "under_sofa": ("under the sofa", 40, [_f("floor", len=1.5), _f("step", "carpet", h=1.0),
                                          _f("low", "carpet", clear=9.0, len=0.8)]),
    "bedroom": ("the bedroom, through the door left ajar", 60, [_f("floor", len=3.0), _f("gap", w=20.0, off=2.0),
                                                               _f("floor", "carpet", len=2.0)]),
    "hallway": ("the far end of the hallway at night", 45, [_f("floor", len=2.0), _f("dark", len=8.0, n=10),
                                                            _f("floor", len=1.0)]),
    "porch": ("the front porch", 40, [_f("floor", "concrete", len=2.0), _f("step", "concrete", h=5.5),
                                      _f("floor", "concrete", len=1.5)]),
    "garden": ("the vegetable garden, past the gate and the drain", 60, [
        _f("floor", "concrete", len=2.0), _f("gap", "concrete", w=28.0, off=-3.0),
        _f("trench", "concrete", w=9.0, depth=5.0), _f("floor", "grass", len=3.0)]),
    "sandbox": ("the far side of the sandbox", 60, [_f("floor", "grass", len=3.0), _f("step", "grass", h=1.0),
                                                     _f("sand", "sand", len=1.5, kmod=0.75),
                                                     _f("floor", "grass", len=0.5)]),
    "shed": ("inside the garden shed", 60, [_f("floor", "grass", len=3.0), _f("ramp", "gravel", deg=22.0, len=1.5),
                                            _f("step", "wood", h=1.2)]),
    # D-069: several edges in a row, round things across the floor, and a strip where carpet meets tile
    "patio": ("the patio, up the three garden steps", 60, [_f("floor", "concrete", len=2.0),
                                                          _f("stairs", "concrete", h=7.5, run=18.0, n=3),
                                                          _f("floor", "concrete", len=1.0)]),
    "study": ("the study, over the power cord and onto the rug", 45, [
        _f("floor", len=2.0), _f("bump", d=1.2), _f("floor", len=1.0), _f("threshold", "carpet", h=1.4, w=6.0),
        _f("floor", "carpet", len=2.0)]),
    "greenhouse": ("the greenhouse, over the garden hose", 60, [_f("floor", "grass", len=2.0), _f("bump", "grass", d=2.5),
                                                              _f("floor", "gravel", len=2.5)]),
    "lawn": ("every corner of the lawn (a survey)", 14400, [_f("floor", "concrete", len=1.0),
                                                           _f("step", "concrete", h=1.0),
                                                           _f("floor", "grass", len=5000.0)]),
}


# ---------------------------------------------------------------- what a body is, in numbers
@dataclass(frozen=True)
class Spec:
    loco: str
    m: float  # kg
    width: float  # cm, overall
    height: float  # cm, overall
    length: float  # cm, chassis
    belly: float  # cm, belly clearance
    r: float  # cm, wheel radius, or the tracks' sprocket radius
    n: int  # drive motors
    f_stall: float  # N, total push at the ground at stall and full duty
    v_nl: float  # m/s, no-load speed at full duty
    eta: float  # battery-to-ground efficiency
    drift: float  # cm, lateral steering error (sigma) at full speed
    sense: float  # m, sensing range in the dark
    wh: float  # Wh, pack capacity
    p_w: float  # W, electronics and sensor
    area: float  # cm^2, contact area on soft ground
    volts: float = 6.0  # V, the pack's nominal voltage across the motors
    tub_w: float = 18.0  # cm, the tub's inner width (the running gear adds ``(width - tub_w) / 2`` a side)
    loco_g: float = 0.0  # g, the running gear: wheels, or both tracks' rollers, idlers and belts
    rotor: float = 0.0  # kg m^2, a motor's rotor inertia seen at its wheel (J ratio^2)
    f: float = 0.5  # weight share on the climbing wheels
    edge: float = 1.0  # traction factor of a wheel on a step edge
    loose: float = 1.0  # traction factor on loose ground
    base: float = 18.0  # cm, wheelbase
    idler: float = 0.0  # cm, tracks: front idler height
    track: float = 0.0  # cm, tracks: contact length
    bogie: float = 0.0  # cm, rocker-bogie: bogie length

    @property
    def usable_j(self) -> float:
        return 0.85 * self.wh * 3600.0


def _ratio(part: Part) -> int:
    return min(RATIOS[part.type], key=lambda x: abs(x - part.params.get("gear_ratio", 100.0)))


def _cells(part: Part) -> int:
    return max(1, math.ceil(part.params.get("capacity", 0.0) / CELL[part.type][1] - 1e-6))


def _tub_g(w: float, length: float) -> float:
    return 0.18 * w * length + 0.5 * (w + length) * TUB_H


def _motors(body: Body) -> int:
    return {"tracks": 2, "rocker_bogie6": 6}.get(body.parts["locomotion"].type, 4)


def _loco_g(lo: Part) -> tuple[float, float]:
    """(total mass, printed PLA) in grams for the locomotion part."""
    p = lo.params
    if lo.type == "tracks":
        pla = 2 * (0.8 * p["track_length"] + 2.0 * p["idler_height"])
        return 180.0 + pla, pla
    if lo.type == "mecanum4":
        return 4 * (20.0 + 2.0 * p["wheel_radius"] ** 2), 0.0
    n = 6 if lo.type == "rocker_bogie6" else 4
    pla = n * (3.0 + 0.8 * p["wheel_radius"] ** 2) + (7.5 * p["bogie_length"] if n == 6 else 0.0)
    return pla + n * 3.0 + (6 * 12.0 if n == 6 else 0.0), pla  # hubs and arms, tyres, pivot bearings


def _spec(body: Body) -> Spec:
    ch = body.parts["chassis"].params
    lo, dr, bt, se = (body.parts[k] for k in ("locomotion", "drive", "battery", "sensor"))
    w, ln, ride = ch["width"], ch["length"], ch["ride_height"]
    tau, w0, geta, motor_g, _, steer = MOTOR[dr.type]
    sense, sh, sg, sp = SENSOR[se.type][:4]
    _, cell_wh, cell_g, _, _ = CELL[bt.type]
    n, p = _motors(body), lo.params
    grams = (ELEC_G + _tub_g(w, ln) + _cells(bt) * cell_g + 8.0 + sg + _loco_g(lo)[0] + n * motor_g  # 8 g: holder
             + math.ceil(n / 2) * DRIVER[dr.type][1])
    kw: dict[str, Any] = {}
    if lo.type == "tracks":
        r, side, drift, belly = 1.8, 3.0, 1.2, ride
        height = max(ride + TUB_H + sh, p["idler_height"] + 1.5)
        kw = {"area": 2 * 3.0 * p["track_length"] * 0.8, "loose": 1.35, "idler": p["idler_height"],
              "track": p["track_length"], "base": p["track_length"]}
    elif lo.type == "rocker_bogie6":
        r, side, drift, belly = p["wheel_radius"], 3.5, 1.0, ride + p["wheel_radius"]
        height = max(belly + TUB_H + sh, 2 * r + 2.0)
        kw = {"area": 6 * 3.0 * 0.5 * r, "f": 1 / 3, "bogie": p["bogie_length"], "base": 2 * p["bogie_length"]}
    else:
        r, belly = p["wheel_radius"], ride
        mec = lo.type == "mecanum4"
        side, drift = (3.0, 0.3) if mec else (2.5, 1.0)
        height = max(ride + TUB_H + sh, 2 * r)
        kw = {"area": 4 * side * 0.5 * r * (0.5 if mec else 1.0), "base": ln - 2.0}
        if mec:
            kw.update(loose=0.55)  # rollers clog in loose ground; at an edge they are in the model (D-070)
    ratio, volts = _ratio(dr), _cells(bt) * VOLTS[bt.type][0]
    k = volts / WINDING[dr.type]  # a DC motor's stall torque and no-load speed both scale with its supply
    return Spec(loco=lo.type, m=grams / 1000, width=w + 2 * side, height=height, length=ln, belly=belly, r=r, n=n,
                f_stall=k * n * tau * ratio * geta / (r / 100), v_nl=k * w0 / ratio * r / 100, eta=0.7 * geta,
                drift=drift * steer, sense=sense, wh=_cells(bt) * cell_wh, p_w=ELEC_W + sp, volts=volts, tub_w=w,
                loco_g=_loco_g(lo)[0], rotor=ROTOR[dr.type] * ratio ** 2, **kw)


# ---------------------------------------------------------------- terrain mechanics
def _mu(s: Spec, surf: str) -> float:
    mu = SURF[surf][0]
    return mu * s.loose if surf in LOOSE else mu  # a mecanum wheel's rollers are in the model (D-070)


def _crr(s: Spec, surf: str) -> float:
    c = SURF[surf][1]
    if s.loco == "tracks":
        return 0.6 * c + 0.05  # spread load, plus the track's own bending and skid-steer losses
    c *= math.sqrt(3.0 / s.r)  # a smaller wheel sinks into and rolls over more of the surface
    return c * 1.2 + 0.02 if s.loco == "mecanum4" else c + (0.008 if s.loco == "rocker_bogie6" else 0.0)


def _part_name(geom: str) -> str:
    """A model geom's name as a part: 'w10' -> 'wheel w10', 'w00_roller3' -> 'roller of wheel w00'."""
    if "_roller" in geom:
        return f"roller of wheel {geom.split('_')[0]}"
    if geom.startswith(("rocker_link", "bogie_link")):
        return geom.replace("_link", " link ")
    return f"wheel {geom}" if geom[:1] == "w" else geom


def _robot(s: Spec, surf: str) -> MM.Robot:
    """The body as the MuJoCo model takes it (SI units), on ``surf``."""
    per = {"tracks": 2, "rocker_bogie6": 6}.get(s.loco, 4)
    return MM.Robot(loco=s.loco, m=s.m, wheel_kg=s.loco_g / 1000 / per, width=s.tub_w / 100, length=s.length / 100,
                    tub_h=TUB_H / 100, belly=s.belly / 100, r=s.r / 100, torque=s.f_stall / s.n * s.r / 100,
                    omega0=s.v_nl / (s.r / 100), crr=_crr(s, surf), rotor=s.rotor, track=s.track / 100,
                    idler=s.idler / 100, bogie=s.bogie / 100)


def _drive(s: Spec, kind: str, f: dict[str, Any], mu: float, duty: float, y0: float = 0.0,
           standing: bool = False, trace: bool = False) -> MM.Episode:
    """One MuJoCo episode at an obstacle, its arguments quantised so that search shares physics (D-063).  With
    ``trace`` (a film), the same episode run afresh with its trajectory kept."""
    q = {"h": 0.1, "w": 0.1, "depth": 0.1, "deg": 0.25, "off": 0.25, "run": 0.5, "n": 1.0, "d": 0.1}  # cm, deg, count
    items = tuple(sorted((k, round(f[k] / q[k]) * q[k]) for k in q if k in f))
    args = (_robot(s, f["s"]), kind, items, round(mu / 0.03) * 0.03, round(duty / 0.05) * 0.05,
            round(y0 / 0.25) * 0.25 / 100)
    if trace:
        return MM.episode(*args[:2], dict(items), *args[3:], trace=True, standing=standing)
    return MM.cached(*args, standing)


@functools.cache
def _climb(s: Spec, surf: str, standing: bool = False) -> float:
    """The highest edge (cm) the body climbs at full duty on ``surf``, measured in MuJoCo by bisection: from a
    run-up, or ``standing`` against it (out of its own rut, where there is no run-up)."""
    mu = _mu(s, surf) * s.edge
    lo, hi = 0.0, max(2.0 * s.r, s.idler, 2.0)
    for _ in range(7):
        h = (lo + hi) / 2
        lo, hi = (h, hi) if _drive(s, "step", {"k": "step", "s": surf, "h": h}, mu, 1.0, standing=standing).ok \
            else (lo, h)
    return lo


def _speed(s: Spec, duty: float, force: float) -> float:
    return s.v_nl * (duty - force / s.f_stall)


def _vsafe(s: Spec, mu: float) -> tuple[float, bool]:
    """(safe speed among obstacles in the dark, whether the robot can steer around them)."""
    if s.sense < s.length / 100:
        return BUMP_V, False
    a, room = 0.5 * mu * G, max(0.0, s.sense - 0.1)
    return a * (-REACT_S + math.sqrt(REACT_S ** 2 + 2 * room / a)), True


DESC = {"floor": "{ln} m of {s}", "step": "a {h:.1f} cm step up ({s}, mu {mu:.2f})",
        "stairs": "{n:.0f} steps of {h:.1f} cm on {run:.0f} cm treads ({s}, mu {mu:.2f})",
        "bump": "a {d:.1f} cm {thing} across the way ({s})", "threshold": "a {h:.1f} cm threshold strip, {w:.1f} cm wide, "
        "onto {s}",
        "gap": "a {w:.1f} cm opening centred {off:+.1f} cm off the robot's line",
        "trench": "a {w:.1f} cm wide, {depth:.1f} cm deep trench ({s} edges, mu {mu:.2f})",
        "ramp": "a {deg:.1f} deg {s} ramp {ln} m long", "low": "{ln} m under a {clear:.1f} cm high overhang",
        "sand": "{ln} m of loose sand (it takes {kmod:.2f} kPa to sink 1 cm)",
        "dark": "{ln} m of dark corridor with {n:.0f} obstacles"}


def _desc(f: dict[str, Any]) -> str:
    ln = f.get("len", 0.0)
    return DESC[f["k"]].format(ln=f"{ln:.0f}" if ln >= 100 else f"{ln:.1f}", mu=SURF[f["s"]][0],
                               thing="hose" if f.get("d", 0) >= 2 else "cord", **f)


def _edge(s: Spec, f: dict[str, Any], mu: float, duty: float) -> tuple[str, str, float] | None:
    """Drive into a step or trench in MuJoCo at approach ``duty``: None, or (mechanism, numbers, partial credit)."""
    ep = _drive(s, f["k"], f, mu, duty)
    if ep.ok:
        return None
    what = {"step": "the {h:.2f} cm edge", "trench": "the {w:.1f} cm wide, {depth:.1f} cm deep trench",
            "stairs": "the {n:.0f} {h:.1f} cm steps on {run:.0f} cm treads", "bump": "the {d:.1f} cm " + ("hose" if f.get("d", 0) >= 2 else "cord"),
            "threshold": "the {h:.1f} cm threshold strip"}[f["k"]].format(**f) + \
        f" (edge mu {mu:.2f}), at approach duty {duty:.2f}: {ep.detail}"
    part = 0.5 * ep.progress
    if ep.why == "belly":
        return "belly_hits_step_edge", f"the tub's belly at {s.belly:.1f} cm grounded on {what}", part
    if ep.why == "bounced":
        return "bounced_off_step", f"thrown back off {what}", part
    if ep.why == "tipped":
        return "tipped_over", f"tipped over on {what}", part
    if ep.why in ("stuck_stalled", "slow"):
        if duty < 0.999:
            return "stalled_at_step", f"the drive stalled on {what}", part
        return "motor_torque_short_on_step", f"the drive stalls at {s.f_stall:.1f} N on {what}", part
    climb = _climb(s, f["s"])
    if f["k"] == "trench":
        return "gap_longer_than_support", f"its running gear slipped in {what}; at full duty it climbs edges up " \
            f"to {climb:.2f} cm", part
    return "step_higher_than_climbable", f"its running gear slipped on {what}; at full duty it climbs edges up to " \
        f"{climb:.2f} cm", part


class _Calm:
    """A noise source that always returns the mean: the route at its nominal dimensions."""

    def uniform(self, lo: float, hi: float) -> float:
        return (lo + hi) / 2

    def normal(self, mean: float, sd: float) -> float:
        return mean


PROBE = {"speed": 1.0, "line": 0.0, "approach": 0.5, "dark": 1.0}


def _attempt(s: Spec, route: list[dict[str, Any]], secs: float, ctl: dict[str, float],
             rng: np.random.Generator | _Calm, probe: bool = False) -> tuple[bool, float, str, str]:
    """Drive one route once: (success, score, mechanism, detail).  A ``probe`` ignores control-level failures
    and drives at the best speed everywhere, so only a limit of the body can stop it."""
    n, t, t_best, e, e_best, dark_note = len(route), 0.0, 0.0, 0.0, 0.0, ""
    for i, nominal in enumerate(route):
        f = {key: v * (1 + rng.uniform(-0.02, 0.02)) if isinstance(v, float) and key != "off" else v
             for key, v in nominal.items()}
        k, surf = f["k"], f["s"]
        mu, crr = _mu(s, surf) * (1 + rng.normal(0.0, 0.03)), _crr(s, surf)
        dist, force, duty = f.get("len", 0.3), crr * s.m * G, ctl["dark" if k == "dark" else "speed"]
        vsafe, steer, fail = math.inf, True, None
        approach = 1.0 if probe else ctl.get("approach", 1.0)
        if k in MM.EDGES:
            fail = _edge(s, f, mu * s.edge, approach)
        elif k == "gap":
            margin, sig = (f["w"] - s.width) / 2, s.drift * (0.3 + 0.7 * duty)
            err = (0.0 if probe else ctl["line"] - nominal["off"]) + rng.normal(0.0, sig)
            narrow = margin < 2 * s.drift * (0.3 + 0.7 * 0.15)  # the error even at the slowest duty
            if margin <= 0:
                fail = ("chassis_wider_than_gap", f"the robot is {s.width:.1f} cm wide overall; the opening is "
                        f"{f['w']:.1f} cm", 0.0)
            elif probe and narrow:
                fail = ("gap_margin_below_drift", f"{margin:.2f} cm to spare per side; its steering error is sigma "
                        f"{s.drift * 0.405:.2f} cm even at the slowest duty", 0.0)
            else:  # it drives straight at the jambs from its line's error
                ep = _drive(s, "gap", f | {"off": -err}, mu, 1.0 if probe else duty)
                if not ep.ok:
                    fail = ("gap_margin_below_drift" if narrow else "clipped_gap_edge",
                            f"{margin:.2f} cm to spare per side; it came {err:+.2f} cm off centre (steering error "
                            f"sigma {sig:.2f} cm at duty {duty:.2f}) and met the jamb: {ep.detail}", 0.5 * ep.progress)
        elif k == "ramp":
            th = math.radians(f["deg"])
            force = s.m * G * (math.sin(th) + crr * math.cos(th))
            ep = _drive(s, "ramp", f, mu, 1.0 if probe else duty)
            what = f"the {f['deg']:.1f} deg {surf} ramp (mu {mu:.2f}) at duty {duty:.2f}: {ep.detail}"
            if ep.ok:
                pass
            elif ep.why == "tipped":
                fail = ("tipped_over", f"tipped over on {what}", 0.5 * ep.progress)
            elif ep.why in ("stuck_stalled", "slow"):
                fail = (("stalled_at_low_duty", f"the drive stalled on {what}") if duty < 0.999 and not probe else
                        ("motor_torque_short_on_ramp", f"climbing needs {force:.1f} N of push; the drive stalls at "
                         f"{s.f_stall:.1f} N; on {what}")) + (0.5 * ep.progress,)
            else:
                fail = ("slips_on_ramp", f"its contacts slipped on {what}", 0.5 * ep.progress)
        elif k == "sand":
            p = s.m * G / (s.area * 1e-4) / 1000
            z, out = p / f["kmod"], s.belly if s.loco == "tracks" else min(s.belly, _climb(s, surf, standing=True))
            if z > out:
                fail = ("sinks_in_sand", f"ground pressure {p:.2f} kPa on {s.area:.0f} cm2 of contact sinks it "
                        f"{z:.2f} cm; it climbs out of {out:.2f} cm (belly clearance {s.belly:.1f} cm)", 0.0)
        elif k == "low" and s.height + 0.2 > f["clear"]:
            fail = ("too_tall_for_clearance", f"the robot is {s.height:.1f} cm tall; the overhang is "
                    f"{f['clear']:.1f} cm", 0.0)
        elif k == "dark":
            vsafe, steer = _vsafe(s, mu)
            if _speed(s, duty, force) > vsafe:
                fail = ("collided_in_dark", f"drove at {_speed(s, duty, force):.2f} m/s; with {s.sense:.2f} m of "
                        f"sensing it can stop from {vsafe:.2f} m/s", 0.5 * vsafe / _speed(s, duty, force))
        what = f"stopped at feature {i + 1}/{n}, {_desc(nominal)}"
        if fail is not None and not (probe and fail[0] in CONTROL_LEVEL):
            return False, (i + fail[2]) / n, fail[0], f"{what}: {fail[1]}"
        v, vb = _speed(s, duty, force), min(vsafe, _speed(s, 1.0, force))
        pen = 0.0 if steer else f["n"] * BACKOFF_S
        if k == "dark" and (vb < _speed(s, 1.0, force) or not steer):
            dark_note = f"; in the dark it sees {s.sense:.2f} m ahead, " + (
                f"enough to stop from {vsafe:.2f} m/s" if steer else f"less than its {s.length / 100:.2f} m length, so "
                f"it touches each of {f['n']:.0f} obstacles at <= {vsafe:.2f} m/s and backs off ({pen:.0f} s)")
        if vb <= 0.02:
            return False, i / n, "motor_torque_short", f"{what}: moving needs {force:.1f} N; the drive stalls at " \
                f"{s.f_stall:.1f} N"
        if v <= 0.02 and not probe:
            return False, (i + 0.25) / n, "stalled_at_low_duty", f"{what}: at duty {duty:.2f} the drive cannot " \
                f"move against {force:.1f} N"
        v = vb if probe else v
        e_seg = force * dist / s.eta + s.p_w * (dist / v + pen)
        e_best += force * dist / s.eta + s.p_w * (dist / vb + pen)
        t, t_best = t + dist / v + pen, t_best + dist / vb + pen
        if e + e_seg > s.usable_j:
            frac = (s.usable_j - e) / e_seg
            return False, (i + frac) / n, "battery_exhausted" if e_best > s.usable_j else "drained_at_chosen_speed", (
                f"{what}: the {s.wh:.1f} Wh pack ({s.usable_j / 3600:.2f} Wh usable) ran out {100 * frac:.0f}% of the "
                f"way; at full speed it spends {e_best / 3600:.2f} Wh to here ({force / s.eta + s.p_w / vb:.2f} J/m "
                f"on {surf} at {s.m:.2f} kg)")
        e += e_seg
    if t > secs:
        mech = ("sensing_range_short" if dark_note else "too_slow_for_time_limit") if t_best > secs \
            else "too_slow_at_chosen_speed"
        return False, 1 - 0.5 / n * (1 - secs / t), mech, (
            f"reached the end in {t:.0f} s ({t_best:.0f} s at best); {secs:.0f} s allowed{dark_note}")
    return True, 1.0, "", f"reached in {t:.0f} s of {secs:.0f}, using {e / 3600:.3f} Wh"


CONTROLS = {  # System 1's controls: range, starting value, the features that call for them
    "speed": ((0.15, 1.0), 0.6, {"floor", "gap", "ramp", "sand", "low", "dark", *MM.EDGES}),
    "line": ((-6.0, 6.0), 0.0, {"gap"}), "approach": ((0.1, 1.0), 0.5, set(MM.EDGES)),
    "dark": ((0.1, 1.0), 0.5, {"dark"})}


def _space(route: list[dict[str, Any]]) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    use = [c for c, (_, _, kinds) in CONTROLS.items() if kinds & {f["k"] for f in route}]
    return {c: CONTROLS[c][0] for c in use}, {c: CONTROLS[c][1] for c in use}


# ---------------------------------------------------------------- the catalogue
def _p(lo: float, hi: float, unit: str, doc: str) -> Param:
    return Param(lo=lo, hi=hi, unit=unit, doc=doc)


_RATIO_DOC = "gearbox reduction, rounded to the nearest stocked ratio ({}): more reduction multiplies the stall " \
    "force at the ground and divides the top speed"
CATALOGUE = [
    PartType(name="printed_tub", slot="chassis", stock=True, doc="a PLA tub that carries the electronics, motors "
             "and battery; the running gear mounts on its sides", provides="the body's width, length and belly height",
             params={"width": _p(10, 26, "cm", "inner width; the running gear adds its own width on each side"),
                     "length": _p(14, 34, "cm", "length; sets the wheelbase and the floor area inside"),
                     "ride_height": _p(0.8, 6, "cm", "belly clearance above the contact line; the tub's 4 cm depth "
                                       "sits on top of it")}),
    PartType(name="wheels4", slot="locomotion", stock=True, doc="four printed hubs with rubber tyres, each driven, "
             "steered by skidding", provides="simple, light rolling on firm floors",
             params={"wheel_radius": _p(2.0, 4.5, "cm", "a larger wheel meets an edge lower on its rim, spans "
                                        "wider gaps, has a longer soft-ground patch and rolls easier, and is taller")}),
    PartType(name="mecanum4", slot="locomotion", doc="four mecanum wheels whose free rollers at 45 degrees let the "
             "robot move sideways and turn without skidding", provides="holonomic motion and fine positioning on "
             "smooth floors; the rollers spin freely across their axis, so traction is lower",
             params={"wheel_radius": _p(2.4, 4.0, "cm", "wheel size (48 to 80 mm wheels)")}),
    PartType(name="rocker_bogie6", slot="locomotion", doc="six driven wheels on a rocker-bogie linkage that keeps "
             "every wheel on the ground; the tub hangs from the rocker pivots, its ride_height counted from the axles",
             provides="the wheels behind push a climbing wheel into an edge; six contact patches; a tall linkage",
             params={"wheel_radius": _p(3.0, 6.0, "cm", "wheel size; the linkage stands about 2 cm above the wheels"),
                     "bogie_length": _p(8, 18, "cm", "distance between a bogie's two wheels; the bogie can tilt "
                                        "about 40 degrees")}),
    PartType(name="tracks", slot="locomotion", doc="a pair of rubber tracks with grousers around a drive sprocket "
             "(1.8 cm radius) and a raised front idler, one motor per side", provides="a long contact patch with "
             "low ground pressure and grip on loose ground; a sloped front run meets edges; heavier and with more "
             "rolling resistance than wheels",
             params={"track_length": _p(14, 32, "cm", "ground contact length"),
                     "idler_height": _p(2, 8, "cm", "height of the front idler's axle; the track stands 1.5 cm "
                                        "above it")}),
    PartType(name="n20_gearmotor", slot="drive", stock=True, doc="N20 micro gearmotors with 6 V windings, 11 g each, "
             "one per driven wheel or track, on TB6612 dual drivers; torque and speed scale with the pack's voltage, "
             "which must be 3-9 V", provides="light drive, 0.75 A stall current per motor at 6 V",
             params={"gear_ratio": _p(50, 298, "", _RATIO_DOC.format("50, 100, 150, 298"))}),
    PartType(name="n20_encoder_gearmotor", slot="drive", doc="N20 gearmotors with magnetic encoders for closed-loop "
             "wheel speed", provides="the same force and speed as the plain N20, with 40% less steering error",
             params={"gear_ratio": _p(50, 298, "", _RATIO_DOC.format("50, 100, 150, 298"))}),
    PartType(name="jga25_gearmotor", slot="drive", doc="25 mm metal gearmotors with 12 V windings, 95 g each, on "
             "L298N drivers; torque and speed scale with the pack's voltage, which must be 6-18 V",
             provides="about 6.5 times the N20's shaft torque at 12 V; 2.2 A stall current per motor at 12 V",
             params={"gear_ratio": _p(21, 90, "", _RATIO_DOC.format("21, 34, 46, 90"))}),
    PartType(name="nimh_pack", slot="battery", stock=True, doc="AA NiMH cells in series, 1.2 V (1.4 V charged), 2.4 Wh and 27 g each",
             provides="energy at about 11 g per Wh; at most 4 A", params={
                 "capacity": _p(2.4, 19.2, "Wh", "pack energy, rounded up to whole 2.4 Wh cells; each adds 27 g")}),
    PartType(name="liion_pack", slot="battery", doc="18650 Li-ion cells in series with a protection board, 3.7 V "
             "(4.2 V charged), 9.4 Wh and 46 g each", provides="energy at about 5 g per Wh; up to 10 A", params={
                 "capacity": _p(9.4, 28.2, "Wh", "pack energy, rounded up to whole 9.4 Wh cells; each adds 46 g")}),
    PartType(name="bump_switch", slot="sensor", stock=True, doc="two microswitches behind a sprung front bumper",
             provides="detects an obstacle by touching it, at up to 0.25 m/s"),
    PartType(name="ultrasonic", slot="sensor", doc="an HC-SR04 sonar on the tub's front face",
             provides="ranges obstacles to 2 m in any light; adds no height; 0.08 W"),
    PartType(name="camera", slot="sensor", doc="a small camera on a 2.5 cm mast with an IR LED",
             provides="a wide view to 3 m in a lit room; in the dark its LED lights 0.12 m; 0.7 W"),
    PartType(name="lidar", slot="sensor", doc="a 2D scanning lidar on a 4 cm turret",
             provides="ranges 360 degrees to 6 m in any light; 45 g, 1.5 W"),
]


class MobileWorld:
    name = "mobile"
    description = ("A small ground robot explores a house and its garden.  Each task is a route from its dock to "
                   "an exploration target, over floors, edges, openings, trenches, ramps, sand, low furniture and "
                   "dark corridors.  System 1 drives each route (cruise duty, the line through an opening, the "
                   "approach duty at an edge, the speed in the dark); the body decides what any driving can reach.")

    def tasks(self) -> list[Task]:
        out = []
        for name, (where, secs, route) in ROUTES.items():
            out.append(self._task(name, "train", 0, route, where, secs))
            for k in range(2):
                rng = np.random.default_rng([zlib.crc32(name.encode()), k])
                varied = [{key: v * rng.uniform(0.95, 1.05) if isinstance(v, float) and key != "n" else v
                           for key, v in f.items()} for f in route]
                out.append(self._task(name, "test", k, varied, where, secs))
        return out

    @staticmethod
    def _task(name: str, split: str, k: int, route: list[dict[str, Any]], where: str, secs: float) -> Task:
        facts = f"{where}: " + ", then ".join(_desc(f) for f in route) + f"; {secs:g} s allowed"
        return Task(id=f"{name}_{split}{k}", split=split, group=name, facts=facts,  # type: ignore[arg-type]
                    data={"route": route, "secs": secs, "seed": k + (100 if split == "test" else 0)})

    def catalogue(self) -> list[PartType]:
        return list(CATALOGUE)

    def stock(self) -> Body:
        return Body(parts={
            "chassis": Part(type="printed_tub", params={"width": 18.0, "length": 20.0, "ride_height": 2.0}),
            "locomotion": Part(type="wheels4", params={"wheel_radius": 3.0}),
            "drive": Part(type="n20_gearmotor", params={"gear_ratio": 100.0}),
            "battery": Part(type="nimh_pack", params={"capacity": 9.6}),
            "sensor": Part(type="bump_switch")})

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        s = _spec(body)
        route, secs = task.data["route"], float(task.data["secs"])
        key = zlib.crc32(task.id.encode())
        space, start = _space(route)

        def run(ctl: dict[str, float], k: int) -> Outcome:
            t0 = time.perf_counter()
            ok, score, mech, detail = _attempt(s, route, secs, ctl, np.random.default_rng([key, k & 0xFFFFFFFF]))
            return Outcome(task=task.id, success=ok, score=score, mechanism=mech, detail=detail,
                           sim_s=time.perf_counter() - t0)

        out = local_search(run, space, start, m=32, confirm=2, seed=seed + int(task.data.get("seed", 0)))
        if not out.success and out.mechanism in CONTROL_LEVEL:  # a limit of the body further on, or the search's
            ok, _, mech, detail = _attempt(s, route, secs, PROBE, _Calm(), probe=True)
            out = out.model_copy(update={"mechanism": "not_found_by_search", "detail": f"{out.mechanism}: {out.detail}"}
                                 if ok else {"mechanism": mech, "detail": detail})
        return out

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        s = _spec(body)
        feats = [f for t in ([t for t in tasks if t.split == "train"] or tasks) for f in t.data["route"]]
        edges = sorted({f["s"] for f in feats if f["k"] in MM.EDGES}, key=lambda x: SURF[x][0])
        rows = [f"mass {s.m:.2f} kg; {s.width:.1f} cm wide overall, {s.length:.1f} cm long, {s.height:.1f} cm tall; "
                f"belly clearance {s.belly:.1f} cm",
                f"drive: {s.n} motors at {s.volts:.1f} V; stall force at the ground {s.f_stall:.1f} N ({s.f_stall / (s.m * G):.2f} of its "
                f"weight); no-load speed {s.v_nl:.2f} m/s",
                "edges it climbs at full duty, driven in MuJoCo: " + ", ".join(
                    f"{_climb(s, e):.2f} cm on {e} (mu {_mu(s, e) * s.edge:.2f})" for e in edges)]
        rows.append(f"gaps: a {s.track:.0f} cm track bridges about {0.45 * s.track:.1f} cm with its centre of mass "
                    "supported" if s.loco == "tracks" else
                    f"gaps: a wheel spans {2 * s.r:.1f} cm; over a wider trench it drops to the bottom")
        for surf in sorted({f["s"] for f in feats if f["k"] == "ramp"}):
            c, mu = _crr(s, surf), _mu(s, surf)
            arg = 0.95 * s.f_stall / (s.m * G * math.sqrt(1 + c * c))
            torque = "any slope" if arg >= 1 else f"{max(0.0, math.degrees(math.asin(arg) - math.atan(c))):.1f} deg"
            rows.append(f"slopes on {surf}: traction holds to {math.degrees(math.atan(max(0.0, mu - c))):.1f} deg "
                        f"(mu {mu:.2f}, rolling resistance {c:.3f}); the stall force alone would climb {torque}")
        for kmod in sorted({round(f["kmod"], 2) for f in feats if f["k"] == "sand"}):
            p = s.m * G / (s.area * 1e-4) / 1000
            out = s.belly if s.loco == "tracks" else min(s.belly, _climb(s, "sand", standing=True))
            rows.append(f"soft ground: {s.area:.0f} cm2 of contact, ground pressure {p:.2f} kPa; it would sink "
                        f"{p / kmod:.2f} cm where {kmod:.2f} kPa sinks 1 cm, and is stuck deeper than {out:.2f} cm "
                        + ("(its belly)" if s.loco == "tracks" else "(its belly, or the rut edge it can climb)"))
        jm = {u: _crr(s, u) * s.m * G / s.eta + s.p_w / max(_speed(s, 1.0, _crr(s, u) * s.m * G), 1e-3)
              for u in ("wood", "grass")}
        rows.append(f"energy: {s.wh:.1f} Wh pack ({s.usable_j / 3600:.2f} Wh usable); at full speed {jm['wood']:.2f} "
                    f"J/m on wood, {jm['grass']:.2f} J/m on grass: range {s.usable_j / jm['grass'] / 1000:.2f} km on "
                    f"grass ({s.p_w:.2f} W of electronics)")
        rows.append(f"steering error sigma {s.drift * 0.405:.2f} cm at the slowest duty, {s.drift:.2f} cm at full")
        vs, steer = _vsafe(s, _mu(s, "wood"))
        rows.append(f"sensing in the dark: {s.sense:.2f} m ahead (body length {s.length / 100:.2f} m); " +
                    (f"stops in time from {vs:.2f} m/s" if steer else
                     f"it touches each obstacle, at most {BUMP_V:.2f} m/s, and backs off ({BACKOFF_S:.0f} s each)"))
        return "\n".join(rows)

    def principles(self) -> list[str]:
        return [
            "A driven wheel climbs an edge by friction at the contact: alone it climbs h = r(1 - 1/sqrt(1+mu^2)); "
            "when the wheels behind push, the edge force can lean back further, h = r(1 - cos(atan mu + "
            "atan(mu (1-f)/f))) with f the weight share on the climbing wheels; it never exceeds the radius r.",
            "A rigid chassis high-centres on an edge unless its belly stays above the edge as the rear wheels "
            "arrive; a track climbs any edge that meets its sloped front run, up to about its front idler height.",
            "A wheel bridges a gap narrower than its diameter, dipping r - sqrt(r^2 - w^2/4); over a wider gap it "
            "drops to the bottom and must climb out.  A rigid vehicle bridges a gap while its centre of mass stays "
            "over support, about 0.45 of a track's contact length.",
            "Ground pressure is weight over contact area; in soft soil sinkage grows with pressure, and a vehicle "
            "that sinks deeper than it can climb out of its own rut, or deeper than its belly, is stuck.",
            "A slope holds while tan(theta) + C_rr stays below the traction coefficient of the driven contacts, and "
            "climbs while the stall force at the ground exceeds m g (sin + C_rr cos); a gear reduction trades top "
            "speed for that force, and every added gram raises the force needed.",
            "Energy per metre is the resisting force over drive efficiency plus electronics power over speed: mass "
            "and rolling resistance raise the first term, a slow robot pays more of the second.",
            "A robot passes an opening only if its overall width plus its steering error fits, and passes under "
            "something only if its overall height does; the running gear adds to both.",
            "Among obstacles a robot may go only as fast as it can stop within its sensing range; one that cannot "
            "see an obstacle a body length ahead must touch it, back off and turn.",
        ]

    def bom(self, body: Body) -> list[BomLine]:
        ch, lo, dr, bt, se = (body.parts[k] for k in ("chassis", "locomotion", "drive", "battery", "sensor"))
        n = _motors(body)
        out = [BomLine(item="print:PLA", qty=round(_tub_g(ch.params["width"], ch.params["length"]), 1), slot="chassis")]
        pla = _loco_g(lo)[1]
        if pla:
            out.append(BomLine(item="print:PLA", qty=round(pla, 1), slot="locomotion", note="hubs, arms, mounts"))
        out += [BomLine(item=i, qty=q, slot="locomotion") for i, q in {
            "tracks": [("rubber_track_kit", 1)], "mecanum4": [("mecanum_wheel", 4)],
            "rocker_bogie6": [("tire_oring", 6), ("bearing_608", 6)]}.get(lo.type, [("tire_oring", 4)])]
        out += [BomLine(item=f"{dr.type}_{_ratio(dr)}", qty=n, slot="drive"),
                BomLine(item=DRIVER[dr.type][0], qty=math.ceil(n / 2), slot="drive"),
                BomLine(item=CELL[bt.type][0], qty=_cells(bt), slot="battery")]
        if bt.type == "liion_pack":
            out.append(BomLine(item="bms_board", qty=1, slot="battery"))
        return out + [BomLine(item=SENSOR[se.type][4], qty=SENSOR[se.type][5], slot="sensor")]

    def check(self, body: Body) -> list[str]:
        errs = []
        ch, lo, dr, bt = (body.parts[k] for k in ("chassis", "locomotion", "drive", "battery"))
        w, ln, ride = ch.params["width"], ch.params["length"], ch.params["ride_height"]
        s, n = _spec(body), _motors(body)
        inner = (w - 1) * (ln - 1)
        motor_cm2 = n * (15.0 if dr.type == "jga25_gearmotor" else 4.2)
        need = 50.0 + _cells(bt) * CELL[bt.type][3] + motor_cm2
        if inner < need:
            errs.append(f"the tub's {inner:.0f} cm2 floor cannot hold the electronics (50 cm2), {_cells(bt)} cells "
                        f"({_cells(bt) * CELL[bt.type][3]:.0f} cm2) and {n} motors ({motor_cm2:.0f} cm2)")
        if lo.type in ("wheels4", "mecanum4") and ln < 2 * s.r + 3:
            errs.append(f"{s.r:.1f} cm wheels need a tub at least {2 * s.r + 3:.1f} cm long so front and rear "
                        f"wheels do not touch; it is {ln:.1f} cm")
        if lo.type in ("wheels4", "mecanum4") and ride > s.r + 1.5:
            errs.append(f"the axle brackets hold the belly at most 1.5 cm above the {s.r:.1f} cm axle; "
                        f"ride_height is {ride:.1f} cm")
        if lo.type == "tracks":
            tl, hi = lo.params["track_length"], lo.params["idler_height"]
            if abs(tl - ln) > 4:
                errs.append(f"the track frames mount on the tub's sides: a {tl:.1f} cm track needs a tub "
                            f"{tl - 4:.1f}-{tl + 4:.1f} cm long; it is {ln:.1f} cm")
            if hi > 0.35 * tl:
                errs.append(f"a {hi:.1f} cm idler on a {tl:.1f} cm track makes the front run a wall, not a slope "
                            f"(at most {0.35 * tl:.1f} cm)")
            if ride > hi + 1.5:
                errs.append(f"the tub hangs between the track frames: its belly can be at most {hi + 1.5:.1f} cm up")
        if lo.type == "rocker_bogie6" and ln < lo.params["bogie_length"] + 2:
            errs.append(f"the rocker pivots need a tub at least {lo.params['bogie_length'] + 2:.1f} cm long")
        lo_v, hi_v = (f * WINDING[dr.type] for f in V_RANGE)
        full = _cells(bt) * VOLTS[bt.type][1]
        if s.volts < lo_v or full > hi_v:
            errs.append(f"{_cells(bt)} {bt.type} cells in series give {s.volts:.1f} V ({full:.1f} V charged); the "
                        f"{WINDING[dr.type]:.0f} V {dr.type} winding takes {lo_v:.1f}-{hi_v:.1f} V")
        amps, cap = n * MOTOR[dr.type][4] * s.volts / WINDING[dr.type], CELL[bt.type][4]
        if amps > cap:
            errs.append(f"{n} {dr.type} motors draw up to {amps:.1f} A at stall; the {bt.type} supplies at most "
                        f"{cap:.0f} A")
        if s.f_stall < 0.3 * s.m * G:
            errs.append(f"the drive's {s.f_stall:.1f} N stall force cannot start {s.m:.2f} kg up a modest slope "
                        f"(it needs at least {0.3 * s.m * G:.1f} N)")
        if not errs:  # parts cannot share space: the built model, its rockers and bogies swung (D-077)
            seen = set()
            for a, b, depth, pose in MM.self_clashes(_robot(s, "wood")):
                if (pair := (_part_name(a), _part_name(b))) in seen:
                    continue
                seen.add(pair)
                errs.append(f"the {_part_name(a)} and the {_part_name(b)} pass {1000 * depth:.1f} mm into each other "
                            f"({pose}); two parts cannot be in one place")
        return errs

    def manifest(self) -> dict[str, Any]:
        items = {k: dict(zip(("on_hand", "usd", "lead_days", "doc"), v, strict=True)) for k, v in ITEMS.items()}
        for kind, ratios in RATIOS.items():
            for r in ratios:
                items[f"{kind}_{r}"] = {"on_hand": 4 if (kind, r) == ("n20_gearmotor", 100) else 0,
                                        "usd": {"n20_gearmotor": 3.5, "n20_encoder_gearmotor": 6.0}.get(kind, 7.5),
                                        "lead_days": 3 if kind == "n20_gearmotor" else 4, "doc": f"{kind} at {r}:1"}
        return {"budget": {"usd": 60, "lead_days": 7, "fab_minutes": 600},
                "fabrication": {"print:PLA": {"minutes_per_unit": 1.0, "unit": "g", "doc": "FDM printer"}},
                "items": items}
