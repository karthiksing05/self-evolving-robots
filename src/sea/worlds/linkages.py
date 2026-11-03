"""The talking robot's drives: how each servo moves the articulator it serves (D-071).

The phonology world gives System 1 position controls (``phonology.rig``); this module is the mechanism between a servo
and each control, the one ``rig_cad`` draws and ``phonology.check`` holds to physical law.  Each drive is planar and in
mm: a servo turns an arm (a horn, a crank, an eccentric or a spool) and a rod or cord carries the motion to a pin on
the moving part.  Its laws, at every sampled posture over the control ranges (the corners of a coupled pair included):
- reach: the rod spans the gap from the arm to the pin (no rod stretches);
- travel: the arm turns no more than the MG90S's 180 degrees over the whole range;
- transmission: the push reaches the part along its guide.  The angle between the rod and the normal to a slide (or
  between the two rods of the five-bar at the tip blade) is at least 30 degrees; below it the rod presses the guide
  more than it moves the part, and the five-bar's tip loses stiffness;
- no dead centre: the horn and its rod stay at least 10 degrees from lying in line, where a servo's motion stops moving
  the pin and noise can fold the linkage over;
- draw: a cord's spool takes up the cord the opening needs within the servo's travel, and a pinion moves its rack
  the slide's whole travel within it;
- the tongue's rail keeps its sled clear of the glottis: the reed housing, or the larynx slide's carriage.

Frame (mm): x along the tract from the glottis to the teeth at x = 0, y across, z up with the bore's axis at z = 0
(``rig_cad``'s frame).  A drive acting in the x-z plane is drawn by a servo lying on its side (shaft along -y); one in
the x-y plane by a standing servo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SPAN = math.pi  # an MG90S turns 180 degrees
MU_MIN = math.radians(30.0)  # the smallest transmission angle a drive may run at
DEAD = math.radians(10.0)  # a horn and its rod stay this far from lying in line
SAMPLES = 21  # postures sampled along each control (a coupled pair: a grid of SAMPLES x SAMPLES)

BORE_MM, WALL_MM = 12.0, 4.0  # the cast's bore radius and its silicone wall: the outer radius is their sum
ZP = -66.0  # the base plate's top
FOOT_DOWN = 9.6  # a lying servo's shaft above the foot of its mount
FOOT_STAND = 29.7  # a standing servo's dome top above the foot of its mount
GRID = 5.0  # where fixed parts stand on the base plate (``design.language.GRID``)
X_STEM = -10.0  # the tip stem's front limit: its crosshead and pushrod eyes clear the cast's flange
RAIL_MM = {"cam_tongue": (70.0, 100.0), "tip_tongue": (70.0, 100.0), "long_rail_tongue": (100.0, 150.0)}  # travel, rail
ROLLER = 3.0  # the follower's roller radius
CAM_R = 9.0  # the eccentric's radius
SLED_BACK = 25.0  # the tongue's sled reaches this far behind its pad (the cam servo lies along it)
GLOTTIS_FRONT = {"larynx_tube": 3.0}  # mm in front of the glottis the glottal hardware reaches at the sled's height:
# the reed housing's stand ends at the glottis; the larynx slide's carriage reaches 2 mm in front of it (+1 clearance)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """The angle between two 2-D vectors, 0..pi."""
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.acos(max(-1.0, min(1.0, c)))


def _wrap(a: np.ndarray) -> np.ndarray:
    """Angles unwrapped about their first value: the span a servo turns through."""
    return np.unwrap(np.asarray(a, float))


@dataclass(frozen=True)
class Dyad:
    """A servo at ``s`` turning a horn of length ``a``; a rod of length ``b`` from its tip to a pin.  ``elbow`` (+1 or
    -1) picks the side of the line from the servo to the pin the horn's tip lies on."""

    s: tuple[float, float]
    a: float
    b: float
    elbow: float

    def reaches(self, p: np.ndarray) -> bool:
        n = float(np.linalg.norm(np.asarray(p) - self.s))
        return abs(self.b - self.a) < n < self.a + self.b

    def tip(self, p: np.ndarray) -> np.ndarray:
        """The horn's tip when the rod's far end is at ``p`` (clamped to the reachable ring: check ``reaches``)."""
        s = np.asarray(self.s, float)
        d = np.asarray(p, float) - s
        n = float(np.linalg.norm(d))
        n = min(max(n, abs(self.b - self.a) + 1e-6), self.a + self.b - 1e-6)
        along = (self.a ** 2 - self.b ** 2 + n * n) / (2 * n)
        h = math.sqrt(max(self.a ** 2 - along ** 2, 0.0))
        u = d / np.linalg.norm(d)
        return s + u * along + self.elbow * h * np.array([-u[1], u[0]])

    def horn(self, p: np.ndarray) -> float:
        """The horn's angle (rad) when the pin is at ``p``."""
        t = self.tip(p) - self.s
        return math.atan2(t[1], t[0])

    def toggle(self, p: np.ndarray) -> float:
        """How far (rad) the horn and its rod are from lying in line: 0 at a dead centre, pi/2 at best."""
        t = self.tip(p)
        g = _angle(t - np.asarray(self.s), np.asarray(p) - t)
        return min(g, math.pi - g)


@dataclass(frozen=True)
class Crank:
    """An offset slider-crank: a standing servo at (``x_c``, ``y_c``) turns a crank of radius ``a``; a rod of length
    ``b`` drives a pin that slides along x at ``y_pin``.  The travel lies on one arc between the two dead centres."""

    x_c: float
    y_c: float
    y_pin: float
    a: float
    b: float

    def x_of(self, th: float) -> float:
        return self.a * math.cos(th) + math.sqrt(self.b ** 2 - (self.y_pin - self.y_c - self.a * math.sin(th)) ** 2)

    def arc(self) -> tuple[float, float]:
        """The crank angles of the two dead centres, ordered so the pin's x rises from the first to the second."""
        ths = np.linspace(0.0, 2 * math.pi, 1441)
        xs = np.array([self.x_of(t) for t in ths])
        lo, hi = float(ths[int(np.argmin(xs))]), float(ths[int(np.argmax(xs))])
        return lo, hi if hi > lo else hi + 2 * math.pi

    def span(self) -> tuple[float, float]:
        """The pin's x (model frame) from one dead centre to the other."""
        lo, hi = self.arc()
        return self.x_c + self.x_of(lo), self.x_c + self.x_of(hi)

    def angle(self, x: float) -> float:
        """The crank angle (rad) that puts the pin at ``x`` (model frame), on the arc; the nearer end if out of span."""
        lo, hi = self.arc()
        target = x - self.x_c
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.x_of(mid) < target else (lo, mid)
        return (lo + hi) / 2

    def tip(self, th: float) -> np.ndarray:
        return np.array([self.x_c + self.a * math.cos(th), self.y_c + self.a * math.sin(th)])

    def mu(self, x: float) -> float:
        """The transmission angle at the slide: the rod's angle to the slide's normal."""
        rod = np.array([x, self.y_pin]) - self.tip(self.angle(x))
        return _angle(rod, np.array([0.0, 1.0])) if rod[1] >= 0 else _angle(rod, np.array([0.0, -1.0]))


def _crank_for(x_far: float, travel: float, y_pin: float, y_c: float, extra: float) -> Crank:
    """A slider-crank whose pin covers ``x_far`` .. ``x_far + travel`` with 1 mm to spare at the far end, its servo on
    the base plate's grid: the crank's radius is half the travel plus 2 mm (so the stroke between its dead centres
    leaves a grid step to place the servo in), the rod reaches past the offset by ``extra``."""
    a = travel / 2 + 2.0
    b = (y_pin - y_c) + a + extra
    probe = Crank(0.0, y_c, y_pin, a, b)
    lo, hi = probe.span()
    x_c = math.floor((x_far - lo - 1.0) / GRID) * GRID  # the far end inside the span
    return Crank(x_c, y_c, y_pin, a, b)


# ---------------------------------------------------------------- the drives
def rail_crank(rail_start: float, kind: str) -> Crank:
    """The rail servo's slider-crank: it slides the tongue's sled, whose pin is 18.75 mm behind the pad."""
    travel = RAIL_MM[kind][0]
    return _crank_for(-10 * rail_start - travel - 18.75, travel, -8.75, -36.0, 24.0)


def sled_pin(place: float) -> float:
    return -10 * place - 18.75


def cam_rise(stroke: float, phi: float) -> float:
    """How far the eccentric (offset ``stroke``/2 on the cam servo's horn) has lifted the follower at horn angle phi."""
    e = 5 * stroke
    return -e * math.cos(phi) + math.sqrt(CAM_R ** 2 - (e * math.sin(phi)) ** 2) - (CAM_R - e)


def cam_angle(stroke: float, lift: float) -> float:
    """The cam servo's angle (0..pi) that lifts the follower ``lift`` mm."""
    lo, hi = 0.0, math.pi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cam_rise(stroke, mid) < lift else (lo, mid)
    return (lo + hi) / 2


def cam_mu(stroke: float, phi: float) -> float:
    """The transmission angle at the follower: the contact normal (through the cam's centre) against its guide."""
    e = 5 * stroke
    return math.pi / 2 - math.asin(e * math.sin(phi) / (CAM_R + ROLLER))


TIP_BRIDGE = 44.0  # the palate bridge's top: the tip servos lie on it
TIP_DYADS = (Dyad((-27.0, TIP_BRIDGE + FOOT_DOWN), 12.0, 30.0, 1.0), Dyad((-63.0, TIP_BRIDGE + FOOT_DOWN), 24.0, 46.0, -1.0))
TIP_CROSS = 33.0  # the crosshead's height with the blade up


def tip_pin(place: float, height: float) -> np.ndarray:
    """Where the tip blade's crosshead sits (x, z)."""
    return np.array([min(-10 * place, X_STEM), TIP_CROSS - 10 * height])


def jaw_drop(rest_area: float, span: float, jaw: float) -> float:
    """How far the jaw has come down (mm) at jaw opening ``jaw`` (0..1) from the cast rest gap."""
    return 20 * (math.sqrt((rest_area + span * jaw) / math.pi) - math.sqrt(rest_area / math.pi))


def jaw_dyad(tube: float, rest_area: float, span: float) -> tuple[Dyad, np.ndarray]:
    """The jaw servo's horn and link, and the jaw's pin with the jaw shut (x, z)."""
    s = (round((tube + 34) / GRID) * GRID, ZP + FOOT_DOWN)
    pin = np.array([tube + 5, -(BORE_MM + WALL_MM) - 2.5])
    return Dyad(s, 12.0, float(np.linalg.norm(pin - s)), -1.0), pin


def ring_clip(lip_radius: float, protrusion: float) -> np.ndarray:
    """The ring clip's pin (x, z): forward with the protrusion, up as the drawstring closes the ring."""
    ring = 10 * max(lip_radius, 0.0) + 2.0
    return np.array([10.0 + 10 * protrusion, -(ring + 2) - 2.5 - 3.0])


def protrusion_dyad(max_protrusion: float, ring_open: float) -> Dyad:
    s = (50.0, ZP + FOOT_DOWN)
    mid = ring_clip(ring_open, max_protrusion / 2)
    return Dyad(s, 14.0, float(np.linalg.norm(mid - s)), -1.0)


SPOOL_R = 16.0  # the drawstring servo's spool: both ends of the cord wind onto it


def draw_needed(lo: float, hi: float) -> float:
    """Cord the drawstring takes up from the ring fully open to fully drawn (mm): the ring's circumference change.
    The cord runs to the servo in a sheath fixed to the ring clip, so pushing the ring forward draws none."""
    return 2 * math.pi * 10 * (max(hi, 0.0) - max(lo, 0.0))


def spool_angle(lip_radius: float, lo: float, hi: float) -> float:
    """The drawstring servo's turn (rad) from the ring fully open: each end of the cord takes up half the draw."""
    return draw_needed(lip_radius, hi) / (2 * SPOOL_R)


PINION = (0.5, 40)  # the larynx servo's pinion: module (mm) and teeth
PINION_R = PINION[0] * PINION[1] / 2  # its pitch radius: the rack moves this far per radian


def pinion_angle(drop: float) -> float:
    """The larynx servo's turn (rad) that lowers the larynx ``drop`` mm."""
    return drop / PINION_R


# ---------------------------------------------------------------- the laws
def _crank_faults(name: str, c: Crank, xs: np.ndarray, what: str) -> list[str]:
    out = []
    lo, hi = c.span()
    if xs.min() < lo - 1e-6 or xs.max() > hi + 1e-6:
        out.append(f"{name}: the crank reaches {lo:.1f} to {hi:.1f} mm along the tract, and {what} needs "
                   f"{xs.min():.1f} to {xs.max():.1f} mm")
    ths = _wrap([c.angle(x) for x in xs])
    if np.ptp(ths) > SPAN:
        out.append(f"{name}: the crank turns {math.degrees(np.ptp(ths)):.0f} degrees over the travel, past the servo's "
                   f"{math.degrees(SPAN):.0f}")
    mus = np.array([c.mu(x) for x in xs])
    if mus.min() < MU_MIN:
        out.append(f"{name}: the rod meets the slide at a {math.degrees(mus.min()):.0f} degree transmission angle "
                   f"(at {xs[int(np.argmin(mus))]:.1f} mm), below {math.degrees(MU_MIN):.0f}")
    return out


STAND_X = (17.0, 1.0)  # the reed housing's stand: from 17 mm behind the glottis to 1 mm behind it
STAND_Y = (6.0, 18.0)  # and across, a bracket on the +y side: the rail crank and its rod pass on the other
ROD_EYE = 3.2  # the connecting rod's eye radius: the widest the crank and rod are


def stand(r, tract: str) -> tuple[float, float, float, float]:
    """The reed housing's stand (x0, x1, y0, y1) in the plane of the rail crank, over the larynx's whole travel: it
    stands from the base plate to under the cast, so the crank and its rod cross its height."""
    g_hi = -10 * r.length
    g_lo = g_hi - 10 * r.controls.get("larynx", (0.0, 0.0))[1]
    return g_lo - STAND_X[0], g_hi - STAND_X[1], STAND_Y[0], STAND_Y[1]


def _crank_clear(name: str, c: Crank, xs: np.ndarray, box: tuple[float, float, float, float]) -> list[str]:
    """The crank and its rod sweep the plane of the crank: they must pass clear of the box (grown by the rod's eye)."""
    x0, x1, y0, y1 = box[0] - ROD_EYE, box[1] + ROD_EYE, box[2] - ROD_EYE, box[3] + ROD_EYE
    t = np.linspace(0.0, 1.0, 41)[:, None]
    for x in xs:
        tip = c.tip(c.angle(x))
        pts = np.vstack([np.array([c.x_c, c.y_c]) + t * (tip - [c.x_c, c.y_c]), tip + t * (np.array([x, c.y_pin]) - tip)])
        if ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)).any():
            return [f"{name}: with the pad {(-x - 18.75) / 10:.2f} cm behind the teeth the crank and its rod pass "
                    f"through the reed housing's stand ({-box[1] / 10:.1f}-{-box[0] / 10:.1f} cm behind the teeth)"]
    return []


def _dyad_faults(name: str, d: Dyad, pins: list[np.ndarray], normal: np.ndarray | None, where: list[str]) -> list[str]:
    """Reach, travel, dead centre, and (``normal``: the slide's normal, or None) the transmission at a slide."""
    out = []
    bad = [w for p, w in zip(pins, where, strict=True) if not d.reaches(p)]
    if bad:
        out.append(f"{name}: a {d.a:g} mm horn and a {d.b:.1f} mm rod cannot reach the pin at {bad[0]}")
        return out
    ang = _wrap([d.horn(p) for p in pins])
    if np.ptp(ang) > SPAN:
        out.append(f"{name}: the horn turns {math.degrees(np.ptp(ang)):.0f} degrees over the range, past the servo's "
                   f"{math.degrees(SPAN):.0f}")
    tog = np.array([d.toggle(p) for p in pins])
    if tog.min() < DEAD:
        i = int(np.argmin(tog))
        out.append(f"{name}: the horn and rod come within {math.degrees(tog[i]):.0f} degrees of a dead centre at "
                   f"{where[i]}")
    if normal is not None:
        mus = []
        for p in pins:
            rod = np.asarray(p) - d.tip(p)
            g = _angle(rod, normal)
            mus.append(min(g, math.pi - g))
        i = int(np.argmin(mus))
        if mus[i] < MU_MIN:
            out.append(f"{name}: the rod meets the slide at a {math.degrees(mus[i]):.0f} degree transmission angle at "
                       f"{where[i]}, below {math.degrees(MU_MIN):.0f}")
    return out


def faults(r, parts: dict) -> list[str]:
    """Every law a body's drives break over its control ranges (``r``: its ``phonology.Rig``; ``parts``: its parts)."""
    from sea.worlds import phonology as P

    out: list[str] = []
    to, li = parts["tongue"], parts["lips"]
    u = np.linspace(0.0, 1.0, SAMPLES)
    span = lambda c: r.controls[c][0] + u * (r.controls[c][1] - r.controls[c][0])  # noqa: E731
    if "tongue_place" in r.controls and to.type != "pneumatic_tongue":
        lo, hi = r.controls["tongue_place"]
        room = r.length - (SLED_BACK + GLOTTIS_FRONT.get(parts["tract"].type, 0.0)) / 10
        if hi > room + 1e-9:
            out.append(f"tongue rail: it carries the pad to {hi:.2f} cm behind the teeth, and its sled, "
                       f"{SLED_BACK / 10:g} cm behind the pad, would run into the glottis's hardware (the glottis is at "
                       f"{r.length:.2f} cm): the pad may go back to {room:.2f} cm")
        ck, xs = rail_crank(to.params["rail_start"], to.type), np.array([sled_pin(p) for p in span("tongue_place")])
        out += _crank_faults("tongue rail", ck, xs, "the sled")
        out += _crank_clear("tongue rail", ck, xs, stand(r, parts["tract"].type))
        stroke = to.params["stroke"]
        mus = [cam_mu(stroke, cam_angle(stroke, 10 * h)) for h in span("tongue_height")]
        if min(mus) < MU_MIN:
            out.append(f"tongue cam: the follower runs at a {math.degrees(min(mus)):.0f} degree transmission angle")
    if "tip_place" in r.controls and to.type == "tip_tongue":
        grid = [(p, h) for p in span("tip_place") for h in span("tip_height")]
        pins = [tip_pin(p, h) for p, h in grid]
        where = [f"tip_place {p:.2f}, tip_height {h:.2f}" for p, h in grid]
        for k, d in enumerate(TIP_DYADS):
            out += _dyad_faults(f"tip servo {k + 1}", d, pins, None, where)
        if not any("cannot reach" in f for f in out):
            gs = []
            for pin in pins:
                rods = [pin - d.tip(pin) for d in TIP_DYADS]
                g = _angle(*rods)
                gs.append(min(g, math.pi - g))
            i = int(np.argmin(gs))
            if gs[i] < MU_MIN:
                out.append(f"tip five-bar: its two rods meet the crosshead at {math.degrees(gs[i]):.0f} degrees at "
                           f"{where[i]}, below {math.degrees(MU_MIN):.0f}: the blade loses its stiffness")
    if li.type in ("jaw_lips", "spread_lips"):
        span_a, tube = P.JAW[li.type]
        d, pin0 = jaw_dyad(10 * tube, li.params["rest_area"], span_a)
        js = span("jaw")
        pins = [pin0 - np.array([0.0, jaw_drop(li.params["rest_area"], span_a, j)]) for j in js]
        out += _dyad_faults("jaw link", d, pins, np.array([1.0, 0.0]), [f"jaw {j:.2f}" for j in js])
    if li.type == "round_lips":
        grid = [(lr, pr) for lr in span("lip_radius") for pr in span("protrusion")]
        d = protrusion_dyad(li.params["max_protrusion"], P.RING / 2)
        out += _dyad_faults("protrusion pushrod", d, [ring_clip(lr, pr) for lr, pr in grid], np.array([0.0, 1.0]),
                            [f"lip_radius {lr:.2f}, protrusion {pr:.2f}" for lr, pr in grid])
        need = draw_needed(*r.controls["lip_radius"])
        if need > 2 * SPOOL_R * SPAN:
            out.append(f"drawstring: closing the ring takes up {need:.0f} mm of cord and the {SPOOL_R:g} mm spool "
                       f"winds {2 * SPOOL_R * SPAN:.0f} mm over the servo's travel")
    if "larynx" in r.controls:
        need = 10 * r.controls["larynx"][1]
        if need > PINION_R * SPAN:
            out.append(f"larynx slide: lowering the larynx {need:.0f} mm and the {PINION_R:g} mm pinion moves the "
                       f"rack {PINION_R * SPAN:.0f} mm over the servo's travel")
    return out


# ---------------------------------------------------------------- how far each servo turns (D-072)
SERVO_SPEED = math.radians(60.0) / 0.1  # an MG90S at 4.8 V: 60 degrees in 0.1 s


def _near(a: list[float], ref: float) -> np.ndarray:
    """Angles on the branch within half a turn of ``ref``: every drive turns less than that over its range."""
    return ref + (np.asarray(a) - ref + math.pi) % (2 * math.pi) - math.pi


def servo_angles(r, parts: dict, ctl: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Each servo's angle (rad) at a batch of postures, each on the branch around its angle mid-range: the names, and
    (B, S).  A pneumatic tongue has no servos; its valves are not timed."""
    from sea.worlds import phonology as P

    to, li = parts["tongue"], parts["lips"]
    names, cols = [], []
    if "larynx" in ctl:
        names.append("larynx servo")
        cols.append(np.array([pinion_angle(10 * v) for v in ctl["larynx"]]))
    if to.type != "pneumatic_tongue":
        ck, stroke = rail_crank(to.params["rail_start"], to.type), to.params["stroke"]
        names += ["rail servo", "cam servo"]
        cols.append(np.array([ck.angle(sled_pin(p)) for p in ctl["tongue_place"]]))
        cols.append(np.array([cam_angle(stroke, 10 * min(h, stroke)) for h in ctl["tongue_height"]]))
    if to.type == "tip_tongue":
        pins = [tip_pin(p, h) for p, h in zip(ctl["tip_place"], ctl["tip_height"], strict=True)]
        mid = tip_pin(sum(r.controls["tip_place"]) / 2, sum(r.controls["tip_height"]) / 2)
        for k, d in enumerate(TIP_DYADS):
            names.append(f"tip servo {k + 1}")
            cols.append(_near([d.horn(p) for p in pins], d.horn(mid)))
    if "jaw" in ctl:
        span, tube = P.JAW[li.type]
        d, pin0 = jaw_dyad(10 * tube, li.params["rest_area"], span)
        names.append("jaw servo")
        pin = lambda j: pin0 - np.array([0.0, jaw_drop(li.params["rest_area"], span, j)])  # noqa: E731
        cols.append(_near([d.horn(pin(j)) for j in ctl["jaw"]], d.horn(pin(0.5))))
    if li.type == "round_lips":
        d = protrusion_dyad(li.params["max_protrusion"], P.RING / 2)
        lo, hi = r.controls["lip_radius"]
        names += ["protrusion servo", "drawstring servo"]
        ref = d.horn(ring_clip((lo + hi) / 2, sum(r.controls["protrusion"]) / 2))
        cols.append(_near([d.horn(ring_clip(a, p)) for a, p in zip(ctl["lip_radius"], ctl["protrusion"], strict=True)],
                          ref))
        cols.append(np.array([spool_angle(a, lo, hi) for a in ctl["lip_radius"]]))
    n = len(next(iter(ctl.values())))
    return names, (np.stack(cols, 1) if cols else np.zeros((n, 0)))
