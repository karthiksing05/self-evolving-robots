"""The phonology world: a mechanical talking robot learns a vowel and stop inventory.

The robot follows the Waseda Talker and Sawada's talking robot: a cast-silicone tube from the glottis
(a reed driven by an air pump) to the lips, squeezed from outside by servo-driven articulators.  The
body is three parts: the cast ``tract``, the ``tongue`` mechanism and the ``lips``.  System 1 searches
the servo positions for each sound; System 2 changes the parts.

Acoustics.  The tract is a chain of cylindrical sections (an area function).  Each section is a lossy
two-port (chain / ABCD matrix, wall and viscous loss ``alpha ~ sqrt(f)``); the glottis end is a
volume-velocity source, the lip end radiates into Flanagan's piston-in-baffle load (a resistance in
parallel with an inertance).  Formants F1-F3 are the first three peaks of ``|U_lips / U_glottis|`` on a
10 Hz grid from 40 to 5000 Hz, refined by parabolic interpolation.  A uniform 17.5 cm tube of 2 cm2
gives 480/1444/2410 Hz against the ideal quarter-wave 500/1500/2500: the radiation end correction.

Articulation (after Stevens & House 1955).  The cast has a fixed bore (radius 1.2 cm) and articulators
only narrow it.  The tongue body is a pad pressed in by ``tongue_height`` at ``tongue_place`` (cm behind
the teeth) along its rail; the radius around it follows ``r = r0 + k (R - r0) (d - place)^2``.  A tongue
tip is a narrower pad on its own linkage.  The lips are a short tube in front of the teeth: the stock
lips only open with the jaw from a cast rest gap; a rounding ring sets the opening (down to pressed shut)
and a protrusion that lengthens the lip tube.  A motorized larynx lengthens the pharynx.  Each control's
range comes from its part (rail travel, cam stroke, linkage), and a control exists for System 1 only if
a part provides it.  Each control is driven by a servo through a real linkage (``worlds/linkages.py``): a body whose
linkage cannot reach, turn, push or draw through its ranges is refused before any trial (D-071).  Every attempt adds servo noise (1% of each range), so confirmation on fresh seeds
means robustness.  System 1 keeps a map of 600 sampled postures per body (babbling), runs ``local_search``
from the nearest three distinct ones, and the same map grounds the brief and the diagnosis.

Tasks.  Ten American English vowels at the adult-male averages of Peterson & Barney (1952), table II:
success if F1 and F2 (and F3 for /er/, whose low F3 is its cue) are each within 10% of the target, about
the spread of one talker group's tokens and inside the distance between neighbouring vowels, with the
narrowest point at least 0.15 cm2 (narrower, the flow turns turbulent: a fricative, not a vowel).  Three
stops: the narrowest point (below 0.05 cm2) at the place (the lips; the alveolar ridge, 0-1.5 cm behind
the teeth; the velum, 5.5-8.5 cm), with the airway beyond the closure open (above 0.2 cm2) so pressure
builds and the release radiates.  Test tasks repeat each sound under other noise with targets moved 3%.

The stock body holds the front, central and open vowels.  What it cannot do is structural and visible in
numbers: unrounded lips and a single pad keep F2 high at low F1, a low F3 needs constrictions at several
places at once, and no articulator reaches or seals the lips or the alveolar ridge.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sea.evolve.core import Body, BomLine, Outcome, Param, Part, PartType, Task
from sea.evolve.search import local_search
from sea.worlds import linkages

# ---------------------------------------------------------------- acoustics
C = 35000.0  # speed of sound, cm/s
RHO = 1.14e-3  # air density, g/cm3
FREQ = np.arange(40.0, 5000.0, 10.0)
_W = 2 * np.pi * FREQ
_GAMMA = 0.0055 * np.sqrt(FREQ / 500.0) + 1j * _W / C  # propagation constant per cm, lightly lossy


def transfer_db(areas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``20 log10 |U_lips / U_glottis|`` for a batch of area functions, glottis first: (B, N) -> (B, F)."""
    areas, lengths = np.atleast_2d(areas), np.atleast_2d(lengths)
    a = np.ones((areas.shape[0], FREQ.size), complex)
    b, c, d = np.zeros_like(a), np.zeros_like(a), np.ones_like(a)
    ch = sh = a
    for i in range(areas.shape[1]):
        if i == 0 or not np.array_equal(lengths[:, i], lengths[:, i - 1]):
            gl = _GAMMA[None, :] * lengths[:, i:i + 1]
            ch, sh = np.cosh(gl), np.sinh(gl)
        z0 = (RHO * C / areas[:, i])[:, None]
        shz, szz = sh / z0, sh * z0
        a, b, c, d = a * ch + b * shz, a * szz + b * ch, c * ch + d * shz, c * szz + d * ch
    al = areas[:, -1:]
    rr = 128 * RHO * C / (9 * np.pi ** 2 * al)  # radiation resistance
    lr = 8 * RHO * np.sqrt(al / np.pi) / (3 * np.pi * al)  # radiation inertance
    zr = 1j * _W * lr * rr / (rr + 1j * _W * lr)
    return -20 * np.log10(np.abs(c * zr + d))  # P_lips = Zr U_lips, U_glottis = (C Zr + D) U_lips


def formants(areas: np.ndarray, lengths: np.ndarray, n: int = 3) -> np.ndarray:
    """The first ``n`` peaks of the transfer function, Hz, (B, n); nan where fewer peaks exist."""
    mag = transfer_db(areas, lengths)
    peak = (mag[:, 1:-1] > mag[:, :-2]) & (mag[:, 1:-1] >= mag[:, 2:])
    out = np.full((mag.shape[0], n), np.nan)
    rows, cols = np.nonzero(peak)
    cols = cols + 1
    rank = np.arange(rows.size) - np.searchsorted(rows, rows)  # index of each peak within its row
    keep = rank < n
    rows, cols, rank = rows[keep], cols[keep], rank[keep]
    y0, y1, y2 = mag[rows, cols - 1], mag[rows, cols], mag[rows, cols + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    out[rows, rank] = FREQ[cols] + off * (FREQ[1] - FREQ[0])
    return out


# ---------------------------------------------------------------- the robot's mechanics
NT = 24  # sections of the cast tube
BORE = 1.2  # cast bore radius, cm (4.5 cm2)
K_BODY, K_TIP = 0.05, 0.4  # parabola sharpness of the body pad and of the tip, 1/cm2
A_CLOSE, A_OPEN = 0.05, 0.2  # cm2: a closure, and an open airway
A_VOWEL = 0.15  # cm2: a vowel's narrowest point; below it the flow turns turbulent (a fricative, not a vowel)
NOISE = 0.01  # servo position noise, fraction of each control's range
CHANNELS = 6  # servo channels on the controller
PLACES = {"lips": (-9.0, -0.01), "alveolar": (0.0, 1.5), "velar": (5.5, 8.5)}  # cm behind the teeth
RING = 1.45  # radius of the rounding lip ring fully open, cm
JAW = {"jaw_lips": (3.9, 1.0), "spread_lips": (5.4, 0.6)}  # opening added by the jaw (cm2), lip tube (cm)


@dataclass
class Rig:
    """A body's mechanics, read from its parts: what System 1 can move and how far."""

    length: float  # cast tube, glottis to teeth, cm
    controls: dict[str, tuple[float, float]] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)  # non-servo settings (a jaw-lips rest gap...)
    lips: str = "jaw_lips"
    servos: int = 0
    opposed: bool = False  # the tip blade presses from the palate, the body pad from the floor (a tip tongue)

    def lip_range(self) -> tuple[float, float, float, float]:
        """Smallest and largest lip opening (cm2), shortest and longest lip tube (cm)."""
        if self.lips == "round_lips":
            return 0.0, float(np.pi * RING ** 2), 1.0, 1.0 + self.controls["protrusion"][1]
        span, tube = JAW[self.lips]
        return self.fixed["lip_rest"], self.fixed["lip_rest"] + span, tube, tube


def rig(body: Body) -> Rig:
    tr, to, li = body.parts["tract"], body.parts["tongue"], body.parts["lips"]
    r = Rig(length=tr.params.get("length", 13.5))
    if tr.type == "larynx_tube":
        r.controls["larynx"] = (0.0, tr.params["travel"])
        r.servos += 1
    if to.type == "pneumatic_tongue":
        r.controls |= {"tongue_place": (1.0, 11.0), "tongue_height": (0.0, BORE), "tip_place": (0.3, 6.0),
                       "tip_height": (0.0, BORE)}
    else:
        travel = 10.0 if to.type == "long_rail_tongue" else 7.0
        start = to.params["rail_start"]
        r.controls |= {"tongue_place": (start, start + travel), "tongue_height": (0.0, to.params["stroke"])}
        r.servos += 2
        if to.type == "tip_tongue":
            r.controls |= {"tip_place": (0.3, 5.0), "tip_height": (0.0, BORE + 0.1)}
            r.servos += 2
            r.opposed = True
    r.lips = li.type
    if li.type == "round_lips":
        r.controls |= {"lip_radius": (-0.2, RING), "protrusion": (0.0, li.params["max_protrusion"])}
        r.servos += 2
    else:
        r.fixed["lip_rest"] = li.params["rest_area"]
        r.controls["jaw"] = (0.0, 1.0)
        r.servos += 2 if li.type == "spread_lips" else 1
    return r


def tract(r: Rig, ctl: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area function for a batch of postures: areas and section lengths (B, NT+2), glottis first, and the
    distance of each section behind the teeth (lips negative)."""
    n = next(iter(ctl.values())).shape[0]
    length = r.length + ctl.get("larynx", np.zeros(n))
    d = (1.0 - (np.arange(NT) + 0.5) / NT)[None, :] * length[:, None]
    rad = np.full((n, NT), BORE)
    for place, height, k in (("tongue_place", "tongue_height", K_BODY), ("tip_place", "tip_height", K_TIP)):
        if place in ctl:
            r0 = np.maximum(0.0, BORE - ctl[height])[:, None]
            rad = np.minimum(rad, r0 + k * (BORE - r0) * (d - ctl[place][:, None]) ** 2)
    if r.lips == "round_lips":  # the drawstring closes past contact (radius < 0): pressed shut
        lip_a, lip_l = np.pi * np.maximum(0.0, ctl["lip_radius"]) ** 2, 1.0 + ctl["protrusion"]
    else:
        span, tube = JAW[r.lips]
        lip_a, lip_l = r.fixed["lip_rest"] + span * ctl["jaw"], np.full(n, tube)
    areas = np.concatenate([np.pi * rad ** 2, np.repeat(lip_a[:, None], 2, 1)], 1)
    lens = np.concatenate([np.repeat((length / NT)[:, None], NT, 1), np.repeat((lip_l / 2)[:, None], 2, 1)], 1)
    dist = np.concatenate([d, -np.stack([lip_l / 4, 3 * lip_l / 4], 1)], 1)
    return areas, lens, dist


CLASH = 0.025  # cm: the blade and the pad may press 0.5 mm into each other (twice this: both are drawn as gaps)


def clash(r: Rig, ctl: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """How far (cm) the tip blade and the body pad of a tip tongue press past each other, and where (cm behind the
    teeth).  They press from opposite walls, the blade from the palate and the pad from the floor, so where both narrow
    one section the gaps they leave must sum to the bore; the concentric tract model would let each pass through the
    other.  Zero for every other tongue (the hydrostat's body and tip are one cast)."""
    n = next(iter(ctl.values())).shape[0]
    if not r.opposed:
        return np.zeros(n), np.zeros(n)
    length = r.length + ctl.get("larynx", np.zeros(n))
    d = (1.0 - (np.arange(NT) + 0.5) / NT)[None, :] * length[:, None]
    rads = []
    for place, height, k in (("tongue_place", "tongue_height", K_BODY), ("tip_place", "tip_height", K_TIP)):
        r0 = np.maximum(0.0, BORE - ctl[height])[:, None]
        rads.append(np.minimum(BORE, r0 + k * (BORE - r0) * (d - ctl[place][:, None]) ** 2))
    over = BORE - rads[0] - rads[1]
    i = np.argmax(over, 1)
    rows = np.arange(n)
    return np.maximum(over[rows, i], 0.0), d[rows, i]


def acoustics(r: Rig, ctl: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Formants (B, 3), and the narrowest point (B, 3): its area, where it is, and the airway beyond it."""
    areas, lens, dist = tract(r, ctl)
    f = formants(np.maximum(areas, 0.08), lens)
    i = np.argmin(areas, 1)
    rows = np.arange(areas.shape[0])
    where = dist[rows, i]
    away = (np.abs(dist - where[:, None]) > 2.5) | ((dist < 0) & (where[:, None] >= 0))  # past the pad's flanks
    open_else = np.where(away, areas, np.inf).min(1)
    return f, np.stack([areas[rows, i], where, open_else], 1)


# ---------------------------------------------------------------- tasks
# Peterson & Barney (1952), table II, adult males: F1, F2, F3 in Hz.
VOWELS = {
    "iy": ("i", "heed", 270, 2290, 3010), "ih": ("ɪ", "hid", 390, 1990, 2550),
    "eh": ("ɛ", "head", 530, 1840, 2480), "ae": ("æ", "had", 660, 1720, 2410),
    "aa": ("ɑ", "hod", 730, 1090, 2440), "ao": ("ɔ", "hawed", 570, 840, 2410),
    "uh": ("ʊ", "hood", 440, 1020, 2240), "uw": ("u", "who'd", 300, 870, 2240),
    "ah": ("ʌ", "hud", 640, 1190, 2390), "er": ("ɝ", "heard", 490, 1350, 1690),
}
STOPS = {"p": ("lips", "bilabial"), "t": ("alveolar", "alveolar"), "k": ("velar", "velar")}
TOL = 0.10


def _vowel_error(f: np.ndarray, target: Sequence[float]) -> np.ndarray:
    """Per-formant error in tolerances (B, k): signed, 1.0 = at the edge of the 10% window."""
    t = np.asarray(target, float)
    e = (f[:, :t.size] / t - 1.0) / TOL
    return np.where(np.isnan(e), 10.0, e)


def _stop_error(clo: np.ndarray, place: str) -> np.ndarray:
    """How far each posture is from a clean stop at ``place``, in units where <= 1 passes."""
    lo, hi = PLACES[place]
    a_min, where, open_else = clo[:, 0], clo[:, 1], clo[:, 2]
    miss = np.maximum(0.0, np.maximum(lo - where, where - hi))  # cm outside the place
    off_place = np.where(miss > 0, 1.0 + miss, 0.0)
    return np.maximum.reduce([a_min / A_CLOSE, off_place, A_OPEN / np.maximum(open_else, 1e-6)])


# ---------------------------------------------------------------- the catalogue and the bench
def _pt(name: str, slot: str, doc: str, provides: str, stock: bool = False, **params: Param) -> PartType:
    return PartType(name=name, slot=slot, doc=doc, provides=provides, stock=stock, params=params)


CATALOGUE = [
    _pt("short_cast", "tract", "a small cast-silicone vocal tract (adult-female size), bore 2.4 cm",
        "a fixed tube from the glottis to the teeth; the lips add their own length", stock=True,
        length=Param(lo=12.5, hi=14.5, unit="cm", doc="glottis to teeth, set by the mold; every resonance scales "
                     "with 1/length")),
    _pt("long_cast", "tract", "a large cast-silicone vocal tract (adult-male size), bore 2.4 cm",
        "a longer fixed tube from the glottis to the teeth",
        length=Param(lo=15.5, hi=17.5, unit="cm", doc="glottis to teeth, set by the mold")),
    _pt("larynx_tube", "tract", "the small cast with a silicone bellows at the glottal end; the reed housing rides a "
        "slide whose rack a pinion on a servo drives",
        "a tube whose pharynx lengthens during an utterance as the larynx is lowered",
        length=Param(lo=12.5, hi=14.5, unit="cm", doc="glottis to teeth with the larynx raised"),
        travel=Param(lo=1.0, hi=3.0, unit="cm", doc="how far the slide lowers the larynx")),
    _pt("cam_tongue", "tongue", "a silicone pad pressed into the tube by a cam, on a carriage riding a 7 cm rail",
        "one constriction: place along the rail and degree by the cam (2 servos)", stock=True,
        rail_start=Param(lo=2.5, hi=4.0, unit="cm", doc="front end of the rail, behind the teeth; the pad reaches "
                         "7 cm back from it"),
        stroke=Param(lo=0.6, hi=1.3, unit="cm", doc="cam lift: how far the pad presses into the 1.2 cm bore radius "
                     "(past 1.2 it squeezes the tube shut)")),
    _pt("long_rail_tongue", "tongue", "the cam pad on a carriage riding a 10 cm rail",
        "one constriction over a longer stretch of the tube (2 servos)",
        rail_start=Param(lo=2.5, hi=4.0, unit="cm", doc="front end of the rail; the pad reaches 10 cm back from it, "
                         "and the rail must end 0.5 cm in front of the glottis"),
        stroke=Param(lo=0.6, hi=1.3, unit="cm", doc="cam lift into the 1.2 cm bore radius")),
    _pt("tip_tongue", "tongue", "the cam pad on its 7 cm rail plus a narrow tip blade on a two-servo linkage under "
        "the palate", "two constrictions at once: the body pad, and a short tip that slides 0.3-5 cm behind the "
        "teeth and can seal the full bore (4 servos)",
        rail_start=Param(lo=2.5, hi=4.0, unit="cm", doc="front end of the body pad's rail"),
        stroke=Param(lo=0.6, hi=1.3, unit="cm", doc="cam lift of the body pad into the 1.2 cm bore radius")),
    _pt("pneumatic_tongue", "tongue", "a three-chamber silicone hydrostat tongue inflated through a valve manifold",
        "a body and a tip shape over 1-11 cm behind the teeth, each able to seal the bore (no servos)"),
    _pt("jaw_lips", "lips", "cast lips on the jaw; one servo opens the jaw", "a lip opening from the rest gap to "
        "rest + 3.9 cm2, lip tube 1 cm (1 servo)", stock=True,
        rest_area=Param(lo=2.5, hi=3.5, unit="cm2", doc="lip opening with the jaw shut, set by the cast lip shape")),
    _pt("spread_lips", "lips", "jaw lips with a second servo pulling the mouth corners apart",
        "a lip opening from the rest gap to rest + 5.4 cm2, lip tube 0.6 cm (2 servos)",
        rest_area=Param(lo=2.5, hi=3.5, unit="cm2", doc="lip opening with the jaw shut")),
    _pt("round_lips", "lips", "a silicone lip ring closed by a drawstring wound on a servo's spool and pushed forward "
        "by a second servo's pushrod",
        "a lip opening of 0-6.6 cm2 and a lip tube 1 cm long plus the protrusion (2 servos)",
        max_protrusion=Param(lo=0.6, hi=1.6, unit="cm", doc="how far the ring can be pushed forward of the teeth")),
]

PARTS = {p.name: p for p in CATALOGUE}

MANIFEST: dict[str, Any] = {
    "budget": {"usd": 40, "lead_days": 7, "fab_minutes": 600},
    "fabrication": {
        "print:PLA": {"minutes_per_unit": 1.2, "unit": "g", "doc": "FDM printer: molds, mounts, carriages"},
        "cast:silicone": {"minutes_per_unit": 2.5, "unit": "g", "doc": "platinum silicone: mix, degas, cure"},
    },
    "items": {
        "servo_mg90s": {"on_hand": 1, "usd": 8, "lead_days": 3, "doc": "metal-gear micro servo, one controller "
                        "channel each"},
        "rail_100mm": {"on_hand": 0, "usd": 9, "lead_days": 4, "doc": "miniature linear rail and carriage"},
        "rail_150mm": {"on_hand": 0, "usd": 12, "lead_days": 4, "doc": "miniature linear rail and carriage"},
        "pushrod_kit": {"on_hand": 0, "usd": 4, "lead_days": 2, "doc": "steel pushrods and ball links"},
        "slide_40mm": {"on_hand": 0, "usd": 15, "lead_days": 5, "doc": "40 mm linear slide with a rack and pinion"},
        "valve_manifold_3ch": {"on_hand": 0, "doc": "proportional pneumatic valves; not stocked, no supplier"},
    },
}


def _bom_part(slot: str, p: Part) -> list[BomLine]:
    def ln(item: str, qty: float) -> BomLine:
        return BomLine(item=item, qty=round(qty, 1), slot=slot)

    length = p.params.get("length", 0.0)
    table: dict[str, list[BomLine]] = {
        "short_cast": [ln("cast:silicone", 3.0 * length), ln("print:PLA", 4.0 * length)],
        "long_cast": [ln("cast:silicone", 3.0 * length), ln("print:PLA", 4.0 * length)],
        "larynx_tube": [ln("cast:silicone", 3.0 * length + 12), ln("print:PLA", 4.0 * length + 20),
                        ln("servo_mg90s", 1), ln("slide_40mm", 1)],
        "cam_tongue": [ln("servo_mg90s", 2), ln("rail_100mm", 1), ln("print:PLA", 25), ln("cast:silicone", 8)],
        "long_rail_tongue": [ln("servo_mg90s", 2), ln("rail_150mm", 1), ln("print:PLA", 32), ln("cast:silicone", 8)],
        "tip_tongue": [ln("servo_mg90s", 4), ln("rail_100mm", 1), ln("pushrod_kit", 1), ln("print:PLA", 38),
                       ln("cast:silicone", 12)],
        "pneumatic_tongue": [ln("valve_manifold_3ch", 1), ln("cast:silicone", 30), ln("print:PLA", 20)],
        "jaw_lips": [ln("servo_mg90s", 1), ln("print:PLA", 15), ln("cast:silicone", 10)],
        "spread_lips": [ln("servo_mg90s", 2), ln("print:PLA", 18), ln("cast:silicone", 10)],
        "round_lips": [ln("servo_mg90s", 2), ln("pushrod_kit", 1), ln("print:PLA", 18), ln("cast:silicone", 14)],
    }
    return table[p.type]


# ---------------------------------------------------------------- the world
@dataclass
class Babble:
    """System 1's map of a body: postures sampled over every control, with what each sounds like."""

    keys: list[str]
    x: np.ndarray  # (n, k) postures
    f: np.ndarray  # (n, 3) formants
    clo: np.ndarray  # (n, 3) closure: smallest area, its place, the airway elsewhere
    clash: np.ndarray  # (n,) how far the tip blade and the body pad press past each other, cm


def _fmt(f: Sequence[float] | np.ndarray) -> str:
    return "/".join(f"{v:.0f}" for v in f)


def _span(lo: float, hi: float) -> str:
    return f"{lo:.3g}" if abs(hi - lo) < 1e-6 else f"{lo:.3g}-{hi:.3g}"


class PhonologyWorld:
    name = "phonology"
    description = ("A mechanical talking robot (after the Waseda Talker): a cast-silicone vocal tract excited by a "
                   "reed at the glottis and squeezed by servo-driven tongue and lip mechanisms.  It must produce "
                   "ten English vowels at adult-male formant targets and three stops (/p/, /t/, /k/).  The body is "
                   "the cast tract, the tongue mechanism and the lips.")
    n_babble = 600
    starts = 3

    def __init__(self) -> None:
        self._babble: dict[str, Babble] = {}

    # -------------------------------------------------- tasks and parts
    def tasks(self) -> list[Task]:
        out = []
        for g, (ipa, word, *fs) in VOWELS.items():
            k = 3 if g == "er" else 2
            for split, i, scale in (("train", 0, 1.0), ("test", 1, 1.03), ("test", 2, 0.97)):
                tgt = [round(v * scale) for v in fs[:k]]
                names = "/".join(f"F{j + 1}" for j in range(k))
                out.append(Task(id=f"{g}_{split}{i}", split=split, group=g,  # type: ignore[arg-type]
                                facts=f"vowel /{ipa}/ as in '{word}': {names} {_fmt(tgt)} Hz, each within 10%",
                                data={"kind": "vowel", "target": tgt, "noise": 7 + 101 * i}))
        for g, (place, name) in STOPS.items():
            lo, hi = PLACES[place]
            where = "at the lips" if place == "lips" else f"{lo:g}-{hi:g} cm behind the teeth"
            for split, i in (("train", 0), ("test", 1), ("test", 2)):
                out.append(Task(id=f"{g}_{split}{i}", split=split, group=g,  # type: ignore[arg-type]
                                facts=f"{name} stop /{g}/: seal the tube {where} (area < {A_CLOSE:g} cm2) with the "
                                      f"airway open elsewhere (> {A_OPEN:g} cm2)",
                                data={"kind": "stop", "place": place, "noise": 7 + 101 * i}))
        return out

    def catalogue(self) -> list[PartType]:
        return list(CATALOGUE)

    def stock(self) -> Body:
        return Body(parts={"tract": Part(type="short_cast", params={"length": 13.5}),
                           "tongue": Part(type="cam_tongue", params={"rail_start": 3.25, "stroke": 0.95}),
                           "lips": Part(type="jaw_lips", params={"rest_area": 3.0})})

    def bom(self, body: Body) -> list[BomLine]:
        return [ln for slot, p in sorted(body.parts.items()) for ln in _bom_part(slot, p)]

    def check(self, body: Body) -> list[str]:
        errs = [f"slot {s!r}: {p.type!r} is not a {s} part" for s, p in body.parts.items()
                if p.type not in PARTS or PARTS[p.type].slot != s]
        missing = sorted({"tract", "tongue", "lips"} - set(body.parts))
        errs += [f"slot {s!r} is empty" for s in missing]
        if errs:
            return errs
        r = rig(body)
        if r.servos > CHANNELS:
            errs.append(f"the body needs {r.servos} servos and the servo controller drives {CHANNELS} channels")
        return errs + linkages.faults(r, body.parts)

    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(MANIFEST)

    # -------------------------------------------------- System 1
    def babble(self, body: Body) -> Babble:
        key = body.key()
        if key not in self._babble:
            r = rig(body)
            keys = sorted(r.controls)
            rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
            u = rng.beta(0.5, 0.5, (self.n_babble, len(keys)))  # dense near each servo's end stops
            lo = np.array([r.controls[k][0] for k in keys])
            hi = np.array([r.controls[k][1] for k in keys])
            x = lo + u * (hi - lo)
            ctl = {k: x[:, j] for j, k in enumerate(keys)}
            f, clo = acoustics(r, ctl)
            self._babble[key] = Babble(keys, x, f, clo, clash(r, ctl)[0])
        return self._babble[key]

    @staticmethod
    def _errors(task: Task, f: np.ndarray, clo: np.ndarray, over: np.ndarray) -> np.ndarray:
        if task.data["kind"] == "vowel":
            formant = np.abs(_vowel_error(f, task.data["target"])).max(1)
            e = np.maximum(formant, A_VOWEL / np.maximum(clo[:, 0], 1e-6))  # a vowel keeps its airway open
        else:
            e = _stop_error(clo, task.data["place"])
        return np.where(over > CLASH, np.maximum(e, 1.0 + over / CLASH), e)  # a posture the parts cannot take

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        """System 1: ``local_search`` from the nearest postures in the body's map, up to ``starts`` of them."""
        r, bab = rig(body), self.babble(body)
        span = np.array([hi - lo for lo, hi in (r.controls[k] for k in bab.keys)])
        picks: list[int] = []
        # nearest first, each start 15% of the range apart
        for i in np.argsort(self._errors(task, bab.f, bab.clo, bab.clash)):
            if all(np.abs((bab.x[i] - bab.x[j]) / span).max() > 0.15 for j in picks):
                picks.append(int(i))
            if len(picks) == self.starts:
                break

        def attempt(ctl: dict[str, float], s: int) -> Outcome:
            t0 = time.perf_counter()
            rng = np.random.default_rng([task.data["noise"], s])
            x = {k: np.array([np.clip(ctl[k] + rng.normal(0.0, NOISE * (hi - lo)), lo, hi)])
                 for k, (lo, hi) in r.controls.items()}
            f, clo = acoustics(r, x)
            over, at = clash(r, x)
            e = float(self._errors(task, f, clo, over)[0])
            said = self._said(task, f[0], clo[0])
            if over[0] > CLASH:
                said += (f"; the tip blade presses {20 * over[0]:.1f} mm into the tongue pad {at[0]:.1f} cm behind the "
                         "teeth: no posture the parts can take")
            return Outcome(task=task.id, success=e <= 1.0, score=1.0 / (1.0 + e), detail=said,
                           sim_s=time.perf_counter() - t0)

        out, attempts, sim_s = None, 0, 0.0
        for j, i in enumerate(picks):
            start = dict(zip(bab.keys, bab.x[i].tolist(), strict=True))
            o = local_search(attempt, r.controls, start, m=32, confirm=2, seed=seed + 7919 * j, step=0.1, shrink=0.9)
            attempts, sim_s = attempts + o.attempts, sim_s + o.sim_s
            if out is None or o.success or o.score > out.score:
                out = o
            if o.success:
                break
        assert out is not None
        out = out.model_copy(update={"attempts": attempts, "sim_s": round(sim_s, 3)})
        if out.success:
            return out
        mech, why = self._diagnose(body, task, out)
        return out.model_copy(update={"mechanism": mech, "detail": f"{out.detail}; {why}"})

    @staticmethod
    def _said(task: Task, f: np.ndarray, clo: np.ndarray) -> str:
        if task.data["kind"] == "vowel":
            tgt = task.data["target"]
            got = f[:len(tgt)]
            off = ", ".join(f"F{j + 1} {100 * (g / t - 1):+.0f}%" for j, (g, t) in enumerate(zip(got, tgt, strict=True))
                            if not np.isnan(g) and abs(g / t - 1) > TOL)
            shut = f"; narrowest {clo[0]:.2f} cm2, a vowel needs {A_VOWEL:g}" if clo[0] < A_VOWEL else ""
            return f"F {_fmt(np.nan_to_num(got))} Hz vs target {_fmt(tgt)}" + (f" ({off})" if off else "") + shut
        where = "the lips" if clo[1] < 0 else f"{clo[1]:.1f} cm behind the teeth"
        return f"narrowest {clo[0]:.2f} cm2 at {where}, airway elsewhere {min(clo[2], 99):.2f} cm2"

    # -------------------------------------------------- diagnosis
    def _diagnose(self, body: Body, task: Task, out: Outcome) -> tuple[str, str]:
        r, bab = rig(body), self.babble(body)
        best = {k: np.array([out.controls[k]]) for k in r.controls}
        fb, cb = acoustics(r, best)
        keep = np.concatenate([bab.clash, clash(r, best)[0]]) <= CLASH  # only postures the parts can take
        f, clo = np.vstack([bab.f, fb])[keep], np.vstack([bab.clo, cb])[keep]
        noisy = "passed once" in out.detail
        if task.data["kind"] == "stop":
            return self._diagnose_stop(r, task, clo, noisy)
        tgt = task.data["target"]
        vowel = clo[:, 0] >= A_VOWEL  # postures that keep the airway open enough to be a vowel
        if not vowel.any():
            return "airway_too_narrow", f"no posture keeps the narrowest point at {A_VOWEL:g} cm2 or more"
        f = f[vowel]
        e = _vowel_error(f, tgt)
        tot = np.abs(e).max(1)
        b = int(np.argmin(tot))
        if tot[b] <= 1.0:  # in the window noise-free: at its edge, servo noise is the cause; well inside, the search
            return ("not_robust_to_motor_noise" if noisy or tot[b] > 0.6 else "not_found_by_search",
                    f"the closest open posture gives F {_fmt(f[b, :len(tgt)])} Hz noise-free ({tot[b]:.0%} of the "
                    f"10% window used); System 1 found none that holds under {NOISE:.0%} servo noise")
        k = int(np.argmax(np.abs(e[b])))
        name = f"F{k + 1}"
        mech = f"f{k + 1}_floor_too_high" if e[b, k] > 0 else f"f{k + 1}_ceiling_too_low"
        fk = f[:, k][~np.isnan(f[:, k])]
        others = [j for j in range(len(tgt)) if j != k]
        ok = np.all(np.abs(e[:, others]) <= 1.0, 1) & ~np.isnan(f[:, k])
        cond = " and ".join(f"F{j + 1} within 10% of {tgt[j]}" for j in others)
        if ok.any():
            span = f"with {cond} Hz it spans {f[ok, k].min():.0f}-{f[ok, k].max():.0f} Hz"
        else:
            span = f"no sampled posture has {cond} Hz"
        return mech, (f"over {len(f)} open postures (System 1's map and its best) this body's {name} spans "
                      f"{fk.min():.0f}-{fk.max():.0f} Hz; {span}; target {name} {tgt[k]} Hz")

    def _closers(self, r: Rig) -> list[tuple[str, float, float, float]]:
        """Each articulator that can narrow the tube: name, place range (cm behind the teeth), smallest area."""
        out = []
        for name, place, height in (("tongue body", "tongue_place", "tongue_height"), ("tongue tip", "tip_place",
                                                                                       "tip_height")):
            if place in r.controls:
                lo, hi = r.controls[place]
                out.append((name, lo, hi, float(np.pi * max(0.0, BORE - r.controls[height][1]) ** 2)))
        out.append(("lips", -1.0, -0.01, r.lip_range()[0]))
        return out

    def _diagnose_stop(self, r: Rig, task: Task, clo: np.ndarray, noisy: bool) -> tuple[str, str]:
        place = task.data["place"]
        lo, hi = PLACES[place]
        closers = self._closers(r)
        there = [c for c in closers if c[1] <= hi and c[2] >= lo]
        facts = "; ".join(("lips: smallest opening" if c[0] == "lips" else
                           f"{c[0]}: {_span(c[1], c[2])} cm behind the teeth, smallest area") + f" {c[3]:.2f} cm2"
                          for c in closers)
        window = "the lips" if place == "lips" else f"the {place} place ({lo:g}-{hi:g} cm behind the teeth)"
        if not there:
            return f"no_closure_at_{place}", f"no articulator reaches {window}; {facts}"
        least = min(c[3] for c in there)
        if least > A_CLOSE:
            who, one = " and ".join(c[0] for c in there), len(there) == 1
            how = (f"the lips close to {least:.2f} cm2 at the least" if place == "lips" else
                   f"{who} reach{'es' if one else ''} {window} but narrow{'s' if one else ''} it to {least:.2f} cm2 "
                   "at the least")
            return f"no_closure_at_{place}", f"{how} (a seal is < {A_CLOSE:g} cm2); {facts}"
        e = _stop_error(clo, place)
        b = int(np.argmin(e))
        if e[b] <= 1.0:
            return ("not_robust_to_motor_noise" if noisy else "not_found_by_search",
                    f"a clean closure exists noise-free; System 1 found none that holds under {NOISE:.0%} servo noise")
        if clo[b, 0] <= A_CLOSE and lo <= clo[b, 1] <= hi:
            return "airway_blocked", (f"the closure at {window} forms, but the airway elsewhere narrows to "
                                      f"{clo[b, 2]:.2f} cm2 (needs > {A_OPEN:g})")
        return "not_found_by_search", f"best narrowest {clo[b, 0]:.2f} cm2; {facts}"

    # -------------------------------------------------- what System 2 is told
    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        r, bab = rig(body), self.babble(body)
        lar = r.controls.get("larynx", (0.0, 0.0))[1]
        a_lo, a_hi, l_lo, l_hi = r.lip_range()
        lips = f"opening {a_lo:.2f}-{a_hi:.2f} cm2, lip tube " + (f"{l_lo:g}-{l_hi:g} cm (protrusion)" if l_hi > l_lo
                                                                  else f"{l_lo:g} cm (no protrusion)")
        lines = [f"- tract: cast tube {r.length:g} cm glottis to teeth" + (f" plus 0-{lar:g} cm as the larynx lowers"
                                                                           if lar else "") +
                 f"; {_span(r.length + l_lo, r.length + lar + l_hi)} cm glottis to lip opening; bore "
                 f"{BORE:g} cm radius ({np.pi * BORE ** 2:.1f} cm2), and the articulators only narrow it"]
        for name, lo, hi, amin in self._closers(r)[:-1]:
            lines.append(f"- {name}: {_span(lo, hi)} cm behind the teeth ({_span(r.length - hi, r.length - lo)} cm "
                         f"above the glottis{' with the larynx up' if lar else ''}), smallest area {amin:.2f} cm2")
        lines.append(f"- lips: {lips}")
        lines.append(f"- servos: {r.servos} of {CHANNELS} controller channels; controls: "
                     + ", ".join(f"{k} {lo:.2f}-{hi:.2f}" for k, (lo, hi) in sorted(r.controls.items())))
        f = bab.f[~np.isnan(bab.f).any(1) & (bab.clo[:, 0] >= A_VOWEL)]
        lines.append(f"- sampled vowel space ({len(f)} of {len(bab.f)} sampled postures keep the narrowest point at "
                     f"or above {A_VOWEL:g} cm2): F1 {f[:, 0].min():.0f}-"
                     f"{f[:, 0].max():.0f} Hz, F2 {f[:, 1].min():.0f}-{f[:, 1].max():.0f} Hz, F3 "
                     f"{f[:, 2].min():.0f}-{f[:, 2].max():.0f} Hz")
        bands = []
        for lo, hi in ((0, 350), (350, 450), (450, 550), (550, 650), (650, 900), (900, 5000)):
            m = (f[:, 0] >= lo) & (f[:, 0] < hi)
            if m.any():
                bands.append(f"F1 {lo}-{hi}: F2 {f[m, 1].min():.0f}-{f[m, 1].max():.0f}, lowest F3 {f[m, 2].min():.0f}")
        lines.append("- F2 and F3 reached, by F1 band (Hz): " + "; ".join(bands))
        for place, (lo, hi) in PLACES.items():
            m = (bab.clo[:, 1] >= lo) & (bab.clo[:, 1] <= hi)
            smallest = f"{bab.clo[m, 0].min():.2f} cm2" if m.any() else "never the narrowest point"
            there = [c[0] for c in self._closers(r) if c[1] <= hi and c[2] >= lo]
            seal = [c[0] for c in self._closers(r) if c[1] <= hi and c[2] >= lo and c[3] <= A_CLOSE]
            at = "at the lips" if place == "lips" else f"{place}, {lo:g}-{hi:g} cm behind the teeth"
            lines.append(f"- closure {at}: reached by {', '.join(there) or 'no articulator'}; sealed by "
                         f"{', '.join(seal) or 'none'}; smallest area sampled there {smallest}")
        return "\n".join(lines)

    def principles(self) -> list[str]:
        return [
            "A tube closed at the glottis and open at the lips resonates near (2n-1)c/4L: every formant scales "
            "with 1/L, so a 14.5 cm tract sits about 20% above a 17.5 cm one, and lengthening the tube or the lip "
            "tube lowers all formants together.",
            "Perturbation theory: narrowing the tube where formant n has a volume-velocity maximum lowers it, and "
            "narrowing it where n has a pressure maximum raises it; the lip opening is a velocity maximum of every "
            "formant, so closing and protruding the lips (rounding) lowers F1, F2 and F3.",
            "F1 is the low resonance of the back cavity through the constriction: it falls as the narrowest point "
            "tightens or moves toward the lips, and rises as the constriction moves down into the pharynx (the "
            "glottal half of the tube) and the mouth opens.",
            "F2 rises as the constriction moves forward under the hard palate (a long back cavity and a short front "
            "one) and falls as it moves back to the velum or pharynx, or as the lips round.",
            "F3 has velocity maxima near 1/5 and 3/5 of the way from glottis to lips and at the lips; a very low F3, "
            "as in /er/, takes constrictions near those places at the same time.",
            "A tube that can only be narrowed cannot push a formant past what its bore and length allow: a longer "
            "tube lowers the ceilings of F1 and F2 as well as their floors.",
            "A stop needs an articulator that reaches its place and seals the bore there, with the rest of the airway "
            "open so pressure builds behind the closure and the release radiates; an articulator acts only within its "
            "mechanism's travel and stroke.",
            "Servo noise moves every posture; a tight constriction makes the formants, F1 above all, sensitive to it, "
            "so a sound counts only when it repeats on fresh noise.",
        ]
