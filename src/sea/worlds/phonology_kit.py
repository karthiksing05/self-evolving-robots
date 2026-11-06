"""The talking robot from a kit (D-112): the same sounds and acoustics, and no mechanism anyone has designed.

A body is one part, ``robot``: a graph of levers and pads around a cast tube.  The tube has no solid of its own: it lies
along +x from the teeth at x = 0 to the body named ``glottis``, and the lips run from the teeth to the body named
``lip_end``.  At each section the bore is the distance from the tube's axis to the nearest solid, up to the cast's bore
or the lip ring's, as the concentric area function of ``worlds.phonology`` takes it.

Servos turn hinges; anything else a graph moves, it moves through loops, so every motion comes through a real linkage.
The world holds the brief, the acoustics and what can be bought, and no advice.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np

from sea.evolve.core import Body, BomLine, Instrument, Outcome, Part, PartType, Task
from sea.mech import graph as G
from sea.mech.base import WORLD
from sea.mech.build import compile as compile_graph
from sea.mech.build import part_kg
from sea.mech.laws import floating, interference
from sea.mech.world import SLOT, GraphWorld, graph_of
from sea.sim import tract_sound as S
from sea.sim.soft_tract import SLEW, WALL, SoftTract
from sea.sim.soft_tract import E as WALL_E
from sea.worlds import phonology as P

CM = 0.01
HOLD = 0.85  # of lung pressure, behind a stop's seal: 0.05 cm2 holds about 0.89 (D-121)
GLIDE_S = 0.05  # s: an opening slower than this is heard as a glide (Liberman et al. 1956)
REACHED = 0.15  # of each section's area: a tract within this of the vowel's has reached it
SERVO_NM = 0.216  # an MG90S at 4.8 V: 2.2 kg cm, its datasheet's stall
Y, X = (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)
ALONG_X = (0.7071068, 0.0, 0.7071068, 0.0)  # a round part's axis along x
RULES = [
    f"The robot has {P.CHANNELS} servo channels.  A Servo is an MG90S ({SERVO_NM:g} N m) on a hinge with a range; a sliding "
    "motion comes through a loop.  It turns at most 60 degrees in 0.1 s and pushes with at most that torque.",
    f"The cast's bore is 1.2 cm in radius and the lip ring's 1.45.  Its wall is silicone of E = {WALL_E / 1000:g} kPa and "
    f"{WALL * 1000:g} mm: solids press it, it presses back, and it closes when it meets its far side.  Air is what remains "
    "inside it, and sound is air stepped in time.",
]


def _talker(p: dict[str, float]) -> G.Graph:
    """A tube, a pad on a two-lever arm from below, and a lever that sets the lips' opening."""
    ln = p["length_cm"] * CM
    arm, lift = 12.0 * CM, 0.95 * CM
    shoulder = (p["place_cm"] * CM, 0.0, -14.7 * CM)
    plate = G.Box(name="base", pos=(0.04, 0.0, -0.16), half=(0.11, 0.05, 0.003))
    post = G.Box(name="arm_post", pos=(shoulder[0], 0.0, (shoulder[2] - 0.157) / 2),
                 half=(0.005, 0.005, abs(shoulder[2] + 0.157) / 2 + 0.002))
    lip_post = G.Box(name="lip_post", pos=(-0.005, -0.02, -0.085), half=(0.003, 0.003, 0.072))
    bodies = (
        G.Body(name="frame", parent="world", parts=(plate, post, lip_post)),
        G.Body(name="glottis", parent="frame", pos=(ln, 0.0, 0.0)),
        G.Body(name="lip_end", parent="frame", pos=(-1.0 * CM, 0.0, 0.0)),
        G.Body(name="arm", parent="frame", pos=shoulder,
               joints=(G.Joint(name="arm_q", type="hinge", axis=Y, range=(-0.29, 0.29)),),
               parts=(G.Box(name="arm_bar", pos=(0.0, 0.0, arm / 2), half=(0.003, 0.003, arm / 2)),)),
        G.Body(name="lever", parent="arm", pos=(0.0, 0.0, arm),
               joints=(G.Joint(name="lever_q", type="hinge", axis=Y, range=(0.0, 1.57)),),
               parts=(G.Box(name="lever_bar", pos=(0.0, 0.0, lift / 2), half=(0.002, 0.002, lift / 2)),
                      G.Capsule(name="pad", pos=(0.0, 0.0, lift), radius=0.015, half_length=0.005, quat=ALONG_X))),
        G.Body(name="jaw", parent="frame", pos=(-0.005, -0.02, -0.013),
               joints=(G.Joint(name="jaw_q", type="hinge", axis=X, range=(-0.15, 0.085)),),
               parts=(G.Box(name="jaw_plate", pos=(0.0, 0.02, 0.0), half=(0.005, 0.02, 0.0015)),)))
    excludes = (("frame", "arm"), ("arm", "lever"), ("frame", "jaw"))
    return G.Graph(name="talker", bodies=bodies, excludes=excludes, actuators=tuple(
        G.Servo(name=n, joint=f"{n}_q", stall_nm=SERVO_NM) for n in ("arm", "lever", "jaw")))


def _model(g: G.Graph) -> tuple[mujoco.MjModel, int]:
    """The graph on an empty world, its solids in contype 2, and a probe: a point the bore is measured from."""
    spec = WORLD.model()
    compile_graph(spec, g, WORLD)
    names = {b.name for b in g.bodies}
    for x in spec.geoms:
        if x.parent.name in names and (x.contype or x.conaffinity):
            x.contype, x.conaffinity = 2, 1
    spec.worldbody.add_geom(name="bore_probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-5, 0.0, 0.0], pos=[0.0, 0.0, 1.0],
                            contype=0, conaffinity=0, group=4)
    m = spec.compile()
    return m, m.geom("bore_probe").id


class _Air:
    """A kit robot's air, stepped in time (D-121)."""

    @staticmethod
    def formants(areas: np.ndarray, lens: np.ndarray) -> np.ndarray:
        return S.formants(areas, lens)

    @staticmethod
    def hiss(areas: np.ndarray, lens: np.ndarray) -> np.ndarray:
        return S.hiss(S.Tube(areas, lens))


@lru_cache(maxsize=8)
def _tract(graph_json: str) -> SoftTract:
    return SoftTract(G.Graph.model_validate_json(graph_json))


class PhonologyKit(GraphWorld, P.PhonologyWorld):
    name = "phonology_kit"
    description = ("A mechanical talking robot built from parts rather than a catalogue: a cast-silicone vocal tract excited "
                   "by a reed at the glottis and squeezed by whatever a graph puts around it.  It must produce English vowels, "
                   "stops, fricatives and syllables.")
    frame = ("A robot is a graph (lengths in metres; x along the tube from the teeth at 0 toward the glottis, z up from its "
             "axis).  The tube runs to the body 'glottis', and the lips from the teeth to the body 'lip_end'.  It is soft: it "
             "lies around the axis, and a solid moves its wall as far as a servo's torque can push it.  A posture is a command "
             "to the servos; what is judged is where everything came to rest.")
    modules = {"talker": (_talker, {"length_cm": (10.0, 20.0), "place_cm": (3.0, 10.0)})}
    starts = 3

    def __init__(self) -> None:
        P.PhonologyWorld.__init__(self)
        self._stalled: dict[str, float] = {}

    def catalogue(self) -> list[PartType]:
        return []

    def stock(self) -> Body:
        return Body(parts={SLOT: Part(type="built", mechanism=self.module("talker", {"length_cm": 13.5, "place_cm": 6.75}).model_dump_json())})

    def rig_of(self, body: Body) -> P.Rig:
        g = graph_of(body)
        ranges = {j.name: j.range for b in g.bodies for j in b.joints}
        controls = {a.name: ranges[a.joint] for a in g.actuators if isinstance(a, G.Servo)}
        return P.Rig(length=0.0, controls=controls, servos=len(controls), lips="graph",
                     areas=_tract(body.parts[SLOT].mechanism), sound=_Air)

    def tasks(self) -> list[Task]:
        def felt(t: Task) -> Task:
            kind = t.data["kind"]
            if kind == "stop":
                return t.model_copy(update={"facts": f"{t.facts}, and holding at least {HOLD:.0%} of the lungs' pressure behind it"})
            if kind == "syllable":
                head = t.facts.split(", with no servo")[0]
                return t.model_copy(update={"facts": f"{head}, the tract opening to the vowel's shape within {1000 * GLIDE_S:.0f} ms "
                                                     "of the seal breaking, as driven (slower, it is heard as a glide, not a stop)"})
            return t
        return [felt(t) for t in super().tasks()]

    def babble(self, body: Body) -> P.Babble:
        new = body.key() not in self._babble
        bab = super().babble(body)
        if new:
            st = _tract(body.parts[SLOT].mechanism)
            st(dict(zip(bab.keys, bab.x.T, strict=True)))
            self._stalled[body.key()] = float(np.mean(st.load >= 0.99 * SERVO_NM))
        return bab

    def _felt(self, r: P.Rig, x: dict[str, np.ndarray], d: dict, e: float) -> tuple[float, str]:
        """What the mechanism and the air did, measured: a stall is said, and a seal or a channel is blown through."""
        st = r.areas
        areas, lens, _ = st(x)
        note = ""
        if st.load[0] >= 0.99 * SERVO_NM and st.short[0] > np.radians(1.0):
            note = f"; a servo stalls at {SERVO_NM:g} N m, {np.degrees(st.short[0]):.0f} degrees short of its command"
        if e > 1.0 or d["kind"] == "vowel":  # no need: it has already failed, or nothing is blown through a vowel
            return 0.0, note
        air = S.blow(S.Tube(areas, lens), seconds=0.03)
        if d["kind"] == "stop":
            held = float(air["behind"][0]) / S.PS
            return HOLD / max(held, 1e-6), f"{note}; it holds {100 * held:.0f}% of the lungs' pressure behind it"
        re = float(air["re"][0])
        return S.RE_C / max(re, 1e-6), f"{note}; {air['flow'][0]:.0f} cm3/s through it, Reynolds {re:.0f}"

    def _timed(self, r: P.Rig, body: Body, closure: dict[str, np.ndarray], vowel: dict[str, np.ndarray], e: float) -> tuple[float, str]:
        """The whole syllable, driven, with air flowing, once both its postures pass.  Before that, what the servos' speed
        allows is a guide for the search and no verdict."""
        names, a0 = self.angles(r, body, closure)
        turn = float(np.abs(self.angles(r, body, vowel)[1][0] - a0[0]).max(initial=0.0))
        if e > 1.0:
            return turn / (GLIDE_S * SLEW), f"the servos alone would take {1000 * turn / SLEW:.0f} ms between them"
        st, frame = r.areas, 0.005
        frames, lens = st.transition({k: float(v[0]) for k, v in closure.items()}, {k: float(v[0]) for k, v in vowel.items()}, frame=frame)
        end = st({k: v for k, v in vowel.items()})[0][0]
        broke = np.nonzero(frames.min(1) > P.A_CLOSE)[0]
        if not broke.size:
            return 2.0, "the seal never breaks"
        off = np.abs(frames - end) > np.maximum(REACHED * end, 0.05)
        last = np.nonzero(off.any(1))[0]
        opens = (float(last[-1] + 1) - float(broke[0])) * frame if last.size else 0.0
        air = S.speak(frames, lens, frame)
        burst = float(np.abs(air["lips"]).max())
        return max(opens, 0.0) / GLIDE_S, (f"from the seal breaking, the tract takes {1000 * max(opens, 0.0):.0f} ms to reach the vowel "
                                           f"(at most {1000 * GLIDE_S:.0f}), the air bursting to {burst:.0f} cm3/s")

    def angles(self, r: P.Rig, body: Body, ctl: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
        names = sorted(r.controls)
        n = next(iter(ctl.values())).shape[0]
        return names, np.stack([ctl[k] for k in names], 1) if names else np.zeros((n, 0))

    def _diagnose(self, body: Body, task: Task, out: Outcome) -> tuple[str, str]:
        """What it reached against what the sound asks: the body's map, nearest posture first."""
        bab = self.babble(body)
        d = task.data if task.data["kind"] != "syllable" else task.data["vowel"]
        e = self._errors(d, bab.f, bab.clo, bab.clash, bab.peak)
        i = int(np.argmin(e))
        said = self._said(d, bab.f[i], bab.clo[i], float(bab.peak[i]))
        if e[i] <= 1.0:
            return "not_found_by_search", f"its map holds a posture that passes: {said}"
        return f"{d['kind']}_out_of_reach", f"the nearest posture of {len(e)} sampled: {said}"

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        bab, g = self.babble(body), graph_of(body)
        lo, hi = np.nanmin(bab.f, 0), np.nanmax(bab.f, 0)
        joints = [f"{j.name} ({j.type}{f', {j.range[0]:.3g}..{j.range[1]:.3g}' if j.range else ''})" for b in g.bodies
                  for j in b.joints]
        return (f"over {len(bab.x)} sampled postures: F1 {lo[0]:.0f}-{hi[0]:.0f}, F2 {lo[1]:.0f}-{hi[1]:.0f}, F3 {lo[2]:.0f}-{hi[2]:.0f} Hz; "
                f"narrowest area {np.min(bab.clo[:, 0]):.2f} cm2, at {np.min(bab.clo[:, 1]):.1f}..{np.max(bab.clo[:, 1]):.1f} cm behind "
                f"the teeth; {sum(isinstance(a, G.Servo) for a in g.actuators)} of {P.CHANNELS} servos, stalled on "
                f"{100 * self._stalled.get(body.key(), 0.0):.0f}% of those postures; joints: " + (", ".join(joints) or "none"))

    def brief_rules(self) -> list[str]:
        return list(RULES)

    def principles(self) -> list[str]:
        return []

    def bom(self, body: Body) -> list[BomLine]:
        g = graph_of(body)
        pla = sum(part_kg(p) * 1000 for b in g.bodies for p in b.parts if getattr(p, "material", "pla") == "pla")
        length = float(self.rig_of(body).areas({})[1][0, : P.NT].sum())
        return [BomLine(item="print:PLA", qty=round(pla, 1), slot=SLOT), BomLine(item="cast:silicone", qty=round(3.0 * length, 1), slot=SLOT),
                BomLine(item="servo_mg90s", qty=sum(isinstance(a, G.Servo) for a in g.actuators), slot=SLOT)]

    def check(self, body: Body) -> list[str]:
        return list(_laws(body.parts[SLOT].mechanism))

    def instruments(self) -> list[Instrument]:
        mk = Instrument
        return [mk(name=f"{n}_{e}", unit="Hz", doc=f"the {'lowest' if e == 'lo' else 'highest'} {n.upper()} over the body's map")
                for n in ("f1", "f2", "f3") for e in ("lo", "hi")] + [
            mk(name="narrowest_cm2", unit="cm2", doc="the smallest area over the body's map"),
            mk(name="stalled_share", unit="", doc="the share of the body's map on which a servo holds at its stall torque"),
            mk(name="servos", unit="", doc="servo channels it uses")]

    def read(self, body: Body, name: str, args: dict[str, Any], task: Task | None = None) -> float:
        bab = self.babble(body)
        if name == "servos":
            return float(len(bab.keys))
        if name == "narrowest_cm2":
            return float(np.min(bab.clo[:, 0]))
        if name == "stalled_share":
            return self._stalled[body.key()]
        if name[:2] in ("f1", "f2", "f3") and name[3:] in ("lo", "hi"):
            col = bab.f[:, int(name[1]) - 1]
            return float(np.nanmin(col) if name[3:] == "lo" else np.nanmax(col))
        raise ValueError(f"no instrument {name!r}")


@lru_cache(maxsize=64)
def _laws(graph_json: str) -> tuple[str, ...]:
    g = G.Graph.model_validate_json(graph_json)
    names = {b.name for b in g.bodies}
    errs = [f"the graph has no body {n!r}, which says where the tube ends" for n in ("glottis", "lip_end") if n not in names]
    joints = {j.name: j for b in g.bodies for j in b.joints}
    servos = [a for a in g.actuators if isinstance(a, G.Servo)]
    if len(servos) > P.CHANNELS:
        errs.append(f"it uses {len(servos)} servos; the controller drives {P.CHANNELS}")
    for a in g.actuators:
        if not isinstance(a, G.Servo):
            errs.append(f"{a.name} is no servo: the robot's motions all come from servos")
    for a in servos:
        j = joints.get(a.joint)
        if abs(a.stall_nm - SERVO_NM) > 1e-6:
            errs.append(f"{a.name} states {a.stall_nm:g} N m: an MG90S is {SERVO_NM:g}")
        if j is None or j.type != "hinge" or j.range is None:
            errs.append(f"{a.name} must turn a hinge with a range: a sliding motion comes through a loop")
    if errs:
        return tuple(errs)
    try:
        m = _model(g)[0]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return (f"it cannot be built: {str(exc)[:300]}",)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return (*floating(g, m, d), *interference(g, m, (), (CM, "cm"), 0.002))
