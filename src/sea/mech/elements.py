"""Machine elements as graph fragments (D-096): the seed library, general to any robot in any world.

Each is what a machine-elements text would show, built from real parts on real joints, with its transmission: a lever,
a four-bar, a slider-crank, a rack and pinion, a gear train, a cam and follower, a ratchet and pawl, a latch, a guide and
carriage.  None is designed for a task.  ``fragment`` builds one mounted on any body; ``compose`` merges fragments into
a graph.  Each element's joints are named with its prefix, so a patch can drive and connect them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from sea.mech.graph import Body, Box, Cylinder, Gear, Graph, Joint, Latch, Loop, Mesh, Quat, Vec

Y = (0.0, 1.0, 0.0)
Z = (0.0, 0.0, 1.0)
X = (1.0, 0.0, 0.0)
PLATE = 0.003  # half thickness of a link plate
BOSS = 0.008  # radius of a pin's boss


def _roty(phi: float) -> Quat:
    """Turning a body's +x by ``phi`` from +x toward -z, about +y."""
    return (math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0)


def _rotz(phi: float) -> Quat:
    return (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))


def _bar(name: str, length: float, half_w: float = 0.006) -> tuple:
    """A bar along +x from the body's origin, with a boss at each pin, swinging in the x-z plane."""
    return (Box(name=name, pos=(length / 2, 0.0, 0.0), half=(length / 2, PLATE, half_w)),
            Cylinder(name=f"{name}_boss0", pos=(0.0, 0.0, 0.0), radius=BOSS, half_length=PLATE, quat=(0.7071068, 0.7071068, 0.0, 0.0)),
            Cylinder(name=f"{name}_boss1", pos=(length, 0.0, 0.0), radius=BOSS, half_length=PLATE, quat=(0.7071068, 0.7071068, 0.0, 0.0)))


def _mount(p: str, on: str, at: Vec, quat: Quat, half: Vec = (0.01, 0.01, 0.01), centre: Vec = (0.0, 0.0, 0.0)) -> Body:
    """The bracket an element is fastened with."""
    return Body(name=f"{p}_mount", parent=on, pos=at, quat=quat, parts=(Box(name=f"{p}_bracket", pos=centre, half=half),))


def lever(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """A bar on a hinge about +y: output travel is input angle times arm."""
    arm = a["arm_m"]
    return Graph(name=p, bodies=(_mount(p, on, at, quat), Body(
        name=f"{p}_arm", parent=f"{p}_mount", joints=(Joint(name=f"{p}_q", type="hinge", axis=Y, range=(-a["swing_rad"], a["swing_rad"])),),
        parts=_bar(f"{p}_bar", arm))), excludes=((f"{p}_mount", f"{p}_arm"),))


def four_bar(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """Crank, coupler and rocker on a ground link along +x, assembled at the crank's start angle: Grashof's relation
    between the four lengths decides whether the crank can turn fully."""
    r1, r2, r3, d, th = a["crank_m"], a["coupler_m"], a["rocker_m"], a["ground_m"], a["start_rad"]
    bx, bz = r1 * math.cos(th), -r1 * math.sin(th)  # the crank's far pin (a positive y-hinge angle turns +x to -z)
    gap = math.hypot(d - bx, bz)
    if gap < 1e-6 or gap > r2 + r3 or gap < abs(r2 - r3):
        raise ValueError(f"the four-bar cannot assemble at start_rad {th:g}: coupler and rocker cannot span {gap:.4f} m")
    alpha = math.atan2(-bz, d - bx)  # the direction from the crank's pin to the rocker's pivot, as a y-hinge angle
    cosb = (r2 * r2 + gap * gap - r3 * r3) / (2 * r2 * gap)
    phi2 = alpha - math.acos(max(-1.0, min(1.0, cosb)))  # the coupler, world angle: the open assembly
    cx, cz = bx + r2 * math.cos(phi2), bz - r2 * math.sin(phi2)
    phi3 = math.atan2(-(cz - 0.0), cx - d)
    return Graph(name=p, bodies=(
        _mount(p, on, at, quat, half=(d / 2 + 0.01, 0.004, 0.01), centre=(d / 2, 0.012, 0.0)),  # behind the links' plane
        Body(name=f"{p}_crank", parent=f"{p}_mount", quat=_roty(th), joints=(Joint(name=f"{p}_q", type="hinge", axis=Y),),
             parts=_bar(f"{p}_crankbar", r1)),
        Body(name=f"{p}_coupler", parent=f"{p}_crank", pos=(r1, 0.0, 0.0), quat=_roty(phi2 - th),
             joints=(Joint(name=f"{p}_q2", type="hinge", axis=Y),), parts=_bar(f"{p}_couplerbar", r2)),
        Body(name=f"{p}_rocker", parent=f"{p}_mount", pos=(d, 0.0, 0.0), quat=_roty(phi3),
             joints=(Joint(name=f"{p}_q3", type="hinge", axis=Y),), parts=_bar(f"{p}_rockerbar", r3))),
        loops=(Loop(a=f"{p}_coupler", b=f"{p}_rocker", point=(r2, 0.0, 0.0)),),
        excludes=((f"{p}_mount", f"{p}_crank"), (f"{p}_crank", f"{p}_coupler"), (f"{p}_coupler", f"{p}_rocker"),
                  (f"{p}_mount", f"{p}_rocker"), (f"{p}_crank", f"{p}_rocker")))


def slider_crank(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """A crank, a connecting rod and a slider along +x: the slider's stroke is twice the crank."""
    r, ln, th = a["crank_m"], a["rod_m"], a["start_rad"]
    if ln <= r:
        raise ValueError("the connecting rod must be longer than the crank")
    bx, bz = r * math.cos(th), -r * math.sin(th)
    sx = bx + math.sqrt(ln * ln - bz * bz)
    phi = math.atan2(bz, sx - bx)  # as a y-hinge angle: the rod runs from the pin back down to the slider's line
    return Graph(name=p, bodies=(
        _mount(p, on, at, quat, half=((r + ln) / 2 + 0.02, 0.004, 0.004), centre=((r + ln) / 2, 0.012, 0.0)),
        Body(name=f"{p}_crank", parent=f"{p}_mount", quat=_roty(th), joints=(Joint(name=f"{p}_q", type="hinge", axis=Y),),
             parts=_bar(f"{p}_crankbar", r)),
        Body(name=f"{p}_rod", parent=f"{p}_crank", pos=(r, 0.0, 0.0), quat=_roty(phi - th),
             joints=(Joint(name=f"{p}_q2", type="hinge", axis=Y),), parts=_bar(f"{p}_rodbar", ln)),
        Body(name=f"{p}_slider", parent=f"{p}_mount", pos=(sx, 0.0, 0.0),
             joints=(Joint(name=f"{p}_s", type="slide", axis=X, range=(-2 * r - 0.01, 2 * r + 0.01)),),
             parts=(Box(name=f"{p}_block", pos=(0.0, 0.006, 0.0), half=(0.015, 0.006, 0.01)),))),
        loops=(Loop(a=f"{p}_rod", b=f"{p}_slider", point=(ln, 0.0, 0.0)),),
        excludes=((f"{p}_mount", f"{p}_crank"), (f"{p}_crank", f"{p}_rod"), (f"{p}_rod", f"{p}_slider"),
                  (f"{p}_mount", f"{p}_slider"), (f"{p}_mount", f"{p}_rod"), (f"{p}_crank", f"{p}_slider")))


def rack_pinion(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """A pinion about +z driving a rack along +x: the rack moves pitch radius times angle."""
    r, w, tr = a["pitch_r_m"], a["width_m"], a["travel_m"]
    return Graph(name=p, bodies=(
        _mount(p, on, at, quat, half=(0.008, 0.008, 0.006), centre=(0.0, 0.0, -w / 2 - 0.006)),
        Body(name=f"{p}_pinion", parent=f"{p}_mount", joints=(Joint(name=f"{p}_q", type="hinge", axis=Z),),
             parts=(Gear(name=f"{p}_gear", r_pitch=r, width=w, z=0.0),)),
        Body(name=f"{p}_rack", parent=f"{p}_mount", pos=(0.0, -r, 0.0),
             joints=(Joint(name=f"{p}_s", type="slide", axis=X, range=(-tr, tr)),),
             parts=(Box(name=f"{p}_bar", pos=(0.0, -0.006, 0.0), half=(tr + 0.01, 0.006, w / 2)),))),
        meshes=(Mesh(follower=f"{p}_s", driver=f"{p}_q", ratio=-r),),
        excludes=((f"{p}_mount", f"{p}_pinion"), (f"{p}_pinion", f"{p}_rack"), (f"{p}_mount", f"{p}_rack")))


def gear_train(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """Two spur gears on parallel shafts: the follower turns the other way, by the ratio of their pitch radii."""
    r1, r2, w = a["driver_r_m"], a["follower_r_m"], a["width_m"]
    return Graph(name=p, bodies=(
        _mount(p, on, at, quat, half=((r1 + r2) / 2 + 0.008, 0.008, 0.004), centre=((r1 + r2) / 2, 0.0, -w / 2 - 0.004)),
        Body(name=f"{p}_driver", parent=f"{p}_mount", joints=(Joint(name=f"{p}_q", type="hinge", axis=Z),),
             parts=(Gear(name=f"{p}_gear1", r_pitch=r1, width=w, z=0.0),)),
        Body(name=f"{p}_follower", parent=f"{p}_mount", pos=(r1 + r2, 0.0, 0.0),
             joints=(Joint(name=f"{p}_q2", type="hinge", axis=Z),), parts=(Gear(name=f"{p}_gear2", r_pitch=r2, width=w, z=0.0),))),
        meshes=(Mesh(follower=f"{p}_q2", driver=f"{p}_q", ratio=-r1 / r2),),
        excludes=((f"{p}_mount", f"{p}_driver"), (f"{p}_mount", f"{p}_follower"), (f"{p}_driver", f"{p}_follower")))


def cam(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """An eccentric disc under a flat follower on a vertical guide: the follower rises twice the eccentricity per turn,
    by contact, and gravity or a spring returns it."""
    r, e = a["disc_r_m"], a["eccentric_m"]
    if e >= r:
        raise ValueError("the eccentricity must be less than the disc's radius")
    disc = Cylinder(name=f"{p}_disc", pos=(e, 0.0, 0.0), radius=r, half_length=0.005, quat=(0.7071068, 0.7071068, 0.0, 0.0))
    hub = Cylinder(name=f"{p}_hub", pos=(0.0, 0.0, 0.0), radius=BOSS, half_length=0.006, quat=(0.7071068, 0.7071068, 0.0, 0.0))
    lift = r + e  # the disc's top with its eccentric point up: the follower's start clears it there
    return Graph(name=p, bodies=(
        _mount(p, on, at, quat, half=(0.006, 0.006, (lift + 0.03) / 2), centre=(0.0, 0.02, (lift + 0.03) / 2)),
        Body(name=f"{p}_cam", parent=f"{p}_mount", quat=_roty(-math.pi / 2), joints=(Joint(name=f"{p}_q", type="hinge", axis=Y),),
             parts=(disc, hub)),
        Body(name=f"{p}_follower", parent=f"{p}_mount", pos=(0.0, 0.0, lift),
             joints=(Joint(name=f"{p}_s", type="slide", axis=Z, range=(-2 * e - 0.005, 0.03)),),
             parts=(Box(name=f"{p}_flat", pos=(0.0, 0.0, 0.003), half=(r + e, 0.008, 0.003)),
                    Box(name=f"{p}_stem", pos=(0.0, 0.012, 0.003), half=(0.004, 0.006, 0.003))))),
        contacts=((f"{p}_disc", f"{p}_flat"),),
        excludes=((f"{p}_mount", f"{p}_cam"), (f"{p}_mount", f"{p}_follower"), (f"{p}_cam", f"{p}_follower")))


def ratchet(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """A toothed wheel and a sprung pawl lying along the wheel's tangent.  Turned positively, each tooth rides up under
    the pawl and lifts it; turned back, a tooth's face meets the pawl's end and pushes along it into its pivot, and locks.
    Tested to hold 10 N at the teeth: above that the pawl's stop gives."""
    r, n = a["wheel_r_m"], int(round(a["teeth"]))
    h = max(0.006, 0.3 * 2 * math.pi * r / n)  # how far a tooth stands proud
    angle = [-math.pi / 4 + (k + 0.5) * 2 * math.pi / n for k in range(n)]  # the pawl starts in a gap, not in a tooth
    teeth = tuple(Box(name=f"{p}_tooth{k}", pos=((r + h / 2) * math.cos(th), 0.0, -(r + h / 2) * math.sin(th)),
                      half=(h / 2 + 0.002, 0.004, 0.002), quat=_roty(th)) for k, th in enumerate(angle))
    disc = Cylinder(name=f"{p}_disc", pos=(0.0, 0.0, 0.0), radius=r, half_length=0.004, quat=(0.7071068, 0.7071068, 0.0, 0.0))
    # the pawl's line: just inside the tangent at 45 degrees up the wheel, so a load on its end draws it deeper in
    rest, ln = math.pi / 4 + 0.45, 1.2 * r  # steeper than friction at the tooth (tan 0.45 > 0.3) can lift it
    tip = (r + 0.4 * h + 0.003) * math.sqrt(0.5)
    px, pz = tip - ln * math.cos(rest), tip + ln * math.sin(rest)
    return Graph(name=p, bodies=(
        _mount(p, on, at, quat, half=(r + h + 0.01, 0.004, (pz + r + h) / 2 + 0.01), centre=(0.0, 0.012, (pz - r - h) / 2)),
        Body(name=f"{p}_wheel", parent=f"{p}_mount", joints=(Joint(name=f"{p}_q", type="hinge", axis=Y, damping=0.005),),
             parts=(disc, *teeth)),
        Body(name=f"{p}_pawl", parent=f"{p}_mount", pos=(px, 0.0, pz), quat=_roty(rest), joints=(Joint(
            name=f"{p}_pawl_q", type="hinge", axis=Y, range=(-0.8, 0.02), stiffness=3.6 * r, springref=0.5, damping=0.1 * r, armature=0.002),),
            parts=(Box(name=f"{p}_pawlbar", pos=(ln / 2, 0.0, 0.0), half=(ln / 2, 0.004, 0.003)),))),
        contacts=tuple((f"{p}_tooth{k}", f"{p}_pawlbar") for k in range(n)),
        excludes=((f"{p}_mount", f"{p}_wheel"), (f"{p}_mount", f"{p}_pawl"), (f"{p}_wheel", f"{p}_pawl")))


def latch(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """A bar on a hinge, held until released: what stores a spring's energy until its moment."""
    g = lever(p, on, at, quat, a)
    return g.model_copy(update={"latches": (Latch(name=f"{p}_latch", joint=f"{p}_q"),)})


def guide(p: str, on: str, at: Vec, quat: Quat, a: dict[str, float]) -> Graph:
    """A carriage on two rods along +x, held at both ends: straight-line travel."""
    ln, gap, tr = a["rod_m"], a["rod_gap_m"], a["travel_m"]
    if tr >= ln / 2:
        raise ValueError("the carriage's travel must be less than half the rods' length")
    rods = tuple(Cylinder(name=f"{p}_rod{k}", pos=(0.0, s * gap / 2, 0.0), radius=0.0015, half_length=ln / 2, material="steel",
                          role="rod", quat=(0.7071068, 0.0, 0.7071068, 0.0)) for k, s in enumerate((1, -1)))
    ends = tuple(Box(name=f"{p}_end{k}", pos=(s * ln / 2, 0.0, 0.0), half=(0.004, gap / 2 + 0.005, 0.006))
                 for k, s in enumerate((1, -1)))
    return Graph(name=p, bodies=(
        Body(name=f"{p}_mount", parent=on, pos=at, quat=quat, parts=(*rods, *ends)),
        Body(name=f"{p}_carriage", parent=f"{p}_mount", joints=(Joint(name=f"{p}_s", type="slide", axis=X, range=(-tr, tr)),),
             parts=(Box(name=f"{p}_block", pos=(0.0, 0.0, 0.0), half=(0.015, gap / 2 + 0.004, 0.008)),))),
        excludes=((f"{p}_mount", f"{p}_carriage"),))


@dataclass(frozen=True)
class Element:
    build: Callable[[str, str, Vec, Quat, dict[str, float]], Graph]
    params: dict[str, tuple[float, float]]
    joints: str  # what a patch or a plan connects to
    provides: str


_LEN = (0.01, 0.4)
ELEMENTS: dict[str, Element] = {
    "lever": Element(lever, {"arm_m": _LEN, "swing_rad": (0.1, 3.2)}, "<p>_q", "turns an angle into travel at its arm"),
    "four_bar": Element(four_bar, {"crank_m": _LEN, "coupler_m": _LEN, "rocker_m": _LEN, "ground_m": _LEN,
                                   "start_rad": (-3.2, 3.2)}, "<p>_q drives; <p>_q2 and <p>_q3 follow",
                        "turns a rotation into a rocking one, and a coupler point along a curve"),
    "slider_crank": Element(slider_crank, {"crank_m": _LEN, "rod_m": _LEN, "start_rad": (-3.2, 3.2)},
                            "<p>_q drives, <p>_s follows; or the other way", "turns rotation into straight travel"),
    "rack_pinion": Element(rack_pinion, {"pitch_r_m": (0.005, 0.1), "width_m": (0.003, 0.03), "travel_m": (0.01, 0.4)},
                           "<p>_q drives, <p>_s follows", "turns rotation into straight travel, by teeth"),
    "gear_train": Element(gear_train, {"driver_r_m": (0.005, 0.1), "follower_r_m": (0.005, 0.1), "width_m": (0.003, 0.03)},
                          "<p>_q drives, <p>_q2 follows", "trades speed for torque"),
    "cam": Element(cam, {"disc_r_m": (0.01, 0.1), "eccentric_m": (0.002, 0.05)}, "<p>_q drives, <p>_s follows by contact",
                   "turns rotation into a timed lift"),
    "ratchet": Element(ratchet, {"wheel_r_m": (0.015, 0.1), "teeth": (6.0, 16.0)}, "<p>_q turns; <p>_pawl_q follows",
                       "lets a shaft turn one way only, and holds what it wound"),
    "latch": Element(latch, {"arm_m": _LEN, "swing_rad": (0.1, 3.2)}, "<p>_q; <p>_latch holds it until released",
                     "stores energy until a chosen moment"),
    "guide": Element(guide, {"rod_m": (0.05, 0.6), "rod_gap_m": (0.01, 0.1), "travel_m": (0.01, 0.28)}, "<p>_s",
                     "carries a body along a straight line"),
}


def fragment(name: str, prefix: str, on: str, at: Vec = (0.0, 0.0, 0.0), quat: Quat = (1.0, 0.0, 0.0, 0.0),
             params: dict[str, float] | None = None) -> Graph:
    """Element ``name``, its names prefixed, mounted on body ``on``.  Parameters left out take their range's middle."""
    if name not in ELEMENTS:
        raise ValueError(f"no element {name!r}; the elements are {sorted(ELEMENTS)}")
    el = ELEMENTS[name]
    given = dict(params or {})
    bad = [f"{k} {v:g} outside {el.params[k][0]:g}..{el.params[k][1]:g}" if k in el.params else f"no parameter {k!r}"
           for k, v in given.items() if k not in el.params or not el.params[k][0] <= v <= el.params[k][1]]
    if bad:
        raise ValueError(f"{name}: " + "; ".join(bad))
    return el.build(prefix, on, at, quat, {k: given.get(k, (lo + hi) / 2) for k, (lo, hi) in el.params.items()})


def merge(g: Graph, frag: Graph) -> Graph:
    """``g`` with a fragment's bodies and constraints added.  A name used twice is refused."""
    have = {b.name for b in g.bodies}
    twice = sorted(have & {b.name for b in frag.bodies})
    if twice:
        raise ValueError(f"bodies {twice} already exist: give the element another prefix")
    if frag.bodies[0].parent not in have and frag.bodies[0].parent not in {"world"}:
        raise ValueError(f"no body {frag.bodies[0].parent!r} to mount on; the bodies are {sorted(have)}")
    fields = ("bodies", "loops", "meshes", "guides", "excludes", "linked", "actuators", "latches", "sensors", "carried", "contacts")
    return g.model_copy(update={f: (*getattr(g, f), *getattr(frag, f)) for f in fields})


def compose(g: Graph, uses: list) -> Graph:
    """``g`` with each element in ``uses`` mounted (a mechanism's ``uses``), in order, so one may mount on another."""
    for u in uses:
        g = merge(g, fragment(u.element, u.prefix, u.on, tuple(u.at), tuple(u.quat), u.params))
    return g
