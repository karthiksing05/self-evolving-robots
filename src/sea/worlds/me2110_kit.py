"""ME 2110 from the kit (D-094): the brief's arena, scoring and kit, and no mechanism anyone has designed.

A body is one part, ``robot``: a graph of plates, wheels, links, the kit's actuators and sensors.  It starts as a
plain plate with the controller on it, which scores nothing, as a student starts from a kit and no robot.  System 2
composes everything else, and each robot is operated by its graph's own controls (``mech.program``).

The world holds the brief and the physics, and no advice: what it knows about a robot is what it measures.  Only the
``match`` is a task: the brief scores one robot in one 40 s round.  ``sea.worlds.me2110`` stays as the old catalogue's
baseline.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np

from sea.evolve.core import Body, BomLine, Instrument, Outcome, Part, PartType, Task
from sea.evolve.search import local_search, planned
from sea.mech import graph as G
from sea.mech.build import part_kg
from sea.mech.patch import Op, apply_patch
from sea.mech.patch import __doc__ as PATCH_DOC
from sea.mech.program import controls
from sea.sim import kit_match as K
from sea.sim import me2110_model as MM
from sea.worlds.me2110 import BOX, VARIANTS, ME2110World, fits_box, judged, law_label

IN = MM.IN
SLOT = "robot"
ATTEMPTS = 24  # System 1's matches per body and track; a plan's ``search`` multiplies them
SCORED_OVER = 3  # matches a score is the mean of, at System 1's best controls (D-094)
TOUCH = 0.002  # m: two parts of one body closer than this are fastened
SWEEP = 5  # poses over each driven joint's range
PASS_TOL = 0.1 * IN  # m: how far two parts may overlap where one is fastened to the other
# the brief's p. 26 list; a limit switch, the IR and ultrasonic rangers, and the encoder and potentiometer
KIT = {"gearmotors": 2, "pistons": 2, "mousetraps": 4, "bands": 5, "limit_switches": 2, "rangefinders": 2, "encoders": 2}
RULES = [
    "The robot starts inside a 12 x 24 x 18 in box (it may turn in plan; 18 in is always height), its parts behind the "
    "start line, motionless, its stored energy held by latches.  At 40 s it must be static with its pistons retracted, "
    "or the round is a disqualification and scores nothing.",
    "Energy may come only from the kit: 2 gearmotors, 2 pistons on one 7 bar tank, 4 mousetraps, 5 #64 rubber bands, the "
    "controller, and gravity.  The parts bought must cost under $120; prints cost $0.05 a gram.",
    "A projectile may fly at most 12 in unsupported and may never rise after leaving the robot.  No magnets on the robot.",
    "Leaving the quadrant is a disqualification; the fortress's cylinder is common ground, but a ball does not score if "
    "any part of the robot is inside it at the end.",
]
ARENA = (
    f"All in inches, x forward from the start line, y left, from the centre of our dock's width.  The dock is {MM.DOCK:g} "
    f"wide.  The fortress: centre x = {MM.XF:g}, radius {MM.RF:g}, height {MM.HF:g}, turning clockwise; four ports "
    f"{MM.PORT_W:g} wide with barriers 6, 7, 8 and 8.75 up; ours is the task's barrier, its colour tab beside it.  The "
    f"coins (radius {MM.COIN_R:g}, {MM.COIN_T:g} thick) and jewels (radius {MM.JEWEL_R:g}) sit on its top's rim.  The pole "
    f"rises from its centre, {2 * MM.POLE_HALF:g} square, to {MM.POLE_TOP:g}; the flag ({MM.FLAG_W:g} x {MM.FLAG_H:g}) must "
    f"sit flat on a face at 24 or above.  Mounds, {MM.MOUND_X:g} x {MM.MOUND_Y:g} and {MM.MOUND_H:g} tall, {MM.BORDER:g} "
    f"border: gold at {MM.MOUNDS['gold'][:2]}, jewels at {MM.MOUNDS['jewel'][:2]}.  Merfolk, on {2 * MM.MARK_R:g} marks: "
    f"{ {k: v[:2] for k, v in MM.MERFOLK.items()} }.  The chest: its near face at x = {MM.CHEST_X:g}, y = 0, "
    f"{MM.CHEST_D:g} deep, {MM.CHEST_W:g} wide, {MM.CHEST_H:g} tall; its lid is propped open and shuts if the chest moves "
    f"{MM.PROP_MOVE:g} or tilts; the heart ({MM.HEART_L:g} long) lies just under its lip.  Items a graph's `carried` may "
    "start on the robot: ball0..ball3 (40 mm, 2.7 g), and the flag, clipped to its body until a pull of 1 N, held to the "
    "pole by its magnet up to 3 N.")


def _chassis(params: dict[str, float]) -> G.Graph:
    """A plate on four feet, with the controller on it."""
    ln, w = params["length_in"] * IN / 2, params["width_in"] * IN / 2
    t, belly = 0.125 * IN, MM.BELLY * IN
    feet = [G.Box(name=f"foot{k}", pos=(sx * (ln - 0.5 * IN), sy * (w - 0.5 * IN), belly / 2), material="plywood",
                  half=(0.5 * IN, 0.5 * IN, belly / 2), mu=0.8)
            for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))]
    return G.Graph(name="chassis", bodies=(G.Body(
        name=K.ROOT, parent="world", joints=(G.Joint(name="root", type="free"),),
        parts=(G.Box(name="plate", pos=(0.0, 0.0, belly + t), half=(ln, w, t), material="plywood"), *feet,
               G.Box(name="controller", pos=(-ln / 2, 0.0, belly + 2 * t + 0.75 * IN),
                     half=(min(2.5 * IN, ln / 2), min(2.0 * IN, w / 2), 0.75 * IN), massive=False))),))


MODULES: dict[str, tuple[Any, dict[str, tuple[float, float]]]] = {
    "chassis": (_chassis, {"length_in": (6.0, 24.0), "width_in": (6.0, 24.0)})}


def _module(name: str, params: dict[str, float]) -> G.Graph:
    if name not in MODULES:
        raise ValueError(f"no module {name!r}; the modules are {sorted(MODULES)}")
    build, ranges = MODULES[name]
    bad = [f"{k} {v:g} outside {ranges[k][0]:g}..{ranges[k][1]:g}" if k in ranges else f"no parameter {k!r}"
           for k, v in params.items() if k not in ranges or not ranges[k][0] <= v <= ranges[k][1]]
    if bad:
        raise ValueError("; ".join(bad))
    return build({k: params.get(k, (lo + hi) / 2) for k, (lo, hi) in ranges.items()})


def graph_of(body: Body) -> G.Graph:
    return G.Graph.model_validate_json(body.parts[SLOT].mechanism)


def _usage(g: G.Graph) -> dict[str, int]:
    return {"gearmotors": sum(isinstance(a, G.Gearmotor) for a in g.actuators),
            "pistons": sum(isinstance(a, G.Piston) for a in g.actuators),
            "mousetraps": sum(isinstance(a, G.TorsionSpring) for a in g.actuators),
            "bands": sum(isinstance(a, G.Band) for a in g.actuators),
            "limit_switches": sum(isinstance(s, G.LimitSwitch) for s in g.sensors),
            "rangefinders": sum(isinstance(s, G.RangeFinder) for s in g.sensors),
            "encoders": sum(isinstance(s, G.Encoder) for s in g.sensors)}


# ---------------------------------------------------------------- the laws
def _floating(g: G.Graph, m: mujoco.MjModel, d: mujoco.MjData) -> list[str]:
    """A body's solids must touch: each is fastened to another or it is floating."""
    out = []
    for b in g.bodies:
        ids = [x for x in range(m.ngeom) if m.geom_bodyid[x] == m.body(b.name).id and m.geom_contype[x] == 2]
        if len(ids) < 2:
            continue
        reached, todo = {ids[0]}, [ids[0]]
        while todo:
            a = todo.pop()
            for c in ids:
                if c not in reached and mujoco.mj_geomDistance(m, d, a, c, TOUCH * 2, None) <= TOUCH:
                    reached.add(c)
                    todo.append(c)
        loose = sorted(m.geom(x).name for x in ids if x not in reached)
        if loose:
            out.append(f"on body {b.name!r}, {', '.join(loose)} touch(es) nothing fastened to {m.geom(ids[0]).name}: floating")
    return out


def _driven(g: G.Graph) -> list[str]:
    names = [a.joint for a in g.actuators if not isinstance(a, G.Band)] + [lt.joint for lt in g.latches]
    return list(dict.fromkeys(names))


def _close(m: mujoco.MjModel, d: mujoco.MjData, free: list[int]) -> None:
    """Move the joints of closed loops until the loops close, by least squares from where they are."""
    from scipy.optimize import least_squares

    adrs = [m.jnt_qposadr[j] for j in free]

    def res(x: np.ndarray) -> np.ndarray:
        d.qpos[adrs] = x
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_makeConstraint(m, d)
        rows = d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        return d.efc_pos[: d.nefc][rows].copy()

    if free and res(d.qpos[adrs].copy()).size:
        d.qpos[adrs] = least_squares(res, d.qpos[adrs].copy(), max_nfev=40).x


def _interference(g: G.Graph, m: mujoco.MjModel) -> list[str]:
    """Parts may not pass through each other: each driven joint swept over its range, with closed loops solved at each
    pose.  Parts of bodies joined to each other may overlap as they are fastened, not further as they move."""
    d = mujoco.MjData(m)
    names = {b.name for b in g.bodies}
    ids = [x for x in range(m.ngeom) if m.geom_contype[x] == 2 and m.body(m.geom_bodyid[x]).name in names]
    excl = {frozenset(e) for e in g.excludes}
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if ba != bb and frozenset((m.body(ba).name, m.body(bb).name)) not in excl:
                pairs.append((a, b, m.body_parentid[ba] == bb or m.body_parentid[bb] == ba))
    keep = [i for i in range(m.neq) if m.eq(i).name in {lt.name for lt in g.latches} or m.eq(i).name in ("prop", "clip")]
    looped = [m.joint(j.name).id for lp in g.loops for b in g.bodies if b.name in (lp.a, lp.b) for j in b.joints
              if j.type != "free"]
    over: dict[tuple[int, int], list[float]] = {}
    worst_pose: dict[tuple[int, int], str] = {}
    for jn in _driven(g) or [""]:
        jid = m.joint(jn).id if jn else -1
        lo, hi = (m.jnt_range[jid] if jn and m.jnt_limited[jid] else (0.0, 0.0))
        for q in np.linspace(lo, hi, SWEEP if hi > lo else 1):
            mujoco.mj_resetData(m, d)
            d.eq_active[keep] = 0
            if jn:
                d.qpos[m.jnt_qposadr[jid]] = q
            _close(m, d, sorted({j for j in looped if j != jid}))
            mujoco.mj_kinematics(m, d)
            for a, b, _ in pairs:
                r = m.geom_rbound[a] + m.geom_rbound[b]
                gap = 0.0 if np.sum((d.geom_xpos[a] - d.geom_xpos[b]) ** 2) > r * r else \
                    min(0.0, mujoco.mj_geomDistance(m, d, a, b, 0.01, None))
                over.setdefault((a, b), []).append(-gap)
                if -gap >= max(over[(a, b)]):
                    worst_pose[(a, b)] = f"{jn or 'at rest'} at {q:.2f}" if jn else "at rest"
    rows = []
    for a, b, joined in pairs:
        ov = over.get((a, b), [0.0])
        bad = max(ov) - (min(ov) if joined else 0.0)
        if bad > PASS_TOL:
            rows.append((bad, f"{m.geom(a).name} and {m.geom(b).name} pass {bad / IN:.2f} in into each other "
                              f"({worst_pose[(a, b)]})"))
    return [t for _, t in sorted(rows, reverse=True)[:6]]


@lru_cache(maxsize=64)
def _laws(graph_json: str) -> tuple[str, ...]:
    g = G.Graph.model_validate_json(graph_json)
    errs = [f"it uses {n} {k}; the kit has {KIT[k]}" for k, n in _usage(g).items() if n > KIT[k]]
    stated = [p.name for b in g.bodies for p in b.parts if getattr(p, "kg", None) is not None]
    if stated:
        errs.append(f"{', '.join(stated)} state their mass: a part's mass is its material's volume")
    try:
        cp = K.compile_robot(g, VARIANTS["train"][0])
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return (*errs, f"it cannot be built: {str(exc)[:300]}")
    x0, x1, y0, y1, z0, z1 = cp.extent_in
    env = (x1 - x0, y1 - y0, z1 - min(z0, 0.0))
    if not fits_box(env):
        errs.append(f"at its start pose it is {env[0]:.1f} x {env[1]:.1f} x {env[2]:.1f} in: it does not fit the "
                    f"{BOX[0]:g} x {BOX[1]:g} x {BOX[2]:g} in go/no-go box")
    if z0 < -0.05:
        errs.append(f"a part starts {-z0:.2f} in under the floor")
    m = cp.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return (*errs, *_floating(g, m, d), *_interference(g, m))


class KitWorld:
    name = "me2110_kit"
    description = ("The ME 2110 'Pirates of the Caribbean' robot, from the kit: one machine that in a 40 s match leaves "
                   "the dock (11 points), sorts coins and jewels off a turning fortress onto their mounds (10 or 4 a coin, "
                   "15 or 9 a jewel, 175 at most), runs a flag up a magnetic pole (40), drops 4 cannonballs into its "
                   "colour's port (9 each), topples two merfolk (12 each, or 5 if only moved) and brings a heart back "
                   f"from a chest (61, or 39 on the robot; the chest, 32 or 11).  {ARENA}")
    objective = "points"
    points_noise = 9.0  # one ball (D-094)
    sprint_bars = (60.0, 150.0, 215.0)
    max_points = float(sum(MM.MAX_POINTS.values()))
    research_blocked = ME2110World.research_blocked

    def tasks(self) -> list[Task]:
        return [t for t in ME2110World().tasks() if t.group == "match"]

    def catalogue(self) -> list[PartType]:
        return []

    def stock(self) -> Body:
        return Body(parts={SLOT: Part(type="built", mechanism=_module("chassis", {"length_in": 11.0, "width_in": 12.0})
                                      .model_dump_json())})

    # mechanisms System 2 composes
    def build_idea(self, body: Body, idea: Any, new_id: str) -> Body:
        mech = idea.mechanism
        if idea.slot not in ("", SLOT):
            raise ValueError(f"a robot here is one graph: slot {idea.slot!r} does not exist")
        if mech.module == "self":
            if mech.params:
                raise ValueError("module self has no parameters: write the change in the patch")
            g = graph_of(body)
        else:
            g = _module(mech.module, mech.params)
        g = apply_patch(g, [Op.model_validate(o) for o in mech.patch]).model_copy(update={"name": idea.name})
        return Body(id=new_id, parent=body.id, parts={SLOT: Part(type="built", mechanism=g.model_dump_json())})

    def module_names(self) -> list[str]:
        return sorted(MODULES)

    def mechanism_text(self, body: Body) -> str:
        return body.parts[SLOT].mechanism

    def module_text(self, name: str, params: dict[str, float]) -> str:
        return _module(name, params).model_dump_json()

    def modules_text(self) -> str:
        kinds = []
        for cls in (G.Box, G.Cylinder, G.Capsule, G.Sphere, G.Wheel, G.Bought, G.Body, G.Joint, G.Loop, G.Servo, G.Gearmotor,
                    G.Piston, G.TorsionSpring, G.Band, G.Latch, G.LimitSwitch, G.RangeFinder, G.Encoder, G.Carry):
            fields = ", ".join(f"{n}" + ("" if f.is_required() else f"={f.default!r}") for n, f in cls.model_fields.items()
                               if n not in ("kind", "kg"))
            kinds.append(f"- {cls.__name__}: {' '.join((cls.__doc__ or '').split())} ({fields})")
        out = ["A robot is a graph (lengths in metres; a part's coordinates in its body's frame; x forward, z up from the "
               "floor).  Its first body is 'robot' on a free joint 'root'.  Every part fastened to a body must touch "
               "another on it.  Joints, actuators, latches and sensors are what operate it: each gives System 1 controls, "
               "which your plan sets.", "Kinds:", *kinds, (PATCH_DOC or "").strip(), "Modules:"]
        for name, (_, ranges) in MODULES.items():
            mid = {k: (lo + hi) / 2 for k, (lo, hi) in ranges.items()}
            out += [f"## {name} ({', '.join(f'{k} {lo:g}..{hi:g}' for k, (lo, hi) in ranges.items())}); at {mid}:",
                    _module(name, mid).model_dump_json()]
        return "\n".join(out)

    def control_names(self, body: Body) -> dict[str, tuple[float, float]]:
        return {k: v[:2] for k, v in self._space(graph_of(body)).items()}

    @staticmethod
    def _space(g: G.Graph) -> dict[str, tuple[float, float, float]]:
        half = K.compile_robot(g, VARIANTS["train"][0]).extent_in
        room = MM.DOCK / 2 - (half[3] - half[2]) / 2
        return {**controls(g, 40.0), "y0": (-max(room, 0.0), max(room, 0.0), 0.0)}

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        g = graph_of(body)
        v = VARIANTS[task.split][task.data["variant"]]
        trig = body.plan.triggers if body.plan is not None else {}
        spec = self._space(g)
        space = {k: x[:2] for k, x in spec.items() if x[1] > x[0]}
        start, m = planned(body, space, {k: spec[k][2] for k in space}, ATTEMPTS)

        def label(_: str, r: MM.Result) -> str:
            broke = law_label(r)
            if broke:
                return broke
            return "scored_nothing_at:" + ",".join(k for k, p in r.points.items() if p <= 0)

        def run(ctl: dict[str, float], k: int) -> Outcome:
            t0 = time.perf_counter()
            r = K.KitMatch(g, v, ctl, seed=int(task.data["seed"]) * 1000 + k, horizon=40.0, triggers=trig).run()
            return judged("match", r, label, t0)

        if not space:  # nothing on the robot can be commanded
            return run({}, seed).model_copy(update={"task": task.id})
        best = local_search(run, space, start, m=m, confirm=0, seed=seed + int(task.data["seed"]))
        again = [run(best.controls, seed + 2000 + j) for j in range(SCORED_OVER)]  # a lucky match is not a score
        keys = {k for o in again for k in o.states}
        return best.model_copy(update={
            "task": task.id, "points": float(np.mean([o.points for o in again])),
            "states": {k: float(np.mean([o.states.get(k, 0.0) for o in again])) for k in keys},
            "success": all(o.success for o in again), "attempts": best.attempts + SCORED_OVER,
            "sim_s": round(best.sim_s + sum(o.sim_s for o in again), 3),
            "detail": f"over {SCORED_OVER} matches {[round(o.points or 0.0) for o in again]} points; {again[0].detail}"})

    # what is known about a body
    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        g = graph_of(body)
        cp = K.compile_robot(g, VARIANTS["train"][0])
        e = cp.extent_in
        used = _usage(g)
        joints = [f"{j.name} ({j.type}{f', {j.range[0]:.3g}..{j.range[1]:.3g}' if j.range else ''})" for b in g.bodies
                  for j in b.joints if j.type != "free"]
        return (f"at its start pose {e[1] - e[0]:.1f} x {e[3] - e[2]:.1f} x {e[5]:.1f} in (box {BOX[0]:g} x {BOX[1]:g} x "
                f"{BOX[2]:g}); {self.read(body, 'mass_kg', {}, None):.2f} kg with the kit's parts; kit used: "
                + ", ".join(f"{n} of {KIT[k]} {k}" for k, n in used.items()) + "; joints: " + (", ".join(joints) or "none"))

    def principles(self) -> list[str]:
        return list(RULES)

    def bom(self, body: Body) -> list[BomLine]:
        g = graph_of(body)
        out = []
        wheels = screws = plywood = 0.0
        pla = 0.0
        for b in g.bodies:
            for p in b.parts:
                mat = getattr(p, "material", "pla")
                if isinstance(p, G.Wheel):
                    wheels += 1
                elif mat == "plywood" and isinstance(p, G.Box):
                    plywood += 4 * p.half[0] * p.half[1] / IN / IN
                elif mat == "pla":
                    pla += part_kg(p) * 1000
                screws += 2 if mat != "stock" else 0
        if plywood:
            out.append(BomLine(item="plywood_in2", qty=round(plywood, 1), slot=SLOT))
        if wheels:
            out.append(BomLine(item="wheel_pair", qty=float(np.ceil(wheels / 2)), slot=SLOT))
        if pla:
            out.append(BomLine(item="print:PLA", qty=round(pla, 1), slot=SLOT))
        out.append(BomLine(item="m3_screw", qty=screws, slot=SLOT))
        return out

    def check(self, body: Body) -> list[str]:
        return list(_laws(body.parts[SLOT].mechanism))

    def manifest(self) -> dict[str, Any]:
        return ME2110World().manifest()

    # D-088
    def instruments(self) -> list[Instrument]:
        mk = Instrument
        return [mk(name="length_in", unit="in", doc="its length along x at the start pose"),
                mk(name="width_in", unit="in", doc="its width along y at the start pose"),
                mk(name="height_in", unit="in", doc="its height at the start pose"),
                mk(name="mass_kg", unit="kg", doc="its mass, the kit's parts included"),
                mk(name="kit_used", unit="", doc="kit items of a kind used", args={"kind": f"one of {sorted(KIT)}"}),
                mk(name="kit_left", unit="", doc="kit items of a kind still free", args={"kind": f"one of {sorted(KIT)}"}),
                mk(name="excludes", unit="", doc="how many pairs of bodies the interference law is told to ignore"),
                mk(name="point_in", unit="in", doc="where a point of a body is, from the robot's start-pose frame, with "
                  "one joint moved", args={"body": "a body's name", "axis": "x, y or z", "joint": "a joint's name, or "
                                           "omit", "q": "its value", "x": "m in the body", "y": "m", "z": "m"})]

    def read(self, body: Body, name: str, args: dict[str, Any], task: Task | None = None) -> float:
        g = graph_of(body)
        if name in ("kit_used", "kit_left"):
            kind = str(args.get("kind", ""))
            if kind not in KIT:
                raise ValueError(f"no kit kind {kind!r}; the kinds are {sorted(KIT)}")
            return float(_usage(g)[kind] if name == "kit_used" else KIT[kind] - _usage(g)[kind])
        if name == "excludes":
            return float(len(g.excludes))
        cp = K.compile_robot(g, VARIANTS["train"][0])
        e = cp.extent_in
        if name in ("length_in", "width_in", "height_in"):
            return {"length_in": e[1] - e[0], "width_in": e[3] - e[2], "height_in": e[5]}[name]
        m = cp.model
        if name == "mass_kg":
            return float(m.body_subtreemass[m.body(K.ROOT).id])
        if name == "point_in":
            d = mujoco.MjData(m)
            if args.get("joint"):
                d.qpos[m.jnt_qposadr[m.joint(str(args["joint"])).id]] = float(args.get("q", 0.0))
            mujoco.mj_kinematics(m, d)
            b = m.body(str(args.get("body", K.ROOT))).id
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([float(args.get(k, 0.0)) for k in "xyz"])
            root = d.xpos[m.body(K.ROOT).id]
            return float((p - root)["xyz".index(str(args.get("axis", "x")))]) / IN
        raise ValueError(f"no instrument {name!r}")

    def task_facts(self, task: Task) -> dict[str, float]:
        return {k: float(task.data[k]) for k in ("barrier_in", "rpm", "mu", "phase_rad")}

