"""System 1 between attempts (spec 8.1 ``local_search.py``, 3.3).

``run_system1`` runs one program on one control the way System 1 handles it:

1. the first attempt;
2. the **fixed retry rule**: once, with the target re-localized (a fresh perception draw when
   perception noise is on) and offset 1 mm toward the control centre;
3. **local search**: CMA-ES (seeded) over the free continuous parameters of the failing
   primitive, at most ``m`` trials, stopping at the first pass.  Bounds are the declared range
   (or a System 2 ``reparam`` range) intersected with ``x0 +/- max_param_delta x width``.

Structural verdicts (``unreachable``, ``rule_violation``, ``no_applicable_primitive``) return at
once: parameter tuning cannot fix them (escalation rule 1).  ``S1Outcome.escalate`` says whether
System 1 gave up (rule 1 or rule 2: its ``m`` trials, or bounds, are exhausted without a pass).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from sea.compose import dsl
from sea.contracts import ControlResult, PrimitiveDef, Program, StepTrace
from sea.control.executor import EpisodeConfig, reset_episode, run_program

STRUCTURAL = frozenset({"unreachable", "rule_violation", "no_applicable_primitive"})
RETRY_OFFSET_M = 1e-3
PERCEPTION_SIGMA_M = 1e-3  # spec 8.9: target pose noise N(0, 1 mm)
SIGMA0 = 0.3  # CMA-ES step size in the normalized [0, 1] box


@dataclass
class S1Outcome:
    result: ControlResult
    trace: StepTrace
    trials: int  # local-search trials (the fixed retry is not counted toward m)
    params: dict[str, dict[str, float]] = field(default_factory=dict)  # overrides of the returned attempt
    history: list[tuple[dict, str]] = field(default_factory=list)  # (overrides or {"retry": ...}, verdict)
    escalate: str | None = None  # "structural" | "s1_exhausted" | None (passed)
    searched: str = ""  # primitive whose parameters were searched


# ---------------------------------------------------------------- search space
def _explicit_args(root, name: str, library: Mapping[str, PrimitiveDef]) -> set[str]:
    """Parameters of ``name`` fixed anywhere in the program (named, positional numeric, or Bind)."""
    fixed: set[str] = set()
    defn = library.get(name)
    params = list(defn.params) if defn is not None else []
    for n in dsl.walk(root):
        if n.prim is None or n.prim.name != name:
            continue
        args = dict(n.bind_args) if n.op == "bind" else {}
        args.update(n.prim.args)
        pi = 0
        target_seen = False
        for k in sorted((k for k in args if k.startswith("_") and k[1:].isdigit()), key=lambda k: int(k[1:])):
            v = args[k]
            if isinstance(v, str) and not target_seen:
                target_seen = True
                continue
            if pi < len(params) and not isinstance(v, str):
                fixed.add(params[pi])
            pi += 1
        fixed |= {k for k, v in args.items() if k in (defn.params if defn else {}) and not isinstance(v, str)}
    return fixed


def _composite_motor_names(root, library: Mapping[str, PrimitiveDef], depth: int = 0) -> list[str]:
    out: list[str] = []
    for name in dsl.primitive_names(root):
        d = library.get(name)
        if d is not None and d.kind == "composite" and d.body is not None and depth < 8:
            out += _composite_motor_names(d.body.root, library, depth + 1)
        else:
            out.append(name)
    return out


def default_value(name: str, param: str, sim, cid: str, cfg: EpisodeConfig) -> float | None:
    """The value a motor primitive uses for ``param`` when the program leaves it out (D-009)."""
    from sea.control import motor

    fn = motor.MOTOR.get(name)
    if fn is not None:
        p = inspect.signature(fn).parameters.get(param)
        if p is not None and isinstance(p.default, (int, float)) and not isinstance(p.default, bool):
            return float(p.default)
    h = sim.controls[cid]
    travel = h.spec.travel
    if param == "force":
        return float(cfg.force_cap)
    if param == "speed":
        return {"push": 0.015, "hook_pull": 0.02, "slide": 0.02, "insert": 0.01}.get(name, cfg.limits.max_tip_speed_mps)
    if name == "grasp" and param == "depth":  # searched: a jaw with parts low over the object grips shallower (D-065)
        return motor.pick_depth(h.spec)
    if name == "push" and param == "depth":
        span = h.geo.lever_arm * travel if h.geo.rotary else travel
        return span * 1.25 + 3e-3
    if name == "rotate" and param == "angle":
        return travel + 0.08
    if name == "slide" and param == "distance":
        return travel * 1.15 + 7e-3
    if name == "insert" and param == "depth":
        return travel + 3e-3
    return None  # control-dependent budgets (hook_pull distance) are not searched


def modes_of(program: Program, library: Mapping[str, PrimitiveDef]) -> list[tuple[str, dict[str, float]]]:
    """The program's motor primitives' other ways to be used, as (primitive, values), in the library's order (D-102)."""
    names = dict.fromkeys(_composite_motor_names(program.root, library))
    return [(n, dict(m)) for n in names if (d := library.get(n)) is not None for m in d.modes]


def _reparam_for(name: str, reparam: Mapping[str, tuple[float, float]] | None) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key, rng in (reparam or {}).items():
        prim, _, param = key.rpartition(".")
        if prim in ("", name):
            out[param] = (float(rng[0]), float(rng[1]))
    return out


@dataclass(frozen=True)
class Dim:
    param: str
    x0: float
    lo: float
    hi: float


def search_space(program: Program, name: str, library: Mapping[str, PrimitiveDef], sim, cid: str,
                 cfg: EpisodeConfig, max_param_delta: float,
                 reparam: Mapping[str, tuple[float, float]] | None = None) -> list[Dim]:
    """Free continuous parameters of ``name`` with their search bounds (empty = nothing to tune)."""
    defn = library.get(name)
    if defn is None or not defn.params:
        return []
    fixed = _explicit_args(program.root, name, library)
    ranges = {**defn.params, **_reparam_for(name, reparam)}
    dims: list[Dim] = []
    for param, (lo, hi) in ranges.items():
        if param in fixed or param not in defn.params:
            continue
        half = max_param_delta * (hi - lo)  # of the declared (or System 2) range
        hi = min(hi, cfg.force_cap, cfg.limits.max_force_n) if param in ("force", "normal_force") else hi
        hi = min(hi, cfg.limits.max_tip_speed_mps) if param == "speed" else hi
        x0 = cfg.param_overrides.get(name, {}).get(param, default_value(name, param, sim, cid, cfg))
        if x0 is None or hi <= lo:
            continue
        x0 = float(np.clip(x0, lo, hi))
        b_lo, b_hi = max(lo, x0 - half), min(hi, x0 + half)
        if b_hi - b_lo > 1e-12:
            dims.append(Dim(param, x0, b_lo, b_hi))
    return dims


def pick_primitive(program: Program, trace: StepTrace, library: Mapping[str, PrimitiveDef]) -> str | None:
    """The failing primitive, else the first actuating primitive with parameters, else None."""
    names = _composite_motor_names(program.root, library)
    if trace.failed_primitive and library.get(trace.failed_primitive) is not None \
            and library[trace.failed_primitive].params:
        return trace.failed_primitive
    for n in names:
        d = library.get(n)
        if d is not None and d.provides and d.params:
            return n
    return None


# ---------------------------------------------------------------- objective
def progress(trace: StepTrace, travel: float) -> float:
    """Fraction of the required travel reached (max over the trace)."""
    if not trace.control_q or travel <= 0:
        return 0.0
    return float(max(abs(q - trace.q_rest) for q in trace.control_q) / travel)


def objective(result: ControlResult, trace: StepTrace, travel: float) -> float:
    """Lower is better: pass = -10, otherwise minus the travel fraction, structural verdicts last."""
    if result.verdict == "pass":
        return -10.0
    penalty = 5.0 if result.verdict in STRUCTURAL else 0.0
    return penalty - min(progress(trace, travel), 1.5)


def relocalized_offset(sim, cid: str, cfg: EpisodeConfig, rng: np.random.Generator) -> np.ndarray:
    """Fixed retry: a fresh perception draw (if noise is on) plus 1 mm toward the control centre."""
    h = sim.controls[cid]
    noisy = bool(np.any(np.asarray(cfg.perception_offset) != 0.0))
    off = rng.normal(0.0, PERCEPTION_SIGMA_M, 3) if noisy else np.zeros(3)
    n = h.T_world_ctl[:3, 2]
    off = off - n * float(off @ n)  # re-localize in the panel plane
    contact = sim.control_point_world(cid, h.geo.contact_body)
    to_centre = h.T_world_ctl[:3, 3] - contact
    to_centre = to_centre - n * float(to_centre @ n)
    norm = float(np.linalg.norm(to_centre))
    if norm > 1e-6:
        off = off + to_centre / norm * min(RETRY_OFFSET_M, norm)
    return off


# ---------------------------------------------------------------- System 1
def run_system1(sim, program: Program, control_id: str, cfg: EpisodeConfig,
                judge_fn: Callable[[StepTrace], ControlResult], *, m: int, max_param_delta: float, seed: int,
                panel_id: str, reparam: dict[str, tuple[float, float]] | None = None) -> S1Outcome:
    """First attempt, the fixed retry, then CMA-ES on the failing primitive (see module doc)."""
    rng = np.random.default_rng(seed)
    travel = sim.controls[control_id].spec.travel
    attempt_no = [0]
    history: list[tuple[dict, str]] = []

    def attempt(c: EpisodeConfig, tag: dict) -> tuple[StepTrace, ControlResult]:
        reset_episode(sim)
        tr = run_program(sim, program, control_id, c, panel_id=panel_id, attempt=attempt_no[0])
        attempt_no[0] += 1
        res = judge_fn(tr)
        history.append((tag, res.verdict))
        return tr, res

    def done(tr: StepTrace, res: ControlResult, trials: int, params: dict, escalate: str | None,
             searched: str = "") -> S1Outcome:
        tr.s1_trials = trials
        res = res.model_copy(update={"s1_trials": trials})
        return S1Outcome(res, tr, trials, params, history, escalate, searched)

    tr, res = attempt(cfg, {})
    if res.verdict == "pass":
        return done(tr, res, 0, {}, None)
    if res.verdict in STRUCTURAL:
        return done(tr, res, 0, {}, "structural")

    retry_cfg = replace(cfg, perception_offset=relocalized_offset(sim, control_id, cfg, rng))
    tr1, res1 = attempt(retry_cfg, {"retry": [round(float(v), 6) for v in retry_cfg.perception_offset]})
    if res1.verdict == "pass":
        return done(tr1, res1, 0, {}, None)
    if res1.verdict in STRUCTURAL:
        return done(tr1, res1, 0, {}, "structural")
    best = min(((tr, res, {}), (tr1, res1, {})), key=lambda b: objective(b[1], b[0], travel))

    lib = cfg.library
    name = pick_primitive(program, tr, lib)
    dims = search_space(program, name, lib, sim, control_id, cfg, max_param_delta, reparam) if name else []
    if not dims or m <= 0:
        return done(best[0], best[1], 0, best[2], "s1_exhausted", name or "")

    import cma

    lo = np.array([d.lo for d in dims])
    hi = np.array([d.hi for d in dims])
    x0 = (np.array([d.x0 for d in dims]) - lo) / (hi - lo)
    # pycma cannot rescale a 1-D search whose step size outgrows its limit (flat fitness raises in
    # tell); an extra, ignored coordinate keeps it on the N >= 2 path (docs/decisions.md D-027).
    pad = [0.5] if len(dims) == 1 else []
    es = cma.CMAEvolutionStrategy(x0.tolist() + pad, SIGMA0, {
        "bounds": [0.0, 1.0], "seed": int(seed) % (2**31 - 2) + 1, "verbose": -9, "maxfevals": m,
        "popsize": max(2, min(m, 4 + int(3 * np.log(len(dims))))),
    })
    trials = 0
    while trials < m:
        xs = es.ask()
        fs: list[float] = []
        for x in xs:
            if trials >= m:
                break
            vals = lo + np.clip(np.asarray(x)[: len(dims)], 0.0, 1.0) * (hi - lo)
            over = {d.param: round(float(v), 6) for d, v in zip(dims, vals, strict=True)}
            params = {**cfg.param_overrides, name: {**cfg.param_overrides.get(name, {}), **over}}
            c = replace(cfg, param_overrides=params)
            trx, resx = attempt(c, {name: over})
            trials += 1
            f = objective(resx, trx, travel)
            fs.append(f)
            if f < objective(best[1], best[0], travel):
                best = (trx, resx, params)
            if resx.verdict == "pass":
                return done(trx, resx, trials, params, None, name)
        if len(fs) < len(xs):
            break
        es.tell(xs, fs)
        if es.stop():
            break
    return done(best[0], best[1], trials, best[2], "s1_exhausted", name)
