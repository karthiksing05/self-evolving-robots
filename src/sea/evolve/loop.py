"""The evolution loop over any world (docs/evolve.md, D-041).

Each iteration:

1. **Evaluate.** System 1 solves every train task with the current body (cached by body and task).
2. **Stop** when every group is held, or the iterations run out.
3. **Propose.** System 2 (or the random baseline) sees the failures with their mechanisms, the brief,
   the catalogue, the resources and the record, and returns up to three body changes.
4. **Screen**, before any simulation: the edit must fit the catalogue, the world must be able to build
   the body, and the resource check must pass.  Ideas are recorded with what they would need.
5. **Trial.** Each admitted body solves the held groups first (stopping at the first loss), then the
   failing ones.
6. **Merge** kept-eligible proposals that change disjoint slots into one body, trialled once: a gain
   in one slot and a gain in another are not alternatives (D-045).
7. **Tune** the most promising new part type (a swap that gained a group but broke a held one, and gained
   more groups than the best kept-eligible candidate) for up to ``tune_trials`` more trials, before it is
   judged against a body whose parameters have had several iterations of tuning.  A structural change is
   protected until its own parameters are fitted (D-044).
8. **Accept** the best candidate that gains a group and loses none: the held set only grows (a ratchet,
   the PASS_TO_PASS rule).  Its purchases and fabrication are charged.

Everything is written to ``out``: every prompt and answer, every screen, trial and decision, so a run
reads back as a case study.  The final body and the stock body are both scored on the test split.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from sea.evolve import concepts as C
from sea.evolve import operation as OP
from sea.evolve import requirements as R
from sea.evolve import research as RS
from sea.evolve.core import (
    Body,
    Edit,
    OperatingPlan,
    Outcome,
    Proposal,
    ProposalBatch,
    Task,
    World,
    apply,
    edit_rung,
)
from sea.evolve.llm import Proposer
from sea.evolve.prompt import SYSTEM, build
from sea.evolve.resources import ResourceReport, charge, check, cost_idea, idea_text, start
from sea.propose.llm import PlanBillingError, PlanLimitReached

Arm = Literal["s2", "random", "s1"]


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world: str
    arm: Arm = "s2"
    iterations: int = 10
    seed: int = 0
    max_proposals: int = 3
    full_trials: bool = False  # score every group even after a loss; swaps always are (the frontier needs it)
    tune_trials: int = 4  # per iteration: trials spent fitting a promising swap's parameters before judging it
    merge: bool = True  # trial the union of kept-eligible proposals that change disjoint slots
    workers: int = 1
    budget: dict[str, float] = {}  # overrides the manifest's budget
    model: str = "claude-opus-5-5"
    effort: str = "high"
    tools: bool = False  # System 2 may query the harness through the ``sea`` tools while it proposes (D-086)
    requirements: bool = False  # System 2 keeps a requirements list the harness measures (D-089)
    research: bool = False  # System 2 searches and reads the web and keeps the notebook; needs the tools (D-090)
    notebook: str = "library/notebook.jsonl"  # shared across runs and worlds
    concepts: bool = False  # the concept set, functions, computed comparison and failure modes (D-091)
    plans: bool = False  # operating plans, and whether the operation or the body failed (D-092)


# ---------------------------------------------------------------- parallel evaluation
_WORLDS: dict[str, Callable[[], World]] = {}


def register(name: str, factory: Callable[[], World]) -> None:
    _WORLDS[name] = factory


def make_world(name: str) -> World:
    if name not in _WORLDS:
        import sea.worlds  # noqa: F401  (registers the built-in worlds)
    return _WORLDS[name]()


_worker_world: dict[str, World] = {}


def _solve_one(args: tuple[str, str, str, int]) -> str:
    wname, body_json, task_json, seed = args
    w = _worker_world.get(wname) or make_world(wname)
    _worker_world[wname] = w
    out = w.solve(Body.model_validate_json(body_json), Task.model_validate_json(task_json), seed)
    return out.model_dump_json()


class Evaluator:
    """Solves (body, task) pairs, in a process pool when ``workers`` > 1, and caches every result."""

    def __init__(self, world: World, wname: str, workers: int = 1, seed: int = 0):
        self.world, self.wname, self.workers, self.seed = world, wname, workers, seed
        self.cache: dict[tuple[str, str], Outcome] = {}
        self.sim_s = 0.0
        self.attempts = 0
        self._pool: ProcessPoolExecutor | None = None

    def pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    def solve(self, body: Body, tasks: Sequence[Task]) -> dict[str, Outcome]:
        todo = [t for t in tasks if (body.key(), t.id) not in self.cache]
        if self.workers > 1 and len(todo) > 1:
            args = [(self.wname, body.model_dump_json(), t.model_dump_json(), self.seed) for t in todo]
            results = [Outcome.model_validate_json(s) for s in self.pool().map(_solve_one, args)]
        else:
            results = [self.world.solve(body, t, self.seed) for t in todo]
        for t, o in zip(todo, results, strict=True):
            self.cache[(body.key(), t.id)] = o
            self.sim_s += o.sim_s
            self.attempts += o.attempts
        return {t.id: self.cache[(body.key(), t.id)] for t in tasks}


def held_groups(tasks: Sequence[Task], outcomes: dict[str, Outcome]) -> set[str]:
    groups: dict[str, list[bool]] = {}
    for t in tasks:
        groups.setdefault(t.group, []).append(outcomes[t.id].success)
    return {g for g, ok in groups.items() if all(ok)}


# ---------------------------------------------------------------- the random baseline
class RandomProposer:
    """Blind body changes from the catalogue with the same screen and trial budget: what reasoning buys."""

    def __init__(self, world: World, seed: int = 0):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.calls: list[dict[str, Any]] = []
        self.body: Body | None = None

    def propose(self, system: str, prompt: str, tag: str = "") -> ProposalBatch:
        assert self.body is not None
        self.calls.append({"tag": tag, "tokens_in": 0, "tokens_out": 0, "usd_list_price": 0.0, "billing": "none"})
        cat = self.world.catalogue()
        out = []
        for _ in range(3):
            slot = str(self.rng.choice(sorted({p.slot for p in cat})))
            types = [p for p in cat if p.slot == slot]
            cur = self.body.parts.get(slot)
            if self.rng.random() < 0.5 and cur is not None and [p for p in types if p.name == cur.type][0].params:
                pt = next(p for p in types if p.name == cur.type)
                keys = list(pt.params)
                pick = self.rng.choice(keys, size=min(len(keys), int(self.rng.integers(1, 3))), replace=False)
                edit = Edit(slot=slot, params={str(k): float(self.rng.uniform(pt.params[k].lo, pt.params[k].hi))
                                               for k in pick})
                move = "params"
            else:
                pt = types[int(self.rng.integers(len(types)))]
                edit = Edit(slot=slot, type=pt.name, params={k: float(self.rng.uniform(p.lo, p.hi))
                                                             for k, p in pt.params.items()})
                move = "swap"
            out.append(Proposal(move=move, edits=[edit], mechanism="random", principle="random",  # type: ignore[arg-type]
                                hypothesis="random baseline"))
        return ProposalBatch(proposals=out)


# ---------------------------------------------------------------- the loop
def _edit_text(edits: Sequence[Edit]) -> str:
    return "; ".join(f"{e.slot}" + (f" -> {e.type}" if e.type else "") +
                     (" (" + ", ".join(f"{k}={v:g}" for k, v in sorted(e.params.items())) + ")" if e.params else "")
                     for e in edits)


def _score_prediction(predicted: Sequence[str], gained: Sequence[str], lost: Sequence[str]) -> dict[str, Any]:
    """How the proposal's predicted gains compare with what the trial found (groups)."""
    p, g = set(predicted), set(gained)
    return {"predicted": sorted(p), "hit": sorted(p & g), "missed": sorted(p - g), "unpredicted": sorted(g - p),
            "precision": round(len(p & g) / len(p), 3) if p else None, "lost": sorted(lost)}


def points_line(world: World, got: tuple[float, dict[str, float]]) -> str:
    """The score against the brief's bars, for the prompt."""
    total, parts = got
    bars = ", ".join(f"{b:g}" for b in getattr(world, "sprint_bars", ()))
    return (f"The current body scores {total:.1f} of {getattr(world, 'max_points', 0.0):g} points in a match"
            + (f" (the brief's bars: {bars})" if bars else "") + ": "
            + ", ".join(f"{k.split('.', 1)[1]} {v:.1f}" for k, v in sorted(parts.items())))


def _proposal_stats(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tried = [h for h in history if h.get("tried")]
    refused = [h for h in history if h.get("result", "").startswith("refused")]
    prec = [h["prediction"]["precision"] for h in tried if h.get("prediction", {}).get("precision") is not None]
    return {"total": len(history), "tried": len(tried), "refused": len(refused),
            "ideas": sum(h["result"].startswith("idea") for h in history),
            "kept": sum(h["result"].startswith("KEPT") for h in history),
            "rejected_regression": sum(bool(h.get("lost")) for h in tried),
            "moves": {m: sum(h.get("move") == m for h in history) for m in ("params", "swap", "idea", "tune", "merge")},
            "mean_prediction_precision": round(sum(prec) / len(prec), 3) if prec else None}


def _diff(a: Body, b: Body) -> list[Edit]:
    """The edits that turn ``a`` into ``b``."""
    return [Edit(slot=s, type=p.type, params=p.params) for s, p in b.parts.items() if a.parts.get(s) != p]


def _merge(body: Body, ranked: Sequence[Body], it: int) -> tuple[Body, list[str]] | None:
    """The best candidate plus every other one whose changed slots are disjoint from those taken so far, if that
    joins at least two: the parts of each, by slot."""
    parts, taken, ids = dict(body.parts), set(), []
    for cand in ranked:
        slots = {e.slot for e in _diff(body, cand)}
        if slots & taken:
            continue
        parts.update({s: cand.parts[s] for s in slots})
        taken |= slots
        ids.append(cand.id)
    if len(ids) < 2:
        return None
    plan = next((c.plan for c in ranked if c.id in ids and c.plan is not None), body.plan)
    return Body(id=f"b{it:02d}m", parent=body.id, parts=parts, plan=plan), ids


# ---------------------------------------------------------------- one path from proposal to trial, shared with System 2's tools (D-086)
def builds(world: World, prop: Proposal) -> bool:
    """An idea that carries a mechanism is built, with any edits to its other slots (D-083)."""
    return prop.idea is not None and prop.idea.mechanism is not None and hasattr(world, "build_idea")


def recorded_only(world: World, prop: Proposal) -> bool:
    """An idea the loop records with what it would need, and does not try."""
    return (prop.move == "idea" or prop.idea is not None and not prop.edits) and not builds(world, prop)


def candidate(world: World, body: Body, bases: dict[str, Body], prop: Proposal, catalogue: dict[str, Any],
              new_id: str) -> Body:
    """The body ``prop`` describes, on the current body or a promising one in ``bases``.  Raises ``ValueError``
    naming what does not apply."""
    if prop.base is not None and prop.base not in bases:
        raise ValueError(f"no promising body {prop.base!r} in the record; ids: {sorted(bases)}")
    base = body if prop.base is None else bases[prop.base]
    cand = _built(world, base, prop, catalogue, new_id)
    plan = prop.plan if prop.plan is not None else base.plan
    if plan is None:
        return cand
    names = world.control_names(cand) if hasattr(world, "control_names") else {}
    for k, v in plan.controls.items():
        if k not in names:
            raise ValueError(f"the plan sets {k!r}, which is not a control here; the controls are {sorted(names)}")
        if not names[k][0] - 1e-9 <= v <= names[k][1] + 1e-9:
            raise ValueError(f"the plan sets {k}={v:g}, outside its range [{names[k][0]:g}, {names[k][1]:g}]")
    return cand.model_copy(update={"plan": plan})


def _built(world: World, base: Body, prop: Proposal, catalogue: dict[str, Any], new_id: str) -> Body:
    if not builds(world, prop):
        return apply(base, prop.edits, catalogue, new_id=new_id)
    slot = prop.idea.slot  # a mechanism the catalogue lacks, built from the world's modules (D-079)
    if slot and any(e.slot == slot for e in prop.edits):
        raise ValueError(f"the idea builds slot {slot!r}; an edit to it as well is ambiguous: "
                         "edit the other slots, or put the change in the mechanism")
    cand = world.build_idea(base, prop.idea, new_id=new_id)
    return apply(cand, prop.edits, catalogue, new_id=cand.id) if prop.edits else cand


def opens_concept(world: World, concepts: C.ConceptSet, body: Body, bases: dict[str, Body], prop: Proposal,
                  cand: Body) -> bool:
    """Whether this proposal starts a new structure rather than developing one (D-091)."""
    built = builds(world, prop)
    base = bases.get(prop.base, body) if prop.base is not None else body
    return C.opens(built, prop.idea.mechanism.module if built else None, edit_rung(base, _diff(base, cand)), prop.base,
                   [c.id for c in concepts.live()])


def built_text(prop: Proposal) -> str:
    mech = prop.idea.mechanism
    return "; ".join([f"{prop.idea.slot or 'mechanism'}: built {prop.idea.name!r} ({mech.module} + "
                      f"{len(mech.patch)} changes)", *([_edit_text(prop.edits)] if prop.edits else [])])


def screen(world: World, cand: Body, body: Body, manifest: dict[str, Any], spent: Any,
           tried: dict[str, str]) -> tuple[list[str], ResourceReport]:
    """What refuses ``cand`` before its trial: the world's build rules and laws, the resources, and the record."""
    errs = world.check(cand)
    res = check(cand, world.bom(cand), body, world.bom(body), manifest, spent)
    if cand.key() == body.key():
        errs = ["the edit leaves the body unchanged", *errs]
    elif cand.key() in tried:
        errs = [f"this exact body was already trialled: {tried[cand.key()]}", *errs]
    return [*errs, *res.errors] if errs or not res.ok else [], res


def _tune_key(entry: dict[str, Any]) -> tuple[float, ...]:
    """Tuning's objective: kept-eligible first, then points where the world counts them, then net groups, then tasks
    newly passed, then partial credit."""
    return (float(entry["eligible"] if "eligible" in entry else bool(entry["gained"]) and not entry["lost"]),
            entry.get("points_gain", 0.0), len(entry["gained"]) - len(entry["lost"]), entry["new_tasks"],
            entry.get("score_sum", 0.0))


def points_of(tasks: Sequence[Task], outcomes: dict[str, Outcome]) -> tuple[float, dict[str, float]]:
    """The mean points over the tasks that carry them, and of each part of that total (D-093)."""
    scored = [outcomes[t.id] for t in tasks if t.id in outcomes and outcomes[t.id].points is not None]
    if not scored:
        return 0.0, {}
    parts = {k: sum(o.states.get(k, 0.0) for o in scored) / len(scored)
             for o in scored for k in o.states if k.startswith("points.")}
    return sum(o.points for o in scored) / len(scored), parts


def run(cfg: RunConfig, out: Path, proposer: Proposer | None = None, world: World | None = None,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    world = world or make_world(cfg.world)
    manifest = world.manifest()
    if cfg.budget:
        manifest = {**manifest, "budget": {**manifest.get("budget", {}), **cfg.budget}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    from sea.harness import fingerprint, judge_key

    fp = fingerprint()
    (out / "harness.json").write_text(json.dumps({"judge_key": judge_key(fp), "components": fp}, indent=1))
    catalogue = {p.name: p for p in world.catalogue()}
    all_tasks = world.tasks()
    train = [t for t in all_tasks if t.split == "train"]
    test = [t for t in all_tasks if t.split == "test"]
    if proposer is None and cfg.arm == "s2":
        from sea.evolve.llm import PlanClaude

        proposer = PlanClaude(cfg.model, cfg.effort, log_dir=out / "llm")
    if proposer is None and cfg.arm == "random":
        proposer = RandomProposer(world, cfg.seed)
    ev = Evaluator(world, cfg.world, cfg.workers, cfg.seed)
    body = world.stock()
    spent = start(world.bom(body))
    history: list[dict[str, Any]] = []
    reqs: list[Any] = []  # System 2's requirements list (D-089)
    notebook = Path(cfg.notebook)
    concepts = C.ConceptSet()  # new structures, developed before they are judged (D-091)
    functions: list[Any] = []  # System 2's function list
    cur_rows: list[Any] = []  # the list measured on the current body
    ideas: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    tried: dict[str, str] = {}  # body key -> what its trial found: a body is never paid for twice
    frontier: dict[str, dict[str, Any]] = {}  # promising bodies not kept: they gained more groups than they lost
    t_start = time.time()

    by_points = getattr(world, "objective", "groups") == "points"
    noise = float(getattr(world, "points_noise", 0.0))

    def bases() -> dict[str, Body]:
        """The bodies a proposal may build on by id: the promising ones, and the live concepts (D-091)."""
        return {**{i: f["body"] for i, f in frontier.items()}, **concepts.bases()}

    def trial(cand: Body, entry: dict[str, Any], full_always: bool) -> tuple[float, ...] | None:
        """Screen ``cand`` against the current body and, if admitted, solve it, filling ``entry``.  Returns its
        acceptance score if it gains a group and loses none."""
        errs, res = screen(world, cand, body, manifest, spent, tried)
        entry["resources"] = res.model_dump()
        if cfg.requirements and reqs:  # what the candidate does to each requirement, measured before any trial
            entry["requirements"] = R.moved(cur_rows, R.measure(world, cand, reqs, train, body, manifest, spent))
        if errs:
            entry.update(result="refused before trial: " + "; ".join(errs), tried=False)
            if cfg.concepts:  # a law's refusal leaves a structure to develop; money or a repeat does not
                entry["laws"] = world.check(cand)
            return None
        held_tasks = [t for t in train if t.group in held]
        cand_out = ev.solve(cand, held_tasks)
        lost = sorted({t.group for t in held_tasks if not cand_out[t.id].success})
        full = not lost or cfg.full_trials or full_always or by_points
        if full:
            cand_out.update(ev.solve(cand, [t for t in train if t.group not in held]))
        cand_held = held_groups(train, cand_out) if full else held - set(lost)
        gained = sorted(cand_held - held)
        n_new = sum(cand_out[t.id].success for t in train if t.group not in held) if full else 0
        still = sorted({cand_out[t.id].mechanism for t in train if t.id in cand_out and not cand_out[t.id].success})
        entry.update(tried=True, gained=gained, lost=lost, new_tasks=n_new, still_failing=still,
                     score_sum=round(sum(o.score for o in cand_out.values()), 3),
                     outcomes={t: o.model_dump() for t, o in cand_out.items()})
        why = (f"lost {', '.join(lost)} (" + "; ".join(f"{t.group}: {cand_out[t.id].mechanism}, "
                                                       f"{cand_out[t.id].detail[:140]}"
                                                       for t in held_tasks if t.group in lost
                                                       and not cand_out[t.id].success)[:600] + ")") if lost else ""
        score = None
        if by_points:  # the score decides; the groups are what it shows (D-093)
            now, now_parts = points_of(train, outcomes)
            new, new_parts = points_of(train, cand_out)
            drops = sorted(k for k, v in now_parts.items() if new_parts.get(k, 0.0) < v - noise)
            entry.update(points=round(new, 2), points_gain=round(new - now, 2), points_lost=drops,
                         eligible=new > now and not drops)
            told = f"{now:.1f} -> {new:.1f} points"
            if entry["eligible"]:
                entry["result"] = (f"kept-eligible: {told}; costs ${res.usd:.2f}, {res.fab_minutes:.0f} machine min")
                score = (new - now, n_new, -res.usd - res.fab_minutes / 60,
                         -int(edit_rung(body, _diff(body, cand)) == "swap"))
            elif new > now:
                entry["result"] = f"rejected: {told}, but {', '.join(drops)} dropped more than {noise:g}"
                frontier[cand.id] = {"id": cand.id, "body": cand, "gained": gained or [told], "lost": drops,
                                     "why_lost": entry["result"], "iteration": it, "edit": entry["edit"]}
                entry["result"] += f"; kept on the frontier as {cand.id}"
            else:
                entry["result"] = f"rejected: {told}; still failing: " + ", ".join(still[:6])
        elif lost:
            entry["result"] = f"rejected: {why}" + (f"; it would gain {', '.join(gained)}" if gained else "")
            if len(gained) > len(lost):
                frontier[cand.id] = {"id": cand.id, "body": cand, "gained": gained, "lost": lost,
                                     "why_lost": why, "iteration": it, "edit": entry["edit"]}
                entry["result"] += f"; kept on the frontier as {cand.id}"
        elif not gained and n_new == 0:
            entry["result"] = "rejected: gained nothing; still failing: " + ", ".join(still[:6])
        else:
            entry["result"] = (f"kept-eligible: gained {', '.join(gained) or f'{n_new} tasks'}, lost none; "
                               f"costs ${res.usd:.2f}, {res.fab_minutes:.0f} machine min")
            score = (len(gained), n_new, -res.usd - res.fab_minutes / 60,
                     -int(edit_rung(body, _diff(body, cand)) == "swap"))
        tried[cand.key()] = f"iteration {it + 1}: {entry['result'][:200]}"
        return score

    def tune(seed: Body, seed_entry: dict[str, Any], rng: np.random.Generator):
        """Fit the parameters of the slots a promising swap edited: a Gaussian step in each parameter's range,
        centred on the best body so far, until one is kept-eligible or ``tune_trials`` trials have run.  Each step
        moves one or two parameters: a swap's values are System 2's reasoned guess, and moving all of them at
        once breaks a different held group each time (gripper_s2v3_s0, iteration 1)."""
        space = [(slot, k, catalogue[seed.parts[slot].type].params[k]) for slot in seed_entry["slots"]
                 for k in sorted(catalogue[seed.parts[slot].type].params)]
        if not space:
            return
        x = np.array([(seed.parts[s].params[k] - p.lo) / ((p.hi - p.lo) or 1.0) for s, k, p in space])
        best_x, best_key, sigma, ran = x, _tune_key(seed_entry), 0.25, 0
        for i in range(3 * cfg.tune_trials):
            if ran >= cfg.tune_trials:
                break
            step = np.zeros(len(space))
            pick = rng.choice(len(space), size=min(len(space), int(rng.integers(1, 3))), replace=False)
            step[pick] = rng.normal(0.0, sigma, len(pick))
            xi = np.clip(best_x + step, 0.0, 1.0)
            edits: dict[str, dict[str, float]] = {}
            for (s, k, p), v in zip(space, xi, strict=True):
                edits.setdefault(s, {})[k] = round(float(p.lo + v * (p.hi - p.lo)), 3)
            ed = [Edit(slot=s, params=ps) for s, ps in edits.items()]
            cand = apply(seed, ed, catalogue, new_id=f"b{it:02d}t{i}").model_copy(update={"plan": seed.plan})
            entry: dict[str, Any] = {"k": f"t{i}", "iteration": it, "move": "tune", "base": seed.id,
                                     "slots": seed_entry["slots"], "edit": _edit_text(ed),
                                     "hypothesis": f"the loop fits {seed.id}'s new {', '.join(seed_entry['slots'])} "
                                                   f"before judging it (tuning trial {ran + 1} of {cfg.tune_trials})"}
            score = trial(cand, entry, full_always=True)
            if entry.get("tried"):
                ran += 1
                key = _tune_key(entry)
                if key > best_key:
                    best_x, best_key = xi, key
                sigma = max(0.05, sigma * 0.8)
            yield score, cand, entry
            if score is not None:
                return

    records = (out / "iterations.jsonl").open("w")
    try:
        for it in range(cfg.iterations + 1):
            t0 = time.time()
            outcomes = ev.solve(body, train)
            held = held_groups(train, outcomes)
            n_ok = sum(o.success for o in outcomes.values())
            curve.append({"iteration": it, "body": body.id, "held_groups": len(held), "tasks_passed": n_ok,
                          "sim_s": round(ev.sim_s, 1), "wall_s": round(time.time() - t_start, 1),
                          **({"points": round(points_of(train, outcomes)[0], 2)} if by_points else {})})
            log(f"[{cfg.world}/{cfg.arm}] iteration {it}: {len(held)}/{len({t.group for t in train})} groups, "
                f"{n_ok}/{len(train)} tasks, body {body.describe()}")
            rec: dict[str, Any] = {"iteration": it, "body": body.model_dump(), "held": sorted(held),
                                   "failing": {t.id: outcomes[t.id].model_dump() for t in train
                                               if not outcomes[t.id].success}}
            done = len(held) == len({t.group for t in train})
            if done or it == cfg.iterations or proposer is None:
                rec["stop"] = "all groups held" if done else ("no proposer" if proposer is None else "iterations")
                records.write(json.dumps(rec) + "\n")
                break
            if isinstance(proposer, RandomProposer):
                proposer.body = body
            front = sorted(frontier.values(), key=lambda f: (-(len(f["gained"]) - len(f["lost"])), f["iteration"]))
            budget, req_text, notes_text, concept_text = None, None, None, None
            if cfg.research:
                notes_text = RS.text(RS.load(notebook))
            groups = sorted({t.group for t in train})
            if cfg.requirements:
                cur_rows = R.measure(world, body, reqs, train, body, manifest, spent)
                req_text = R.text(reqs, cur_rows, held, groups)
            if cfg.concepts:
                concepts.refresh(reqs, partial(R.measure, world, reqs=reqs, tasks=train, base=body, manifest=manifest,
                                               spent=spent))
                concept_text = C.text(concepts, reqs, R.measure(world, body, reqs, train, body, manifest, spent), functions,
                                      {*catalogue, *(world.module_names() if hasattr(world, "module_names") else [])},
                                      getattr(world, "mechanism_text", None))
            if cfg.tools or cfg.research:  # what System 2's tools check a draft against: this iteration's body and record (D-086)
                from sea.evolve import tools as T

                ctx = out / "tools" / f"iter{it:02d}.json"
                ctx.parent.mkdir(exist_ok=True)
                ctx.write_text(T.Context(world=cfg.world, iteration=it, body=body, tried=tried, manifest=manifest,
                                         bases=bases(), spent=spent,
                                         concepts=concepts if cfg.concepts else None,
                                         train=[t.id for t in train], requirements=reqs,
                                         held=sorted(held), notebook=str(notebook.resolve()) if cfg.research else "",
                                         research_log=str((out / "research.jsonl").resolve()),
                                         run=out.name).model_dump_json())
                budget = {"calls": T.CALLS, "sims": T.SIMS}
                if hasattr(proposer, "tools"):
                    proposer.tools = (ctx, out / "tool_calls.jsonl")
                if cfg.research and hasattr(proposer, "research"):  # the guard on this iteration's research
                    guard = ctx.with_suffix(".research.json")
                    guard.write_text(RS.Config(blocked=list(getattr(world, "research_blocked", [])),
                                               log=str((out / "research.jsonl").resolve()), iteration=it).model_dump_json())
                    hooks = ctx.with_suffix(".settings.json")
                    hooks.write_text(json.dumps(RS.settings(guard)))
                    proposer.research = hooks
            prompt = build(world, body, train, outcomes, held, history, ideas, spent, manifest, it, cfg.iterations,
                           cfg.max_proposals, frontier=front[:5], tune_trials=cfg.tune_trials, tools=budget,
                           requirements=req_text, notebook=notes_text, concepts=concept_text,
                           points=points_line(world, points_of(train, outcomes)) if by_points else None,
                           plans=OP.text(sorted(world.control_names(body)) if hasattr(world, "control_names") else [])
                           if cfg.plans else None, current_plan=body.plan if cfg.plans else None)
            try:
                batch = proposer.propose(SYSTEM, prompt, tag=f"iter{it:02d}")
            except (PlanLimitReached, PlanBillingError):
                raise  # out of plan, or not on the plan: stop rather than run on without System 2
            except RuntimeError as exc:  # System 2 gave nothing usable: the iteration proposes nothing (D-082)
                log(f"  iter {it}: System 2 gave no proposals: {exc}")
                batch = ProposalBatch.model_construct(proposals=[], notes=f"System 2 gave no proposals: {exc}")
            rec["llm"] = proposer.calls[-1] if getattr(proposer, "calls", None) else {}
            if cfg.tools or cfg.research:
                rec["tools"] = T.summary(out / "tool_calls.jsonl", it)
            if cfg.research:
                rec["research"] = RS.summary(out / "research.jsonl", it)
            rec["notes"] = batch.notes
            if cfg.requirements and batch.requirements:  # System 2 revised its list: measure the new one
                reqs = list(batch.requirements)
                cur_rows = R.measure(world, body, reqs, train, body, manifest, spent)
                rec["requirements"] = [r.model_dump() for r in reqs]
                (out / "requirements.json").write_text(json.dumps(rec["requirements"], indent=1))
            if cfg.requirements:
                rec["requirement_rows"] = [r.model_dump() for r in cur_rows]
                rec["uncovered"] = R.coverage(reqs, groups)
            if cfg.concepts and batch.functions:
                functions = list(batch.functions)
                rec["functions"] = [f.model_dump() for f in functions]
            trials = []
            swaps: list[tuple[Body, dict[str, Any]]] = []
            eligible: list[tuple[tuple[float, ...], Body, dict[str, Any]]] = []
            best: tuple[tuple[float, ...], Body, dict[str, Any]] | None = None
            for k, prop in enumerate(batch.proposals[: cfg.max_proposals]):
                entry: dict[str, Any] = {"k": k, "proposal": prop.model_dump(), "iteration": it, "move": prop.move,
                                         "hypothesis": prop.hypothesis, "edit": _edit_text(prop.edits)}
                built = builds(world, prop)
                if recorded_only(world, prop):
                    idea = prop.idea.model_dump() if prop.idea else {"name": "?", "needs": []}
                    cost = cost_idea(prop.idea.needs if prop.idea else [], manifest, spent)
                    ideas.append({**idea, "iteration": it, "cost": cost})
                    entry.update(result=f"idea recorded, not tested: {idea['name']}; it would need {idea_text(cost)}",
                                 tried=False)
                    trials.append(entry)
                    history.append(entry)
                    continue
                opening = False
                try:
                    gone = concepts.base_refusal(prop.base) if cfg.concepts else ""
                    if gone:
                        raise ValueError(gone)
                    cand = candidate(world, body, bases(), prop, catalogue, new_id=f"b{it:02d}{k}")
                    if built:
                        entry["edit"] = built_text(prop)
                    if cfg.concepts:
                        opening = opens_concept(world, concepts, body, bases(), prop, cand)
                        full = concepts.full_refusal() if opening else ""
                        if full:
                            raise ValueError(full)
                        if cfg.plans and opening:  # a new structure is searched longer; what is developed from it inherits that
                            cand = cand.model_copy(update={"plan": (cand.plan or OperatingPlan()).model_copy(
                                update={"search": OP.SEARCH_NEW})})
                except ValueError as exc:
                    entry.update(result=f"refused before trial: {exc}", tried=False)
                    trials.append(entry)
                    history.append(entry)
                    continue
                change = _diff(body, cand)  # against the current body, also when the proposal builds on another
                entry["move"] = "idea" if built else edit_rung(body, change)
                entry["slots"] = sorted({e.slot for e in change})
                score = trial(cand, entry, full_always=entry["move"] in ("swap", "idea"))  # a redesign is scored whole
                if entry.get("tried"):
                    entry["prediction"] = _score_prediction(prop.gains, entry["gained"], entry["lost"])
                    if entry["move"] == "swap":
                        swaps.append((cand, entry))
                if cfg.plans and entry.get("tried"):
                    entry["operation"] = OP.diagnose(cand.plan, entry["outcomes"], {t.id: t.group for t in train})
                if cfg.concepts:
                    entry["failure_modes"], entry["unpredicted_failures"] = C.rate(prop.failure_modes, entry,
                                                                                   {t.id: t.group for t in train})
                    name = prop.idea.name if prop.idea is not None else entry["edit"]
                    cid = concepts.record(prop.base, opening, name, cand, entry, it, dict(prop.options))
                    if cid is not None:
                        entry["concept"] = cid
                if score is not None:
                    eligible.append((score, cand, entry))
                    if best is None or score > best[0]:
                        best = (score, cand, entry)
                trials.append(entry)
                history.append(entry)
                log(f"    proposal {k} [{entry['move']}] {entry['edit']}: {entry['result'][:160]}")
            if cfg.merge and len(eligible) > 1:
                merged = _merge(body, [c for _, c, _ in sorted(eligible, key=lambda x: x[0], reverse=True)], it)
                if merged is not None:
                    mbody, parts = merged
                    entry = {"k": "m", "iteration": it, "move": "merge", "edit": _edit_text(_diff(body, mbody)),
                             "slots": sorted({e.slot for e in _diff(body, mbody)}), "merged": parts,
                             "hypothesis": f"the union of kept-eligible proposals {', '.join(parts)}, which change "
                                           "disjoint slots"}
                    score = trial(mbody, entry, full_always=True)
                    trials.append(entry)
                    history.append(entry)
                    log(f"    merge of {', '.join(parts)}: {entry['edit']}: {entry['result'][:160]}")
                    if score is not None and (best is None or score > best[0]):
                        best = (score, mbody, entry)
            # worth tuning only if it already gains more groups than the best candidate this iteration keeps
            promising = ([(c, e) for c, e in swaps if e["points_gain"] > 0 and e["points_lost"]
                          and (best is None or e["points_gain"] > best[0][0])] if by_points else
                         [(c, e) for c, e in swaps if e["gained"] and e["lost"]
                          and (best is None or len(e["gained"]) > best[0][0])])
            if promising and cfg.tune_trials > 0:
                cand, entry = max(promising, key=lambda ce: (ce[1].get("points_gain", 0.0),
                                                              len(ce[1]["gained"]) - len(ce[1]["lost"]),
                                                              len(ce[1]["gained"]), ce[1]["new_tasks"]))
                for score, tcand, tentry in tune(cand, entry, np.random.default_rng([cfg.seed, it])):
                    if "concept" in entry:
                        tentry["concept"] = entry["concept"]
                    trials.append(tentry)
                    history.append(tentry)
                    log(f"    tune {tentry['k']} of {cand.id}: {tentry['edit']}: {tentry['result'][:160]}")
                    if score is not None and (best is None or score > best[0]):
                        best = (score, tcand, tentry)
            accepted = None
            if best is not None:
                _, cand, entry = best
                spent = charge(spent, ResourceReport.model_validate(entry["resources"]))
                entry["result"] = entry["result"].replace("kept-eligible", "KEPT")
                body = cand
                accepted = cand.id
                if cfg.research:  # the notes the kept change cites earn their credit
                    entry["credited"] = RS.credit(notebook, json.dumps(entry.get("proposal", {})))
                if cfg.concepts:
                    concepts.keep([entry.get("concept", ""), *entry.get("merged", [])])
            for e in trials:
                if e.get("result", "").startswith("kept-eligible"):
                    e["result"] = e["result"].replace("kept-eligible", "not kept (another proposal gained more)")
            if cfg.concepts:  # narrow on this iteration's evidence, measured against the body it ends with
                concepts.refresh(reqs, partial(R.measure, world, reqs=reqs, tasks=train, base=body, manifest=manifest,
                                               spent=spent))
                rec["concepts_dropped"] = concepts.narrow(it, reqs)
                rec["concepts"] = C.summary(concepts)
                (out / "concepts.json").write_text(concepts.model_dump_json(indent=1))
            rec.update(trials=[{k: v for k, v in e.items() if k != "outcomes"} | {"outcomes": e.get("outcomes", {})}
                               for e in trials], accepted=accepted, spent=spent.model_dump(),
                       seconds=round(time.time() - t0, 1))
            records.write(json.dumps(rec) + "\n")
            records.flush()
        final_test = ev.solve(body, test) if test else {}
        stock_test = ev.solve(world.stock(), test) if test else {}
    finally:
        records.close()
        ev.close()
    summary = {
        "world": cfg.world, "arm": cfg.arm, "seed": cfg.seed, "final_body": body.model_dump(), "curve": curve,
        "train_groups": len({t.group for t in train}), "final_held_train": sorted(held_groups(train, ev.solve(body, train))),
        "test": {"final_held": sorted(held_groups(test, final_test)) if test else [],
                 "stock_held": sorted(held_groups(test, stock_test)) if test else [],
                 "final_tasks": sum(o.success for o in final_test.values()),
                 "stock_tasks": sum(o.success for o in stock_test.values()), "n": len(test)},
        "spent": spent.model_dump(), "ideas": ideas, "sim_s": round(ev.sim_s, 1), "attempts": ev.attempts,
        "proposals": _proposal_stats(history),
        "frontier": [{k: (v.describe() if k == "body" else v) for k, v in f.items()} for f in frontier.values()],
        "wall_s": round(time.time() - t_start, 1),
        "llm": {"calls": len(getattr(proposer, "calls", []) or []),
                "tokens_in": sum(c.get("tokens_in", 0) for c in getattr(proposer, "calls", []) or []),
                "tokens_out": sum(c.get("tokens_out", 0) for c in getattr(proposer, "calls", []) or []),
                "usd_list_price": round(sum(c.get("usd_list_price", 0.0) for c in getattr(proposer, "calls", []) or []), 3)},
    }
    if by_points:
        by_task = {t.id: ev.solve(body, [t])[t.id] for t in test} if test else {}
        held_out = points_of(test, by_task)[0]
        summary["points"] = {"train": round(points_of(train, ev.solve(body, train))[0], 2), "test": round(held_out, 2),
                             "test_by_task": {t: o.points for t, o in by_task.items() if o.points is not None},
                             "stock_test": round(points_of(test, stock_test)[0], 2),
                             "bars_met_test": [b for b in getattr(world, "sprint_bars", ()) if held_out >= b]}
    if cfg.research:
        summary["research"] = RS.summary(out / "research.jsonl")
    if cfg.concepts:
        summary["concepts"] = C.summary(concepts)
    if cfg.tools or cfg.research:
        from sea.evolve.tools import summary as tool_summary

        summary["tools"] = tool_summary(out / "tool_calls.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
