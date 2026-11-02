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
6. **Accept** the best candidate that gains a group and loses none: the held set only grows (a ratchet,
   the PASS_TO_PASS rule).  Its purchases and fabrication are charged.

Everything is written to ``out``: every prompt and answer, every screen, trial and decision, so a run
reads back as a case study.  The final body and the stock body are both scored on the test split.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from sea.evolve.core import Body, Edit, Outcome, Proposal, ProposalBatch, Task, World, apply, edit_rung
from sea.evolve.llm import Proposer
from sea.evolve.prompt import SYSTEM, build
from sea.evolve.resources import Spent, check

Arm = Literal["s2", "random", "s1"]


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world: str
    arm: Arm = "s2"
    iterations: int = 10
    seed: int = 0
    max_proposals: int = 3
    workers: int = 1
    budget: dict[str, float] = {}  # overrides the manifest's budget
    model: str = "claude-opus-5-5"
    effort: str = "high"


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


def run(cfg: RunConfig, out: Path, proposer: Proposer | None = None, world: World | None = None,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    world = world or make_world(cfg.world)
    manifest = world.manifest()
    if cfg.budget:
        manifest = {**manifest, "budget": {**manifest.get("budget", {}), **cfg.budget}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
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
    spent = Spent()
    history: list[dict[str, Any]] = []
    ideas: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    t_start = time.time()
    records = (out / "iterations.jsonl").open("w")
    try:
        for it in range(cfg.iterations + 1):
            t0 = time.time()
            outcomes = ev.solve(body, train)
            held = held_groups(train, outcomes)
            n_ok = sum(o.success for o in outcomes.values())
            curve.append({"iteration": it, "body": body.id, "held_groups": len(held), "tasks_passed": n_ok,
                          "sim_s": round(ev.sim_s, 1), "wall_s": round(time.time() - t_start, 1)})
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
            prompt = build(world, body, train, outcomes, held, history, ideas, spent, manifest, it, cfg.iterations,
                           cfg.max_proposals)
            batch = proposer.propose(SYSTEM, prompt, tag=f"iter{it:02d}")
            rec["llm"] = proposer.calls[-1] if getattr(proposer, "calls", None) else {}
            rec["notes"] = batch.notes
            trials = []
            best: tuple[tuple[float, ...], Body, dict[str, Any]] | None = None
            for k, prop in enumerate(batch.proposals[: cfg.max_proposals]):
                entry: dict[str, Any] = {"k": k, "proposal": prop.model_dump(), "iteration": it, "move": prop.move,
                                         "hypothesis": prop.hypothesis, "edit": _edit_text(prop.edits)}
                if prop.move == "idea" or prop.idea is not None and not prop.edits:
                    idea = prop.idea.model_dump() if prop.idea else {"name": "?", "needs": []}
                    known = set(manifest.get("items", {}))
                    idea["provided"] = [n for n in idea.get("needs", []) if n.split(" ")[0] in known]
                    ideas.append({**idea, "iteration": it})
                    entry.update(result=f"idea recorded, not tested: {idea['name']}", tried=False)
                    trials.append(entry)
                    history.append(entry)
                    continue
                try:
                    cand = apply(body, prop.edits, catalogue, new_id=f"b{it:02d}{k}")
                except ValueError as exc:
                    entry.update(result=f"refused before trial: {exc}", tried=False)
                    trials.append(entry)
                    history.append(entry)
                    continue
                entry["move"] = edit_rung(body, prop.edits)
                entry["slots"] = sorted({e.slot for e in prop.edits})
                errs = world.check(cand)
                res = check(cand, world.bom(cand), body, world.bom(body), manifest, spent)
                entry["resources"] = res.model_dump()
                if cand.key() == body.key():
                    errs = ["the edit leaves the body unchanged", *errs]
                if errs or not res.ok:
                    entry.update(result="refused before trial: " + "; ".join([*errs, *res.errors]), tried=False)
                    trials.append(entry)
                    history.append(entry)
                    continue
                held_tasks = [t for t in train if t.group in held]
                cand_out = ev.solve(cand, held_tasks)
                lost = sorted({t.group for t in held_tasks if not cand_out[t.id].success})
                if not lost:
                    cand_out.update(ev.solve(cand, [t for t in train if t.group not in held]))
                cand_held = held_groups(train, cand_out) if not lost else held - set(lost)
                gained = sorted(cand_held - held) if not lost else []
                n_new = sum(cand_out[t.id].success for t in train if t.group not in held) if not lost else 0
                still = sorted({cand_out[t.id].mechanism for t in train if t.id in cand_out and not cand_out[t.id].success})
                entry.update(tried=True, gained=gained, lost=lost, new_tasks=n_new, still_failing=still,
                             outcomes={t: o.model_dump() for t, o in cand_out.items()})
                why = (f"lost {', '.join(lost)} (" + "; ".join(f"{t.group}: {cand_out[t.id].mechanism}"
                                                               for t in held_tasks if t.group in lost
                                                               and not cand_out[t.id].success)[:300] + ")") if lost else ""
                if lost:
                    entry["result"] = f"rejected: {why}"
                elif not gained and n_new == 0:
                    entry["result"] = "rejected: gained nothing; still failing: " + ", ".join(still[:6])
                else:
                    entry["result"] = (f"kept-eligible: gained {', '.join(gained) or f'{n_new} tasks'}, lost none; "
                                       f"costs ${res.usd:.2f}, {res.fab_minutes:.0f} machine min")
                    score = (len(gained), n_new, -res.usd - res.fab_minutes / 60, -int(entry["move"] == "swap"))
                    if best is None or score > best[0]:
                        best = (score, cand, entry)
                trials.append(entry)
                history.append(entry)
                log(f"    proposal {k} [{entry['move']}] {entry['edit']}: {entry['result'][:160]}")
            accepted = None
            if best is not None:
                _, cand, entry = best
                res = entry["resources"]
                spent = Spent(usd=spent.usd + res["usd"], fab_minutes=spent.fab_minutes + res["fab_minutes"],
                              lead_days=max(spent.lead_days, res["lead_days"]))
                entry["result"] = entry["result"].replace("kept-eligible", "KEPT")
                body = cand
                accepted = cand.id
            for e in trials:
                if e.get("result", "").startswith("kept-eligible"):
                    e["result"] = e["result"].replace("kept-eligible", "not kept (another proposal gained more)")
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
        "wall_s": round(time.time() - t_start, 1),
        "llm": {"calls": len(getattr(proposer, "calls", []) or []),
                "tokens_in": sum(c.get("tokens_in", 0) for c in getattr(proposer, "calls", []) or []),
                "tokens_out": sum(c.get("tokens_out", 0) for c in getattr(proposer, "calls", []) or []),
                "usd_list_price": round(sum(c.get("usd_list_price", 0.0) for c in getattr(proposer, "calls", []) or []), 3)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
