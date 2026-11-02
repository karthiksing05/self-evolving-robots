"""System 2's prompt for the evolution core: the same sections in every world (docs/evolve.md).

The harness's own principles come first and never change between worlds; the world supplies its
description, a computed brief of the current body, its physics principles, its catalogue and its
resource manifest.  The loop supplies the evidence: which task groups are held, which fail and with
what mechanism, and every past proposal with what it gained, lost or was refused for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sea.evolve.core import Body, Outcome, PartType, Task, World
from sea.evolve.resources import Spent, availability, manifest_text

SYSTEM = """You are System 2 of a self-evolving robot.  System 1 has already searched the control \
parameters of every task; what still fails is yours.  You change the robot's body, and only the body.  \
Answer with a ProposalBatch.  Work like an engineer who must justify every part to a budget:

1. Diagnose before you design.  Each failing task comes with a mechanism read from the simulator and the \
numbers behind it.  Name the mechanism you are fixing, and do not fix a symptom.
2. Know the body you have.  The brief states computed facts about the current parts: their ranges, \
their kinematic limits, what they can and cannot reach.  Reason from those numbers, and quote them.
3. Climb the ladder only as far as you must.  A `params` move (new values for the current parts) is \
cheapest.  A `swap` replaces a part with another type from the catalogue: a redesign of that part, worth \
its cost when the brief or the history shows that no value in the current part's range fixes the \
mechanism.  An `idea` is a part the catalogue lacks; it is recorded with what it needs and is not \
tested, so propose one only when no catalogue part can address the mechanism.
4. The robot's history comes first.  A change is kept only if every task group already held still \
passes.  Say which held groups your change puts at risk and why they will survive.
5. Resources are real.  Every part has a bill of materials.  Items on hand are free, items with a price \
cost budget and lead time, items the manifest does not provide make the design impossible.  Use what is \
provided, and say what you would buy.
6. Learn from the record.  Past proposals list what they gained, what they broke, and why some were \
refused before any trial.  Do not repeat a refused or failed change; change your reasoning instead.
7. Predict, then let the trial decide.  List the groups you expect to gain.  The simulator, not your \
prediction, decides what is kept."""


def _catalogue_text(world: World, body: Body, manifest: dict[str, Any]) -> str:
    out = []
    cat = world.catalogue()
    for slot in sorted({p.slot for p in cat}):
        cur = body.parts.get(slot)
        out.append(f"slot `{slot}` (now: {cur.type if cur else 'empty'})")
        for pt in [p for p in cat if p.slot == slot]:
            probe = Body(parts={**body.parts, slot: _mid(pt)})
            needs = availability([b for b in world.bom(probe) if b.slot == slot], manifest)
            out.append(f"  - {pt.name}{' [stock]' if pt.stock else ''}: {pt.doc}")
            if pt.provides:
                out.append(f"      provides: {pt.provides}")
            for k, p in pt.params.items():
                out.append(f"      {k} [{p.lo:g}, {p.hi:g}] {p.unit}: {p.doc}")
            out.append(f"      needs (at mid-range): {needs or 'nothing'}")
            facts = getattr(world, "part_facts", None)
            if facts is not None:  # what the part would do on this body, computed, at its parameters' ranges
                out.extend(f"      {line}" for line in facts(body, pt))
    return "\n".join(out)


def _mid(pt: PartType):
    from sea.evolve.core import Part

    return Part(type=pt.name, params={k: (p.lo + p.hi) / 2 for k, p in pt.params.items()})


def _outcome_line(task: Task, o: Outcome) -> str:
    ctl = ", ".join(f"{k}={v:.3g}" for k, v in sorted(o.controls.items()))
    return (f"  - {task.id} [{task.group}] {task.facts}\n      mechanism: {o.mechanism or 'unlabelled'}"
            f" | {o.detail}\n      System 1: {o.attempts} attempts, best score {o.score:.2f}" + (f" at {ctl}" if ctl else ""))


def build(world: World, body: Body, tasks: Sequence[Task], outcomes: dict[str, Outcome], held: set[str],
          history: Sequence[dict[str, Any]], ideas: Sequence[dict[str, Any]], spent: Spent, manifest: dict[str, Any],
          iteration: int, iterations: int, max_proposals: int = 3, frontier: Sequence[dict[str, Any]] = ()) -> str:
    by_id = {t.id: t for t in tasks}
    groups = sorted({t.group for t in tasks})
    failing = [t for t in tasks if not outcomes[t.id].success]
    held_txt = ", ".join(sorted(held)) or "none"
    fail_groups = sorted({t.group for t in failing})
    sec = [f"# World: {world.name}\n{world.description}",
           f"# Iteration {iteration + 1} of {iterations}\n{len(held)}/{len(groups)} task groups held "
           f"({sum(o.success for o in outcomes.values())}/{len(tasks)} tasks).",
           f"# Current body\n{body.describe()}",
           f"# Brief: computed facts about the current body\n{world.brief(body, tasks)}",
           "# Physics principles of this world\n" + "\n".join(f"- {p}" for p in world.principles()),
           f"# Catalogue: every part type, by slot\n{_catalogue_text(world, body, manifest)}",
           f"# Resources\n{manifest_text(manifest, spent)}",
           f"# Held task groups (each must keep passing)\n{held_txt}",
           f"# Failing tasks ({len(fail_groups)} groups)\n" + "\n".join(_outcome_line(by_id[t.id], outcomes[t.id])
                                                                        for t in failing)]
    if history:
        rows = []
        for h in history[-12:]:
            rows.append(f"- iteration {h['iteration'] + 1}, {h['move']}: {h['edit']}\n    hypothesis: {h['hypothesis'][:300]}"
                        f"\n    result: {h['result']}")
        sec.append("# Record of past proposals (most recent last)\n" + "\n".join(rows))
        tried = _exhaustion(history)
        if tried:
            sec.append("# What parameter moves have not fixed\n" + tried)
    if frontier:
        sec.append("# Promising bodies not kept (they gained more groups than they lost)\n" + "\n".join(
            f"- {f['id']} (iteration {f['iteration'] + 1}): {f['body'].describe()}\n    would gain {', '.join(f['gained'])}"
            f"\n    {f['why_lost'][:400]}" for f in frontier) +
            "\nTo build on one, set `base` to its id: your edits apply to that body, and the result must still keep "
            "every held group.")
    if ideas:
        sec.append("# Ideas recorded so far (untested: no builder in the simulator)\n" +
                   "\n".join(f"- {i['name']}: needs {', '.join(i['needs']) or 'nothing listed'}" for i in ideas[-6:]))
    sec.append(f"Propose 1 to {max_proposals} body changes for the failing groups.  Each proposal is tried on "
               "its own and kept only if it gains a group and loses none.")
    return "\n\n".join(sec)


def _exhaustion(history: Sequence[dict[str, Any]]) -> str:
    """Per slot: how many parameter-only trials ran, and the mechanisms still failing after them."""
    per: dict[str, list[dict[str, Any]]] = {}
    for h in history:
        if h["move"] == "params" and h.get("tried"):
            for slot in h.get("slots", []):
                per.setdefault(slot, []).append(h)
    out = []
    for slot, hs in sorted(per.items()):
        mech = sorted({m for h in hs for m in h.get("still_failing", [])})
        gained = sum(len(h.get("gained", [])) for h in hs)
        out.append(f"- slot `{slot}`: {len(hs)} parameter-only trials, {gained} groups gained in total; mechanisms "
                   f"still failing after them: {', '.join(mech[:8]) or 'none'}")
    return "\n".join(out)
