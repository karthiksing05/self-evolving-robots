"""System 2's prompt for the evolution core: the same sections in every world (docs/evolve.md).

The harness's own principles come first and never change between worlds; the world supplies its
description, a computed brief of the current body, its physics principles, its catalogue and its
resource manifest.  The loop supplies the evidence: which task groups are held, which fail and with
what mechanism, and every past proposal with what it gained, lost or was refused for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sea.evolve.core import Body, BomLine, Outcome, PartType, Task, World
from sea.evolve.resources import Spent, check, is_fab, totals

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


def _catalogue_text(world: World, body: Body, manifest: dict[str, Any], spent: Spent) -> str:
    out = []
    cat = world.catalogue()
    cur_bom = world.bom(body)
    for slot in sorted({p.slot for p in cat}):
        cur = body.parts.get(slot)
        out.append(f"slot `{slot}` (now: {cur.type if cur else 'empty'})")
        for pt in [p for p in cat if p.slot == slot]:
            probe = Body(parts={**body.parts, slot: _mid(pt)})
            probe_bom = world.bom(probe)
            needs = availability([b for b in probe_bom if b.slot == slot], manifest, cur_bom)
            out.append(f"  - {pt.name}{' [stock]' if pt.stock else ''}: {pt.doc}")
            if pt.provides:
                out.append(f"      provides: {pt.provides}")
            for k, p in pt.params.items():
                out.append(f"      {k} [{p.lo:g}, {p.hi:g}] {p.unit}: {p.doc}")
            out.append(f"      needs (at mid-range): {needs or 'nothing'}")
            if cur is not None and cur.type != pt.name:  # the swap's true marginal cost from this body
                res = check(probe, probe_bom, body, cur_bom, manifest, spent)
                out.append(f"      swapping it in now costs ${res.usd:.2f}, {res.lead_days:g} days, "
                           f"{res.fab_minutes:.0f} machine min" if res.ok else
                           f"      swapping it in now is refused: {'; '.join(res.errors)}")
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
          iteration: int, iterations: int, max_proposals: int = 3, frontier: Sequence[dict[str, Any]] = (),
          tune_trials: int = 0) -> str:
    by_id = {t.id: t for t in tasks}
    groups = sorted({t.group for t in tasks})
    failing = [t for t in tasks if not outcomes[t.id].success]
    first = {}
    for t in tasks:
        first.setdefault(t.group, t)
    held_txt = "\n".join(f"- {g}: {first[g].facts}" if first[g].facts else f"- {g}" for g in sorted(held)) or "none"
    fail_groups = sorted({t.group for t in failing})
    sec = [f"# World: {world.name}\n{world.description}",
           f"# Iteration {iteration + 1} of {iterations}\n{len(held)}/{len(groups)} task groups held "
           f"({sum(o.success for o in outcomes.values())}/{len(tasks)} tasks).",
           f"# Current body\n{body.describe()}",
           f"# Brief: computed facts about the current body\n{world.brief(body, tasks)}",
           "# Physics principles of this world\n" + "\n".join(f"- {p}" for p in world.principles()),
           f"# Catalogue: every part type, by slot\n{_catalogue_text(world, body, manifest, spent)}",
           f"# Resources\n{manifest_text(manifest, spent)}",
           f"# Held task groups (each must keep passing)\n{held_txt}",
           f"# Failing tasks ({len(fail_groups)} groups)\n" + "\n".join(_outcome_line(by_id[t.id], outcomes[t.id])
                                                                        for t in failing)]
    if history:
        rows = []
        for h in history[-16:]:
            if h["move"] == "tune":
                rows.append(f"- iteration {h['iteration'] + 1}, tuning {h['base']}: {h['edit']}\n    result: {h['result']}")
                continue
            rows.append(f"- iteration {h['iteration'] + 1}, {h['move']}: {h['edit']}\n    hypothesis: {h['hypothesis'][:300]}"
                        f"\n    result: {h['result']}")
        sec.append("# Record of past proposals (most recent last)\n" + "\n".join(rows))
        cal = _calibration(history)
        if cal:
            sec.append(f"# Your predictions against the trials\n{cal}")
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
    close = (f"Propose 1 to {max_proposals} body changes for the failing groups.  Each proposal is tried on "
             "its own and kept only if it gains a group and loses none.")
    if tune_trials:
        close += (f"  A swap that gains a group but breaks a held one is not judged on your values alone: the loop "
                  f"spends up to {tune_trials} more trials fitting the parameters of the slots it changed, so judge a "
                  "part type by what it can do across its range.")
    sec.append(close)
    return "\n\n".join(sec)


def _calibration(history: Sequence[dict[str, Any]]) -> str:
    """System 2's predicted gains and named risks against what its trials found."""
    tried = [h for h in history if h.get("prediction")]
    if not tried:
        return ""
    pred = sum(len(h["prediction"]["predicted"]) for h in tried)
    hit = sum(len(h["prediction"]["hit"]) for h in tried)
    unpred = sum(len(h["prediction"]["unpredicted"]) for h in tried)
    lost = [(g, h) for h in tried for g in h["prediction"]["lost"]]
    unwarned = sum(not any(g in r for r in h.get("proposal", {}).get("risks", [])) for g, h in lost)
    return (f"Over {len(tried)} trials: {hit} of the {pred} gains you predicted happened, {unpred} gains came "
            f"unpredicted, and {len(lost)} held groups were lost, {unwarned} of them not named in your risks.")


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


def availability(bom: Sequence[BomLine], manifest: dict[str, Any], have: Sequence[BomLine] = ()) -> str:
    """One line per item a part type needs, saying whether the bench provides it: for the catalogue.  Items in
    ``have`` (the current body's bill) are reused."""
    items = manifest.get("items", {})
    fab = manifest.get("fabrication", {})
    owned = totals([b for b in have if not is_fab(b.item)])
    out = []
    for item, qty in sorted(totals(bom).items()):
        if is_fab(item):
            out.append(f"{item} {qty:.0f}{fab.get(item, {}).get('unit', '')}" + ("" if item in fab else " (NO MACHINE)"))
            continue
        spec = items.get(item)
        if spec is None:
            out.append(f"{item} x{qty:g} (NOT PROVIDED)")
        elif owned.get(item, 0.0) >= qty:
            out.append(f"{item} x{qty:g} (reused from the current body)")
        elif float(spec.get("on_hand", 0)) + owned.get(item, 0.0) >= qty:
            out.append(f"{item} x{qty:g} (on hand)")
        elif spec.get("usd") is not None:
            out.append(f"{item} x{qty:g} (buy ${float(spec['usd']):g} each, {float(spec.get('lead_days', 0)):g} d)")
        else:
            out.append(f"{item} x{qty:g} (NOT PROVIDED)")
    return ", ".join(out)


def manifest_text(manifest: dict[str, Any], spent: Spent) -> str:
    b = manifest.get("budget", {})
    head = (f"budget left: ${float(b.get('usd', 0)) - spent.usd:.2f} to buy parts, at most "
            f"{float(b.get('lead_days', 0)):g} days of lead time, "
            f"{float(b.get('fab_minutes', 0)) - spent.fab_minutes:.0f} machine minutes")
    rows = []
    for name, spec in sorted(manifest.get("items", {}).items()):
        buy = f"buy ${float(spec['usd']):g}, {float(spec.get('lead_days', 0)):g} d" if spec.get("usd") is not None \
            else "cannot be bought"
        rows.append(f"  {name}: {float(spec.get('on_hand', 0)):g} on hand, {buy}" +
                    (f" ({spec['doc']})" if spec.get("doc") else ""))
    for name, spec in sorted(manifest.get("fabrication", {}).items()):
        rows.append(f"  {name}: {float(spec.get('minutes_per_unit', 0)):g} min per {spec.get('unit', 'unit')}" +
                    (f" ({spec['doc']})" if spec.get("doc") else ""))
    return head + "\n" + "\n".join(rows)
