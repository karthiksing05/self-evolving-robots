"""Whether the operation or the body failed (D-092; the general framework, F5).

System 2's operating plan says what each state should reach.  A failed task then says which of two things is
wrong: System 1 never got the mechanism to the intended state, or it did and the task still failed.  These call
for different changes, and without the difference a mechanism is judged by how well it was operated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sea.evolve.core import Intent, OperatingPlan, Outcome

NOT_FOUND = "operation_not_found"
CANNOT = "body_cannot"
UNOBSERVED = "unobserved"
SEARCH_NEW = 2  # how many times longer System 1 searches a new concept, and what is developed from it


def _met(i: Intent, v: float) -> bool:
    return (i.lo is None or v >= i.lo) and (i.hi is None or v <= i.hi)


def label(plan: OperatingPlan | None, o: Outcome, group: str) -> tuple[str, list[str]]:
    """A failed outcome's label, with each intent's reading.  A world's own probe that says the body could do it
    (``not_found_by_search``) counts as the operation not being found."""
    if o.mechanism == "not_found_by_search":
        return NOT_FOUND, []
    intents: Sequence[Intent] = [i for i in (plan.intents if plan else []) if not i.tasks or group in i.tasks]
    if not intents:
        return UNOBSERVED, []
    rows = [f"{i.state} {o.states[i.state]:.3g}" + (" (reached)" if _met(i, o.states[i.state]) else " (not reached)")
            if i.state in o.states else f"{i.state} unread" for i in intents]
    if any(i.state not in o.states for i in intents):
        return UNOBSERVED, rows
    return (CANNOT if all(_met(i, o.states[i.state]) for i in intents) else NOT_FOUND), rows


def diagnose(plan: OperatingPlan | None, outcomes: dict[str, dict[str, Any]], groups: dict[str, str]) -> dict[str, Any]:
    """Every failed task of a trial, labelled, and how many of each."""
    tasks, counts = {}, {NOT_FOUND: 0, CANNOT: 0, UNOBSERVED: 0}
    for t, raw in outcomes.items():
        if raw["success"]:
            continue
        lab, rows = label(plan, Outcome.model_validate(raw), groups.get(t, ""))
        counts[lab] += 1
        tasks[t] = {"label": lab, "states": rows}
    return {"counts": counts, "tasks": tasks}


def text(names: Sequence[str]) -> str:
    """The plan's section for the prompt: how it works, and what this world lets a plan set."""
    return ("A proposal's `plan` says how the body is to be operated: `controls`, the values System 1 starts its search "
            "from; `triggers`, where a sensor starts an actuator's clock; and `intents`, the states the body should "
            "reach, with bounds.  A plan travels with its body and may be proposed alone, with no edits.  Each failed "
            "task in a trial is labelled from your intents: operation_not_found (an intended state was not reached), "
            "body_cannot (each was reached, and the task still failed), or unobserved (no intent, or a state this "
            "world does not read).\n"
            + (f"Controls here: {', '.join(names)}" if names else "This world's System 1 does not take starting "
               "values: a plan here carries intents."))
