"""The requirements list (D-089; the general framework, F2): written by System 2, measured by the harness.

An engineer starts from a requirements list: every need as a metric with a target, marked a demand (it must
hold) or a wish, traced to where it came from (Pahl and Beitz; Ulrich and Eppinger's target specifications).
Here System 2 writes the list, from the tasks, the brief and its own research; it keeps it in its answers and
revises it as it learns.  The harness never writes a requirement.  It:

- evaluates each one's metric (arithmetic over instrument readings and task facts, ``evolve/instruments``) on
  the current body and on any draft, per task where the requirement names tasks;
- marks it a **demand** if it guards a task group the robot already holds (the ratchet), else a **wish**;
- reports **coverage**: which train groups no requirement traces to, since LLM-written lists measurably leave
  most needs out (Massoudi and Fuge, 2025: under 20% coverage);
- and records which rows each kept change moved.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from sea.evolve import instruments as I
from sea.evolve.core import Body, Requirement, Task, World


class Row(BaseModel):
    """One requirement, measured on one body: its worst value over the tasks it names."""

    model_config = ConfigDict(extra="forbid")
    id: str
    value: float | None
    worst_task: str = ""
    meets: Literal["ideal", "marginal", "no", "unknown"]
    error: str = ""


def _ok(v: float, sense: str, target: float) -> bool:
    return v >= target if sense == ">=" else v <= target


def measure(world: World, body: Body, reqs: Sequence[Requirement], tasks: Sequence[Task],
            base: Body | None = None, manifest: dict[str, Any] | None = None, spent: Any = None) -> list[Row]:
    """Each requirement on ``body``: its worst value over its tasks (by group), and whether it meets the ideal,
    the marginal value, neither, or could not be measured (with why)."""
    by_group: dict[str, list[Task]] = {}
    for t in tasks:
        by_group.setdefault(t.group, []).append(t)
    out = []
    for r in reqs:
        cases: list[Task | None] = [t for g in r.tasks for t in by_group.get(g, [])] if r.tasks else [None]
        if r.tasks and not cases:
            out.append(Row(id=r.id, value=None, meets="unknown", error=f"no train task in groups {r.tasks}"))
            continue
        worst: tuple[float, str] | None = None
        error = ""
        for t in cases:
            facts = I.facts(world, t) if t is not None else {}
            try:
                v = I.evaluate(r.metric, lambda n, a, t=t: I.read(world, body, n, a, t, base, manifest, spent),
                               lambda n, facts=facts: facts[n])
            except KeyError as exc:
                error = f"no fact {exc} for {t.id if t else 'the body'}"
                break
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
                error = str(exc)[:200]
                break
            if math.isnan(v):
                error = "the metric is not a number here"
                break
            key = v if r.sense == ">=" else -v
            if worst is None or key < (worst[0] if r.sense == ">=" else -worst[0]):
                worst = (v, t.id if t else "")
        if error or worst is None:
            out.append(Row(id=r.id, value=None, meets="unknown", error=error))
            continue
        v, at = worst
        meets = ("ideal" if r.ideal is not None and _ok(v, r.sense, r.ideal) else
                 "marginal" if _ok(v, r.sense, r.marginal) else "no")
        out.append(Row(id=r.id, value=v, worst_task=at, meets=meets))
    return out


def kind(r: Requirement, held: set[str]) -> str:
    """A demand guards a group the robot holds (losing it loses the group); otherwise it is a wish."""
    return "demand" if set(r.tasks) & held else "wish"


def coverage(reqs: Sequence[Requirement], groups: Sequence[str]) -> list[str]:
    """The train groups no requirement traces to: measured on it, or named as where the need comes from (D-099)."""
    named = {g for r in reqs for g in r.tasks}
    named |= {g for g in groups if any(re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", r.source) for r in reqs)}
    return sorted(set(groups) - named)


def rule_coverage(reqs: Sequence[Requirement], rules: Sequence[str]) -> list[int]:
    """The brief's rules (numbered from 1) no requirement cites in its source (D-095)."""
    cited = {int(n) for r in reqs for n in re.findall(r"\brule (\d+)", r.source)}
    return [k for k in range(1, len(rules) + 1) if k not in cited]


def text(reqs: Sequence[Requirement], rows: Sequence[Row], held: set[str], groups: Sequence[str],
         draft: Sequence[Row] | None = None, rules: Sequence[str] = ()) -> str:
    """The table for the prompt (and, with ``draft``, the draft's values beside the current body's)."""
    if not reqs:
        return ("No requirements yet.  Write them in your answer's `requirements`: each a metric over the "
                "instruments and task facts, with its target and where the need comes from.")
    lines = []
    for r, row, *d in zip(reqs, rows, *([draft] if draft is not None else []), strict=True):
        target = f"{r.sense} {r.marginal:g}" + (f" (ideal {r.ideal:g})" if r.ideal is not None else "")
        now = (f"{row.value:.4g}" + (f" at {row.worst_task}" if row.worst_task else "") + f": {row.meets}"
               if row.value is not None else f"unmeasured: {row.error}")
        cell = f"; draft {d[0].value:.4g}: {d[0].meets}" if d and d[0].value is not None else \
            (f"; draft unmeasured: {d[0].error}" if d else "")
        lines.append(f"- {r.id} [{kind(r, held)}] {r.statement}: `{r.metric}` {target}; now {now}{cell} "
                     f"(source: {r.source}; groups: {', '.join(r.tasks) or 'the body'})")
    missing = coverage(reqs, groups)
    if missing:
        lines.append(f"Coverage: no requirement is measured on {len(missing)} train groups: {', '.join(missing)}")
    uncited = rule_coverage(reqs, rules)
    if uncited:
        lines.append("Coverage: no requirement cites " + ", ".join(f"rule {k}" for k in uncited) + " in its source")
    return "\n".join(lines)


def moved(before: Sequence[Row], after: Sequence[Row]) -> list[str]:
    """The rows a change moved, and how."""
    out = []
    for a, b in zip(before, after, strict=True):
        if a.meets != b.meets or (a.value is not None and b.value is not None and abs(a.value - b.value) > 1e-9):
            out.append(f"{a.id}: {_fmt(a)} -> {_fmt(b)}")
    return out


def _fmt(r: Row) -> str:
    return f"{r.value:.4g} ({r.meets})" if r.value is not None else "unmeasured"
