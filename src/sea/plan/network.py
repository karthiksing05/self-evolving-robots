"""The task network between System 2 and System 1 (docs/goals.md, D-034, D-037).  Policy tier.

Goals form a tree.  Each goal carries:
- a test the judge evaluates: a check expression on one control (``compose/checks.py``), or a test
  over many controls or a body change (``verify/goals.py``);
- a budget slice, or none to share its parent's;
- the task that serves it: a System 1 skill, an analysis, a body change, or a network (a method
  that decomposes it into children).

Whoever serves a goal answers with a ``GoalReport``: the judge's verdict, what it spent, and the
impasse if it could not proceed.  The impasse type picks the sub-goal (``IMPASSE_SUBGOAL``).

``DegenerateNetwork`` is the loop as it runs today, written as a network (docs/goals.md 3.6): one
root goal (every training control passes), one leaf per control served by System 1, and one
one-shot System 2 node per iteration that proposes.  It only records, so the loop's numbers do not
change.  ``GoalLog`` writes the network to ``goals.jsonl``, one event per line, and ``replay`` reads
it back.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from sea.compose import checks
from sea.contracts import Budgets, ControlSpec, CostVector, Decision, PanelSpec, Proposal
from sea.sim.env import ControlRun, attempts
from sea.spec.ledger import RESOURCES, Ledger
from sea.verify import goals as goal_tests

TaskKind = Literal["skill", "analyze", "design", "network"]
Impasse = Literal["structural", "s1_exhausted", "surprise", "unfamiliar", "reflex", "slice_exhausted"]
GoalStatus = Literal["open", "passed", "failed", "exhausted"]

ROOT = "root"
REFLEX_STOPS = frozenset({"load_limit", "slip", "force_limit"})  # trace notes that are System 1 reflexes (D-036)

# What each impasse creates when System 2 deliberates (docs/goals.md 3.3).  The one-shot fast path
# answers every impasse with a design proposal instead.  None: return to the parent.
IMPASSE_SUBGOAL: dict[str, TaskKind | None] = {
    "structural": "design",  # no method's preconditions hold with this body: change the body
    "s1_exhausted": "network",  # the search ran out: decompose with another method
    "unfamiliar": "network",  # memory has no method for this kind of goal yet
    "surprise": "analyze",  # a sensor residual or a failed prediction: look before acting again
    "reflex": "analyze",  # load_limit, slip, force_limit: never retry blindly
    "slice_exhausted": None,
}

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class Task(BaseModel):
    model_config = _FROZEN

    kind: TaskKind
    program: str = ""  # skill: the library program to run, or "" for System 1's own retrieval
    needs: tuple[str, ...] = ()  # sensor parts the task relies on (configs/sensors.yaml)
    question: str = ""  # analyze
    change: str = ""  # design: the proposal kind
    method: str = ""  # network: the method that decomposes the goal


class Goal(BaseModel):
    model_config = _FROZEN

    id: str
    parent: str | None = None
    test: str
    target: str = ""  # a control key when ``test`` is a check expression on that control
    serves: tuple[str, ...] = ()  # goals whose impasse this goal answers
    why: str = ""  # System 2's rationale
    task: Task
    slice: CostVector | None = None  # None: shares the parent's slice


class Method(BaseModel):
    """A way to decompose a goal: HTN's method, the library's composite program one level up."""

    model_config = _FROZEN

    name: str
    achieves: str  # the goal test atom it serves, such as pass_share
    requires: dict[str, Any] = {}
    body: tuple[Task, ...] = ()
    source: Literal["seed", "learned"] = "seed"


class GoalReport(BaseModel):
    model_config = _FROZEN

    goal: str
    iteration: int
    verdict: str  # the judge's: a control verdict, or pass / fail for a test over many
    impasse: Impasse | None = None
    stopped_by: str = ""  # a System 1 reflex or limit that ended the attempt
    program: str = ""
    spent: CostVector = CostVector()


# The degenerate network's only method: serve the root by one System 1 leaf per control.
COVERAGE = Method(name="coverage", achieves="pass_share", body=(Task(kind="skill"),))


# ---------------------------------------------------------------- slices
def cost_sum(costs: Sequence[CostVector]) -> CostVector:
    return CostVector(**{f: sum(getattr(c, f) for c in costs) for f in CostVector.model_fields})


def cost_diff(a: CostVector, b: CostVector) -> CostVector:
    return CostVector(**{f: getattr(a, f) - getattr(b, f) for f in CostVector.model_fields})


def fits(child: CostVector, parent: CostVector) -> bool:
    """Every budgeted resource of ``child`` within ``parent``'s (servo heat is priced, not sliced)."""
    return all(getattr(child, r) <= getattr(parent, r) + 1e-9 for r in RESOURCES)


def overspent(slice_: CostVector, spent: CostVector) -> list[str]:
    return [r for r in RESOURCES if getattr(spent, r) > getattr(slice_, r) + 1e-9]


def run_slice(budgets: Budgets) -> CostVector:
    """The root's slice: every budget less the reserve kept for the final evaluation."""
    ledger = Ledger(budgets)
    keep = 1.0 - budgets.reserve_fraction
    return CostVector(**{r: ledger.budget(r) * keep for r in RESOURCES})


def effective_slice(goal_id: str, goals: Mapping[str, Goal]) -> CostVector | None:
    seen: set[str] = set()
    g: Goal | None = goals.get(goal_id)
    while g is not None and g.slice is None and g.parent is not None and g.id not in seen:
        seen.add(g.id)
        g = goals.get(g.parent)
    return None if g is None else g.slice


# ---------------------------------------------------------------- validation
def validate(goals: Sequence[Goal], *, mounted: Collection[str] = (), programs: Collection[str] | None = None,
             sets: Collection[str] | None = None) -> list[str]:
    """Why a network cannot be dispatched; empty when it can.

    - one root, unique ids, every parent and every served goal present, no cycles;
    - every test drawn from the judge's vocabulary, naming only known sets;
    - the root has a slice, and the explicit slices of a goal's children fit inside its own;
    - every skill names a program that exists (or leaves the choice to System 1);
    - every task's sensors are mounted on the current body (D-033)."""
    errors: list[str] = []
    by_id: dict[str, Goal] = {}
    for g in goals:
        if g.id in by_id:
            errors.append(f"{g.id}: duplicate id")
        by_id[g.id] = g
    roots = [g.id for g in goals if g.parent is None]
    if len(roots) != 1:
        errors.append(f"needs exactly one root, has {len(roots)}")
    for g in goals:
        if g.parent is not None and g.parent not in by_id:
            errors.append(f"{g.id}: unknown parent {g.parent}")
        errors += [f"{g.id}: serves unknown goal {s}" for s in g.serves if s not in by_id]
        errors += _test_errors(g, sets)
        if g.task.kind == "skill" and g.task.program and programs is not None and g.task.program not in programs:
            errors.append(f"{g.id}: skill {g.task.program} is not in the library")
        missing = sorted(set(g.task.needs) - set(mounted))
        if missing:
            errors.append(f"{g.id}: needs unmounted sensors {', '.join(missing)}")
        if _cycles(g, by_id):
            errors.append(f"{g.id}: its parents form a cycle")
    for rid in roots:
        if by_id[rid].slice is None:
            errors.append(f"{rid}: the root needs a slice")
    for g in goals:
        own = effective_slice(g.id, by_id)
        kids = [c.slice for c in goals if c.parent == g.id and c.slice is not None]
        if own is not None and kids and not fits(cost_sum(kids), own):
            errors.append(f"{g.id}: children's slices exceed its own")
    return errors


def _test_errors(g: Goal, sets: Collection[str] | None) -> list[str]:
    try:
        if g.target:
            checks.parse(g.test)
            return []
        named = goal_tests.sets(goal_tests.parse(g.test))
    except checks.CheckSyntaxError as exc:
        return [f"{g.id}: test is not in the judge's vocabulary ({exc})"]
    unknown = sorted(named - set(sets)) if sets is not None else []
    return [f"{g.id}: test names unknown sets {', '.join(unknown)}"] if unknown else []


def _cycles(g: Goal, by_id: Mapping[str, Goal]) -> bool:
    seen = {g.id}
    p = g.parent
    while p is not None and p in by_id:
        if p in seen:
            return True
        seen.add(p)
        p = by_id[p].parent
    return False


# ---------------------------------------------------------------- the log
class GoalLog:
    """``goals.jsonl``: one event per line (open, report, close), so a run's network can be replayed."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def _write(self, event: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def open(self, goal: Goal, iteration: int) -> None:
        self._write({"event": "open", "iteration": iteration, "goal": goal.model_dump(mode="json")})

    def report(self, rep: GoalReport) -> None:
        self._write({"event": "report", "report": rep.model_dump(mode="json")})

    def close(self, goal_id: str, status: GoalStatus, iteration: int, reason: str = "") -> None:
        self._write({"event": "close", "iteration": iteration, "goal": goal_id, "status": status, "reason": reason})


def replay(path: Path) -> tuple[dict[str, Goal], list[GoalReport], dict[str, GoalStatus]]:
    goals: dict[str, Goal] = {}
    reports: list[GoalReport] = []
    status: dict[str, GoalStatus] = {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "open":
            g = Goal.model_validate(e["goal"])
            goals[g.id] = g
            status[g.id] = "open"
        elif e["event"] == "report":
            reports.append(GoalReport.model_validate(e["report"]))
        else:
            status[e["goal"]] = e["status"]
    return goals, reports, status


# ---------------------------------------------------------------- the loop as a network
class DegenerateNetwork:
    """Records the loop as one root, a System 1 leaf per control, and a one-shot System 2 node per proposal."""

    def __init__(self, log: GoalLog, budgets: Budgets, train: Sequence[PanelSpec]):
        self.log = log
        self.members = {"train": [f"{p.id}:{c.id}" for p in train for c in p.controls]}
        self.root = Goal(id=ROOT, test="pass_share(train, 1)", why="every training control passes",
                         task=Task(kind="network", method=COVERAGE.name), slice=run_slice(budgets))
        self.goals: dict[str, Goal] = {ROOT: self.root}
        self.verdicts: dict[str, str] = {}
        log.open(self.root, 0)

    def observe(self, iteration: int, runs: Sequence[ControlRun], impasses: Mapping[str, str],
                controls: Mapping[str, tuple[ControlSpec, PanelSpec]]) -> None:
        """One report per control System 1 served this iteration; a leaf opens the first time it is served."""
        for run in runs:
            key = run.key
            if key not in self.goals:
                leaf = Goal(id=key, parent=ROOT, test=controls[key][0].success, target=key, task=Task(kind="skill"))
                self.goals[key] = leaf
                self.log.open(leaf, iteration)
            tr = run.trace
            spent = CostVector(sim_seconds=round(attempts(run) * tr.sim_seconds, 4)) if tr is not None else CostVector()
            self.log.report(GoalReport(goal=key, iteration=iteration, verdict=run.result.verdict,
                                       impasse=impasses.get(key),  # type: ignore[arg-type]
                                       stopped_by=tr.note if tr is not None else "",
                                       program=run.program or run.result.program_id, spent=spent))
            self.verdicts[key] = run.result.verdict

    def one_shot(self, iteration: int, serves: Sequence[str], proposal: Proposal | None, decision: Decision | None,
                 spent: CostVector) -> None:
        """System 2's one call at this iteration's impasses, closed by the acceptance rule's decision."""
        node = Goal(id=f"s2.{iteration:03d}", parent=ROOT, test="accepted()", serves=tuple(serves),
                    why=proposal.hypothesis if proposal is not None else "no proposal was tried",
                    task=Task(kind="design", change=proposal.kind if proposal is not None else ""))
        self.goals[node.id] = node
        self.log.open(node, iteration)
        ok = goal_tests.holds(node.test, goal_tests.Evidence(members={}, verdicts={},
                                                             accepted=decision.accepted if decision else None))
        self.log.report(GoalReport(goal=node.id, iteration=iteration, verdict="pass" if ok else "fail", spent=spent))
        self.log.close(node.id, "passed" if ok else "failed", iteration)

    def close(self, iteration: int, stop_reason: str) -> GoalStatus:
        """The root exits on its test; otherwise its slice or the iterations ran out."""
        ok = goal_tests.holds(self.root.test, goal_tests.Evidence(members=self.members, verdicts=self.verdicts))
        status: GoalStatus = "passed" if ok else "exhausted"
        self.log.close(ROOT, status, iteration, reason=stop_reason)
        return status
