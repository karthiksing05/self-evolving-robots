"""The world-agnostic core of body evolution (docs/evolve.md, D-041).

A *world* supplies tasks, a simulator, and a catalogue of parts.  A *body* is one part per slot.  The
loop never knows what a gripper or a vocal tract is: it asks the world to solve tasks with System 1,
shows System 2 what failed and why, and lets it change the body in one of three ways, cheapest first:

- ``params``: new values for parameters of the parts the body already has;
- ``swap``: a different part type in a slot, i.e. a redesign of that part;
- ``idea``: a part the catalogue does not have.  It cannot be simulated, so it is recorded with what it
  would need from the resource manifest, never trialled.

Every part carries a bill of materials.  A body change is admitted only if the manifest provides what it
needs or the run's budget can buy it (``resources.py``), so a redesign is always a costed, grounded move.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class _Model(BaseModel):
    model_config = _FROZEN


# ---------------------------------------------------------------- parts and bodies
class Param(_Model):
    lo: float
    hi: float
    unit: str = ""
    doc: str = ""  # what changing it does, in physical terms


class BomLine(_Model):
    """One item a part needs.  ``item`` names a manifest item, or ``print:<material>`` in grams."""

    item: str
    qty: float
    slot: str = ""
    note: str = ""


class PartType(_Model):
    name: str
    slot: str
    doc: str  # what it is, physically
    params: dict[str, Param] = {}
    provides: str = ""  # the capability it gives, in words; never a promise about a task
    stock: bool = False  # the part the body starts with in this slot


class Part(_Model):
    type: str
    params: dict[str, float] = {}
    mechanism: str = ""  # a mechanism System 2 composed, as its graph's JSON (sea.mech, D-079); "" for the catalogue's


class Body(_Model):
    id: str = "stock"
    parent: str | None = None
    parts: dict[str, Part]

    def key(self) -> str:
        """Content hash: two bodies with the same parts are the same body, whatever their ids."""
        blob = json.dumps({s: p.model_dump(exclude=set() if p.mechanism else {"mechanism"})
                           for s, p in sorted(self.parts.items())}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        out = []
        for slot, p in sorted(self.parts.items()):
            ps = ", ".join(f"{k}={v:g}" for k, v in sorted(p.params.items()))
            if p.mechanism:  # a built mechanism: its graph's name
                ps = json.loads(p.mechanism).get("name", "")
            out.append(f"{slot}: {p.type}" + (f" ({ps})" if ps else ""))
        return "; ".join(out)


# ---------------------------------------------------------------- tasks and outcomes
Split = Literal["train", "test"]


class Task(_Model):
    id: str
    split: Split = "train"
    group: str = ""  # tasks that differ only in seed or placement share a group (an object, a phoneme)
    facts: str = ""  # what System 2 may know about the task, in words
    data: dict[str, Any] = {}


class Instrument(_Model):
    """A quantity a world can measure on a body, or on a body facing a task (D-088): a reading, never advice."""

    name: str
    unit: str
    doc: str
    args: dict[str, str] = {}  # argument -> what it is, with its range
    task: bool = False  # it reads the body facing one task


class Outcome(_Model):
    """One task solved (or not) by one body.  ``mechanism`` is the world's label for *why*, read from the
    simulator after the attempt (a twin's analysis, not the robot's senses); the verdict is ``success``."""

    task: str
    success: bool
    score: float = 0.0  # 0..1 partial credit, for diagnosis and curves; acceptance reads ``success``
    mechanism: str = ""
    detail: str = ""
    controls: dict[str, float] = {}
    attempts: int = 1
    sim_s: float = 0.0


# ---------------------------------------------------------------- System 2's moves
class Edit(_Model):
    slot: str
    type: str | None = Field(default=None, description="a part type from the catalogue to swap in; omit to keep "
                             "the slot's current type")
    params: dict[str, float] = Field(default_factory=dict, description="parameter values to set; a swapped-in part "
                                     "takes the type's midpoint for any parameter left out")


class Need(_Model):
    item: str = Field(description="a manifest item's exact name, print:<material> (qty in grams), or a new item")
    qty: float = 1.0
    usd_each: float | None = Field(default=None, description="for an item the manifest does not list: your estimated "
                                   "price; leave empty for a manifest item")
    lead_days: float | None = Field(default=None, description="for an item the manifest does not list: your estimate")


class Mechanism(_Model):
    """A mechanism the catalogue lacks, written as a change to one of the world's modules (D-079)."""

    module: str = Field(description="the module it starts from, by name, as the mechanisms section lists them")
    params: dict[str, float] = Field(default_factory=dict, description="the module's parameters, within its ranges")
    patch: list[dict[str, Any]] = Field(default_factory=list, description="operations on the module's graph, each "
                                        "{op: add|set|remove, path, value}, addressed by names as the section says")


class Idea(_Model):
    name: str
    description: str = Field(description="the part and how it works physically")
    slot: str = Field(default="", description="the slot it would fill, or a new slot")
    needs: list[Need] = Field(default_factory=list, description="its bill of materials, item by item")
    test: str = Field(default="", description="what would have to be built in the simulator to try it")
    mechanism: Mechanism | None = Field(default=None, description="the mechanism as a change to a module: given, it "
                                        "is built and trialled like a swap; omitted, the idea is only recorded")


Move = Literal["params", "swap", "idea"]


class Proposal(_Model):
    move: Move
    base: str | None = Field(default=None, description="build on a promising body from the record by its id; "
                             "omit to change the current body")
    edits: list[Edit] = Field(default_factory=list)
    idea: Idea | None = None
    mechanism: str = Field(description="the failure mechanism this addresses, as the diagnosis names it")
    principle: str = Field(description="the principle that says this change fixes that mechanism")
    hypothesis: str = Field(description="why this change, and why the cheaper moves will not do")
    gains: list[str] = Field(default_factory=list, description="task groups predicted to start passing")
    risks: list[str] = Field(default_factory=list, description="held task groups this could break, and why")


class Requirement(_Model):
    """A need as System 2 states it (D-089): a metric over instruments and task facts, with its target."""

    id: str = Field(description="a short id, kept across revisions (R1, R2, ...)")
    statement: str = Field(description="the need in words")
    metric: str = Field(description="arithmetic over instrument readings, written as calls (squeeze_n(width_mm=35)), "
                        "and the task's facts by name (mass_g); see the instruments section")
    sense: Literal[">=", "<="] = Field(description="the metric must be at least, or at most, the target")
    marginal: float = Field(description="the value at which the need is just met")
    ideal: float | None = Field(default=None, description="the value with the margin you want (a safety factor)")
    tasks: list[str] = Field(default_factory=list, description="the task groups it is measured on, each task "
                             "facing the body (its facts and per-task readings); empty: the body alone")
    source: str = Field(description="where the need comes from: a task group, a rule of the brief, or a note id")


class ProposalBatch(_Model):
    proposals: list[Proposal] = Field(min_length=1, max_length=3)
    notes: str = ""
    requirements: list[Requirement] = Field(default_factory=list, description="the whole requirements list, "
                                            "revised; empty: keep the last one (D-089)")


def apply(body: Body, edits: Sequence[Edit], catalogue: dict[str, PartType], new_id: str) -> Body:
    """``body`` with ``edits`` applied.  Raises ``ValueError`` naming the first edit the catalogue rejects."""
    parts = dict(body.parts)
    for e in edits:
        cur = parts.get(e.slot)
        if cur is None and e.type is None:
            raise ValueError(f"slot {e.slot!r} does not exist; slots: {sorted(parts)}")
        tname = e.type or (cur.type if cur is not None else "")
        pt = catalogue.get(tname)
        if pt is None:
            raise ValueError(f"part type {tname!r} is not in the catalogue")
        if pt.slot != e.slot:
            raise ValueError(f"{tname} fills slot {pt.slot!r}, not {e.slot!r}")
        unknown = sorted(set(e.params) - set(pt.params))
        if unknown:
            raise ValueError(f"{tname} has no parameter {unknown[0]!r}; it has {sorted(pt.params)}")
        base = dict(cur.params) if cur is not None and cur.type == tname else {
            k: (p.lo + p.hi) / 2 for k, p in pt.params.items()}
        for k, v in e.params.items():
            p = pt.params[k]
            if not p.lo - 1e-9 <= v <= p.hi + 1e-9:
                raise ValueError(f"{tname}.{k}={v:g} is outside its range [{p.lo:g}, {p.hi:g}] {p.unit}")
            base[k] = float(v)
        parts[e.slot] = Part(type=tname, params=base)
    return Body(id=new_id, parent=body.id, parts=parts)


def edit_rung(body: Body, edits: Sequence[Edit]) -> Move:
    """``swap`` if any edit changes a slot's part type, else ``params``."""
    return "swap" if any(e.type is not None and (e.slot not in body.parts or body.parts[e.slot].type != e.type)
                         for e in edits) else "params"


# ---------------------------------------------------------------- the world contract
@runtime_checkable
class World(Protocol):
    """What a domain supplies.  ``solve`` runs System 1 on one task; ``brief`` states computed facts about a
    body (capability readouts, kinematic limits) without saying what to change; ``principles`` are the
    domain's physics in words; ``bom`` lists what a body needs; ``check`` lists reasons it cannot be built."""

    name: str
    description: str

    def tasks(self) -> list[Task]: ...

    def catalogue(self) -> list[PartType]: ...

    def stock(self) -> Body: ...

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome: ...

    def brief(self, body: Body, tasks: Sequence[Task]) -> str: ...

    def principles(self) -> list[str]: ...

    def bom(self, body: Body) -> list[BomLine]: ...

    def check(self, body: Body) -> list[str]: ...

    def manifest(self) -> dict[str, Any]: ...

    # Optional: ``part_facts(body, part_type) -> list[str]``, computed facts about a part type fitted to this
    # body (its reach, force, range at the ends of its parameters), shown in the catalogue.  Facts, not advice.
    # Optional (D-088): ``instruments() -> list[Instrument]`` and ``read(body, name, args, task) -> float``, the
    # quantities System 2 can measure on any draft; ``task_facts(task) -> dict[str, float]``, a task's numbers
    # by name (else every number in ``Task.data``).
