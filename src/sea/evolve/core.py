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


class Body(_Model):
    id: str = "stock"
    parent: str | None = None
    parts: dict[str, Part]

    def key(self) -> str:
        """Content hash: two bodies with the same parts are the same body, whatever their ids."""
        blob = json.dumps({s: p.model_dump() for s, p in sorted(self.parts.items())}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        out = []
        for slot, p in sorted(self.parts.items()):
            ps = ", ".join(f"{k}={v:g}" for k, v in sorted(p.params.items()))
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


class Idea(_Model):
    name: str
    description: str = Field(description="the part and how it works physically")
    needs: list[str] = Field(default_factory=list, description="items it needs: manifest item names, or new items "
                             "with an estimated price")
    test: str = Field(default="", description="what would have to be built in the simulator to try it")


Move = Literal["params", "swap", "idea"]


class Proposal(_Model):
    move: Move
    edits: list[Edit] = Field(default_factory=list)
    idea: Idea | None = None
    mechanism: str = Field(description="the failure mechanism this addresses, as the diagnosis names it")
    principle: str = Field(description="the principle that says this change fixes that mechanism")
    hypothesis: str = Field(description="why this change, and why the cheaper moves will not do")
    gains: list[str] = Field(default_factory=list, description="task groups predicted to start passing")
    risks: list[str] = Field(default_factory=list, description="held task groups this could break, and why")


class ProposalBatch(_Model):
    proposals: list[Proposal] = Field(min_length=1, max_length=3)
    notes: str = ""


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
