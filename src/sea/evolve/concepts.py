"""Concepts developed before they are judged (D-091; the general framework, F4).

An engineer keeps several concepts alive, gives each a budget to develop it, and narrows on evidence (set-based
design: Sobek, Ward and Liker).  They start from the functions a design must perform and the options for each
(a morphological chart: Pahl and Beitz), compare candidates against the incumbent on measured requirements
(Pugh's matrix, without its guessed ratings), and predict how each could fail (FMEA).

System 2 writes the functions, the concepts and the failure modes.  The harness holds no design knowledge here.  It:

- keeps the set: a new structure (a built mechanism, a swapped part type) that reaches a trial or is refused by a
  law is a concept, developed by proposing with ``base`` set to its id;
- narrows it on computed evidence only: an allowance spent without progress, a developed concept another live one
  dominates on every requirement and on the groups it gained, or one left undeveloped;
- computes the comparison from the requirements' readings, and rates each predicted failure mode against what
  the laws and the trial then found.  None of this keeps or refuses a body: the trial does that.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from sea.evolve.core import Body, FailureMode, Function, Requirement

N_LIVE = 3  # concepts alive at once
TRIALS, REFUSALS = 4, 6  # a concept's allowance between looks at its progress
IDLE = 5  # iterations undeveloped before it gives up its place
BEFORE_DOMINANCE = 2  # developments before a concept can be judged against another: it is developed first
TOL = 0.01  # of the larger of the value and the target: a smaller move in a reading is not progress


class Step(BaseModel):
    """One development of a concept: what it was, and what the laws or the trial found."""

    model_config = ConfigDict(extra="forbid")
    iteration: int
    body: str
    edit: str
    result: str
    tried: bool
    gained: list[str] = []
    lost: list[str] = []


class Concept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    body: Body  # its last development: a patch continues from here
    opened: int
    last: int
    status: Literal["live", "dropped", "kept"] = "live"
    why: str = ""
    steps: list[Step] = []
    options: dict[str, str] = {}  # function -> the option it takes (the chart's row)
    trials: int = 0  # spent from the allowance since it was last renewed
    refusals: int = 0
    gained: list[str] = []  # every group any of its trials gained
    progress: list[str] = []  # groups gained since the allowance was last renewed
    baseline: dict[str, float] = {}  # readings by metric when the allowance was last renewed
    rows: dict[str, float | None] = {}  # readings by metric on its current body
    first_gain: int | None = None  # how many developments it took


def _key(r: Requirement) -> str:
    return f"{r.metric} {r.sense}"


def sign(r: Requirement, ref: float | None, v: float | None) -> str:
    """Whether ``v`` is better (+), the same (0) or worse (-) than ``ref`` by the requirement's sense; ? unmeasured."""
    if ref is None or v is None:
        return "?"
    if abs(v - ref) <= TOL * max(abs(ref), abs(r.marginal)) + 1e-9:
        return "0"
    return "+" if (v > ref) == (r.sense == ">=") else "-"


class ConceptSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concepts: list[Concept] = []

    def live(self) -> list[Concept]:
        return [c for c in self.concepts if c.status == "live"]

    def bases(self) -> dict[str, Body]:
        return {c.id: c.body for c in self.live()}

    def base_refusal(self, base: str | None) -> str:
        """Why a proposal cannot build on ``base``, if it names a concept no longer in the set."""
        c = next((c for c in self.concepts if c.id == base), None)
        if c is None or c.status == "live":
            return ""
        if c.status == "kept":
            return f"concept {c.id} is the current body now: omit `base` to develop it"
        return f"concept {c.id} was dropped ({c.why}): open a new concept to try that structure again"

    def full_refusal(self) -> str:
        """Why no new concept can open now."""
        live = self.live()
        if len(live) < N_LIVE:
            return ""
        return (f"{N_LIVE} concepts are alive ({', '.join(c.id for c in live)}): develop one by setting `base` to its "
                "id; a place frees when one's allowance is spent without progress, when another dominates it, or "
                f"after {IDLE} iterations undeveloped")

    def record(self, base: str | None, opening: bool, name: str, cand: Body, entry: dict[str, Any], it: int,
               options: dict[str, str]) -> str | None:
        """Enter one proposal's outcome: a development of the concept ``base`` names, or a new concept if it opens
        one and reached a trial or a law's refusal.  Returns the concept's id, if any."""
        tried, laws = bool(entry.get("tried")), bool(entry.get("laws"))
        c = next((c for c in self.live() if c.id == base), None)
        if c is None:
            if not opening or not (tried or laws):
                return None
            c = Concept(id=f"C{len(self.concepts) + 1}", name=name, body=cand, opened=it, last=it, options=options)
            self.concepts.append(c)
        else:
            c.trials += tried
            c.refusals += not tried
            c.options = {**c.options, **options}
            if tried or laws:  # a refusal for money or a repeat changes nothing about the structure
                c.body = cand
        c.last = it
        gained = list(entry.get("gained", []))
        c.steps.append(Step(iteration=it, body=cand.id, edit=entry.get("edit", ""), result=entry.get("result", "")[:300],
                            tried=tried, gained=gained, lost=list(entry.get("lost", []))))
        c.gained = sorted(set(c.gained) | set(gained))
        c.progress = sorted(set(c.progress) | set(gained))
        if gained and c.first_gain is None:
            c.first_gain = len(c.steps)
        return c.id

    def refresh(self, reqs: Sequence[Requirement], measure: Callable[[Body], Sequence[Any]]) -> None:
        """Each live concept's readings on its current body.  A requirement first seen sets its baseline."""
        for c in self.live():
            c.rows = {_key(r): row.value for r, row in zip(reqs, measure(c.body), strict=True)}
            c.baseline.update({k: v for k, v in c.rows.items() if k not in c.baseline and v is not None})

    def _improved(self, c: Concept, reqs: Sequence[Requirement]) -> list[str]:
        return [r.id for r in reqs if _key(r) in c.baseline
                and sign(r, c.baseline[_key(r)], c.rows.get(_key(r))) == "+"]

    def _dominator(self, c: Concept, reqs: Sequence[Requirement]) -> Concept | None:
        """A live concept at least as good as ``c`` on every requirement, better on one, holding every group it gained."""
        if len(c.steps) <= BEFORE_DOMINANCE or not reqs:
            return None
        for o in self.live():
            if o.id == c.id or not set(c.gained) <= set(o.gained):
                continue
            signs = [sign(r, c.rows.get(_key(r)), o.rows.get(_key(r))) for r in reqs]
            if "+" in signs and "-" not in signs and "?" not in signs:
                return o
        return None

    def narrow(self, it: int, reqs: Sequence[Requirement]) -> list[str]:
        """Drop what the evidence says to, with why; renew the allowance of what made progress.  Returns the ids dropped."""
        out = []
        for c in self.live():
            why = ""
            if c.trials >= TRIALS or c.refusals >= REFUSALS:
                up = self._improved(c, reqs)
                if c.progress or up:
                    c.trials, c.refusals, c.progress = 0, 0, []
                    c.baseline = {k: v for k, v in c.rows.items() if v is not None}
                else:
                    why = (f"allowance spent ({c.trials} trials, {c.refusals} refusals) with no group gained and no "
                           "requirement improved")
            if not why and (o := self._dominator(c, reqs)) is not None:
                why = f"{o.id} is as good on every requirement, better on one, and holds every group it gained"
            if not why and it - c.last >= IDLE:
                why = f"undeveloped for {it - c.last} iterations"
            if why:
                c.status, c.why = "dropped", why
                out.append(c.id)
        return out

    def keep(self, ids: Sequence[str]) -> None:
        for c in self.live():
            if c.id in ids or c.body.id in ids:
                c.status = "kept"


def opens(built: bool, module: str | None, rung: str, base: str | None, live: Sequence[str]) -> bool:
    """Whether a proposal starts a new structure: a built mechanism or a swapped part type, not on a live concept and
    not continuing a mechanism's own graph."""
    return (rung == "swap" or (built and module != "self")) and base not in live


def matrix(reqs: Sequence[Requirement], incumbent: Sequence[Any], columns: Sequence[tuple[str, Sequence[Any]]]) -> str:
    """Each column against the incumbent on every requirement, with its reading and a computed +, 0 or -."""
    rows = []
    for k, r in enumerate(reqs):
        ref = incumbent[k].value
        cells = [f"{name} " + (f"{col[k].value:.4g} " if col[k].value is not None else "") + sign(r, ref, col[k].value)
                 for name, col in columns]
        rows.append(f"- {r.id} `{r.metric}` {r.sense} {r.marginal:g}: current "
                    f"{f'{ref:.4g}' if ref is not None else 'unmeasured'}; " + "; ".join(cells))
    return "\n".join(rows)


def chart(functions: Sequence[Function], known: set[str], concepts: Sequence[Concept]) -> str:
    """The functions and their options: which exist to build from, and which no concept has taken yet."""
    taken: dict[str, set[str]] = {}
    for c in concepts:
        for f, o in c.options.items():
            taken.setdefault(f, set()).add(o)
    rows = []
    for f in functions:
        cells = [o + ("" if o in known else " (not in the catalogue or modules)")
                 + ("" if o in taken.get(f.name, set()) else " [untried]") for o in f.options]
        rows.append(f"- {f.name}: " + "; ".join(cells))
    return "\n".join(rows)


def rate(modes: Sequence[FailureMode], entry: dict[str, Any],
         groups: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Each predicted mode against what came of the proposal: whether its signature showed in the laws' refusal or in
    the trial's diagnoses, and how many held groups went with it; then the failures no mode predicted (``groups``:
    each task's group)."""
    refused = entry["result"] if not entry.get("tried") else ""
    failed = {o["task"]: o for o in entry.get("outcomes", {}).values() if not o["success"]}
    lost = entry.get("lost", [])
    texts = [f"{o['mechanism']} {o['detail']}".lower() for o in failed.values()]
    laws = list(entry.get("laws", []))
    rated = []
    seen_laws, seen_labels = set(), set()
    for m in modes:
        sig = m.signature.strip().lower()
        where = "" if not sig else "laws" if sig in refused.lower() else "trial" if any(sig in t for t in texts) else ""
        seen_laws |= {x for x in laws if sig and sig in x.lower()}
        seen_labels |= {o["mechanism"] for o in failed.values() if sig and sig in f"{o['mechanism']} {o['detail']}".lower()}
        rated.append({**m.model_dump(), "occurred": bool(where), "detected": where,
                      "severity": len(lost) if where == "trial" else 0})
    labels = {o["mechanism"] for t, o in failed.items() if groups.get(t) in lost}
    return rated, sorted((set(laws) - seen_laws) | (labels - seen_labels))


def calibration(history: Sequence[dict[str, Any]]) -> str:
    """Modes predicted against what happened, over the record."""
    rated = [m for h in history for m in h.get("failure_modes", [])]
    unpredicted = sum(len(h.get("unpredicted_failures", [])) for h in history)
    if not rated and not unpredicted:
        return ""
    hit = sum(m["occurred"] for m in rated)
    laws = sum(m["detected"] == "laws" for m in rated)
    sev = sum(m["severity"] for m in rated)
    return (f"Failure modes: {hit} of the {len(rated)} you predicted occurred ({laws} showed in the laws, "
            f"{hit - laws} only in a trial, costing {sev} held groups); {unpredicted} failures came that no mode predicted.")


def text(cs: ConceptSet, reqs: Sequence[Requirement], incumbent: Sequence[Any], functions: Sequence[Function],
         known: set[str], graph: Callable[[Body], str] | None = None) -> str:
    """The set for the prompt: how it works, each live concept with its lineage and comparison, what left it."""
    out = [f"A new structure (a built mechanism, or a swapped part type) that reaches a trial or is refused by a law "
           f"becomes a concept.  Up to {N_LIVE} live at once, each allowed {TRIALS} trials and {REFUSALS} refusals "
           "before the harness looks at its progress.  Develop one by proposing with `base` set to its id; in a "
           "mechanism, module `self` with a patch continues its last graph.  A concept leaves the set when its "
           "allowance is spent with no group gained and no requirement improved, when after "
           f"{BEFORE_DOMINANCE} developments another live one is as good on every requirement and holds every group it "
           f"gained, or after {IDLE} iterations undeveloped.  Your answer's `functions`, when given, replaces your "
           "function list: the functions the design must perform and the options you consider for each; a proposal's "
           "`options` says which it takes.  A proposal's `failure_modes` are scored against the laws and the trial."]
    live = cs.live()
    for c in live:
        out.append(f"## {c.id} {c.name}: {c.body.describe()}; {c.trials} of {TRIALS} trials, {c.refusals} of {REFUSALS} "
                   f"refusals since its allowance began; groups gained {', '.join(c.gained) or 'none'}")
        out += [f"- iteration {s.iteration + 1}: {s.edit}: {s.result}" for s in c.steps[-4:]]
        if graph is not None:
            out.append(graph(c.body))
    if not live:
        out.append("No concept is alive.")
    if live and reqs:
        cols = [(c.id, [_Cell(c.rows.get(_key(r))) for r in reqs]) for c in live]
        out.append("Against the current body (+ better, 0 the same, - worse, ? unmeasured):\n"
                   + matrix(reqs, incumbent, cols))
    gone = [c for c in cs.concepts if c.status != "live"]
    if gone:
        out.append("Left the set:\n" + "\n".join(
            f"- {c.id} {c.name}: " + ("kept: it is the current body" if c.status == "kept" else f"dropped: {c.why}")
            for c in gone))
    if functions:
        out.append("Your functions and options:\n" + chart(functions, known, cs.concepts))
    return "\n".join(out)


class _Cell:
    def __init__(self, value: float | None):
        self.value = value


def summary(cs: ConceptSet) -> dict[str, Any]:
    """What the set did over a run: concepts opened, how far each was developed, and how each left."""
    return {"opened": len(cs.concepts), "live": len(cs.live()),
            "kept": sum(c.status == "kept" for c in cs.concepts),
            "dropped": {c.id: c.why for c in cs.concepts if c.status == "dropped"},
            "developments": {c.id: len(c.steps) for c in cs.concepts},
            "reached_trial": sum(any(s.tried for s in c.steps) for c in cs.concepts),
            "first_gain": {c.id: c.first_gain for c in cs.concepts if c.first_gain is not None}}
