"""A library of mechanical-engineering skills, reused over time (D-096; the general framework, F8).

An engineer starts from machine elements and from what their own past work has proven.  System 2's seeds are the
elements (``mech.elements``), valid in any world.  Everything else here is earned, with provenance, in ``library/``:

- **mechanisms** (``mechanisms.json``): when a body is kept, each new sub-tree of bodies with a joint in it is entered
  under a structural signature (its tree, joints, part kinds and constraints, with names and numbers left out).  Seen
  in 2 or more (world, mount) contexts it is promoted; the numbers that differ between its contexts are its arguments;
- **methods** (``methods.json``): each requirement metric System 2 measured.  Used in more than one run it is promoted,
  and shown wherever its instruments exist;
- **notes** (``notebook.jsonl``): ``evolve/research``, which keeps them and counts what cited them.

A promoted mechanism reaches a graph as patch operations, so any world that builds a patch can use it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sea.evolve import instruments as I
from sea.evolve.core import Idea, Requirement, Use, World
from sea.mech.elements import ELEMENTS
from sea.mech.graph import Graph

MECHANISMS, METHODS = "mechanisms.json", "methods.json"
CONTEXTS = 2  # (world, mount) contexts that promote a mechanism
UNSEEN_RUNS = 10  # runs after which an unpromoted mechanism is removed
ON = "@on"  # the body a mechanism mounts on, in a stored template
_LISTS = ("loops", "meshes", "guides", "excludes", "linked", "actuators", "latches", "sensors", "carried", "contacts")


# ---------------------------------------------------------------- a mechanism's structure
def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _names(x: Any, key: str = "") -> Iterator[str]:
    """Every name a part of a graph gives itself."""
    if isinstance(x, dict):
        if isinstance(x.get("name"), str) and x["name"]:
            yield x["name"]
        for k, v in x.items():
            yield from _names(v, k)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _names(v, key)


def _strings(x: Any) -> Iterator[str]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def _mask(x: Any, names: set[str]) -> Any:
    """``x`` with its numbers and names blanked: what two instances of one structure have in common."""
    if _number(x):
        return "#"
    if isinstance(x, str):
        return "$" if x in names else x
    if isinstance(x, dict):
        return {k: _mask(v, names) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_mask(v, names) for v in x]
    return x


def _key(x: Any, names: set[str]) -> str:
    return json.dumps(_mask(x, names), sort_keys=True)


def _rename(x: Any, table: dict[str, str]) -> Any:
    if isinstance(x, str):
        return table.get(x, x)
    if isinstance(x, dict):
        return {k: _rename(v, table) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rename(v, table) for v in x]
    return x


def _template(d: dict[str, Any], root: str, mount: str, known: set[str]) -> dict[str, Any] | None:
    """The sub-tree of graph ``d`` under body ``root``, in canonical order and names, with what ties only it (and its
    mount) together; None if nothing in it moves."""
    children: dict[str, list[dict]] = {}
    by_name = {b["name"]: b for b in d["bodies"]}
    for b in d["bodies"]:
        children.setdefault(b["parent"], []).append(b)
    names = set(known)

    def subtree(b: dict) -> list[dict]:
        return [b, *(x for c in children.get(b["name"], []) for x in subtree(c))]

    def shape(b: dict) -> str:  # a body and everything under it, as a sort key
        return json.dumps([_key({**b, "parts": sorted(_key(p, names) for p in b["parts"])}, names),
                           sorted(shape(c) for c in children.get(b["name"], []))])

    def ordered(b: dict) -> list[dict]:
        return [b, *(x for c in sorted(children.get(b["name"], []), key=shape) for x in ordered(c))]

    bodies = [by_name[b["name"]] for b in ordered(by_name[root])]
    if not any(b["joints"] for b in bodies):
        return None
    local: set[str] = set()
    for b in subtree(by_name[root]):
        local.update(_names(b))
    def ours(e: Any) -> bool:
        refs = set(_strings(e)) & known
        return bool(refs & local) and refs <= local | {mount}

    els = {f: [e for e in d[f] if ours(e)] for f in _LISTS}
    table = {mount: ON}
    for i, b in enumerate(bodies):
        table[b["name"]] = f"$b{i}"
    bodies = [{**b, "parts": sorted(b["parts"], key=lambda p: _key(p, names))} for b in bodies]
    k = 0
    for b in bodies:
        for n in _names({"joints": b["joints"], "parts": b["parts"]}):
            if n not in table:
                table[n] = f"$n{k}"
                k += 1
    els = {f: sorted(v, key=lambda e: _key(e, names | set(_names(e)))) for f, v in els.items()}
    for f in _LISTS:
        for e in els[f]:
            for n in _names(e):
                if n not in table:
                    table[n] = f"$n{k}"
                    k += 1
    bodies = [{**b, **({"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]} if i == 0 else {})} for i, b in enumerate(bodies)]
    return _rename({"bodies": bodies, **els}, table)


def _leaves(x: Any, path: str = "") -> Iterator[tuple[str, float]]:
    """A template's numbers, each with where it is."""
    if _number(x):
        yield path, float(x)
    elif isinstance(x, dict):
        for k in sorted(x):
            yield from _leaves(x[k], f"{path}.{k}" if path else k)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}.{i}")


def _with(x: Any, values: dict[str, float], path: str = "") -> Any:
    if _number(x):
        return values.get(path, x)
    if isinstance(x, dict):
        return {k: _with(v, values, f"{path}.{k}" if path else k) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_with(v, values, f"{path}.{i}") for i, v in enumerate(x)]
    return x


def signature(template: dict[str, Any]) -> str:
    return "M-" + hashlib.sha1(json.dumps(_mask(template, set()), sort_keys=True).encode()).hexdigest()[:8]


def fragments(parent: Graph | None, g: Graph) -> list[tuple[str, dict[str, Any]]]:
    """What ``g`` added to ``parent``: each new sub-tree with a joint in it, as (the body it mounts on, its template)."""
    old = set(parent.new_bodies()) if parent is not None else set()
    d = json.loads(g.model_dump_json())
    added = {b["name"] for b in d["bodies"] if not b["stock"] and b["name"] not in old}
    known = set(_names(d["bodies"]))
    roots = [b for b in d["bodies"] if b["name"] in added and b["parent"] not in added]
    out = []
    for b in roots:
        t = _template(d, b["name"], b["parent"], known)
        if t is not None:
            out.append((b["parent"], t))
    return out


# ---------------------------------------------------------------- the store
class Instance(BaseModel):
    """Where a mechanism was kept: its provenance."""

    model_config = ConfigDict(extra="forbid")
    world: str
    mount: str
    run: str
    iteration: int
    body: str
    values: dict[str, float]


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    template: dict[str, Any]
    instances: list[Instance]
    last_seen: int = 0  # the store's run count when last kept

    def contexts(self) -> set[tuple[str, str]]:
        return {(i.world, i.mount) for i in self.instances}

    def promoted(self) -> bool:
        return len(self.contexts()) >= CONTEXTS

    def arguments(self) -> dict[str, tuple[float, float]]:
        """The numbers that differed between its contexts, with the range seen."""
        out = {}
        for k in self.instances[0].values:
            vs = [i.values[k] for i in self.instances]
            if max(vs) - min(vs) > 1e-9:
                out[k] = (min(vs), max(vs))
        return out

    def ops(self, use: Use) -> list[dict[str, Any]]:
        """This mechanism at ``use`` as patch operations.  An argument left out takes the middle of its range; a value
        outside the range seen is allowed, and the laws and the trial judge it."""
        args = self.arguments()
        extra = sorted(set(use.params) - set(args))
        if extra:
            raise ValueError(f"{self.id} has no argument {extra}; it has {sorted(args)}")
        values = {k: use.params.get(k, (lo + hi) / 2) for k, (lo, hi) in args.items()}
        t = _with(self.template, {**self.instances[0].values, **values})
        table = {ON: use.on}
        for n in {s for s in _strings(t) if s.startswith("$")}:
            table[n] = f"{use.prefix}_{n[1:]}"
        t = _rename(t, table)
        first = {**t["bodies"][0], "parent": use.on, "pos": list(use.at), "quat": list(use.quat)}
        ops = [{"op": "add", "path": f"bodies/{b['name']}", "value": b} for b in [first, *t["bodies"][1:]]]
        return ops + [{"op": "add", "path": f, "value": e} for f in _LISTS for e in t[f]]


class Method(BaseModel):
    """A requirement metric System 2 wrote and the harness could measure."""

    model_config = ConfigDict(extra="forbid")
    metric: str
    statement: str
    instruments: list[str]
    worlds: list[str]
    runs: list[str]


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """One writer at a time: runs in parallel share the library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f".{path.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _save(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def load(root: Path) -> tuple[int, dict[str, Entry]]:
    path = root / MECHANISMS
    if not path.exists():
        return 0, {}
    d = json.loads(path.read_text())
    return d["runs"], {k: Entry.model_validate(v) for k, v in d["entries"].items()}


def _store(root: Path, runs: int, entries: dict[str, Entry]) -> None:
    _save(root / MECHANISMS, {"runs": runs, "entries": {k: json.loads(v.model_dump_json()) for k, v in entries.items()}})


def graph_of(world: Any, body: Any) -> Graph | None:
    text = world.mechanism_text(body) if hasattr(world, "mechanism_text") else ""
    return Graph.model_validate_json(text) if text else None


def keep(root: Path, world: Any, parent: Any, body: Any, name: str, run: str, iteration: int) -> list[str]:
    """Enter what the kept ``body`` added to ``parent``; returns the ids entered."""
    g = graph_of(world, body)
    if g is None:
        return []
    ids = []
    with _locked(root / MECHANISMS):
        runs, entries = load(root)
        for mount, t in fragments(graph_of(world, parent), g):
            sig = signature(t)
            inst = Instance(world=world.name, mount=mount, run=run, iteration=iteration, body=body.id, values=dict(_leaves(t)))
            e = entries.get(sig)
            if e is None:
                e = Entry(id=sig, name=name, template=t, instances=[])
            entries[sig] = e.model_copy(update={"name": name, "instances": [*e.instances, inst], "last_seen": runs})
            ids.append(sig)
        if ids:
            _store(root, runs, entries)
    return ids


def finish_run(root: Path) -> list[str]:
    """One run has ended: remove what has gone unseen too long and was never promoted; returns what was removed."""
    with _locked(root / MECHANISMS):
        runs, entries = load(root)
        runs += 1
        gone = [k for k, e in entries.items() if not e.promoted() and runs - e.last_seen - 1 >= UNSEEN_RUNS]
        _store(root, runs, {k: e for k, e in entries.items() if k not in gone})
    return gone


def promoted(root: Path) -> dict[str, Entry]:
    return {k: e for k, e in load(root)[1].items() if e.promoted()}


def expand(idea: Idea, entries: dict[str, Entry]) -> Idea:
    """``idea`` with each promoted mechanism it uses written out as patch operations, ahead of its own."""
    mech = idea.mechanism
    if mech is None or not any(u.element in entries for u in mech.uses):
        return idea
    ops = [o for u in mech.uses if u.element in entries for o in entries[u.element].ops(u)]
    keep_uses = [u for u in mech.uses if u.element not in entries]
    return idea.model_copy(update={"mechanism": mech.model_copy(update={"uses": keep_uses, "patch": [*ops, *mech.patch]})})


# ---------------------------------------------------------------- methods
def instruments_of(metric: str) -> list[str]:
    return sorted(set(re.findall(r"([A-Za-z_]\w*)\(", metric)) - set(I._FUNCS))


def remember(root: Path, reqs: Sequence[Requirement], rows: Sequence[Any], world: str, run: str) -> None:
    """Record the metrics that could be measured."""
    measured = {r.id for r in rows if not r.error}
    path = root / METHODS
    with _locked(path):
        store = json.loads(path.read_text()) if path.exists() else {}
        for r in reqs:
            if r.id not in measured or not instruments_of(r.metric):
                continue
            key = " ".join(r.metric.split())
            m = store.get(key, {"metric": key, "statement": r.statement, "instruments": instruments_of(r.metric),
                                "worlds": [], "runs": []})
            m["worlds"] = sorted({*m["worlds"], world})
            m["runs"] = sorted({*m["runs"], run})
            store[key] = m
        _save(path, store)


def methods(root: Path, world: World) -> list[Method]:
    """The promoted methods whose instruments exist here."""
    path = root / METHODS
    if not path.exists():
        return []
    here = {i.name for i in I.instruments(world)}
    all_ = [Method.model_validate(m) for m in json.loads(path.read_text()).values()]
    return [m for m in all_ if len(m.runs) > 1 and set(m.instruments) <= here]


# ---------------------------------------------------------------- what System 2 is shown
def text(root: Path, world: World) -> str:
    els = [f"- {n} ({', '.join(f'{k} {lo:g}..{hi:g}' for k, (lo, hi) in e.params.items())}): {e.provides}; {e.joints}"
           for n, e in ELEMENTS.items()]
    out = ["A mechanism's `uses` mounts these on any body of its graph, each as {element, prefix, on, at, quat, params}; "
           "`<p>` below is its prefix.  They are added after its module and before its patch, and the patch connects "
           "them, drives them and changes them.  Each still has to pass the laws and earn its place in a trial.",
           "Machine elements:", *els]
    entries = promoted(root)
    if entries:
        out.append("Mechanisms your past runs proved, in more than one context (`element` is the id; `params` its arguments):")
        for e in entries.values():
            args = ", ".join(f"{k} {lo:g}..{hi:g}" for k, (lo, hi) in e.arguments().items()) or "none"
            where = ", ".join(sorted(f"{w}/{m}" for w, m in e.contexts()))
            out.append(f"- {e.id} {e.name!r}: {len(e.template['bodies'])} bodies; kept on {where}; arguments: {args}")
    ms = methods(root, world)
    if ms:
        out.append("Metrics your past runs measured, which this world's instruments can read:")
        out += [f"- {m.metric}: {m.statement} (runs: {len(m.runs)}; worlds: {', '.join(m.worlds)})" for m in ms]
    return "\n".join(out)
