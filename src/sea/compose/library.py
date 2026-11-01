"""Primitive library: seed load, applicability, promotion and pruning (spec 3.5, 8.7).

The store maps names to ``PrimitiveDef``.  Composite primitives (seed programs and promoted
programs) carry their ``body``.  ``provides`` strings use ``|`` between control classes and an
optional ``[a/b]`` handle filter, e.g. ``door[tab/pull]|drawer[tab/pull]``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sea.compose import dsl
from sea.contracts import D_DUP, RETIRE_AFTER, ControlClass, ControlSpec, Node, PrimitiveDef, Program

ROOT = Path(__file__).resolve().parents[3]
SEED_PATH = ROOT / "configs" / "primitives.yaml"
_ITEM = re.compile(r"^(?P<cls>[a-z_]+)(?:\[(?P<handles>[a-z_/]+)\])?$")


def parse_provides(text: str) -> list[tuple[str, set[str] | None]]:
    out = []
    for item in text.split("|"):
        item = item.strip()
        if not item:
            continue
        m = _ITEM.match(item)
        if m is None:
            raise ValueError(f"bad provides item {item!r}")
        hs = set(m.group("handles").split("/")) if m.group("handles") else None
        out.append((m.group("cls"), hs))
    return out


def provides_control(defn: PrimitiveDef, control: ControlSpec) -> bool:
    for value in defn.provides.values():
        for cls, handles in parse_provides(value):
            if cls == control.cls and (handles is None or control.handle in handles):
                return True
    return False


def requires_contacts(defn: PrimitiveDef) -> set[str] | None:
    """Contact types a motor primitive accepts (None = any)."""
    c = defn.requires.get("contact", "any")
    if not isinstance(c, str) or c == "any":
        return None
    return set(c.split("|"))


def load_seed(path: Path = SEED_PATH) -> dict[str, PrimitiveDef]:
    data = yaml.safe_load(path.read_text())
    lib: dict[str, PrimitiveDef] = {}
    for m in data["motor"]:
        lib[m["name"]] = PrimitiveDef(
            name=m["name"], kind="motor", doc=m["doc"], requires=m.get("requires", {}),
            provides=m.get("provides", {}), params={k: (float(v[0]), float(v[1])) for k, v in m["params"].items()},
        )
    for c in data["checks"]:
        lib[c["name"]] = PrimitiveDef(name=c["name"], kind="check", doc=c["doc"])
    for p in data["programs"]:
        root = dsl.parse(p["text"])
        prog = Program(id=p["name"], target_cls=p["target_cls"], root=root, source="seed")
        lib[p["name"]] = PrimitiveDef(name=p["name"], kind="composite", doc=p["doc"], provides=p["provides"],
                                      requires=composite_requires(root, lib), body=prog)
    return lib


def composite_requires(root: Node, lib: dict[str, PrimitiveDef]) -> dict[str, str | float]:
    """Intersection of the contact requirements of the motor primitives a program uses."""
    allowed: set[str] | None = None
    min_force = 0.0
    for name in dsl.primitive_names(root):
        d = lib.get(name)
        if d is None:
            continue
        rc = requires_contacts(d)
        if rc is not None:
            allowed = rc if allowed is None else allowed & rc
        mf = d.requires.get("min_force")
        if isinstance(mf, (int, float)):
            min_force = max(min_force, float(mf))
    out: dict[str, str | float] = {"contact": "any" if allowed is None else "|".join(sorted(allowed))}
    if min_force:
        out["min_force"] = min_force
    return out


@dataclass
class Library:
    """Mutable primitive store with usage stats, promotion and pruning."""

    prims: dict[str, PrimitiveDef]
    size_budget: int = 40
    retired: list[PrimitiveDef] = field(default_factory=list)
    last_used: dict[str, int] = field(default_factory=dict)
    uses_log: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # program id -> (panel, control)

    @classmethod
    def seed(cls, size_budget: int = 40, path: Path = SEED_PATH) -> Library:
        return cls(load_seed(path), size_budget)

    def __contains__(self, name: str) -> bool:
        return name in self.prims

    def get(self, name: str) -> PrimitiveDef | None:
        return self.prims.get(name)

    def programs(self) -> list[PrimitiveDef]:
        return [p for p in self.prims.values() if p.kind == "composite" and p.body is not None]

    def programs_for(self, control: ControlSpec) -> list[PrimitiveDef]:
        """Composite programs whose ``provides`` covers the control (seed programs first)."""
        order = {"seed": 0, "promoted": 1, "s2": 2}
        cands = [p for p in self.programs() if provides_control(p, control)]
        return sorted(cands, key=lambda p: (order.get(p.body.source if p.body else "s2", 3), p.name))

    def add(self, defn: PrimitiveDef) -> None:
        self.prims[defn.name] = defn

    def record_use(self, name: str, passed: bool, iteration: int, panel_id: str = "", control_id: str = "") -> None:
        d = self.prims.get(name)
        if d is None:
            return
        self.prims[name] = d.model_copy(update={"uses": d.uses + 1, "passes": d.passes + int(passed)})
        self.last_used[name] = iteration
        if passed:
            self.uses_log.setdefault(name, []).append((panel_id, control_id))

    def to_list(self) -> list[dict]:
        return [p.model_dump(mode="json") for p in self.prims.values()]

    # ---------------------------------------------------------------- promotion (3.5)
    def promotion_ready(self, program_name: str, k_promote: int) -> bool:
        """Passed on >= k_promote distinct controls across >= 2 panels."""
        uses = set(self.uses_log.get(program_name, []))
        panels = {p for p, _ in uses}
        return len(uses) >= k_promote and len(panels) >= 2

    def promote(self, program: Program, name: str, doc: str, variants: Iterable[Node], iteration: int,
                is_duplicate: Callable[[Program], bool] | None = None) -> PrimitiveDef | None:
        """Abstract ``variants`` (bodies that passed) into one composite primitive.

        Numeric arguments that differ between variants become parameters (``$p<i>`` placeholders,
        range = observed min..max); identical ones stay fixed.  Returns None when the program is a
        duplicate (``is_duplicate``, e.g. program-tree distance < d_dup) or the name is taken.
        """
        if name in self.prims:
            return None
        if is_duplicate is not None and is_duplicate(program):
            return None
        body, params = abstract(list(variants) or [program.root])
        prog = Program(id=name, target_cls=program.target_cls, root=body, source="promoted")
        provides = {"actuates": program.target_cls}
        defn = PrimitiveDef(name=name, kind="composite", doc=doc, requires=composite_requires(body, self.prims),
                            provides=provides, params=params, body=prog, created_iter=iteration)
        self.prims[name] = defn
        self.last_used[name] = iteration
        return defn

    # ---------------------------------------------------------------- pruning
    def prune(self, iteration: int, protected: set[str] | None = None) -> list[str]:
        """Retire promoted primitives unused for RETIRE_AFTER iterations or dominated by another
        (same provides, lower cost, pass rate >= theirs), then enforce the size budget."""
        protected = protected or set()
        gone: list[str] = []
        promoted = [p for p in self.prims.values() if p.body is not None and p.body.source == "promoted"
                    and p.name not in protected]
        for p in promoted:
            idle = iteration - self.last_used.get(p.name, p.created_iter)
            if idle >= RETIRE_AFTER or self._dominated(p):
                gone.append(p.name)
        while len(self.prims) - len(gone) > self.size_budget:
            rest = [p for p in promoted if p.name not in gone]
            if not rest:
                break
            worst = min(rest, key=lambda p: (pass_rate(p), self.last_used.get(p.name, p.created_iter), p.name))
            gone.append(worst.name)
        for n in gone:
            self.retired.append(self.prims.pop(n))
        return gone

    def _dominated(self, p: PrimitiveDef) -> bool:
        for q in self.prims.values():
            if q.name == p.name or q.kind != "composite" or q.provides != p.provides or q.uses == 0:
                continue
            if program_cost(q) < program_cost(p) and pass_rate(q) >= pass_rate(p):
                return True
        return False


def pass_rate(p: PrimitiveDef) -> float:
    return p.passes / p.uses if p.uses else 0.0


def program_cost(p: PrimitiveDef) -> float:
    """Proxy execution cost: number of primitive calls plus tool swaps (each swap = 5 s)."""
    if p.body is None:
        return 1.0
    return float(len(dsl.primitive_names(p.body.root)) + 5 * len(dsl.tool_indices(p.body.root)))


def abstract(variants: list[Node]) -> tuple[Node, dict[str, tuple[float, float]]]:
    """Anti-unify structurally identical program trees over their numeric arguments."""
    base = variants[0]
    texts = [dsl.pretty(v) for v in variants]
    shapes = {re.sub(r"-?\d+\.\d+(e-?\d+)?", "#", t) for t in texts}
    if len(shapes) > 1:  # structures differ: keep the first verbatim
        return base, {}
    params: dict[str, tuple[float, float]] = {}
    counter = [0]

    def walk(nodes: list[Node]) -> Node:
        n0 = nodes[0]
        update: dict = {}
        if n0.prim is not None and n0.op == "prim":
            args = dict(n0.prim.args)
            for k, v in n0.prim.args.items():
                vals = [n.prim.args.get(k) for n in nodes if n.prim is not None]
                if isinstance(v, float) and len({float(x) for x in vals if isinstance(x, float)}) > 1:
                    pname = f"p{counter[0]}"
                    counter[0] += 1
                    nums = [float(x) for x in vals if isinstance(x, float)]
                    params[pname] = (min(nums), max(nums))
                    args[k] = f"${pname}"
            update["prim"] = n0.prim.model_copy(update={"args": args})
        if n0.children:
            update["children"] = [walk([n.children[i] for n in nodes]) for i in range(len(n0.children))]
        return n0.model_copy(update=update)

    return walk(variants), params


def target_classes(defn: PrimitiveDef) -> list[ControlClass]:
    out: list[ControlClass] = []
    for v in defn.provides.values():
        out.extend(c for c, _ in parse_provides(v))  # type: ignore[misc]
    return out


DUP_DISTANCE = D_DUP
