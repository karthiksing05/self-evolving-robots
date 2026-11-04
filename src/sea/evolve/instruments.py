"""What the harness can measure about a body, the same way in every world (D-088; the general framework, F1).

A world declares its **instruments**: named quantities it can compute on a body, or on a body facing one task
(an opening, a squeeze, an edge it climbs, a formant it reaches).  An instrument is a reading with a unit,
never advice.  The harness adds the readings every world has: what a body costs, how long it takes to make
and to arrive, and how many of the world's laws refuse it.  A task's own numbers (an object's width, a step's
height) are its **facts**.

System 2 gathers the data it needs through these: it reads instruments on drafts (``evolve/tools.py``), sweeps
a parameter through them, and states its requirements as expressions over them (``evolve/requirements.py``).
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Sequence
from typing import Any

from sea.evolve.core import Body, Instrument, Task, World

GENERIC = [
    Instrument(name="cost_usd", unit="$", doc="what building this body from the current one costs to buy"),
    Instrument(name="fab_minutes", unit="min", doc="machine time to make this body's new parts"),
    Instrument(name="lead_days", unit="days", doc="the longest wait for a part this body needs"),
    Instrument(name="refusals", unit="", doc="how many build rules and mechanism laws refuse this body"),
]


def instruments(world: World) -> list[Instrument]:
    return [*GENERIC, *(world.instruments() if hasattr(world, "instruments") else [])]


def facts(world: World, task: Task) -> dict[str, float]:
    """A task's numbers by name: the world's own list, else every number in its data (nested names joined by
    dots, list items by index)."""
    if hasattr(world, "task_facts"):
        return dict(world.task_facts(task))
    out: dict[str, float] = {}

    def walk(prefix: str, v: Any, depth: int) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, int | float):
            out[prefix] = float(v)
        elif isinstance(v, dict) and depth < 3:
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x, depth + 1)
        elif isinstance(v, list | tuple) and depth < 3:
            for i, x in enumerate(v):
                walk(f"{prefix}[{i}]", x, depth + 1)

    walk("", task.data, 0)
    return out


def read(world: World, body: Body, name: str, args: dict[str, Any] | None = None, task: Task | None = None,
         base: Body | None = None, manifest: dict[str, Any] | None = None, spent: Any = None) -> float:
    """One reading.  ``base``, ``manifest`` and ``spent`` price a body as a change from ``base``.  Raises
    ``ValueError`` naming what is wrong with the request."""
    args = dict(args or {})
    if name in ("cost_usd", "fab_minutes", "lead_days"):
        from sea.evolve.resources import check

        cur = base or world.stock()
        res = check(body, world.bom(body), cur, world.bom(cur), manifest or world.manifest(), spent)
        return float({"cost_usd": res.usd, "fab_minutes": res.fab_minutes, "lead_days": res.lead_days}[name])
    if name == "refusals":
        return float(len(world.check(body)))
    known = {i.name: i for i in instruments(world)}
    if name not in known:
        raise ValueError(f"no instrument {name!r}; the instruments are {sorted(known)}")
    if known[name].task and task is None:
        raise ValueError(f"{name} reads a body facing a task: give its task id")
    unknown = sorted(set(args) - set(known[name].args))
    if unknown:
        raise ValueError(f"{name} takes {sorted(known[name].args) or 'no arguments'}, not {unknown}")
    return float(world.read(body, name, args, task))


# ---------------------------------------------------------------- expressions over readings
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan": math.atan,
          "radians": math.radians, "degrees": math.degrees, "abs": abs, "min": min, "max": max, "log": math.log,
          "exp": math.exp}
_CONSTS = {"pi": math.pi, "g": 9.81}
_FACT = re.compile(r"[A-Za-z_]\w*(\[\d+\]|\.[A-Za-z_]\w*)+")


def evaluate(expr: str, reading, fact) -> float:
    """``expr``, arithmetic over instrument readings and task facts.  A reading is a call ``name(arg=value, ...)``
    of an instrument, ``reading(name, args)`` its value; a bare name is a task fact, ``fact(name)``.  Only
    arithmetic, the functions in ``_FUNCS`` and the constants in ``_CONSTS``: anything else is refused."""
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int | float) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name):
            return _CONSTS[n.id] if n.id in _CONSTS else float(fact(n.id))
        if isinstance(n, ast.Attribute | ast.Subscript) and _FACT.fullmatch(ast.unparse(n)):  # a nested fact
            return float(fact(ast.unparse(n)))                                                 # (route[2].h)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in _FUNCS and not n.keywords:
                return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
            if not n.args:
                return float(reading(n.func.id, {k.arg: _literal(k.value) for k in n.keywords}))
        raise ValueError(f"not allowed in a metric: {ast.unparse(n)!r}")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not an expression: {expr!r}") from exc
    return ev(tree)


def _literal(n: ast.AST) -> Any:
    v = ast.literal_eval(n)
    if not isinstance(v, int | float | str) or isinstance(v, bool):
        raise ValueError(f"an argument must be a number or a word: {ast.unparse(n)!r}")
    return v


def table(world: World, names: Sequence[str] | None = None) -> str:
    """The instruments, for the prompt."""
    rows = []
    for i in instruments(world):
        if names is not None and i.name not in names:
            continue
        args = ", ".join(f"{k}: {v}" for k, v in i.args.items())
        rows.append(f"- `{i.name}({args})`" + (" [per task]" if i.task else "") + f" {i.unit}: {i.doc}")
    return "\n".join(rows)
