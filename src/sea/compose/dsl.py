"""Program DSL: constructors, pretty-printer, parser and hashing (spec 3.2, Appendix A).

The JSON form (``contracts.Node``) and the text form round-trip:
``parse(pretty(node)) == node``.  Positional primitive arguments are stored under the
keys ``"_0"``, ``"_1"``, ... and printed first, in order; named arguments follow, sorted.
"""

from __future__ import annotations

import hashlib

from sea.compose import checks
from sea.compose.checks import CheckSyntaxError, TokenStream, tokenize
from sea.contracts import Node, PrimRef, Program

TARGET = checks.TARGET


class ProgramSyntaxError(ValueError):
    pass


# ---------------------------------------------------------------- constructors
def prim(name: str, *pos: float | str, **named: float | str) -> Node:
    args: dict[str, float | str] = {f"_{i}": v for i, v in enumerate(pos)}
    args.update(named)
    return Node(op="prim", prim=PrimRef(name=name, args=args))


def seq(*children: Node) -> Node:
    return Node(op="seq", children=list(children))


def guard(child: Node, until: str, cap: str) -> Node:
    return Node(op="guard", children=[child], check=checks.canonical(until), cap=checks.canonical(cap))


def repeat(child: Node, n: int | None = None, until: str | None = None) -> Node:
    if (n is None) == (until is None):
        raise ProgramSyntaxError("Repeat needs exactly one of n or until")
    return Node(op="repeat", children=[child], n=n, check=checks.canonical(until) if until else None)


def branch(check: str, if_true: Node, if_false: Node) -> Node:
    return Node(op="branch", children=[if_true, if_false], check=checks.canonical(check))


def with_tool(tool_index: int, child: Node) -> Node:
    return Node(op="with", children=[child], tool_index=tool_index)


def bind(name: str, **fixed: float) -> Node:
    return Node(op="bind", prim=PrimRef(name=name), bind_args={k: float(v) for k, v in fixed.items()})


# ---------------------------------------------------------------- validation
ARITY = {"prim": (0, 0), "seq": (2, None), "guard": (1, 1), "repeat": (1, 1), "branch": (2, 2),
         "with": (1, 1), "bind": (0, 0)}


def validate(node: Node) -> list[str]:
    """Structural errors per the Appendix A arity table (empty list = well formed)."""
    errs: list[str] = []
    lo, hi = ARITY[node.op]
    n = len(node.children)
    if n < lo or (hi is not None and n > hi):
        errs.append(f"{node.op}: expected {lo}..{hi if hi is not None else 'n'} children, got {n}")
    if node.op in ("prim", "bind") and node.prim is None:
        errs.append(f"{node.op}: missing prim")
    if node.op == "bind" and not node.bind_args:
        errs.append("bind: needs at least one fixed argument")
    if node.op == "guard" and (node.check is None or node.cap is None):
        errs.append("guard: needs until and cap")
    if node.op == "repeat" and (node.n is None) == (node.check is None):
        errs.append("repeat: needs exactly one of n or until")
    if node.op == "repeat" and node.n is not None and node.n < 1:
        errs.append("repeat: n must be >= 1")
    if node.op == "branch" and node.check is None:
        errs.append("branch: needs a check")
    if node.op == "with" and (node.tool_index is None or node.tool_index < 0):
        errs.append("with: needs a non-negative tool_index")
    for expr in (node.check, node.cap):
        if expr is not None:
            try:
                checks.parse(expr)
            except CheckSyntaxError as exc:
                errs.append(f"{node.op}: bad check {expr!r}: {exc}")
    for c in node.children:
        errs.extend(validate(c))
    return errs


# ---------------------------------------------------------------- printing
def _fmt_value(v: float | str) -> str:
    if isinstance(v, str):
        return v if v == TARGET else '"' + v + '"'
    return repr(float(v))


def _fmt_args(args: dict[str, float | str]) -> str:
    pos = sorted((k for k in args if k.startswith("_") and k[1:].isdigit()), key=lambda k: int(k[1:]))
    named = sorted(k for k in args if k not in pos)
    parts = [_fmt_value(args[k]) for k in pos] + [f"{k}={_fmt_value(args[k])}" for k in named]
    return ", ".join(parts)


def pretty(node: Node) -> str:
    op = node.op
    if op == "prim":
        assert node.prim is not None
        return f"{node.prim.name}({_fmt_args(node.prim.args)})"
    if op == "seq":
        return "Seq(" + ", ".join(pretty(c) for c in node.children) + ")"
    if op == "guard":
        return f"Guard({pretty(node.children[0])}, until={_ck(node.check)}, cap={_ck(node.cap)})"
    if op == "repeat":
        tail = f"n={node.n}" if node.n is not None else f"until={_ck(node.check)}"
        return f"Repeat({pretty(node.children[0])}, {tail})"
    if op == "branch":
        return f"Branch({_ck(node.check)}, {pretty(node.children[0])}, {pretty(node.children[1])})"
    if op == "with":
        return f"With({node.tool_index}, {pretty(node.children[0])})"
    assert node.prim is not None
    fixed = ", ".join(f"{k}: {float(v)!r}" for k, v in sorted(node.bind_args.items()))
    return f"Bind({node.prim.name}, {{{fixed}}})"


def _ck(expr: str | None) -> str:
    return checks.canonical(expr or "")


def pretty_program(program: Program) -> str:
    return pretty(program.root)


# ---------------------------------------------------------------- parsing
_COMBINATORS = {"Seq", "Guard", "Repeat", "Branch", "With", "Bind"}


def parse(src: str) -> Node:
    try:
        ts = TokenStream(tokenize(src))
        node = _parse_node(ts)
        if ts.peek().kind != "end":
            raise ProgramSyntaxError(f"trailing input at {ts.peek().pos}: {ts.peek().text!r}")
    except CheckSyntaxError as exc:
        raise ProgramSyntaxError(str(exc)) from exc
    errs = validate(node)
    if errs:
        raise ProgramSyntaxError("; ".join(errs))
    return node


def _check_text(ts: TokenStream) -> str:
    return checks.pretty(checks.parse_stream(ts))


def _parse_node(ts: TokenStream) -> Node:
    tok = ts.expect_kind("name")
    name = tok.text
    ts.expect("(")
    if name == "Seq":
        kids = [_parse_node(ts)]
        while ts.accept(","):
            kids.append(_parse_node(ts))
        ts.expect(")")
        return Node(op="seq", children=kids)
    if name == "Guard":
        child = _parse_node(ts)
        ts.expect(",")
        _expect_kw(ts, "until")
        until = _check_text(ts)
        ts.expect(",")
        _expect_kw(ts, "cap")
        cap = _check_text(ts)
        ts.expect(")")
        return Node(op="guard", children=[child], check=until, cap=cap)
    if name == "Repeat":
        child = _parse_node(ts)
        ts.expect(",")
        kw = ts.expect_kind("name").text
        ts.expect("=")
        if kw == "n":
            n = int(ts.expect_kind("num").text)
            ts.expect(")")
            return Node(op="repeat", children=[child], n=n)
        if kw == "until":
            until = _check_text(ts)
            ts.expect(")")
            return Node(op="repeat", children=[child], check=until)
        raise ProgramSyntaxError(f"Repeat: unexpected keyword {kw!r}")
    if name == "Branch":
        cond = _check_text(ts)
        ts.expect(",")
        a = _parse_node(ts)
        ts.expect(",")
        b = _parse_node(ts)
        ts.expect(")")
        return Node(op="branch", children=[a, b], check=cond)
    if name == "With":
        idx = int(ts.expect_kind("num").text)
        ts.expect(",")
        child = _parse_node(ts)
        ts.expect(")")
        return Node(op="with", children=[child], tool_index=idx)
    if name == "Bind":
        pname = ts.expect_kind("name").text
        ts.expect(",")
        ts.expect("{")
        fixed: dict[str, float] = {}
        while True:
            k = ts.expect_kind("name").text
            ts.expect(":")
            fixed[k] = float(ts.expect_kind("num").text)
            if not ts.accept(","):
                break
        ts.expect("}")
        ts.expect(")")
        return Node(op="bind", prim=PrimRef(name=pname), bind_args=fixed)
    # primitive call
    args: dict[str, float | str] = {}
    npos = 0
    if not ts.accept(")"):
        while True:
            if ts.peek().kind == "name" and ts.peek(1).text == "=":
                key = ts.next().text
                ts.next()
                args[key] = _parse_value(ts)
            else:
                args[f"_{npos}"] = _parse_value(ts)
                npos += 1
            if ts.accept(")"):
                break
            ts.expect(",")
    return Node(op="prim", prim=PrimRef(name=name, args=args))


def _expect_kw(ts: TokenStream, kw: str) -> None:
    tok = ts.expect_kind("name")
    if tok.text != kw:
        raise ProgramSyntaxError(f"expected {kw}= at {tok.pos}")
    ts.expect("=")


def _parse_value(ts: TokenStream) -> float | str:
    tok = ts.next()
    if tok.kind == "num":
        return float(tok.text)
    if tok.kind == "target":
        return TARGET
    if tok.kind == "str":
        return tok.text[1:-1]
    raise ProgramSyntaxError(f"expected a value at {tok.pos}, got {tok.text!r}")


# ---------------------------------------------------------------- hashing and walking
def program_hash(node: Node) -> str:
    """Stable content hash (sha256 of the canonical text, 16 hex chars)."""
    return hashlib.sha256(pretty(node).encode()).hexdigest()[:16]


def walk(node: Node):
    yield node
    for c in node.children:
        yield from walk(c)


def primitive_names(node: Node) -> list[str]:
    return [n.prim.name for n in walk(node) if n.prim is not None]


def tool_indices(node: Node) -> list[int]:
    return [n.tool_index for n in walk(node) if n.op == "with" and n.tool_index is not None]
