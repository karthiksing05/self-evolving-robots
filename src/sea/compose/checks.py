"""CheckExpr parser and evaluator (spec Section 6 and Appendix A).

Hand-written recursive descent; nothing here calls ``eval``.  Grammar::

    check  := term ("|" term)*
    term   := factor ("&" factor)*
    factor := "~" factor | "(" check ")" | atom
    atom   := moved_at_least(ref, value) | latched(ref) | detent_crossed(ref, INT)
            | force_below(NUMBER) | no_banned_contact()
    ref    := "$target" | CONTROL_ID
    value  := NUMBER | "travel"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

TARGET = "$target"


class CheckSyntaxError(ValueError):
    pass


# ---------------------------------------------------------------- tokens
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)
  | (?P<target>\$target)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<str>"[^"\\]*")
  | (?P<op>[()\[\]{},=:&|~])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str  # num | target | name | str | op | end
    text: str
    pos: int


def tokenize(src: str) -> list[Token]:
    out: list[Token] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if m is None:
            raise CheckSyntaxError(f"unexpected character {src[i]!r} at {i}")
        kind = m.lastgroup or ""
        if kind != "ws":
            out.append(Token(kind, m.group(), i))
        i = m.end()
    out.append(Token("end", "", len(src)))
    return out


class TokenStream:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self, k: int = 0) -> Token:
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def next(self) -> Token:
        tok = self.peek()
        self.i += 1
        return tok

    def accept(self, text: str) -> bool:
        if self.peek().kind == "op" and self.peek().text == text:
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> Token:
        tok = self.next()
        if tok.text != text:
            raise CheckSyntaxError(f"expected {text!r} at {tok.pos}, got {tok.text!r}")
        return tok

    def expect_kind(self, kind: str) -> Token:
        tok = self.next()
        if tok.kind != kind:
            raise CheckSyntaxError(f"expected {kind} at {tok.pos}, got {tok.text!r}")
        return tok


# ---------------------------------------------------------------- AST
@dataclass(frozen=True)
class Atom:
    name: str
    ref: str | None = None  # "$target" or a control id
    value: float | str | None = None  # number, int, or "travel"


@dataclass(frozen=True)
class Not:
    arg: Expr


@dataclass(frozen=True)
class And:
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Or:
    args: tuple[Expr, ...]


Expr = Atom | Not | And | Or

ATOMS = ("moved_at_least", "latched", "detent_crossed", "force_below", "no_banned_contact")


def _parse_ref(ts: TokenStream) -> str:
    tok = ts.next()
    if tok.kind in ("target", "name"):
        return tok.text
    raise CheckSyntaxError(f"expected a control reference at {tok.pos}, got {tok.text!r}")


def _parse_number(ts: TokenStream) -> float:
    return float(ts.expect_kind("num").text)


def _parse_int(ts: TokenStream) -> int:
    tok = ts.expect_kind("num")
    try:
        return int(tok.text)
    except ValueError as exc:
        raise CheckSyntaxError(f"expected an integer at {tok.pos}, got {tok.text!r}") from exc


def _parse_atom(ts: TokenStream) -> Atom:
    tok = ts.expect_kind("name")
    name = tok.text
    if name not in ATOMS:
        raise CheckSyntaxError(f"unknown check {name!r} at {tok.pos}")
    ts.expect("(")
    if name == "moved_at_least":
        ref = _parse_ref(ts)
        ts.expect(",")
        nxt = ts.peek()
        if nxt.kind == "name" and nxt.text == "travel":
            ts.next()
            value: float | str = "travel"
        else:
            value = _parse_number(ts)
        atom = Atom(name, ref, value)
    elif name == "latched":
        atom = Atom(name, _parse_ref(ts))
    elif name == "detent_crossed":
        ref = _parse_ref(ts)
        ts.expect(",")
        atom = Atom(name, ref, float(_parse_int(ts)))
    elif name == "force_below":
        atom = Atom(name, None, _parse_number(ts))
    else:  # no_banned_contact
        atom = Atom(name)
    ts.expect(")")
    return atom


def _parse_factor(ts: TokenStream) -> Expr:
    if ts.accept("~"):
        return Not(_parse_factor(ts))
    if ts.accept("("):
        inner = parse_stream(ts)
        ts.expect(")")
        return inner
    return _parse_atom(ts)


def _parse_term(ts: TokenStream) -> Expr:
    args = [_parse_factor(ts)]
    while ts.accept("&"):
        args.append(_parse_factor(ts))
    return args[0] if len(args) == 1 else And(tuple(args))


def parse_stream(ts: TokenStream) -> Expr:
    """Parse one check from a token stream, leaving the stream after it."""
    args = [_parse_term(ts)]
    while ts.accept("|"):
        args.append(_parse_term(ts))
    return args[0] if len(args) == 1 else Or(tuple(args))


def parse(src: str) -> Expr:
    ts = TokenStream(tokenize(src))
    expr = parse_stream(ts)
    if ts.peek().kind != "end":
        raise CheckSyntaxError(f"trailing input at {ts.peek().pos}: {ts.peek().text!r}")
    return expr


# ---------------------------------------------------------------- printing
def pretty(expr: Expr) -> str:
    """Canonical text form; ``parse(pretty(e)) == e``."""
    return _pretty(expr, 0)


def _pretty(expr: Expr, parent_prec: int) -> str:
    # precedence: Or=1, And=2, Not/Atom=3
    if isinstance(expr, Atom):
        if expr.name == "moved_at_least":
            v = expr.value if expr.value == "travel" else repr(float(expr.value))  # type: ignore[arg-type]
            return f"moved_at_least({expr.ref}, {v})"
        if expr.name == "latched":
            return f"latched({expr.ref})"
        if expr.name == "detent_crossed":
            return f"detent_crossed({expr.ref}, {int(expr.value)})"  # type: ignore[arg-type]
        if expr.name == "force_below":
            return f"force_below({float(expr.value)!r})"  # type: ignore[arg-type]
        return "no_banned_contact()"
    if isinstance(expr, Not):
        return "~" + _pretty(expr.arg, 3)
    if isinstance(expr, And):
        s = " & ".join(_pretty(a, 2) for a in expr.args)
        return f"({s})" if parent_prec > 2 else s
    s = " | ".join(_pretty(a, 1) for a in expr.args)
    return f"({s})" if parent_prec > 1 else s


def canonical(src: str) -> str:
    return pretty(parse(src))


# ---------------------------------------------------------------- evaluation
class CheckEnv(Protocol):
    """Ground-truth sim state as seen by the verifier."""

    def moved(self, ref: str) -> float: ...
    def travel(self, ref: str) -> float: ...
    def latched(self, ref: str) -> bool: ...
    def detents(self, ref: str) -> int: ...
    def force(self) -> float: ...
    def banned_contact(self) -> bool: ...


def evaluate(expr: Expr, env: CheckEnv) -> bool:
    if isinstance(expr, Not):
        return not evaluate(expr.arg, env)
    if isinstance(expr, And):
        return all(evaluate(a, env) for a in expr.args)
    if isinstance(expr, Or):
        return any(evaluate(a, env) for a in expr.args)
    name = expr.name
    if name == "moved_at_least":
        assert expr.ref is not None
        need = env.travel(expr.ref) if expr.value == "travel" else float(expr.value)  # type: ignore[arg-type]
        return env.moved(expr.ref) >= need - 1e-9
    if name == "latched":
        assert expr.ref is not None
        return env.latched(expr.ref)
    if name == "detent_crossed":
        assert expr.ref is not None
        return env.detents(expr.ref) >= int(expr.value)  # type: ignore[arg-type]
    if name == "force_below":
        return env.force() < float(expr.value)  # type: ignore[arg-type]
    return not env.banned_contact()


def check(src: str, env: CheckEnv) -> bool:
    return evaluate(parse(src), env)


def atoms(expr: Expr) -> list[Atom]:
    if isinstance(expr, Atom):
        return [expr]
    if isinstance(expr, Not):
        return atoms(expr.arg)
    out: list[Atom] = []
    for a in expr.args:
        out.extend(atoms(a))
    return out


def max_force_bound(expr: Expr) -> float | None:
    """Smallest ``force_below`` threshold that appears un-negated, if any (used by typecheck)."""
    best: float | None = None

    def walk(e: Expr, negated: bool) -> None:
        nonlocal best
        if isinstance(e, Atom):
            if e.name == "force_below" and not negated:
                v = float(e.value)  # type: ignore[arg-type]
                best = v if best is None else min(best, v)
        elif isinstance(e, Not):
            walk(e.arg, not negated)
        else:
            for a in e.args:
                walk(a, negated)

    walk(expr, False)
    return best
