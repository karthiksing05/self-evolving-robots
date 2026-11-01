"""CheckExpr parser and evaluator (Appendix A; test list 12.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sea.compose import checks
from sea.compose.checks import And, Atom, CheckSyntaxError, Not, Or

SRC = Path(__file__).resolve().parents[1] / "src" / "sea"


@dataclass
class Env:
    moved_by: dict[str, float] = field(default_factory=dict)
    travels: dict[str, float] = field(default_factory=dict)
    latch: dict[str, bool] = field(default_factory=dict)
    det: dict[str, int] = field(default_factory=dict)
    f: float = 0.0
    banned: bool = False

    def moved(self, ref):
        return self.moved_by.get(ref, 0.0)

    def travel(self, ref):
        return self.travels.get(ref, 0.01)

    def latched(self, ref):
        return self.latch.get(ref, False)

    def detents(self, ref):
        return self.det.get(ref, 0)

    def force(self):
        return self.f

    def banned_contact(self):
        return self.banned


@pytest.mark.parametrize("src, expected", [
    ("moved_at_least($target, travel)", Atom("moved_at_least", "$target", "travel")),
    ("moved_at_least(c3, 0.002)", Atom("moved_at_least", "c3", 0.002)),
    ("latched($target)", Atom("latched", "$target")),
    ("detent_crossed($target, 2)", Atom("detent_crossed", "$target", 2.0)),
    ("force_below(9.5)", Atom("force_below", None, 9.5)),
    ("no_banned_contact()", Atom("no_banned_contact")),
])
def test_every_atom_parses(src, expected):
    assert checks.parse(src) == expected
    assert checks.parse(checks.pretty(expected)) == expected


def test_precedence_not_and_or():
    e = checks.parse("~latched(a) & force_below(3) | no_banned_contact()")
    assert isinstance(e, Or)
    left, right = e.args
    assert isinstance(left, And) and isinstance(left.args[0], Not)
    assert right == Atom("no_banned_contact")
    grouped = checks.parse("~(latched(a) | force_below(3))")
    assert isinstance(grouped, Not) and isinstance(grouped.arg, Or)
    assert checks.parse("latched(a) & (force_below(1) | force_below(2))").args[1] == Or(
        (Atom("force_below", None, 1.0), Atom("force_below", None, 2.0)))


@pytest.mark.parametrize("src", [
    "~latched(a) & force_below(3) | no_banned_contact()",
    "(latched(a) | latched(b)) & ~(force_below(1) & detent_crossed($target, 1))",
    "~~latched($target)",
    "moved_at_least($target, travel) & detent_crossed($target, 1) & ~latched($target)",
])
def test_pretty_round_trip(src):
    e = checks.parse(src)
    assert checks.parse(checks.pretty(e)) == e
    assert checks.canonical(checks.canonical(src)) == checks.canonical(src)


@pytest.mark.parametrize("src", [
    "", "latched(", "latched()", "unknown($target)", "force_below(x)", "detent_crossed($target, 1.5)",
    "moved_at_least($target travel)", "latched(a) &", "latched(a) latched(b)", "force_below(1) @", "(latched(a)",
    "no_banned_contact(1)", "~", "moved_at_least(1, 2)",
])
def test_invalid_strings_raise(src):
    with pytest.raises(CheckSyntaxError):
        checks.parse(src)


def test_evaluate_semantics():
    env = Env(moved_by={"$target": 0.004, "c1": 0.5}, travels={"$target": 0.004}, latch={"c2": True},
              det={"$target": 1}, f=3.0)
    assert checks.check("moved_at_least($target, travel)", env)
    assert not checks.check("moved_at_least($target, 0.0041)", env)
    assert checks.check("moved_at_least(c1, 0.5)", env)
    assert checks.check("latched(c2) & ~latched($target)", env)
    assert checks.check("detent_crossed($target, 1) & ~detent_crossed($target, 2)", env)
    assert checks.check("force_below(3.5)", env) and not checks.check("force_below(3.0)", env)
    assert checks.check("no_banned_contact()", env)
    env.banned = True
    assert not checks.check("no_banned_contact()", env)
    assert checks.check("no_banned_contact() | force_below(4)", env)


def test_atoms_and_force_bound():
    e = checks.parse("force_below(9.5) & ~force_below(1.0) | (latched(a) & force_below(4))")
    assert [a.name for a in checks.atoms(e)] == ["force_below", "force_below", "latched", "force_below"]
    assert checks.max_force_bound(e) == 4.0
    assert checks.max_force_bound(checks.parse("latched(a)")) is None


def test_no_eval_or_exec_in_source():
    """Static guard: nothing in the package evaluates strings (spec 12.2)."""
    pat = re.compile(r"(?<![\w.])(eval|exec)\s*\(")
    hits = []
    for path in SRC.rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pat.search(code):
                hits.append(f"{path.relative_to(SRC)}:{i}: {line.strip()}")
    assert not hits, hits
