"""The goal-test vocabulary above one control (docs/goals.md section 3.1, D-037).  Judge tier.

A goal on one control is tested by that control's check expression (``compose/checks.py``) during
the attempt.  A goal over many controls, or over a body change, is tested here, from what the judge
has already decided: verdicts, and the acceptance rule's decision on a paired trial.  Grammar::

    test := atom ("&" atom)*
    atom := pass_share(SET, NUMBER) | still_passes(SET) | accepted()

- ``pass_share(SET, x)``: at least a share ``x`` of SET's controls pass;
- ``still_passes(SET)``: every control of SET that passed before the change still passes;
- ``accepted()``: the paired trial of a body change passed the acceptance rule (``spec/posture.py``).

System 2 composes goal tests from this vocabulary and the check atoms.  It cannot write a new kind,
so a planner can never grade its own sub-goals (D-034).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sea.compose.checks import CheckSyntaxError, TokenStream, tokenize

SET_ATOMS = ("pass_share", "still_passes", "accepted")


@dataclass(frozen=True)
class SetAtom:
    name: str
    set: str = ""  # a named set of control keys: a split, the targets of a trial, a regression set
    share: float = 0.0  # pass_share only


def _atom(ts: TokenStream) -> SetAtom:
    tok = ts.expect_kind("name")
    if tok.text not in SET_ATOMS:
        raise CheckSyntaxError(f"unknown goal test {tok.text!r} at {tok.pos}")
    ts.expect("(")
    if tok.text == "accepted":
        ts.expect(")")
        return SetAtom("accepted")
    name = ts.expect_kind("name").text
    share = 0.0
    if tok.text == "pass_share":
        ts.expect(",")
        num = ts.expect_kind("num")
        share = float(num.text)
        if not 0.0 <= share <= 1.0:
            raise CheckSyntaxError(f"pass_share needs a share in [0, 1] at {num.pos}, got {num.text}")
    ts.expect(")")
    return SetAtom(tok.text, name, share)


def parse(src: str) -> tuple[SetAtom, ...]:
    ts = TokenStream(tokenize(src))
    out = [_atom(ts)]
    while ts.accept("&"):
        out.append(_atom(ts))
    if ts.peek().kind != "end":
        raise CheckSyntaxError(f"unexpected {ts.peek().text!r} at {ts.peek().pos}")
    return tuple(out)


def pretty(atoms: Sequence[SetAtom]) -> str:
    def one(a: SetAtom) -> str:
        if a.name == "accepted":
            return "accepted()"
        if a.name == "pass_share":
            return f"pass_share({a.set}, {a.share:g})"
        return f"{a.name}({a.set})"

    return " & ".join(one(a) for a in atoms)


def sets(atoms: Sequence[SetAtom]) -> set[str]:
    """The named sets a test refers to."""
    return {a.set for a in atoms if a.set}


@dataclass
class Evidence:
    """What the judge has decided, which is all a goal test may read."""

    members: Mapping[str, Sequence[str]]  # set name -> control keys
    verdicts: Mapping[str, str]  # control key -> latest verdict
    before: Mapping[str, str] = field(default_factory=dict)  # control key -> verdict before the change
    accepted: bool | None = None  # the acceptance rule's decision, when a trial ran


def holds(test: str | Sequence[SetAtom], ev: Evidence) -> bool:
    atoms = parse(test) if isinstance(test, str) else test
    for a in atoms:
        if a.name == "accepted":
            ok = bool(ev.accepted)
        else:
            if a.set not in ev.members:
                raise KeyError(f"goal test names an unknown set {a.set!r}")
            keys = ev.members[a.set]
            if a.name == "pass_share":
                share = sum(ev.verdicts.get(k) == "pass" for k in keys) / len(keys) if keys else 0.0
                ok = share >= a.share
            else:
                ok = all(ev.verdicts.get(k) == "pass" for k in keys if ev.before.get(k) == "pass")
        if not ok:
            return False
    return True
