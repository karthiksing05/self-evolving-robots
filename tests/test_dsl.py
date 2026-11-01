"""Program DSL (Appendix A; test list 12.2): pretty/parse round-trip on random programs, hashing."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sea.compose import dsl
from sea.compose.dsl import ProgramSyntaxError
from sea.contracts import Node, PrimRef

PRIMS = ["approach", "contact", "push", "rotate", "hook_pull", "slide", "insert", "release", "retreat", "my_prim"]
ARG_NAMES = ["force", "speed", "depth", "distance", "angle", "offset_u", "x1"]
REFS = st.sampled_from(["$target", "c0", "c12", "knob_a"])
NUM = st.floats(allow_nan=False, allow_infinity=False, width=64)
SMALL_INT = st.integers(min_value=0, max_value=8)


def _atom() -> st.SearchStrategy[str]:
    return st.one_of(
        st.builds(lambda r, v: f"moved_at_least({r}, {v})", REFS, st.one_of(st.just("travel"), NUM.map(repr))),
        st.builds(lambda r: f"latched({r})", REFS),
        st.builds(lambda r, n: f"detent_crossed({r}, {n})", REFS, SMALL_INT),
        st.builds(lambda v: f"force_below({v!r})", NUM),
        st.just("no_banned_contact()"),
    )


CHECK = st.recursive(_atom(), lambda inner: st.one_of(
    inner.map(lambda c: f"~{c}" if c[0] != "~" else f"~({c})"),
    st.builds(lambda a, b: f"({a}) & ({b})", inner, inner),
    st.builds(lambda a, b: f"({a}) | ({b})", inner, inner),
), max_leaves=5)

VALUE = st.one_of(NUM, st.just("$target"), st.text(alphabet="abcxyz_", min_size=1, max_size=5).map(lambda s: "s" + s))


def _prim_node() -> st.SearchStrategy[Node]:
    pos = st.lists(VALUE, max_size=3)
    named = st.dictionaries(st.sampled_from(ARG_NAMES), VALUE, max_size=3)
    return st.builds(lambda n, p, kw: dsl.prim(n, *p, **kw), st.sampled_from(PRIMS), pos, named)


def _bind_node() -> st.SearchStrategy[Node]:
    return st.builds(lambda n, kw: dsl.bind(n, **kw), st.sampled_from(PRIMS),
                     st.dictionaries(st.sampled_from(ARG_NAMES), NUM, min_size=1, max_size=3))


LEAF = st.one_of(_prim_node(), _bind_node())


def _extend(inner: st.SearchStrategy[Node]) -> st.SearchStrategy[Node]:
    return st.one_of(
        st.lists(inner, min_size=2, max_size=4).map(lambda kids: dsl.seq(*kids)),
        st.builds(dsl.guard, inner, CHECK, CHECK),
        st.builds(lambda c, n: dsl.repeat(c, n=n), inner, st.integers(min_value=1, max_value=9)),
        st.builds(lambda c, u: dsl.repeat(c, until=u), inner, CHECK),
        st.builds(dsl.branch, CHECK, inner, inner),
        st.builds(dsl.with_tool, SMALL_INT, inner),
    )


PROGRAMS = st.recursive(LEAF, _extend, max_leaves=8)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(PROGRAMS)
def test_round_trip_random_programs(node):
    text = dsl.pretty(node)
    again = dsl.parse(text)
    assert again == node
    assert dsl.pretty(again) == text
    assert dsl.program_hash(again) == dsl.program_hash(node)
    assert Node.model_validate_json(node.model_dump_json()) == node
    assert dsl.validate(node) == []


def test_hash_is_stable_across_runs():
    text = ("Seq(approach($target), contact($target), Guard(push($target), until=moved_at_least($target, travel), "
            "cap=force_below(9.5)), retreat())")
    node = dsl.parse(text)
    assert dsl.pretty(node) == text
    assert dsl.program_hash(node) == "03edce8f6b5061e2"
    assert dsl.program_hash(dsl.parse(text.replace("9.5", "9.6"))) != "03edce8f6b5061e2"


def test_whitespace_and_canonical_checks_do_not_change_the_hash():
    a = dsl.parse("Guard(push($target),until=force_below(1)|latched(c1),cap=(force_below(9.5)))")
    b = dsl.parse("Guard( push( $target ), until = force_below(1.0) | latched(c1), cap = force_below(9.5) )")
    assert a == b and dsl.program_hash(a) == dsl.program_hash(b)


def test_walk_helpers():
    node = dsl.parse("Seq(With(1, push($target)), Bind(rotate, {angle: 1.0}), With(0, retreat()))")
    assert dsl.primitive_names(node) == ["push", "rotate", "retreat"]
    assert dsl.tool_indices(node) == [1, 0]
    assert len(list(dsl.walk(node))) == 6


@pytest.mark.parametrize("src", [
    "Seq(push($target))",  # seq needs >= 2 children
    "Repeat(push($target), n=0)",
    "Repeat(push($target), k=2)",
    "Guard(push($target), until=latched(a))",
    "Guard(push($target), until=latched(a), lim=force_below(1))",
    "Bind(push, {})",
    "push($target",
    "push($target) retreat()",
    "push(@)",
    "With(-1, push($target))",
    "Branch(latched(a), push($target))",
])
def test_invalid_programs_raise(src):
    with pytest.raises(ProgramSyntaxError):
        dsl.parse(src)


def test_constructor_and_validate_errors():
    with pytest.raises(ProgramSyntaxError):
        dsl.repeat(dsl.prim("push"))
    bad = [
        Node(op="prim"),
        Node(op="bind", prim=PrimRef(name="push")),
        Node(op="guard", children=[dsl.prim("push")]),
        Node(op="repeat", children=[dsl.prim("push")], n=0),
        Node(op="branch", children=[dsl.prim("push"), dsl.prim("push")]),
        Node(op="with", children=[dsl.prim("push")]),
        Node(op="guard", children=[dsl.prim("push")], check="latched(", cap="force_below(1)"),
    ]
    for node in bad:
        assert dsl.validate(node), node
