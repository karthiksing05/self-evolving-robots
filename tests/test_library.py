"""Primitive library: seed, promotion, duplicates, pruning (spec 3.5, 8.7; test list 12.2)."""

from __future__ import annotations

import pytest

from sea.compose import dsl
from sea.compose import library as L
from sea.compose.library import Library
from sea.contracts import D_DUP, RETIRE_AFTER, ControlSpec, Pose, PrimitiveDef, Program, Vec3

PRESS = ("Seq(approach($target), contact($target), Guard(push($target, force={f}), "
         "until=moved_at_least($target, travel), cap=force_below(9.5)), retreat())")


def _prog(text: str, pid: str = "s2_x", cls: str = "push_button") -> Program:
    return Program(id=pid, target_cls=cls, root=dsl.parse(text), source="s2")


def _ctl(cls: str = "door", handle: str = "none") -> ControlSpec:
    return ControlSpec(id="c", cls=cls, pose_panel=Pose(pos=Vec3(x=0, y=0, z=0)), size_mm=(10.0,), stiffness=1.0,
                       travel=0.01, friction=0.01, handle=handle, success="moved_at_least($target, travel)")


def test_seed_library_contents():
    lib = Library.seed()
    kinds = {k: [p.name for p in lib.prims.values() if p.kind == k] for k in ("motor", "check", "composite")}
    assert len(kinds["motor"]) == 10 and len(kinds["check"]) == 6  # grasp and held_for: the grasp world
    assert sorted(kinds["composite"]) == sorted(["press_button", "flip_toggle", "turn_knob", "pull_lever",
                                                 "open_door_pull", "slide_slider", "pick_up"])
    assert lib.get("grasp").requires == {"contact": "pinch", "aperture": "width"}
    assert lib.get("press_button").requires["contact"] == "edge|pad|probe"
    assert lib.get("turn_knob").requires == {"contact": "cup|pinch", "min_force": 0.5}
    assert "push" in lib and "fly" not in lib and lib.get("fly") is None
    assert lib.to_list()[0]["name"] == "approach"


def test_provides_parsing_and_applicability():
    assert L.parse_provides("door[tab/pull]|drawer|") == [("door", {"tab", "pull"}), ("drawer", None)]
    with pytest.raises(ValueError):
        L.parse_provides("Door[")
    lib = Library.seed()
    assert [p.name for p in lib.programs_for(_ctl("door", "tab"))] == ["open_door_pull"]
    assert lib.programs_for(_ctl("door", "push_latch")) == []
    assert L.target_classes(lib.get("open_door_pull")) == ["door", "drawer"]
    assert L.requires_contacts(lib.get("approach")) is None


def test_promotion_requires_k_uses_across_two_panels():
    lib = Library.seed()
    name = "s2_press"
    lib.add(PrimitiveDef(name=name, kind="composite", doc="d", provides={"actuates": "push_button"},
                         body=_prog(PRESS.format(f=3.0), name)))
    for c in ("c0", "c1", "c2"):
        lib.record_use(name, passed=True, iteration=1, panel_id="p0", control_id=c)
    assert not lib.promotion_ready(name, k_promote=3)  # 3 controls but a single panel
    lib.record_use(name, passed=False, iteration=1, panel_id="p1", control_id="c0")
    assert not lib.promotion_ready(name, k_promote=3)  # failures do not count
    lib.record_use(name, passed=True, iteration=2, panel_id="p1", control_id="c0")
    assert lib.promotion_ready(name, k_promote=3)
    assert not lib.promotion_ready(name, k_promote=5)
    lib.record_use(name, passed=True, iteration=2, panel_id="p1", control_id="c0")  # same control again
    assert not lib.promotion_ready(name, k_promote=5)
    d = lib.get(name)
    assert d.uses == 6 and d.passes == 5 and lib.last_used[name] == 2
    lib.record_use("unknown", passed=True, iteration=3)  # ignored


def test_promote_abstracts_differing_arguments():
    lib = Library.seed()
    variants = [dsl.parse(PRESS.format(f=f)) for f in (3.0, 4.5, 6.0)]
    d = lib.promote(_prog(PRESS.format(f=3.0)), "firm_press", "Press with a tuned force.", variants, iteration=4)
    assert d is not None and d.body.source == "promoted" and d.created_iter == 4
    assert d.params == {"p0": (3.0, 6.0)}
    assert '"$p0"' in dsl.pretty(d.body.root)
    assert d.requires["contact"] == "edge|pad|probe" and d.provides == {"actuates": "push_button"}
    assert lib.promote(_prog(PRESS.format(f=3.0)), "firm_press", "again", variants, iteration=5) is None
    # structurally different variants keep the first one verbatim
    body, params = L.abstract([dsl.parse(PRESS.format(f=1.0)), dsl.parse("Seq(approach($target), retreat())")])
    assert params == {} and dsl.pretty(body) == PRESS.format(f=1.0)


def test_duplicate_detection_blocks_near_identical_programs():
    from sea.memory import ConceptMemory, FakeEmbedder

    mem = ConceptMemory(seed=0, embedder=FakeEmbedder())
    existing = _prog(PRESS.format(f=3.0), "press_firm")
    mem.add_program(existing, {"push_button": 1.0})

    def is_dup(p: Program) -> bool:
        return any(dist < D_DUP for _, dist in mem.similar_programs(p, {"push_button": 1.0}, k=1))

    lib = Library.seed()
    near = _prog(PRESS.format(f=3.0), "s2_near")
    assert lib.promote(near, "press_again", "dup", [near.root], iteration=1, is_duplicate=is_dup) is None
    far = _prog("Seq(approach($target), contact($target), Repeat(Seq(rotate($target), release(), contact($target)), "
                "until=moved_at_least($target, travel)), retreat())", "s2_far", "knob")
    assert lib.promote(far, "twist", "turns", [far.root], iteration=1, is_duplicate=is_dup) is not None


def _promoted(lib: Library, name: str, it: int, n_prims: int = 2, uses: int = 0, passes: int = 0,
              provides: str = "push_button") -> None:
    text = "Seq(" + ", ".join(["approach($target)"] * (n_prims - 1) + ["push($target)"]) + ")"
    d = lib.promote(_prog(text, name), name, "d", [dsl.parse(text)], iteration=it)
    assert d is not None
    lib.prims[name] = d.model_copy(update={"uses": uses, "passes": passes, "provides": {"actuates": provides}})


def test_prune_retires_idle_and_dominated():
    lib = Library.seed()
    _promoted(lib, "idle", it=0, provides="toggle")
    _promoted(lib, "fresh", it=8, provides="slider")
    _promoted(lib, "cheap_good", it=8, n_prims=2, uses=4, passes=4)
    _promoted(lib, "costly_bad", it=8, n_prims=4, uses=4, passes=1)
    gone = lib.prune(iteration=RETIRE_AFTER, protected=set())
    assert sorted(gone) == ["costly_bad", "idle"]
    assert {p.name for p in lib.retired} == {"costly_bad", "idle"}
    assert "fresh" in lib and "cheap_good" in lib and "press_button" in lib  # seeds are never pruned
    lib.record_use("fresh", passed=True, iteration=15)
    assert lib.prune(iteration=17) == []  # cheap_good idle 9 < RETIRE_AFTER
    assert lib.prune(iteration=18) == ["cheap_good"]
    assert lib.prune(iteration=30, protected={"fresh"}) == []


def test_prune_respects_the_size_budget():
    lib = Library.seed()
    n_seed = len(lib.prims)
    lib.size_budget = n_seed + 2
    for i, (u, p) in enumerate([(4, 4), (4, 1), (4, 3), (4, 2)]):
        _promoted(lib, f"p{i}", it=5, uses=u, passes=p, provides=["toggle", "slider", "lever", "knob"][i])
    gone = lib.prune(iteration=6)
    assert sorted(gone) == ["p1", "p3"]  # lowest pass rates go first
    assert len(lib.prims) == lib.size_budget
    lib.size_budget = 1  # only promoted primitives can be dropped
    lib.prune(iteration=6)
    assert len(lib.prims) == n_seed


def test_program_cost_and_pass_rate():
    lib = Library.seed()
    assert L.pass_rate(lib.get("press_button")) == 0.0
    assert L.program_cost(lib.get("push")) == 1.0
    with_tool = PrimitiveDef(name="w", kind="composite", doc="d", body=_prog("With(0, Seq(push($target), retreat()))"))
    assert L.program_cost(with_tool) == 2 + 5
