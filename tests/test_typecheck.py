"""Static program checks (spec 8.7, D-008; test list 12.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sea.compose import dsl
from sea.compose import typecheck as T
from sea.compose.library import Library
from sea.contracts import Design, HardLimits, PrimitiveDef, Program, ToolSpec
from sea.control.reach import ReachGrid, design_key
from sea.sim.panels import panel_pose, reference_panel

ROOT = Path(__file__).resolve().parents[1]
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
LIB = Library.seed()
PANEL = reference_panel()
CTL = {c.cls if c.cls not in ("door", "drawer") else c.cls: c for c in PANEL.controls}
REACH = ReachGrid.load(ROOT / "tests" / "fixtures" / "reach_stock.npz")

HOOK = ToolSpec(template="hook", params={"depth": 14, "throat": 6, "length": 45}, contact="hook")
CUP = ToolSpec(template="knob_cup", params={"inner_diameter": 14, "depth": 10, "ribs": 4}, contact="cup")
PAD = ToolSpec(template="pad", params={"w": 16, "h": 16, "thickness": 5, "friction": 0.9}, contact="pad")
BLADE = ToolSpec(template="slot_inserter", params={"blade_thickness": 1.4, "width": 24, "length": 40},
                 contact="edge")
STOCK = Design(id="stock")
RACK = Design(id="rack", tool_rack=[HOOK, CUP, PAD, BLADE])


def _prog(text: str, cls: str = "push_button") -> Program:
    return Program(id="t", target_cls=cls, root=dsl.parse(text), source="s2")


def _codes(errs) -> set[str]:
    return {e.code for e in errs}


def _seed(name: str) -> Program:
    p = LIB.get(name)
    assert p is not None and p.body is not None
    return p.body


# ---------------------------------------------------------------- required cases (12.2)
def test_valid_seed_program_passes():
    c = CTL["push_button"]
    assert T.typecheck(_seed("press_button"), STOCK, LIB.prims, LIMITS, REACH, control=c, panel=PANEL) == []


def test_hook_primitive_without_hook_tool_is_rejected():
    door = CTL["door"]
    assert door.handle == "pull"
    errs = T.typecheck(_seed("open_door_pull"), STOCK, LIB.prims, LIMITS, REACH, control=door, panel=PANEL)
    assert _codes(errs) == {"contact"} and "hook" in errs[0].message
    lever = CTL["lever"]  # levers accept pad and hook, but hook_pull needs hook or pinch: a pad cannot pull
    with_pad = _prog("Seq(approach($target), With(2, hook_pull($target)), retreat())", "lever")
    assert _codes(T.typecheck(with_pad, RACK, LIB.prims, LIMITS, control=lever)) == {"contact"}
    errs = T.typecheck(_prog("With(2, hook_pull($target))", "door"), RACK, LIB.prims, LIMITS, control=door)
    assert _codes(errs) == {"contact"}
    assert T.typecheck(_prog("With(0, hook_pull($target))", "door"), RACK, LIB.prims, LIMITS, control=door) == []


def test_out_of_reach_target_is_rejected():
    far = PANEL.model_copy(update={"pose_world": panel_pose(0.32, 0.0, 0.0, 0.0)})
    c = next(c for c in far.controls if c.cls == "recessed_button")  # ~5 cm further out than the panel centre
    assert T.control_target(far, c)[0][0] > 0.36
    errs = T.typecheck(_seed("press_button"), STOCK, LIB.prims, LIMITS, REACH, control=c, panel=far)
    assert _codes(errs) == {"reach"}
    # without a reach model the same program type-checks
    assert T.typecheck(_seed("press_button"), STOCK, LIB.prims, LIMITS, control=c, panel=far) == []


def test_parameter_beyond_hard_limit_is_rejected():
    c = CTL["push_button"]
    over = _prog("Guard(push($target, force=20.0), until=moved_at_least($target, travel), cap=force_below(9.5))")
    assert "hard_limit" in _codes(T.typecheck(over, STOCK, LIB.prims, LIMITS, control=c))
    tight = LIMITS.model_copy(update={"max_force_n": 8.0, "max_tip_speed_mps": 0.03})
    inside = _prog("push($target, force=10.0, speed=0.04)")
    errs = T.typecheck(inside, STOCK, LIB.prims, tight, control=c)
    assert _codes(errs) == {"hard_limit"} and len(errs) == 2
    cap = _prog("Guard(push($target), until=latched($target), cap=force_below(16.0))")
    assert _codes(T.typecheck(cap, STOCK, LIB.prims, LIMITS, control=c)) == {"hard_limit"}
    # a parameter *range* (System 2 reparam, or a promoted primitive's argument range)
    assert _codes(T.reparam_errors({"push.force": (1.0, 20.0)}, LIB.prims, LIMITS)) == {"hard_limit", "range"}
    assert T.reparam_errors({"force": (1.0, 5.0)}, LIB.prims, LIMITS, primitive="push") == []
    body = _prog('Seq(approach($target), push($target, force="$p0"))')
    risky = PrimitiveDef(name="press_hard", kind="composite", doc="d", provides={"actuates": "push_button"},
                         params={"p0": (1.0, 20.0)}, body=body)
    lib = {**LIB.prims, "press_hard": risky}
    errs = T.typecheck(_prog("press_hard($target)"), STOCK, lib, LIMITS, control=c)
    assert "hard_limit" in _codes(errs)
    ok = T.typecheck(_prog("press_hard($target, 4.0)"), STOCK, lib, LIMITS, control=c)
    assert ok == []


# ---------------------------------------------------------------- other rules
def test_structure_unknowns_and_args():
    c = CTL["push_button"]
    bad = Program(id="b", target_cls="push_button", root=dsl.seq(dsl.prim("push", "$target"), dsl.prim("x"))
                  .model_copy(update={"children": [dsl.prim("push", "$target")]}), source="s2")
    assert _codes(T.typecheck(bad, STOCK, LIB.prims, LIMITS, control=c)) == {"structure"}
    errs = T.typecheck(_prog("Seq(fly($target), push($target, wobble=1.0, force=3.0))"), STOCK, LIB.prims, LIMITS,
                       control=c)
    assert _codes(errs) == {"unknown", "arg"}
    errs = T.typecheck(_prog("Seq(latched($target), push($target, 1.0, 2.0, 0.01, 0.0, 0.0, 9.0))"), STOCK,
                       LIB.prims, LIMITS, control=c)
    assert {"unknown", "arg"} <= _codes(errs)
    errs = T.typecheck(_prog('push($target, depth=0.5, force="$nope")'), STOCK, LIB.prims, LIMITS, control=c)
    assert _codes(errs) == {"range", "arg"}
    s = str(errs[0])
    assert s.startswith("root:") and "[" in s


def test_tool_slots_and_slot_inserter():
    slot = CTL["slot"]
    assert "tool" in _codes(T.typecheck(_prog("With(4, insert($target))", "slot"), RACK, LIB.prims, LIMITS,
                                        control=slot))
    errs = T.typecheck(_prog("insert($target)", "slot"), RACK, LIB.prims, LIMITS, control=slot)
    assert _codes(errs) == {"contact"}  # the stock edge is accepted, but slots need the blade
    assert T.typecheck(_prog("With(3, insert($target))", "slot"), RACK, LIB.prims, LIMITS, control=slot) == []
    nested = _prog("With(0, Seq(approach($target), With(3, insert($target))))", "slot")
    assert T.typecheck(nested, RACK, LIB.prims, LIMITS, control=slot) == []


def test_contact_typing_follows_appendix_b():
    knob, toggle, lever = CTL["knob"], CTL["toggle"], CTL["lever"]
    # stock pinch is accepted by knobs (banned contacts are the verifier's business, D-008)
    assert T.typecheck(_seed("turn_knob"), STOCK, LIB.prims, LIMITS, control=knob) == []
    assert T.typecheck(_prog("With(1, rotate($target))", "knob"), RACK, LIB.prims, LIMITS, control=knob) == []
    assert _codes(T.typecheck(_prog("With(2, rotate($target))", "knob"), RACK, LIB.prims, LIMITS,
                              control=knob)) == {"contact"}
    assert T.typecheck(_seed("flip_toggle"), STOCK, LIB.prims, LIMITS, control=toggle) == []
    assert _codes(T.typecheck(_seed("pull_lever"), STOCK, LIB.prims, LIMITS, control=lever)) == {"contact"}
    # without a control the program's target class is used (any handle for doors)
    assert T.typecheck(_seed("open_door_pull"), STOCK, LIB.prims, LIMITS) == []
    assert _codes(T.typecheck(_prog("push($target)", "knob"), STOCK, LIB.prims, LIMITS)) == {"contact"}
    assert T.accepted_contacts(cls="door") == frozenset({"pinch", "hook", "pad", "probe"})
    assert T.tool_contacts(None) == T.STOCK_CONTACTS == T.tool_contacts(ToolSpec(template="stock", params={},
                                                                                 contact="edge"))


def test_provides_and_references():
    c = CTL["push_button"]
    errs = T.typecheck(_prog("Seq(approach($target), retreat())"), STOCK, LIB.prims, LIMITS, control=c)
    assert _codes(errs) == {"provides"}
    errs = T.typecheck(_prog("slide($target)"), STOCK, LIB.prims, LIMITS, control=c)
    assert _codes(errs) == {"provides"}
    other = PANEL.controls[1]
    assert other.cls == "toggle"
    cross = _prog(f"Seq(push(\"{other.id}\"), push($target))")  # Appendix A: other controls by STRING
    assert T.typecheck(cross, STOCK, LIB.prims, LIMITS, control=c, panel=PANEL) == []
    ghost = _prog("Seq(push(\"ghost\"), Guard(push($target), until=latched(ghost2), cap=force_below(5)))")
    errs = T.typecheck(ghost, STOCK, LIB.prims, LIMITS, control=c, panel=PANEL)
    assert _codes(errs) == {"ref"} and len(errs) == 2


def test_composites_recurse_and_nesting_limit():
    c = CTL["push_button"]
    assert T.typecheck(_prog("Seq(press_button($target), retreat())"), STOCK, LIB.prims, LIMITS, control=c) == []
    loop = PrimitiveDef(name="loop", kind="composite", doc="d", provides={"actuates": "push_button"},
                        body=_prog("Seq(loop($target), push($target))"))
    errs = T.typecheck(_prog("loop($target)"), STOCK, {**LIB.prims, "loop": loop}, LIMITS, control=c)
    assert "structure" in _codes(errs)


def test_reparam_errors_unknowns():
    errs = T.reparam_errors({"push.wobble": (0, 1), "nope.force": (0, 1), "force": (5.0, 1.0)}, LIB.prims, LIMITS,
                            primitive="push")
    assert [e.code for e in errs] == ["arg", "arg", "range"]
    assert _codes(T.reparam_errors({"push.speed": (0.01, float("inf"))}, LIB, LIMITS)) == {"range"}


# ---------------------------------------------------------------- reach plumbing
class _FakeGrid:
    def __init__(self, ok: bool):
        self.ok = ok
        self.calls: list = []

    def reachable(self, pos, axis):
        self.calls.append(np.asarray(pos))
        return self.ok


def test_reach_with_tool_tip_offset_and_callable():
    c = CTL["push_button"]
    grid = _FakeGrid(True)
    prog = _prog("With(2, push($target))")
    assert T.typecheck(prog, RACK, LIB.prims, LIMITS, grid, control=c, panel=PANEL) == []
    point, approach = T.control_target(PANEL, c)
    # the pad tip sits thickness + 12 mm past the jaw: the frame is queried that far back
    np.testing.assert_allclose(grid.calls[-1], point - approach * 0.017, atol=1e-9)
    seen = []

    def provider(tip):
        seen.append(tip)
        return _FakeGrid(False)

    errs = T.typecheck(prog, RACK, LIB.prims, LIMITS, provider, control=c, panel=PANEL)
    assert _codes(errs) == {"reach"} and np.isclose(seen[0][2, 3], 0.017)


def test_reach_provider_uses_the_disk_cache(tmp_path):
    from sea.control.executor import stock_ik

    key = design_key(stock_ik().urdf, None, LIMITS)
    REACH.save(tmp_path / f"reach_{key}.npz")
    prov = T.ReachProvider(LIMITS, cache_dir=tmp_path)
    g = prov(None)
    assert g is prov(None)
    np.testing.assert_array_equal(g.occ, REACH.occ)


def test_control_target_matches_the_sim():
    from sea.sim.build_model import build_model

    sim = build_model(STOCK, PANEL)
    for c in PANEL.controls:
        h = sim.controls[c.id]
        point, approach = T.control_target(PANEL, c)
        body_pt = h.geo.contact_body
        if c.handle in ("tab", "pull") and h.geo.handle_body is not None and h.geo.flap is None:
            body_pt = h.geo.handle_body
        np.testing.assert_allclose(point, sim.control_point_world(c.id, body_pt), atol=1e-6)
        np.testing.assert_allclose(approach, sim.ctl_dir_world(c.id, h.geo.approach_ctl), atol=1e-9)


# ---------------------------------------------------------------- program and tool selection
def test_wrap_with_tool_prefers_stock_then_rack_and_avoids_banned():
    lib = LIB.prims
    press = _seed("press_button")
    assert T.wrap_with_tool(press, RACK, CTL["push_button"], lib) is press  # stock edge works
    knob = T.wrap_with_tool(_seed("turn_knob"), RACK, CTL["knob"], lib, LIMITS, banned=["pinch"])
    assert knob is not None and dsl.tool_indices(knob.root) == [1]  # cup, not a banned stock pinch
    stock_knob = T.wrap_with_tool(_seed("turn_knob"), STOCK, CTL["knob"], lib, banned=["pinch"])
    assert stock_knob is not None and dsl.tool_indices(stock_knob.root) == []  # only option: the verifier flags it
    lever = T.wrap_with_tool(_seed("pull_lever"), RACK, CTL["lever"], lib)
    assert lever is not None and dsl.tool_indices(lever.root) == [2]
    assert T.wrap_with_tool(_seed("pull_lever"), STOCK, CTL["lever"], lib) is None
    mounted = _prog("With(0, hook_pull($target))", "door")
    assert T.wrap_with_tool(mounted, RACK, CTL["door"], lib) is mounted
    assert T.wrap_with_tool(_prog("With(2, hook_pull($target))", "door"), RACK, CTL["door"], lib) is None


def test_applicable_programs_orders_by_smoothed_pass_rate():
    lib = Library.seed()
    door = CTL["door"]
    assert T.applicable_programs(door, STOCK, lib, LIMITS, REACH, PANEL) == []
    got = T.applicable_programs(door, RACK, lib, LIMITS, REACH, PANEL)
    assert [p.name for p in got] == ["open_door_pull"] and dsl.tool_indices(got[0].body.root) == [0]
    assert T.applicable_programs(CTL["slot"], RACK, lib, LIMITS) == []  # no seed program inserts
    c = CTL["push_button"]
    alt = PrimitiveDef(name="a_press", kind="composite", doc="d", provides={"actuates": "push_button"},
                       body=_prog("Seq(approach($target), push($target), retreat())"))
    lib.add(alt)
    assert [p.name for p in T.applicable_programs(c, STOCK, lib, LIMITS)] == ["a_press", "press_button"]
    for _ in range(3):
        lib.record_use("a_press", passed=False, iteration=0)
    lib.record_use("press_button", passed=True, iteration=0)
    assert [p.name for p in T.applicable_programs(c, STOCK, lib, LIMITS)] == ["press_button", "a_press"]


@pytest.mark.parametrize("cls", ["door", "drawer"])
def test_accepts_without_handle_is_union(cls):
    assert T.accepted_contacts(cls=cls, handle="none") == frozenset({"pinch", "hook", "pad", "probe"})
