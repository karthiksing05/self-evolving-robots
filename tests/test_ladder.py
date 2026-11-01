"""Escalation ladder, posture and structure checks on proposals (spec 3.4, 9)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sea.compose import dsl
from sea.contracts import (
    PROPOSAL_RUNG,
    ArmSpec,
    Design,
    DesignDelta,
    Posture,
    Prediction,
    PrimitiveDef,
    Program,
    Proposal,
    ToolSpec,
)
from sea.propose import ladder as L
from sea.spec.posture import derive

PAD = ToolSpec(template="pad", params={"w": 12.0, "h": 12.0, "thickness": 4.0, "friction": 0.8}, contact="pad")
PROGRAM = Program(id="s2_001", target_cls="door", source="s2",
                  root=dsl.seq(dsl.prim("approach", "$target"), dsl.prim("push", "$target"), dsl.prim("retreat")))


def _payload(kind: str) -> dict:
    if kind == "program":
        return {"program": PROGRAM}
    if kind == "reparam":
        return {"reparam": {"push.force": (2.0, 9.0)}}
    if kind == "promote":
        return {"promote": PrimitiveDef(name="open_push_latch_door", kind="composite", doc="d", body=PROGRAM)}
    if kind in ("use_tool", "new_tool"):
        return {"design_delta": DesignDelta(rack_add=[PAD])}
    if kind == "finger":
        return {"design_delta": DesignDelta(fingers={"reach_mm": 10.0})}
    if kind == "arm":
        return {"design_delta": DesignDelta(arm={"fore_scale": 1.1})}
    return {"design_delta": DesignDelta(servo_voltage={"wrist_flex": 12.0})}


def _prop(kind: str, ruled_out: list[str] | None = None, rung: int | None = None, **kw) -> Proposal:
    r = PROPOSAL_RUNG[kind]
    if ruled_out is None:
        ruled_out = [f"rung{i}: cheaper rung {i} cannot fix it" for i in range(r or 0)]
    fields = {**_payload(kind), **kw}
    return Proposal(kind=kind, rung=rung if rung is not None else (r or 0), rationale="peak_force=9.5",
                    hypothesis="h", ruled_out=ruled_out, prediction=Prediction(confidence=0.6), **fields)


OPT = derive(Posture(beta=1.0))
CONS = derive(Posture(beta=-1.0))
STOCK = Design(id="d000")


@pytest.mark.parametrize("kind", [k for k, r in PROPOSAL_RUNG.items() if r])
def test_rung_k_missing_one_entry_is_rejected(kind):
    k = PROPOSAL_RUNG[kind]
    full = _prop(kind)
    assert L.ladder_errors(full) == []
    for missing in range(k):
        entries = [f"rung{i}: reason {i}" for i in range(k) if i != missing]
        errs = L.ladder_errors(_prop(kind, ruled_out=entries))
        assert any(f"rung{missing}:" in e for e in errs)


@settings(derandomize=True, max_examples=40)
@given(st.sampled_from(sorted(PROPOSAL_RUNG)), st.lists(st.integers(0, 5), max_size=6))
def test_ladder_rule_property(kind, present):
    entries = [f"rung{i}: because" for i in present]
    errs = L.ladder_errors(_prop(kind, ruled_out=entries))
    k = PROPOSAL_RUNG[kind]
    if k is None:
        assert errs == []
    else:
        assert (errs == []) == set(range(k)).issubset(present)


def test_ladder_rejects_empty_reason_and_wrong_rung():
    assert L.ladder_errors(_prop("program", ruled_out=["rung0:   "]))
    assert L.ladder_errors(_prop("program", ruled_out=["rung0 no colon"]))
    errs = L.ladder_errors(_prop("program", rung=4))
    assert any("is rung 1" in e for e in errs)
    assert L.ladder_errors(_prop("promote", ruled_out=[], rung=9)) == []


def test_posture_max_rung_unless_forced():
    p = _prop("new_tool")
    assert L.posture_errors(p, OPT, STOCK) == []
    errs = L.posture_errors(p, CONS, STOCK)
    assert any("max rung" in e for e in errs)
    assert not any("max rung" in e for e in L.posture_errors(p, CONS, STOCK, forced=True))
    assert L.posture_errors(_prop("promote"), CONS, STOCK) == []


def test_posture_edit_count_and_param_delta():
    many = _prop("arm", design_delta=DesignDelta(arm={"upper_scale": 1.1, "fore_scale": 1.1}))
    assert any("max_edits" in e for e in L.posture_errors(many, CONS, STOCK))
    assert L.count_edits(many) == 2
    assert L.count_edits(_prop("reparam")) == 1
    assert L.count_edits(_prop("new_tool")) == 1
    big = _prop("arm", design_delta=DesignDelta(arm={"upper_scale": 1.5}))
    assert any("max_param_delta" in e for e in L.posture_errors(big, CONS, STOCK, forced=True))
    assert not any("max_param_delta" in e for e in L.posture_errors(big, OPT, STOCK))
    zero_base = _prop("arm", design_delta=DesignDelta(arm={"wrist_offset_mm": 5.0}))  # 0 -> 5 mm, scale 20 mm
    assert L.posture_errors(zero_base, CONS, STOCK, forced=True) == []
    fingers = _prop("finger", design_delta=DesignDelta(fingers={"pad_tpu_mm": 3.0}))
    assert any("fingers.pad_tpu_mm" in e for e in L.posture_errors(fingers, CONS, STOCK, forced=True))
    bogus = _prop("arm", design_delta=DesignDelta(arm={"wingspan": 2.0}))
    assert any("unknown arm field" in e for e in L.posture_errors(bogus, OPT, STOCK))
    sv = _prop("arm", design_delta=DesignDelta(arm={"servo_voltage": 12.0}))
    assert any("unknown arm field" in e for e in L.posture_errors(sv, OPT, STOCK))
    scaled = Design(id="d1", arm=ArmSpec(upper_scale=1.2))
    ok = _prop("arm", design_delta=DesignDelta(arm={"upper_scale": 1.5}))
    assert L.posture_errors(ok, CONS, scaled, forced=True) == []  # |1.5-1.2|/1.2 = 0.25


def test_structure_payload_consistency():
    for kind in PROPOSAL_RUNG:
        assert L.structure_errors(_prop(kind), STOCK, rack_slots=4) == [], kind
    bare = {k: _prop(k, program=None, reparam=None, promote=None, design_delta=None)
            for k in PROPOSAL_RUNG}
    assert L.structure_errors(bare["program"], STOCK, 4) == ["program proposal without a program"]
    assert L.structure_errors(bare["reparam"], STOCK, 4) == ["reparam proposal without new ranges"]
    assert L.structure_errors(bare["promote"], STOCK, 4) == ["promote proposal without a primitive"]
    for kind in L.BODY_KINDS:
        assert L.structure_errors(bare[kind], STOCK, 4) == [f"{kind} proposal without a design_delta"]
    empty = DesignDelta()
    expect = {"use_tool": "use_tool needs", "new_tool": "without rack_add", "finger": "without finger",
              "arm": "without arm", "actuator": "without servo_voltage"}
    for kind, msg in expect.items():
        errs = L.structure_errors(_prop(kind, design_delta=empty), STOCK, 4)
        assert len(errs) == 1 and msg in errs[0], kind
    via_program = _prop("use_tool", design_delta=empty, program=PROGRAM)
    assert L.structure_errors(via_program, STOCK, 4) == []


def test_structure_rack_slots_and_removals():
    full = Design(id="d2", tool_rack=[PAD, PAD])
    add = _prop("new_tool")
    assert any("name tools to remove" in e for e in L.structure_errors(add, full, rack_slots=2))
    swap = _prop("new_tool", design_delta=DesignDelta(rack_add=[PAD], rack_remove=[1]))
    assert L.structure_errors(swap, full, rack_slots=2) == []
    bad = _prop("new_tool", design_delta=DesignDelta(rack_add=[PAD], rack_remove=[5]))
    errs = L.structure_errors(bad, full, rack_slots=2)
    assert any("not on the rack" in e for e in errs) and any("name tools to remove" in e for e in errs)
    dup = _prop("new_tool", design_delta=DesignDelta(rack_add=[PAD], rack_remove=[0, 0]))
    assert any("twice" in e for e in L.structure_errors(dup, full, rack_slots=2))


def test_validate_combines_all_checks():
    p = _prop("new_tool", ruled_out=["rung0: no"])
    errs = L.validate(p, STOCK, CONS, rack_slots=4)
    assert any("rung1:" in e for e in errs) and any("max rung" in e for e in errs)
    assert L.validate(_prop("program"), STOCK, CONS, rack_slots=4) == []
