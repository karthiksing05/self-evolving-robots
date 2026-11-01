"""Proposers: scripted System 2, fixture recording, body-search baselines, s1_only (spec 8.6, E1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sea.compose.library import Library
from sea.contracts import (
    Budgets,
    ControlResult,
    CostVector,
    Decision,
    Design,
    HardLimits,
    Posture,
    Report,
    Vec3,
)
from sea.design.space import make_tool_spec
from sea.propose.baselines import (
    BOProposer,
    CMAProposer,
    ProposeContext,
    RandomProposer,
    RecordingLLM,
    S1OnlyProposer,
    S2Proposer,
    ScriptedLLM,
    _rack_delta,
    make_proposer,
    ruled_out_for,
)
from sea.propose.ladder import ladder_errors
from sea.propose.llm import MockLLM
from sea.propose.prompt import TraceSummary
from sea.propose.targets import Escalation, TargetGroup
from sea.sim.panels import reference_panel
from sea.spec.posture import derive

LIMITS = HardLimits(workspace_box=(Vec3(x=0.1, y=-0.2, z=-0.1), Vec3(x=0.4, y=0.2, z=0.25)))
PANEL = reference_panel()
BY_CLS = {c.cls: c for c in PANEL.controls}


def _result(cls: str, verdict: str) -> ControlResult:
    c = BY_CLS[cls]
    return ControlResult(panel_id=PANEL.id, control_id=c.id, cls=cls, verdict=verdict, peak_force=1.0,
                         travel_reached=0.1, detents=0, time_s=3.0)


def _ctx(targets: list[tuple[str, str]], *, design: Design | None = None, contacts: tuple[str, ...] = (),
         failed: str = "", tool_library=(), beta: float = 1.0, library: Library | None = None) -> ProposeContext:
    results = [_result(cls, v) for cls, v in targets]
    groups = [TargetGroup(concept_id=i, escalations=[Escalation(result=r, reason="structural", concept_id=i,
                                                                 uncertainty=0.5)], score=1.0)
              for i, r in enumerate(results)]
    summaries = {r.key: TraceSummary(key=r.key, cls=r.cls, verdict=r.verdict, peak_force=1.0, travel_reached=0.1,
                                     travel_required=BY_CLS[r.cls].travel, contact_types=list(contacts),
                                     handle=BY_CLS[r.cls].handle) for r in results}
    report = Report(run_id="t", iteration=0, design_id="d", panels=[PANEL.id], results=results, control_coverage=1.0,
                    pass_rate=0.0, coverage_reward=0.0, cost=CostVector(), reward=0.0)
    return ProposeContext(
        iteration=0, targets=groups, report=report, summaries=summaries,
        controls={f"{PANEL.id}:{c.id}": (c, PANEL) for c in PANEL.controls}, design=design or Design(id="d"),
        library=library or Library.seed(), eff=derive(Posture(beta=beta)), limits=LIMITS, budgets=Budgets(),
        remaining={"print_minutes": 1.0}, tool_library=list(tool_library),
        failed_primitive={r.key: failed for r in results} if failed else {},
    )


def _scripted(ctx: ProposeContext, tmp: Path | None = None):
    s2 = S2Proposer(ScriptedLLM(record_dir=tmp), model="scripted")
    return s2, s2.propose(ctx)


def test_pinch_on_a_tab_gets_a_new_hook_with_a_full_ladder() -> None:
    _, (p,) = _scripted(_ctx([("drawer", "rule_violation")], contacts=("pinch",)))
    assert p.kind == "new_tool" and p.rung == 3 and ladder_errors(p) == []
    assert p.design_delta is not None and p.design_delta.rack_add[0].template == "hook"
    assert p.prediction.flips_to_pass == [f"{PANEL.id}:{BY_CLS['drawer'].id}"]


def test_known_tool_is_reused_before_printing_a_new_one() -> None:
    hook = make_tool_spec("hook", depth=14.0, throat=6.0, length=45.0)
    _, (p,) = _scripted(_ctx([("door", "no_applicable_primitive")], tool_library=[hook]))
    assert p.kind == "use_tool" and p.rung == 2 and p.design_delta.rack_add == [hook]


def test_conservative_posture_cannot_reach_new_templates() -> None:
    _, (p,) = _scripted(_ctx([("door", "no_applicable_primitive")], beta=-1.0))
    assert p.kind == "reparam"  # rung 3 is above the conservative max_rung: fall back to the cheap probe


def test_tool_sizing_follows_the_control() -> None:
    llm = ScriptedLLM()
    llm.bind(_ctx([("knob", "rule_violation")]))
    assert llm._tool_need(BY_CLS["knob"])[1]["inner_diameter"] == pytest.approx(BY_CLS["knob"].size_mm[0] + 2)
    blade = llm._tool_need(BY_CLS["slot"])[1]["blade_thickness"]
    assert 1.2 <= blade < BY_CLS["slot"].size_mm[1]
    assert llm._tool_need(BY_CLS["recessed_button"])[1]["length"] >= 20.0
    assert llm._tool_need(BY_CLS["slider"]) is None


def test_missing_program_comes_with_its_tool_or_alone() -> None:
    _, (p,) = _scripted(_ctx([("slot", "no_applicable_primitive")]))
    assert p.kind == "new_tool" and p.program is not None and p.design_delta.rack_add[0].template == "slot_inserter"
    rack = Design(id="d", tool_rack=[make_tool_spec("slot_inserter", blade_thickness=1.2, width=24, length=40)])
    _, (p,) = _scripted(_ctx([("slot", "no_applicable_primitive")], design=rack))
    assert p.kind == "program" and p.rung == 1 and ladder_errors(p) == [] and p.program.target_cls == "slot"


def test_stuck_control_gets_a_rung0_range_for_the_failing_primitive() -> None:
    _, (p,) = _scripted(_ctx([("push_button", "stuck")], failed="push"))
    assert p.kind == "reparam" and p.rung == 0 and p.reparam
    assert all(k.startswith("push.") for k in p.reparam)
    assert max(hi for k, (lo, hi) in p.reparam.items() if "force" in k) <= LIMITS.max_force_n
    assert list(p.reparam)[0] == "push.force"  # force before extents, within max_edits


def test_no_rule_matches_falls_back_to_one_cheap_probe() -> None:
    _, out = _scripted(_ctx([("slider", "unreachable")]))
    assert [p.kind for p in out] == ["reparam"]


def test_recorded_fixture_replays_through_mock_llm(tmp_path: Path) -> None:
    ctx = _ctx([("drawer", "rule_violation")], contacts=("pinch",))
    s2, out = _scripted(ctx, tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1
    replay = S2Proposer(MockLLM(tmp_path), model="scripted").propose(ctx)
    assert [p.model_dump() for p in replay] == [p.model_dump() for p in out]
    assert s2.llm.calls[0].n_proposals == 1 and s2.llm.total_usd == 0.0
    s2.observe(out[0], None)
    nomem = S2Proposer(ScriptedLLM(), model="m", use_memory=False)
    assert nomem.name == "s2_nomem"


def test_recording_wrapper_passes_attributes_through(tmp_path: Path) -> None:
    inner = ScriptedLLM()
    rec = RecordingLLM(inner, tmp_path / "fx")
    rec.log_path = tmp_path / "calls.jsonl"
    assert inner.log_path == tmp_path / "calls.jsonl"
    ctx = _ctx([("drawer", "rule_violation")], contacts=("pinch",))
    s2 = S2Proposer(rec, model="scripted")
    s2.propose(ctx)
    assert (tmp_path / "calls.jsonl").exists() and len(list((tmp_path / "fx").glob("*.json"))) == 1
    assert rec.total_usd == 0.0


def _decision(delta: float) -> Decision:
    return Decision(accepted=delta > 0, rule="mean", delta_mean=delta, delta_lo=delta, delta_hi=delta)


@pytest.mark.parametrize("cls", [RandomProposer, CMAProposer, BOProposer])
def test_body_search_proposes_whole_racks_and_learns(cls) -> None:
    prop = cls(["L0", "L2"], rack_slots=2, limits=LIMITS, seed=3)
    ctx = _ctx([("door", "no_applicable_primitive")], design=Design(id="d", tool_rack=[make_tool_spec("pad")]))
    seen = []
    for i in range(8):
        (p,) = prop.propose(ctx)
        assert p.kind == "new_tool" and ladder_errors(p) == [] and 1 <= len(p.design_delta.rack_add) <= 2
        assert p.design_delta.rack_remove == [0]
        seen.append(tuple(t.template for t in p.design_delta.rack_add))
        prop.observe(p, _decision(0.1 * (i % 3)) if i % 2 else None)
    assert len(set(seen)) > 1
    prop.observe(p, None)  # unknown or already-observed proposals are ignored


def test_s1_only_and_factory() -> None:
    (p,) = S1OnlyProposer().propose(_ctx([("push_button", "stuck")], failed="push"))
    assert p.kind == "reparam"
    S1OnlyProposer().observe(p, None)
    kw = dict(llm=ScriptedLLM(), model="m", levels=["L0"], rack_slots=4, limits=LIMITS, seed=0)
    assert {make_proposer(k, **kw).name for k in ("s2", "s2_nomem", "random", "cma", "bo", "s1_only")} == {
        "s2", "s2_nomem", "random", "cma", "bo", "s1_only"}
    with pytest.raises(ValueError):
        make_proposer("nope", **kw)


def test_rack_delta_names_a_removal_only_when_full() -> None:
    hook = make_tool_spec("hook")
    assert _rack_delta(Design(id="d"), hook, 4).rack_remove == []
    full = Design(id="d", tool_rack=[make_tool_spec("pad")] * 4)
    assert _rack_delta(full, hook, 4).rack_remove == [0]
    assert ruled_out_for(2, "x") == ["rung0: x", "rung1: x"]
