"""Escalation rule (spec 3.3) and the ladder's mechanical check (spec 3.4)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.contracts import ControlResult, Design, HardLimits, Posture, Prediction, Proposal, RunSpec, Vec3
from sea.loop import State, escalations
from sea.propose.ladder import ladder_errors, validate
from sea.sim.env import ControlRun, SuiteRun
from sea.spec.posture import derive

LIMITS = HardLimits(workspace_box=(Vec3(x=0.1, y=-0.2, z=-0.1), Vec3(x=0.4, y=0.2, z=0.25)))


def _result(cid: str, verdict: str, *, z: float = 0.0, concept_n: int = 50) -> ControlResult:
    return ControlResult(panel_id="p0", control_id=cid, cls="push_button", verdict=verdict, peak_force=3.0,
                         travel_reached=0.2, detents=0, time_s=4.0, surprise_z={"peak_force": z} if z else {},
                         concept_n=concept_n)


class _Memory:
    """Stand-in for ConceptMemory: non-empty context tree, fixed concepts."""

    class _T:
        empty = False

    trees = {"context": _T()}

    def concept_id(self, ctx: np.ndarray) -> int:
        return int(ctx[0])

    def uncertainty(self, ctx: np.ndarray) -> float:
        return 0.5


def _state(memory=None, beta: float = 0.0) -> State:
    spec = RunSpec(seed=0, limits=LIMITS, posture=Posture(beta=beta))
    return State(spec=spec, built=None, library=None, ledger=None, posture=spec.posture, beta0=beta,  # type: ignore[arg-type]
                 encoder=None, memory=memory)


def _suite(*runs: ControlRun) -> SuiteRun:
    return SuiteRun(list(runs))


@pytest.mark.parametrize("verdict", ["unreachable", "rule_violation", "no_applicable_primitive"])
def test_structural_verdicts_escalate(verdict: str) -> None:
    st = _state(_Memory())
    run = ControlRun(_result("c0", verdict), None)
    st.contexts[run.key] = np.zeros(4)
    (e,) = escalations(st, _suite(run))
    assert e.reason == "structural" and e.result.escalated and e.result.escalation_reason == "structural"


def test_exhausted_local_search_escalates() -> None:
    st = _state(_Memory())
    run = ControlRun(_result("c0", "stuck"), None, s1_exhausted=True)
    st.contexts[run.key] = np.zeros(4)
    assert [e.reason for e in escalations(st, _suite(run))] == ["s1_exhausted"]


def test_surprise_above_tau_escalates_depending_on_posture() -> None:
    # tau = 3 sigma conservative, 5 sigma optimistic; a 4 sigma surprise splits them.
    for beta, expect in ((-1.0, ["surprise"]), (1.0, [])):
        st = _state(_Memory(), beta)
        assert derive(st.posture).surprise_tau == (3.0 if beta < 0 else 5.0)
        run = ControlRun(_result("c0", "stuck", z=-4.0), None)
        st.contexts[run.key] = np.zeros(4)
        assert [e.reason for e in escalations(st, _suite(run))] == expect


def test_unfamiliar_control_escalates() -> None:
    st = _state(_Memory())
    run = ControlRun(_result("c0", "wrong_behavior", concept_n=2), None)
    st.contexts[run.key] = np.zeros(4)
    assert [e.reason for e in escalations(st, _suite(run))] == ["unfamiliar"]


def test_passes_unconfirmed_and_familiar_failures_do_not_escalate() -> None:
    st = _state(_Memory())
    runs = [ControlRun(_result("c0", "pass"), None),
            ControlRun(_result("c1", "unreachable"), None, confirmed=False),
            ControlRun(_result("c2", "stuck"), None)]  # familiar, not surprising, S1 not exhausted
    for r in runs:
        st.contexts[r.key] = np.zeros(4)
    assert escalations(st, _suite(*runs)) == []


def test_concepts_group_and_count_siblings() -> None:
    st = _state(_Memory())
    runs = [ControlRun(_result(f"c{i}", "unreachable"), None) for i in range(3)]
    for i, r in enumerate(runs):
        st.contexts[r.key] = np.array([float(i % 2), 0, 0, 0])
    esc = {e.key: e for e in escalations(st, _suite(*runs))}
    assert esc["p0:c0"].concept_id == esc["p0:c2"].concept_id != esc["p0:c1"].concept_id
    assert esc["p0:c0"].concept_size == 2 and esc["p0:c1"].concept_size == 1


def test_without_memory_every_class_is_one_concept() -> None:
    st = _state(None)
    runs = [ControlRun(_result(f"c{i}", "rule_violation"), None) for i in range(2)]
    for r in runs:
        st.contexts[r.key] = np.zeros(4)
    esc = escalations(st, _suite(*runs))
    assert len({e.concept_id for e in esc}) == 1 and all(e.uncertainty == 1.0 for e in esc)


def _proposal(kind: str, rung: int, ruled_out: list[str]) -> Proposal:
    from sea.contracts import DesignDelta
    from sea.design.space import make_tool_spec

    delta = DesignDelta(rack_add=[make_tool_spec("hook")]) if kind in ("new_tool", "use_tool") else None
    return Proposal(kind=kind, rung=rung, rationale="r", hypothesis="h", ruled_out=ruled_out, design_delta=delta,
                    reparam={"force": (0.5, 9.0)} if kind == "reparam" else None,
                    prediction=Prediction(confidence=0.5))


def test_ladder_requires_every_cheaper_rung() -> None:
    full = ["rung0: parameters cannot add a contact", "rung1: no composition makes a hook",
            "rung2: no known tool"]
    assert ladder_errors(_proposal("new_tool", 3, full)) == []
    missing = ladder_errors(_proposal("new_tool", 3, full[:1] + full[2:]))
    assert len(missing) == 1 and "rung1:" in missing[0]
    assert ladder_errors(_proposal("new_tool", 3, [*full[:2], "rung2:   "]))  # empty reason
    assert ladder_errors(_proposal("reparam", 0, [])) == []
    eff = derive(Posture(beta=1.0))
    assert validate(_proposal("new_tool", 3, full), Design(id="d"), eff, rack_slots=4) == []
