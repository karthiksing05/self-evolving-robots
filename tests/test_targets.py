"""Target selection (spec 8.4; test list 12.2)."""

from __future__ import annotations

import pytest

from sea.contracts import Budgets, ControlResult, CostVector
from sea.propose import targets as T
from sea.spec.ledger import Ledger


def _result(key: str, verdict: str = "stuck") -> ControlResult:
    panel, control = key.split(":")
    return ControlResult(panel_id=panel, control_id=control, cls="push_button", verdict=verdict, peak_force=9.5,
                         travel_reached=0.0005, detents=0, time_s=4.0)


def _esc(key: str, verdict: str = "stuck", concept: int = 0, unc: float = 0.5, size: int = 1) -> T.Escalation:
    return T.Escalation(result=_result(key, verdict), reason="trials_exhausted", concept_id=concept,
                        uncertainty=unc, concept_size=size)


def test_gain_and_cost_tables():
    assert T.gain_est("unreachable") == T.gain_est("no_applicable_primitive") == 1.0
    assert T.gain_est("stuck") == T.gain_est("wrong_behavior") == 0.7
    assert T.gain_est("no_actuation") == T.gain_est("rule_violation") == 0.4
    assert T.cost_est("unreachable")["print_minutes"] > T.cost_est("stuck").get("print_minutes", 0.0)
    assert T.cost_est("pass") == {}
    c = T.cost_est("stuck")
    c["llm_usd"] = 99.0  # callers get a copy
    assert T.cost_est("stuck")["llm_usd"] != 99.0


def test_scarcity_weights_rise_as_a_budget_falls():
    led = Ledger(Budgets())
    before = led.scarcity_weights()["print_minutes"]
    led.charge(CostVector(print_minutes=150))
    after = led.scarcity_weights()["print_minutes"]
    assert after > before
    # a scarcer print budget makes a body-change target score lower
    esc = _esc("p1:c1", "unreachable")
    assert T.score(esc, 0.0, led.scarcity_weights()) < T.score(esc, 0.0, Ledger(Budgets()).scarcity_weights())


def test_beta_sign_orders_by_uncertainty():
    w = Ledger(Budgets()).scarcity_weights()
    sure, unsure = _esc("p1:a", unc=0.1, concept=1), _esc("p1:b", unc=0.9, concept=2)
    cons = T.pick_targets([unsure, sure], beta=-1.0, weights=w)
    opt = T.pick_targets([sure, unsure], beta=1.0, weights=w)
    assert cons[0].keys == ["p1:a"]
    assert opt[0].keys == ["p1:b"]


def test_gap_mode_ignores_uncertainty():
    w = Ledger(Budgets()).scarcity_weights()
    a, b = _esc("p1:a", unc=0.1, concept=1), _esc("p1:b", unc=0.9, concept=2)
    assert T.score(a, 1.0, w, mode="gap") == pytest.approx(T.score(b, 1.0, w, mode="gap"))
    picked = T.pick_targets([b, a], beta=1.0, weights=w, mode="gap")
    assert picked[0].keys == ["p1:a"]  # tie broken by key


def test_concept_size_multiplies_gain_and_groups_and_cap():
    w = Ledger(Budgets()).scarcity_weights()
    big = [_esc(f"p{i}:c1", "stuck", concept=7, unc=0.2, size=3) for i in range(3)]
    lone = _esc("p9:c4", "unreachable", concept=8, unc=0.2, size=1)
    extra = _esc("p8:c2", "no_actuation", concept=9, unc=0.2)
    groups = T.pick_targets([lone, extra, *big], beta=0.0, weights=w, max_targets=3)
    assert len(groups) == 1 and groups[0].concept_id == 7
    assert groups[0].keys == ["p0:c1", "p1:c1", "p2:c1"]
    assert groups[0].panel_ids == ["p0", "p1", "p2"]
    groups = T.pick_targets([lone, extra, *big], beta=0.0, weights=w, max_targets=5)
    assert [g.concept_id for g in groups] == [7, 8, 9]
    assert groups[0].score >= groups[1].score >= groups[2].score
    assert T.pick_targets([], beta=0.0, weights=w) == []
    assert big[0].key == "p0:c1"
