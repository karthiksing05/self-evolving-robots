"""Posture: derive, accept, anneal (spec 8.9, 9; test list 12.2)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sea.contracts import Posture, Prediction
from sea.spec import posture as P

CONSERVATIVE = dict(force_cap_fraction=0.6, max_rung=2, max_edits=1, max_param_delta=0.3, repeats_before_fail=3,
                    accept_rule="lcb", target_sign=-1.0, regression_panel="full", surprise_tau=3.0, k_promote=4)
OPTIMISTIC = dict(force_cap_fraction=1.0, max_rung=5, max_edits=3, max_param_delta=1.0, repeats_before_fail=1,
                  accept_rule="ucb", target_sign=1.0, regression_panel="sample", surprise_tau=5.0, k_promote=2)


def _matches(eff: P.EffectiveParams, expected: dict) -> None:
    for name, value in expected.items():
        got = getattr(eff, name)
        if isinstance(value, float):
            assert got == pytest.approx(value), name
        else:
            assert got == value, name


def test_derive_matches_table_at_plus_minus_one():
    _matches(P.derive(Posture(beta=-1.0)), CONSERVATIVE)
    _matches(P.derive(Posture(beta=1.0)), OPTIMISTIC)
    assert P.derive(Posture(beta=1.0)).regression_sample_n == 4


def test_derive_interpolates_between_columns():
    mid = P.derive(Posture(beta=0.0))
    _matches(mid, dict(force_cap_fraction=0.8, max_rung=4, max_edits=2, max_param_delta=0.65, repeats_before_fail=2,
                       accept_rule="mean", target_sign=0.0, regression_panel="full", surprise_tau=4.0, k_promote=3))
    half = P.derive(Posture(beta=0.5))
    assert half.force_cap_fraction == pytest.approx(0.9)
    assert half.surprise_tau == pytest.approx(4.5)
    assert half.max_param_delta == pytest.approx(0.825)
    assert half.regression_panel == "sample"


@settings(derandomize=True, max_examples=60)
@given(st.floats(-1.0, 1.0), st.floats(-1.0, 1.0))
def test_derive_is_monotone_in_beta(b1, b2):
    lo, hi = sorted((b1, b2))
    a, b = P.derive(Posture(beta=lo)), P.derive(Posture(beta=hi))
    assert a.force_cap_fraction <= b.force_cap_fraction + 1e-12
    assert a.surprise_tau <= b.surprise_tau + 1e-12
    assert a.max_rung <= b.max_rung and a.max_edits <= b.max_edits
    assert a.repeats_before_fail >= b.repeats_before_fail and a.k_promote >= b.k_promote


def test_derive_clamps_beyond_columns():
    two, five, minus = P.derive(Posture(beta=2.0)), P.derive(Posture(beta=5.0)), P.derive(Posture(beta=-3.0))
    _matches(two, OPTIMISTIC)
    assert two.beta == 2.0 and five.beta == 2.0 and minus.beta == -2.0
    _matches(minus, CONSERVATIVE)


def test_accept_rule_thresholds():
    assert P.derive(Posture(beta=-0.34)).accept_rule == "lcb"
    assert P.derive(Posture(beta=-0.33)).accept_rule == "mean"
    assert P.derive(Posture(beta=0.33)).accept_rule == "mean"
    assert P.derive(Posture(beta=0.34)).accept_rule == "ucb"


def test_explicit_fields_override_beta():
    eff = P.derive(Posture(beta=1.0, accept_rule="lcb", k_promote=7, force_cap_fraction=0.5, max_edits=1,
                           repeats_before_fail=4, surprise_tau=2.5, regression_panel="full", max_param_delta=0.1))
    assert (eff.accept_rule, eff.k_promote, eff.force_cap_fraction, eff.max_edits) == ("lcb", 7, 0.5, 1)
    assert (eff.repeats_before_fail, eff.surprise_tau, eff.regression_panel) == (4, 2.5, "full")
    assert eff.max_param_delta == 0.1
    assert eff.max_rung == 5  # not overridden


def test_per_area_overrides_apply():
    pst = Posture(beta=1.0, per_area={"force": -1.0, "library": 0.0})
    _matches(P.derive(pst, area="force"), CONSERVATIVE)
    _matches(P.derive(pst, area="shape"), OPTIMISTIC)  # no override -> global beta
    assert P.derive(pst, area="library").k_promote == 3
    assert P.derive(pst).beta == 1.0
    with pytest.raises(ValueError):
        P.derive(pst, area="vibes")


def test_score_knob():
    assert P.score(1.0, 0.5, 0.2, beta=2.0) == pytest.approx(1.8)
    assert P.score(1.0, 0.5, 0.2, beta=-1.0) == pytest.approx(0.3)


# ---------------------------------------------------------------- accept
def _eff(rule: str) -> P.EffectiveParams:
    return P.derive(Posture(beta=0.0, accept_rule=rule))


def _reps(values: list[list[float]]) -> list[dict[str, float]]:
    return [{f"p:c{i}": v for i, v in enumerate(rep)} for rep in values]


def test_accept_rules_on_clear_improvement_and_regression():
    base = _reps([[0, 0, 1, 1]] * 3)
    trial = _reps([[1, 1, 1, 1]] * 3)
    for rule in ("lcb", "mean", "ucb"):
        d = P.accept(trial, base, _eff(rule))
        assert d.accepted and d.rule == rule
        assert d.delta_mean == pytest.approx(0.5)
        assert d.delta_lo <= d.delta_mean <= d.delta_hi
    d = P.accept(trial, base, _eff("mean"), regressions=["ref:c1", "ref:c1"])
    assert not d.accepted and d.regressions == ["ref:c1"]


def test_accept_rule_strictness_on_a_noisy_small_gain():
    base = _reps([[0, 0, 0, 0, 0, 0, 0, 0]] * 3)
    trial = _reps([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    assert not P.accept(trial, base, _eff("lcb")).accepted
    assert P.accept(trial, base, _eff("mean")).accepted
    assert P.accept(trial, base, _eff("ucb")).accepted


@settings(derandomize=True, max_examples=60, deadline=None)
@given(st.lists(st.lists(st.sampled_from([0.0, 1.0]), min_size=5, max_size=5), min_size=3, max_size=3),
       st.lists(st.lists(st.sampled_from([0.0, 1.0]), min_size=5, max_size=5), min_size=3, max_size=3),
       st.floats(-0.3, 0.3), st.integers(0, 3))
def test_accept_is_monotone_across_rules(trial_v, base_v, cost_delta, seed):
    trial, base = _reps(trial_v), _reps(base_v)
    got = {r: P.accept(trial, base, _eff(r), cost_delta=cost_delta, seed=seed).accepted for r in ("lcb", "mean", "ucb")}
    if not got["ucb"]:
        assert not got["mean"] and not got["lcb"]
    if got["lcb"]:
        assert got["mean"] and got["ucb"]


def test_accept_unpaired_base_uses_key_means_and_cost_delta():
    base = [{"a": 0.0, "b": 1.0}]  # one base repeat vs three trial repeats
    trial = [{"a": 1.0, "b": 1.0}] * 3
    d = P.accept(trial, base, _eff("mean"))
    assert d.delta_mean == pytest.approx(0.5)
    d2 = P.accept(trial, base, _eff("mean"), cost_delta=0.6)
    assert d2.delta_mean == pytest.approx(-0.1) and not d2.accepted


def test_accept_with_no_overlapping_controls_rejects():
    d = P.accept([{"a": 1.0}], [{"b": 0.0}], _eff("ucb"))
    assert not d.accepted and d.delta_mean == 0.0 and d.delta_lo == 0.0 and d.delta_hi == 0.0


def test_bootstrap_is_seeded():
    import numpy as np

    x = np.array([0.0, 1.0, 0.0, 1.0, 1.0, -1.0])
    assert P.bootstrap_interval(x, seed=3) == P.bootstrap_interval(x, seed=3)


def test_prediction_score_and_extra_runs():
    pred = Prediction(flips_to_pass=["p:a", "p:b"], may_regress=[], confidence=0.7)
    assert P.prediction_score(pred, ["p:a", "p:z"]) == 0.5
    assert P.prediction_score(None, ["p:a"]) == 0.0
    assert P.prediction_score(Prediction(confidence=0.5), ["p:a"]) == 0.0
    d = P.accept(_reps([[1, 1]]), _reps([[0, 1]]), _eff("mean"), prediction=pred, flipped=["p:b"])
    assert d.prediction_score == 0.5
    assert P.extra_trial_runs("ucb") == 1
    assert P.extra_trial_runs("mean") == 0 and P.extra_trial_runs("lcb") == 0


# ---------------------------------------------------------------- anneal
@settings(derandomize=True, max_examples=80)
@given(st.floats(-2.0, 2.0), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_anneal_moves_beta_toward_conservative_as_budgets_fall(beta0, r1, r2):
    hi, lo = max(r1, r2), min(r1, r2)
    pst = Posture(beta=beta0)
    b_hi = P.anneal(pst, beta0, {"print_minutes": hi, "llm_usd": 1.0}).beta
    b_lo = P.anneal(pst, beta0, {"print_minutes": lo, "llm_usd": 1.0}).beta
    assert abs(b_lo - P.BETA_MIN) <= abs(b_hi - P.BETA_MIN) + 1e-12
    assert P.anneal(pst, beta0, {"x": 1.0}).beta == pytest.approx(beta0)
    assert P.anneal(pst, beta0, {"x": 0.0}).beta == pytest.approx(P.BETA_MIN)


def test_anneal_edge_cases():
    pst = Posture(beta=1.0, per_area={"force": -1.0})
    out = P.anneal(pst, 1.0, {"a": 0.5})
    assert out.beta == pytest.approx(0.0) and out.per_area == {"force": -1.0}
    assert P.anneal(pst, 1.0, {}).beta == 1.0
    assert P.anneal(pst, 1.0, {"a": 1.7}).beta == 1.0  # clamped fraction
    frozen = Posture(beta=1.0, anneal_with_budget=False)
    assert P.anneal(frozen, 1.0, {"a": 0.0}) is frozen
