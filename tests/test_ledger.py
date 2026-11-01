"""Ledger and servo thermal model (spec 8.9, 10; test list 12.2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sea.contracts import Budgets, CostVector, DesignDelta, Prediction, Proposal, ToolSpec
from sea.spec.ledger import RESOURCES, Ledger, ReserveError, ServoThermal


def _proposal(kind: str = "new_tool") -> Proposal:
    rung = {"reparam": 0, "program": 1, "use_tool": 2, "new_tool": 3}.get(kind, 0)
    delta = DesignDelta(rack_add=[ToolSpec(template="pad", params={"w": 10.0}, contact="pad")])
    return Proposal(kind=kind, rung=rung, rationale="r", hypothesis="h", ruled_out=[],
                    design_delta=delta if kind in ("use_tool", "new_tool") else None,
                    prediction=Prediction(confidence=0.5))


cost_st = st.builds(
    CostVector,
    print_minutes=st.floats(0, 120), grams_pla=st.floats(0, 120), grams_tpu=st.floats(0, 20),
    llm_usd=st.floats(0, 8), approvals=st.integers(0, 2), sim_seconds=st.floats(0, 40000),
    servo_heat_j=st.floats(0, 100),
)


@settings(derandomize=True, max_examples=60, deadline=None)
@given(st.lists(cost_st, max_size=12))
def test_spend_never_negative_and_reserve_never_crossed(costs):
    led = Ledger(Budgets())
    for c in costs:
        try:
            led.charge(c)
        except ReserveError:
            pass
        for r in RESOURCES:
            assert getattr(led.spent, r) >= 0
            if r != "approvals":
                assert getattr(led.spent, r) <= led.budget(r) * (1 - led.budgets.reserve_fraction) + 1e-9
        assert all(0.0 <= v <= 1.0 for v in led.remaining().values())


@settings(derandomize=True, max_examples=30)
@given(st.sampled_from(RESOURCES + ("servo_heat_j",)), st.floats(-100, -1e-6))
def test_negative_charge_raises(resource, amount):
    led = Ledger(Budgets())
    with pytest.raises(ValueError):
        led.charge(CostVector(**{resource: amount}))
    assert led.spent == CostVector()


def test_reserve_boundary_and_final_evaluation():
    led = Ledger(Budgets(print_minutes=100))
    led.charge(CostVector(print_minutes=80))  # exactly at the reserve line
    assert led.would_break_reserve(CostVector(print_minutes=0.5))
    assert not led.would_break_reserve(CostVector(llm_usd=1.0))
    with pytest.raises(ReserveError):
        led.charge(CostVector(print_minutes=1))
    led.charge(CostVector(print_minutes=15), final=True)
    assert led.spent.print_minutes == 95
    assert led.remaining()["print_minutes"] == pytest.approx(0.05)


def test_approval_rate_limit_per_sim_hour():
    led = Ledger(Budgets(approvals_per_hour=2))
    led.charge(CostVector(approvals=1, sim_seconds=10))
    led.charge(CostVector(approvals=1, sim_seconds=10))
    assert led.would_break_reserve(CostVector(approvals=1))
    with pytest.raises(ReserveError):
        led.charge(CostVector(approvals=1))
    assert not led.would_break_reserve(CostVector(approvals=1, sim_seconds=3600))
    led.charge(CostVector(approvals=1, sim_seconds=3600))
    assert led.spent.approvals == 3


def test_estimate_by_kind():
    led = Ledger(Budgets())
    prog = led.estimate(_proposal("program"), llm_usd=0.05, sim_seconds=30, added_volume_cm3=5.0)
    assert prog == CostVector(llm_usd=0.05, sim_seconds=30)
    tool = led.estimate(_proposal("new_tool"), added_volume_cm3=10.0, n_parts=1, llm_usd=0.05)
    assert tool.print_minutes == pytest.approx(1.6 * 10 + 6)
    assert tool.grams_pla == pytest.approx(12.4) and tool.grams_tpu == 0 and tool.approvals == 1
    soft = led.estimate(_proposal("new_tool"), added_volume_cm3=2.0, material="tpu", grams=3.0)
    assert soft.grams_tpu == 3.0 and soft.grams_pla == 0 and soft.print_minutes == pytest.approx(1.6 * 2 + 6)
    reuse = led.estimate(_proposal("use_tool"))
    assert reuse.print_minutes == 0 and reuse.approvals == 1
    assert led.print_minutes_for(0.0, 0) == 0.0


def test_refit_lowers_error_on_synthetic_data():
    rng = np.random.default_rng(0)
    led = Ledger(Budgets(), refit_every=10)
    vols = rng.uniform(2, 30, size=20)
    actual_fn = lambda v: 2.3 * v + 11.0  # noqa: E731
    err_before, err_after = [], []
    for i, v in enumerate(vols):
        est = led.estimate(_proposal("new_tool"), added_volume_cm3=float(v), n_parts=1)
        act = CostVector(print_minutes=actual_fn(v) + float(rng.normal(0, 0.2)))
        (err_before if i < 10 else err_after).append(abs(act.print_minutes - est.print_minutes))
        led.record(est, act, float(v))
    assert led.print_a == pytest.approx(2.3, abs=0.05) and led.print_b == pytest.approx(11.0, abs=1.0)
    assert np.mean(err_after) < 0.25 * np.mean(err_before)
    assert len(led.history) == 20


def test_refit_needs_spread_volumes():
    led = Ledger(Budgets())
    assert not led.refit()
    for _ in range(3):
        led.record(CostVector(print_minutes=10), CostVector(print_minutes=12), 5.0)
    assert not led.refit()  # all volumes equal
    assert (led.print_a, led.print_b) == (1.6, 6.0)


def test_scarcity_weights_rise_as_a_budget_falls():
    led = Ledger(Budgets(llm_usd=10))
    w0 = led.scarcity_weights()
    assert sum(w0.values()) == pytest.approx(1.0)
    assert w0["llm_usd"] == pytest.approx(w0["print_minutes"])
    led.charge(CostVector(llm_usd=6))
    w1 = led.scarcity_weights()
    assert w1["llm_usd"] > w0["llm_usd"] and w1["llm_usd"] > w1["print_minutes"]
    led.charge(CostVector(llm_usd=10), final=True)  # past the end: floor at 0.05
    assert led.remaining()["llm_usd"] == 0.0
    assert led.scarcity_weights()["llm_usd"] > w1["llm_usd"]


def test_value_per_cost_floor_and_scaling():
    led = Ledger(Budgets())
    assert led.value_per_cost(1.0, CostVector()) == pytest.approx(1.0 / 0.002)
    cheap = led.value_per_cost(1.0, CostVector(llm_usd=0.1))
    dear = led.value_per_cost(1.0, CostVector(print_minutes=60, grams_pla=20, approvals=1))
    assert cheap > dear > 0


def test_zero_budgets_and_iterations():
    led = Ledger(Budgets(grams_tpu=0, design_iterations=0))
    assert led.remaining()["grams_tpu"] == 1.0 and led.remaining()["design_iterations"] == 1.0
    assert led.value_per_cost(2.0, CostVector(grams_tpu=5)) == pytest.approx(2.0 / 0.002)
    led2 = Ledger(Budgets(design_iterations=4))
    led2.tick_iteration()
    assert led2.remaining()["design_iterations"] == pytest.approx(0.75)


def test_round_trip_dict():
    led = Ledger(Budgets(print_minutes=500))
    led.charge(CostVector(print_minutes=20, approvals=1, sim_seconds=60))
    led.record(CostVector(print_minutes=20), CostVector(print_minutes=22), 9.0)
    led.tick_iteration()
    back = Ledger.from_dict(led.to_dict())
    assert back.to_dict() == led.to_dict()


def test_thermal_heats_under_load_and_cools_without():
    th = ServoThermal()
    load = np.array([1.5, 1.0, 0.5, 0.0, 0.0, 0.0])
    for _ in range(100):
        th.step(load, 0.1)
    hot = th.heat.copy()
    assert hot[0] > hot[1] > hot[2] > 0 and hot[3] == 0
    assert th.total_j == pytest.approx(float((load**2).sum()) * 10.0)
    for _ in range(100):
        th.step(np.zeros(6), 0.1)
    assert np.all(th.heat[:3] < hot[:3]) and np.all(th.heat[3:] == 0)
    assert th.heat[0] == pytest.approx(hot[0] * np.exp(-10 / 60), rel=1e-6)


def test_thermal_pause_hysteresis():
    th = ServoThermal(n_joints=1)
    paused = [th.step(np.array([3.0]), 1.0) for _ in range(10)]  # 9 J/s
    assert not paused[0] and paused[-1]
    states = []
    while th.heat[0] > 20:
        states.append(th.step(np.zeros(1), 1.0))
    assert states[0] is True  # still paused between 60 J and 30 J
    assert states[-1] is False and not th.paused
