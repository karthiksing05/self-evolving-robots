"""Spec 12.3: the verifier's own acceptance suite on planted-fault panels.

Synthetic traces (fast, M2): a 1-D guarded actuator against the real control dynamics.
Real-sim traces (slow, M3): the seed library on the compiled MuJoCo model.
"""

from __future__ import annotations

import pytest
from planted import CASES, LIMITS, Planted, sim_trace, synthetic_trace

from sea.contracts import ControlResult
from sea.verify import coverage
from sea.verify.verdicts import judge

SEEDS = range(5)


def _judge(case: Planted, trace) -> ControlResult:
    panel = case.panel()
    return judge(trace, panel.controls[0], panel.rules, LIMITS)


def test_ten_faults_cover_the_spec_list():
    names = {c.name for c in CASES}
    assert len(CASES) == 10
    assert {"stiff_button", "no_detent", "latch_above_cap", "deep_recess", "tab_door_no_pinch", "knob_over_torque",
            "reversed_slider"} <= names
    assert len({c.expected for c in CASES}) >= 5  # every failure verdict judge() can give for a complete trace


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_synthetic_fault_gets_declared_verdict(case: Planted):
    assert _judge(case, synthetic_trace(case)).verdict == case.expected


def test_synthetic_twins_pass_over_seeds():
    results = [_judge(c.twin(), synthetic_trace(c.twin(), seed)) for c in CASES for seed in SEEDS]
    assert coverage.passes(results) / len(results) >= 0.95


def test_synthetic_coverage_matches_hand_counts():
    results = [_judge(c, synthetic_trace(c)) for c in CASES] + [_judge(c.twin(), synthetic_trace(c.twin())) for c in CASES]
    # Hand count: 10 twins pass; no fault is no_applicable_primitive or inconclusive, so all 20 are covered.
    assert coverage.covered(results) == 20
    assert coverage.control_coverage(results) == 1.0
    assert coverage.pass_rate(results) == 10 / 20
    assert coverage.coverage_reward(results) == 0.5


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_sim_fault_gets_declared_verdict(case: Planted):
    if case.sim_xfail:
        pytest.xfail(case.sim_xfail)
    assert _judge(case, sim_trace(case)).verdict == case.expected


@pytest.mark.slow
def test_sim_twins_pass_over_seeds():
    cases = [c for c in CASES if not c.sim_xfail]
    results = [_judge(c.twin(), sim_trace(c.twin(), seed)) for c in cases for seed in SEEDS]
    failing = [(r.panel_id, r.verdict) for r in results if r.verdict != "pass"]
    assert coverage.passes(results) / len(results) >= 0.95, failing
    summary = coverage.summarize(results)
    assert summary["n"] == len(cases) * len(SEEDS)
    assert summary["control_coverage"] == coverage.covered(results) / len(results)
