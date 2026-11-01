"""Section 8.2 verdicts on synthetic traces: one trace per verdict, plus the precedence order.

The executor-side ``no_applicable_primitive`` (emitted before simulation when no program
type-checks) is produced by the suite runner through ``verdicts.no_applicable``; that runner is
tested with the loop.  Here we check the result it builds.
"""

from __future__ import annotations

import pytest

from sea.contracts import ControlSpec, HardLimits, Pose, Rules, StepTrace, Vec3
from sea.verify.verdicts import judge, no_applicable, verdict_for

LIMITS = HardLimits(workspace_box=(Vec3(x=0.1, y=-0.2, z=-0.1), Vec3(x=0.4, y=0.2, z=0.25)))
RULES = Rules(force_cap_by_class={"push_button": 10.0, "door": 15.0})
BUTTON = ControlSpec(id="c0", cls="push_button", pose_panel=Pose(pos=Vec3(x=0, y=0, z=0)), size_mm=(12.0,),
                     stiffness=800.0, travel=0.003, friction=0.05, success="moved_at_least($target, travel)")
N = 50


def trace(*, q_frac: float = 0.0, tip: float = 0.0, ctl: float = 0.0, **kw) -> StepTrace:
    """A complete 1 s trace whose control ends at ``q_frac`` of travel with peak forces ``tip``/``ctl``."""
    base = dict(
        panel_id="p0", control_id="c0", program_id="press_button", attempt=0,
        t=[0.02 * (i + 1) for i in range(N)],
        tip_force=[tip * i / (N - 1) for i in range(N)],
        control_q=[q_frac * BUTTON.travel * i / (N - 1) for i in range(N)],
        push_phase=[True] * N, in_contact=[True] * N, expected_samples=N, force_cap=8.0,
        peak_control_force=ctl, sim_seconds=1.0, ik_residual_mm=0.7,
    )
    base.update(kw)
    return StepTrace(**base)


def v(tr: StepTrace, rules: Rules = RULES, limits: HardLimits = LIMITS) -> str:
    return verdict_for(tr, BUTTON, rules, limits)


# ---------------------------------------------------------------- one trace per verdict
def test_pass():
    assert v(trace(q_frac=1.1, tip=3, ctl=3, success_time_s=0.8)) == "pass"


def test_pass_needs_the_time_limit():
    assert v(trace(q_frac=1.1, tip=3, ctl=3, success_time_s=25.0)) != "pass"


def test_stuck_at_the_guard_cap():
    assert v(trace(q_frac=0.2, tip=7.6, ctl=7.6)) == "stuck"


def test_stuck_falls_back_to_the_class_cap_without_a_guard_cap():
    assert v(trace(q_frac=0.1, tip=9.6, ctl=9.6, force_cap=0.0)) == "stuck"


def test_no_actuation():
    assert v(trace(q_frac=0.0, tip=0.1, ctl=0.1)) == "no_actuation"


def test_wrong_behavior_no_detent_or_latch():
    assert v(trace(q_frac=1.2, tip=3, ctl=3)) == "wrong_behavior"


def test_wrong_behavior_wrong_direction():
    assert v(trace(q_frac=-1.0, tip=3, ctl=3)) == "wrong_behavior"


def test_unreachable_by_residual():
    assert v(trace(ik_residual_mm=6.0)) == "unreachable"


def test_unreachable_by_workspace_refusal():
    assert v(trace(note="workspace", ik_residual_mm=0.0)) == "unreachable"


def test_workspace_abort_after_contact_is_judged_on_the_evidence():
    assert v(trace(note="workspace", q_frac=1.2, tip=3, ctl=3)) == "wrong_behavior"


@pytest.mark.parametrize("kw", [
    dict(contact_types=["pinch"]),
    dict(contacts=[(0.5, "pinch", "c0")]),
    dict(tip=10.5, ctl=3.0),
    dict(tip=3.0, ctl=10.5),
], ids=["banned_type", "banned_contact_log", "tip_over_class_cap", "control_over_class_cap"])
def test_rule_violation(kw):
    assert v(trace(q_frac=0.5, **kw)) == "rule_violation"


def test_rule_violation_from_hard_limit_bans():
    rules = RULES.model_copy(update={"banned_contacts": []})
    assert v(trace(contact_types=["pinch"]), rules) == "rule_violation"
    assert v(trace(contact_types=["pinch"]), rules, LIMITS.model_copy(update={"banned_contacts": []})) != "rule_violation"


def test_banned_contact_on_another_control_is_ignored():
    assert v(trace(q_frac=1.1, tip=3, ctl=3, success_time_s=0.8, contacts=[(0.5, "pinch", "c1")])) == "pass"


def test_inconclusive_when_aborted():
    assert v(trace(aborted=True, success_time_s=0.5)) == "inconclusive"


def test_inconclusive_with_too_few_samples():
    assert v(trace(expected_samples=2 * N + 1)) == "inconclusive"


def test_inconclusive_when_contact_is_lost():
    lost = [i % 3 == 0 for i in range(N)]  # in contact a third of the push phase
    assert v(trace(q_frac=0.1, tip=2.0, ctl=2.0, in_contact=lost)) == "inconclusive"


def test_contact_loss_outside_the_push_phase_does_not_count():
    lost = [i % 3 == 0 for i in range(N)]
    assert v(trace(q_frac=0.1, tip=2.0, ctl=2.0, in_contact=lost, push_phase=[False] * N)) == "stuck"


def test_fallback_stuck_when_pushed_but_unmoved():
    assert v(trace(q_frac=0.1, tip=2.0, ctl=2.0)) == "stuck"


def test_fallback_wrong_behavior_when_pushed_backwards():
    assert v(trace(q_frac=-0.1, tip=2.0, ctl=2.0)) == "wrong_behavior"


# ---------------------------------------------------------------- precedence (Section 8.2 order)
@pytest.mark.parametrize("kw,expected", [
    (dict(aborted=True, contact_types=["pinch"], ik_residual_mm=9.0), "inconclusive"),
    (dict(contact_types=["pinch"], ik_residual_mm=9.0, success_time_s=0.5, q_frac=1.1), "rule_violation"),
    (dict(ik_residual_mm=9.0, success_time_s=0.5, q_frac=1.1, tip=3, ctl=3), "unreachable"),
    (dict(success_time_s=0.5, q_frac=1.1, tip=7.9, ctl=7.9), "pass"),
    (dict(q_frac=0.0, tip=7.9, ctl=0.0), "stuck"),  # cap reached on the rim: stuck before no_actuation
    (dict(q_frac=0.0, tip=0.1, ctl=0.1, in_contact=[False] * N), "no_actuation"),
    (dict(q_frac=1.2, tip=3, ctl=3, in_contact=[False] * N), "wrong_behavior"),
])
def test_precedence(kw, expected):
    assert v(trace(**kw)) == expected


# ---------------------------------------------------------------- measurements
def test_judge_fills_the_measurement_vector():
    r = judge(trace(q_frac=1.1, tip=4.0, ctl=3.5, success_time_s=0.8, detents_crossed=1, s1_trials=3), BUTTON,
              RULES, LIMITS)
    assert (r.verdict, r.cls, r.panel_id, r.control_id, r.program_id) == ("pass", "push_button", "p0", "c0",
                                                                           "press_button")
    assert r.peak_force == 3.5
    assert r.travel_reached == pytest.approx(1.1)
    assert (r.detents, r.time_s, r.ik_residual_mm, r.s1_trials) == (1, 0.8, 0.7, 3)


def test_judge_time_and_travel_fallbacks():
    tr = trace(q_frac=-0.5, tip=2, ctl=2, sim_seconds=0.0)
    r = judge(tr, BUTTON, RULES, LIMITS)
    assert r.travel_reached == pytest.approx(0.5)  # |q - q_rest|
    assert r.time_s == tr.t[-1]
    empty = StepTrace(panel_id="p0", control_id="c0", program_id="x", attempt=0)
    r = judge(empty, BUTTON, RULES, LIMITS)
    assert (r.verdict, r.time_s, r.travel_reached) == ("no_actuation", 0.0, 0.0)


def test_unknown_class_cap_uses_the_hard_limit():
    rules = Rules(force_cap_by_class={})
    assert v(trace(q_frac=0.5, tip=12.0, ctl=3.0), rules) != "rule_violation"
    assert v(trace(q_frac=0.5, tip=15.5, ctl=3.0), rules) == "rule_violation"


def test_no_applicable_result():
    r = no_applicable("p3", BUTTON)
    assert (r.verdict, r.panel_id, r.control_id, r.cls, r.peak_force) == ("no_applicable_primitive", "p3", "c0",
                                                                          "push_button", 0.0)
    assert r.key == "p3:c0"
