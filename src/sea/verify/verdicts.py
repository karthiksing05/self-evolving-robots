"""Verdicts (spec 8.2): turn one ``StepTrace`` into a ``ControlResult``.

``judge`` applies the Section 8.2 precedence; the first rule that holds wins:

0. ``inconclusive``    the trace is incomplete (sim failure, or under 50% of the expected samples)
1. ``rule_violation``  a banned contact touched the control, or force exceeded the class cap
2. ``unreachable``     IK residual over 5 mm at the contact waypoint (or a workspace refusal
                       before the control was touched)
3. ``pass``            the success check held within the time limit
4. ``stuck``           tip force reached the guard cap with under 30% of the travel
5. ``no_actuation``    peak force on the control under 0.2 N
6. ``wrong_behavior``  the control moved (>= 30% of travel, either sign) but success failed
7. ``inconclusive``    contact was lost for more than 50% of the push phase
   fallback            ``wrong_behavior`` if it moved backwards, else ``stuck`` (docs/decisions.md V-002)

``no_applicable_primitive`` is never produced here: the suite runner emits it (via
``no_applicable``) before simulation when no library program type-checks for the control.
"""

from __future__ import annotations

from sea.contracts import ControlResult, ControlSpec, HardLimits, Rules, StepTrace, Verdict

MIN_SAMPLE_FRACTION = 0.5
UNREACHABLE_MM = 5.0
CAP_TOLERANCE_N = 0.5  # guarded moves stop within 0.5 N of the cap (test_motor)
STUCK_TRAVEL = 0.3
NO_ACTUATION_N = 0.2
MOVED_TRAVEL = 0.3  # "moved" mirrors the stuck threshold, so rules 4 and 6 partition force-at-cap
LOST_CONTACT_FRACTION = 0.5
BACKWARD_TRAVEL = 0.05  # fallback: a control pushed backwards by more than this behaves wrongly


def _travel_fraction(trace: StepTrace, control: ControlSpec) -> tuple[float, float]:
    """(furthest forward, furthest backward) displacement from rest, as fractions of the travel."""
    travel = max(abs(control.travel), 1e-9)
    moved = [q - trace.q_rest for q in trace.control_q] or [0.0]
    return max(0.0, max(moved)) / travel, max(0.0, -min(moved)) / travel


def _peak_tip_force(trace: StepTrace) -> float:
    return max(trace.tip_force, default=0.0)


def _contact_types(trace: StepTrace, control: ControlSpec) -> set[str]:
    types = set(trace.contact_types)
    types.update(c[1] for c in trace.contacts if c[2] == control.id)
    return types


def _incomplete(trace: StepTrace) -> bool:
    return trace.aborted or len(trace.t) < MIN_SAMPLE_FRACTION * trace.expected_samples


def _contact_lost(trace: StepTrace) -> bool:
    pushing = [c for p, c in zip(trace.push_phase, trace.in_contact, strict=False) if p]
    return bool(pushing) and sum(not c for c in pushing) > LOST_CONTACT_FRACTION * len(pushing)


def verdict_for(trace: StepTrace, control: ControlSpec, rules: Rules, limits: HardLimits) -> Verdict:
    """The Section 8.2 verdict for one trace (see the module docstring for the order)."""
    if _incomplete(trace):
        return "inconclusive"
    banned = set(rules.banned_contacts) | set(limits.banned_contacts)
    class_cap = rules.force_cap_by_class.get(control.cls, limits.max_force_n)
    peak_ctl = trace.peak_control_force
    if _contact_types(trace, control) & banned or max(_peak_tip_force(trace), peak_ctl) > class_cap:
        return "rule_violation"
    refused = trace.note == "workspace" and peak_ctl < NO_ACTUATION_N
    if trace.ik_residual_mm > UNREACHABLE_MM or refused:
        return "unreachable"
    if trace.success_time_s is not None and trace.success_time_s <= rules.time_limit_s:
        return "pass"
    fwd, back = _travel_fraction(trace, control)
    frac = max(fwd, back)
    cap = trace.force_cap if trace.force_cap > 0 else class_cap
    if _peak_tip_force(trace) >= cap - CAP_TOLERANCE_N and frac < STUCK_TRAVEL:
        return "stuck"
    if peak_ctl < NO_ACTUATION_N:
        return "no_actuation"
    if frac >= MOVED_TRAVEL:
        return "wrong_behavior"
    if _contact_lost(trace):
        return "inconclusive"
    return "wrong_behavior" if back > max(fwd, BACKWARD_TRAVEL) else "stuck"


def judge(trace: StepTrace, control: ControlSpec, rules: Rules, limits: HardLimits) -> ControlResult:
    """Verdict plus the measurement vector the concept memory learns from."""
    verdict = verdict_for(trace, control, rules, limits)
    frac = max(_travel_fraction(trace, control))
    if verdict == "pass" and trace.success_time_s is not None:
        time_s = trace.success_time_s
    else:
        time_s = trace.sim_seconds or (trace.t[-1] if trace.t else 0.0)
    return ControlResult(
        panel_id=trace.panel_id, control_id=control.id, cls=control.cls, verdict=verdict,
        peak_force=round(trace.peak_control_force, 4), travel_reached=round(frac, 6),
        detents=trace.detents_crossed, time_s=round(float(time_s), 4),
        ik_residual_mm=round(trace.ik_residual_mm, 4), program_id=trace.program_id, s1_trials=trace.s1_trials,
    )


def no_applicable(panel_id: str, control: ControlSpec) -> ControlResult:
    """Result for a control that no library program type-checks for (emitted before simulation)."""
    return ControlResult(panel_id=panel_id, control_id=control.id, cls=control.cls,
                         verdict="no_applicable_primitive", peak_force=0.0, travel_reached=0.0, detents=0, time_s=0.0)
