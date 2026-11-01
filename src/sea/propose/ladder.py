"""Escalation-ladder and posture checks on System 2 proposals (spec Sections 3.4 and 9).

Every check returns a list of human-readable errors; an empty list means the proposal may
proceed to costing.  These run before any simulation and before the ledger sees the proposal.
"""

from __future__ import annotations

from typing import Any

from sea.contracts import PROPOSAL_RUNG, Design, Proposal
from sea.spec.posture import EffectiveParams

BODY_KINDS = ("use_tool", "new_tool", "finger", "arm", "actuator")

# Scale used for the relative-change bound when a field's current value is zero or small
# (e.g. wrist_offset_mm and riser_m are 0 on the stock arm).
FIELD_SCALE: dict[str, float] = {
    "upper_scale": 1.0,
    "fore_scale": 1.0,
    "wrist_offset_mm": 20.0,
    "riser_m": 0.1,
    "base_pitch_rad": 0.5,
    "reach_mm": 30.0,
    "pad_angle_deg": 45.0,
    "groove_mm": 6.0,
    "lip_mm": 4.0,
    "pad_tpu_mm": 3.0,
    "width_mm": 24.0,
}


def ladder_errors(proposal: Proposal) -> list[str]:
    """Rung must match the kind; every cheaper rung needs a ``rung<i>: reason`` entry."""
    rung = PROPOSAL_RUNG[proposal.kind]
    if rung is None:  # promote has no rung
        return []
    errs: list[str] = []
    if proposal.rung != rung:
        errs.append(f"kind {proposal.kind!r} is rung {rung}, proposal says rung {proposal.rung}")
    for i in range(rung):
        prefix = f"rung{i}:"
        if not any(e.startswith(prefix) and e[len(prefix):].strip() for e in proposal.ruled_out):
            errs.append(f"missing ruled_out entry '{prefix} <reason>' for a rung-{rung} proposal")
    return errs


def count_edits(proposal: Proposal, design: Design | None = None) -> int:
    """Parameters a proposal edits.  Each added tool counts once (one template instantiation).  Given the
    current ``design``, a field restated at its current value is not an edit (D-039)."""
    n = len(proposal.reparam or {})
    d = proposal.design_delta
    if d is None:
        return n

    def changed(new: dict[str, float], current: Any) -> int:
        if design is None:
            return len(new)
        fields = current if isinstance(current, dict) else current.model_dump()
        same = {k for k, v in new.items() if isinstance(fields.get(k), int | float) and float(fields[k]) == float(v)}
        return len(new) - len(same)

    return (n + changed(d.arm, design and design.arm) + changed(d.fingers, design and design.fingers)
            + changed(d.servo_voltage, design and design.arm.servo_voltage) + len(d.rack_add))


def _relative_change(field: str, old: float, new: float) -> float:
    return abs(new - old) / max(abs(old), FIELD_SCALE.get(field, 1.0))


def posture_errors(proposal: Proposal, eff: EffectiveParams, design: Design, forced: bool = False) -> list[str]:
    errs: list[str] = []
    rung = PROPOSAL_RUNG[proposal.kind]
    if rung is not None and rung > eff.max_rung and not forced:
        errs.append(f"rung {rung} exceeds the posture's max rung {eff.max_rung}")
    n = count_edits(proposal, design)
    if n > eff.max_edits:
        errs.append(f"{n} parameter edits exceed the posture's max_edits {eff.max_edits}")
    d = proposal.design_delta
    if d is not None:
        for group, current in (("arm", design.arm), ("fingers", design.fingers)):
            for name, new in getattr(d, group).items():
                if name not in type(current).model_fields or name == "servo_voltage":
                    errs.append(f"unknown {group} field {name!r}")
                    continue
                old = float(getattr(current, name))
                rel = _relative_change(name, old, float(new))
                if rel > eff.max_param_delta + 1e-12:
                    errs.append(f"{group}.{name}: change {rel:.2f} exceeds max_param_delta {eff.max_param_delta:.2f}")
    return errs


def structure_errors(proposal: Proposal, design: Design, rack_slots: int) -> list[str]:
    errs: list[str] = []
    kind = proposal.kind
    if kind == "program" and proposal.program is None:
        errs.append("program proposal without a program")
    if kind == "reparam" and not proposal.reparam:
        errs.append("reparam proposal without new ranges")
    if kind == "promote" and proposal.promote is None:
        errs.append("promote proposal without a primitive")
    d = proposal.design_delta
    if kind in BODY_KINDS:
        if d is None:
            errs.append(f"{kind} proposal without a design_delta")
        elif kind == "use_tool" and not d.rack_add and proposal.program is None:
            errs.append("use_tool needs a rack_add or a program using a mounted tool")
        elif kind == "new_tool" and not d.rack_add:
            errs.append("new_tool proposal without rack_add")
        elif kind == "finger" and not d.fingers:
            errs.append("finger proposal without finger changes")
        elif kind == "arm" and not d.arm:
            errs.append("arm proposal without arm changes")
        elif kind == "actuator" and not d.servo_voltage:
            errs.append("actuator proposal without servo_voltage changes")
    if d is not None:
        rack = len(design.tool_rack)
        bad = [i for i in d.rack_remove if not 0 <= i < rack]
        if bad:
            errs.append(f"rack_remove indices {bad} not on the rack (size {rack})")
        if len(set(d.rack_remove)) != len(d.rack_remove):
            errs.append("rack_remove lists an index twice")
        after = rack - len(set(d.rack_remove) - set(bad)) + len(d.rack_add)
        if after > rack_slots:
            errs.append(f"rack would hold {after} tools but has {rack_slots} slots; name tools to remove")
    return errs


def validate(proposal: Proposal, design: Design, eff: EffectiveParams, rack_slots: int,
             forced: bool = False) -> list[str]:
    return (ladder_errors(proposal) + posture_errors(proposal, eff, design, forced)
            + structure_errors(proposal, design, rack_slots))

