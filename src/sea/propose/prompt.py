"""System 2 prompt construction (spec Section 8.6).

``build_prompt`` renders the eight sections in the order the spec lists them.  Everything is
formatted deterministically (sorted keys, fixed float precision) so the prompt hash, which keys
the response cache and the ``MockLLM`` fixtures, is stable across runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from sea.contracts import Budgets, Design, FixRecord, HardLimits, PrimitiveDef
from sea.sim.sensors import manifest
from sea.spec.posture import EffectiveParams

SECTION_TITLES: tuple[str, ...] = (
    "Hard limits",
    "Posture and remaining budgets",
    "Escalation ladder",
    "Primitive library",
    "Current design",
    "Sensors on this body",
    "Failing controls",
    "Similar past fixes",
    "Output schema and rules",
)

SYSTEM = (
    "You are System 2 of a self-evolving robot arm (a simulated SO-101). System 1 runs scripted "
    "motor primitives with guarded contact and gave up on the controls listed below. Diagnose why "
    "each failed and propose fixes, preferring the cheapest rung of the escalation ladder that can "
    "plausibly work. You cannot change the hard limits, the verifier, or the reward. Every proposal "
    "is type-checked, priced, simulated, and kept only if the verifiable reward rises with no "
    "regressions. Ground every claim in the evidence fields you are given and cite them by name."
)

LADDER_TEXT = """Consider fixes in this order and move down only when the cheaper rung cannot plausibly fix the failure.
| Rung | kind | Change | Typical cost |
| 0 | reparam | Re-parameterize: hand System 1 new parameter ranges | sim seconds |
| 1 | program | Recompose existing primitives into a new program | sim seconds + 1 LLM call |
| 2 | use_tool | Use or adapt an existing tool from the library | print minutes only if adapted |
| 3 | new_tool | New tool from a template (L0) | print minutes, grams |
| 4 | finger | Finger change (L1) | more print minutes |
| 5 | arm, actuator | Kinematic (L2) or actuator (L3) change | most expensive; full regression |
`promote` (turn a passing composed program into a named primitive) has no rung.
Rule: a proposal at rung k must list, for every i < k, a `ruled_out` entry that starts with `rung<i>:` followed by a non-empty reason, e.g. "rung0: stuck at the 9.5 N cap with travel 0.4 mm, so wider ranges cannot help". Proposals missing an entry are rejected before costing."""

GRAMMAR_TEXT = """Programs are trees of primitives (Appendix A grammar, text form shown for reference):
  node   := prim | Seq(node, node, ...) | Guard(node, until=check, cap=check) | Repeat(node, n=INT | until=check)
          | Branch(check, if_true, if_false) | With(rack_index, node) | Bind(name, {param: number, ...})
  prim   := name(args)   args are numbers (SI: m, N, rad, s), $target, or name=value
  check  := atoms combined with & | ~ and parentheses; atoms: moved_at_least(ref, d|travel), latched(ref),
            detent_crossed(ref, n), force_below(N), no_banned_contact()   ref := $target | control id
Give programs as `Program` JSON: {"id", "target_cls", "source": "s2", "root": Node}, where a Node is
{"op": "prim"|"seq"|"guard"|"repeat"|"branch"|"with"|"bind", "prim": {"name", "args"}, "children": [...],
 "check": until/branch check, "cap": guard cap, "n": repeat count, "tool_index": rack slot, "bind_args": {...}}.
Positional primitive arguments use the keys "_0", "_1", ...; `$target` is the control the program runs on."""


class TraceSummary(BaseModel):
    """What System 2 sees about one failing control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    cls: str
    verdict: str
    peak_force: float
    travel_reached: float
    travel_required: float
    contact_types: list[str] = []
    ik_residual_mm: float = 0.0
    joint_limit_hits: list[str] = []
    surprise_z: dict[str, float] = {}
    escalation_reason: str | None = None
    program_id: str = ""
    handle: str = "none"
    recess_mm: float = 0.0
    tilt_deg: float = 0.0


def request_hash(payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a request payload."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class S2Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    system: str
    sections: list[tuple[str, str]]
    target_keys: list[str] = []
    max_proposals: int = 3

    def render(self) -> str:
        return "\n\n".join(f"## {i}. {title}\n{body}" for i, (title, body) in enumerate(self.sections, 1))

    def hash(self) -> str:
        return request_hash({
            "model": self.model,
            "system": self.system,
            "sections": [list(s) for s in self.sections],
            "target_keys": list(self.target_keys),
            "max_proposals": self.max_proposals,
        })


# ---------------------------------------------------------------- formatting helpers
def _num(x: float) -> str:
    return f"{float(x):.4g}"


def _kv(d: Mapping[str, Any]) -> str:
    parts = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            parts.append(f"{k}=[{_num(v[0])}, {_num(v[1])}]")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={_num(v)}")
    return ", ".join(parts) if parts else "-"


def _limits_section(limits: HardLimits) -> str:
    lo, hi = limits.workspace_box
    return "\n".join([
        "These are fixed facts. No proposal can change them; parameters outside them are rejected by the typechecker.",
        f"- max tip force: {_num(limits.max_force_n)} N",
        f"- max tip speed: {_num(limits.max_tip_speed_mps)} m/s",
        f"- workspace box (base frame, m): x {_num(lo.x)}..{_num(hi.x)}, y {_num(lo.y)}..{_num(hi.y)}, "
        f"z {_num(lo.z)}..{_num(hi.z)}",
        f"- banned contact types: {', '.join(limits.banned_contacts) or 'none'}",
        f"- max tool mass: {_num(limits.max_tool_mass_g)} g; max link scale: {_num(limits.max_link_scale)}; "
        f"min wall: {_num(limits.min_wall_mm)} mm",
    ])


def _posture_section(eff: EffectiveParams, remaining: Mapping[str, float], budgets: Budgets) -> str:
    rem = ", ".join(f"{k} {100 * remaining[k]:.0f}%" for k in sorted(remaining))
    return "\n".join([
        f"- beta {_num(eff.beta)}: accept rule {eff.accept_rule}, max rung {eff.max_rung} (unless forced), "
        f"max edits {eff.max_edits}, max relative parameter change {_num(eff.max_param_delta)}",
        f"- force cap fraction {_num(eff.force_cap_fraction)} of each class cap; failures confirmed after "
        f"{eff.repeats_before_fail} repeats; promotion after {eff.k_promote} distinct controls",
        f"- remaining budget fractions: {rem or '-'}",
        f"- budgets: print {_num(budgets.print_minutes)} min, PLA {_num(budgets.grams_pla)} g, TPU "
        f"{_num(budgets.grams_tpu)} g, LLM ${_num(budgets.llm_usd)}, rack slots {budgets.rack_slots}, "
        f"library size {budgets.library_size}, reserve {100 * budgets.reserve_fraction:.0f}%",
    ])


def _library_section(library: Sequence[PrimitiveDef]) -> str:
    lines = []
    for p in sorted(library, key=lambda p: (p.kind, p.name)):
        lines.append(f"- {p.name} [{p.kind}] {p.doc}")
        lines.append(f"    requires: {_kv(p.requires)}; provides: {_kv(p.provides)}; params: {_kv(p.params)}")
    return "\n".join(lines) if lines else "(empty)"


def _design_section(design: Design) -> str:
    a, f = design.arm, design.fingers
    lines = [
        f"- id {design.id} (parent {design.parent or '-'})",
        f"- arm: upper_scale {_num(a.upper_scale)}, fore_scale {_num(a.fore_scale)}, wrist_offset_mm "
        f"{_num(a.wrist_offset_mm)}, riser_m {_num(a.riser_m)}, base_pitch_rad {_num(a.base_pitch_rad)}",
        f"- servo voltage: {_kv(a.servo_voltage)}",
        f"- fingers: jaw_len_mm {_num(f.jaw_len_mm)}, tip_radius_mm {_num(f.tip_radius_mm)}, softness {_num(f.softness)}",
    ]
    if design.tool_rack:
        for i, t in enumerate(design.tool_rack):
            lines.append(f"- rack[{i}]: {t.template} contact={t.contact} {_kv(t.params)}")
    else:
        lines.append("- rack: empty (stock gripper only: contacts edge, or pinch when both jaws touch)")
    e = design.est
    lines.append(f"- build estimate: {_num(e.print_minutes)} print min, {_num(e.grams_pla)} g PLA, "
                 f"{_num(e.grams_tpu)} g TPU, added mass {_num(e.added_mass_g)} g")
    return "\n".join(lines)


def _sensors_section(design: Design) -> str:
    """The sensor manifest (D-033): what this body can feel, so plans use only what exists."""
    lines = []
    for part in manifest(design):
        tags = [part["priority"], "toggleable" if part["toggle"] else "fixed", "bench" if part["bench"] else "sim only"]
        state = "mounted" if part["mounted"] else "NOT mounted"
        lines.append(f"- {part['part']} [{state}; {', '.join(tags)}; {_num(part['mass_g'])} g; {part['mount']}]: "
                     f"{part['tells']}")
        lines.append("    channels: " + ", ".join(
            f"{c} ({v['unit']}, resolution {_num(v['resolution'])}, noise {_num(v['noise'])})"
            for c, v in part["channels"].items()))
    lines.append("- derived from servo telemetry and the arm's own model: torque, payload_g (valid only while the "
                 "arm is still), kinetic energy, momentum")
    lines.append("Do not plan on a channel whose part is not mounted. Sensors inform your choices; pass and fail "
                 "are judged from ground truth, never from sensor readings.")
    return "\n".join(lines)


def _failing_section(failing: Sequence[TraceSummary]) -> str:
    if not failing:
        return "(none)"
    lines = []
    for s in sorted(failing, key=lambda s: s.key):
        lines.append(
            f"- {s.key} cls={s.cls} handle={s.handle} recess_mm={_num(s.recess_mm)} tilt_deg={_num(s.tilt_deg)} "
            f"verdict={s.verdict} escalation_reason={s.escalation_reason or '-'} program_id={s.program_id or '-'}"
        )
        lines.append(
            f"    peak_force={_num(s.peak_force)} N travel_reached={_num(s.travel_reached)} of "
            f"travel_required={_num(s.travel_required)} contact_types={','.join(s.contact_types) or '-'} "
            f"ik_residual_mm={_num(s.ik_residual_mm)} joint_limit_hits={','.join(s.joint_limit_hits) or '-'} "
            f"surprise_z={_kv(s.surprise_z)}"
        )
    return "\n".join(lines)


def _fixes_section(fixes: Sequence[FixRecord], k: int = 5) -> str:
    if not fixes:
        return "(no similar past fixes yet)"
    lines = []
    for fx in sorted(fixes, key=lambda f: (f.distance, f.id))[:k]:
        p, d = fx.proposal, fx.decision
        outcome = "accepted" if d.accepted else "rejected"
        lines.append(
            f"- {fx.id} (iter {fx.iteration}, distance {_num(fx.distance)}): {p.kind} rung {p.rung} -> {outcome} "
            f"by {d.rule} (delta mean {_num(d.delta_mean)}, interval [{_num(d.delta_lo)}, {_num(d.delta_hi)}], "
            f"regressions {len(d.regressions)}, prediction score {_num(d.prediction_score)})"
        )
        lines.append(f"    hypothesis: {p.hypothesis}")
    return "\n".join(lines)


def _rules_section(max_proposals: int) -> str:
    return "\n".join([
        f"Call the `propose` tool exactly once with 1 to {max_proposals} proposals (a `ProposalBatch`).",
        "Each proposal must:",
        "- set `kind` and the matching `rung`, and give `ruled_out` entries for every cheaper rung (see ladder);",
        "- give a `hypothesis` (the root cause) and a `rationale` that cites evidence by field name, "
        "e.g. peak_force, travel_reached, ik_residual_mm, contact_types, joint_limit_hits, surprise_z;",
        "- include a `prediction`: control keys expected to flip to pass, keys that may regress, and a confidence in [0, 1];",
        "- carry exactly the payload its kind needs: `program` (program), `reparam` (reparam), `design_delta` "
        "(use_tool, new_tool, finger, arm, actuator), `promote` (promote).",
        "If the rack is full, `design_delta.rack_remove` must name the rack indices to free.",
        "`reparam` keys are `primitive.param` (e.g. `push.force`); a bare `param` applies to whichever "
        "primitive System 1 is searching.  New ranges must stay inside the hard limits.",
        GRAMMAR_TEXT,
    ])


def build_prompt(*, limits: HardLimits, eff: EffectiveParams, remaining: Mapping[str, float], budgets: Budgets,
                 library: Sequence[PrimitiveDef], design: Design, failing: Sequence[TraceSummary],
                 fixes: Sequence[FixRecord], model: str, max_proposals: int = 3) -> S2Prompt:
    bodies = (
        _limits_section(limits),
        _posture_section(eff, remaining, budgets),
        LADDER_TEXT,
        _library_section(library),
        _design_section(design),
        _sensors_section(design),
        _failing_section(failing),
        _fixes_section(fixes),
        _rules_section(max_proposals),
    )
    return S2Prompt(
        model=model,
        system=SYSTEM,
        sections=list(zip(SECTION_TITLES, bodies, strict=True)),
        target_keys=sorted(s.key for s in failing),
        max_proposals=max_proposals,
    )
