"""Target selection among escalated controls (spec Section 8.4).

``score = gain_est * concept_size + beta * uncertainty - sum_r w_r * cost_est_r``

The loop precomputes each escalation's concept id, concept-level uncertainty and how many
escalated controls share the concept, so this module stays pure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sea.contracts import MAX_TARGETS, ControlResult
from sea.spec.posture import score as posture_score

TargetMode = Literal["gap", "gap_uncertainty"]

# Expected cost of fixing a control with this verdict, as fractions of the *default* budgets
# (240 print min, 300 g PLA, 50 g TPU, $20 LLM, 240 approvals, 40 sim-hours).  Heuristic:
# structural failures (unreachable, no applicable primitive, rule violation) usually end in a
# body change (a ~30 min / ~15 g tool and one approval); stuck and wrong behaviour are usually
# fixed by re-parameterizing or recomposing (LLM call plus sim retries); the rest are sim-only.
_COST_TABLE: dict[str, dict[str, float]] = {
    "unreachable": {"print_minutes": 0.125, "grams_pla": 0.05, "llm_usd": 0.0025, "approvals": 0.004,
                    "sim_seconds": 0.002},
    "no_applicable_primitive": {"print_minutes": 0.06, "grams_pla": 0.025, "llm_usd": 0.003,
                                "approvals": 0.002, "sim_seconds": 0.002},
    "rule_violation": {"print_minutes": 0.06, "grams_pla": 0.025, "llm_usd": 0.003, "approvals": 0.004,
                       "sim_seconds": 0.002},
    "stuck": {"llm_usd": 0.002, "sim_seconds": 0.004},
    "wrong_behavior": {"llm_usd": 0.003, "sim_seconds": 0.003},
    "no_actuation": {"llm_usd": 0.001, "sim_seconds": 0.002},
    "inconclusive": {"sim_seconds": 0.004},
}


def gain_est(verdict: str) -> float:
    if verdict in ("unreachable", "no_applicable_primitive"):
        return 1.0
    if verdict in ("stuck", "wrong_behavior"):
        return 0.7
    return 0.4


def cost_est(verdict: str) -> dict[str, float]:
    return dict(_COST_TABLE.get(verdict, {}))


@dataclass(frozen=True)
class Escalation:
    result: ControlResult
    reason: str
    concept_id: int
    uncertainty: float
    concept_size: int = 1

    @property
    def key(self) -> str:
        return self.result.key


def score(esc: Escalation, beta: float, weights: Mapping[str, float], mode: TargetMode = "gap_uncertainty") -> float:
    gain = gain_est(esc.result.verdict) * max(esc.concept_size, 1)
    unc = esc.uncertainty if mode == "gap_uncertainty" else 0.0
    cost = sum(weights.get(r, 0.0) * c for r, c in cost_est(esc.result.verdict).items())
    return posture_score(gain, unc, cost, beta)


@dataclass
class TargetGroup:
    concept_id: int
    escalations: list[Escalation] = field(default_factory=list)
    score: float = float("-inf")

    @property
    def keys(self) -> list[str]:
        return [e.key for e in self.escalations]

    @property
    def panel_ids(self) -> list[str]:
        return sorted({e.result.panel_id for e in self.escalations})


def pick_targets(escalations: Sequence[Escalation], beta: float, weights: Mapping[str, float],
                 max_targets: int = MAX_TARGETS, mode: TargetMode = "gap_uncertainty") -> list[TargetGroup]:
    """Top ``max_targets`` escalated controls by score, grouped by concept (best group first)."""
    ranked = sorted(((score(e, beta, weights, mode), e) for e in escalations), key=lambda se: (-se[0], se[1].key))
    groups: dict[int, TargetGroup] = {}
    for s, esc in ranked[:max_targets]:
        g = groups.setdefault(esc.concept_id, TargetGroup(concept_id=esc.concept_id))
        g.escalations.append(esc)
        g.score = max(g.score, s)
    return sorted(groups.values(), key=lambda g: (-g.score, g.keys[0]))
