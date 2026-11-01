"""Coverage metrics and the reward (spec 8.2, 8.3).

::

    control_coverage = (# controls with verdict not in {no_applicable_primitive, inconclusive}) / N
    pass_rate        = passes / covered            (0 when covered = 0)
    coverage_reward  = control_coverage * pass_rate  (= passes / N)
    reward           = coverage_reward - l_print * cum_print_minutes / budget_print_minutes
                                       - l_mass * added_mass_g / max_tool_mass_g

``verify/reward.py`` recomputes the reward from run logs with these same functions, so the
in-loop value and the audit agree to the bit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import get_args

from sea.contracts import LAMBDA_MASS, LAMBDA_PRINT, ControlResult, Verdict

UNCOVERED: frozenset[str] = frozenset({"no_applicable_primitive", "inconclusive"})
VERDICTS: tuple[str, ...] = get_args(Verdict)


def covered(results: Sequence[ControlResult]) -> int:
    return sum(r.verdict not in UNCOVERED for r in results)


def passes(results: Sequence[ControlResult]) -> int:
    return sum(r.verdict == "pass" for r in results)


def control_coverage(results: Sequence[ControlResult]) -> float:
    return covered(results) / len(results) if results else 0.0


def pass_rate(results: Sequence[ControlResult]) -> float:
    c = covered(results)
    return passes(results) / c if c else 0.0


def coverage_reward(results: Sequence[ControlResult]) -> float:
    """``control_coverage * pass_rate``, computed as passes / N (same value, no float drift)."""
    return passes(results) / len(results) if results else 0.0


def reward(cov_reward: float, cum_print_minutes: float, budget_print_minutes: float, added_mass_g: float,
           max_tool_mass_g: float, cost_terms: bool = True) -> float:
    """Section 8.3.  ``cost_terms=False`` is the E7 ablation.  A zero budget drops its term."""
    if not cost_terms:
        return cov_reward
    r = cov_reward
    if budget_print_minutes > 0:
        r -= LAMBDA_PRINT * cum_print_minutes / budget_print_minutes
    if max_tool_mass_g > 0:
        r -= LAMBDA_MASS * added_mass_g / max_tool_mass_g
    return r


def summarize(results: Sequence[ControlResult]) -> dict:
    """Coverage numbers plus verdict counts overall and per control class."""
    by_class: dict[str, dict] = {}
    for r in results:
        entry = by_class.setdefault(r.cls, {"n": 0, "passes": 0, "by_verdict": Counter()})
        entry["n"] += 1
        entry["passes"] += r.verdict == "pass"
        entry["by_verdict"][r.verdict] += 1
    counts = Counter(r.verdict for r in results)
    return {
        "n": len(results),
        "covered": covered(results),
        "passes": passes(results),
        "control_coverage": control_coverage(results),
        "pass_rate": pass_rate(results),
        "coverage_reward": coverage_reward(results),
        "by_verdict": {v: counts.get(v, 0) for v in VERDICTS},
        "by_class": {c: {**e, "by_verdict": dict(e["by_verdict"])} for c, e in sorted(by_class.items())},
    }
