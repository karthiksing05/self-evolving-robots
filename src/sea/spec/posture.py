"""Posture: how boldly the system acts inside the hard limits (spec Sections 8.9 and 9).

One knob, beta, ties the decisions together: ``score = gain + beta * uncertainty - cost``.
``derive`` turns beta into concrete settings by linear interpolation between the
conservative (beta = -1) and optimistic (beta = +1) columns of the Section 9 table, clamped
beyond them.  ``accept`` is the trial acceptance rule and ``anneal`` moves beta toward
conservative as budgets run down.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from sea.contracts import Decision, Posture, Prediction

AREAS: tuple[str, ...] = ("force", "shape", "kinematics", "library")
BETA_MIN = -1.0  # annealing floor (Sec 8.9)
BETA_CLAMP = 2.0
LCB_BELOW = -0.34  # accept rule thresholds (Sec 9 table)
UCB_ABOVE = 0.34
BOOTSTRAP_RESAMPLES = 1000
INTERVAL = 0.90

# Section 9, as (conservative beta=-1, optimistic beta=+1).  Non-numeric rows are documentation
# for the threshold rules in ``derive``.
TABLE: dict[str, tuple[object, object]] = {
    "force_cap_fraction": (0.6, 1.0),  # of the class cap, clamped to the hard limit (D-011)
    "max_rung": (2, 5),  # "only tried templates; rung <= 2 unless forced" vs all rungs
    "max_edits": (1, 3),
    "max_param_delta": (0.3, 1.0),
    "repeats_before_fail": (3, 1),
    "accept_rule": ("lcb", "ucb"),  # mean in between (|beta| < 0.34)
    "target_choice": ("lowest uncertainty first", "highest uncertainty first"),
    "regression_panel": ("full", "sample"),  # sample = 4 controls
    "surprise_tau": (3.0, 5.0),
    "k_promote": (4, 2),
}
_NUMERIC = ("force_cap_fraction", "max_rung", "max_edits", "max_param_delta", "repeats_before_fail",
            "surprise_tau", "k_promote")
_INTEGER = {"max_rung", "max_edits", "repeats_before_fail", "k_promote"}
_OVERRIDABLE = ("max_param_delta", "max_edits", "repeats_before_fail", "accept_rule", "surprise_tau",
                "regression_panel", "force_cap_fraction", "k_promote")

AcceptRule = Literal["lcb", "mean", "ucb"]


class EffectiveParams(BaseModel):
    """Concrete settings for one posture (and optionally one area)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    beta: float
    force_cap_fraction: float
    max_rung: int
    max_edits: int
    max_param_delta: float
    repeats_before_fail: int
    accept_rule: AcceptRule
    target_sign: float  # -1: lowest uncertainty first, +1: highest first
    regression_panel: Literal["full", "sample"]
    regression_sample_n: int = 4
    surprise_tau: float
    k_promote: int


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def accept_rule_for(beta: float) -> AcceptRule:
    if beta <= LCB_BELOW:
        return "lcb"
    if beta >= UCB_ABOVE:
        return "ucb"
    return "mean"


def area_beta(posture: Posture, area: str | None = None) -> float:
    if area is not None and area not in AREAS:
        raise ValueError(f"unknown posture area {area!r}; expected one of {AREAS}")
    beta = posture.per_area.get(area, posture.beta) if area is not None else posture.beta
    return max(-BETA_CLAMP, min(BETA_CLAMP, float(beta)))


def derive(posture: Posture, area: str | None = None) -> EffectiveParams:
    """Beta -> settings.  Explicit (non-None) ``Posture`` fields override derived values."""
    beta = area_beta(posture, area)
    t = (max(-1.0, min(1.0, beta)) + 1.0) / 2.0
    values: dict[str, object] = {}
    for name in _NUMERIC:
        lo, hi = TABLE[name]
        v = float(lo) + t * (float(hi) - float(lo))  # type: ignore[arg-type]
        values[name] = _round_half_up(v) if name in _INTEGER else v
    values["accept_rule"] = accept_rule_for(beta)
    values["regression_panel"] = "full" if beta <= 0 else "sample"
    for name in _OVERRIDABLE:
        explicit = getattr(posture, name)
        if explicit is not None:
            values[name] = explicit
    sign = 0.0 if beta == 0 else math.copysign(1.0, beta)
    return EffectiveParams(beta=beta, target_sign=sign, **values)  # type: ignore[arg-type]


def score(gain: float, uncertainty: float, cost: float, beta: float) -> float:
    """The posture knob: ``gain + beta * uncertainty - cost`` (Sec 8.4, 9)."""
    return gain + beta * uncertainty - cost


def anneal(posture: Posture, beta0: float, remaining: Mapping[str, float]) -> Posture:
    """Move beta from beta0 toward BETA_MIN as the scarcest budget runs down (Sec 8.9).

    Per-area overrides are left fixed (spec Sec 17 default: force conservative, others annealed).
    """
    if not posture.anneal_with_budget:
        return posture
    r = min(remaining.values()) if remaining else 1.0
    r = max(0.0, min(1.0, float(r)))
    beta = beta0 - (beta0 - BETA_MIN) * (1.0 - r)
    return posture.model_copy(update={"beta": beta})


def extra_trial_runs(rule: AcceptRule) -> int:
    """``ucb`` charges one extra trial run (Sec 8.9)."""
    return 1 if rule == "ucb" else 0


def _pooled_deltas(trial: Sequence[Mapping[str, float]], base: Sequence[Mapping[str, float]],
                   cost_delta: float) -> np.ndarray:
    paired = len(base) == len(trial)
    base_mean: dict[str, float] = {}
    if not paired:
        keys = {k for b in base for k in b}
        for k in sorted(keys):
            vals = [b[k] for b in base if k in b]
            base_mean[k] = float(np.mean(vals))
    out: list[float] = []
    for i, tr in enumerate(trial):
        ref = base[i] if paired else base_mean
        for key in sorted(tr):
            if key in ref:
                out.append(float(tr[key]) - float(ref[key]) - cost_delta)
    return np.asarray(out, dtype=float)


def bootstrap_interval(deltas: np.ndarray, seed: int = 0, resamples: int = BOOTSTRAP_RESAMPLES,
                       level: float = INTERVAL) -> tuple[float, float, float]:
    """Mean and a percentile-bootstrap interval (seeded, deterministic)."""
    if deltas.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(resamples, deltas.size))
    means = deltas[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(deltas.mean()), float(lo), float(hi)


def accept(trial: Sequence[Mapping[str, float]], base: Sequence[Mapping[str, float]],
           eff: EffectiveParams, *, cost_delta: float = 0.0, regressions: Iterable[str] = (),
           seed: int = 0, prediction: Prediction | None = None, flipped: Iterable[str] = ()) -> Decision:
    """Accept a trial if its pooled per-control reward delta clears the posture's rule.

    ``trial`` and ``base`` hold one ``{control_key: reward}`` map per randomization repeat
    (1.0 for pass, 0.0 otherwise).  Deltas are paired by repeat when the repeat counts match,
    otherwise compared against the per-key base mean.  ``cost_delta`` (the change in cost terms)
    is subtracted from every delta.  Any regression vetoes acceptance.
    """
    deltas = _pooled_deltas(trial, base, cost_delta)
    mean, lo, hi = bootstrap_interval(deltas, seed)
    rule = eff.accept_rule
    if deltas.size == 0:
        ok = False
    elif rule == "lcb":
        ok = lo > 0
    elif rule == "mean":
        ok = mean > 0
    else:
        ok = hi > 0
    regs = sorted(set(regressions))
    return Decision(
        accepted=bool(ok and not regs),
        rule=rule,
        delta_mean=mean,
        delta_lo=lo,
        delta_hi=hi,
        regressions=regs,
        prediction_score=prediction_score(prediction, flipped),
    )


def prediction_score(prediction: Prediction | None, flipped: Iterable[str]) -> float:
    """Fraction of the predicted flips that actually flipped to pass."""
    if prediction is None or not prediction.flips_to_pass:
        return 0.0
    got = set(flipped)
    hits = sum(1 for k in prediction.flips_to_pass if k in got)
    return hits / len(prediction.flips_to_pass)
