"""System 1 for any world: local search over a task's control parameters, then confirmation.

``local_search(run, space, start)`` tries the world's default controls, then samples around the best
score so far with a shrinking step, and stops at the first success.  A success counts only if it repeats
on ``confirm`` fresh seeds (spec 8.5): a lucky draw is not a skill.  ``score`` guides the search;
``success`` decides.  A world may keep its own System 1 instead.

A failure is not yet the body's (D-102, D-108).  Before it counts, System 1 searches other ways of using the same
body: those System 2's plan names, then starts spread over the control space.  Each draws its own samples, because
ways that share them fail together (D-105).  The first way runs exactly as if it were the only one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from sea.evolve.core import Body, Outcome

Run = Callable[[dict[str, float], int], Outcome]
WAYS = 3  # starts spread over the control space, after the first and any the plan names
OTHER_SEED = 7919


def planned(body: Body, space: dict[str, tuple[float, float]], start: dict[str, float], m: int) -> tuple[dict[str, float], int]:
    """The search's start and attempts for ``body``: its operating plan's values over ``start``, and its allowance (D-092)."""
    if body.plan is None:
        return start, m
    return {**start, **{k: v for k, v in body.plan.controls.items() if k in space}}, m * body.plan.search


def ways_of(body: Body, space: dict[str, tuple[float, float]]) -> list[dict[str, float]]:
    """The other ways System 2's plan names for ``body``, in terms of the controls this search has."""
    if body.plan is None:
        return []
    return [w for w in ({k: v for k, v in way.items() if k in space} for way in body.plan.ways) if w]


def spread(space: dict[str, tuple[float, float]], n: int, seed: int) -> list[dict[str, float]]:
    """``n`` starts apart from each other: one per stratum of each control's range, in a shuffled order per control."""
    rng = np.random.default_rng([seed, OTHER_SEED])
    orders = {k: rng.permutation(n) for k in sorted(space)}
    return [{k: space[k][0] + (float(orders[k][i]) + 0.5) / n * (space[k][1] - space[k][0]) for k in orders} for i in range(n)]


def local_search(run: Run, space: dict[str, tuple[float, float]], start: dict[str, float], m: int = 16,
                 confirm: int = 2, seed: int = 0, step: float = 0.3, shrink: float = 0.85,
                 ways: Sequence[dict[str, float]] = (), other: int | None = None) -> Outcome:
    """The search from ``start``; if it fails, from ``ways`` and ``other`` starts (``WAYS`` if None) spread over the
    space, each with half the attempts, until one passes.  The best outcome, saying how many ways it took."""
    first = _search(run, space, start, m, confirm, seed, step, shrink)
    starts = [{**start, **w} for w in ways] + spread(space, WAYS if other is None else other, seed)
    if first.success or not space or not starts:
        return first
    best, attempts, sim_s, tried = first, first.attempts, first.sim_s, 1
    for k, s in enumerate(starts, 1):
        o = _search(run, space, s, max(4, m // 2), confirm, seed + OTHER_SEED * k, step, shrink)
        attempts, sim_s, tried = attempts + o.attempts, sim_s + o.sim_s, tried + 1
        if o.success or o.score > best.score:
            best = o
        if o.success:
            break
    return best.model_copy(update={"attempts": attempts, "sim_s": round(sim_s, 3), "ways": tried, "nominal": first.nominal})


def _search(run: Run, space: dict[str, tuple[float, float]], start: dict[str, float], m: int, confirm: int, seed: int,
            step: float, shrink: float) -> Outcome:
    rng = np.random.default_rng(seed)
    keys = sorted(space)
    lo = np.array([space[k][0] for k in keys])
    hi = np.array([space[k][1] for k in keys])
    x = np.clip(np.array([start.get(k, (space[k][0] + space[k][1]) / 2) for k in keys]), lo, hi)
    best_x, best, attempts, sim_s = x, None, 0, 0.0
    sigma = step
    nominal = None
    for trial in range(m):
        cand = x if trial == 0 else np.clip(best_x + rng.normal(0.0, sigma, len(keys)) * (hi - lo), lo, hi)
        ctl = {k: float(v) for k, v in zip(keys, cand, strict=True)}
        out = run(ctl, seed)
        attempts += 1
        if trial == 0:
            nominal = out.score
        sim_s += out.sim_s
        if best is None or out.score > best.score or (out.success and not best.success):
            best, best_x = out, cand
        if out.success:
            reruns = [run(ctl, seed + 1000 + k) for k in range(confirm)]
            attempts += len(reruns)
            sim_s += sum(r.sim_s for r in reruns)
            failed = next((r for r in reruns if not r.success), None)
            if failed is None:
                return out.model_copy(update={"controls": ctl, "attempts": attempts, "sim_s": round(sim_s, 3),
                                              "nominal": nominal})
            best = failed.model_copy(update={"detail": f"passed once, failed a confirmation: {failed.detail}"})
        sigma = max(0.05, sigma * shrink)
    assert best is not None
    return best.model_copy(update={"controls": {k: float(v) for k, v in zip(keys, best_x, strict=True)},
                                   "attempts": attempts, "sim_s": round(sim_s, 3), "success": False, "nominal": nominal})
