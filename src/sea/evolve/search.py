"""System 1 for any world: local search over a task's control parameters, then confirmation.

``local_search(run, space, start)`` tries the world's default controls, then samples around the best
score so far with a shrinking step, and stops at the first success.  A success counts only if it repeats
on ``confirm`` fresh seeds (spec 8.5): a lucky draw is not a skill.  ``score`` guides the search;
``success`` decides.  Worlds with their own System 1 (the MuJoCo grasp world) call theirs instead.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sea.evolve.core import Body, Outcome

Run = Callable[[dict[str, float], int], Outcome]


def planned(body: Body, space: dict[str, tuple[float, float]], start: dict[str, float], m: int) -> tuple[dict[str, float], int]:
    """The search's start and attempts for ``body``: its operating plan's values over ``start``, and its allowance (D-092)."""
    if body.plan is None:
        return start, m
    return {**start, **{k: v for k, v in body.plan.controls.items() if k in space}}, m * body.plan.search


def local_search(run: Run, space: dict[str, tuple[float, float]], start: dict[str, float], m: int = 16,
                 confirm: int = 2, seed: int = 0, step: float = 0.3, shrink: float = 0.85) -> Outcome:
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
