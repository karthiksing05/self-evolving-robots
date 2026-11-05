"""Coverage suite runner (spec Section 4 ``run_suite``): System 1 on every control of every panel.

For each control the runner picks the library programs that type-check for it
(``compose.typecheck.applicable_programs``; none -> ``no_applicable_primitive`` without simulating),
mounts the rack tool the program needs, and hands the first program to System 1
(``control.local_search.run_system1``: one attempt, the fixed retry rule, then CMA-ES within ``m``
trials).  Remaining applicable programs get one attempt each if System 1 did not pass
(docs/decisions.md D-016).

Every repeat rebuilds the panel model with its own domain-randomization draw (friction x U(0.8, 1.2),
stiffness x U(0.9, 1.1), perceived pose noise N(0, 1 mm)), seeded from ``(seed, panel id, repeat)``
so trial and base runs are paired.  A failure is *confirmed* when its verdict repeats on
``repeats_before_fail`` independent draws (spec 8.5 "Learning"); unconfirmed failures are neither
learned nor escalated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from sea.compose.library import Library
from sea.contracts import (
    S1_TRIALS,
    ControlResult,
    ControlSpec,
    Design,
    HardLimits,
    PanelSpec,
    Program,
    StepTrace,
)
from sea.control.executor import EpisodeConfig, reset_episode, run_program
from sea.sim.build_model import DomainRandomization, Sim, build_model
from sea.sim.controls import Faults
from sea.spec.ledger import ServoThermal

STRUCTURAL = frozenset({"unreachable", "rule_violation", "no_applicable_primitive"})


def dr_seed(seed: int, panel_id: str, repeat: int) -> int:
    """Stable per-(run seed, panel, repeat) randomization seed."""
    digest = hashlib.sha256(f"{seed}:{panel_id}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass
class SuiteConfig:
    """Everything a suite run needs besides the panels."""

    design: Design
    tools: list[Any]  # design.build.BuiltTool per rack slot
    library: Library
    limits: HardLimits
    force_cap_fraction: float = 0.8
    max_param_delta: float = 0.65
    repeats_before_fail: int = 2
    m: int = S1_TRIALS
    reach: Any = None  # control.reach.ReachGrid for the stock tip, or None to skip reach checks
    reparam: dict[str, tuple[float, float]] = field(default_factory=dict)
    faults: dict[str, dict[str, Faults]] = field(default_factory=dict)  # panel id -> control id -> faults
    seed: int = 0
    thermal: bool = True
    confirm: bool = True  # re-run failures on fresh draws to confirm them
    trace_dir: Path | None = None
    only: set[str] | None = None  # restrict to these control keys (regression samples)
    on_control: Callable[[ControlRun], None] | None = None


@dataclass
class ControlRun:
    result: ControlResult
    trace: StepTrace | None
    program: str = ""  # library name of the program System 1 settled on
    params: dict[str, dict[str, float]] = field(default_factory=dict)
    confirmed: bool = True  # passes are always confirmed
    s1_exhausted: bool = False  # local search used all m trials without a pass
    repeats: list[str] = field(default_factory=list)  # verdicts of the confirmation re-runs
    ways: int = 1  # ways of using the mechanism System 1 searched before this result (D-102)

    @property
    def key(self) -> str:
        return self.result.key


@dataclass
class SuiteRun:
    runs: list[ControlRun]
    sim_seconds: float = 0.0
    servo_heat_j: float = 0.0

    @property
    def results(self) -> list[ControlResult]:
        return [r.result for r in self.runs]

    def by_key(self) -> dict[str, ControlRun]:
        return {r.key: r for r in self.runs}


def force_cap(control: ControlSpec, panel: PanelSpec, limits: HardLimits, fraction: float) -> float:
    """Posture fraction of the class cap, never above the hard limit (D-011)."""
    return float(min(fraction * panel.rules.force_cap_by_class[control.cls], limits.max_force_n))


def episode_config(control: ControlSpec, panel: PanelSpec, cfg: SuiteConfig, noise: np.ndarray,
                   thermal: ServoThermal | None, sensor_seed: int = 0) -> EpisodeConfig:
    banned = frozenset(set(panel.rules.banned_contacts) | set(cfg.limits.banned_contacts))
    return EpisodeConfig(
        limits=cfg.limits,
        force_cap=force_cap(control, panel, cfg.limits, cfg.force_cap_fraction),
        class_cap=panel.rules.force_cap_by_class[control.cls],
        banned=banned,
        time_limit_s=panel.rules.time_limit_s,
        perception_offset=np.asarray(noise, dtype=float),
        library=cfg.library.prims,
        thermal=thermal,
        sensor_seed=sensor_seed,
    )


def candidate_programs(control: ControlSpec, panel: PanelSpec, cfg: SuiteConfig) -> list[tuple[str, Program]]:
    """Applicable library programs for ``control``, already wrapped in the tool they need."""
    from sea.compose.typecheck import applicable_programs

    banned = tuple(sorted(set(panel.rules.banned_contacts) | set(cfg.limits.banned_contacts)))
    out: list[tuple[str, Program]] = []
    for defn in applicable_programs(control, cfg.design, cfg.library.prims, cfg.limits, cfg.reach, panel=panel,
                                    banned=banned):
        assert defn.body is not None
        out.append((defn.name, defn.body))
    return out


def unreachable(panel: PanelSpec, control: ControlSpec) -> ControlResult:
    """Pre-simulation ``unreachable``: every applicable program fails only the reach check.

    The residual is the stock IK's best residual at the contact point with the control's approach.
    """
    from sea.compose.typecheck import control_target
    from sea.control.executor import stock_ik

    point, axis = control_target(panel, control)
    _, res, _ = stock_ik().solve(point, axis)
    return ControlResult(panel_id=panel.id, control_id=control.id, cls=control.cls, verdict="unreachable",
                         peak_force=0.0, travel_reached=0.0, detents=0, time_s=0.0, ik_residual_mm=round(res, 4))


def _judge(control: ControlSpec, panel: PanelSpec, limits: HardLimits) -> Callable[[StepTrace], ControlResult]:
    from sea.verify.verdicts import judge

    def fn(trace: StepTrace) -> ControlResult:
        r = judge(trace, control, panel.rules, limits)
        s = trace.sensors  # what the robot felt rides along for memory and System 2, after the verdict (D-036)
        return r if s is None else r.model_copy(update={"peak_acc": s.peak_acc, "stall_s": s.stall_s})

    return fn


def _save_trace(trace: StepTrace, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = trace.model_dump(mode="json", exclude={"t", "q", "tip_force", "control_q", "push_phase", "in_contact"})
    np.savez_compressed(
        path,
        t=np.asarray(trace.t, dtype=np.float32),
        q=np.asarray(trace.q, dtype=np.float32).reshape(len(trace.t), -1) if trace.q else np.zeros((0, 6)),
        tip_force=np.asarray(trace.tip_force, dtype=np.float32),
        control_q=np.asarray(trace.control_q, dtype=np.float32),
        push_phase=np.asarray(trace.push_phase, dtype=bool),
        in_contact=np.asarray(trace.in_contact, dtype=bool),
        meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
    )
    return f"{path.parent.parent.name}/{path.parent.name}/{path.name}"  # relative to the run dir: runs stay comparable


def load_trace(path: str | Path) -> StepTrace:
    data = np.load(path)
    meta = json.loads(bytes(data["meta"]).decode())
    return StepTrace(**meta, t=data["t"].tolist(), q=data["q"].tolist(), tip_force=data["tip_force"].tolist(),
                     control_q=data["control_q"].tolist(), push_phase=data["push_phase"].tolist(),
                     in_contact=data["in_contact"].tolist())


class PanelRunner:
    """One panel under one randomization draw: compiled model plus per-control episode configs."""

    def __init__(self, panel: PanelSpec, cfg: SuiteConfig, repeat: int):
        self.panel = panel
        self.cfg = cfg
        self.seed = dr_seed(cfg.seed, panel.id, repeat)
        self.dr = DomainRandomization(seed=self.seed)
        self.noise = {c: v[2] for c, v in self.dr.draws([c.id for c in panel.controls]).items()}
        self.sim: Sim = build_model(cfg.design, panel, cfg.tools, faults=cfg.faults.get(panel.id), dr=self.dr)
        self.thermal = ServoThermal() if cfg.thermal else None

    def config(self, control: ControlSpec) -> EpisodeConfig:
        return episode_config(control, self.panel, self.cfg, self.noise[control.id], self.thermal,
                              sensor_seed=self.seed)

    def attempt(self, control: ControlSpec, program: Program, params: dict[str, dict[str, float]],
                attempt: int = 0) -> StepTrace:
        reset_episode(self.sim)
        ecfg = self.config(control)
        ecfg.param_overrides = {k: dict(v) for k, v in params.items()}
        return run_program(self.sim, program, control.id, ecfg, panel_id=self.panel.id, attempt=attempt)


def other_ways(runner: PanelRunner, control: ControlSpec, prog: Program, first: ControlRun, judge_fn) -> ControlRun:
    """A failure is not yet the mechanism's: search each other way its primitives can be used, as System 1 searched the
    plain one, and stop at the first that passes (D-102).  Otherwise the best of them, saying how many were tried."""
    from sea.control.local_search import modes_of, objective, run_system1

    cfg, panel = runner.cfg, runner.panel
    if first.result.verdict == "pass":
        return first
    travel = runner.sim.controls[control.id].spec.travel
    best, trials, ways = first, first.result.s1_trials, 1
    for k, (name, mode) in enumerate(modes_of(prog, cfg.library.prims), 1):
        reset_episode(runner.sim)
        ecfg = runner.config(control)
        ecfg.param_overrides = {name: dict(mode)}
        out = run_system1(runner.sim, prog, control.id, ecfg, judge_fn, m=cfg.m, max_param_delta=cfg.max_param_delta,
                          seed=runner.seed + k, panel_id=panel.id, reparam=cfg.reparam or None)  # its own samples (D-105)
        params = {**out.params, name: {**mode, **out.params.get(name, {})}}
        run = ControlRun(out.result, out.trace, first.program, params, s1_exhausted=out.escalate == "s1_exhausted")
        trials, ways = trials + out.result.s1_trials + 1, ways + 1
        if objective(run.result, run.trace, travel) < objective(best.result, best.trace, travel):
            best = run
        if run.result.verdict == "pass":
            break
    best.ways = ways
    best.result = best.result.model_copy(update={"s1_trials": trials})
    return best


def run_control(runner: PanelRunner, control: ControlSpec) -> ControlRun:
    """System 1 on one control: candidates in order, local search on the first."""
    from sea.control.local_search import run_system1
    from sea.verify.verdicts import no_applicable

    cfg, panel = runner.cfg, runner.panel
    cands = candidate_programs(control, panel, cfg)
    if not cands:
        if cfg.reach is not None and candidate_programs(control, panel, replace(cfg, reach=None)):
            return ControlRun(unreachable(panel, control), None)  # only the reach check failed (D-022)
        return ControlRun(no_applicable(panel.id, control), None)
    judge_fn = _judge(control, panel, cfg.limits)
    best: ControlRun | None = None
    for i, (name, prog) in enumerate(cands):
        if i == 0:
            reset_episode(runner.sim)
            out = run_system1(runner.sim, prog, control.id, runner.config(control), judge_fn, m=cfg.m,
                              max_param_delta=cfg.max_param_delta, seed=runner.seed, panel_id=panel.id,
                              reparam=cfg.reparam or None)
            run = ControlRun(out.result, out.trace, name, out.params, s1_exhausted=out.escalate == "s1_exhausted")
            run = other_ways(runner, control, prog, run, judge_fn)
        else:
            trace = runner.attempt(control, prog, {})
            run = ControlRun(judge_fn(trace), trace, name, {})
            run.result = run.result.model_copy(update={"program_id": name})
        if run.result.verdict != "pass" and best is not None and best.result.verdict in STRUCTURAL:
            best = run  # prefer an attempt that got past the structural failure
        elif best is None or run.result.verdict == "pass":
            best = run
        if best.result.verdict == "pass":
            break
    assert best is not None
    best.result = best.result.model_copy(update={"program_id": best.program or best.result.program_id})
    return best


def confirm(run: ControlRun, panel: PanelSpec, control: ControlSpec, cfg: SuiteConfig) -> ControlRun:
    """Re-run a failure on fresh draws; confirmed when every re-run repeats its verdict."""
    need = max(0, cfg.repeats_before_fail - 1)
    if run.result.verdict == "pass":
        return run
    if run.result.verdict == "no_applicable_primitive" or need == 0 or not cfg.confirm:
        run.confirmed = True
        return run
    prog = dict(candidate_programs(control, panel, cfg)).get(run.program)
    if prog is None:
        run.confirmed = True
        return run
    judge_fn = _judge(control, panel, cfg.limits)
    for k in range(need):
        rr = PanelRunner(panel, cfg, repeat=1000 + k)
        run.repeats.append(judge_fn(rr.attempt(control, prog, run.params)).verdict)
    run.confirmed = all(v == run.result.verdict for v in run.repeats)
    return run


def run_panel(panel: PanelSpec, cfg: SuiteConfig, repeat: int = 0) -> list[ControlRun]:
    runner = PanelRunner(panel, cfg, repeat)
    out: list[ControlRun] = []
    for control in panel.controls:
        key = f"{panel.id}:{control.id}"
        if cfg.only is not None and key not in cfg.only:
            continue
        run = run_control(runner, control)
        if repeat == 0:
            run = confirm(run, panel, control, cfg)
        if cfg.trace_dir is not None and run.trace is not None:
            name = f"{panel.id}_{control.id}" + (f"_r{repeat}" if repeat else "") + ".npz"
            path = _save_trace(run.trace, cfg.trace_dir / name)
            run.result = run.result.model_copy(update={"trace_path": path})
        if cfg.on_control is not None:
            cfg.on_control(run)
        out.append(run)
    return out


def attempts(run: ControlRun) -> int:
    """Simulated attempts behind a result: the final one, earlier System 1 trials and confirmation re-runs."""
    return 1 + max(run.result.s1_trials - 1, 0) + len(run.repeats)


def run_suite(panels: Sequence[PanelSpec], cfg: SuiteConfig, repeat: int = 0) -> SuiteRun:
    """One pass over ``panels`` (all controls, or ``cfg.only``) under randomization draw ``repeat``."""
    runs: list[ControlRun] = []
    for panel in panels:
        runs.extend(run_panel(panel, cfg, repeat))
    sim_s = sum(r.trace.sim_seconds for r in runs if r.trace is not None)
    heat = sum(r.trace.servo_heat_j for r in runs if r.trace is not None)
    # Confirmation re-runs and local-search trials also cost sim time; traces carry the final attempt only,
    # so charge each System 1 trial and re-run at the final attempt's duration (an estimate, D-016).
    extra = sum((attempts(r) - 1) * r.trace.sim_seconds for r in runs if r.trace is not None)
    return SuiteRun(runs, round(sim_s + extra, 4), round(heat, 4))


def reward_map(suite: SuiteRun) -> dict[str, float]:
    """Per-control reward for acceptance: 1.0 for pass, 0.0 otherwise."""
    return {r.key: 1.0 if r.result.verdict == "pass" else 0.0 for r in suite.runs}
