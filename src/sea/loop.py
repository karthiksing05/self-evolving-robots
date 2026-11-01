"""The evolve loop (spec Section 4) and the run log (spec 8.10).

One iteration:

1. Run System 1 on a sample of train panels (failing panels first) -> ``ControlResult`` per control.
2. Score surprise against concept memory *before* learning, then learn passes and confirmed failures.
3. Escalate failing controls by the Section 3.3 rule and pick targets (Section 8.4).
4. Ask the proposer (System 2 or a baseline); screen each proposal (ladder, posture, typecheck,
   build checks), price it (ledger), and keep the best by value per cost (Section 10).
5. Run the change on the target panels plus the reference panel, paired with the current state on
   the same randomization draws, and accept or reject by the posture's rule (Section 8.9).
6. Charge actual costs, record the fix, commit, promote and prune, anneal the posture, write logs.

The harness owns everything that judges: the verifier, the reward, hard limits and the ledger.
Proposers only ever see facts and return typed proposals (docs/harness.md).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sea.compose import dsl
from sea.compose.library import Library
from sea.contracts import (
    D_DUP,
    MEASUREMENTS,
    N_FAM,
    ControlResult,
    ControlSpec,
    CostVector,
    Decision,
    Design,
    Node,
    PanelSpec,
    PrimitiveDef,
    Program,
    Proposal,
    Report,
    RunSpec,
    ToolSpec,
    recess_bin,
    tilt_bin,
)
from sea.design.build import BuiltDesign, build
from sea.design.space import tools_equivalent
from sea.propose.baselines import ProposeContext, Proposer, make_proposer
from sea.propose.prompt import TraceSummary
from sea.propose.targets import Escalation, TargetGroup, gain_est, pick_targets
from sea.sim.env import STRUCTURAL, ControlRun, SuiteConfig, SuiteRun, reward_map, run_panel, run_suite
from sea.sim.panels import generate, reference_panel
from sea.spec.ledger import Ledger, ReserveError
from sea.spec.posture import EffectiveParams, accept, anneal, derive, extra_trial_runs
from sea.verify import coverage

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"
EPISODE_EST_S = 12.0  # sim seconds per control attempt, for cost-before-action estimates
AREA_FOR = {"reparam": "force", "program": "library", "promote": "library", "use_tool": "shape",
            "new_tool": "shape", "finger": "shape", "arm": "kinematics", "actuator": "kinematics"}
LEVEL_FOR = {"use_tool": "L0", "new_tool": "L0", "finger": "L1", "arm": "L2", "actuator": "L3"}
SUPPORTED_LEVELS = {"L0", "L3"}  # L1 fingers and L2 kinematics need design/arm.py (milestone M9)


def load_spec(path: str | Path) -> RunSpec:
    return RunSpec.model_validate(yaml.safe_load(Path(path).read_text()))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable))


def _jsonable(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, set | frozenset):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# ---------------------------------------------------------------- state
@dataclass
class State:
    spec: RunSpec
    built: BuiltDesign
    library: Library
    ledger: Ledger
    posture: Any  # contracts.Posture
    beta0: float
    encoder: Any  # memory.ContextEncoder
    memory: Any | None = None  # memory.ConceptMemory (None for the s2_nomem ablation)
    reach: Any = None
    reparam: dict[str, tuple[float, float]] = field(default_factory=dict)
    version: int = 0
    tool_library: list[ToolSpec] = field(default_factory=list)
    variants: dict[str, list[Node]] = field(default_factory=dict)  # program name -> passing bodies
    origin: dict[str, Proposal] = field(default_factory=dict)  # S2 program name -> its proposal
    promoted: set[str] = field(default_factory=set)
    contexts: dict[str, np.ndarray] = field(default_factory=dict)
    base_cache: dict[tuple, list[ControlRun]] = field(default_factory=dict)

    @property
    def design(self) -> Design:
        return self.built.design

    def eff(self, area: str | None = None) -> EffectiveParams:
        return derive(self.posture, area)


@dataclass
class Candidate:
    """A proposal after screening and costing."""

    proposal: Proposal
    errors: list[str]
    cost: CostVector = field(default_factory=CostVector)
    value: float = 0.0
    vpc: float = 0.0
    built: BuiltDesign | None = None
    library: Library | None = None
    reparam: dict[str, tuple[float, float]] | None = None
    new_tools: list[ToolSpec] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"proposal": self.proposal.model_dump(mode="json"), "errors": self.errors,
                "cost": self.cost.model_dump(), "value": self.value, "vpc": self.vpc}


def suite_config(state: State, built: BuiltDesign | None = None, library: Library | None = None,
                 reparam: dict[str, tuple[float, float]] | None = None, **kw: Any) -> SuiteConfig:
    built = built or state.built
    force, shape = state.eff("force"), state.eff()
    return SuiteConfig(design=built.design, tools=built.tools, library=library or state.library,
                       limits=state.spec.limits, force_cap_fraction=force.force_cap_fraction,
                       max_param_delta=shape.max_param_delta, repeats_before_fail=shape.repeats_before_fail,
                       m=state.spec.s1_trials, reach=state.reach,
                       reparam=state.reparam if reparam is None else reparam, seed=state.spec.seed, **kw)


def make_llm(spec: RunSpec, run_dir: Path) -> Any:
    from sea.propose.baselines import RecordingLLM, ScriptedLLM
    from sea.propose.llm import AnthropicLLM, ClaudeCodeLLM, MockLLM

    log = run_dir / "llm_calls.jsonl"
    if spec.llm == "mock":
        assert spec.llm_fixtures, "llm: mock needs llm_fixtures"
        return MockLLM(Path(spec.llm_fixtures), log_path=log)
    if spec.llm == "scripted":
        return ScriptedLLM(Path(spec.llm_fixtures) if spec.llm_fixtures else None, log_path=log)
    if spec.llm == "claude_code":  # pragma: no cover - live CLI
        live: Any = ClaudeCodeLLM(spec.llm_model, cache_dir=CACHE / "llm", log_path=log)
    else:  # pragma: no cover - live API
        live = AnthropicLLM(spec.llm_model, cache_dir=CACHE / "llm", log_path=log)
    return RecordingLLM(live, Path(spec.llm_fixtures)) if spec.llm_fixtures else live  # pragma: no cover


def init_state(spec: RunSpec, train: Sequence[PanelSpec]) -> State:
    from sea.control.executor import stock_ik
    from sea.control.reach import reach_for
    from sea.memory import ConceptMemory, ContextEncoder, FakeEmbedder

    if spec.embedder == "real":  # pragma: no cover - loads real models
        from sea.memory.embed import real_embedder

        embedder = real_embedder()
    else:
        embedder = FakeEmbedder()
    encoder = ContextEncoder(embedder, seed=spec.seed)
    items = [(c, p) for p in train for c in p.controls]
    if len(items) >= encoder.dim:
        encoder.fit(encoder.raw_many(items))
    memory = None
    if spec.proposer != "s2_nomem":
        memory = ConceptMemory(seed=spec.seed, embedder=embedder)
    library = Library.seed(spec.budgets.library_size)
    if memory is not None:
        for d in library.programs():
            assert d.body is not None
            memory.add_program(d.body, {})
    built = build(Design(id="d000"), spec.limits)
    reach = reach_for(stock_ik(), spec.limits, None, CACHE / "reach")
    return State(spec=spec, built=built, library=library, ledger=Ledger(spec.budgets), posture=spec.posture,
                 beta0=spec.posture.beta, encoder=encoder, memory=memory, reach=reach)


# ---------------------------------------------------------------- iteration pieces
def sample_iteration_panels(train: Sequence[PanelSpec], failing: Sequence[str], n: int, seed: int) -> list[PanelSpec]:
    """Failing panels from the last iteration first, then a fixed-seed sample of the rest (Sec 4)."""
    by_id = {p.id: p for p in train}
    first = [by_id[i] for i in failing if i in by_id][:n]
    rest = [p for p in train if p.id not in set(failing)]
    order = np.random.default_rng(seed).permutation(len(rest))
    return first + [rest[i] for i in order[: max(0, n - len(first))]]


def control_index(panels: Sequence[PanelSpec]) -> dict[str, tuple[ControlSpec, PanelSpec]]:
    return {f"{p.id}:{c.id}": (c, p) for p in panels for c in p.controls}


def measurements(r: ControlResult) -> dict[str, float]:
    return {m: float(getattr(r, m)) for m in MEASUREMENTS}


def context_vec(state: State, key: str, control: ControlSpec, panel: PanelSpec) -> np.ndarray:
    if key not in state.contexts:
        state.contexts[key] = state.encoder.encode(control, panel)
    return state.contexts[key]


def annotate(state: State, suite: SuiteRun, index: dict[str, tuple[ControlSpec, PanelSpec]]) -> None:
    """Fill surprise and familiarity from memory as it was *before* this iteration's results."""
    mem = state.memory
    for run in suite.runs:
        c, p = index[run.key]
        ctx = context_vec(state, run.key, c, p)
        if mem is None or mem.trees["context"].empty:
            continue
        z, nll = mem.surprise(ctx, measurements(run.result))
        run.result = run.result.model_copy(update={"surprise_z": z, "surprise_nll": nll,
                                                   "concept_n": mem.concept_n(ctx)})


def learn(state: State, suite: SuiteRun, iteration: int) -> None:
    if state.memory is None:
        return
    confirmed = {r.key for r in suite.runs if r.result.verdict != "pass" and r.confirmed}
    state.memory.learn(suite.results, confirmed, {r.key: state.contexts[r.key] for r in suite.runs}, iteration)


def bind_params(root: Node, params: dict[str, dict[str, float]]) -> Node:
    """Program body with System 1's tuned arguments written in (a promotion variant)."""
    if not params:
        return root
    upd: dict[str, Any] = {}
    if root.op == "prim" and root.prim is not None and root.prim.name in params:
        upd["prim"] = root.prim.model_copy(update={"args": {**root.prim.args, **params[root.prim.name]}})
    if root.children:
        upd["children"] = [bind_params(ch, params) for ch in root.children]
    return root.model_copy(update=upd) if upd else root


def record_uses(state: State, suite: SuiteRun, iteration: int) -> None:
    for run in suite.runs:
        if not run.program:
            continue
        passed = run.result.verdict == "pass"
        state.library.record_use(run.program, passed, iteration, run.result.panel_id, run.result.control_id)
        defn = state.library.get(run.program)
        if passed and defn is not None and defn.body is not None:
            state.variants.setdefault(run.program, []).append(bind_params(defn.body.root, run.params))


def make_report(state: State, run_id: str, iteration: int, panels: Sequence[PanelSpec],
                results: list[ControlResult], cost: CostVector, design: Design) -> Report:
    cov = coverage.coverage_reward(results)
    spec = state.spec
    rew = coverage.reward(cov, cost.print_minutes, spec.budgets.print_minutes, design.est.added_mass_g,
                          spec.limits.max_tool_mass_g, cost_terms=spec.ablations.cost_terms)
    return Report(run_id=run_id, iteration=iteration, design_id=design.id, panels=[p.id for p in panels],
                  results=results, control_coverage=coverage.control_coverage(results),
                  pass_rate=coverage.pass_rate(results), coverage_reward=cov, cost=cost, reward=rew)


def escalations(state: State, suite: SuiteRun) -> list[Escalation]:
    """Section 3.3: confirmed failures that are structural, exhausted System 1, surprising, or unfamiliar."""
    eff = state.eff()
    out: list[tuple[ControlRun, str]] = []
    for run in suite.runs:
        r = run.result
        if r.verdict == "pass" or not run.confirmed:
            continue
        if r.verdict in STRUCTURAL:
            reason = "structural"
        elif run.s1_exhausted:
            reason = "s1_exhausted"
        elif r.surprise_z and max(abs(v) for v in r.surprise_z.values()) > eff.surprise_tau:
            reason = "surprise"
        elif r.concept_n < N_FAM:
            reason = "unfamiliar"
        else:
            continue
        out.append((run, reason))
    concept: dict[str, int] = {}
    unc: dict[str, float] = {}
    for run, _ in out:
        ctx = state.contexts[run.key]
        if state.memory is not None and not state.memory.trees["context"].empty:
            concept[run.key] = state.memory.concept_id(ctx)
            unc[run.key] = state.memory.uncertainty(ctx)
        else:  # no memory: one concept per class, maximal uncertainty
            concept[run.key] = int(hashlib.sha256(run.result.cls.encode()).hexdigest()[:6], 16)
            unc[run.key] = 1.0
    sizes: dict[int, int] = {}
    for k in concept.values():
        sizes[k] = sizes.get(k, 0) + 1
    escs = []
    for run, reason in out:
        cid = concept[run.key]
        r = run.result.model_copy(update={"escalated": True, "escalation_reason": reason})
        run.result = r
        escs.append(Escalation(result=r, reason=reason, concept_id=cid, uncertainty=unc[run.key],
                               concept_size=sizes[cid]))
    return escs


def trace_summary(run: ControlRun, control: ControlSpec, panel: PanelSpec) -> TraceSummary:
    r, tr = run.result, run.trace
    s = tr.sensors if tr else None
    felt = {} if s is None else {"felt": True, "peak_wrist_acc": s.peak_acc, "stall_s": s.stall_s,
                                 "stall_joint": s.stall_joint, "payload_held_g": s.payload_held_g,
                                 "pad_drop_t": s.pad_drop_t}
    return TraceSummary(
        key=r.key, cls=r.cls, verdict=r.verdict, peak_force=r.peak_force, travel_reached=r.travel_reached,
        travel_required=control.travel, contact_types=list(tr.contact_types) if tr else [],
        ik_residual_mm=r.ik_residual_mm, joint_limit_hits=list(tr.joint_limit_hits) if tr else [],
        surprise_z=r.surprise_z, escalation_reason=r.escalation_reason, program_id=r.program_id,
        handle=control.handle, recess_mm=control.recess_depth_mm, tilt_deg=float(np.degrees(panel.tilt_rad)),
        stopped_by=tr.note if tr else "", **felt,
    )


def fix_query(targets: Sequence[TargetGroup], index: dict[str, tuple[ControlSpec, PanelSpec]]) -> str:
    parts = []
    for g in targets:
        for e in g.escalations:
            c, p = index[e.key]
            parts.append(f"{c.cls} {c.handle} recess={recess_bin(c.recess_depth_mm)} tilt={tilt_bin(p.tilt_rad)} "
                         f"{e.result.verdict}")
    return "; ".join(parts)


# ---------------------------------------------------------------- applying proposals
def provides_for(keys: Sequence[str], index: dict[str, tuple[ControlSpec, PanelSpec]]) -> str:
    handles: dict[str, set[str]] = {}
    for k in keys:
        if k in index:
            c, _ = index[k]
            handles.setdefault(c.cls, set()).add(c.handle)
    items = []
    for cls in sorted(handles):
        hs = sorted(h for h in handles[cls] if h != "none")
        items.append(f"{cls}[{'/'.join(hs)}]" if hs and cls in ("door", "drawer") else cls)
    return "|".join(items)


def apply_design(design: Design, p: Proposal, new_id: str) -> Design:
    d = p.design_delta
    if d is None:
        return design.model_copy(update={"id": new_id, "parent": design.id})
    rack = [t for i, t in enumerate(design.tool_rack) if i not in set(d.rack_remove)] + list(d.rack_add)
    arm = design.arm.model_copy(update={**d.arm, **({"servo_voltage": {**design.arm.servo_voltage,
                                                                      **d.servo_voltage}}
                                                    if d.servo_voltage else {})})
    fingers = design.fingers.model_copy(update=dict(d.fingers))
    return Design(id=new_id, parent=design.id, arm=arm, fingers=fingers, tool_rack=rack)


def add_program(state: State, p: Proposal, keys: Sequence[str], index: dict[str, tuple[ControlSpec, PanelSpec]],
                design: Design, iteration: int) -> tuple[Library | None, list[str]]:
    """A library copy with ``p.program`` added, if it type-checks (tool-wrapped as at run time) for a target."""
    from sea.compose.library import composite_requires
    from sea.compose.typecheck import typecheck, wrap_with_tool

    assert p.program is not None
    lib = copy.deepcopy(state.library)
    name = f"s2_{dsl.program_hash(p.program.root)[:8]}"
    prog = p.program.model_copy(update={"id": name, "source": "s2"})
    errors: list[str] = []
    ok = 0
    for k in keys:
        c, panel = index[k]
        wrapped = wrap_with_tool(prog, design, c, lib.prims, state.spec.limits, panel.rules.banned_contacts)
        if wrapped is None:
            errors.append(f"{k}: no rack tool provides the contacts this program needs")
            continue
        errs = typecheck(wrapped, design, lib.prims, state.spec.limits, state.reach, control=c, panel=panel)
        errors.extend(f"{k}: {e.code}: {e.message}" for e in errs)
        ok += not errs
    if ok == 0:
        return None, errors or ["program type-checks for no target"]
    lib.add(PrimitiveDef(name=name, kind="composite", doc=p.hypothesis[:200],
                         requires=composite_requires(prog.root, lib.prims),
                         provides={"actuates": provides_for(keys, index) or prog.target_cls},
                         body=prog, created_iter=iteration))
    return lib, []


def screen(state: State, p: Proposal, ctx: ProposeContext, index: dict[str, tuple[ControlSpec, PanelSpec]],
           iteration: int, s2: bool) -> Candidate:
    """Ladder/posture checks, typecheck, build checks, then cost and value (Sections 3.4, 9, 10)."""
    from sea.propose.ladder import ladder_errors, structure_errors, validate

    spec = state.spec
    eff = state.eff(AREA_FOR[p.kind])
    rack_slots = spec.budgets.rack_slots
    errors = validate(p, state.design, eff, rack_slots) if s2 else \
        ladder_errors(p) + structure_errors(p, state.design, rack_slots)
    level = LEVEL_FOR.get(p.kind)
    if level is not None and level not in spec.levels:
        errors.append(f"{p.kind} is level {level}, not enabled in this run ({spec.levels})")
    elif level is not None and level not in SUPPORTED_LEVELS:
        errors.append(f"level {level} changes need design/arm.py (milestone M9)")
    cand = Candidate(p, errors)
    if errors:
        return cand
    keys = [k for k in p.prediction.flips_to_pass if k in index] or ctx.target_keys
    if p.kind == "program":
        assert p.program is not None
        lib, cand.errors = add_program(state, p, keys, index, state.design, iteration)
        if cand.errors:
            return cand
        cand.library = lib
    elif p.kind == "reparam":
        assert p.reparam is not None
        from sea.compose.typecheck import reparam_errors

        failed = next((ctx.failed_primitive[k] for k in keys if ctx.failed_primitive.get(k)), None)
        errs = reparam_errors(p.reparam, state.library.prims, spec.limits, primitive=failed)
        if errs:
            cand.errors = [f"{e.code}: {e.message}" for e in errs]
            return cand
        cand.reparam = {**state.reparam, **{k: (float(lo), float(hi)) for k, (lo, hi) in p.reparam.items()}}
    elif p.kind == "promote":
        assert p.promote is not None
        lib = copy.deepcopy(state.library)
        if p.promote.name in lib.prims or p.promote.body is None:
            cand.errors.append("promote: name taken or no body")
            return cand
        lib.add(p.promote.model_copy(update={"created_iter": iteration}))
        cand.library = lib
    else:  # body change
        new_design = apply_design(state.design, p, f"d{iteration + 1:03d}_{state.version + 1}")
        built = build(new_design, spec.limits)
        if not built.ok:
            cand.errors = [f"build: {e}" for e in built.errors] or ["build checks failed"]
            return cand
        cand.built = built
        old = state.design.tool_rack
        cand.new_tools = [t for t in new_design.tool_rack if not any(t == o for o in old)]
        if p.program is not None:  # a tool together with the program that uses it
            lib, cand.errors = add_program(state, p, keys, index, built.design, iteration)
            if cand.errors:
                return cand
            cand.library = lib
    printed = [t for t in cand.new_tools if not any(tools_equivalent(t, k) for k in state.tool_library)]
    vol, grams, parts, material = 0.0, 0.0, 0, "pla"
    if cand.built is not None:
        for t in cand.built.tools:
            if any(t.spec == s for s in printed):
                vol += t.volume_cm3
                grams += t.grams_pla + t.grams_tpu
                parts += 1
                material = t.material if t.material in ("pla", "tpu") else "pla"
    n_ctl = sum(len(index_panel(index, pid).controls) for pid in {index[k][1].id for k in keys})
    sim_est = EPISODE_EST_S * spec.trial_repeats * 2 * (n_ctl + len(reference_panel().controls))
    cand.cost = state.ledger.estimate(p, added_volume_cm3=vol, material=material, sim_seconds=sim_est,
                                      n_parts=parts, grams=grams if parts else None)
    verdicts = {r.key: r.verdict for r in ctx.report.results}
    cand.value = p.prediction.confidence * sum(gain_est(verdicts[k]) for k in p.prediction.flips_to_pass
                                               if k in verdicts)
    cand.vpc = state.ledger.value_per_cost(cand.value, cand.cost)
    return cand


# ---------------------------------------------------------------- trials and acceptance
def trial_panels(state: State, targets: Sequence[TargetGroup], index: dict[str, tuple[ControlSpec, PanelSpec]],
                 eff: EffectiveParams, seed: int) -> tuple[list[PanelSpec], set[str], set[str]]:
    """Target panels plus the reference panel (full, or a sample of controls), and the keys to run."""
    ids = sorted({index[k][1].id for g in targets for k in g.keys})
    panels = [index_panel(index, i) for i in ids]
    keys = {f"{p.id}:{c.id}" for p in panels for c in p.controls}
    ref_keys: set[str] = set()
    if state.spec.ablations.reference_regression:
        ref = reference_panel()
        all_ref = [f"{ref.id}:{c.id}" for c in ref.controls]
        if eff.regression_panel == "sample":
            rng = np.random.default_rng(seed)
            all_ref = sorted(rng.choice(all_ref, size=min(eff.regression_sample_n, len(all_ref)), replace=False))
        ref_keys = set(all_ref)
        panels.append(ref)
        keys |= ref_keys
    return panels, keys, ref_keys


def index_panel(index: dict[str, tuple[ControlSpec, PanelSpec]], panel_id: str) -> PanelSpec:
    return next(p for c, p in index.values() if p.id == panel_id)


def paired_runs(cfg: SuiteConfig, panels: Sequence[PanelSpec], keys: set[str], repeats: int,
                cache: dict[tuple, list[ControlRun]] | None, version: int | None) -> tuple[list[dict[str, float]],
                                                                                      float, float]:
    """``repeats`` passes over ``panels`` restricted to ``keys``; returns reward maps, sim s, heat J."""
    maps: list[dict[str, float]] = []
    sim_s = heat = 0.0
    for rep in range(repeats):
        runs: list[ControlRun] = []
        for panel in panels:
            only = {k for k in keys if k.startswith(panel.id + ":")}
            ck = (version, panel.id, rep, frozenset(only))
            if cache is not None and ck in cache:
                runs.extend(cache[ck])
                continue
            cfg.only = only
            got = run_panel(panel, cfg, repeat=100 + rep)
            sim_s += sum(r.trace.sim_seconds for r in got if r.trace is not None)
            heat += sum(r.trace.servo_heat_j for r in got if r.trace is not None)
            if cache is not None:
                cache[ck] = got
            runs.extend(got)
        maps.append(reward_map(SuiteRun(runs)))
    return maps, sim_s, heat


def mean_by_key(maps: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for m in maps for k in m})
    return {k: float(np.mean([m[k] for m in maps if k in m])) for k in keys}


def cost_delta(state: State, cand: Candidate) -> float:
    if not state.spec.ablations.cost_terms:
        return 0.0
    from sea.contracts import LAMBDA_MASS, LAMBDA_PRINT

    b, lim = state.spec.budgets, state.spec.limits
    d = LAMBDA_PRINT * cand.cost.print_minutes / max(b.print_minutes, 1e-9)
    if cand.built is not None:
        d += LAMBDA_MASS * (cand.built.design.est.added_mass_g - state.design.est.added_mass_g) / lim.max_tool_mass_g
    return d


def run_trial(state: State, cand: Candidate, targets: Sequence[TargetGroup],
              index: dict[str, tuple[ControlSpec, PanelSpec]], iteration: int) -> tuple[Decision, CostVector, dict]:
    eff = state.eff(AREA_FOR[cand.proposal.kind])
    panels, keys, ref_keys = trial_panels(state, targets, index, eff, state.spec.seed + iteration)
    reps = state.spec.trial_repeats + extra_trial_runs(eff.accept_rule)
    base_cfg = suite_config(state, confirm=False)
    trial_cfg = suite_config(state, built=cand.built, library=cand.library, reparam=cand.reparam, confirm=False)
    base, s_b, h_b = paired_runs(base_cfg, panels, keys, reps, state.base_cache, state.version)
    trial, s_t, h_t = paired_runs(trial_cfg, panels, keys, reps, None, None)
    bm, tm = mean_by_key(base), mean_by_key(trial)
    regressions = sorted(k for k in ref_keys if bm.get(k, 0.0) >= 0.5 and tm.get(k, 0.0) < 0.5)
    flipped = sorted(k for k in tm if bm.get(k, 0.0) < 0.5 <= tm[k])
    decision = accept(trial, base, eff, cost_delta=cost_delta(state, cand), regressions=regressions,
                      seed=state.spec.seed + iteration, prediction=cand.proposal.prediction, flipped=flipped)
    body = cand.proposal.kind in ("use_tool", "new_tool", "finger", "arm", "actuator")
    actual = CostVector(print_minutes=cand.cost.print_minutes, grams_pla=cand.cost.grams_pla,
                        grams_tpu=cand.cost.grams_tpu, approvals=1 if body else 0,
                        sim_seconds=round(s_b + s_t, 4), servo_heat_j=round(h_b + h_t, 4))
    return decision, actual, {"base": bm, "trial": tm, "repeats": reps}


def commit(state: State, cand: Candidate, iteration: int, index: dict[str, tuple[ControlSpec, PanelSpec]]) -> None:
    p = cand.proposal
    if cand.built is not None:
        state.built = cand.built
        for t in cand.new_tools:
            if not any(tools_equivalent(t, k) for k in state.tool_library):
                state.tool_library.append(t)
    if cand.library is not None:
        new = set(cand.library.prims) - set(state.library.prims)
        state.library = cand.library
        for name in new:
            state.origin[name] = p
            body = state.library.prims[name].body
            if state.memory is not None and body is not None:
                state.memory.add_program(body, {})
    if cand.reparam is not None:
        state.reparam = cand.reparam
    state.version += 1


def maybe_promote(state: State, iteration: int) -> list[str]:
    """Promote S2 programs that passed on k_promote controls across 2 panels (Section 3.5)."""
    if not state.spec.ablations.library_learning:
        return []
    k = state.eff("library").k_promote
    out = []
    for name in sorted(state.origin):
        defn = state.library.get(name)
        if defn is None or defn.body is None or name in state.promoted or not state.library.promotion_ready(name, k):
            continue
        mem = state.memory

        def is_dup(prog: Program, mem: Any = mem) -> bool:
            return mem is not None and any(d < D_DUP for _, d in mem.similar_programs(prog, k=1))

        origin = state.origin[name]
        cls = defn.body.target_cls
        new_name = f"{cls}_{dsl.program_hash(defn.body.root)[:6]}"
        new = state.library.promote(defn.body, new_name, origin.hypothesis[:200], state.variants.get(name, []),
                                    iteration, is_duplicate=is_dup)
        state.promoted.add(name)
        if new is None:
            continue
        state.library.prims[new.name] = new.model_copy(update={"provides": defn.provides})
        if mem is not None and new.body is not None:
            mem.add_program(new.body, {})
        out.append(new.name)
    return out


# ---------------------------------------------------------------- the loop
def write_state(run_dir: Path, it: int, state: State) -> None:
    dump_json(run_dir / f"design_{it:03d}.json", state.design)
    dump_json(run_dir / f"library_{it:03d}.json", state.library.to_list())


def write_results(path: Path, results: Sequence[ControlResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in results))


def evolve(spec: RunSpec, run_dir: str | Path, *, proposer: Proposer | None = None,
           log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the loop and the final evaluation; everything lands in ``run_dir`` (spec 8.10)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    from sea.harness import fingerprint, judge_key

    fp = fingerprint()
    dump_json(run_dir / "harness.json", {"judge_key": judge_key(fp), "components": fp})
    train = generate("train", spec.splits.get("train", 200), spec.panel_seed)
    state = init_state(spec, train)
    llm = make_llm(spec, run_dir) if spec.proposer in ("s2", "s2_nomem") else None
    proposer = proposer or make_proposer(spec.proposer, llm=llm, model=spec.llm_model, levels=spec.levels,
                                         rack_slots=spec.budgets.rack_slots, limits=spec.limits, seed=spec.seed)
    timing: dict[str, float] = {}
    failing: list[str] = []
    stall = 0
    stop_reason = "iterations"
    it = -1
    for it in range(spec.budgets.design_iterations):
        t0 = time.time()
        it_dir = run_dir / f"iter_{it:03d}"
        write_state(run_dir, it, state)
        panels = sample_iteration_panels(train, failing, spec.iter_sample, spec.seed)
        index = control_index(panels)
        est = CostVector(sim_seconds=EPISODE_EST_S * len(index))
        if state.ledger.would_break_reserve(est):
            stop_reason = "sim budget reserve"
            break
        suite = run_suite(panels, suite_config(state, trace_dir=it_dir / "traces"))
        suite_cost = CostVector(sim_seconds=suite.sim_seconds, servo_heat_j=suite.servo_heat_j)
        breach = _charge(state, suite_cost)
        annotate(state, suite, index)
        learn(state, suite, it)
        record_uses(state, suite, it)
        esc = escalations(state, suite)  # marks escalated results before they are logged
        results = suite.results
        report = make_report(state, run_dir.name, it, panels, results, state.ledger.spent, state.design)
        write_results(it_dir / "results.jsonl", results)
        dump_json(it_dir / "report.json", report)
        cost_now = state.ledger.spent
        failing = sorted({r.panel_id for r in results if r.verdict != "pass"})
        log(f"iter {it}: coverage_reward={report.coverage_reward:.3f} reward={report.reward:.3f} "
            f"escalated={len(esc)}")
        proposal_log: dict[str, Any] = {"escalations": [e.key for e in esc], "candidates": [], "chosen": None}
        decision: Decision | None = None
        if breach:
            stop_reason = "sim budget reserve"
        elif report.coverage_reward >= spec.stop.target_reward:
            stop_reason = "target reward"
        elif esc:
            decision, proposal_log, stalled = _propose_and_try(state, proposer, esc, report, suite, index, it, it_dir,
                                                               proposal_log, log)
            stall = stall + 1 if stalled else 0
            if stall >= spec.stop.patience:
                stop_reason = "stalled"
        promoted = maybe_promote(state, it)
        pruned = state.library.prune(it, protected={r.program for r in suite.runs if r.program})
        proposal_log.update({"promoted": promoted, "pruned": pruned})
        if spec.posture.anneal_with_budget:
            state.posture = anneal(state.posture, state.beta0, state.ledger.remaining())
        state.ledger.tick_iteration()
        dump_json(it_dir / "cost.json", {"cumulative": cost_now, "iteration": suite_cost,
                                          "end": state.ledger.spent, "remaining": state.ledger.remaining()})
        dump_json(it_dir / "proposal.json", proposal_log)
        dump_json(it_dir / "decision.json", decision.model_dump(mode="json") if decision else None)
        if state.memory is not None:
            state.memory.save(run_dir / "memory")
        timing[f"iter_{it:03d}"] = round(time.time() - t0, 3)
        dump_json(run_dir / "timing.json", timing)
        if stop_reason != "iterations":
            break
    write_state(run_dir, it + 1, state)
    t0 = time.time()
    final = final_eval(state, run_dir, spec.eval_split)
    timing["final"] = round(time.time() - t0, 3)
    dump_json(run_dir / "timing.json", timing)
    final["stop_reason"] = stop_reason
    final["iterations"] = it + 1
    dump_json(run_dir / "final" / "summary.json", final)
    return final


def _charge(state: State, cost: CostVector) -> bool:
    """Charge ``cost``; returns True (and charges as spent) when it crossed the reserve."""
    try:
        state.ledger.charge(cost)
        return False
    except ReserveError:
        state.ledger.charge(cost, final=True)  # the simulation already ran: record it, then stop
        return True


def _propose_and_try(state: State, proposer: Proposer, esc: list[Escalation], report: Report, suite: SuiteRun,
                     index: dict[str, tuple[ControlSpec, PanelSpec]], it: int, it_dir: Path,
                     proposal_log: dict[str, Any], log: Callable[[str], None]) -> tuple[Decision | None, dict, bool]:
    spec = state.spec
    eff = state.eff()
    weights = state.ledger.scarcity_weights()
    targets = pick_targets(esc, eff.beta, weights, mode=spec.ablations.target_mode)
    runs = suite.by_key()
    fixes = []
    if state.memory is not None and spec.ablations.use_fix_tree:
        fixes = state.memory.similar_fixes(fix_query(targets, index), k=5)
    ctx = ProposeContext(
        iteration=it, targets=targets, report=report,
        summaries={k: trace_summary(runs[k], *index[k]) for g in targets for k in g.keys},
        controls=index, design=state.design, library=state.library, eff=eff, limits=spec.limits,
        budgets=spec.budgets, remaining=state.ledger.remaining(), fixes=fixes, tool_library=state.tool_library,
        failed_primitive={k: (runs[k].trace.failed_primitive if runs[k].trace is not None else "")
                          for g in targets for k in g.keys},
        seed=spec.seed + it,
    )
    llm = getattr(proposer, "llm", None)
    usd0 = float(getattr(llm, "total_usd", 0.0))
    if llm is not None and hasattr(llm, "log_path"):
        llm.log_path = it_dir / "llm_calls.jsonl"
    proposals = proposer.propose(ctx)
    usd = float(getattr(llm, "total_usd", 0.0)) - usd0
    if usd > 0:
        _charge(state, CostVector(llm_usd=usd))
    s2 = spec.proposer in ("s2", "s2_nomem")
    cands = [screen(state, p, ctx, index, it, s2) for p in proposals]
    proposal_log["targets"] = [g.keys for g in targets]
    proposal_log["candidates"] = [c.summary() for c in cands]
    valid = sorted((c for c in cands if not c.errors), key=lambda c: -c.vpc)
    best = valid[0] if valid else None
    if best is None or best.vpc < spec.stop.min_value_per_cost or state.ledger.would_break_reserve(best.cost):
        for c in cands:
            proposer.observe(c.proposal, None)
        log(f"  no proposal tried ({len(cands)} candidates, {len(valid)} valid)")
        return None, proposal_log, True
    proposal_log["chosen"] = cands.index(best)
    decision, actual, shares = run_trial(state, best, targets, index, it)
    proposal_log["trial"] = shares  # per-control pass share, base vs trial
    _charge(state, actual)
    proposer.observe(best.proposal, decision)
    if state.memory is not None:
        state.memory.record_fix(f"it{it:03d}", best.proposal, decision, it)
    if decision.accepted:
        commit(state, best, it, index)
    log(f"  tried {best.proposal.kind} (vpc={best.vpc:.2f}): accepted={decision.accepted} "
        f"delta={decision.delta_mean:+.3f} [{decision.delta_lo:+.3f}, {decision.delta_hi:+.3f}]")
    return decision, proposal_log, False


def final_eval(state: State, run_dir: Path, split: str, out_name: str = "final") -> dict[str, Any]:
    """Evaluate the final design and library on ``split`` (no learning, no proposals)."""
    spec = state.spec
    panels = generate(split, spec.splits.get(split, 50), spec.panel_seed) if split != "reference" \
        else [reference_panel()]
    out = run_dir / out_name
    index = control_index(panels)
    suite = run_suite(panels, suite_config(state, confirm=False, trace_dir=out / "traces"))
    state.ledger.charge(CostVector(sim_seconds=suite.sim_seconds, servo_heat_j=suite.servo_heat_j), final=True)
    annotate(state, suite, index)
    results = suite.results
    report = make_report(state, run_dir.name, -1, panels, results, state.ledger.spent, state.design)
    write_results(out / "results.jsonl", results)
    dump_json(out / "report.json", report)
    dump_json(out / "cost.json", {"cumulative": state.ledger.spent})
    dump_json(out / "design.json", state.design)
    dump_json(out / "library.json", state.library.to_list())
    spent = state.ledger.spent
    cov = report.coverage_reward
    per = {"per_print_hour": spent.print_minutes / 60.0, "per_gram": spent.grams_pla + spent.grams_tpu,
           "per_llm_usd": spent.llm_usd, "per_approval": float(spent.approvals),
           "per_sim_hour": spent.sim_seconds / 3600.0}
    return {"split": split, "coverage_reward": cov, "reward": report.reward,
            "control_coverage": report.control_coverage, "pass_rate": report.pass_rate,
            "summary": coverage.summarize(results), "cost": spent.model_dump(),
            "efficiency": {k: (cov / v if v > 0 else None) for k, v in per.items()},
            "design": state.design.id, "library_size": len(state.library.prims)}


def load_state(run_dir: str | Path) -> State:
    """Rebuild the final design and library of a finished run (for ``sea eval``)."""
    from sea.memory import ContextEncoder, FakeEmbedder

    run_dir = Path(run_dir)
    spec = load_spec(run_dir / "spec.yaml")
    design = Design.model_validate_json((run_dir / "final" / "design.json").read_text())
    prims = [PrimitiveDef.model_validate(d) for d in json.loads((run_dir / "final" / "library.json").read_text())]
    library = Library({p.name: p for p in prims}, spec.budgets.library_size)
    from sea.control.executor import stock_ik
    from sea.control.reach import reach_for

    return State(spec=spec, built=build(design, spec.limits), library=library, ledger=Ledger(spec.budgets),
                 posture=spec.posture, beta0=spec.posture.beta, encoder=ContextEncoder(FakeEmbedder(), seed=spec.seed),
                 reach=reach_for(stock_ik(), spec.limits, None, CACHE / "reach"))


def evaluate(run_dir: str | Path, split: str) -> dict[str, Any]:
    state = load_state(run_dir)
    out = final_eval(state, Path(run_dir), split, out_name=f"eval_{split}")
    dump_json(Path(run_dir) / f"eval_{split}" / "summary.json", out)
    return out


__all__ = ["State", "evaluate", "evolve", "final_eval", "load_spec", "load_state"]
