"""Proposers: System 2 and the comparison baselines (spec 8.6, E1).

Every proposer turns the iteration's escalated targets into ``Proposal`` objects; the loop then
runs the same filters (ladder, typecheck, build checks), costing, value-per-cost selection and
acceptance for all of them, so differences in results come from the proposals alone.

- ``S2Proposer``: builds the ``S2Prompt`` (spec 8.6) and asks an ``LLM``.  ``use_memory=False`` is
  the ``s2_nomem`` ablation (no similar fixes in the prompt).
- ``ScriptedLLM``: an offline, rule-based stand-in for Claude that follows the ladder.  It answers
  the same prompt (it reads the structured context bound by the proposer) and can record
  ``MockLLM`` fixtures, which is how the integration fixtures are made (docs/decisions.md D-017).
- ``RandomProposer``, ``CMAProposer``, ``BOProposer``: body search over ``design.space.DesignSpace``
  (L0 rack contents and parameters) with no program changes.
- ``S1OnlyProposer``: re-parameterization only (rung 0); it cannot change program structure or the body.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sea.compose.library import Library
from sea.contracts import (
    PROPOSAL_RUNG,
    Budgets,
    ControlSpec,
    Decision,
    Design,
    DesignDelta,
    FixRecord,
    HardLimits,
    PanelSpec,
    Prediction,
    Program,
    Proposal,
    Report,
    ToolSpec,
)
from sea.design.space import DesignSpace, make_tool_spec, tools_equivalent
from sea.propose.llm import LLM, LLMCall, _CallLog, write_fixture
from sea.propose.prompt import Rejection, S2Prompt, TraceSummary, build_prompt
from sea.propose.targets import TargetGroup
from sea.spec.posture import EffectiveParams


@dataclass
class ProposeContext:
    """What a proposer may look at (the same facts System 2's prompt is built from)."""

    iteration: int
    targets: list[TargetGroup]
    report: Report
    summaries: dict[str, TraceSummary]  # result key -> trace summary
    controls: dict[str, tuple[ControlSpec, PanelSpec]]  # result key -> specs
    design: Design
    library: Library
    eff: EffectiveParams
    limits: HardLimits
    budgets: Budgets
    remaining: dict[str, float]
    fixes: list[FixRecord] = field(default_factory=list)
    tool_library: list[ToolSpec] = field(default_factory=list)  # every tool built so far
    failed_primitive: dict[str, str] = field(default_factory=dict)  # result key -> primitive name
    seed: int = 0
    rejected: list[Rejection] = field(default_factory=list)  # recent proposals screening refused

    @property
    def target_keys(self) -> list[str]:
        return [k for g in self.targets for k in g.keys]


class Proposer(Protocol):
    name: str

    def propose(self, ctx: ProposeContext) -> list[Proposal]: ...

    def observe(self, proposal: Proposal, decision: Decision | None) -> None: ...


def ruled_out_for(rung: int, reason: str) -> list[str]:
    """``rung<i>: reason`` for every rung below ``rung`` (spec 3.4 mechanical check)."""
    return [f"rung{i}: {reason}" for i in range(rung)]


def _rack_delta(design: Design, add: ToolSpec, rack_slots: int) -> DesignDelta:
    """Add ``add`` to the rack, naming the oldest slot to remove when the rack is full (Sec 10)."""
    remove = [0] if len(design.tool_rack) >= rack_slots else []
    return DesignDelta(rack_add=[add], rack_remove=remove)


# ---------------------------------------------------------------- System 2
class S2Proposer:
    """System 2: prompt -> LLM -> proposals."""

    def __init__(self, llm: LLM, model: str, use_memory: bool = True, max_proposals: int = 3):
        self.llm = llm
        self.model = model
        self.use_memory = use_memory
        self.max_proposals = max_proposals
        self.name = "s2" if use_memory else "s2_nomem"
        self.last_prompt: S2Prompt | None = None

    def prompt(self, ctx: ProposeContext) -> S2Prompt:
        failing = [ctx.summaries[k] for k in ctx.target_keys if k in ctx.summaries]
        return build_prompt(limits=ctx.limits, eff=ctx.eff, remaining=ctx.remaining, budgets=ctx.budgets,
                            library=list(ctx.library.prims.values()), design=ctx.design, failing=failing,
                            fixes=ctx.fixes if self.use_memory else [], model=self.model,
                            max_proposals=self.max_proposals, rejected=ctx.rejected)

    def propose(self, ctx: ProposeContext) -> list[Proposal]:
        prompt = self.prompt(ctx)
        self.last_prompt = prompt
        bind = getattr(self.llm, "bind", None)
        if bind is not None:
            bind(ctx)
        return list(self.llm.propose(prompt))[: self.max_proposals]

    def observe(self, proposal: Proposal, decision: Decision | None) -> None:
        return None


# Templates and parameters the scripted System 2 reaches for, by what the control needs.
_TOOL_FOR = {
    "pull": ("hook", {"depth": 14.0, "throat": 6.0, "length": 45.0}),
    "tab": ("hook", {"depth": 14.0, "throat": 6.0, "length": 45.0}),
    "push_latch": ("probe", {"diameter": 6.0, "length": 50.0}),
    "knob": ("knob_cup", {"inner_diameter": 0.0, "depth": 10.0, "ribs": 4.0}),
    "slot": ("slot_inserter", {"blade_thickness": 1.4, "width": 24.0, "length": 40.0}),
    "recessed_button": ("probe", {"diameter": 5.0, "length": 0.0}),
    "lever": ("pad", {"w": 16.0, "h": 16.0, "thickness": 5.0, "friction": 0.9}),
}


# Programs the scripted System 2 composes (rung 1) for controls no library program provides.
_PROGRAM_FOR = {
    "slot": "Seq(approach($target), contact($target), insert($target), retreat())",
    "push_latch": "Seq(approach($target), contact($target), push($target), retreat())",
}


class ScriptedLLM(_CallLog):
    """Offline System 2 that follows the escalation ladder with fixed rules.

    It reads the structured ``ProposeContext`` bound by ``S2Proposer`` (the same facts the prompt
    renders) and returns at most one proposal per target group:
    structural failures that need a contact the rack lacks -> rung 2 (reuse a known tool) or rung 3
    (new tool from a template); stuck or wrong behaviour with a failing primitive -> rung 0
    (re-parameterize its force/extent range).  ``record_dir`` writes each answer as a ``MockLLM``
    fixture for the prompt it answered.
    """

    def __init__(self, record_dir: Path | None = None, log_path: Path | None = None):
        super().__init__(log_path)
        self.record_dir = Path(record_dir) if record_dir is not None else None
        self.ctx: ProposeContext | None = None

    def bind(self, ctx: ProposeContext) -> None:
        self.ctx = ctx

    def propose(self, prompt: S2Prompt) -> list[Proposal]:
        assert self.ctx is not None, "ScriptedLLM needs a bound ProposeContext"
        out: list[Proposal] = []
        for group in self.ctx.targets:
            p = self._for_group(group)
            if p is not None and not any(p.model_dump() == q.model_dump() for q in out):
                out.append(p)
        if not out:  # nothing to change: re-parameterize the first target as a cheap probe
            out.append(self._reparam(self.ctx.target_keys[:1], "no rule matched"))
        out = out[:3]
        self._log(LLMCall(prompt_hash=prompt.hash(), model="scripted", attempts=1, n_proposals=len(out)))
        if self.record_dir is not None:
            write_fixture(prompt, out, self.record_dir)
        return out

    def _tool_need(self, control: ControlSpec) -> tuple[str, dict[str, float]] | None:
        if control.cls in ("door", "drawer") and control.handle in _TOOL_FOR:
            return _TOOL_FOR[control.handle]
        if control.cls == "knob":
            t, p = _TOOL_FOR["knob"]
            return t, {**p, "inner_diameter": float(np.clip(control.size_mm[0] + 2.0, 8.0, 40.0))}
        if control.cls == "slot":  # blade thinner than the gap, but no thinner than the printable wall
            t, p = _TOOL_FOR["slot"]
            gap = control.size_mm[1] if len(control.size_mm) > 1 else 2.0
            return t, {**p, "blade_thickness": float(np.clip(0.65 * gap, 1.2, 2.0))}
        if control.cls == "recessed_button":
            t, p = _TOOL_FOR["recessed_button"]
            return t, {**p, "length": float(np.clip(control.recess_depth_mm + 25.0, 20.0, 80.0))}
        if control.cls in _TOOL_FOR:
            return _TOOL_FOR[control.cls]
        return None

    def _rack_has(self, template: str) -> bool:
        assert self.ctx is not None
        return any(t.template == template for t in self.ctx.design.tool_rack)

    def _program(self, control: ControlSpec) -> Program | None:
        """A composed program when no library program provides this control at all."""
        from sea.compose import dsl
        from sea.compose.library import provides_control

        assert self.ctx is not None
        if any(provides_control(d, control) for d in self.ctx.library.programs()):
            return None
        text = _PROGRAM_FOR.get(control.handle if control.handle == "push_latch" else control.cls)
        if text is None:
            return None
        return Program(id="s2_draft", target_cls=control.cls, root=dsl.parse(text), source="s2")

    def _for_group(self, group: TargetGroup) -> Proposal | None:
        ctx = self.ctx
        assert ctx is not None
        keys = group.keys
        esc = group.escalations[0]
        control, _ = ctx.controls[esc.key]
        verdict = esc.result.verdict
        summary = ctx.summaries.get(esc.key)
        need = self._tool_need(control)
        program = self._program(control)
        if program is not None and (need is None or self._rack_has(need[0])):
            return Proposal(kind="program", rung=1, rationale=f"no library program provides {control.cls}",
                            hypothesis=f"{control.cls} needs a composition the library lacks",
                            ruled_out=ruled_out_for(1, "no existing program applies, so parameters cannot help"),
                            program=program, prediction=Prediction(flips_to_pass=keys, confidence=0.5))
        pinched = summary is not None and "pinch" in summary.contact_types
        structural = verdict in ("rule_violation", "no_applicable_primitive", "unreachable")
        if need is not None and (structural or pinched or verdict == "stuck") and not self._rack_has(need[0]):
            template, params = need
            spec = make_tool_spec(template, **params)
            known = next((t for t in ctx.tool_library if tools_equivalent(t, spec)), None)
            if known is not None and ctx.eff.max_rung >= 2:
                return self._tool(keys, known, "use_tool", f"a known {template} provides the missing contact",
                                  program)
            if ctx.eff.max_rung >= 3:
                return self._tool(keys, spec, "new_tool", f"no rack tool provides {spec.contact} contact", program)
        if verdict in ("stuck", "wrong_behavior", "no_actuation", "inconclusive"):
            return self._reparam(keys, f"{verdict}: widen the failing primitive's range")
        return None

    def _tool(self, keys: list[str], spec: ToolSpec, kind: str, why: str, program: Program | None = None) -> Proposal:
        ctx = self.ctx
        assert ctx is not None
        rung = PROPOSAL_RUNG[kind]
        return Proposal(
            kind=kind,  # type: ignore[arg-type]
            rung=rung,
            rationale=f"{why}; targets {', '.join(keys)}",
            hypothesis=f"the targets fail because the mounted end effector cannot make a {spec.contact} contact",
            ruled_out=ruled_out_for(rung, "parameters or recomposition cannot create a missing contact type"),
            design_delta=_rack_delta(ctx.design, spec, ctx.budgets.rack_slots),
            program=program,
            prediction=Prediction(flips_to_pass=keys, confidence=0.6),
        )

    def _reparam(self, keys: list[str], why: str) -> Proposal:
        ctx = self.ctx
        assert ctx is not None
        prim = next((ctx.failed_primitive[k] for k in keys if ctx.failed_primitive.get(k)), "push")
        defn = ctx.library.get(prim)
        ranges = dict(defn.params) if defn is not None else {}
        # Hand System 1 the whole declared range (inside the hard limits) instead of the
        # posture-bounded neighbourhood of the current value; force and extent params first.
        order = sorted(ranges, key=lambda n: (0 if "force" in n else 1 if n in ("depth", "distance", "angle") else 2, n))
        wide = {}
        for name in order:
            lo, hi = ranges[name]
            if "force" in name:
                hi = min(hi, ctx.limits.max_force_n)
            if "speed" in name:
                hi = min(hi, ctx.limits.max_tip_speed_mps)
            wide[f"{prim}.{name}"] = (float(lo), float(hi))
        if not wide:
            wide = {"push.force": (0.5, float(ctx.limits.max_force_n))}
        return Proposal(kind="reparam", rung=0, rationale=f"{why} ({prim})",
                        hypothesis=f"{prim} needs parameters outside the neighbourhood System 1 searched",
                        reparam=dict(list(wide.items())[: max(1, ctx.eff.max_edits)]),
                        prediction=Prediction(flips_to_pass=keys, confidence=0.3))


class RecordingLLM:
    """Wraps any ``LLM`` and writes each answer as a ``MockLLM`` fixture for its prompt."""

    def __init__(self, inner: LLM, record_dir: Path):
        self.inner = inner
        self.record_dir = Path(record_dir)

    def __getattr__(self, name: str) -> Any:  # log_path, total_usd, calls, bind ...
        return getattr(self.inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("inner", "record_dir"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def propose(self, prompt: S2Prompt) -> list[Proposal]:
        out = list(self.inner.propose(prompt))
        write_fixture(prompt, out, self.record_dir)
        return out


# ---------------------------------------------------------------- body-search baselines
class _BodySearch:
    """Shared plumbing: a ``DesignSpace`` vector -> a whole-rack ``new_tool`` proposal."""

    name = "body"

    def __init__(self, levels: Sequence[str], rack_slots: int, limits: HardLimits, seed: int = 0):
        self.space = DesignSpace([lv for lv in levels if lv == "L0"] or ["L0"], rack_slots, limits)
        self.rng = np.random.default_rng(seed)
        self.pending: dict[int, np.ndarray] = {}  # id(proposal) -> vector, until observed

    def _proposal(self, ctx: ProposeContext, vec: np.ndarray) -> Proposal | None:
        cand = self.space.decode(vec, ctx.design, new_id="cand")
        if not cand.tool_rack:
            return None
        delta = DesignDelta(rack_add=list(cand.tool_rack), rack_remove=list(range(len(ctx.design.tool_rack))))
        p = Proposal(kind="new_tool", rung=3, rationale=f"{self.name} body search",
                     hypothesis="a different rack may cover the failing controls",
                     ruled_out=ruled_out_for(3, f"{self.name} baseline searches the body only"),
                     design_delta=delta, prediction=Prediction(flips_to_pass=ctx.target_keys, confidence=0.5))
        self.pending[id(p)] = vec
        return p

    def _vec_of(self, proposal: Proposal) -> np.ndarray | None:
        return self.pending.pop(id(proposal), None)


class RandomProposer(_BodySearch):
    name = "random"

    def propose(self, ctx: ProposeContext) -> list[Proposal]:
        for _ in range(20):
            p = self._proposal(ctx, self.space.sample(self.rng))
            if p is not None:
                return [p]
        return []

    def observe(self, proposal: Proposal, decision: Decision | None) -> None:
        return None


class CMAProposer(_BodySearch):
    """CMA-ES over the design-space vector; one candidate per iteration, told the trial delta."""

    name = "cma"

    def __init__(self, levels: Sequence[str], rack_slots: int, limits: HardLimits, seed: int = 0):
        super().__init__(levels, rack_slots, limits, seed)
        import cma

        self.es = cma.CMAEvolutionStrategy([0.5] * self.space.dim, 0.3,
                                           {"bounds": [0.0, 1.0], "seed": seed + 1, "verbose": -9})
        self._asked: list[np.ndarray] = []
        self._told: list[tuple[np.ndarray, float]] = []

    def propose(self, ctx: ProposeContext) -> list[Proposal]:
        for _ in range(20):
            if not self._asked:
                self._asked = [np.asarray(x) for x in self.es.ask()]
            vec = self._asked.pop(0)
            p = self._proposal(ctx, vec)
            if p is not None:
                return [p]
            self._told.append((vec, 1.0))  # empty rack: worst score
            self._flush()
        return []

    def _flush(self) -> None:
        if len(self._told) >= self.es.popsize:
            xs, fs = zip(*self._told[: self.es.popsize], strict=True)
            self.es.tell(list(xs), list(fs))
            self._told = self._told[self.es.popsize:]

    def observe(self, proposal: Proposal, decision: Decision | None) -> None:
        vec = self._vec_of(proposal)
        if vec is None:
            return
        self._told.append((vec, -(decision.delta_mean if decision is not None else -1.0)))
        self._flush()


class BOProposer(_BodySearch):
    """Bayesian optimization (Optuna TPE, seeded) over the design-space vector."""

    name = "bo"

    def __init__(self, levels: Sequence[str], rack_slots: int, limits: HardLimits, seed: int = 0):
        super().__init__(levels, rack_slots, limits, seed)
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.trials: dict[int, Any] = {}

    def propose(self, ctx: ProposeContext) -> list[Proposal]:
        for _ in range(20):
            trial = self.study.ask()
            vec = np.array([trial.suggest_float(n, 0.0, 1.0) for n in self.space.names()])
            p = self._proposal(ctx, vec)
            if p is not None:
                self.trials[id(p)] = trial
                return [p]
            self.study.tell(trial, -1.0)
        return []

    def observe(self, proposal: Proposal, decision: Decision | None) -> None:
        trial = self.trials.pop(id(proposal), None)
        self.pending.pop(id(proposal), None)
        if trial is not None:
            self.study.tell(trial, decision.delta_mean if decision is not None else -1.0)


class S1OnlyProposer:
    """Re-parameterization only: widen the failing primitive's ranges (rung 0)."""

    name = "s1_only"

    def __init__(self) -> None:
        self._scripted = ScriptedLLM()

    def propose(self, ctx: ProposeContext) -> list[Proposal]:
        self._scripted.bind(ctx)
        return [self._scripted._reparam(g.keys, "System 1 only") for g in ctx.targets][:1]

    def observe(self, proposal: Proposal, decision: Decision | None) -> None:
        return None


def make_proposer(kind: str, *, llm: LLM | None, model: str, levels: Sequence[str], rack_slots: int,
                  limits: HardLimits, seed: int) -> Proposer:
    if kind in ("s2", "s2_nomem"):
        assert llm is not None, "System 2 needs an LLM"
        return S2Proposer(llm, model, use_memory=kind == "s2")
    if kind == "random":
        return RandomProposer(levels, rack_slots, limits, seed)
    if kind == "cma":
        return CMAProposer(levels, rack_slots, limits, seed)
    if kind == "bo":
        return BOProposer(levels, rack_slots, limits, seed)
    if kind == "s1_only":
        return S1OnlyProposer()
    raise ValueError(f"unknown proposer {kind!r}")


__all__ = [
    "BOProposer",
    "CMAProposer",
    "ProposeContext",
    "Proposer",
    "RandomProposer",
    "RecordingLLM",
    "S1OnlyProposer",
    "S2Proposer",
    "ScriptedLLM",
    "make_proposer",
    "ruled_out_for",
]
