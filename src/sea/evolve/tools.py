"""System 2's questions to the harness (D-086): a read-only MCP tool server, ``sea``.

While it writes an iteration's proposals, System 2 can ask what the harness would say about a draft:

- ``check``: whether the loop would trial it, or every reason it would refuse it, word for word;
- ``measure``: the body's computed facts (the world's brief, its cost, and for the gripper the wrist roll its grasps
  need and whether a built mechanism's loops close);
- ``module``: a module's graph at given parameters, to patch against;
- ``simulate``: one train task solved by the draft body.  One task is not a trial, and nothing it finds is kept.

Every answer is the harness's own computation on the same path the loop takes (``loop.candidate``,
``loop.screen``); acceptance stays with the paid trial.  The budget is in the context file the loop writes, never
in a tool's arguments, and every call is appended to the run's ``tool_calls.jsonl``.

The server speaks the subset of MCP that ``claude -p`` uses: newline-delimited JSON-RPC 2.0 over stdio with
``initialize``, ``tools/list`` and ``tools/call``.

    python -m sea.evolve.tools --context runs/<id>/tools/iter03.json --log runs/<id>/tool_calls.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from sea.evolve.concepts import ConceptSet
from sea.evolve.core import Body, Proposal, Requirement, World
from sea.evolve.resources import Spent

NAMES = ("check", "measure", "read", "sweep", "sensitivity", "compare", "facts", "module", "simulate", "note")
CALLS, SIMS = 40, 3  # per iteration
SWEEP_VALUES, SWEEP_READINGS = 12, 6
SENSITIVITY_PARAMS = 12
COMPARE_DRAFTS, DRAFTS_BEFORE_SIMULATE = 3, 2
NOTES_BEFORE_SIMULATE = 2  # in a run's first iteration: an engineer reads before building (D-099)


class Context(BaseModel):
    """What the loop knows when it asks System 2: the bodies a draft may build on and the record it is checked
    against."""

    model_config = ConfigDict(extra="forbid")
    world: str
    iteration: int
    body: Body
    bases: dict[str, Body] = {}  # promising bodies by id
    tried: dict[str, str] = {}  # body key -> what its trial found
    manifest: dict[str, Any]
    spent: Spent
    train: list[str]
    calls: int = CALLS
    sims: int = SIMS
    requirements: list[Requirement] = []  # System 2's list (D-089)
    held: list[str] = []
    notebook: str = ""  # the engineering notebook (D-090), with this run's research log and name
    research_log: str = ""
    run: str = ""
    library: str = ""  # the skills library (D-096)
    concepts: ConceptSet | None = None  # the concept set (D-091), when the run keeps one


class ToolError(Exception):
    pass


def _schema() -> tuple[dict[str, Any], dict[str, Any]]:
    from sea.propose.llm import cli_schema

    s = cli_schema(Proposal.model_json_schema())
    return {k: v for k, v in s.items() if k != "$defs"}, s.get("$defs", {})


def specs() -> list[dict[str, Any]]:
    prop, defs = _schema()
    draft = {"proposal": {**prop, "description": "a draft proposal, as you would answer it"}}

    def tool(name: str, doc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"name": name, "description": doc, "inputSchema": {"type": "object", "properties": props,
                                                                  "required": required, "$defs": defs}}

    return [
        tool("check", "Would the loop trial this proposal?  Returns 'would trial' with its cost, or every reason it "
             "would be refused before its trial (the build rules, the mechanism laws, the resources, the record), as "
             "the loop words them.", draft, ["proposal"]),
        tool("measure", "The computed facts of the body this proposal makes: the world's brief of it (for the "
             "gripper: by object width, the servo angle, the faces' tilt, the fingertips, the squeeze and any rods' "
             "loads), its cost, and for the gripper the wrist roll each task's grasp needs and the range the "
             "simulator holds the wrist to; for a built mechanism, how far its loops are from closing.", draft,
             ["proposal"]),
        tool("read", "One instrument read on the body this proposal makes (the instruments section lists them; "
             "a per-task one needs a train task id).", {**draft, "instrument": {"type": "string"},
                                                       "args": {"type": "object"}, "task": {"type": "string"}},
             ["proposal", "instrument"]),
        tool("sweep", "Instruments read as one parameter of one slot of the body this proposal makes takes each of "
             f"up to {SWEEP_VALUES} values ({SWEEP_READINGS} readings at most): how the readings depend on it.",
             {**draft, "slot": {"type": "string"}, "param": {"type": "string"},
              "values": {"type": "array", "items": {"type": "number"}},
              "readings": {"type": "array", "items": {"type": "object", "properties": {
                  "instrument": {"type": "string"}, "args": {"type": "object"}, "task": {"type": "string"}},
                  "required": ["instrument"]}}}, ["proposal", "slot", "param", "values", "readings"]),
        tool("sensitivity", "How each of your requirements moves when each parameter of the drafted body's parts "
             "moves a tenth of its range up and down, and which parameters help one requirement while hurting "
             "another (a coupling).", draft, ["proposal"]),
        tool("compare", f"Up to {COMPARE_DRAFTS} drafts against the current body and each other on every requirement, "
             "each reading marked + better, 0 the same or - worse than the current body.",
             {"proposals": {"type": "array", "items": prop}}, ["proposals"]),
        tool("facts", "A train task's numbers by name (an object's size and mass, a route's features, a target).",
             {"task": {"type": "string"}}, ["task"]),
        tool("note", "Record a finding from your research in the engineering notebook, with the page you read it "
             "on: it outlives this run, and a kept change that cites its id credits it.  Returns its id.",
             {k: v for k, v in _note_schema().items()}, ["claim", "source_url"]),
        tool("module", "A module's graph at the given parameters (lengths in metres), or a concept's own graph by its "
             "id, to write a patch against.",
             {"name": {"type": "string"}, "params": {"type": "object", "additionalProperties": {"type": "number"}}},
             ["name"]),
        tool("simulate", "One train task solved by the body this proposal makes: whether it succeeded, the failure "
             "mechanism and the diagnosis.  One task is not a trial: nothing it finds is kept, and the budget is "
             "small.  A draft that opens a new concept may simulate once two distinct new drafts have been checked "
             "or measured this iteration.", {**draft, "task": {"type": "string", "description": "a train task id"}}, ["proposal", "task"]),
    ]


class Server:
    def __init__(self, ctx: Context, log: Path | None, world: World | None = None):
        self.ctx, self.log = ctx, log
        self._world = world
        self.drafts: list[str] = []  # the new concepts' drafts this call looked at

    @property
    def world(self) -> World:
        """Loaded at the first call, so the handshake does not wait on it."""
        if self._world is None:
            from sea.evolve.loop import make_world

            self._world = make_world(self.ctx.world)
            if self.ctx.library:
                from sea.evolve.library import promoted

                self._world.promoted = promoted(Path(self.ctx.library))
        return self._world

    @property
    def catalogue(self) -> dict[str, Any]:
        return {p.name: p for p in self.world.catalogue()}

    @property
    def tasks(self) -> dict[str, Any]:
        return {t.id: t for t in self.world.tasks()}

    # ---------------------------------------------------------------- budget and log
    def _used(self) -> tuple[int, int]:
        n = s = 0
        if self.log is not None and self.log.exists():
            for line in self.log.read_text().splitlines():
                r = json.loads(line)
                if r.get("iteration") == self.ctx.iteration and r.get("counted"):
                    n += 1
                    s += r["tool"] == "simulate" and not r["error"]  # a simulation refused ran nothing
        return n, s

    def _record(self, tool: str, args: dict[str, Any], result: str, seconds: float, counted: bool,
                sim_s: float = 0.0, error: bool = False) -> None:
        if self.log is None:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as f:
            f.write(json.dumps({"iteration": self.ctx.iteration, "tool": tool, "args": args, "result": result[:2000],
                                "error": error, "seconds": round(seconds, 2), "sim_s": round(sim_s, 1),
                                "counted": counted, **({"drafts": self.drafts} if self.drafts else {})}) + "\n")

    def _drafted(self) -> set[str]:
        """The new concepts' drafts looked at this iteration, by body, simulations aside."""
        if self.log is None or not self.log.exists():
            return set()
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        return {k for r in rows if r.get("iteration") == self.ctx.iteration and r["tool"] != "simulate"
                for k in r.get("drafts", [])}

    def call(self, tool: str, args: dict[str, Any]) -> tuple[str, bool]:
        """The answer's text, and whether it is an error."""
        t0 = time.time()
        self.drafts = []
        n, s = self._used()
        if tool not in NAMES:
            return f"no tool {tool!r}; the tools are {list(NAMES)}", True
        if n >= self.ctx.calls:
            text = f"this iteration's {self.ctx.calls} tool calls are spent: answer with your proposals"
            self._record(tool, args, text, time.time() - t0, counted=False, error=True)
            return text, True
        if tool == "simulate" and s >= self.ctx.sims:
            text = f"this iteration's {self.ctx.sims} simulations are spent"
            self._record(tool, args, text, time.time() - t0, counted=False, error=True)
            return text, True
        self.sim_s, error = 0.0, False
        try:
            text = getattr(self, f"_{tool}")(args)
        except ToolError as exc:
            text, error = str(exc), True
        self._record(tool, args, text, time.time() - t0, counted=True, sim_s=self.sim_s, error=error)
        return text, error

    # ---------------------------------------------------------------- the tools
    def _draft(self, args: dict[str, Any]) -> tuple[Proposal, Body]:
        from sea.evolve.loop import candidate, recorded_only

        try:
            prop = Proposal.model_validate(args.get("proposal"))
        except ValidationError as exc:
            raise ToolError(f"not a valid proposal: {str(exc)[:600]}") from exc
        if recorded_only(self.world, prop):
            raise ToolError("this idea has no mechanism, so the loop records it and builds nothing to check")
        conc = self.ctx.concepts
        try:
            gone = conc.base_refusal(prop.base) if conc is not None else ""
            if gone:
                raise ValueError(gone)
            cand = candidate(self.world, self.ctx.body, self.ctx.bases, prop, self.catalogue, new_id="draft")
            if conc is not None and self._opens(prop, cand):
                full = conc.full_refusal()
                if full:
                    raise ValueError(full)
                self.drafts.append(cand.key())
        except ValueError as exc:
            raise ToolError(f"refused before trial: {exc}") from exc
        return prop, cand

    def _opens(self, prop: Proposal, cand: Body) -> bool:
        from sea.evolve.loop import opens_concept

        return opens_concept(self.world, self.ctx.concepts, self.ctx.body, self.ctx.bases, prop, cand)

    def _check(self, args: dict[str, Any]) -> str:
        from sea.evolve.loop import screen

        _, cand = self._draft(args)
        errs, res = screen(self.world, cand, self.ctx.body, self.ctx.manifest, self.ctx.spent, self.ctx.tried)
        if errs:
            return "refused before trial: " + "; ".join(errs)
        return f"would trial: costs ${res.usd:.2f}, {res.fab_minutes:.0f} machine min, {res.lead_days:g} days' wait"

    def _measure(self, args: dict[str, Any]) -> str:
        from sea.evolve.resources import check

        _, cand = self._draft(args)
        train = [self.tasks[i] for i in self.ctx.train]
        try:
            out = [self.world.brief(cand, train)]
        except (ValueError, ZeroDivisionError) as exc:
            raise ToolError(f"cannot measure this body: {exc}") from exc
        res = check(cand, self.world.bom(cand), self.ctx.body, self.world.bom(self.ctx.body), self.ctx.manifest,
                    self.ctx.spent)
        out.append(f"cost: ${res.usd:.2f}, {res.fab_minutes:.0f} machine min, {res.lead_days:g} days' wait; "
                   + "; ".join(res.lines))
        extra = EXTRA.get(self.ctx.world)
        if extra is not None:
            out += extra(cand, train)
        if self.ctx.requirements:
            from sea.evolve import requirements as R

            reqs, held, groups = self.ctx.requirements, set(self.ctx.held), sorted({t.group for t in train})
            cur = R.measure(self.world, self.ctx.body, reqs, train, self.ctx.body, self.ctx.manifest, self.ctx.spent)
            new = R.measure(self.world, cand, reqs, train, self.ctx.body, self.ctx.manifest, self.ctx.spent)
            out.append("requirements, now and on the draft:\n" + R.text(reqs, cur, held, groups, draft=new))
        return "\n".join(out)

    def _sensitivity(self, args: dict[str, Any]) -> str:
        from sea.evolve import requirements as R
        from sea.evolve.core import Edit, apply

        reqs = self.ctx.requirements
        if not reqs:
            raise ToolError("there are no requirements yet: write them in your answer's `requirements`")
        _, cand = self._draft(args)
        train = [self.tasks[i] for i in self.ctx.train]
        here = R.measure(self.world, cand, reqs, train, self.ctx.body, self.ctx.manifest, self.ctx.spent)
        space = [(slot, k, p) for slot, part in sorted(cand.parts.items()) if part.type in self.catalogue
                 for k, p in sorted(self.catalogue[part.type].params.items())][:SENSITIVITY_PARAMS]
        if not space:
            return "the drafted body's parts have no parameters"
        rows, coupled = [], []
        for slot, k, p in space:
            v0 = cand.parts[slot].params[k]
            step = (p.hi - p.lo) / 10
            cells = []
            for sign in (1, -1):
                v = min(max(v0 + sign * step, p.lo), p.hi)
                if v == v0:
                    continue
                b = apply(cand, [Edit(slot=slot, params={k: v})], self.catalogue, new_id="sens")
                moved = R.measure(self.world, b, reqs, train, self.ctx.body, self.ctx.manifest, self.ctx.spent)
                helps, hurts = [], []
                for r, a, m in zip(reqs, here, moved, strict=True):
                    if a.value is None or m.value is None or abs(m.value - a.value) < 1e-9:
                        continue
                    better = (m.value > a.value) == (r.sense == ">=")
                    (helps if better else hurts).append(r.id)
                    cells.append(f"{k} {v:g}: {r.id} {a.value:.4g} -> {m.value:.4g} ({'better' if better else 'worse'})")
                if helps and hurts:  # one move, two requirements pulled apart: a coupling
                    coupled.append(f"{slot}.{k} {'up' if sign > 0 else 'down'} helps {', '.join(helps)} and hurts "
                                   f"{', '.join(hurts)}")
            rows.append(f"{slot}.{k} = {v0:g}: " + ("; ".join(cells) or "moves no requirement"))
        return "\n".join(rows + (["couplings: " + "; ".join(coupled)] if coupled else ["no coupling found"]))

    def _compare(self, args: dict[str, Any]) -> str:
        from sea.evolve import concepts as C
        from sea.evolve import requirements as R

        reqs = self.ctx.requirements
        if not reqs:
            raise ToolError("there are no requirements yet: write them in your answer's `requirements`")
        train = [self.tasks[i] for i in self.ctx.train]

        def rows(b: Body) -> list[R.Row]:
            return R.measure(self.world, b, reqs, train, self.ctx.body, self.ctx.manifest, self.ctx.spent)

        columns, out = [], []
        for n, raw in enumerate(list(args.get("proposals") or [])[:COMPARE_DRAFTS]):
            try:
                _, cand = self._draft({"proposal": raw})
            except ToolError as exc:
                out.append(f"draft {n + 1}: {exc}")
                continue
            columns.append((f"draft {n + 1}", rows(cand)))
        if not columns and not out:
            raise ToolError("give the drafts to compare")
        return "\n".join([C.matrix(reqs, rows(self.ctx.body), columns), *out] if columns else out)

    def _reading(self, body: Body, r: dict[str, Any]) -> float:
        from sea.evolve import instruments as I

        task = r.get("task")
        if task is not None and task not in self.ctx.train:
            raise ToolError(f"{task!r} is not a train task")
        try:
            return I.read(self.world, body, str(r.get("instrument", "")), dict(r.get("args") or {}),
                          self.tasks[task] if task else None, base=self.ctx.body, manifest=self.ctx.manifest,
                          spent=self.ctx.spent)
        except (ValueError, KeyError, ZeroDivisionError) as exc:
            raise ToolError(f"cannot read {r.get('instrument')!r}: {exc}") from exc

    def _read(self, args: dict[str, Any]) -> str:
        from sea.evolve import instruments as I

        _, cand = self._draft(args)
        v = self._reading(cand, args)
        unit = next((i.unit for i in I.instruments(self.world) if i.name == args.get("instrument")), "")
        return f"{args.get('instrument')}{_argtext(args)} = {v:.4g} {unit}".rstrip()

    def _sweep(self, args: dict[str, Any]) -> str:
        from sea.evolve.core import Edit, apply

        _, cand = self._draft(args)
        values = list(args.get("values") or [])[:SWEEP_VALUES]
        readings = list(args.get("readings") or [])[:SWEEP_READINGS]
        slot, param = str(args.get("slot", "")), str(args.get("param", ""))
        if not values or not readings:
            raise ToolError("give the values and the readings to take at each")
        rows = []
        for v in values:
            try:
                b = apply(cand, [Edit(slot=slot, params={param: float(v)})], self.catalogue, new_id="sweep")
            except ValueError as exc:
                rows.append(f"{param} = {v:g}: {exc}")
                continue
            cells = []
            for r in readings:
                try:
                    cells.append(f"{r['instrument']}{_argtext(r)} {self._reading(b, r):.4g}")
                except ToolError as exc:
                    cells.append(str(exc))
            rows.append(f"{param} = {v:g}: " + "; ".join(cells))
        return f"{slot}.{param} swept on the drafted body:\n" + "\n".join(rows)

    def _facts(self, args: dict[str, Any]) -> str:
        from sea.evolve import instruments as I

        task = str(args.get("task", ""))
        if task not in self.ctx.train:
            raise ToolError(f"{task!r} is not a train task")
        t = self.tasks[task]
        return f"{task} ({t.group}): {t.facts}\n" + ", ".join(f"{k} = {v:g}" for k, v in I.facts(self.world, t).items())

    def _note(self, args: dict[str, Any]) -> str:
        from sea.evolve import research as RS

        if not self.ctx.notebook:
            raise ToolError("research is off in this run: there is no notebook")
        try:
            note = RS.Note.model_validate({k: v for k, v in args.items() if k in _note_schema()})
            kept = RS.add(Path(self.ctx.notebook), note, RS.read_urls(Path(self.ctx.research_log)), self.ctx.world,
                          self.ctx.run, self.ctx.iteration)
        except ValueError as exc:
            raise ToolError(f"not recorded: {str(exc)[:400]}") from exc
        return f"recorded as {kept.id}"

    def _module(self, args: dict[str, Any]) -> str:
        if not hasattr(self.world, "modules_text"):
            raise ToolError("this world has no modules")
        if args.get("name") in self.ctx.bases and hasattr(self.world, "mechanism_text"):  # a concept's own graph (D-091)
            return self.world.mechanism_text(self.ctx.bases[args["name"]]) or f"{args['name']} has no built mechanism"
        if hasattr(self.world, "module_text"):  # a world that keeps its own modules (D-094)
            try:
                return self.world.module_text(str(args.get("name", "")), dict(args.get("params") or {}))
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        from sea.worlds import gripper as W

        name, params = args.get("name", ""), dict(args.get("params") or {})
        if name not in W.MODULES:
            raise ToolError(f"no module {name!r}; the modules are {sorted(W.MODULES)}")
        ranges = W.MODULES[name][1]
        bad = [f"{k} {v:g} outside {ranges[k][0]:g}..{ranges[k][1]:g}" if k in ranges else f"no parameter {k!r}"
               for k, v in params.items() if k not in ranges or not ranges[k][0] <= v <= ranges[k][1]]
        if bad:
            raise ToolError("; ".join(bad))
        params = {k: params.get(k, (lo + hi) / 2) for k, (lo, hi) in ranges.items()}
        return f"{name} at {params or 'its only setting'}:\n" + W._compact(W._module(name, params))

    def _research_first(self) -> None:
        """Refuse a first iteration's simulation until the run has written down what it read, or spent its searches."""
        from sea.evolve import research as RS

        if not self.ctx.notebook or self.ctx.iteration != 0:
            return
        ours = [n for n in RS.load(Path(self.ctx.notebook)) if n.run == self.ctx.run]
        spent = RS.summary(Path(self.ctx.research_log), 0)["calls"] >= RS.BUDGET
        if len(ours) < NOTES_BEFORE_SIMULATE and not spent:
            raise ToolError(f"research first: this run has recorded {len(ours)} of {NOTES_BEFORE_SIMULATE} notes.  Search "
                            "and read what the failures need you to know, and record each finding with `note`")

    def _simulate(self, args: dict[str, Any]) -> str:
        task = str(args.get("task", ""))
        if task not in self.ctx.train:
            raise ToolError(f"{task!r} is not a train task; the train tasks are {self.ctx.train}")
        from sea.evolve.loop import screen

        self._research_first()
        _, cand = self._draft(args)
        if self.drafts and len(self._drafted()) < DRAFTS_BEFORE_SIMULATE:
            raise ToolError(f"this draft opens a new concept: check or measure {DRAFTS_BEFORE_SIMULATE} distinct new "
                            "drafts this iteration before simulating one")
        errs, _ = screen(self.world, cand, self.ctx.body, self.ctx.manifest, self.ctx.spent, {})
        if errs:
            raise ToolError("the loop would not build this body to trial it: " + "; ".join(errs))
        o = self.world.solve(cand, self.tasks[task], 0)
        self.sim_s = o.sim_s
        return (f"{task}: {'succeeded' if o.success else 'failed'} (score {o.score:.2f})"
                + (f"; {o.mechanism}: {o.detail[:800]}" if not o.success else "")
                + (f"; first attempt scored {o.nominal:.2f}" if o.nominal is not None and o.nominal != o.score else "")
                + (f"; states: {', '.join(f'{k} {v:.3g}' for k, v in o.states.items())}" if o.states else "")
                + " -- one task, not a trial")

    # ---------------------------------------------------------------- JSON-RPC
    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        mid, method = msg.get("id"), msg.get("method", "")
        if mid is None:
            return None  # a notification
        if method == "initialize":
            result: dict[str, Any] = {"protocolVersion": (msg.get("params") or {}).get("protocolVersion", "2025-06-18"),
                                      "capabilities": {"tools": {}}, "serverInfo": {"name": "sea", "version": "1"}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": specs()}
        elif method == "tools/call":
            p = msg.get("params") or {}
            text, error = self.call(p.get("name", ""), p.get("arguments") or {})
            result = {"content": [{"type": "text", "text": text}], "isError": error}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method!r}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, rx: IO[str], tx: IO[str]) -> None:
        for line in rx:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                out: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None,
                                              "error": {"code": -32700, "message": "not JSON"}}
            else:
                out = self.handle(msg)
            if out is not None:
                tx.write(json.dumps(out) + "\n")
                tx.flush()


def _note_schema() -> dict[str, Any]:
    from sea.evolve.research import Note

    props = Note.model_json_schema()["properties"]
    return {k: v for k, v in props.items() if k in ("claim", "source_url", "quote", "numbers", "validity", "applies_to")}


def _argtext(r: dict[str, Any]) -> str:
    a = dict(r.get("args") or {})
    inner = ", ".join(f"{k}={v}" for k, v in a.items())
    return f"({inner})" + (f" on {r['task']}" if r.get("task") else "")


def summary(log: Path, iteration: int | None = None) -> dict[str, Any]:
    """The calls in ``log`` (one iteration's, if given): how many of each tool, refused for the budget, and the
    time and sim-seconds they took."""
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    rows = [r for r in rows if iteration is None or r["iteration"] == iteration]
    by: dict[str, int] = {}
    for r in rows:
        if r["counted"]:
            by[r["tool"]] = by.get(r["tool"], 0) + 1
    return {"calls": sum(by.values()), "by_tool": by, "over_budget": sum(not r["counted"] for r in rows),
            "errors": sum(r["error"] for r in rows if r["counted"]),
            "seconds": round(sum(r["seconds"] for r in rows), 1), "sim_s": round(sum(r["sim_s"] for r in rows), 1)}


# ---------------------------------------------------------------- per-world facts beyond the brief
def _gripper(body: Body, train: list[Any]) -> list[str]:
    """The wrist roll each train task's grasp needs, from the IK at the poses the grasp commands (D-085), and the
    range the simulator holds the wrist to; a built mechanism's worst open loop."""
    from sea.worlds import gripper as W

    f = W.to_design(body).fingers
    lo, hi = W.roll_range(f)
    j_lo, j_hi = W.ik_joint_range("wrist_roll")
    ids = {t.id for t in train}
    need = {k: v for k, v in W.roll_needed(f).items() if k in ids}
    if not need:
        return []
    a, b = min(need, key=lambda k: need[k][0]), max(need, key=lambda k: need[k][1])
    rows = [f"{k} {x:+.2f}" if y - x < 0.01 else f"{k} {x:+.2f}..{y:+.2f}" for k, (x, y) in need.items()]
    out = [f"wrist roll: the simulator holds the wrist to {lo:+.2f}..{hi:+.2f} rad (the joint allows "
           f"{j_lo:+.2f}..{j_hi:+.2f}); the grasps need {need[a][0]:+.2f} ({a}) to {need[b][1]:+.2f} ({b}), above the "
           "object, at the grip and lifted; by task: " + ", ".join(rows)]
    if body.parts["jaw"].type == "built":
        from sea.mech.graph import Graph
        from sea.mech.kinematics import closes

        gap, q = closes(Graph.model_validate_json(body.parts["jaw"].mechanism))
        out.append(f"loops: the worst is {gap * 1000:.3f} mm (or mrad) open, at servo {q:.2f} rad")
    return out


EXTRA = {"gripper": _gripper}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--log", type=Path)
    a = ap.parse_args(argv)
    tx, sys.stdout = sys.stdout, sys.stderr  # stdout carries the protocol alone: anything else printed goes to stderr
    Server(Context.model_validate_json(a.context.read_text()), a.log).serve(sys.stdin, tx)


if __name__ == "__main__":
    main()
