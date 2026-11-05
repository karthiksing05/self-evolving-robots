"""System 2's questions to the harness (D-086): the ``sea`` tool server answers what the loop would, inside a budget
the arguments cannot change, over the protocol ``claude -p`` speaks."""

from __future__ import annotations

import json
import subprocess
import sys

from sea.evolve.core import Edit, Idea, Mechanism, ProposalBatch
from sea.evolve.llm import PlanClaude, Scripted
from sea.evolve.loop import RunConfig, run
from sea.evolve.resources import start
from sea.evolve.tools import Context, Server, summary
from sea.propose.llm import CLAUDE_FLAGS
from tests.test_evolve import Builds, _p


def _ctx(w, **kw) -> Context:
    body = w.stock()
    return Context(world="toy", iteration=0, body=body, manifest=w.manifest(), spent=start(w.bom(body)),
                   train=[t.id for t in w.tasks() if t.split == "train"], **kw)


ROD = Idea(name="long rod", description="a 0.5 m rod", slot="stick", mechanism=Mechanism(module="rod",
                                                                                      params={"length": 0.5}))
PROPS = [_p("idea", idea=ROD), _p("idea", [Edit(slot="sensor", type="lidar")], idea=ROD),
         _p("idea", [Edit(slot="stick", params={"length": 0.2})], idea=ROD), _p("params", [])]


def test_check_says_what_the_loop_says(tmp_path):
    """Each draft's check is the loop's own screen of it, word for word: the same refusals, before any trial."""
    w = Builds()
    run(RunConfig(world="toy", iterations=1, max_proposals=4, merge=False), tmp_path,
        proposer=Scripted([ProposalBatch.model_construct(proposals=PROPS, notes="")]), world=w, log=lambda m: None)
    trials = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])["trials"]
    s = Server(_ctx(w), None, w)
    got = [s.call("check", {"proposal": p.model_dump()})[0] for p in PROPS]
    assert got[0].startswith("would trial") and trials[0]["tried"]
    for k in (1, 2, 3):
        assert got[k] == trials[k]["result"], k
    assert "cannot be bought" in got[1] and "ambiguous" in got[2] and "unchanged" in got[3]


def test_the_budget_is_the_context_s_and_every_call_is_logged(tmp_path):
    w, log = Builds(), tmp_path / "tool_calls.jsonl"
    s = Server(_ctx(w, calls=3, sims=1), log, w)
    draft = PROPS[0].model_dump()
    text, err = s.call("simulate", {"proposal": draft, "task": "d0.45_test0"})
    assert err and "not a train task" in text  # the test split stays unseen
    text, err = s.call("simulate", {"proposal": draft, "task": "d0.45_train0", "calls": 99, "sims": 99})
    assert not err and text.startswith("d0.45_train0: succeeded") and "not a trial" in text
    assert "simulations are spent" in s.call("simulate", {"proposal": draft, "task": "d0.15_train0"})[0]
    assert s.call("measure", {"proposal": draft}) == ("reach 0.5 m\ncost: $0.00, 50 machine min, 0 days' wait; "
                                                     + "; ".join(_lines(w, s)), False)
    assert "calls are spent" in s.call("check", {"proposal": draft})[0]
    assert Server(_ctx(w, calls=3, sims=1), log, w).call("check", {"proposal": draft})[1]  # a new process, same count
    got = summary(log, 0)
    assert got["calls"] == 3 and got["by_tool"] == {"simulate": 2, "measure": 1} and got["over_budget"] == 3
    assert s.call("check", {"proposal": {"move": "params"}})[1]  # a draft that is not a proposal is an error


def _lines(w, s):
    from sea.evolve.loop import candidate
    from sea.evolve.resources import check

    cand = candidate(w, s.ctx.body, {}, PROPS[0], s.catalogue, "draft")
    return check(cand, w.bom(cand), s.ctx.body, w.bom(s.ctx.body), s.ctx.manifest, s.ctx.spent).lines


def test_the_server_speaks_mcp_over_stdio(tmp_path):
    """The handshake ``claude -p`` makes, an unknown probe, the tool list and one call (the gripper's modules)."""
    from sea.worlds.gripper import GripperWorld

    w = GripperWorld()
    ctx = tmp_path / "ctx.json"
    ctx.write_text(_ctx(w).model_copy(update={"world": "gripper"}).model_dump_json())
    msgs = [{"jsonrpc": "2.0", "id": "probe", "method": "server/discover"},
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "module", "arguments": {"name": "parallelogram", "params": {"link_mm": 60}}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "module", "arguments": {"name": "parallelogram", "params": {"link_mm": 200}}}}]
    out = subprocess.run([sys.executable, "-m", "sea.evolve.tools", "--context", str(ctx), "--log",
                          str(tmp_path / "log.jsonl")], input="".join(json.dumps(m) + "\n" for m in msgs),
                         capture_output=True, text=True, timeout=300, check=True)
    got = {r["id"]: r for r in map(json.loads, out.stdout.splitlines())}
    assert set(got) == {"probe", 0, 1, 2, 3}  # the notification is not answered
    assert got["probe"]["error"]["code"] == -32601
    assert got[0]["result"]["protocolVersion"] == "2025-11-25"
    assert [t["name"] for t in got[1]["result"]["tools"]] == ["check", "measure", "read", "sweep", "sensitivity", "compare",
                                                              "facts", "note", "module", "simulate"]
    assert all("$defs" in t["inputSchema"] for t in got[1]["result"]["tools"])
    assert not got[2]["result"]["isError"] and '"name":"parallelogram(60)"' in got[2]["result"]["content"][0]["text"]
    assert got[3]["result"]["isError"] and "outside 50..90" in got[3]["result"]["content"][0]["text"]


def test_measure_reads_a_built_gripper_jaw_s_wrist_and_loops():
    from sea.worlds.gripper import GripperWorld

    w = GripperWorld()
    s = Server(_ctx(w).model_copy(update={"world": "gripper"}), None, w)
    built = _p("idea", idea=Idea(name="plain", description="the stock jaw as a graph", slot="jaw",
                                 mechanism=Mechanism(module="revolute")))
    text, err = s.call("measure", {"proposal": built.model_dump()})
    assert not err and "built mechanism 'plain'" in text
    assert "wrist roll: the simulator holds the wrist to -2.31..+0.20 rad" in text  # D-085's stock range, D-102's ways
    assert "loops: the worst is 0.000" in text


def test_plan_claude_runs_the_tools_only_when_given_them(tmp_path):
    """Without tools the command is today's, flag for flag; with them, only the ``sea`` server and its tools."""
    p = PlanClaude()
    assert p.flags() == list(CLAUDE_FLAGS)
    ctx = tmp_path / "iter00.json"
    ctx.write_text("{}")
    p.tools = (ctx, tmp_path / "tool_calls.jsonl")
    f = p.flags()
    assert "--safe-mode" not in f and f[f.index("--tools") + 1] == "" and "--strict-mcp-config" in f
    assert f[f.index("--allowedTools") + 1] == ",".join(f"mcp__sea__{n}" for n in ("check", "measure", "read", "sweep",
                                                                                 "sensitivity", "compare", "facts", "module",
                                                                                 "simulate", "note"))
    cfg = json.loads((tmp_path / "iter00.mcp.json").read_text())["mcpServers"]
    assert list(cfg) == ["sea"] and cfg["sea"]["command"] == sys.executable and str(ctx.resolve()) in cfg["sea"]["args"]


def test_the_loop_writes_each_iteration_s_context_and_counts_its_calls(tmp_path):
    w = Builds()

    class Asks(Scripted):  # a proposer that checks its draft before answering, as System 2 would
        tools = None

        def propose(self, system, prompt, tag=""):
            ctx, log = self.tools
            Server(Context.model_validate_json(ctx.read_text()), log, w).call("check", {"proposal": PROPS[0].model_dump()})
            return super().propose(system, prompt, tag)

    proposer = Asks([ProposalBatch(proposals=[PROPS[0]])])
    s = run(RunConfig(world="toy", iterations=2, tools=True), tmp_path, proposer=proposer, world=w, log=lambda m: None)
    assert "# Questions to the harness" in proposer.prompts[0]
    rec = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])
    assert rec["tools"]["by_tool"] == {"check": 1}
    assert Context.model_validate_json((tmp_path / "tools" / "iter00.json").read_text()).body == w.stock()
    assert s["tools"]["calls"] == len(proposer.prompts)
    off = tmp_path / "off"
    plain = Scripted([ProposalBatch(proposals=[PROPS[0]])])
    run(RunConfig(world="toy", iterations=1), off, proposer=plain, world=w, log=lambda m: None)
    assert "# Questions to the harness" not in plain.prompts[0] and not (off / "tools").exists()


class Measured(Builds):
    """The toy world with instruments (D-088): its stick's reach, and how far short of a task's target it falls."""

    def instruments(self):
        from sea.evolve.core import Instrument

        return [Instrument(name="reach_m", unit="m", doc="the stick's reach"),
                Instrument(name="short_m", unit="m", doc="how far short of this task's target", task=True),
                Instrument(name="tip_m", unit="m", doc="the tip's height at an angle", args={"deg": "angle"})]

    def read(self, body, name, args, task=None):
        import math

        reach = body.parts["stick"].params["length"]
        if name == "reach_m":
            return reach
        if name == "short_m":
            return max(0.0, task.data["d"] - reach)
        return reach * math.sin(math.radians(args["deg"]))


def test_instruments_read_any_body_and_metrics_are_arithmetic_over_them():
    from sea.evolve import instruments as I

    w = Measured()
    b = w.stock()
    t = next(x for x in w.tasks() if x.id == "d0.45_train0")
    assert [i.name for i in I.instruments(w)][:4] == ["cost_usd", "fab_minutes", "lead_days", "refusals"]
    assert I.read(w, b, "reach_m") == 0.2 and I.read(w, b, "short_m", task=t) == 0.25
    assert I.facts(w, t) == {"d": 0.45}
    for bad, msg in [("nope", "no instrument"), ("short_m", "give its task id")]:
        try:
            I.read(w, b, bad)
        except ValueError as exc:
            assert msg in str(exc)
        else:
            raise AssertionError(bad)
    got = I.evaluate("2 * reach_m() - d + tip_m(deg=90) / sqrt(4)", lambda n, a: I.read(w, b, n, a, t),
                     lambda n: I.facts(w, t)[n])
    assert abs(got - (0.4 - 0.45 + 0.1)) < 1e-12
    for evil in ("__import__('os')", "reach_m().real", "[1, 2]"):
        try:
            I.evaluate(evil, lambda n, a: 0.0, lambda n: 0.0)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError(evil)


def test_read_sweep_and_facts_tools_measure_drafts():
    w = Measured()
    s = Server(_ctx(w), None, w)
    draft = _p("params", [Edit(slot="stick", params={"length": 0.3})]).model_dump()
    assert s.call("read", {"proposal": draft, "instrument": "reach_m"}) == ("reach_m() = 0.3 m", False)
    text, err = s.call("read", {"proposal": draft, "instrument": "short_m", "task": "d0.45_train0"})
    assert not err and text == "short_m() on d0.45_train0 = 0.15 m"
    assert s.call("read", {"proposal": draft, "instrument": "short_m", "task": "d0.45_test0"})[1]  # test unseen
    text, err = s.call("sweep", {"proposal": draft, "slot": "stick", "param": "length", "values": [0.1, 0.3, 0.9],
                                 "readings": [{"instrument": "reach_m"}, {"instrument": "cost_usd"}]})
    assert not err and "length = 0.1: reach_m() 0.1" in text and "outside its range" in text
    text, err = s.call("facts", {"task": "d0.15_train0"})
    assert not err and "d = 0.15" in text
