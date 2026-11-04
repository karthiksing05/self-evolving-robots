"""System 2's research and notebook (D-090): the guard refuses what the world blocks and fails closed, a note must
cite a page this run read, the notebook outlives the run, and a kept citation earns credit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sea.evolve import research as RS
from sea.evolve.core import Edit, ProposalBatch
from sea.evolve.llm import PlanClaude, Scripted
from sea.evolve.loop import RunConfig, run
from sea.evolve.tools import Context, Server
from sea.worlds.me2110 import ME2110World
from tests.test_evolve import _p
from tests.test_tools import Measured

PAGE = "https://example.org/handbook/sticks"


def _cfg(tmp_path: Path, **kw) -> RS.Config:
    return RS.Config(blocked=ME2110World.research_blocked, log=str(tmp_path / "research.jsonl"), iteration=0, **kw)


def _search(q: str) -> dict:
    return {"tool_name": "WebSearch", "tool_input": {"query": q}}


def test_the_guard_refuses_the_benchmark_s_own_material_and_passes_engineering(tmp_path):
    cfg = _cfg(tmp_path)
    for q in ("ME 2110 final project", "me2110 robot ideas", "ME-2110 Georgia Tech", "Creative Decisions and Design notes",
              "Pirates of the Caribbean robot competition", "Davy Jones heart robot", "gold and jewels sorting robot",
              "Georgia Tech robot design competition winners"):
        assert not RS.pre(cfg, _search(q))[0], q
    assert not RS.pre(cfg, {"tool_name": "WebFetch", "tool_input": {"url": "https://x.io/me2110.html", "prompt": "read"}})[0]
    for q in ("pneumatic cylinder force bore pressure", "mousetrap spring torque curve", "rack and pinion module tooth count",
              "friction coefficient plywood rubber", "how to sort coins by size mechanism"):
        assert RS.pre(cfg, _search(q))[0], q
    assert RS.summary(Path(cfg.log)) == {"calls": 5, "refused": 9, "leaks": 0}


def test_the_budget_stops_the_next_search_and_the_guard_fails_closed(tmp_path):
    cfg = _cfg(tmp_path, budget=2)
    assert RS.pre(cfg, _search("a"))[0] and RS.pre(cfg, _search("b"))[0]
    ok, why = RS.pre(cfg, _search("c"))
    assert not ok and "spent" in why
    out = subprocess.run([sys.executable, "-m", "sea.evolve.research", "pre", "--config", str(tmp_path / "missing.json")],
                         input=json.dumps(_search("c")), capture_output=True, text=True, timeout=120)
    assert out.returncode == 2 and "could not check" in out.stderr  # a crash refuses: only exit 2 blocks a call


def test_a_note_must_cite_a_page_this_run_read_and_the_notebook_persists_and_earns_credit(tmp_path):
    cfg, book = _cfg(tmp_path), tmp_path / "notebook.jsonl"
    RS.post(cfg, {"tool_name": "WebFetch", "tool_input": {"url": PAGE + "/"}, "tool_response": {"text": "see https://b.org/x."}})
    read = RS.read_urls(Path(cfg.log))
    assert PAGE in read and "https://b.org/x" in read
    note = RS.Note(claim="a stick bends", source_url=PAGE, numbers={"E_gpa": 3.5})
    try:
        RS.add(book, note.model_copy(update={"source_url": "https://never.read/"}), read, "toy", "r", 0)
    except ValueError as exc:
        assert "cite what you read" in str(exc)
    else:
        raise AssertionError("an unread citation was recorded")
    kept = RS.add(book, note, read, "toy", "r", 0)
    assert RS.add(book, note, read, "toy", "r2", 1).id == kept.id and len(RS.load(book)) == 1  # the same finding once
    assert RS.credit(book, f"because of {kept.id} and N-000000") == [kept.id]
    assert RS.load(book)[0].credit == 1 and "credited 1" in RS.text(RS.load(book)) and "E_gpa = 3.5" in RS.text(RS.load(book))
    RS.post(cfg, {"tool_name": "WebSearch", "tool_input": {"query": "sticks"}, "tool_response": "ME 2110 teams use sticks"})
    assert RS.summary(Path(cfg.log))["leaks"] == 1  # what a search surfaced anyway is counted


def test_the_loop_gives_system_2_the_notebook_and_a_kept_citation_credits_it(tmp_path):
    w, book = Measured(), tmp_path / "library" / "notebook.jsonl"

    class Reads(Scripted):
        tools, research = None, None

        def propose(self, system, prompt, tag=""):
            self.prompts.append(prompt)
            ctx_path, log = self.tools
            ctx = Context.model_validate_json(ctx_path.read_text())
            guard = RS.Config.model_validate_json(ctx_path.with_suffix(".research.json").read_text())
            RS.post(guard, {"tool_name": "WebFetch", "tool_input": {"url": PAGE}, "tool_response": ""})
            text, err = Server(ctx, log, w).call("note", {"claim": "longer reaches further", "source_url": PAGE})
            assert not err, text
            nid = text.split()[-1]
            return ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})],
                                               hypothesis=f"from {nid}, a longer stick reaches")])

    p = Reads()
    s = run(RunConfig(world="toy", iterations=1, research=True, notebook=str(book)), tmp_path / "a", proposer=p, world=w,
            log=lambda m: None)
    assert "The notebook is empty" in p.prompts[0] and "# Research" in p.prompts[0]
    assert s["research"] == {"calls": 0, "refused": 0, "leaks": 0} and s["tools"]["by_tool"] == {"note": 1}
    assert RS.load(book)[0].credit == 1 and RS.load(book)[0].run == "a"
    later = Scripted([ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})])])])
    run(RunConfig(world="toy", iterations=1, research=True, notebook=str(book)), tmp_path / "b", proposer=later, world=w,
        log=lambda m: None)
    assert "longer reaches further" in later.prompts[0]  # another run reads what this one found
    off = Scripted([ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})])])])
    run(RunConfig(world="toy", iterations=1), tmp_path / "c", proposer=off, world=w, log=lambda m: None)
    assert "# Research" not in off.prompts[0]


def test_plan_claude_gets_search_and_fetch_only_with_the_guard(tmp_path):
    p = PlanClaude()
    ctx = tmp_path / "iter00.json"
    ctx.write_text("{}")
    p.tools = (ctx, tmp_path / "log.jsonl")
    assert p.flags()[p.flags().index("--tools") + 1] == "" and "--settings" not in p.flags()
    p.research = tmp_path / "hooks.json"
    f = p.flags()
    assert f[f.index("--tools") + 1] == "WebSearch,WebFetch" and f[f.index("--settings") + 1] == str(p.research.resolve())
    assert "WebSearch" in f[f.index("--allowedTools") + 1] and "mcp__sea__note" in f[f.index("--allowedTools") + 1]
    hooks = RS.settings(tmp_path / "guard.json")["hooks"]
    assert hooks["PreToolUse"][0]["matcher"] == "WebSearch|WebFetch" and " pre " in hooks["PreToolUse"][0]["hooks"][0]["command"]
