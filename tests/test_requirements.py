"""System 2's requirements list, measured by the harness (D-089): worst case over the groups it names, demand or
wish by the ratchet, coverage, the rows a trial moves, and couplings found by sensitivity."""

from __future__ import annotations

import json

from sea.evolve import requirements as R
from sea.evolve.core import Edit, ProposalBatch, Requirement
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, run
from sea.evolve.tools import Server
from tests.test_evolve import _p
from tests.test_tools import Measured, _ctx

REACH = Requirement(id="R1", statement="the stick reaches every target", metric="reach_m() - d", sense=">=",
                    marginal=0.0, ideal=0.02, tasks=["d0.15", "d0.25", "d0.45"], source="every task group")
CHEAP = Requirement(id="R2", statement="the change stays cheap", metric="cost_usd()", sense="<=", marginal=5.0,
                    source="the budget")
LOW = Requirement(id="R3", statement="the tip stays low at 30 degrees", metric="tip_m(deg=30)", sense="<=",
                  marginal=0.12, tasks=["d0.15"], source="the brief")


def test_requirements_are_measured_at_their_worst_task_and_marked_by_the_ratchet():
    w = Measured()
    train = [t for t in w.tasks() if t.split == "train"]
    rows = R.measure(w, w.stock(), [REACH, CHEAP], train)
    assert rows[0].worst_task == "d0.45_train0" and abs(rows[0].value + 0.25) < 1e-12 and rows[0].meets == "no"
    assert rows[1].value == 0.0 and rows[1].meets == "marginal"
    assert R.kind(REACH, {"d0.15"}) == "demand" and R.kind(CHEAP, {"d0.15"}) == "wish"
    assert R.coverage([LOW, CHEAP], ["d0.15", "d0.25", "d0.45"]) == ["d0.25", "d0.45"]
    bad = Requirement(id="R9", statement="x", metric="nope()", sense=">=", marginal=0, source="s")
    row = R.measure(w, w.stock(), [bad], train)[0]
    assert row.meets == "unknown" and "no instrument" in row.error
    text = R.text([REACH, LOW], R.measure(w, w.stock(), [REACH, LOW], train), {"d0.15"},
                  ["d0.15", "d0.25", "d0.45"])
    assert "R1 [demand]" in text and "R3 [demand]" in text and "Coverage" not in text


def test_the_loop_keeps_the_list_and_each_trial_records_the_rows_it_moved(tmp_path):
    w = Measured()
    batch = ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})])],
                          requirements=[REACH, CHEAP])
    proposer = Scripted([batch])
    run(RunConfig(world="toy", iterations=2, requirements=True), tmp_path, proposer=proposer, world=w,
        log=lambda m: None)
    assert "# Requirements" in proposer.prompts[0] and "No requirements yet" in proposer.prompts[0]
    assert "R1 [demand]" in proposer.prompts[1]  # the second prompt shows the list, measured on the kept body
    recs = [json.loads(x) for x in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    moved = recs[0]["trials"][0]["requirements"]
    assert moved == ["R1: -0.25 (no) -> -0.15 (no)"]
    assert [r["id"] for r in json.loads((tmp_path / "requirements.json").read_text())] == ["R1", "R2"]
    off = Scripted([batch])
    run(RunConfig(world="toy", iterations=1), tmp_path / "off", proposer=off, world=w, log=lambda m: None)
    assert "# Requirements" not in off.prompts[0]


def test_sensitivity_finds_a_parameter_that_pulls_two_requirements_apart():
    w = Measured()
    s = Server(_ctx(w, requirements=[REACH, LOW], held=["d0.15"]), None, w)
    draft = _p("params", [Edit(slot="stick", params={"length": 0.2})]).model_dump()
    text, err = s.call("sensitivity", {"proposal": draft})
    assert not err and "stick.length up helps R1 and hurts R3" in text and "down helps R3 and hurts R1" in text
    text, err = s.call("measure", {"proposal": draft})
    assert not err and "requirements, now and on the draft" in text and "draft -0.25: no" in text
    assert "no requirements yet" in Server(_ctx(w), None, w).call("sensitivity", {"proposal": draft})[0]


def test_rule_coverage_reads_the_sources():
    a = Requirement(id="R1", statement="s", metric="1", sense=">=", marginal=0, source="rule 2 of the brief, and rule 3")
    assert R.rule_coverage([a], ["x", "y", "z", "w"]) == [1, 4]
