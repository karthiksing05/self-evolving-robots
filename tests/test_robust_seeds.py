"""A run does not end on one seed's luck, and it can continue from a body a past run built (D-101)."""

from __future__ import annotations

import json

import pytest

from sea.evolve.core import Body, Edit, Outcome, Part, ProposalBatch, Task
from sea.evolve.llm import Scripted
from sea.evolve.loop import Evaluator, RunConfig, run
from tests.test_evolve import Reach, _p


class Lucky(Reach):
    """A short stick reaches d0.25 under seed 0 alone; only a longer one reaches it under every seed."""

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        length = body.parts["stick"].params["length"]
        need = task.data["d"] if seed == 0 or task.data["d"] != 0.25 else 0.28
        ok = length >= need
        return Outcome(task=task.id, success=ok, score=float(ok), mechanism="" if ok else "too_short",
                       detail=f"seed {seed} needs {need}", sim_s=0.01)

    def tasks(self) -> list[Task]:
        return [t for t in super().tasks() if t.data["d"] != 0.45]


def _body(length: float) -> Body:
    return Body(parts={"stick": Part(type="rigid", params={"length": length}), "sensor": Part(type="none")})


def test_a_task_holds_only_under_every_seed_and_a_loss_names_its_seed():
    w = Lucky()
    t = next(x for x in w.tasks() if x.data["d"] == 0.25)
    ev = Evaluator(w, "lucky")
    assert ev.solve(_body(0.26), [t])[t.id].success
    ev.seeds = [0, 100]
    got = ev.solve(_body(0.26), [t])[t.id]
    assert not got.success and got.detail.startswith("under seed 100:")
    assert ev.solve(_body(0.3), [t])[t.id].success


def test_each_seed_is_solved_once():
    w = Lucky()
    t = w.tasks()[0]
    ev = Evaluator(w, "lucky")
    ev.seeds = [0, 100]
    ev.solve(_body(0.3), [t])
    ev.solve(_body(0.3), [t])
    assert ev.attempts == 2


def test_a_lucky_body_does_not_end_the_run_and_the_loss_goes_back_to_system_2(tmp_path):
    fix = ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})])])
    llm = Scripted([fix])
    start = {"final_body": _body(0.26).model_dump()}
    (tmp_path / "from.json").write_text(json.dumps(start))
    s = run(RunConfig(world="lucky", iterations=2, robust=1, start=str(tmp_path / "from.json")), tmp_path / "run",
            proposer=llm, world=Lucky(), log=lambda m: None)
    assert s["final_body"]["parts"]["stick"]["params"]["length"] == pytest.approx(0.3)
    rows = [json.loads(x) for x in (tmp_path / "run" / "iterations.jsonl").read_text().splitlines()]
    assert "under seed 100" in rows[0]["failing"][next(iter(rows[0]["failing"]))]["detail"]
    assert rows[-1]["stop"] == "all groups held"


def test_without_robust_one_seed_still_ends_the_run(tmp_path):
    (tmp_path / "from.json").write_text(json.dumps({"final_body": _body(0.26).model_dump()}))
    s = run(RunConfig(world="lucky", iterations=2, start=str(tmp_path / "from.json")), tmp_path / "run",
            proposer=Scripted([]), world=Lucky(), log=lambda m: None)
    assert s["final_body"]["parts"]["stick"]["params"]["length"] == pytest.approx(0.26)
    assert s["final_body"]["id"] == "start"


def test_a_second_seed_judges_every_body_from_the_first_iteration(tmp_path):
    (tmp_path / "from.json").write_text(json.dumps({"final_body": _body(0.26).model_dump()}))
    run(RunConfig(world="lucky", iterations=0, robust=3, start=str(tmp_path / "from.json")), tmp_path / "run",
        proposer=Scripted([]), world=Lucky(), log=lambda m: None)
    row = json.loads((tmp_path / "run" / "iterations.jsonl").read_text().splitlines()[0])
    assert any("under seed 100" in o["detail"] for o in row["failing"].values())
    assert not any("under seed 200" in o["detail"] for o in row["failing"].values())


def test_the_run_says_what_stopped_it(tmp_path):
    s = run(RunConfig(world="lucky", iterations=0), tmp_path / "run", proposer=Scripted([]), world=Lucky(), log=lambda m: None)
    assert s["stopped"]["reason"] == "iterations"
    assert s["stopped"]["failing"] and all(v["mechanism"] == "too_short" for v in s["stopped"]["failing"].values())
    assert set(s["stopped"]["left"]) == {"usd", "fab_minutes"}
