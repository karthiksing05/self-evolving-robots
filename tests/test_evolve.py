"""The evolution core on a toy world: parts, resources, System 1, the ratchet, ideas (D-041)."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from sea.evolve.core import (
    Body,
    BomLine,
    Edit,
    Idea,
    Need,
    Outcome,
    Param,
    Part,
    PartType,
    Proposal,
    ProposalBatch,
    Task,
    apply,
)
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, held_groups, run
from sea.evolve.prompt import SYSTEM, build
from sea.evolve.resources import Spent, charge, check, start
from sea.evolve.search import local_search


class Reach:
    """A stick must reach targets at fixed distances; System 1 aims it (angle must be near 0.5)."""

    name = "toy"
    description = "a stick that must touch targets at several distances"

    def tasks(self) -> list[Task]:
        out = []
        for d in (0.15, 0.25, 0.45):
            for split, n in (("train", 1), ("test", 2)):
                for k in range(n):
                    out.append(Task(id=f"d{d}_{split}{k}", split=split, group=f"d{d}", facts=f"target {d} m away",
                                    data={"d": d}))
        return out

    def catalogue(self) -> list[PartType]:
        return [
            PartType(name="rigid", slot="stick", stock=True, doc="a printed stick",
                     params={"length": Param(lo=0.1, hi=0.3, unit="m", doc="reach")}),
            PartType(name="telescopic", slot="stick", doc="a sliding stick on a rail",
                     params={"length": Param(lo=0.2, hi=0.6, unit="m", doc="reach")}),
            PartType(name="none", slot="sensor", stock=True, doc="no sensor"),
            PartType(name="lidar", slot="sensor", doc="a lidar"),
        ]

    def stock(self) -> Body:
        return Body(parts={"stick": Part(type="rigid", params={"length": 0.2}), "sensor": Part(type="none")})

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        length = body.parts["stick"].params["length"]

        def attempt(ctl: dict[str, float], s: int) -> Outcome:
            aimed = abs(ctl["angle"] - 0.5) < 0.1
            far = length >= task.data["d"]
            return Outcome(task=task.id, success=aimed and far, score=0.5 * far + 0.5 * (1 - abs(ctl["angle"] - 0.5)),
                           mechanism="" if far else "too_short", detail=f"length {length:g} vs {task.data['d']}",
                           sim_s=0.01)

        return local_search(attempt, {"angle": (0.0, 1.0)}, {"angle": 0.1}, m=30, seed=seed)

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        return f"reach {body.parts['stick'].params['length']:g} m"

    def principles(self) -> list[str]:
        return ["a stick touches only what is closer than its length"]

    def bom(self, body: Body) -> list[BomLine]:
        st = body.parts["stick"]
        lines = [BomLine(item="print:PLA", qty=100 * st.params["length"], slot="stick")]
        if st.type == "telescopic":
            lines.append(BomLine(item="rail", qty=1, slot="stick"))
        if body.parts["sensor"].type == "lidar":
            lines.append(BomLine(item="lidar", qty=1, slot="sensor"))
        return lines

    def check(self, body: Body) -> list[str]:
        return []

    def manifest(self) -> dict:
        return {"budget": {"usd": 10, "lead_days": 5, "fab_minutes": 200},
                "fabrication": {"print:PLA": {"minutes_per_unit": 1.0, "unit": "g"}},
                "items": {"rail": {"on_hand": 0, "usd": 4, "lead_days": 2}, "lidar": {"on_hand": 0}}}


def test_apply_checks_the_catalogue():
    w = Reach()
    cat = {p.name: p for p in w.catalogue()}
    b = w.stock()
    assert apply(b, [Edit(slot="stick", params={"length": 0.3})], cat, "x").parts["stick"].params["length"] == 0.3
    swapped = apply(b, [Edit(slot="stick", type="telescopic")], cat, "y").parts["stick"]
    assert swapped.type == "telescopic" and swapped.params["length"] == pytest.approx(0.4)  # midpoint
    for bad, msg in [(Edit(slot="arm"), "does not exist"), (Edit(slot="stick", type="jet"), "not in the catalogue"),
                     (Edit(slot="stick", type="lidar"), "fills slot"), (Edit(slot="stick", params={"mass": 1}), "no parameter"),
                     (Edit(slot="stick", params={"length": 0.5}), "outside its range")]:
        with pytest.raises(ValueError, match=msg):
            apply(b, [bad], cat, "z")


def test_resources_reuse_buy_refuse():
    w = Reach()
    cat = {p.name: p for p in w.catalogue()}
    stock = w.stock()
    tele = apply(stock, [Edit(slot="stick", type="telescopic", params={"length": 0.5})], cat, "t")
    r = check(tele, w.bom(tele), stock, w.bom(stock), w.manifest())
    assert r.ok and r.usd == 4 and r.lead_days == 2 and r.fab_minutes == pytest.approx(50)
    back = check(stock, w.bom(stock), tele, w.bom(tele), w.manifest())  # the rail comes back; reprint the stick
    assert back.ok and back.usd == 0
    lidar = apply(stock, [Edit(slot="sensor", type="lidar")], cat, "l")
    r = check(lidar, w.bom(lidar), stock, w.bom(stock), w.manifest())
    assert not r.ok and "cannot be bought" in r.errors[0] and r.fab_minutes == 0  # the stick did not change
    r = check(tele, w.bom(tele), stock, w.bom(stock), w.manifest(), Spent(usd=8))
    assert not r.ok and "left" in r.errors[0]


def test_a_spare_is_used_once():
    """Two spare cells on hand and two in the stock pack: a 4-cell pack is free, then a 6-cell pack buys two."""
    man = {"budget": {"usd": 10, "lead_days": 5}, "items": {"cell": {"on_hand": 2, "usd": 1.0, "lead_days": 1}}}

    def pack(n: int) -> tuple[Body, list[BomLine]]:
        return Body(id=f"p{n}", parts={"battery": Part(type="pack", params={"cells": n})}), [BomLine(item="cell", qty=n)]

    (b2, m2), (b4, m4), (b6, m6) = pack(2), pack(4), pack(6)
    spent = start(m2)
    r4 = check(b4, m4, b2, m2, man, spent)
    assert r4.ok and r4.usd == 0 and r4.buy == {}
    spent = charge(spent, r4)
    r6 = check(b6, m6, b4, m4, man, spent)
    assert r6.ok and r6.usd == 2 and r6.buy == {"cell": 2}
    assert check(b6, m6, b4, m4, man).usd == 0  # without the run's ledger the two spares would count twice
    spent = charge(spent, r6)
    assert spent.owned == {"cell": 4} and spent.usd == 2
    assert check(b2, m2, b6, m6, man, spent).usd == 0  # going back needs nothing new


def test_local_search_confirms():
    calls = []

    def attempt(ctl, s):
        calls.append(s)
        return Outcome(task="t", success=abs(ctl["x"] - 0.8) < 0.05, score=1 - abs(ctl["x"] - 0.8))

    out = local_search(attempt, {"x": (0.0, 1.0)}, {"x": 0.2}, m=60, confirm=2, seed=3)
    assert out.success and abs(out.controls["x"] - 0.8) < 0.05 and out.attempts == len(calls)
    assert sum(s >= 1000 for s in calls) == 2  # two confirmation seeds


def _p(move, edits=(), **kw):
    return Proposal(move=move, edits=list(edits), mechanism=kw.pop("mechanism", "too_short"),
                    principle="reach", hypothesis=kw.pop("hypothesis", "h"), **kw)


def test_loop_ratchet_swap_and_ideas(tmp_path):
    batches = [
        ProposalBatch(proposals=[
            _p("params", [Edit(slot="stick", params={"length": 0.1})]),  # loses d0.15: rejected
            _p("params", [Edit(slot="stick", params={"length": 0.3})]),  # gains d0.25: kept
            _p("swap", [Edit(slot="sensor", type="lidar")]),  # not provided: refused
        ]),
        ProposalBatch(proposals=[
            _p("idea", idea=Idea(name="jet pack", description="fly to it",
                                 needs=[Need(item="rail"), Need(item="jet", usd_each=50.0, lead_days=9),
                                        Need(item="print:PLA", qty=30)])),
            _p("swap", [Edit(slot="stick", type="telescopic", params={"length": 0.5})], gains=["d0.45", "d0.15"]),
            _p("params", [Edit(slot="stick", params={"length": 0.1})]),  # the body rejected in iteration 1
        ]),
    ]
    llm = Scripted(batches)
    s = run(RunConfig(world="toy", iterations=4), tmp_path, proposer=llm, world=Reach(), log=lambda m: None)
    assert s["final_body"]["parts"]["stick"]["type"] == "telescopic"
    assert s["final_held_train"] == ["d0.15", "d0.25", "d0.45"]
    assert s["test"]["final_held"] == ["d0.15", "d0.25", "d0.45"] and s["test"]["stock_held"] == ["d0.15"]
    assert s["spent"]["usd"] == 4 and [i["name"] for i in s["ideas"]] == ["jet pack"]
    cost = s["ideas"][0]["cost"]
    assert cost["usd"] == 4 and cost["usd_estimated"] == 50 and cost["fab_minutes"] == 30 and not cost["fits_budget"]
    assert "jet x1: not on the manifest; estimated $50.00" in cost["new_items"]
    recs = [json.loads(x) for x in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    results = [t["result"] for t in recs[0]["trials"]]
    assert results[0].startswith("rejected: lost d0.15") and results[1].startswith("KEPT")
    assert "cannot be bought" in results[2]
    assert recs[1]["trials"][1]["move"] == "swap" and recs[-1]["stop"] == "all groups held"
    assert "already trialled" in recs[1]["trials"][2]["result"]
    assert recs[1]["trials"][1]["prediction"]["hit"] == ["d0.45"] and recs[1]["trials"][1]["prediction"]["precision"] == 0.5
    st = s["proposals"]
    assert st["kept"] == 2 and st["refused"] == 2 and st["ideas"] == 1 and st["rejected_regression"] == 1
    assert (tmp_path / "harness.json").exists()
    assert len(llm.prompts) == 2
    p2 = llm.prompts[1]
    for section in ("# Brief", "# Catalogue", "# Resources", "# Held task groups", "# Record of past proposals",
                    "# What parameter moves have not fixed"):
        assert section in p2
    assert "refused before trial" in p2 and "lost d0.15" in p2 and "rail x1 (buy $4 each" in p2


def test_prompt_and_random_arm(tmp_path):
    w = Reach()
    tasks = [t for t in w.tasks() if t.split == "train"]
    outs = {t.id: w.solve(w.stock(), t) for t in tasks}
    assert held_groups(tasks, outs) == {"d0.15"}
    text = build(w, w.stock(), tasks, outs, {"d0.15"}, [], [], Spent(), w.manifest(), 0, 3)
    assert "too_short" in text and "telescopic" in text and "NOT PROVIDED" in text
    assert "Climb the ladder" in SYSTEM
    s = run(RunConfig(world="toy", arm="random", iterations=3, seed=1), tmp_path, world=Reach(), log=lambda m: None)
    assert s["arm"] == "random" and len(s["curve"]) >= 2
    s1 = run(RunConfig(world="toy", arm="s1", iterations=3), tmp_path / "s1", world=Reach(), log=lambda m: None)
    assert s1["final_held_train"] == ["d0.15"] and len(s1["curve"]) == 1


class Trade(Reach):
    """A swap that gains two groups and breaks one; a parameter of the new part repairs it."""

    def tasks(self) -> list[Task]:
        return [Task(id=f"{g}_train", group=g, data={"g": g}) for g in ("a", "b", "c")]

    def catalogue(self) -> list[PartType]:
        return [PartType(name="old", slot="hand", stock=True, doc="the old hand"),
                PartType(name="new", slot="hand", doc="a new hand",
                         params={"care": Param(lo=0, hi=1, doc="keeps a working")})]

    def stock(self) -> Body:
        return Body(parts={"hand": Part(type="old")})

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        h = body.parts["hand"]
        ok = (task.data["g"] == "a") if h.type == "old" else (task.data["g"] != "a" or h.params["care"] > 0.9)
        return Outcome(task=task.id, success=ok, score=float(ok), mechanism="" if ok else "wrong_hand")

    def bom(self, body: Body) -> list[BomLine]:
        return []

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        return body.describe()


def test_frontier_lets_a_redesign_be_repaired(tmp_path):
    batches = [ProposalBatch(proposals=[_p("swap", [Edit(slot="hand", type="new")])]),
               ProposalBatch(proposals=[_p("params", [Edit(slot="hand", params={"care": 1.0})], base="b000"),
                                        _p("params", [Edit(slot="hand", params={"care": 1.0})], base="nope")])]
    llm = Scripted(batches)
    s = run(RunConfig(world="trade", iterations=3, tune_trials=0), tmp_path, proposer=llm, world=Trade(),
            log=lambda m: None)
    recs = [json.loads(x) for x in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    first = recs[0]["trials"][0]["result"]
    assert first.startswith("rejected: lost a") and "would gain b, c" in first and "frontier as b000" in first
    assert "# Promising bodies not kept" in llm.prompts[1] and "b000" in llm.prompts[1]
    assert recs[1]["trials"][0]["result"].startswith("KEPT") and "no promising body 'nope'" in recs[1]["trials"][1]["result"]
    assert s["final_held_train"] == ["a", "b", "c"] and len(s["frontier"]) == 1


class Gentle(Trade):
    """As Trade, but the new hand keeps ``a`` once ``care`` is past 0.6: its midpoint is close, not right."""

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        h = body.parts["hand"]
        ok = (task.data["g"] == "a") if h.type == "old" else (task.data["g"] != "a" or h.params["care"] > 0.6)
        score = float(ok) if h.type == "old" or ok else h.params["care"]
        return Outcome(task=task.id, success=ok, score=score, mechanism="" if ok else "wrong_hand")


def test_tuning_protects_a_new_part(tmp_path):
    """A swap judged at its midpoint would break ``a``; the loop fits its parameter first (D-044)."""
    llm = Scripted([ProposalBatch(proposals=[_p("swap", [Edit(slot="hand", type="new")], gains=["b", "c"])])])
    s = run(RunConfig(world="gentle", iterations=2, tune_trials=6), tmp_path, proposer=llm, world=Gentle(),
            log=lambda m: None)
    rec = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])
    first, *tuned = rec["trials"]
    assert first["result"].startswith("rejected: lost a") and first["move"] == "swap"
    assert tuned and all(t["move"] == "tune" and t["base"] == "b000" for t in tuned)
    assert tuned[-1]["result"].startswith("KEPT") and rec["accepted"] == f"b00{tuned[-1]['k']}"
    assert s["final_held_train"] == ["a", "b", "c"] and s["final_body"]["parts"]["hand"]["params"]["care"] > 0.6
    assert s["proposals"]["moves"]["tune"] == len(tuned) and len(llm.prompts) == 1
    off = run(RunConfig(world="gentle", iterations=1, tune_trials=0), tmp_path / "off", proposer=Scripted(
        [ProposalBatch(proposals=[_p("swap", [Edit(slot="hand", type="new")])])]), world=Gentle(), log=lambda m: None)
    assert off["final_held_train"] == ["a"] and off["proposals"]["moves"]["tune"] == 0


class Two(Reach):
    """Two independent slots, each with a part that gains its own group."""

    def tasks(self) -> list[Task]:
        return [Task(id=f"{g}_train", group=g, data={"g": g}) for g in ("base", "x", "y")]

    def catalogue(self) -> list[PartType]:
        return [PartType(name=n, slot=s, stock=n.endswith("0"), doc=n) for s in ("sx", "sy") for n in (s + "0", s + "1")]

    def stock(self) -> Body:
        return Body(parts={"sx": Part(type="sx0"), "sy": Part(type="sy0")})

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        g = task.data["g"]
        ok = g == "base" or body.parts["s" + g].type.endswith("1")
        return Outcome(task=task.id, success=ok, score=float(ok), mechanism="" if ok else "missing_part")

    def bom(self, body: Body) -> list[BomLine]:
        return []

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        return body.describe()


def test_disjoint_gains_are_merged(tmp_path):
    """Two kept-eligible swaps in different slots: the loop trials and keeps their union (D-045)."""
    batch = ProposalBatch(proposals=[_p("swap", [Edit(slot="sx", type="sx1")]), _p("swap", [Edit(slot="sy", type="sy1")])])
    s = run(RunConfig(world="two", iterations=2), tmp_path, proposer=Scripted([batch]), world=Two(), log=lambda m: None)
    rec = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])
    merge = rec["trials"][-1]
    assert merge["move"] == "merge" and merge["merged"] == ["b000", "b001"] and merge["result"].startswith("KEPT")
    assert s["final_held_train"] == ["base", "x", "y"] and s["proposals"]["moves"]["merge"] == 1
    off = run(RunConfig(world="two", iterations=1, merge=False), tmp_path / "off", proposer=Scripted([batch]),
              world=Two(), log=lambda m: None)
    assert len(off["final_held_train"]) == 2


class Wide(Gentle):
    """Gentle's new hand with five more parameters that do nothing: tuning must not stir them all at once."""

    def catalogue(self) -> list[PartType]:
        extra = {k: Param(lo=0, hi=1, doc="inert") for k in "vwxyz"}
        return [PartType(name="old", slot="hand", stock=True, doc="the old hand"),
                PartType(name="new", slot="hand", doc="a new hand", params={"care": Param(lo=0, hi=1, doc="keeps a"), **extra})]


def test_tuning_moves_one_or_two_parameters_at_a_time(tmp_path):
    llm = Scripted([ProposalBatch(proposals=[_p("swap", [Edit(slot="hand", type="new")], gains=["b", "c"])])])
    run(RunConfig(world="wide", iterations=1, tune_trials=6), tmp_path, proposer=llm, world=Wide(), log=lambda m: None)
    rec = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])
    tuned = [t for t in rec["trials"] if t["move"] == "tune"]
    vals = dict(kv.split("=") for kv in tuned[0]["edit"].split("(")[1].rstrip(")").split(", "))
    assert len(vals) == 6 and 1 <= sum(abs(float(v) - 0.5) > 1e-9 for v in vals.values()) <= 2  # vs the midpoints


def test_plan_claude_waits_out_an_outage_without_spending_an_attempt():
    """Three seeds of the D-061 campaign died when the network dropped: the CLI's "can't reach the API server"
    was read as a malformed answer, retried once and fatal.  An outage is waited out instead."""
    import json
    from types import SimpleNamespace

    from sea.evolve.llm import PlanClaude

    def cli(*replies):
        replies, calls = list(replies), []

        def run(args, **kw):
            calls.append(kw.get("input"))
            body = {"loggedIn": True, "authMethod": "claude.ai"} if args[1:3] == ["auth", "status"] else replies.pop(0)
            return SimpleNamespace(stdout=json.dumps(body), stderr="", returncode=0)
        return run, calls

    down = {"is_error": True, "result": "API Error: Can't reach the API server (ENOTFOUND)"}
    batch = ProposalBatch(proposals=[_p("swap", [Edit(slot="hand", type="new")])]).model_dump(mode="json")
    run, calls = cli(down, down, {"is_error": False, "structured_output": batch, "usage": {}})
    slept: list[float] = []
    assert len(PlanClaude(run=run, sleep=slept.append).propose("sys", "prompt").proposals) == 1
    assert slept == [30.0, 60.0] and "not a valid ProposalBatch" not in calls[-1]  # the outage spent no attempt
    run, _ = cli(*[down] * 9)
    with pytest.raises(RuntimeError, match="unreachable"):
        PlanClaude(run=run, sleep=lambda s: None, outage_s=100.0).propose("sys", "prompt")
