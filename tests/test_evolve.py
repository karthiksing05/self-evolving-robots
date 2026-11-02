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
from sea.evolve.resources import Spent, check
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
            _p("idea", idea=Idea(name="jet pack", description="fly to it", needs=["rail", "jet"])),
            _p("swap", [Edit(slot="stick", type="telescopic", params={"length": 0.5})]),  # gains d0.45 with a bought rail
        ]),
    ]
    llm = Scripted(batches)
    s = run(RunConfig(world="toy", iterations=4), tmp_path, proposer=llm, world=Reach(), log=lambda m: None)
    assert s["final_body"]["parts"]["stick"]["type"] == "telescopic"
    assert s["final_held_train"] == ["d0.15", "d0.25", "d0.45"]
    assert s["test"]["final_held"] == ["d0.15", "d0.25", "d0.45"] and s["test"]["stock_held"] == ["d0.15"]
    assert s["spent"]["usd"] == 4 and [i["name"] for i in s["ideas"]] == ["jet pack"]
    assert s["ideas"][0]["provided"] == ["rail"]
    recs = [json.loads(x) for x in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    results = [t["result"] for t in recs[0]["trials"]]
    assert results[0].startswith("rejected: lost d0.15") and results[1].startswith("KEPT")
    assert "cannot be bought" in results[2]
    assert recs[1]["trials"][1]["move"] == "swap" and recs[-1]["stop"] == "all groups held"
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
