"""Concepts developed before they are judged (D-091): the set, its narrowing on computed evidence, the functions,
the comparison, the parallel-draft rule and the failure modes."""

from __future__ import annotations

import json

from sea.evolve import concepts as C
from sea.evolve.core import (
    Body,
    Edit,
    FailureMode,
    Function,
    Idea,
    Mechanism,
    Part,
    ProposalBatch,
    Requirement,
)
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, run
from sea.evolve.requirements import Row
from sea.evolve.tools import Server
from tests.test_evolve import _p
from tests.test_tools import Measured, _ctx

REACH = Requirement(id="R1", statement="reaches", metric="reach_m()", sense=">=", marginal=0.5, source="tasks")
COST = Requirement(id="R2", statement="light", metric="reach_m() * 10", sense="<=", marginal=2.0, source="the brief")


class Patches(Measured):
    """The toy world whose rods can be developed: module ``self`` continues a body's rod, a patch changes its length."""

    def build_idea(self, body, idea, new_id):
        mech = idea.mechanism
        if mech.module == "self":
            if body.parts["stick"].type != "built":
                raise ValueError("no built rod to continue")
            length = body.parts["stick"].params["length"]
        elif mech.module == "rod":
            length = mech.params["length"]
        else:
            raise ValueError(f"no module {mech.module!r}")
        for op in mech.patch:
            length = op["value"]
        return Body(id=new_id, parent=body.id, parts={**body.parts, "stick": Part(
            type="built", params={"length": length}, mechanism=json.dumps({"name": idea.name}))})

    def check(self, body):
        return ["the rod buckles"] if body.parts["stick"].params["length"] > 0.9 else []

    def solve(self, body, task, seed=0):
        return super().solve(body, task, seed)


def _rod(name="rod", length=0.28, **kw):
    return _p("idea", idea=Idea(name=name, description="d", slot="stick",
                                mechanism=Mechanism(module="rod", params={"length": length})), **kw)


def _dev(base, length, **kw):
    return _p("idea", idea=Idea(name="rod", description="d", slot="stick", mechanism=Mechanism(
        module="self", patch=[{"op": "set", "path": "length", "value": length}])), base=base, **kw)


def _run(tmp_path, batches, iterations=None, world=None, **cfg):
    proposer = Scripted([ProposalBatch(proposals=b) if isinstance(b, list) else b for b in batches])
    s = run(RunConfig(world="toy", iterations=iterations or len(batches), concepts=True, merge=False, **cfg), tmp_path,
            proposer=proposer, world=world or Patches(), log=lambda m: None)
    recs = [json.loads(x) for x in (tmp_path / "iterations.jsonl").read_text().splitlines()]
    return s, recs, proposer, C.ConceptSet.model_validate_json((tmp_path / "concepts.json").read_text())


def test_a_trialled_or_law_refused_structure_opens_a_concept_and_self_continues_it(tmp_path):
    s, recs, proposer, cs = _run(tmp_path, [
        [_rod("short", 0.21), _rod("buckling", 0.95), _p("idea", idea=Idea(
            name="nope", description="d", slot="stick", mechanism=Mechanism(module="whip")))],
        [_dev("C2", 0.22), _dev("C1", 0.23)]])
    t = recs[0]["trials"]
    assert t[0]["concept"] == "C1" and t[0]["tried"] and t[1]["concept"] == "C2" and not t[1]["tried"]
    assert "concept" not in t[2]  # a mechanism that could not be built leaves nothing to develop
    assert "# Concepts" in proposer.prompts[1] and "## C2 buckling" in proposer.prompts[1]
    t = recs[1]["trials"]
    assert t[0]["concept"] == "C2" and t[1]["concept"] == "C1"
    assert t[0]["tried"]  # the refused concept, patched back inside the law, reaches its trial
    assert cs.concepts[1].body.parts["stick"].params["length"] == 0.22
    assert len(cs.concepts[0].steps) == 2 and s["concepts"]["opened"] == 2


def test_a_fourth_concept_waits_and_the_check_says_what_the_loop_says(tmp_path):
    w = Patches()
    _, recs, _, cs = _run(tmp_path, [[_rod("a", 0.21), _rod("b", 0.22), _rod("c", 0.23)], [_rod("d", 0.24), _dev("C1", 0.245)]],
                          world=w)
    assert [c.id for c in cs.live()] == ["C1", "C2", "C3"]
    refused = recs[1]["trials"][0]["result"]
    assert refused.startswith("refused before trial: 3 concepts are alive (C1, C2, C3)")
    assert recs[1]["trials"][1]["concept"] == "C1"
    s = Server(_ctx(w, concepts=cs, bases=cs.bases()), None, w)
    assert s.call("check", {"proposal": _rod("d", 0.24).model_dump()})[0] == refused
    assert s.call("check", {"proposal": _dev("C1", 0.235).model_dump()})[0].startswith("would trial")


def test_a_swap_is_a_concept_too_and_params_on_it_develop_it(tmp_path):
    swap = _p("swap", [Edit(slot="stick", type="telescopic", params={"length": 0.21})])
    dev = _p("params", [Edit(slot="stick", params={"length": 0.22})], base="C1")
    _, recs, _, cs = _run(tmp_path, [[swap], [dev]])
    assert recs[0]["trials"][0]["concept"] == "C1" and recs[1]["trials"][0]["concept"] == "C1"
    assert len(cs.concepts) == 1 and cs.concepts[0].body.parts["stick"].params["length"] == 0.22


def test_an_allowance_spent_without_progress_drops_the_concept_and_it_cannot_be_built_on(tmp_path):
    lengths = [0.21, 0.21, 0.211, 0.2105, 0.2102]
    batches = [[_rod("flat", lengths[0])]] + [[_dev("C1", x)] for x in lengths[1:]] + [[_dev("C1", 0.22), _rod("flat", 0.2103)]]
    batches[0] = ProposalBatch(proposals=batches[0], requirements=[REACH])
    _, recs, proposer, cs = _run(tmp_path, batches)
    c = cs.concepts[0]
    assert c.status == "dropped" and "allowance spent" in c.why
    assert "C1" in next(r["concepts_dropped"] for r in recs if r.get("concepts_dropped"))
    last = recs[-2]["trials"]
    assert last[0]["result"].startswith("refused before trial: concept C1 was dropped")
    assert last[1]["concept"] == "C2"  # the same structure may open again: nothing is outlawed
    assert "Left the set" in proposer.prompts[-1] and "dropped: allowance spent" in proposer.prompts[-1]


def test_progress_renews_an_allowance():
    cs = C.ConceptSet(concepts=[C.Concept(id="C1", name="x", body=Measured().stock(), opened=0, last=3, trials=4,
                                          baseline={C._key(REACH): 0.2}, rows={C._key(REACH): 0.3})])
    assert cs.narrow(3, [REACH]) == [] and cs.concepts[0].trials == 0 and cs.concepts[0].baseline[C._key(REACH)] == 0.3
    cs.concepts[0].trials, cs.concepts[0].progress = 4, ["d0.25"]  # no requirement moved, but a group was gained
    cs.concepts[0].rows = {C._key(REACH): 0.3}
    assert cs.narrow(3, [REACH]) == [] and cs.concepts[0].trials == 0 and cs.concepts[0].progress == []
    cs.concepts[0].trials = 4
    assert cs.narrow(3, [REACH]) == ["C1"]


def _step(n):
    return [C.Step(iteration=i, body="b", edit="e", result="r", tried=True) for i in range(n)]


def test_dominance_waits_for_development_and_respects_groups():
    body = Measured().stock()

    def concept(i, v, steps, gained=()):
        return C.Concept(id=i, name=i, body=body, opened=0, last=3, steps=_step(steps), gained=list(gained),
                         rows={C._key(REACH): v}, baseline={C._key(REACH): v})

    assert C.ConceptSet(concepts=[concept("C1", 0.2, 3), concept("C2", 0.4, 3)]).narrow(3, [REACH]) == ["C1"]
    assert C.ConceptSet(concepts=[concept("C1", 0.2, 1), concept("C2", 0.4, 3)]).narrow(3, [REACH]) == []  # undeveloped
    assert C.ConceptSet(concepts=[concept("C1", 0.2, 3, ["g"]), concept("C2", 0.4, 3)]).narrow(3, [REACH]) == []  # its gain
    assert C.ConceptSet(concepts=[concept("C1", 0.2, 3), concept("C2", 0.4, 3)]).narrow(3, []) == []  # nothing measured


def test_a_concept_left_alone_gives_up_its_place(tmp_path):
    batches = [[_rod("a", 0.21)]] + [[_p("params", [Edit(slot="stick", params={"length": 0.2 + i / 1000})])]
                                     for i in range(1, 6)]
    _, recs, _, cs = _run(tmp_path, batches)
    assert cs.concepts[0].status == "dropped" and "undeveloped for 5" in cs.concepts[0].why
    assert recs[-2]["concepts_dropped"] == ["C1"]


def test_a_kept_concept_leaves_the_set_as_the_current_body(tmp_path):
    _, recs, _, cs = _run(tmp_path, [[_rod("long", 0.5)], [_dev(None, 0.6), _dev("C1", 0.7)]], iterations=2)
    assert cs.concepts[0].status == "kept" and recs[0]["trials"][0]["result"].startswith("KEPT")
    assert "stop" in recs[1]  # the long rod holds every group: the run ends


def test_the_matrix_s_signs_follow_each_requirement_s_sense(tmp_path):
    def row(v):
        return Row(id="x", value=v, meets="no")

    assert C.sign(REACH, 0.2, 0.3) == "+" and C.sign(REACH, 0.3, 0.2) == "-" and C.sign(REACH, 0.2, 0.2) == "0"
    assert C.sign(COST, 2.0, 3.0) == "-" and C.sign(COST, 3.0, 2.0) == "+" and C.sign(COST, None, 1.0) == "?"
    assert C.sign(REACH, 0.5, 0.5001) == "0"  # within 1%: not a move
    m = C.matrix([REACH, COST], [row(0.2), row(2.0)], [("C1", [row(0.4), row(4.0)])])
    assert "R1 `reach_m()` >= 0.5: current 0.2; C1 0.4 +" in m and "R2 `reach_m() * 10` <= 2: current 2; C1 4 -" in m
    w = Patches()
    s = Server(_ctx(w, requirements=[REACH, COST]), None, w)
    text, err = s.call("compare", {"proposals": [_rod("a", 0.3).model_dump(), _rod("b", 0.1).model_dump(),
                                                 _p("idea", idea=Idea(name="n", description="d", slot="stick",
                                                                      mechanism=Mechanism(module="whip"))).model_dump()]})
    assert not err and "draft 1 0.3 +; draft 2 0.1 -" in text and "draft 1 3 -; draft 2 1 +" in text
    assert "draft 3: refused before trial: no module 'whip'" in text
    assert "no requirements yet" in Server(_ctx(w), None, w).call("compare", {"proposals": []})[0]


def test_a_new_concept_simulates_only_after_two_drafts(tmp_path):
    w, log = Patches(), tmp_path / "log.jsonl"
    s = Server(_ctx(w, concepts=C.ConceptSet()), log, w)
    a, b = _rod("a", 0.3).model_dump(), _rod("b", 0.4).model_dump()
    task = {"task": "d0.15_train0"}
    text, err = s.call("simulate", {"proposal": a, **task})
    assert err and "opens a new concept" in text
    s.call("check", {"proposal": a})
    s.call("check", {"proposal": a})  # the same draft twice is one draft
    assert "opens a new concept" in s.call("simulate", {"proposal": a, **task})[0]
    s.call("measure", {"proposal": b})
    text, err = s.call("simulate", {"proposal": a, **task})
    assert not err and "not a trial" in text
    # developing a concept, and a run without the set, are not held to it
    live = C.ConceptSet(concepts=[C.Concept(id="C1", name="x", body=w.build_idea(w.stock(), _rod().idea, "c"), opened=0,
                                            last=0)])
    s2 = Server(_ctx(w, concepts=live, bases=live.bases()), tmp_path / "log2.jsonl", w)
    assert not s2.call("simulate", {"proposal": _dev("C1", 0.3).model_dump(), **task})[1]
    assert not Server(_ctx(w), tmp_path / "log3.jsonl", w).call("simulate", {"proposal": a, **task})[1]


def test_failure_modes_are_scored_against_the_laws_and_the_trial(tmp_path):
    held_lost = _p("params", [Edit(slot="stick", params={"length": 0.1})], failure_modes=[
        FailureMode(mode="too short", cause="c", effect="e", signature="too_short"),
        FailureMode(mode="it melts", cause="c", effect="e", signature="melts")])
    buckles = _rod("x", 0.95, failure_modes=[FailureMode(mode="buckling", cause="c", effect="e", signature="BUCKLES")])
    unwarned = _rod("y", 0.97)
    _, recs, proposer, _ = _run(tmp_path, [[held_lost, buckles, unwarned], [_p("params", [])]])
    t = recs[0]["trials"]
    assert t[0]["failure_modes"][0]["occurred"] and t[0]["failure_modes"][0]["detected"] == "trial"
    assert t[0]["failure_modes"][0]["severity"] == 1 and not t[0]["failure_modes"][1]["occurred"]
    assert t[1]["failure_modes"][0]["detected"] == "laws" and t[1]["failure_modes"][0]["severity"] == 0
    assert t[1]["unpredicted_failures"] == [] and t[2]["unpredicted_failures"] == ["the rod buckles"]
    assert ("Failure modes: 2 of the 3 you predicted occurred (1 showed in the laws, 1 only in a trial, costing 1 held "
            "groups); 1 failures came that no mode predicted.") in proposer.prompts[1]


def test_the_function_chart_marks_what_exists_and_what_is_untried(tmp_path):
    batch = ProposalBatch(proposals=[_rod("a", 0.21, options={"reach out": "rod"})],
                          functions=[Function(name="reach out", options=["rod", "telescopic", "jet pack"])])
    _, recs, proposer, _ = _run(tmp_path, [batch, [_p("params", [])]])
    assert recs[0]["functions"][0]["name"] == "reach out"
    text = proposer.prompts[1]
    assert "- reach out: rod (not in the catalogue or modules); telescopic [untried]; jet pack (not in the catalogue or modules) [untried]" in text


def test_with_the_switch_off_the_loop_is_as_it_was(tmp_path):
    p = Scripted([ProposalBatch(proposals=[_rod("a", 0.3)])])
    run(RunConfig(world="toy", iterations=1, merge=False), tmp_path, proposer=p, world=Patches(), log=lambda m: None)
    rec = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])
    assert "concepts" not in rec and "concept" not in rec["trials"][0] and "failure_modes" not in rec["trials"][0]
    assert "# Concepts" not in p.prompts[0] and not (tmp_path / "concepts.json").exists()


def test_the_gripper_continues_a_built_jaw():
    from sea.worlds.gripper import GripperWorld

    w = GripperWorld()
    built = w.build_idea(w.stock(), Idea(name="plain", description="d", slot="jaw", mechanism=Mechanism(module="revolute")),
                         "b1")
    again = w.build_idea(built, Idea(name="plain2", description="d", slot="jaw", mechanism=Mechanism(module="self")), "b2")
    assert json.loads(again.parts["jaw"].mechanism)["bodies"] == json.loads(built.parts["jaw"].mechanism)["bodies"]
    assert '"name":"plain"' in w.mechanism_text(built)
    assert w.mechanism_text(w.stock()) == "" and "revolute" in w.module_names()
    for body, idea, msg in [(w.stock(), Mechanism(module="self"), "catalogue part"),
                            (built, Mechanism(module="self", params={"x": 1.0}), "no parameters")]:
        try:
            w.build_idea(body, Idea(name="n", description="d", slot="jaw", mechanism=idea), "b3")
        except ValueError as exc:
            assert msg in str(exc)
        else:
            raise AssertionError(msg)
