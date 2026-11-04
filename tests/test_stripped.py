"""Every world stripped to its brief (D-095): no hand-written design knowledge reaches System 2."""

from __future__ import annotations

import pytest

from sea.evolve.core import ProposalBatch
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, make_world, run
from sea.evolve.prompt import build
from sea.evolve.resources import start
from tests.test_evolve import Reach, _p


@pytest.mark.parametrize("name", ["gripper", "mobile", "phonology", "me2110", "me2110_kit"])
def test_no_principle_reaches_a_stripped_prompt_and_rules_are_not_how(name):
    w = make_world(name)
    body = w.stock()
    tasks = [t for t in w.tasks() if t.split == "train"][:1]
    from sea.evolve.core import Outcome

    outs = {t.id: Outcome(task=t.id, success=False, mechanism="m", detail="d") for t in tasks}
    args = (w, body, tasks, outs, set(), [], [], start(w.bom(body)), w.manifest(), 0, 1)
    stripped = build(*args, rules=list(w.brief_rules()))
    plain = build(*args)
    assert "# Physics principles" not in stripped
    for p in w.principles():
        if p not in w.brief_rules():
            assert p not in stripped, p
    if w.principles():
        assert "# Physics principles" in plain and all(p in plain for p in w.principles())
    assert len(w.brief_rules()) <= 4


def test_the_loop_numbers_the_rules_and_the_requirements_trace_them(tmp_path):
    from sea.evolve.core import Edit, Requirement

    class W(Reach):
        def brief_rules(self):
            return ["the stick may not be sold", "the stick must be yours"]

    req = Requirement(id="R1", statement="s", metric="cost_usd()", sense="<=", marginal=5, source="rule 1")
    p = Scripted([ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})])], requirements=[req])])
    run(RunConfig(world="toy", iterations=2, stripped=True, requirements=True), tmp_path, proposer=p, world=W(),
        log=lambda m: None)
    assert "- rule 2: the stick must be yours" in p.prompts[0] and "a stick touches only" not in p.prompts[0]
    assert "no requirement cites rule 2" in p.prompts[1]
