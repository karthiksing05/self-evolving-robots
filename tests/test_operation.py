"""System 1 carries out System 2's ideas (D-092): controls from the graph, operating plans, operation or body."""

from __future__ import annotations

import json

import mujoco
import pytest

from sea.evolve import operation as OP
from sea.evolve.core import Edit, Intent, OperatingPlan, Outcome, ProposalBatch
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, candidate, run
from sea.evolve.search import local_search, planned
from sea.mech import build
from sea.mech.base import WORLD
from sea.mech.graph import Body as GBody
from sea.mech.graph import Box, Encoder, Gearmotor, Graph, Joint, Latch, Piston, Servo
from sea.mech.program import Program, controls
from sea.mech.runtime import Machine
from tests.test_concepts import _rod, _run
from tests.test_evolve import Reach, _p


def _arm(name: str, jtype: str = "hinge", rng=None, z: float = 0.5) -> GBody:
    return GBody(name=name, parent="world", pos=(0.0, 0.0, z), joints=(Joint(
        name=f"{name}_q", type=jtype, axis=(0.0, 1.0, 0.0) if jtype == "hinge" else (1.0, 0.0, 0.0), range=rng),),
        parts=(Box(name=f"{name}_box", pos=(0.1, 0.0, 0.0), half=(0.1, 0.01, 0.01)),))


def _play(g: Graph, ctl, secs, triggers=None, gravity=False):
    spec = WORLD.model()
    if not gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]
    build.compile(spec, g, WORLD)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mc = Machine(m, d, g)
    prog = Program(g, ctl, secs, triggers)
    rows = []
    while d.time < secs:
        prog.step(mc, d.time)
        rows.append((d.time, d.ctrl.copy(), d.qpos.copy()))
        mujoco.mj_step(m, d)
    return rows, prog, mc


def _at(rows, t):
    return min(rows, key=lambda r: abs(r[0] - t))


def test_a_graph_s_controls_are_derived_from_its_elements():
    g = Graph(name="g", bodies=(_arm("a", rng=(-1.0, 1.0)), _arm("b", z=0.8), _arm("c", "slide", z=1.1)),
              actuators=(Servo(name="s", joint="a_q", stall_nm=2.0), Gearmotor(name="m", joint="b_q", stall_nm=1.0,
                                                                                 no_load_rad_s=10.0),
                         Piston(name="p", joint="c_q", bore_m=0.02, pressure_pa=5e5, valve_m_s=0.3)),
              latches=(Latch(name="l", joint="a_q"),), sensors=(Encoder(name="e", joint="a_q"),))
    c = controls(g, 10.0)
    assert set(c) == {"s.at", "s.target", "m.at", "m.run", "m.duty", "p.at", "p.hold", "l.at", "e.threshold"}
    assert c["s.target"] == (-1.0, 1.0, 0.0) and c["m.duty"][:2] == (-1.0, 1.0) and c["e.threshold"][:2] == (-1.0, 1.0)


def test_a_program_plays_a_motor_s_window_and_a_piston_s_stroke():
    g = Graph(name="g", bodies=(_arm("b"), _arm("c", "slide", rng=(-1.0, 1.0), z=0.9)),
              actuators=(Gearmotor(name="m", joint="b_q", stall_nm=1.0, no_load_rad_s=10.0),
                         Piston(name="p", joint="c_q", bore_m=0.02, pressure_pa=5e5, valve_m_s=0.3)))
    rows, _, _ = _play(g, {"m.at": 0.2, "m.run": 0.3, "m.duty": 0.5, "p.at": 0.4, "p.hold": 0.3}, 1.0)
    assert _at(rows, 0.1)[1][0] == 0.0 and _at(rows, 0.3)[1][0] == 0.5 and _at(rows, 0.6)[1][0] == 0.0
    assert _at(rows, 0.3)[1][1] == 0.0 and _at(rows, 0.5)[1][1] == 1.0 and _at(rows, 0.8)[1][1] == -1.0


def test_a_latch_releases_on_time_and_a_sensor_starts_an_actuator_s_clock():
    g = Graph(name="g", bodies=(_arm("a"), _arm("b", z=0.9)),
              actuators=(Gearmotor(name="m", joint="b_q", stall_nm=1.0, no_load_rad_s=10.0),),
              latches=(Latch(name="trigger", joint="a_q"),), sensors=(Encoder(name="e", joint="a_q"),))
    ctl = {"trigger.at": 0.2, "e.threshold": 0.05, "m.at": 0.1, "m.run": 5.0, "m.duty": 1.0}
    rows, prog, mc = _play(g, ctl, 1.5, triggers={"m": "e"}, gravity=True)
    assert abs(_at(rows, 0.15)[2][0]) < 1e-3 and not mc.held("trigger")
    fired = prog.fired["e"]  # the arm falls once released, passes 0.05 rad, and only then the motor's clock runs
    assert fired > 0.2
    assert _at(rows, fired + 0.05)[1][0] == 0.0 and _at(rows, fired + 0.15)[1][0] == 1.0


class Planned(Reach):
    """The toy world with controls System 1 starts from and a state it reads: how far the stick was swung."""

    def __init__(self):
        self.starts: list[dict] = []

    def control_names(self, body):
        return {"angle": (0.0, 1.0)}

    def solve(self, body, task, seed=0):
        length = body.parts["stick"].params["length"]

        def attempt(ctl, s):
            self.starts.append(dict(ctl))
            aimed = abs(ctl["angle"] - 0.5) < 0.1
            far = length >= task.data["d"]
            return Outcome(task=task.id, success=aimed and far, score=0.5 * far + 0.5 * (1 - abs(ctl["angle"] - 0.5)),
                           mechanism="" if far else "too_short", states={"swing": ctl["angle"]}, sim_s=0.01)

        space, start = {"angle": (0.0, 1.0)}, {"angle": 0.1}
        start, m = planned(body, space, start, 30)
        return local_search(attempt, space, start, m=m, seed=seed)


def test_system_1_starts_from_the_plan_and_a_plan_alone_is_trialled(tmp_path):
    w = Planned()
    plan = OperatingPlan(controls={"angle": 0.3})
    p = _p("params", [], plan=plan)
    proposer = Scripted([ProposalBatch(proposals=[p])])
    w.starts.clear()
    run(RunConfig(world="toy", iterations=1, plans=True, merge=False), tmp_path, proposer=proposer, world=w,
        log=lambda m: None)
    assert "# Operating plans" in proposer.prompts[0] and "Controls here: angle" in proposer.prompts[0]
    t = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])["trials"][0]
    assert t["tried"]  # the same parts with a plan are a different body
    assert {"angle": 0.3} in w.starts
    assert w.stock().key() != w.stock().model_copy(update={"plan": plan}).key()


def test_an_unknown_or_out_of_range_control_is_refused_and_a_plan_is_inherited():
    w = Planned()
    cat = {x.name: x for x in w.catalogue()}
    good = OperatingPlan(controls={"angle": 0.3})
    for plan, msg in [(OperatingPlan(controls={"thrust": 1.0}), "not a control here"),
                      (OperatingPlan(controls={"angle": 2.0}), "outside its range")]:
        with pytest.raises(ValueError, match=msg):
            candidate(w, w.stock(), {}, _p("params", [], plan=plan), cat, "x")
    with pytest.raises(ValueError, match="not a control here"):  # a world without controls takes intents alone
        candidate(Reach(), Reach().stock(), {}, _p("params", [], plan=good), cat, "x")
    assert candidate(Reach(), Reach().stock(), {}, _p("params", [], plan=OperatingPlan(intents=[Intent(state="x")])), cat,
                     "x").plan is not None
    b = candidate(w, w.stock(), {}, _p("params", [], plan=good), cat, "x")
    assert candidate(w, b, {}, _p("params", [Edit(slot="stick", params={"length": 0.3})]), cat, "y").plan == good


def test_the_first_attempt_is_the_plan_and_its_score_is_kept():
    out = local_search(lambda c, s: Outcome(task="t", success=False, score=c["x"]), {"x": (0.0, 1.0)}, {"x": 0.25}, m=5)
    assert out.nominal == 0.25 and out.score >= 0.25


def test_the_three_labels():
    reached = OperatingPlan(intents=[Intent(state="swing", lo=0.4)])
    o = Outcome(task="t", success=False, states={"swing": 0.5})
    assert OP.label(reached, o, "g")[0] == OP.CANNOT
    assert OP.label(reached, o.model_copy(update={"states": {"swing": 0.1}}), "g")[0] == OP.NOT_FOUND
    assert OP.label(reached, o.model_copy(update={"states": {}}), "g")[0] == OP.UNOBSERVED
    assert OP.label(None, o, "g")[0] == OP.UNOBSERVED
    other = OperatingPlan(intents=[Intent(state="swing", lo=0.9, tasks=["h"])])
    assert OP.label(other, o, "g")[0] == OP.UNOBSERVED  # the intent is for another group
    assert OP.label(None, o.model_copy(update={"mechanism": "not_found_by_search"}), "g")[0] == OP.NOT_FOUND


def test_each_failed_task_of_a_trial_is_labelled_and_the_prompt_says_which(tmp_path):
    w = Planned()
    plan = OperatingPlan(intents=[Intent(state="swing", lo=0.4, hi=0.6)])
    proposer = Scripted([ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.21})], plan=plan)])])
    run(RunConfig(world="toy", iterations=2, plans=True, merge=False), tmp_path, proposer=proposer, world=w,
        log=lambda m: None)
    t = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])["trials"][0]
    assert t["operation"]["counts"][OP.CANNOT] == 2 and "(reached)" in t["operation"]["tasks"]["d0.45_train0"]["states"][0]
    assert "Failing tasks, by what failed:" in proposer.prompts[1] and "d0.45_train0: unobserved" in proposer.prompts[1]


def test_a_new_concept_searches_longer(tmp_path):
    from tests.test_concepts import Patches, _dev

    class W(Patches):
        ms: list[int] = []

        def control_names(self, body):
            return {}

        def solve(self, body, task, seed=0):
            W.ms.append(body.plan.search if body.plan else 1)
            return super().solve(body, task, seed)

    _run(tmp_path, [[_rod("a", 0.21)], [_dev("C1", 0.22)]], world=W(), plans=True)
    assert 2 in W.ms and 1 in W.ms  # the stock body keeps its search, the concept's bodies double it
    W.ms.clear()
    _run(tmp_path / "off", [[_rod("a", 0.21)]], world=W())
    assert set(W.ms) == {1}  # without plans nothing changes


def test_with_the_switch_off_a_body_keeps_its_key_and_the_prompt_its_sections(tmp_path):
    w = Reach()
    assert '"(plan)"' not in w.stock().model_dump_json() and w.stock().plan is None
    proposer = Scripted([ProposalBatch(proposals=[_p("params", [Edit(slot="stick", params={"length": 0.3})])])])
    run(RunConfig(world="toy", iterations=1), tmp_path, proposer=proposer, world=w, log=lambda m: None)
    assert "# Operating plans" not in proposer.prompts[0]
    assert "operation" not in json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])["trials"][0]
