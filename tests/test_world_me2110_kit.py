"""ME 2110 from the kit (D-094): a robot is one graph, operated by its own controls, judged by general laws."""

from __future__ import annotations

import pytest

from sea.evolve.core import Body, Part, World
from sea.mech import graph as G
from sea.sim import kit_match as K
from sea.sim import me2110_model as MM
from sea.worlds.me2110_kit import IN, KitWorld, _module

W = KitWorld()
V = MM.Variant(barrier=7.0, rpm=3.0, mu=0.5, phase=0.4, others=(6.0, 8.0, 8.75))


def _body(g: G.Graph) -> Body:
    return Body(parts={"robot": Part(type="built", mechanism=g.model_dump_json())})


def _rover() -> G.Graph:
    """Two gearmotors on wheels and a caster: written here for the test, not offered to System 2."""
    c = _module("chassis", {"length_in": 11.0, "width_in": 12.0})
    root = c.bodies[0]
    parts = tuple(p for p in root.parts if not p.name.startswith("foot")) + (
        G.Sphere(name="caster", pos=(4.5 * IN, 0.0, 0.5 * IN), radius=0.5 * IN, mu=0.02),
        G.Box(name="caster_post", pos=(4.5 * IN, 0.0, 0.75 * IN), half=(0.3 * IN, 0.3 * IN, 0.3 * IN)),
        G.Box(name="axle_l", pos=(-2.75 * IN, 6.3 * IN, 1.25 * IN), half=(0.3 * IN, 0.35 * IN, 0.3 * IN)),
        G.Box(name="axle_r", pos=(-2.75 * IN, -6.3 * IN, 1.25 * IN), half=(0.3 * IN, 0.35 * IN, 0.3 * IN)))
    wheels = tuple(G.Body(name=f"wheel{k}", parent="robot", pos=(-2.75 * IN, sy * 6.9 * IN, 1.5 * IN),
                          joints=(G.Joint(name=f"w{k}", type="hinge", axis=(0.0, 1.0, 0.0), armature=0.0005),),
                          parts=(G.Wheel(name=f"tyre{k}", pos=(0.0, 0.0, 0.0), radius=1.5 * IN, half_width=0.375 * IN,
                                         quat=(0.7071068, 0.7071068, 0.0, 0.0), kg=None),))
                   for k, sy in enumerate((1, -1)))
    return G.Graph(name="rover", bodies=(root.model_copy(update={"parts": parts}), *wheels), actuators=(
        G.Gearmotor(name="left", joint="w0", stall_nm=MM.MOTOR_TS, no_load_rad_s=MM.MOTOR_W0),
        G.Gearmotor(name="right", joint="w1", stall_nm=MM.MOTOR_TS, no_load_rad_s=MM.MOTOR_W0)))


def test_protocol_and_a_stock_that_scores_nothing():
    assert isinstance(W, World)
    tasks = W.tasks()
    assert {t.group for t in tasks} == {"match"} and len([t for t in tasks if t.split == "test"]) == 4
    assert W.catalogue() == [] and W.check(W.stock()) == []
    r = K.KitMatch(_module("chassis", {"length_in": 11.0, "width_in": 12.0}), V, {}, horizon=40.0).run()
    assert sum(r.points.values()) == 0 and not r.laws


def test_a_rover_leaves_the_dock_from_its_program_alone():
    g = _rover()
    assert W.check(_body(g)) == []
    ctl = {"left.duty": 0.5, "right.duty": 0.5, "left.run": 2.0, "right.run": 2.0}
    r = K.KitMatch(g, V, ctl, horizon=40.0).run()
    assert r.points["activate"] == 11 and not r.laws
    assert sum(r.points.values()) == 11


def test_system_1_finds_it_and_the_score_is_a_mean_of_matches(monkeypatch):
    import sea.worlds.me2110_kit as M

    monkeypatch.setattr(M, "ATTEMPTS", 3)
    b = _body(_rover()).model_copy(update={"plan": None})
    plan = {"left.duty": 0.5, "right.duty": 0.5, "left.run": 2.0, "right.run": 2.0}
    from sea.evolve.core import OperatingPlan

    b = b.model_copy(update={"plan": OperatingPlan(controls=plan)})
    o = W.solve(b, W.tasks()[0], 0)
    assert o.points == pytest.approx(11.0) and o.states["points.activate"] == pytest.approx(11.0)
    assert o.attempts == 3 + 3 and "over 3 matches" in o.detail and o.nominal is not None


@pytest.mark.parametrize("fault,msg", [
    (lambda g: g.model_copy(update={"actuators": (*g.actuators, *[G.Gearmotor(
        name=f"m{k}", joint="w0", stall_nm=1.0, no_load_rad_s=10.0) for k in range(1)])}), "3 gearmotors"),
    (lambda g: g.model_copy(update={"bodies": (g.bodies[0].model_copy(update={"parts": (*g.bodies[0].parts, G.Box(
        name="lump", pos=(0.0, 0.0, 0.1), half=(0.01, 0.01, 0.01), kg=0.001))}), *g.bodies[1:])}), "state their mass"),
    (lambda g: g.model_copy(update={"bodies": (g.bodies[0].model_copy(update={"parts": (*g.bodies[0].parts, G.Box(
        name="wing", pos=(0.0, 0.5, 0.05), half=(0.05, 0.5, 0.01)))}), *g.bodies[1:])}), "go/no-go box"),
    (lambda g: g.model_copy(update={"bodies": (g.bodies[0].model_copy(update={"parts": (*g.bodies[0].parts, G.Box(
        name="float", pos=(0.0, 0.0, 0.3), half=(0.01, 0.01, 0.01)))}), *g.bodies[1:])}), "floating"),
    (lambda g: g.model_copy(update={"bodies": (g.bodies[0], *g.bodies[1:], G.Body(
        name="arm", parent="robot", pos=(0.0, 0.0, 0.1), joints=(G.Joint(name="sw", type="hinge", axis=(0.0, 1.0, 0.0),
                                                                         range=(0.0, 3.0)),),
        parts=(G.Box(name="bar", pos=(-0.05, 0.0, 0.0), half=(0.08, 0.01, 0.01)),)), G.Body(
        name="post", parent="robot", pos=(0.0, 0.0, 0.0), parts=(G.Box(name="blk", pos=(0.04, 0.0, 0.1),
                                                                        half=(0.01, 0.01, 0.01)),)),),
        "latches": (G.Latch(name="l", joint="sw"),)}), "pass"),
])
def test_each_law_refuses_a_planted_fault(fault, msg):
    errs = W.check(_body(fault(_rover())))
    assert any(msg in e for e in errs), errs


def test_a_robot_composed_by_a_patch_runs_and_the_tools_see_its_graph(tmp_path):
    from sea.evolve.core import Idea, Mechanism

    b = W.build_idea(W.stock(), Idea(name="plated", description="d", slot="robot", mechanism=Mechanism(
        module="self", patch=[{"op": "add", "path": "bodies/robot/parts", "value": {
            "kind": "box", "name": "rail", "pos": [0.0, 0.0, 0.06], "half": [0.05, 0.01, 0.01]}}])), "b1")
    assert W.check(b) == [] and '"rail"' in W.mechanism_text(b) and W.read(b, "mass_kg", {}, None) > 0.3
    assert W.read(b, "kit_left", {"kind": "gearmotors"}, None) == 2 and W.read(b, "height_in", {}, None) > 1
    assert "y0" in W.control_names(_body(_rover())) and "left.duty" in W.control_names(_body(_rover()))
    assert "Kinds:" in W.modules_text() and "chassis" in W.module_names() and "Gearmotor" in W.modules_text()
    assert W.bom(b)[0].item == "plywood_in2" and W.manifest()["budget"]["usd"] == 120


def test_the_loop_keeps_a_kit_robot_on_its_points(tmp_path, monkeypatch):
    """The whole path: System 2 writes a robot as a patch with a plan, it opens a concept, a match scores it, it is kept."""
    import json

    import sea.worlds.me2110_kit as M
    from sea.evolve.core import Idea, Mechanism, OperatingPlan, ProposalBatch
    from sea.evolve.llm import Scripted
    from sea.evolve.loop import RunConfig, run
    from sea.worlds.me2110_kit import _chassis
    from tests.test_evolve import _p

    monkeypatch.setattr(M, "ATTEMPTS", 1)
    monkeypatch.setitem(M.MODULES, "rover", (lambda p: _rover(), {}))  # the test's robot, not System 2's library
    monkeypatch.setitem(M.MODULES, "chassis", (_chassis, M.MODULES["chassis"][1]))
    plan = OperatingPlan(controls={"left.duty": 0.5, "right.duty": 0.5, "left.run": 2.0, "right.run": 2.0})
    prop = _p("idea", idea=Idea(name="rover", description="d", slot="robot", mechanism=Mechanism(module="rover")), plan=plan)
    proposer = Scripted([ProposalBatch(proposals=[prop])])
    s = run(RunConfig(world="me2110_kit", iterations=1, concepts=True, plans=True, requirements=True), tmp_path,
            proposer=proposer, world=W, log=lambda m: None)
    assert "The current body scores 0.0 of 347 points" in proposer.prompts[0] and "# Concepts" in proposer.prompts[0]
    t = json.loads((tmp_path / "iterations.jsonl").read_text().splitlines()[0])["trials"][0]
    assert t["result"].startswith("KEPT: 0.0 -> 11.0 points"), t["result"]
    assert t["concept"] == "C1" and t["operation"]["counts"]["unobserved"] == 1
    assert s["points"]["train"] == 11.0 and s["points"]["test"] > 0 and s["points"]["stock_test"] == 0.0
