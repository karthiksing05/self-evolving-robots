"""A kit talker's servo is a solid of a real size on what carries its hinge (D-127)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.evolve.core import Outcome, Part
from sea.mech import graph as G
from sea.mech.world import SLOT
from sea.worlds import graph_cad
from sea.worlds import phonology as P
from sea.worlds.phonology_kit import CASE_MM, PhonologyKit, _case, _laws

ARM = 0.05
TALL = CASE_MM[2] / 2000


def staggered(places=((0.11, -1), (0.085, 1), (0.06, -1), (0.035, 1), (0.012, -1), (-0.005, 1)), case=True) -> G.Graph:
    """Six levers from below, on alternate sides of the tube, so that cases 28.5 mm long fit plates 2.5 cm apart."""
    frame = [G.Box(name="base", pos=(0.05, 0.0, -0.033), half=(0.1, 0.05, 0.003))]
    levers = []
    for k, (x, side) in enumerate(places):
        y = side * ARM
        frame.append(G.Box(name=f"post{k}", pos=(x, y, -0.0275), half=(0.003, 0.003, 0.0035)))
        if case:
            frame.append(_case(f"case{k}", (x + 0.003 + TALL, y, -0.0185), "x"))
        levers.append(G.Body(name=f"l{k}", parent="frame", pos=(x, y, -0.024),
                             joints=(G.Joint(name=f"l{k}_q", type="hinge", axis=(-side * 1.0, 0.0, 0.0), range=(0.0, 1.1)),),
                             parts=(G.Box(name=f"l{k}_bar", pos=(0.0, -y / 2, 0.0), half=(0.002, ARM / 2, 0.002)),
                                    G.Box(name=f"l{k}_pad", pos=(0.0, -y, 0.0), half=(0.009, 0.018, 0.002)))))
    bodies = (G.Body(name="frame", parent="world", parts=tuple(frame)), G.Body(name="glottis", parent="frame", pos=(0.135, 0.0, 0.0)),
              G.Body(name="lip_end", parent="frame", pos=(-0.01, 0.0, 0.0)), *levers)
    return G.Graph(name="staggered", bodies=bodies, excludes=tuple(("frame", b.name) for b in levers),
                   actuators=tuple(G.Servo(name=b.name, joint=f"{b.name}_q", stall_nm=0.216) for b in levers))


def _says(g: G.Graph) -> str:
    return " | ".join(_laws(g.model_dump_json()))


def test_six_servos_can_be_laid_out_around_the_tube():
    assert _says(staggered()) == ""


def test_stock_answers_to_the_law():
    w = PhonologyKit()
    assert w.check(w.stock()) == []


def test_a_servo_with_no_case_is_refused():
    assert "has no case" in _says(staggered(case=False))


def test_cases_in_one_line_pass_through_each_other():
    assert "passes through" in _says(staggered(places=tuple((0.11 - 0.012 * k, -1) for k in range(6))))


def test_a_case_of_the_wrong_size_is_not_a_servo():
    g = staggered()
    wrong = g.bodies[0].parts[2].model_copy(update={"half": (0.005, 0.005, 0.005)})
    frame = g.bodies[0].model_copy(update={"parts": (*g.bodies[0].parts[:2], wrong, *g.bodies[0].parts[3:])})
    assert "l0 has no case" in _says(g.model_copy(update={"bodies": (frame, *g.bodies[1:])}))


def test_the_cad_draws_the_tube_and_the_cases():
    w = PhonologyKit()
    body = w.stock().model_copy(update={"parts": {SLOT: Part(type="built", mechanism=staggered().model_dump_json())}})
    parts = graph_cad.talker(body).parts
    assert sum(p.material == "servo" for p in parts) == 6
    tube = next(p for p in parts if p.name == "tube")
    assert tube.material == "silicone" and tube.mesh.extents[0] == pytest.approx(145.0, abs=0.5)


def test_a_later_draw_is_another_sample_and_leaves_the_first_alone():
    world, body = PhonologyKit(), PhonologyKit().stock()
    first = world.babble(body).x.copy()
    assert not np.array_equal(world.babble(body, 1).x, first)
    assert np.array_equal(world.babble(body).x, first)
    assert np.array_equal(PhonologyKit().babble(body).x, first)


def test_a_look_follows_only_a_failure(monkeypatch):
    """Three looks at a task System 1 fails cost more than one; the first look's verdict and attempts are today's."""
    world = PhonologyKit()
    body = world.stock()
    tasks = {t.id: t for t in world.tasks()}
    fails, passes = tasks["iy_train0"], tasks["ae_train0"]
    three = world.solve(body, fails), world.solve(body, passes)
    monkeypatch.setattr(P, "LOOKS", 1)
    one = world.solve(body, fails), world.solve(body, passes)
    assert not three[0].success and not one[0].success and three[0].attempts > one[0].attempts
    assert three[1].success and one[1].success and three[1].attempts == one[1].attempts


def test_a_syllable_the_map_cannot_pair_is_out_of_reach_not_a_search_failure():
    world = PhonologyKit()
    body = world.stock()
    task = next(t for t in world.tasks() if t.id == "ka_train0")
    out = world.solve(body, task)
    assert not out.success and out.mechanism == "syllable_out_of_reach"
    assert "pass together" in out.detail


def test_a_pair_that_only_passes_standing_still_is_out_of_reach_not_a_search_failure(monkeypatch):
    world = PhonologyKit()
    body = world.stock()
    task = next(t for t in world.tasks() if t.id == "ka_train0")
    out = Outcome(task=task.id, success=False, score=0.5, detail="")
    monkeypatch.setattr(world, "_pairs", lambda bab, sides: ([(0, 1), (2, 3)], np.array([0.9, 0.5])))
    monkeypatch.setattr(world, "_timed", lambda *a: (1.4, "the tract takes 70 ms to reach the vowel"))
    mech, why = world._diagnose(body, task, out)
    assert mech == "syllable_out_of_reach" and "70 ms" in why and "none of the 2" in why
    monkeypatch.setattr(world, "_timed", lambda *a: (0.9, "the tract takes 45 ms to reach the vowel"))
    assert world._diagnose(body, task, out)[0] != "syllable_out_of_reach"
