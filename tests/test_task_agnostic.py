"""The harness knows no task (D-107): nothing general imports a world or names one, and what System 2 reads in one
world names nothing of another."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from sea.evolve.core import Outcome
from sea.evolve.loop import make_world
from sea.evolve.prompt import SYSTEM, build
from sea.evolve.resources import start
from sea.evolve.tools import specs
from tests.test_evolve import Reach

SRC = Path(__file__).parent.parent / "src" / "sea"
GENERAL = sorted((SRC / "evolve").glob("*.py"))
MECH = [SRC / "mech" / f"{n}.py" for n in ("base", "elements", "kinematics", "patch", "program", "runtime")]
WORDS = {"gripper": r"gripper|\bjaws?\b|\bfingers?\b|fingertip|wrist|grasp|mustard|\bycb\b|so-?101",
         "mobile": r"mecanum|rocker.bogie|sonar", "phonology": r"formant|vowel|glottis|larynx",
         "me2110": r"mousetrap|me ?2110"}
EVERY = "|".join(WORDS.values())
OTHERS = {"mobile": ["gripper", "phonology", "me2110"], "phonology": ["mobile", "me2110"],  # a mouth has a jaw
          "mobile_kit": ["gripper", "phonology", "me2110"], "phonology_kit": ["mobile", "me2110"],
          "me2110": ["gripper", "mobile", "phonology"], "me2110_kit": ["gripper", "mobile", "phonology"],
          "gripper": ["mobile", "phonology", "me2110"]}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


@pytest.mark.parametrize("path", GENERAL + MECH, ids=lambda p: p.name)
def test_general_code_imports_no_world_of_its_own(path):
    bad = {m for m in _imports(path) if m.startswith("sea.worlds") and m != "sea.worlds"}
    assert not bad


@pytest.mark.parametrize("path", GENERAL, ids=lambda p: p.name)
def test_the_loop_names_no_task(path):
    assert not re.search(EVERY, path.read_text(), re.I), path


def test_no_graph_is_on_a_robot_by_default():
    for path in (SRC / "mech").glob("*.py"):
        assert '"gripper"' not in path.read_text(), path


def test_system_2s_tools_and_instructions_name_no_task():
    text = SYSTEM + json.dumps([[t["name"], t["description"]] for t in specs()])
    assert not re.search(EVERY, text, re.I)


@pytest.mark.parametrize("name", list(OTHERS))
def test_a_prompt_names_nothing_of_another_world(name):
    w = make_world(name)
    body, tasks = w.stock(), [t for t in w.tasks() if t.split == "train"]
    outs = {t.id: Outcome(task=t.id, success=False, mechanism="m", detail="d") for t in tasks}
    text = build(w, body, tasks, outs, set(), [], [], start(w.bom(body)), w.manifest(), 0, 1, rules=list(w.brief_rules()))
    for other in OTHERS[name]:
        assert not re.search(WORDS[other], text, re.I), (name, other)


def test_a_new_world_prompt_names_no_world():
    w = Reach()
    body, tasks = w.stock(), [t for t in w.tasks() if t.split == "train"]
    outs = {t.id: Outcome(task=t.id, success=False, mechanism="m", detail="d") for t in tasks}
    assert not re.search(EVERY, build(w, body, tasks, outs, set(), [], [], start(w.bom(body)), w.manifest(), 0, 1), re.I)


def test_a_failure_is_searched_other_ways_before_it_counts():
    from sea.evolve.search import local_search

    seen = []

    def run(c, s):
        seen.append(c["x"])
        return Outcome(task="t", success=False, score=1.0 - c["x"])

    out = local_search(run, {"x": (0.0, 1.0)}, {"x": 0.9}, m=8, confirm=0, ways=[{"x": 0.5}])
    assert out.ways == 1 + 1 + 3 and out.attempts == len(seen) == 8 + 4 * 4 and out.nominal == pytest.approx(0.1)
    assert seen[8] == 0.5  # the plan's way comes first, before the spread starts


def test_the_first_way_runs_as_if_it_were_alone():
    from sea.evolve.search import local_search

    calls: list[list[float]] = [[], []]

    def run(k):
        def f(c, s):
            calls[k].append(c["x"])
            return Outcome(task="t", success=False, score=c["x"])
        return f

    local_search(run(0), {"x": (0.0, 1.0)}, {"x": 0.3}, m=6, confirm=0, seed=5, other=0)
    local_search(run(1), {"x": (0.0, 1.0)}, {"x": 0.3}, m=6, confirm=0, seed=5)
    assert calls[1][:6] == calls[0]


def test_each_way_draws_its_own_samples():
    from sea.evolve.search import local_search

    seen = []

    def run(c, s):
        seen.append(c["x"])
        return Outcome(task="t", success=False, score=0.0)

    local_search(run, {"x": (0.0, 1.0)}, {"x": 0.5}, m=8, confirm=0, ways=[{"x": 0.5}], other=0)
    assert seen[1:4] != seen[9:12]


def test_a_plan_without_ways_keeps_its_body_key():
    from sea.evolve.core import OperatingPlan

    b = Reach().stock().model_copy(update={"plan": OperatingPlan(controls={"a": 1.0})})
    with_ways = b.model_copy(update={"plan": OperatingPlan(controls={"a": 1.0}, ways=[{"a": 2.0}])})
    assert b.key() != with_ways.key() and "ways" not in b.plan.model_dump(exclude={"ways"})


def _slider_crank():
    from sea.mech.base import WORLD
    from sea.mech.build import compile as compile_graph
    from sea.mech.elements import fragment, merge
    from tests.mech.test_machine_elements import BASE

    g = merge(BASE, fragment("slider_crank", "s", "frame", params={"crank_m": 0.03, "rod_m": 0.1, "start_rad": 0.5}))
    spec = WORLD.model()
    compile_graph(spec, g, WORLD)
    return g, spec.compile()


def test_any_printed_bar_on_a_loop_is_judged_without_any_robot_named():
    from sea.mech.build import printed_members

    g, m = _slider_crank()
    members = {x.name: x for x in printed_members(g, m)}
    assert members["s_rodbar"].kind == "strut" and members["s_crankbar"].kind == "crank"
    assert members["s_rodbar"].length == pytest.approx(g.loops[0].point[0])


def test_a_plywood_or_steel_bar_is_not_read():
    from sea.mech.build import _bar_section
    from sea.mech.graph import Box

    pin = np.array([0.1, 0.0, 0.0])
    bar = dict(name="b", pos=(0.05, 0.0, 0.0), half=(0.05, 0.003, 0.006))
    assert _bar_section(Box(**bar), pin, None) is not None
    assert _bar_section(Box(**bar, material="plywood"), pin, None) is None


def test_a_broken_member_disqualifies_a_kit_match(monkeypatch):
    from sea.sim import kit_match as K
    from sea.worlds.me2110 import VARIANTS, law_label
    from sea.worlds.me2110_kit import graph_of

    w = make_world("me2110_kit")
    k = K.KitMatch(graph_of(w.stock()), VARIANTS["train"][0], {}, seed=0, horizon=0.02)
    k.members = type("M", (), {"read": lambda s, d: None, "broken": lambda s: "unphysical: the printed bar carried 99 MPa"})()
    r = k.run()
    assert law_label(r) == "printed_part_overloaded"


def test_the_kit_worlds_measure_up_to_the_worlds_they_rebuild():
    new, old = make_world("mobile_kit"), make_world("mobile")
    for name in ("kitchen", "bedroom", "porch", "under_sofa"):
        task = next(t for t in new.tasks() if t.id == f"{name}_train0")
        assert new.solve(new.stock(), task, 0).mechanism == old.solve(old.stock(), task, 0).mechanism, name


@pytest.mark.parametrize("name", ["mobile_kit", "phonology_kit"])
def test_a_rebuilt_world_starts_lawful_and_refuses_what_cannot_be_bought(name):
    from sea.evolve.core import Part
    from sea.mech.world import SLOT, graph_of

    w = make_world(name)
    assert w.check(w.stock()) == [] and w.bom(w.stock())
    g = graph_of(w.stock())
    strong = g.model_copy(update={"actuators": tuple(a.model_copy(update={"stall_nm": 99.0}) for a in g.actuators)})
    bad = w.stock().model_copy(update={"parts": {SLOT: Part(type="built", mechanism=strong.model_dump_json())}})
    assert w.check(bad)


class _Lever:
    """A new world in a few lines: a servo's bar must reach a mark, with no line of general code knowing of it."""

    def __new__(cls):
        from sea.evolve.core import Body, BomLine, Part, Task
        from sea.evolve.search import local_search, planned, ways_of
        from sea.mech import graph as G
        from sea.mech.world import SLOT, GraphWorld, graph_of

        def bar(p):
            return G.Graph(name="bar", bodies=(G.Body(name="frame", parent="world", parts=(
                G.Box(name="post", pos=(0.0, 0.0, 0.0), half=(0.005, 0.005, 0.005)),)),
                G.Body(name="arm", parent="frame", joints=(G.Joint(name="q", type="hinge", axis=(0.0, 1.0, 0.0), range=(0.0, 1.5)),),
                       parts=(G.Box(name="beam", pos=(p["length"] / 2, 0.0, 0.0), half=(p["length"] / 2, 0.003, 0.003)),))),
                actuators=(G.Servo(name="s", joint="q", stall_nm=1.0),))

        class World(GraphWorld):
            name, description, objective = "lever", "A bar reaches a mark.", "groups"
            frame = "A graph in metres."
            modules = {"bar": (bar, {"length": (0.05, 0.3)})}

            def tasks(self):
                return [Task(id=f"mark_{s}", split=s, group="mark", facts="reach 0.2 m", data={}) for s in ("train", "test")]

            def catalogue(self):
                return []

            def stock(self):
                return Body(parts={SLOT: Part(type="built", mechanism=bar({"length": 0.1}).model_dump_json())})

            def solve(self, body, task, seed=0):
                ln = graph_of(body).bodies[1].parts[0].half[0] * 2
                space, start = {"s": (0.0, 1.5)}, {"s": 0.1}

                def run(c, k):
                    far = ln * math.cos(c["s"])
                    return Outcome(task=task.id, success=far >= 0.2, score=min(far / 0.2, 1.0), mechanism="" if far >= 0.2 else "short")

                start, m = planned(body, space, start, 8)
                return local_search(run, space, start, m=m, seed=seed, ways=ways_of(body, space))

            def brief(self, body, tasks):
                return "a bar"

            def principles(self):
                return []

            def brief_rules(self):
                return []

            def bom(self, body):
                return [BomLine(item="print:PLA", qty=1.0, slot=SLOT)]

            def check(self, body):
                return []

            def manifest(self):
                return {"budget": {"usd": 10, "lead_days": 1, "fab_minutes": 100},
                        "fabrication": {"print:PLA": {"minutes_per_unit": 1.0, "unit": "g", "doc": "printer"}}, "items": {}}

        return World()


def test_a_world_nobody_planned_for_runs_the_whole_loop(tmp_path):
    from sea.evolve.core import Idea, Mechanism, ProposalBatch
    from sea.evolve.llm import Scripted
    from sea.evolve.loop import RunConfig, run
    from tests.test_evolve import _p

    w = _Lever()
    longer = _p("idea", idea=Idea(name="long_bar", description="a longer bar", slot="robot",
                                  mechanism=Mechanism(module="bar", params={"length": 0.25})))
    out = run(RunConfig(world="lever", iterations=2, stripped=True), tmp_path, proposer=Scripted([ProposalBatch(proposals=[longer])]),
              world=w, log=lambda m: None)
    assert out["curve"][0]["held_groups"] == 0 and out["train_groups"] == 1
    assert "long_bar" in out["final_body"]["parts"]["robot"]["mechanism"]
