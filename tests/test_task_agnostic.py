"""The harness knows no task (D-107): nothing general imports a world or names one, and what System 2 reads in one
world names nothing of another."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

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
