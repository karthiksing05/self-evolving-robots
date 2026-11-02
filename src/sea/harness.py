"""Harness fingerprint: which version of each fundamental produced a run (docs/iteration.md).

Each component hashes the source files and configs that define it.  A run writes the fingerprint
to ``runs/<id>/harness.json``; ``sea report`` refuses to pool runs whose *judge* components differ,
because then a change in the numbers may be a change in the ruler, not in the robot.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "sea"

# component -> (tier, files).  Tiers follow docs/iteration.md:
#   judge   - decides what counts as success or cost (changing it changes the ruler)
#   world   - the simulated physics and kinematics the robot acts in
#   skill   - what the robot can do before any learning (primitives, tools, executor)
#   policy  - how boldly and where the loop searches (posture, ledger prices, prompt)
COMPONENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    # contracts.py holds the HardLimits and Budgets defaults, the verdict and contact enums and the
    # coverage bins; build.py applies the hard limits to a body and prices its print (D-028).
    "contracts": ("judge", ("src/sea/contracts.py",)),
    # verify/goals.py is the goal-test vocabulary above one control: the planner composes, never writes it (D-037)
    "verifier": ("judge", ("src/sea/verify/verdicts.py", "src/sea/compose/checks.py", "src/sea/verify/goals.py")),
    "reward": ("judge", ("src/sea/verify/coverage.py", "src/sea/verify/reward.py")),
    "limits": ("judge", ("configs/limits.yaml", "configs/limits_grasp.yaml")),
    "panels": ("judge", ("configs/panels.yaml", "src/sea/sim/panels.py")),
    # the grasp world's task distribution (docs/gripper-experiment.md): the same ruler, other tasks
    "grasp_world": ("judge", ("configs/grasp.yaml", "src/sea/sim/grasp_world.py", "src/sea/sim/objects.py")),
    "acceptance": ("judge", ("src/sea/spec/posture.py",)),
    "build": ("judge", ("src/sea/design/build.py",)),
    "physics": ("world", ("src/sea/sim/build_model.py", "src/sea/sim/controls.py", "src/sea/sim/jaws.py")),
    # what the robot can feel; the verifier never reads it, so it is world, not judge (D-031)
    "sensors": ("world", ("src/sea/sim/sensors.py", "configs/sensors.yaml")),
    # the fixed pick that measures the gripper's capability table (V1); an instrument, not a skill
    "grasp_probe": ("world", ("src/sea/sim/grasp_probe.py",)),
    "kinematics": ("world", ("src/sea/control/ik.py", "src/sea/control/frames.py", "src/sea/control/reach.py",
                             "configs/ik_seeds.yaml")),
    "primitives": ("skill", ("configs/primitives.yaml", "src/sea/control/motor.py", "src/sea/control/executor.py",
                             "src/sea/control/local_search.py", "src/sea/compose/typecheck.py",
                             "src/sea/compose/dsl.py")),
    "tools": ("skill", ("src/sea/design/tools.py", "src/sea/design/space.py", "src/sea/design/grip.py",
                         "src/sea/design/fingers.py")),
    "ledger": ("policy", ("src/sea/spec/ledger.py",)),
    "proposer": ("policy", ("src/sea/propose/prompt.py", "src/sea/propose/ladder.py", "src/sea/propose/targets.py",
                            "src/sea/propose/baselines.py", "src/sea/propose/llm.py")),
    "memory": ("policy", ("src/sea/memory/concepts.py", "src/sea/memory/embed.py", "src/sea/compose/library.py")),
    "loop": ("policy", ("src/sea/loop.py", "src/sea/sim/env.py")),
    "network": ("policy", ("src/sea/plan/network.py",)),
    # the world-agnostic evolution core (D-041): its contracts and resource check decide what a body change
    # is and whether the bench can build it; the loop, prompt and backends decide how boldly it searches
    "evolve": ("judge", ("src/sea/evolve/core.py", "src/sea/evolve/resources.py")),
    "evolve_loop": ("policy", ("src/sea/evolve/loop.py", "src/sea/evolve/prompt.py", "src/sea/evolve/llm.py",
                               "src/sea/evolve/search.py")),
    # each world's tasks, physics and parts catalogue: the ruler for runs in that world (D-042)
    "world_gripper": ("judge", ("src/sea/worlds/gripper.py",)),
    "world_mobile": ("judge", ("src/sea/worlds/mobile.py",)),
    "world_phonology": ("judge", ("src/sea/worlds/phonology.py",)),
}

# Modules that read runs but never shape one; every other module under src/sea belongs to a component.
# The CAD modules draw a run's bodies for people (D-049): a picture is not part of the ruler.
TOOLING = ("src/sea/cli.py", "src/sea/harness.py", "src/sea/report.py", "src/sea/design/cad.py",
           "src/sea/worlds/cad.py", "src/sea/worlds/film.py",
           # the mechanism verifier measures and decides nothing yet (D-051); it joins the judge when its laws do
           "src/sea/mechanism/geometry.py", "src/sea/mechanism/audit.py")


def _hash_files(paths: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for rel in paths:
        p = ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:12]


def fingerprint() -> dict[str, dict[str, str]]:
    """``{component: {"tier": ..., "hash": ...}}`` for the working tree."""
    return {name: {"tier": tier, "hash": _hash_files(files)} for name, (tier, files) in COMPONENTS.items()}


def judge_key(fp: dict[str, dict[str, str]]) -> str:
    """One hash over the judge-tier components: runs are comparable only if these match."""
    parts = sorted(f"{k}={v['hash']}" for k, v in fp.items() if v.get("tier") == "judge")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def diff(a: dict[str, dict[str, str]], b: dict[str, dict[str, str]]) -> dict[str, str]:
    """Components whose hash differs, with their tier."""
    return {k: a.get(k, b.get(k, {})).get("tier", "?") for k in sorted(set(a) | set(b))
            if a.get(k, {}).get("hash") != b.get(k, {}).get("hash")}
