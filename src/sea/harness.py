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
    "verifier": ("judge", ("src/sea/verify/verdicts.py", "src/sea/compose/checks.py")),
    "reward": ("judge", ("src/sea/verify/coverage.py", "src/sea/verify/reward.py")),
    "limits": ("judge", ("configs/limits.yaml",)),
    "panels": ("judge", ("configs/panels.yaml", "src/sea/sim/panels.py")),
    "acceptance": ("judge", ("src/sea/spec/posture.py",)),
    "physics": ("world", ("src/sea/sim/build_model.py", "src/sea/sim/controls.py")),
    "kinematics": ("world", ("src/sea/control/ik.py", "src/sea/control/frames.py", "src/sea/control/reach.py",
                             "configs/ik_seeds.yaml")),
    "primitives": ("skill", ("configs/primitives.yaml", "src/sea/control/motor.py", "src/sea/control/executor.py",
                             "src/sea/control/local_search.py", "src/sea/compose/typecheck.py")),
    "tools": ("skill", ("src/sea/design/tools.py", "src/sea/design/build.py", "src/sea/design/space.py")),
    "ledger": ("policy", ("src/sea/spec/ledger.py",)),
    "proposer": ("policy", ("src/sea/propose/prompt.py", "src/sea/propose/ladder.py", "src/sea/propose/targets.py",
                            "src/sea/propose/baselines.py")),
    "memory": ("policy", ("src/sea/memory/concepts.py", "src/sea/memory/embed.py")),
    "loop": ("policy", ("src/sea/loop.py", "src/sea/sim/env.py")),
}


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
