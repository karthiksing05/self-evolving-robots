"""Feedback from the designed product itself (docs/framework-goal.md): a world whose verdicts come from outside.

The loop reads only what ``World.solve`` returns, so it can't tell a simulator from a rig on a bench.  ``ProductFeedback``
keeps a world's brief, kit and laws, and takes each verdict from a command: it gets ``{"body", "task", "seed"}`` as JSON on
stdin, and answers an ``Outcome`` as JSON on stdout.  The command can drive hardware, ask a person or call a lab.  A product
that gives no answer stops the run: no verdict is ever invented.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Any

from sea.evolve.core import Body, Outcome, Task


class ProductFeedback:
    def __init__(self, world: Any, command: Sequence[str]):
        self.world, self.command = world, list(command)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.world, name)

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        ask = json.dumps({"body": body.model_dump(mode="json"), "task": task.model_dump(mode="json"), "seed": seed})
        done = subprocess.run(self.command, input=ask, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"the product gave no verdict on {task.id}: {done.stderr.strip()[-500:]}")
        return Outcome.model_validate_json(done.stdout).model_copy(update={"task": task.id})
