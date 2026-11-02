"""What a simulated attempt did that a real one could not, read from MuJoCo at every step.

MuJoCo's contacts are soft: two bodies may sink into each other until the constraint pushes back, so a
grip can hold an object partly by burying the jaws in it.  ``Penetration`` records, per step, the deepest
penetration of each kind of contact pair (robot, object, table), so an attempt can be judged on how far
it relied on overlap that rigid parts cannot have.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class Penetration:
    """A step hook: per step, the deepest penetration (mm) of each kind of pair in contact.  ``label`` names
    what a body is ("robot", "object", "table"); a kind is the two labels in sorted order."""

    def __init__(self, model, label: Callable[[int], str]):
        self.model = model
        self.label = label
        self.steps: list[dict[str, float]] = []

    def __call__(self, data) -> None:
        worst: dict[str, float] = {}
        bodies = self.model.geom_bodyid
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            kind = "-".join(sorted((self.label(int(bodies[c.geom1])), self.label(int(bodies[c.geom2])))))
            worst[kind] = max(worst.get(kind, 0.0), -1000.0 * float(c.dist))
        self.steps.append(worst)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per kind of pair: the peak, and the 95th percentile over the steps it was in contact (mm)."""
        out: dict[str, dict[str, float]] = {}
        for kind in sorted({k for s in self.steps for k in s}):
            d = np.array([s[kind] for s in self.steps if kind in s])
            out[kind] = {"peak_mm": round(float(d.max()), 2), "p95_mm": round(float(np.percentile(d, 95)), 2),
                         "steps": int(len(d))}
        return out
