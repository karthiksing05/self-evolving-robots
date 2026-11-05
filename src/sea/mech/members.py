"""Printed members under a mechanism's own loads, for any graph on any robot (D-103, D-109).

A loop's pin force passes through what spans it: each such member is judged at that force, held ``HOLD_STEPS`` steps,
in stress and, pushed, against its Euler load over ``PRINT_FACTOR``.  Only printed PLA has sourced values here.  A
plywood or steel part is not read: no citable modulus for the plywood has been found, and steel is far from these loads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

HOLD_STEPS = 10  # a lift or load counts once it lasts 20 ms, as the force limit's does: not an impact's spike

# Printed PLA (D-100): flat prints' measured flexural strength and modulus (ISO 178, Prusa Mini, 2026; a 90 degree
# build reaches 21.3 MPa), over a 1.5 factor.
PLA_FLEX_PA, PLA_E_PA, PRINT_FACTOR = 47.6e6, 1.808e9, 1.5
PLA_ALLOW_PA = PLA_FLEX_PA / PRINT_FACTOR


@dataclass(frozen=True)
class Member:
    """A printed member carrying a loop's pin force (D-103).  A ``strut`` runs from its body's origin to the pin, in
    ``plates`` equal plates or webs; a ``crank`` reaches from a hub of radius ``root`` to the pin.  ``pin`` is in the
    body's frame; ``i_bend`` and ``c`` are its in-plane bending, ``i_min`` its weakest axis's, per plate."""

    name: str
    kind: str
    loop: int
    body: str
    pin: tuple[float, float, float]
    plates: int
    root: float
    area: float
    i_bend: float
    c: float
    i_min: float

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.pin))

    def loads(self, force: np.ndarray, axis: np.ndarray) -> tuple[float, float, float]:
        """With ``force`` on the pin and the unit ``axis`` from the body's origin to it: the stress (Pa), and the load
        along one plate in compression (N) with the load it buckles at (N)."""
        along = float(force @ axis)
        if self.kind == "crank":
            across = float(np.linalg.norm(force - along * axis))
            return across * (self.length - self.root) * self.c / self.i_bend, 0.0, math.inf
        per = along / self.plates
        return abs(per) / self.area, max(-per, 0.0), math.pi**2 * PLA_E_PA * self.i_min / self.length**2


class PrintedMembers:
    """The pin forces on printed members, step by step, each judged as held ``HOLD_STEPS`` steps (D-103)."""

    def __init__(self, model: mujoco.MjModel, members: Sequence[Member]):
        self.members = list(members)
        connects = [i for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT]
        self.eq = [connects[m.loop] for m in self.members]
        self.body = [model.body(m.body).id for m in self.members]
        self.reset()

    def reset(self) -> None:
        self.stress = (0.0, "")  # MPa, the part
        self.buckled = (0.0, math.inf, "")  # N along a plate, the N it buckles at, the part
        self._recent: list[list[tuple[float, float, float]]] = [[] for _ in self.members]

    def read(self, d: mujoco.MjData) -> None:
        if not d.nefc:
            return
        kinds, ids, forces = d.efc_type[: d.nefc], d.efc_id[: d.nefc], d.efc_force[: d.nefc]
        for k, m in enumerate(self.members):
            rows = (kinds == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (ids == self.eq[k])
            f = forces[rows]
            if len(f) != 3:
                continue
            axis = d.xmat[self.body[k]].reshape(3, 3) @ (np.array(m.pin) / m.length)
            recent = [*self._recent[k][-(HOLD_STEPS - 1):], m.loads(f, axis)]
            self._recent[k] = recent
            if len(recent) == HOLD_STEPS:
                held = min(r[0] for r in recent) / 1e6
                if held > self.stress[0]:
                    self.stress = (held, m.name)
                load, at = min(r[1] for r in recent), recent[-1][2]
                if load / at > self.buckled[0] / self.buckled[1]:
                    self.buckled = (load, at, m.name)

    def broken(self) -> str:
        if self.stress[0] > PLA_ALLOW_PA / 1e6:
            return (f"unphysical: the printed {self.stress[1]} carried {self.stress[0]:.0f} MPa at the pin's force; "
                    f"printed PLA allows {PLA_ALLOW_PA / 1e6:.0f}")
        if self.buckled[0] > self.buckled[1] / PRINT_FACTOR:
            return (f"unphysical: the printed {self.buckled[2]} carried {self.buckled[0]:.0f} N along it; it buckles at "
                    f"{self.buckled[1]:.0f} N, so may carry {self.buckled[1] / PRINT_FACTOR:.0f}")
        return ""
