"""The robot a mechanism graph is added to (D-087).

A graph is plain data and belongs to no robot.  A ``Base`` says what it is built onto: the body every pose is
measured in (``root``), the joint that drives the mechanism (``driver``), the robot's stock model, and the body
that carries what the task touches when the graph names none.  The world passes its base in; none is a default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class Base:
    name: str
    root: str
    driver: str
    model: Callable[[], mujoco.MjSpec]
    finger: str = ""


def _empty() -> mujoco.MjSpec:
    """An empty world with a floor: for a robot built whole from a graph (a chassis on a free joint)."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # a graph's joint ranges are radians, as the stock robot's model says of its own
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1])
    return spec


BASES: dict[str, Base] = {}


def register(base: Base) -> Base:
    BASES[base.name] = base
    return base


WORLD = register(Base(name="world", root="world", driver="", model=_empty))


def base(name: str | Base) -> Base:
    return name if isinstance(name, Base) else BASES[name]
