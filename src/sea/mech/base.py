"""The robot a mechanism graph is added to (D-087).

A graph is plain data and belongs to no robot.  A ``Base`` says what it is built onto: the body every pose is
measured in (``root``), the joint that drives the mechanism (``driver``), the robot's stock model, and the body
that carries the finger when the graph names none.  The world passes its base in; the gripper's is the default.
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


def _so101() -> mujoco.MjSpec:
    from sea.control.frames import STOCK_MJCF

    return mujoco.MjSpec.from_file(str(STOCK_MJCF))


def _empty() -> mujoco.MjSpec:
    """An empty world with a floor: for a robot built whole from a graph (a chassis on a free joint)."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1])
    return spec


BASES: dict[str, Base] = {}


def register(base: Base) -> Base:
    BASES[base.name] = base
    return base


GRIPPER = register(Base(name="so101_gripper", root="gripper", driver="gripper", model=_so101,
                        finger="moving_jaw_so101_v1"))
WORLD = register(Base(name="world", root="world", driver="", model=_empty))


def base(name: str | Base | None) -> Base:
    if isinstance(name, Base):
        return name
    return BASES[name or GRIPPER.name]
