"""Worlds for the evolution core (sea.evolve).  Importing this package registers each one by name."""

from sea.evolve.loop import register


def _gripper():
    from sea.worlds.gripper import GripperWorld

    return GripperWorld()


def _gripper_dense():
    """The gripper with four train placements per object (centred plus three offsets) instead of two; the
    test placements are the same.  A lever against fitting the train placements (docs/optimizing-the-harness.md)."""
    from sea.worlds.gripper import GripperWorld

    return GripperWorld(train_offsets=3)


def _phonology():
    from sea.worlds.phonology import PhonologyWorld

    return PhonologyWorld()


def _mobile():
    from sea.worlds.mobile import MobileWorld

    return MobileWorld()


def _me2110():
    from sea.worlds.me2110 import ME2110World

    return ME2110World()


def _me2110_kit():
    from sea.worlds.me2110_kit import KitWorld

    return KitWorld()


register("gripper", _gripper)
register("gripper_dense", _gripper_dense)
register("phonology", _phonology)
register("mobile", _mobile)
register("me2110", _me2110)
register("me2110_kit", _me2110_kit)
