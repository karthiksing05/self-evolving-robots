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


register("gripper", _gripper)
register("gripper_dense", _gripper_dense)
register("phonology", _phonology)
register("mobile", _mobile)
