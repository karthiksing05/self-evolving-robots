"""Worlds for the evolution core (sea.evolve).  Importing this package registers each one by name."""

from sea.evolve.loop import register


def _gripper():
    from sea.worlds.gripper import GripperWorld

    return GripperWorld()


def _phonology():
    from sea.worlds.phonology import PhonologyWorld

    return PhonologyWorld()


def _mobile():
    from sea.worlds.mobile import MobileWorld

    return MobileWorld()


register("gripper", _gripper)
register("phonology", _phonology)
register("mobile", _mobile)
