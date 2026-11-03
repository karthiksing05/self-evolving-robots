"""The gripper's catalogue jaws as mechanism graphs (D-078): the seed modules of the library.

Each function returns the ``Graph`` of one jaw: which parts it adds, on which bodies, how they are joined, and
what moves with the servo.  The layout is the SO-101's (``sim/jaws.py`` holds its measured constants: the
hinge, the side planes, the frame's faces).  These were the hand-written builders ``jaws.four_bar`` and
``jaws.slide``; they compile to the same model (``tests/mech/test_seed_equivalence.py``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from sea.mech.graph import (
    Blade,
    Body,
    Bought,
    Box,
    Bracket,
    Cylinder,
    Gear,
    Graph,
    Guide,
    Inertial,
    Joint,
    Link,
    Loop,
    Mesh,
    RackTeeth,
    Web,
)
from sea.sim import jaws as J

MM = J.MM
SIDES = {"horn": J.HORN_SIDE, "rear": J.REAR_SIDE}  # the jaw's two link planes (jaw frame z)


class Site(NamedTuple):
    """Where a jaw mechanism mounts, as the robot's model has it: the servo's range (rad) and the moving jaw's
    orientation in the gripper (its axes, ``jaws.R0``)."""

    q_range: tuple[float, float]
    jaw_quat: tuple[float, float, float, float]

    @classmethod
    def of(cls, spec) -> Site:
        return cls(tuple(float(x) for x in spec.joint("gripper").range),
                   tuple(float(x) for x in spec.body(J.JAW_MESH).quat))


def _link(name: str, length: float, side: str) -> Link:
    z = SIDES[side]
    return Link(name=name, length=length, z=z, half_width=J.PLATE_HALF_W, boss_r=J.BOSS_R, pin_head=J.PIN_HEAD,
                outward=1 if (z[0] + z[1]) / 2 > J.MID_Z else -1)


def _hub(name: str, at, r: float = J.BOSS_R) -> Cylinder:
    """A hub on a pin between the two side planes (jaw frame)."""
    lo, hi = J.REAR_SIDE[1] + 0.3 * MM, J.HORN_SIDE[0] - 0.3 * MM
    return Cylinder(name=name, pos=(float(at[0]), float(at[1]), (lo + hi) / 2), radius=r, half_length=(hi - lo) / 2)


def _pin(name: str, kind: str = "hinge", **kw) -> Joint:
    return Joint(name=name, type=kind, armature=J.JOINT_ARMATURE, damping=J.JOINT_DAMPING, **kw)


def revolute() -> Graph:
    """The stock jaw: the finger screwed to the servo horn."""
    return Graph(name="revolute", bodies=())


def parallelogram(link_mm: float, site: Site) -> Graph:
    """Two driver plates on the horn and the rear hub, two follower plates on a pin held by a bracket on the
    servo's end; the finger is the coupler, pinned to both pairs, so it keeps its orientation (D-057)."""
    L = link_mm * MM
    e = J.follower_offset()
    f_g = J.HINGE + J.R0 @ e
    jaw = Body(name=J.JAW_MESH, stock=True, parts=(_link("jaw_driver_horn", L, "horn"), _link("jaw_driver_rear", L, "rear")),
               inertial=Inertial(pos=(0.0, -L / 2, 0.0), kind="slender", size=L))
    coupler_parts = [Blade(offset=(0.0, L, 0.0)), _hub("jaw_coupler_hub0", np.zeros(3)), _hub("jaw_coupler_hub1", e),
                     Web(name="jaw_coupler_web", a=(0.0, 0.0), b=(float(e[0]), float(e[1])), z=J.MID_Z,
                         half=(3.5 * MM, 6.0 * MM))]
    if J.blade_gap(np.array([0.0, -L])) > J.BOSS_R - 1.0 * MM:
        # links longer than the finger: the pins hang below the blade, and a strut carries it from them
        tip = J.blade_tip() + [0.0, L]
        coupler_parts.append(Web(name="jaw_coupler_strut", a=(0.0, 0.0), b=(float(tip[0]), float(tip[1])), z=J.MID_Z,
                                 half=(3.5 * MM, 6.0 * MM)))
    coupler = Body(name="jaw_coupler", parent=J.JAW_MESH, pos=(0.0, -L, 0.0), joints=(_pin("jaw_coupler_q", axis=(0.0, 0.0, 1.0)),),
                   parts=tuple(coupler_parts), inertial=Inertial(pos=(0.016, L - 0.057, 0.0), kind="lumped", size=4e-4))
    follower = Body(name="jaw_follower", parent="gripper", pos=tuple(f_g.tolist()), quat=site.jaw_quat,
                    joints=(_pin("jaw_follower_q", axis=(0.0, 0.0, 1.0)),),
                    parts=(_link("jaw_follower_horn", L, "horn"), _link("jaw_follower_rear", L, "rear")),
                    inertial=Inertial(pos=(0.0, -L / 2, 0.0), kind="slender", size=L))
    # the bracket on the servo's end that holds the follower pin (gripper frame: its axis along y)
    lo, hi = J.HINGE[1] - (J.HORN_SIDE[0] - 0.3 * MM), J.HINGE[1] - (J.REAR_SIDE[1] + 0.3 * MM)
    grip = Body(name="gripper", stock=True, parts=(
        Cylinder(name="jaw_bracket_hub", pos=(float(f_g[0]), (lo + hi) / 2, float(f_g[2])), radius=5.0 * MM,
                 half_length=(hi - lo) / 2, quat=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0), role="bracket",
                 piece="jaw_bracket"),
        Box(name="jaw_bracket", pos=(30.4 * MM + 2.8 * MM, float(J.HINGE[1] - J.MID_Z), float(f_g[2] - 6.0 * MM)),
            half=(2.8 * MM, 15.0 * MM, 9.0 * MM), role="bracket", piece="jaw_bracket")))
    return Graph(name=f"parallelogram({link_mm:g})", bodies=(jaw, coupler, follower, grip),
                 remove_stock_mesh=(J.JAW_MESH,), loops=(Loop(a="jaw_follower", b="jaw_coupler", point=(0.0, -L, 0.0)),),
                 excludes=(("gripper", "jaw_coupler"), ("jaw_follower", J.JAW_MESH), ("jaw_follower", "jaw_coupler")),
                 linked=(("jaw_coupler_q", -1.0), ("jaw_follower_q", 1.0)))


def _jaw_box(name: str, lo, hi) -> Box:
    """A block given by gripper-frame corners, on a body whose frame is the jaw's at q = 0."""
    a, b = J._to_jaw(np.asarray(lo, float)), J._to_jaw(np.asarray(hi, float))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return Box(name=name, pos=tuple(((lo + hi) / 2).tolist()), half=tuple(((hi - lo) / 2).tolist()))


def _grip_box(name: str, lo, hi, role: str = "structure", piece: str = "") -> Box:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Box(name=name, pos=tuple(((lo + hi) / 2).tolist()), half=tuple(((hi - lo) / 2).tolist()), role=role,
               piece=piece)


def slide(r_pitch: float, sector: bool, site: Site) -> Graph:
    """A rack and pinion on two rods (D-058): the servo's hinge turns a pinion of pitch radius ``r_pitch`` on its
    horn; the rack's carriage runs on the rods ``r_pitch`` per radian and carries the stock finger's blade.
    ``sector``: the long slide (D-064), a gear sector with teeth only over the arc in mesh, on 150 mm rods
    clamped beside the fixed finger."""
    lo, hi = site.q_range
    s_lo, s_hi = lo * r_pitch, hi * r_pitch
    width = J.GEAR_Y[1] - J.GEAR_Y[0]
    hub = (J.HORN_SIDE[0], J.HINGE[1] - J.GEAR_Y[1])
    n_teeth, m = J.module_for(r_pitch)
    pitch = 2 * math.pi / n_teeth
    # the gear's angle in mesh (at the bottom) is -pi/2 - q: over the range, plus a margin of teeth
    arc = (-math.pi / 2 - hi - J.SECTOR_TEETH * pitch, -math.pi / 2 - lo + J.SECTOR_TEETH * pitch) if sector else None
    z_gear = J.HINGE[1] - sum(J.GEAR_Y) / 2
    jaw = Body(name=J.JAW_MESH, stock=True, parts=(
        Cylinder(name="jaw_pinion_hub", pos=(0.0, 0.0, sum(hub) / 2), radius=J.HUB_R, half_length=(hub[1] - hub[0]) / 2),
        Gear(name="jaw_pinion", r_pitch=r_pitch, width=width, z=float(z_gear), arc=arc)),
        inertial=Inertial(pos=(0.0, 0.0, float(z_gear)), kind="disc", size=r_pitch))

    z_pitch = J.HINGE[2] - r_pitch
    root = z_pitch - 1.25 * m
    rack_x = (J.HINGE[0] - s_hi - 6 * MM, J.HINGE[0] - s_lo + 6 * MM)  # meshed under the pinion at every opening
    car = [_jaw_box("jaw_rack", [rack_x[0], J.GEAR_Y[0], root - 5 * MM], [rack_x[1], J.GEAR_Y[1], root]),
           RackTeeth(name="jaw_rack_teeth", x=rack_x, x_mesh=float(J.HINGE[0]), z_pitch=float(z_pitch), width=width,
                     module=m, y_mid=sum(J.GEAR_Y) / 2)]
    x0, x1 = J.CARRIAGE_X
    if sector:
        # the block beside the gear plane, from under the rack to above the upper rod, a bore round each rod;
        # the arm under the rack carries the blade
        rod_y = J.LONG_ROD_Y
        rods = [z_pitch + d for d in J.LONG_ROD_RISE]
        y0, y1 = rod_y - 5 * MM, J.GEAR_Y[0]
        bottom, top = root - 13 * MM, rods[1] + 6 * MM
        for k, (z0, z1) in enumerate([(bottom, rods[0] - J.BORE), (rods[0] + J.BORE, rods[1] - J.BORE), (rods[1] + J.BORE, top)]):
            car.append(_jaw_box(f"jaw_carriage{k}", [x0, y0, z0], [x1, y1, z1]))
        for k, zr in enumerate(rods):
            for side, (ya, yb) in (("out", (y0, rod_y - J.BORE)), ("in", (rod_y + J.BORE, y1))):
                car.append(_jaw_box(f"jaw_carriage_bore{k}_{side}", [x0, ya, zr - J.BORE], [x1, yb, zr + J.BORE]))
        car.append(_jaw_box("jaw_carriage_arm", [12 * MM, y1, bottom], [26 * MM, -6 * MM, root - 5 * MM]))
    else:
        # the carriage block under the rack, with a square bore for each rod's bushing
        rod_y = J.ROD_Y
        rods = [z_pitch - d for d in J.ROD_DROP]
        top, bottom = root - 5 * MM, rods[1] - 6 * MM
        cy = J.CARRIAGE_Y
        for k, (z0, z1) in enumerate([(rods[0] + J.BORE, top), (rods[1] + J.BORE, rods[0] - J.BORE), (bottom, rods[1] - J.BORE)]):
            car.append(_jaw_box(f"jaw_carriage{k}", [x0, cy[0], z0], [x1, cy[1], z1]))
        for k, zr in enumerate(rods):
            for side, (ya, yb) in (("out", (cy[0], J.ROD_Y - J.BORE)), ("in", (J.ROD_Y + J.BORE, cy[1]))):
                car.append(_jaw_box(f"jaw_carriage_bore{k}_{side}", [x0, ya, zr - J.BORE], [x1, yb, zr + J.BORE]))
        # the arm across to the blade, at a height where both are
        arm_z = min(max(z_pitch - 15 * MM, -95 * MM), -62 * MM)
        arm_x1 = 19 * MM if arm_z < -88 * MM else 26 * MM
        car.append(_jaw_box("jaw_carriage_arm", [12 * MM, cy[1], arm_z - 4 * MM], [arm_x1, -6 * MM, arm_z + 4 * MM]))
    car += [Blade(), Bought(item="lm3uu", kg=J.BUSHING_KG), Bought(item="lm3uu", kg=J.BUSHING_KG)]
    y_mid = (rod_y - 5 * MM + J.GEAR_Y[0]) / 2 if sector else -20 * MM
    carriage = Body(name="jaw_carriage", parent="gripper", pos=tuple(J.HINGE.tolist()), quat=site.jaw_quat,
                    joints=(_pin("jaw_slide", "slide", axis=(1.0, 0.0, 0.0), range=(s_lo, s_hi)),), parts=tuple(car),
                    inertial=Inertial(pos=tuple(J._to_jaw(np.array([sum(J.CARRIAGE_X) / 2, y_mid, (top + bottom) / 2])).tolist()),
                                      kind="lumped", size=4e-4))

    # the rods, their clamp and its bracket to the frame, on the gripper; the rods end just past the carriage
    # fully open, so a long travel leaves the carriage off their back end when closed
    front = J.CARRIAGE_X[1] + s_hi + 2 * MM
    rod_len = J.LONG_ROD if sector else J.ROD_LEN
    fixed = [Cylinder(name=f"jaw_rod{k}", pos=(front - rod_len / 2, rod_y, zr), radius=J.ROD_R, half_length=rod_len / 2,
                      quat=(math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0), material="steel", role="rod") for k, zr in enumerate(rods)]
    if sector:
        # the clamp grips both rods beside the fixed finger; its bracket runs in over the rack's teeth (1 mm clear)
        # and under the sector's sweep to the finger
        clamp_hi = J.CLAMP_X[1]
        fixed.append(_grip_box("jaw_clamp", [J.CLAMP_X[0], rod_y - 5 * MM, rods[0] - 5 * MM],
                               [J.CLAMP_X[1], rod_y + 5 * MM, rods[1] + 5 * MM], "clamp", "jaw_clamp"))
        fixed.append(Bracket(name="jaw_clamp_bracket", x=J.CLAMP_X, z=(z_pitch + m + 1 * MM, z_pitch + m + 8 * MM),
                             y_from=rod_y + 5 * MM, fallback_y=rod_y + 6 * MM, piece="jaw_clamp"))
    else:
        back = front - J.ROD_LEN
        clamp_hi = back + 8 * MM
        clamp_top = -31 * MM  # the bracket's top meets the frame's horn side (y = -28 mm, down to z = -37 mm)
        fixed.append(_grip_box("jaw_clamp", [back, J.CARRIAGE_Y[0], rods[1] - 5 * MM], [back + 8 * MM, J.ROD_Y + 5 * MM, clamp_top],
                               "clamp", "jaw_clamp"))
        # the bracket crosses the gear plane behind the pinion to reach the frame (x >= -19.5 mm there): a pinion
        # too large leaves it no room, and the clamp hangs from nothing
        reach = min(max(back + 8 * MM, -11.0 * MM), J.HINGE[0] - r_pitch - 2 * m - 1 * MM)
        fixed.append(Bracket(name="jaw_clamp_bracket", x=(back, max(reach, back + 1 * MM)), z=(clamp_top, clamp_top + 6 * MM),
                             y_from=J.CARRIAGE_Y[0], fallback_y=J.CARRIAGE_Y[0] + 1 * MM, piece="jaw_clamp"))
    grip = Body(name="gripper", stock=True, parts=tuple(fixed))

    c0 = J._to_jaw(np.array([sum(J.CARRIAGE_X) / 2, rod_y, sum(rods) / 2]))
    return Graph(name=f"{'long_slide' if sector else 'prismatic'}({r_pitch * 1000:g} mm)", bodies=(jaw, carriage, grip),
                 remove_stock_mesh=(J.JAW_MESH,),
                 guides=(Guide(carriage="jaw_carriage", slide="jaw_slide", at=tuple(c0.tolist()), clamp_end=float(clamp_hi),
                               centre=sum(J.CARRIAGE_X) / 2, module=m, mesh_x=float(J.HINGE[0])),),
                 meshes=(Mesh(follower="jaw_slide", driver="gripper", ratio=r_pitch),),
                 excludes=((J.JAW_MESH, "jaw_carriage"),),  # the pinion meshes with the rack
                 linked=(("jaw_slide", r_pitch),))


def jaw_graph(fingers, site: Site) -> Graph:
    """The graph of a ``FingerSpec``'s jaw: a built mechanism's own, or the catalogue's."""
    if fingers.jaw == "built":
        return Graph.model_validate_json(fingers.mechanism)
    if fingers.jaw == "prismatic":
        return slide(fingers.drive_mm_per_rad * MM, fingers.sector, site)
    if fingers.jaw == "parallelogram":
        return parallelogram(fingers.link_mm, site)
    return revolute()
