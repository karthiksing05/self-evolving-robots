"""A mechanism as a graph of typed parts (D-078; docs/theory.md §3.1, body primitives).

A body of a robot is rigid bodies joined by joints and constraints, each body carrying parts.  A part is a
physical thing with a function: a link plate between two pins, a hub on a pin, a gear, a rack, a rod, a
bushing, a clamp, a bracket screwed to the frame, a finger's blade.  Each part knows its solids (what the
simulator collides with and draws), its volume and material (so its mass), and what it is to the laws (its
role).  ``mech.build.compile`` turns a graph into the simulator's bodies; nothing else describes a mechanism.

All lengths are metres.  A part's coordinates are in its body's frame.  The graph is plain data (pydantic),
so System 2 can write one and the record can store it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Material = Literal["pla", "steel", "plywood", "stock"]
# what a part is to the laws: carried structure, a pin or rod others ride on, a clamp holding rods, a bracket
# fastened to the frame, the finger that meets the object
Role = Literal["structure", "pin", "rod", "clamp", "bracket", "finger"]
DENSITY = {"pla": 1240.0, "steel": 7850.0, "plywood": 600.0}  # kg/m^3: PLA printed solid, ground steel rod, birch ply


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- parts
class Box(_M):
    """A printed (or steel) block: centre, half extents and orientation in the body's frame."""

    kind: Literal["box"] = "box"
    name: str
    pos: Vec
    half: Vec
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    material: Material = "pla"
    role: Role = "structure"
    piece: str = ""  # parts with the same piece are one printed part (a clamp and its bracket)
    massive: bool = True  # counts toward the body's mass
    kg: float | None = None  # a stated mass instead of its volume's (a world's laws may refuse it)
    mu: float | None = None  # its surface's friction, where it is not the world's default


class Cylinder(_M):
    """A round part along its local z: a hub, a boss, a rod, a pin head."""

    kind: Literal["cylinder"] = "cylinder"
    name: str
    pos: Vec
    radius: float
    half_length: float
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    material: Material = "pla"
    role: Role = "structure"
    piece: str = ""
    massive: bool = True
    kg: float | None = None
    mu: float | None = None


class Capsule(_M):
    """A rounded bar along its local z: a boom, a dowel, a piston's cylinder or rod."""

    kind: Literal["capsule"] = "capsule"
    name: str
    pos: Vec
    radius: float
    half_length: float
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    material: Material = "pla"
    role: Role = "structure"
    piece: str = ""
    massive: bool = True
    kg: float | None = None
    mu: float | None = None


class Sphere(_M):
    """A ball: a caster, a knob."""

    kind: Literal["sphere"] = "sphere"
    name: str
    pos: Vec
    radius: float
    material: Material = "steel"
    role: Role = "structure"
    piece: str = ""
    massive: bool = True
    kg: float | None = None
    mu: float | None = None


class Link(_M):
    """A link plate from the body's origin to (0, -length, 0), lying between z0 and z1: a web, round bosses at
    both pins (half of each boss outside the web counts), and the far pin's head on the plate's outer face."""

    kind: Literal["link"] = "link"
    name: str
    length: float
    z: tuple[float, float]
    half_width: float
    boss_r: float
    pin_head: tuple[float, float]  # radius, half thickness
    outward: Literal[1, -1]  # which face of the plate is outside: +z or -z


class Web(_M):
    """A printed web joining two points (x, y) of the body's mid-plane z, ``half`` thick across it."""

    kind: Literal["web"] = "web"
    name: str
    a: tuple[float, float]
    b: tuple[float, float]
    z: float
    half: tuple[float, float]  # half thickness in the plane, half depth along z


class Gear(_M):
    """A printed spur gear (or a sector of one, ``arc`` in the gear's own angles) about the body's z."""

    kind: Literal["gear"] = "gear"
    name: str
    r_pitch: float
    width: float
    z: float
    arc: tuple[float, float] | None = None


class RackTeeth(_M):
    """A rack's teeth (drawn: the teeth are the mesh constraint in the simulator), in the gripper frame's x-z
    plane extruded along y, placed in the body's frame by ``to_body`` (rotation rows, then the origin)."""

    kind: Literal["rack_teeth"] = "rack_teeth"
    name: str
    x: tuple[float, float]
    x_mesh: float
    z_pitch: float
    width: float
    module: float
    y_mid: float


class Blade(_M):
    """The stock finger's blade below its cut: drawn as one surface, in contact as its convex parts, offset
    (body frame) from where it sits on the stock jaw."""

    kind: Literal["blade"] = "blade"
    offset: Vec = (0.0, 0.0, 0.0)


class PadMount(_M):
    """Where the pads ride (D-081): this body carries them, placed as on the stock jaw at q = 0 and offset (body
    frame) as a blade is.  No solid of its own; the body's axes must be the jaw's at q = 0."""

    kind: Literal["pad_mount"] = "pad_mount"
    offset: Vec = (0.0, 0.0, 0.0)


class Bracket(_M):
    """A printed bracket from y = ``y_from`` toward +y until it meets the frame mesh over its footprint (x, z),
    seated ``FIT`` into the frame's face; ``fallback_y`` where no frame lies behind it (it then mounts on
    nothing, which the mounted law finds).  On the gripper body, gripper frame."""

    kind: Literal["bracket"] = "bracket"
    name: str
    x: tuple[float, float]
    z: tuple[float, float]
    y_from: float
    fallback_y: float
    role: Role = "bracket"
    piece: str = ""


class Bought(_M):
    """A bought part carried by the body with no solid of its own here (a bushing inside its bore)."""

    kind: Literal["bought"] = "bought"
    item: str
    kg: float


class Wheel(_M):
    """A wheel: a cylinder about its local z that rolls on what it touches, with its tread's friction there
    (D-087).  It turns on its body's hinge."""

    kind: Literal["wheel"] = "wheel"
    name: str
    pos: Vec
    radius: float
    half_width: float
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    mu: float = 0.9
    condim: int = 3
    material: Material = "pla"
    role: Role = "structure"
    piece: str = ""
    massive: bool = True
    kg: float | None = None


Part = Annotated[Box | Cylinder | Capsule | Sphere | Link | Web | Gear | RackTeeth | Blade | PadMount | Bracket | Bought
                 | Wheel,
                 Field(discriminator="kind")]


# ---------------------------------------------------------------- bodies and joints
class Joint(_M):
    """A joint of a body to its parent.  ``free``: the body moves in every way (a chassis on the floor)."""

    name: str
    type: Literal["hinge", "slide", "free"]
    axis: Vec = (0.0, 0.0, 1.0)
    pos: Vec = (0.0, 0.0, 0.0)
    range: tuple[float, float] | None = None
    armature: float = 0.0
    damping: float = 0.0
    stiffness: float = 0.0  # a spring on the joint itself, toward ``springref``
    springref: float = 0.0
    frictionloss: float = 0.0


class Inertial(_M):
    """How a body's inertia is taken from its mass.  ``slender``: a bar of ``size`` along the body's y;
    ``disc``: a disc of radius ``size`` about z; ``lumped``: ``size`` m^2 per kg about every axis."""

    pos: Vec
    kind: Literal["slender", "disc", "lumped"]
    size: float


class Body(_M):
    """A rigid body.  A ``stock`` body is the robot's own (the gripper, the servo's horn body): it is not
    created, and its parts' mass joins its mass, unless ``inertial`` is given, which then sets it whole."""

    name: str
    parent: str = ""
    pos: Vec = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    stock: bool = False
    joints: tuple[Joint, ...] = ()
    parts: tuple[Part, ...] = ()
    inertial: Inertial | None = None


class Loop(_M):
    """Closes a loop: ``point`` of body ``a`` (in ``a``'s frame) is pinned to body ``b``.  With each body's own
    joint at its origin, the loop's pins are ``a``'s and ``b``'s origins and ``point``: the transmission angle is
    the angle at ``point`` between them."""

    kind: Literal["loop"] = "loop"
    a: str
    b: str
    point: Vec


class Mesh(_M):
    """Gear teeth in mesh: joint ``follower`` moves ``ratio`` per unit of joint ``driver`` (a pinion's radius)."""

    kind: Literal["mesh"] = "mesh"
    follower: str
    driver: str
    ratio: float
    pressure_angle_deg: float = 20.0


class Guide(_M):
    """A carriage on two rods clamped at one end (D-076), or at both (D-080): the carriage may move across the
    rods and turn as the bent rods let it, from ``at`` (carriage frame).  ``clamp_end`` is where the rods leave
    the clamp, ``far_end`` where a second clamp holds their other ends (none: they overhang), ``centre`` the
    carriage's middle at zero travel, ``mesh_x`` where its rack meets the gear (all gripper x), ``module`` the
    teeth's, ``bushings`` each bushing's offset along the rods from the carriage's middle (one per rod at each:
    a carriage carries two LM3UU per offset)."""

    kind: Literal["guide"] = "guide"
    carriage: str
    slide: str
    at: Vec
    clamp_end: float
    centre: float
    module: float
    mesh_x: float
    bushings: tuple[float, ...] = (0.0,)
    far_end: float | None = None


# ---------------------------------------------------------------- actuators, springs, latches and sensors (D-087)
# Each takes its numbers from its part's datasheet, which the world's catalogue gives as data.  ``mech.runtime``
# applies the ones whose force depends on the state; a servo and a band are the simulator's own.
class Servo(_M):
    """A position servo on ``joint``: it drives toward its command with gain ``kp`` and at most ``stall_nm``; its
    gearbox damps the joint by ``kv`` and adds ``armature`` (its rotor's inertia seen through the gears)."""

    kind: Literal["servo"] = "servo"
    name: str
    joint: str
    stall_nm: float
    kp: float = 20.0
    kv: float = 0.3
    armature: float = 0.002


class Gearmotor(_M):
    """A DC gearmotor on ``joint``: its torque falls linearly from ``stall_nm`` at rest to zero at
    ``no_load_rad_s``, scaled by its duty (-1..1)."""

    kind: Literal["gearmotor"] = "gearmotor"
    name: str
    joint: str
    stall_nm: float
    no_load_rad_s: float


class Piston(_M):
    """A pneumatic cylinder on slide ``joint``: its valve applies the supply's ``pressure_pa`` over the bore's
    area to extend or retract, and a needle valve limits its speed to about ``valve_m_s`` (a damper P A / v on the
    joint, which the simulator integrates implicitly: as an actuator force it would be unstable).
    ``end_hold_n`` keeps the rod seated at an end of its stroke while the valve is closed."""

    kind: Literal["piston"] = "piston"
    name: str
    joint: str
    bore_m: float
    pressure_pa: float
    valve_m_s: float
    end_hold_n: float = 5.0


class TorsionSpring(_M):
    """A wound torsion spring on hinge ``joint`` (a mousetrap): its torque falls linearly from ``torque_nm[0]``
    with the joint at ``set_rad`` to ``torque_nm[1]`` after it has swung ``span_rad`` in direction ``sign``."""

    kind: Literal["torsion_spring"] = "torsion_spring"
    name: str
    joint: str
    torque_nm: tuple[float, float]
    span_rad: float
    set_rad: float = 0.0
    sign: Literal[1, -1] = 1


class Band(_M):
    """An elastic band from ``a_point`` on body ``a`` to ``b_point`` on body ``b``: it pulls ``k_n_m`` per metre
    past ``rest_m``, never pushes, and breaks past ``max_stretch`` times its rest length (a law)."""

    kind: Literal["band"] = "band"
    name: str
    a: str
    a_point: Vec
    b: str
    b_point: Vec
    k_n_m: float
    rest_m: float
    max_stretch: float = 3.0


Actuator = Annotated[Servo | Gearmotor | Piston | TorsionSpring | Band, Field(discriminator="kind")]


class Latch(_M):
    """Holds ``joint`` at ``at`` until it is released (a trigger, a pin pulled, a solenoid)."""

    kind: Literal["latch"] = "latch"
    name: str
    joint: str
    at: float = 0.0


class LimitSwitch(_M):
    """A switch that closes when anything presses its box (``pos``, ``half``) on ``body``."""

    kind: Literal["limit_switch"] = "limit_switch"
    name: str
    body: str
    pos: Vec
    half: Vec = (0.004, 0.004, 0.002)


class RangeFinder(_M):
    """An IR or ultrasonic range sensor on ``body`` at ``pos``, looking along its local z (``quat``): the distance
    to the first surface, out to ``max_m``."""

    kind: Literal["range"] = "range"
    name: str
    body: str
    pos: Vec
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    max_m: float = 1.5


class Encoder(_M):
    """A joint's position, as a shaft encoder reads it."""

    kind: Literal["encoder"] = "encoder"
    name: str
    joint: str


Sensor = Annotated[LimitSwitch | RangeFinder | Encoder, Field(discriminator="kind")]


class Carry(_M):
    """An item of the world that starts on the robot (a ball on a chute, a flag on its clip), at ``pos`` and ``quat`` in
    ``body``'s frame.  The world says what each item is and how it is held."""

    item: str
    body: str
    pos: Vec = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)


class Graph(_M):
    """A mechanism: its bodies (in build order), what it removes from the stock robot, its constraints, the
    pairs that may touch where they are pinned, and the joints that move with the driving joint."""

    name: str
    bodies: tuple[Body, ...]
    remove_stock_mesh: tuple[str, ...] = ()  # stock meshes on these bodies the mechanism replaces
    loops: tuple[Loop, ...] = ()
    meshes: tuple[Mesh, ...] = ()
    guides: tuple[Guide, ...] = ()
    excludes: tuple[tuple[str, str], ...] = ()
    linked: tuple[tuple[str, float], ...] = ()  # (joint, units per radian of the driving joint)
    actuators: tuple[Actuator, ...] = ()
    latches: tuple[Latch, ...] = ()
    sensors: tuple[Sensor, ...] = ()
    carried: tuple[Carry, ...] = ()

    def new_bodies(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bodies if not b.stock)
