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
Material = Literal["pla", "steel", "stock"]
# what a part is to the laws: carried structure, a pin or rod others ride on, a clamp holding rods, a bracket
# fastened to the frame, the finger that meets the object
Role = Literal["structure", "pin", "rod", "clamp", "bracket", "finger"]
DENSITY = {"pla": 1240.0, "steel": 7850.0}  # kg/m^3: PLA as printed solid, ground steel rod


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


Part = Annotated[Box | Cylinder | Link | Web | Gear | RackTeeth | Blade | PadMount | Bracket | Bought,
                 Field(discriminator="kind")]


# ---------------------------------------------------------------- bodies and joints
class Joint(_M):
    name: str
    type: Literal["hinge", "slide"]
    axis: Vec
    pos: Vec = (0.0, 0.0, 0.0)
    range: tuple[float, float] | None = None
    armature: float = 0.0
    damping: float = 0.0


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

    def new_bodies(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bodies if not b.stock)
