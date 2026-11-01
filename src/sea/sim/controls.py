"""Control geometry and dynamics (spec 7.2).

Every control is a body with one joint (slide or hinge) inside its panel body.  Positive joint
motion is always the actuation direction (press in, turn, open, slide), so ``moved = q - q_rest``.
Static parts (recess walls, slot housings) are panel geoms.  Lengths are meters; ``size_mm`` and
``recess_depth_mm`` are converted here.

Dynamics are written into ``qfrc_applied`` before each ``mj_step``::

    f = -k (q - q_rest) + detent_force(q) - friction * tanh(qdot / v0)
    detent_force(q) = -sum_i peak * h((q - d_i) / w),   h(u) = u * exp(0.5 - u^2 / 2)

Latches are joint limits (stable at 500 Hz) whose constraint force is the "applied force along the
axis"; the latch releases once that force exceeds ``latch_threshold`` (docs/decisions.md D-012).
Push-latches release when pushed 2 mm inward; a spring then opens them to 15 deg / 15 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sea.contracts import ControlSpec

MM = 1e-3
PUSH_LATCH_DEPTH = 2 * MM
PUSH_LATCH_OPEN_ROT = math.radians(15.0)
PUSH_LATCH_OPEN_LIN = 15 * MM
SLIDE_ARMATURE = 0.05  # kg; keeps explicit springs stable at dt = 2 ms
HINGE_ARMATURE = 2e-4  # kg m^2
FRICTION_V0_LIN = 1e-3
FRICTION_V0_ROT = 1e-2


@dataclass
class GeomSpec:
    name: str
    type: str  # box | cylinder | capsule | sphere | mesh
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    vertices: np.ndarray | None = None  # mesh: the convex part's vertices (MuJoCo collides its hull)
    faces: np.ndarray | None = None  # mesh: triangles, for a visual-only mesh
    visual: bool = False  # drawn, never collides


@dataclass
class MovingPart:
    """A jointed body: rest pose in the control frame, joint in body frame, geoms in body frame."""

    name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    joint: str  # slide | hinge | free (a grasp object: its coordinate is its height)
    axis: tuple[float, float, float]
    range: tuple[float, float]
    geoms: list[GeomSpec] = field(default_factory=list)
    mass: float = 0.01
    inertia: tuple[np.ndarray, np.ndarray] | None = None  # (centre of mass, 3x3 tensor): a real object's


@dataclass
class ControlGeometry:
    """What the sim builds and what System 1 needs to act on one control.

    Points marked "body" are in the moving body's frame (they move with the control); vectors
    marked "ctl" are in the control frame at rest.
    """

    moving: MovingPart
    static: list[GeomSpec]
    contact_body: np.ndarray  # where the tool touches (body)
    approach_ctl: np.ndarray  # direction of tool motion into the contact (ctl)
    actuation_body: np.ndarray  # direction to push/slide/insert (body; rotates with hinges)
    roll_ctl: np.ndarray | None = None  # tool x axis to align with (blades, hooks, pinches)
    handle_body: np.ndarray | None = None  # hook/pinch engagement point (body)
    handle_kind: str = "none"
    rotary: bool = False
    lever_arm: float = 0.0  # m from the hinge axis to the contact point
    flap: MovingPart | None = None  # slot cover (compose split)
    flap_handle_body: np.ndarray | None = None
    footprint: tuple[float, float] = (0.03, 0.03)  # x, y extents on the panel (m)


def _box(name, half, pos=(0.0, 0.0, 0.0), rgba=(0.7, 0.7, 0.7, 1.0), quat=(1.0, 0.0, 0.0, 0.0)) -> GeomSpec:
    return GeomSpec(name, "box", tuple(half), tuple(pos), tuple(quat), rgba)


def _cyl(name, radius, half_h, pos=(0.0, 0.0, 0.0), rgba=(0.7, 0.7, 0.7, 1.0)) -> GeomSpec:
    return GeomSpec(name, "cylinder", (radius, half_h), tuple(pos), (1.0, 0.0, 0.0, 0.0), rgba)


def _quat_x(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


def _quat_y(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


RED = (0.85, 0.2, 0.2, 1.0)
BLUE = (0.2, 0.35, 0.85, 1.0)
GREY = (0.55, 0.55, 0.6, 1.0)
GREEN = (0.2, 0.7, 0.35, 1.0)


def recess_walls(half_x: float, half_y: float, depth: float, t: float = 3 * MM) -> list[GeomSpec]:
    """Four walls of height ``depth`` around an inner opening of half extents (half_x, half_y)."""
    if depth <= 0:
        return []
    hz = depth / 2
    return [
        _box("wall_px", (t / 2, half_y + t, hz), (half_x + t / 2, 0, hz), GREY),
        _box("wall_nx", (t / 2, half_y + t, hz), (-half_x - t / 2, 0, hz), GREY),
        _box("wall_py", (half_x, t / 2, hz), (0, half_y + t / 2, hz), GREY),
        _box("wall_ny", (half_x, t / 2, hz), (0, -half_y - t / 2, hz), GREY),
    ]


def _handle_geoms(kind: str, x: float, plate_top: float) -> tuple[list[GeomSpec], np.ndarray | None]:
    """Handle geoms on a plate (body frame) centred at body x; returns geoms and engagement point."""
    if kind == "tab":
        # A fin with a lip pointing back toward the hinge (-x): fingers pinch the fin, a hook
        # coming from the hinge side catches the lip's underside (the tab's "back edge").
        fin_h = 14 * MM
        geoms = [
            _box("tab_fin", (1.5 * MM, 6 * MM, fin_h / 2), (x, 0, plate_top + fin_h / 2), BLUE),
            _box("tab_lip", (4 * MM, 6 * MM, 1.5 * MM), (x - 2.5 * MM, 0, plate_top + fin_h - 1.5 * MM), BLUE),
        ]
        return geoms, np.array([x - 4 * MM, 0.0, plate_top + 7 * MM])
    if kind == "pull":
        post_h = 14 * MM
        geoms = [
            _box("pull_post_p", (3 * MM, 3 * MM, post_h / 2), (x, 21 * MM, plate_top + post_h / 2), BLUE),
            _box("pull_post_n", (3 * MM, 3 * MM, post_h / 2), (x, -21 * MM, plate_top + post_h / 2), BLUE),
            _box("pull_bar", (3 * MM, 24 * MM, 3 * MM), (x, 0, plate_top + post_h + 3 * MM), BLUE),
        ]
        return geoms, np.array([x, 0.0, plate_top + 6 * MM])
    if kind == "push_latch":
        return [_box("latch_pad", (5 * MM, 5 * MM, 0.5 * MM), (x, 0, plate_top + 0.5 * MM), RED)], None
    return [], None


def build_geometry(spec: ControlSpec) -> ControlGeometry:
    """Geometry for one control, in its own frame (z = panel outward normal)."""
    cls = spec.cls
    sz = [s * MM for s in spec.size_mm]
    recess = spec.recess_depth_mm * MM
    travel = spec.travel
    up = np.array([0.0, 0.0, -1.0])

    if cls in ("push_button", "recessed_button"):
        d = sz[0]
        # The board is solid, so the cap must stand at least its full joint range (+1 mm) proud of it;
        # recess walls grow with the cap so the cap still sits (recess - 1 mm) below the rim (V-006).
        press = travel * 1.3 + 1 * MM
        top = max(1 * MM if cls == "recessed_button" else 4 * MM, press + 1 * MM)
        moving = MovingPart("cap", (0, 0, 0), (1, 0, 0, 0), "slide", (0, 0, -1),
                            (0.0, press), [_cyl("cap", d / 2, 3 * MM, (0, 0, top - 3 * MM), RED)], 0.005)
        half = d / 2 + 2 * MM
        walls = recess_walls(half, half, recess + top - 1 * MM) if recess > 0 else []
        geo = ControlGeometry(moving, walls, np.array([0, 0, top]), up, up.copy(),
                              footprint=(d + 12 * MM, d + 12 * MM))
    elif cls == "toggle":
        w, h = sz[0], sz[1]
        zp = (h / 2) * math.sin(travel / 2) + 3 * MM
        moving = MovingPart("rocker", (0, 0, zp), _quat_x(travel / 2), "hinge", (-1, 0, 0),
                            (-0.05, travel + 0.15), [_box("plate", (w / 2, h / 2, 2 * MM), (0, 0, 0), RED)], 0.004)
        arm = h / 2 - 2 * MM
        geo = ControlGeometry(moving, recess_walls(w / 2 + 2 * MM, h / 2 + 2 * MM, recess),
                              np.array([0, arm, 2 * MM]), up, np.array([0, 0, -1.0]), roll_ctl=np.array([0.0, 1.0, 0.0]),
                              rotary=True, lever_arm=arm,
                              footprint=(w + 10 * MM, h + 10 * MM))
    elif cls == "knob":
        d = sz[0]
        moving = MovingPart("knob", (0, 0, 0), (1, 0, 0, 0), "hinge", (0, 0, 1), (-0.3, travel + 1.0),
                            [_cyl("knob", d / 2, 7 * MM, (0, 0, 7 * MM), RED),
                             _box("pointer", (1 * MM, d / 2 - 1 * MM, 0.75 * MM), (0, d / 4, 14.5 * MM), GREY)], 0.008)
        geo = ControlGeometry(moving, recess_walls(d / 2 + 4 * MM, d / 2 + 4 * MM, recess), np.array([0, 0, 14 * MM]),
                              up, up.copy(), roll_ctl=np.array([1.0, 0.0, 0.0]), rotary=True, lever_arm=d / 2,
                              footprint=(d + 14 * MM, d + 14 * MM))
    elif cls == "lever":
        # Hinged at the panel about y, leaning 25 deg away from the base; pressed down on its tip
        # face, it rotates further away (positive q).  Approach is straight down the normal.
        length = sz[0]
        lean = math.radians(25.0)
        moving = MovingPart("lever", (0, 0, 0), _quat_y(lean), "hinge", (0, 1, 0), (-0.05, min(travel + 0.15, 1.15)),
                            [_box("bar", (4 * MM, 4 * MM, length / 2), (0, 0, length / 2), RED),
                             _box("hub", (6 * MM, 6 * MM, 4 * MM), (0, 0, 0), GREY)], 0.01)
        contact = np.array([-4 * MM, 0.0, length - 6 * MM])
        geo = ControlGeometry(moving, recess_walls(length * 0.6, 10 * MM, recess), contact, up.copy(),
                              np.array([1.0, 0.0, 0.0]), handle_body=np.array([0.0, 0.0, length - 4 * MM]),
                              handle_kind="bar", rotary=True, lever_arm=length - 6 * MM,
                              footprint=(length + 12 * MM, 24 * MM))
    elif cls == "door":
        w, h = sz[0], sz[1]
        open_hi = travel + 0.3
        moving = MovingPart("door", (-w / 2, 0, 3 * MM), (1, 0, 0, 0), "hinge", (0, -1, 0),
                            (-(3 * MM) / w, open_hi), [_box("plate", (w / 2, h / 2, 2 * MM), (w / 2, 0, 0), RED)], 0.02)
        hx = w - (15 * MM if spec.handle == "pull" else 5 * MM if spec.handle == "tab" else 10 * MM)
        hgeoms, hpt = _handle_geoms(spec.handle, hx, 2 * MM)
        moving.geoms.extend(hgeoms)
        contact = np.array([w - 10 * MM, 0.0, 2 * MM])
        geo = ControlGeometry(moving, recess_walls(w / 2 + 3 * MM, h / 2 + 3 * MM, recess), contact, up,
                              np.array([0, 0, -1.0]), roll_ctl=np.array([1.0, 0.0, 0.0]), handle_body=hpt,
                              handle_kind=spec.handle, rotary=True, lever_arm=w - 10 * MM,
                              footprint=(w + 10 * MM, h + 10 * MM))
    elif cls == "drawer":
        w, h = sz[0], sz[1]
        moving = MovingPart("drawer", (0, 0, 4 * MM), (1, 0, 0, 0), "slide", (0, 0, 1),
                            (-3 * MM, travel + 20 * MM), [_box("front", (w / 2, h / 2, 3 * MM), (0, 0, 0), RED)], 0.03)
        hgeoms, hpt = _handle_geoms(spec.handle, 0.0, 3 * MM)
        moving.geoms.extend(hgeoms)
        geo = ControlGeometry(moving, recess_walls(w / 2 + 3 * MM, h / 2 + 3 * MM, recess),
                              np.array([0.0 if spec.handle != "tab" else -12 * MM, 0.0, 3 * MM]), up,
                              np.array([0, 0, -1.0]), roll_ctl=np.array([1.0, 0.0, 0.0]), handle_body=hpt,
                              handle_kind=spec.handle, footprint=(w + 10 * MM, h + 10 * MM))
    elif cls == "slider":
        length = sz[0]
        y0 = -length / 2 + 5 * MM
        moving = MovingPart("knob", (0, y0, 0), (1, 0, 0, 0), "slide", (0, 1, 0), (-1 * MM, travel + 3 * MM),
                            [_box("knob", (5 * MM, 4 * MM, 5 * MM), (0, 0, 5 * MM), RED)], 0.006)
        track = _box("track", (2 * MM, length / 2 + 5 * MM, 0.5 * MM), (0, 0, 0.5 * MM), GREY)
        geo = ControlGeometry(moving, [track] + recess_walls(8 * MM, length / 2 + 8 * MM, recess),
                              np.array([0.0, -4 * MM, 5 * MM]), up, np.array([0.0, 1.0, 0.0]),
                              roll_ctl=np.array([0.0, 1.0, 0.0]), footprint=(30 * MM, length + 16 * MM))
    elif cls == "slot":
        w, gap = sz[0], sz[1]
        hh = travel + 10 * MM
        side = 6 * MM
        static = [
            _box("house_py", (w / 2 + 5 * MM, side / 2, hh / 2), (0, gap / 2 + side / 2, hh / 2), GREY),
            _box("house_ny", (w / 2 + 5 * MM, side / 2, hh / 2), (0, -gap / 2 - side / 2, hh / 2), GREY),
            _box("house_px", (2.5 * MM, gap / 2, hh / 2), (w / 2 + 2.5 * MM, 0, hh / 2), GREY),
            _box("house_nx", (2.5 * MM, gap / 2, hh / 2), (-w / 2 - 2.5 * MM, 0, hh / 2), GREY),
        ]
        moving = MovingPart("plunger", (0, 0, hh - 5 * MM), (1, 0, 0, 0), "slide", (0, 0, -1), (0.0, travel + 3 * MM),
                            [_box("plunger", (w / 2 - 0.5 * MM, max(gap / 2 - 0.15 * MM, 0.2 * MM), 2 * MM), (0, 0, 0), RED)],
                            0.004)
        geo = ControlGeometry(moving, static + recess_walls(w / 2 + 6 * MM, gap / 2 + side + 1 * MM, recess),
                              np.array([0.0, 0.0, 2 * MM]), up, np.array([0, 0, -1.0]),
                              roll_ctl=np.array([1.0, 0.0, 0.0]), footprint=(w + 14 * MM, gap + 2 * side + 8 * MM))
        if spec.handle != "none":
            fw, fh = w / 2 + 7 * MM, gap / 2 + side
            flap = MovingPart("flap", (-fw, 0, hh + 1 * MM), (1, 0, 0, 0), "hinge", (0, -1, 0), (0.0, 2.2),
                              [_box("flap", (fw, fh, 1 * MM), (fw, 0, 0), BLUE)], 0.005)
            fg, fpt = _handle_geoms(spec.handle, 2 * fw - 5 * MM, 1 * MM)
            flap.geoms.extend(fg)
            geo.flap = flap
            geo.flap_handle_body = fpt
            geo.handle_kind = spec.handle
    elif cls == "grasp":
        geo = _grasp_geometry(spec)
    else:  # pragma: no cover - ControlClass is a closed Literal
        raise ValueError(cls)
    return geo


GRASP_BOX_DEPTH = 30 * MM  # box extent along the approach (sim/grasp_world.py BOX_DEPTH_MM)


def _grasp_geometry(spec: ControlSpec) -> ControlGeometry:
    """A free object standing on the table: width ``size_mm[0]`` across the jaws (ctl y), height
    ``size_mm[1]``.  The tool comes straight down onto its top and the jaws close along ctl y."""
    w, hgt = spec.size_mm[0] * MM, spec.size_mm[1] * MM
    if spec.asset is not None:
        return _asset_geometry(spec)
    if spec.shape == "sphere":
        g, hgt = GeomSpec("ball", "sphere", (w / 2,), rgba=GREEN), w
    elif spec.shape == "box":
        g = _box("box", (GRASP_BOX_DEPTH / 2, w / 2, hgt / 2), rgba=GREEN)
    else:
        g = _cyl("can", w / 2, hgt / 2, rgba=GREEN)
    moving = MovingPart("object", (0, 0, hgt / 2), (1, 0, 0, 0), "free", (0, 0, 1), (0.0, 1.0), [g],
                        spec.mass_g / 1000)
    depth = GRASP_BOX_DEPTH if spec.shape == "box" else w
    return ControlGeometry(moving, [], contact_body=np.array([0.0, 0.0, hgt / 2]),
                           approach_ctl=np.array([0.0, 0.0, -1.0]), actuation_body=np.array([0.0, 0.0, 1.0]),
                           roll_ctl=np.array([0.0, 1.0, 0.0]), handle_body=np.zeros(3), footprint=(depth, w))


def _asset_geometry(spec: ControlSpec) -> ControlGeometry:
    """A real object (configs/ycb_objects.yaml): its convex parts collide, its scan is drawn, and its
    inertia is the scan's at the published mass.  The asset frame has the object's bottom at z = 0,
    centred, with its narrowest horizontal extent along y (across the jaws)."""
    from sea.sim.objects import load_asset

    a = load_asset(spec.asset or "")
    w, hgt = spec.size_mm[0] * MM, spec.size_mm[1] * MM
    geoms = [GeomSpec(f"part{k}", "mesh", (), rgba=(*a.rgba[:3], 0.0), vertices=v) for k, v in enumerate(a.parts)]
    geoms.append(GeomSpec("scan", "mesh", (), rgba=tuple(a.rgba), vertices=a.visual_v, faces=a.visual_f, visual=True))
    moving = MovingPart("object", (0, 0, 0), (1, 0, 0, 0), "free", (0, 0, 1), (0.0, 1.0), geoms, spec.mass_g / 1000,
                        inertia=(a.com, a.inertia))
    return ControlGeometry(moving, [], contact_body=np.array([0.0, 0.0, hgt]), approach_ctl=np.array([0.0, 0.0, -1.0]),
                           actuation_body=np.array([0.0, 0.0, 1.0]), roll_ctl=np.array([0.0, 1.0, 0.0]),
                           handle_body=np.zeros(3), footprint=(a.extents_mm[1] * MM, w))


# ---------------------------------------------------------------- dynamics
def h(u: float) -> float:
    return u * math.exp(0.5 - u * u / 2.0)


@dataclass
class Faults:
    """Physics that differ from the description (planted-fault panels, tests/verifier)."""

    stiffness_scale: float = 1.0
    no_detent: bool = False
    reverse: bool = False  # joint axis flipped relative to the description
    latch_scale: float = 1.0
    friction_scale: float = 1.0
    jammed: bool = False  # joint range collapsed to the rest position


class ControlDynamics:
    """Per-control state machine and generalized force (one per control per episode)."""

    def __init__(self, spec: ControlSpec, stiffness_scale: float = 1.0, friction_scale: float = 1.0,
                 faults: Faults | None = None):
        self.spec = spec
        self.faults = faults or Faults()
        self.k = spec.stiffness * stiffness_scale * self.faults.stiffness_scale
        # a grasp object's friction is its surface's (build_model mixes it into the jaw contacts);
        # nothing pushes on its height coordinate but gravity and contact
        self.friction = 0.0 if spec.cls == "grasp" else spec.friction * friction_scale * self.faults.friction_scale
        self.rotary = spec.cls in ("toggle", "knob", "lever", "door")
        self.detents = [] if self.faults.no_detent else list(spec.detents)
        self.peak = 0.0 if self.faults.no_detent else spec.detent_peak
        if len(spec.detents) > 1:
            spacing = min(abs(b - a) for a, b in zip(sorted(spec.detents), sorted(spec.detents)[1:], strict=False))
            self.w = 0.15 * max(spacing, 1e-6)
        else:
            self.w = 0.15 * max(abs(spec.travel), 1e-6)
        has_latch = spec.cls in ("door", "drawer") and spec.handle != "none"
        self.has_latch = has_latch
        self.push_latch = has_latch and spec.handle == "push_latch"
        self.latch_threshold = (spec.latch_threshold or 0.0) * self.faults.latch_scale
        self.lever_r = 1.0
        self.reset()

    def reset(self) -> None:
        """Episode state back to rest: detent sides and count, latch, travel extremes."""
        self.q_rest = 0.0
        self.sides = [-1 if self.q_rest < d else 1 for d in self.detents]
        self.crossings = 0
        self.latched = self.has_latch and (self.push_latch or self.latch_threshold > 0)
        self.max_q = 0.0
        self.min_q = 0.0

    def set_lever_arm(self, r: float) -> None:
        """Force-to-torque arm for rotary latches (threshold is a force at the handle)."""
        self.lever_r = max(r, 1e-3)

    @property
    def latch_threshold_gen(self) -> float:
        """Latch threshold as a generalized force (N or N*m)."""
        return self.latch_threshold * (self.lever_r if self.rotary else 1.0)

    def detent_force(self, q: float) -> float:
        return -sum(self.peak * h((q - d) / self.w) for d in self.detents)

    def force(self, q: float, qdot: float) -> float:
        v0 = FRICTION_V0_ROT if self.rotary else FRICTION_V0_LIN
        f = -self.k * (q - self.q_rest) + self.detent_force(q) - self.friction * math.tanh(qdot / v0)
        if self.push_latch and not self.latched:
            k_pop = max(self.k, 0.2 if self.rotary else 30.0)
            f = -k_pop * (q - self.q_rest) - self.friction * math.tanh(qdot / v0)
        return f

    def update(self, q: float, latch_force: float = 0.0) -> None:
        """Advance detent counting and latch state after a step.

        ``latch_force`` is the magnitude of the latch limit's constraint force (generalized).
        """
        self.max_q = max(self.max_q, q)
        self.min_q = min(self.min_q, q)
        hyst = 0.2 * self.w
        for i, d in enumerate(self.detents):
            if self.sides[i] < 0 and q > d + hyst:
                self.sides[i] = 1
                self.crossings += 1
            elif self.sides[i] > 0 and q < d - hyst:
                self.sides[i] = -1
                self.crossings += 1
        if not self.latched:
            return
        if self.push_latch:
            depth = PUSH_LATCH_DEPTH / (self.lever_r if self.rotary else 1.0)
            if q <= -depth * 0.95:
                self.latched = False
                self.q_rest = PUSH_LATCH_OPEN_ROT if self.rotary else PUSH_LATCH_OPEN_LIN
        elif latch_force > self.latch_threshold_gen:
            self.latched = False

    def latch_upper(self) -> float | None:
        """Upper joint limit while latched (None = no latch limit)."""
        return 0.0 if self.latched else None
