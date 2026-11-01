"""The finger family (docs/gripper-experiment.md section 3): printed pads over the stock jaw tips.

Each jaw gets one pad: a block that covers the bottom ``COVER`` of the stock jaw's inner face and
extends ``reach_mm`` past its tip.  The parameters (``FingerSpec``, all zero for the stock jaws):

- ``reach_mm``: how far the pads extend past the stock tips (grasp depth, low objects);
- ``pad_angle_deg``: the moving pad is tilted back by this angle, so its face is parallel to the
  fixed pad's when the jaw has opened by the same angle.  The stock faces are parallel only when
  closed (q = 0), and at the 66 mm a soup can needs they open downward by about 40 degrees, which
  squeezes the can out of the grip (D-038);
- ``groove_mm``: depth of a V-groove across each pad face, which centres round objects;
- ``lip_mm``: an inward hook at the bottom of each pad, for edges and objects lying flat;
- ``pad_tpu_mm``: a TPU skin on each face: friction 0.9 instead of bare PLA's 0.45.  Only the friction
  is modelled: a softer contact time constant, unmeasured, made the skin hold worse than bare PLA (D-038);
- ``width_mm``: the pad width across the jaw (default ``WIDTH``).

``pad_geoms`` gives the boxes ``build_model`` adds to the two jaw bodies; ``gap`` and ``opening_for``
give the distance between the pad faces at any jaw angle, which the capability numbers and the grasp
primitive use.  Stock geometry constants were measured from the SO-101 jaw meshes (D-038).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sea.contracts import FingerSpec

MM = 1e-3
# Fixed jaw (gripper body frame): its inner face is the plane x = FIXED_FACE_X over the bottom 10 mm,
# and its tip is at z = FIXED_TIP_Z.  The stock tool point sits on that face.
FIXED_FACE_X = -7.9 * MM
FIXED_TIP_Z = -104.4 * MM
# Moving jaw: its body frame is the gripper frame rotated by Rx(90 deg), then by q about local z (the
# hinge), with origin HINGE.  Its inner tip face is the plane x_local = MOVING_FACE_X, lowest point
# MOVING_TIP_Y, centred across the jaw at z_local = MOVING_MID_Z.
HINGE = np.array([20.2, 18.8, -23.4]) * MM
MOVING_FACE_X = -12.3 * MM
MOVING_TIP_Y = -80.5 * MM
MOVING_MID_Z = 18.9 * MM
R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # moving body axes in the gripper frame

COVER = 10 * MM  # how much of the stock jaw face a pad covers
THICK = 3 * MM  # PLA backing: at least min_wall everywhere
WIDTH = 12.0  # mm, pad width when width_mm is 0
LIP_THICK = 2 * MM
PLA_FRICTION, TPU_FRICTION = 0.45, 0.9
RANGES: dict[str, tuple[float, float]] = {
    "reach_mm": (0.0, 30.0),
    "pad_angle_deg": (0.0, 45.0),
    "groove_mm": (0.0, 6.0),
    "lip_mm": (0.0, 4.0),
    "pad_tpu_mm": (0.0, 3.0),
    "width_mm": (8.0, 24.0),
}


def is_stock(spec: FingerSpec) -> bool:
    return spec == FingerSpec()


def width(spec: FingerSpec) -> float:
    return (spec.width_mm or WIDTH) * MM


def thickness(spec: FingerSpec) -> float:
    """Pad face offset from the stock jaw face: backing plus skin."""
    return 0.0 if is_stock(spec) else THICK + spec.pad_tpu_mm * MM


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def moving_to_gripper(q: float, p_local: np.ndarray) -> np.ndarray:
    return HINGE + R0 @ (_rz(q) @ p_local)


def tip_rise(q: float) -> float:
    """How much higher the moving jaw's tip is at opening ``q`` than closed (m)."""
    tip = np.array([MOVING_FACE_X, MOVING_TIP_Y, MOVING_MID_Z])
    return float(moving_to_gripper(q, tip)[2] - moving_to_gripper(0.0, tip)[2])


def moving_reach(spec: FingerSpec) -> float:
    """The moving pad is longer by the tip's rise at its parallel opening, so at that opening both
    pads end at the same height."""
    return spec.reach_mm * MM + tip_rise(math.radians(spec.pad_angle_deg))


@dataclass(frozen=True)
class PadGeom:
    body: str  # "gripper" (fixed jaw) or "moving_jaw_so101_v1"
    kind: str  # "jaw_fixed" | "jaw_moving": the verifier's jaw names
    name: str
    half: tuple[float, float, float]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # wxyz, in the body frame
    friction: float


def _quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    from sea.control.frames import mat_to_quat_wxyz

    return tuple(float(v) for v in mat_to_quat_wxyz(rot))  # type: ignore[return-value]


def _face_boxes(spec: FingerSpec, length: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(centre, half sizes, rotation) of the pad boxes in a pad frame where +x points at the object,
    +y runs up the jaw from the pad's bottom end at y = 0, and z runs across the pad."""
    t, w, g = thickness(spec), width(spec), spec.groove_mm * MM
    h = length + COVER
    out = []
    if g <= 0:
        out.append((np.array([t / 2, h / 2, 0.0]), np.array([t / 2, h / 2, w / 2]), np.eye(3)))
    else:  # two half-faces that recess by g at the centre line: a V across the pad
        y = np.array([0.0, 1.0, 0.0])
        for side in (1.0, -1.0):
            a = np.array([t, 0.0, side * w / 2])  # face edge
            b = np.array([t - g, 0.0, 0.0])  # groove bottom
            n = np.cross(b - a, y)
            n = n / np.linalg.norm(n) * (1.0 if n[0] > 0 else -1.0)  # face normal, toward the object
            u = np.cross(n, y)  # along the face, so that (n, y, u) is right-handed
            mid = (a + b) / 2 - n * t / 2
            mid[1] = h / 2
            out.append((mid, np.array([t / 2, h / 2, np.linalg.norm(b - a) / 2]), np.column_stack([n, y, u])))
    if spec.lip_mm > 0:
        lip = spec.lip_mm * MM
        out.append((np.array([t + lip / 2, LIP_THICK / 2, 0.0]), np.array([lip / 2 + t / 2, LIP_THICK / 2, w / 2]),
                    np.eye(3)))
    return out


def pad_geoms(spec: FingerSpec) -> list[PadGeom]:
    """The boxes ``build_model`` adds to the jaw bodies; none for the stock jaws."""
    if is_stock(spec):
        return []
    mu = TPU_FRICTION if spec.pad_tpu_mm > 0 else PLA_FRICTION
    out: list[PadGeom] = []
    # Fixed pad, gripper frame: pad +x = gripper +x (toward the moving jaw), pad +y = gripper +z,
    # pad z = gripper -y; the pad's bottom is reach below the stock tip.
    rf = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # pad axes in gripper coords
    base_f = np.array([FIXED_FACE_X, 0.0, FIXED_TIP_Z - spec.reach_mm * MM])
    for k, (c, half, rot) in enumerate(_face_boxes(spec, spec.reach_mm * MM)):
        out.append(PadGeom("gripper", "jaw_fixed", f"jaw_fixed_pad{k}", tuple(half), tuple(base_f + rf @ c),
                           _quat(rf @ rot), mu))
    # Moving pad, moving body frame, tilted back by the pad angle about the hinge axis through the
    # stock tip's inner corner: pad +x = local -x (toward the fixed jaw), pad +y = local +y (up).
    th = math.radians(spec.pad_angle_deg)
    rm = _rz(-th) @ np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    corner = np.array([MOVING_FACE_X, MOVING_TIP_Y, MOVING_MID_Z])
    lm = moving_reach(spec)
    base_m = corner + _rz(-th) @ np.array([0.0, -lm, 0.0])
    for k, (c, half, rot) in enumerate(_face_boxes(spec, lm)):
        out.append(PadGeom("moving_jaw_so101_v1", "jaw_moving", f"jaw_moving_pad{k}", tuple(half),
                           tuple(base_m + rm @ c), _quat(rm @ rot), mu))
    return out


def gap(spec: FingerSpec, q: float, height: float | None = None) -> float:
    """Distance (m) across the jaws between the two pad faces at jaw opening ``q``, measured at
    ``height`` above the fixed pad's bottom (default: the middle of the reach)."""
    t = thickness(spec)
    z = FIXED_TIP_Z - spec.reach_mm * MM + (spec.reach_mm * MM / 2 if height is None else height)
    th = math.radians(spec.pad_angle_deg)
    lm = moving_reach(spec) if not is_stock(spec) else 0.0
    corner = np.array([MOVING_FACE_X, MOVING_TIP_Y, MOVING_MID_Z])
    # the moving face line (pad +x offset t toward the fixed jaw), in local coords, then gripper
    n_loc = _rz(-th) @ np.array([-1.0, 0.0, 0.0])
    up_loc = _rz(-th) @ np.array([0.0, 1.0, 0.0])
    p0 = moving_to_gripper(q, corner + n_loc * t - up_loc * lm)
    d = R0 @ (_rz(q) @ up_loc)
    s = (z - p0[2]) / d[2] if abs(d[2]) > 1e-9 else 0.0
    x_moving = p0[0] + d[0] * s
    return float(x_moving - (FIXED_FACE_X + t))


Q_RANGE = (-0.17, 1.2)  # closed to GRIP_WIDE: the openings the skills use; the gap grows monotonically


def opening_for(spec: FingerSpec, width_m: float) -> float | None:
    """The jaw angle at which the pads are ``width_m`` apart (bisection), or None past the widest opening.
    Past about 1.2 rad the moving pad swings toward horizontal and the face-line gap stops meaning a width."""
    lo, hi = Q_RANGE
    if gap(spec, hi) < width_m:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if gap(spec, mid) < width_m:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


TOOL_Z = -98.1 * MM  # the stock tool point (gripper_frame_link) in the gripper frame


def _corners(g: PadGeom) -> np.ndarray:
    from sea.control.frames import quat_wxyz_to_mat

    rot = quat_wxyz_to_mat(g.quat)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
    return np.asarray(g.pos) + (signs * np.asarray(g.half)) @ rot.T


def below_tool(spec: FingerSpec, q: float) -> float:
    """How far below the tool point (m) the lowest pad corner reaches at jaw opening ``q``."""
    low = 0.0
    for g in pad_geoms(spec):
        c = _corners(g)
        z = c[:, 2] if g.body == "gripper" else np.array([moving_to_gripper(q, p)[2] for p in c])
        low = max(low, TOOL_Z - float(z.min()))
    return low
