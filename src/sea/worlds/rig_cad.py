"""The talking robot as a bench instrument (docs/design-language.md): the assembly ``worlds/cad.phonology`` draws.

Every moving part is placed by the rig's controls (``P.rig``, System 1's postures) through the mechanism that would
move it:
- a crank on the rail servo slides the tongue's sled along its rail (``tongue_place``);
- an eccentric cam on the sled's servo lifts the follower rod that bulges the tongue pad (``tongue_height``);
- two servos on a bridge over the tract carry the tip blade through a five-bar linkage (``tip_place``, ``tip_height``);
- a lead-screw slide lowers the reed housing and stretches a bellows (``larynx``);
- a servo pulls the jaw down its guide (``jaw``), or pushes the lip ring forward and draws its string
  (``protrusion``, ``lip_radius``).

The acoustics are the world's: this module reads only ``P.rig``, the bore and the articulators' profile constants.

Frame (mm): x along the tract from the glottis to the teeth at x = 0; y across, with the camera at -y; z up, with the
bore's axis at z = 0 and the base plate's top at ``ZP``.  Cast parts, the reed housing and the face plate have a section
mesh with the half toward the camera removed, so the bore shows; mechanisms are drawn whole.  Each part is a
``language.Member`` with its legend line and print orientation, and its joints are declared, so ``language.audit``
can check that nothing floats and that parts share space only where a joint says so.
"""

from __future__ import annotations

import math
from functools import cache

import cadquery as cq
import numpy as np
import trimesh

from sea.design import language as LG
from sea.design.cad import CadModel, CadPart, View
from sea.design.language import tessellate as to_mesh
from sea.evolve.core import Body
from sea.worlds import phonology as P

V = cq.Vector
R = 10 * P.BORE  # bore radius
RO = R + 4.0  # the cast's outer radius: a 4 mm silicone wall
ZP = -66.0  # top of the base plate
UP = (0.0, 0.0, 1.0)
NEAR = cq.Solid.makeBox(4000, 2000, 4000, V(-2000, -2000, -2000))  # y < 0: removed in section
RAIL_MM = {"cam_tongue": (70.0, 100.0), "tip_tongue": (70.0, 100.0), "long_rail_tongue": (100.0, 150.0)}  # travel, rail
LINER_MM = 1.2  # a TPU membrane's thickness; a cast one is silicone's wall
X_STEM = -10.0  # the tip stem's front limit: its crosshead and pushrod eyes clear the cast's flange

# MG90S, in its own frame: the shaft on +z at x = y = 0, the case running toward +x
CASE = (-6.1, 16.7, 6.1, 22.7)  # x from, x to, half width, height
TABS = (15.9, 18.4, 4.75)  # tab underside, top, overhang past each end
DOME = (5.9, 26.7)  # gear dome radius, and its top: where a placement puts ``at``
SPLINE = (2.4, 29.9)
HORN = (3.5, 30.7, 1.6)  # hub radius, top, arm thickness
FOOT_DOWN = 9.6  # a lying servo's shaft above the foot of its mount
FOOT_STAND = 29.7  # a standing servo's dome top above the foot of its mount
ARM_Y = 3.2  # a servo whose shaft points -y has its horn arm centred on y = 0 when its dome top is here


def posture(body: Body) -> dict[str, float]:
    """Each articulator at the end of its travel: the body pad mid-rail at full stroke, the tip blade sealing the
    alveolar ridge, the lips open and fully protruded, the larynx lowered.  The default picture of a body."""
    r = P.rig(body)
    out = {k: hi for k, (lo, hi) in r.controls.items()}
    if "tongue_place" in r.controls:
        lo, hi = r.controls["tongue_place"]
        out["tongue_place"] = (lo + hi) / 2
    if "tip_place" in r.controls:
        out["tip_place"] = 1.5
    out["jaw"] = 0.5 if "jaw" in out else 0.0
    return out


# ---------------------------------------------------------------- shapes
def _cyl(r: float, length: float, at, axis) -> cq.Solid:
    a = V(*axis).normalized()
    return cq.Solid.makeCylinder(r, length, V(*at) - a * (length / 2), a)


def _rod(p0, p1, r: float) -> cq.Solid:
    d = np.subtract(p1, p0)
    return _cyl(r, float(np.linalg.norm(d)), tuple(np.add(p0, p1) / 2), tuple(d))


def _box(lo, hi, fillet: float = 0.0) -> cq.Solid:
    s = cq.Solid.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], V(*lo))
    if fillet > 0:
        s = s.fillet(min(fillet, 0.45 * min(np.subtract(hi, lo))), s.Edges())
    return s


def _slot(r0: float, r1: float, length: float, thick: float) -> cq.Solid:
    """A tapered arm in the xy plane, from a hub of radius ``r0`` at the origin to a tip of ``r1`` at (length, 0)."""
    a = math.asin((r0 - r1) / length)
    p0, p1 = (r0 * math.sin(a), r0 * math.cos(a)), (length + r1 * math.sin(a), r1 * math.cos(a))
    wp = (cq.Workplane("XY").moveTo(*p0).lineTo(*p1).threePointArc((length + r1, 0), (p1[0], -p1[1]))
          .lineTo(p0[0], -p0[1]).threePointArc((-r0, 0), p0).close())
    return wp.extrude(thick).val()


def _frame(at, shaft, body) -> np.ndarray:
    """Servo frame -> model: the shaft along ``shaft``, the case along ``body``, the dome top at ``at``."""
    ez = np.asarray(shaft, float) / np.linalg.norm(shaft)
    ex = np.asarray(body, float) - ez * np.dot(body, ez)
    ex /= np.linalg.norm(ex)
    out = np.eye(4)
    out[:3, :3] = np.column_stack([ex, np.cross(ez, ex), ez])
    out[:3, 3] = np.asarray(at, float) - DOME[1] * ez
    return out


def _moved(mesh: trimesh.Trimesh, pose: np.ndarray) -> trimesh.Trimesh:
    return mesh.copy().apply_transform(pose)


def _pt(pose: np.ndarray, p) -> np.ndarray:
    return pose[:3, :3] @ np.asarray(p, float) + pose[:3, 3]


@cache
def _servo_mesh() -> trimesh.Trimesh:
    x0, x1, hw, h = CASE
    case = _box((x0, -hw, 0), (x1, hw, h), 0.8)
    tabs = _box((x0 - TABS[2], -hw, TABS[0]), (x1 + TABS[2], hw, TABS[1]), 0.6)
    for x in (x0 - 2.4, x1 + 2.4):
        tabs = tabs.cut(_cyl(1.0, 5, (x, 0, TABS[0] + 1), UP))
    dome = _cyl(DOME[0], DOME[1] - h + 0.2, (0, 0, (DOME[1] + h - 0.2) / 2), UP)
    spline = _cyl(SPLINE[0], SPLINE[1] - DOME[1], (0, 0, (SPLINE[1] + DOME[1]) / 2), UP)
    return to_mesh(case.fuse(tabs, dome, spline).clean())


@cache
def _mount_mesh(lying: bool) -> trimesh.Trimesh:
    """The printed cradle a servo's tabs screw to: standing, a foot under the case and a post under each tab;
    lying, an L of a tab plate with a window for the case and a foot along its down side (servo frame -y)."""
    x0, x1, hw, _ = CASE
    xa, xb = x0 - TABS[2] - 1.2, x1 + TABS[2] + 1.2
    if not lying:
        foot = _box((xa, -hw - 5, -3), (xb, hw + 5, 0), 1.0)
        posts = [_box((xa, -hw, -1), (x0 - 0.2, hw, TABS[0]), 1.0), _box((x1 + 0.2, -hw, -1), (xb, hw, TABS[0]), 1.0)]
        return to_mesh(foot.fuse(*posts).clean())
    yd = -hw - 0.5
    prof = [(yd - 3, -1), (yd, -1), (yd, TABS[0] - 5.5), (yd + 3, TABS[0] - 2.5), (hw + 3, TABS[0] - 2.5),
            (hw + 3, TABS[0]), (yd - 3, TABS[0])]
    plate = cq.Workplane("YZ").polyline(prof).close().extrude(xb - xa).translate(V(xa, 0, 0))
    plate = plate.edges("|X").fillet(0.8).val()
    return to_mesh(plate.cut(_box((x0 - 0.2, -hw - 0.2, -2), (x1 + 0.2, hw + 0.2, TABS[0] + 1))))


@cache
def _screws_mesh(lying: bool) -> trimesh.Trimesh:
    """M2 screws through the tabs into the mount; M3 heads on a standing mount's foot."""
    x0, x1, hw, _ = CASE
    heads = [_cyl(1.9, 1.3, (x, 0, TABS[1] + 0.65), UP).fuse(_cyl(0.9, 6.0, (x, 0, TABS[1] - 3.0), UP))
             for x in (x0 - 2.4, x1 + 2.4)]
    if not lying:
        heads += [_cyl(2.8, 1.8, ((x0 + x1) / 2, s * (hw + 2.6), 0.9), UP) for s in (-1, 1)]
    return to_mesh(cq.Compound.makeCompound(heads))


@cache
def _horn_mesh(length: float) -> trimesh.Trimesh:
    hub = _cyl(HORN[0], HORN[1] - DOME[1], (0, 0, (HORN[1] + DOME[1]) / 2), UP)
    arm = _slot(HORN[0], 2.0, length, HORN[2]).translate(V(0, 0, HORN[1] - HORN[2]))
    return to_mesh(hub.fuse(arm).clean())


def _bump(place: float, height: float, k: float, half_w: float, back: float, from_top: bool = False) -> cq.Shape:
    """A soft articulator pressed into the bore at ``place`` cm behind the teeth: the tract's own radius profile
    (``P.tract``), drawn as the gap it leaves, which is the diameter of the model's radius there."""
    r0 = max(0.0, P.BORE - height)
    span = math.sqrt(1.0 / k)
    d = np.linspace(place - span, place + span, 41)
    rad = np.minimum(P.BORE, r0 + k * (P.BORE - r0) * (d - place) ** 2)
    top = np.clip(10 * (-P.BORE + 2 * (P.BORE - rad)), -R + 0.5, R - 0.01)
    pts = [(-10 * a, z) for a, z in zip(d, top, strict=True)] + [(-10 * d[-1], -R - 1), (-10 * d[0], -R - 1)]
    prof = cq.Workplane("XZ").polyline(pts).close().extrude(2 * half_w).translate(V(0, half_w, 0)).val()
    out = prof.intersect(_cyl(R, -back - 5.0, ((back - 4.5) / 2, 0, 0), (1, 0, 0)))  # within the cast
    return out.mirror("XY") if from_top else out


def _liner(x0: float, x1: float, depth: float, from_top: bool = False) -> cq.Shape:
    """An articulator's membrane: a half sleeve of even thickness ``depth`` set into a pocket in the cast's floor (or
    roof), flush with the bore, so the bore is round where nothing presses it and the pad has no feathered edge."""
    ring = _cyl(R + depth, x1 - x0, ((x0 + x1) / 2, 0, 0), (1, 0, 0)).cut(_cyl(R, x1 - x0 + 2, ((x0 + x1) / 2, 0, 0), (1, 0, 0)))
    out = ring.intersect(_box((x0 - 1, -R - depth - 1, -R - depth - 1), (x1 + 1, R + depth + 1, 0)))
    return out.mirror("XY") if from_top else out


def _inverse(f, target: float, lo: float, hi: float) -> float:
    """``x`` in [lo, hi] with f(x) = target, for f monotonic there (bisection; the nearer end if out of range)."""
    up = f(hi) > f(lo)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (f(mid) < target) == up else (lo, mid)
    return (lo + hi) / 2


def _reach(s: np.ndarray, p: np.ndarray, a: float, b: float, elbow: float) -> np.ndarray:
    """The tip (2-D) of a horn of length ``a`` turning about ``s`` whose link of length ``b`` reaches ``p``;
    ``elbow`` (+1 or -1) picks the side of the line from s to p."""
    d = p - s
    n = float(np.linalg.norm(d))
    n = min(max(n, abs(b - a) + 1e-6), a + b - 1e-6)
    along = (a * a - b * b + n * n) / (2 * n)
    h = math.sqrt(max(a * a - along * along, 0.0))
    u = d / np.linalg.norm(d)
    return s + u * along + elbow * h * np.array([-u[1], u[0]])


# ---------------------------------------------------------------- the assembly
class _Rig:
    def __init__(self, body: Body, ctl: dict[str, float] | None):
        self.body, self.r = body, P.rig(body)
        self.c = posture(body) | (ctl or {})
        self.a = LG.Assembly()
        self.L = 10 * (self.r.length + self.c.get("larynx", 0.0))  # glottis to teeth
        self.base = "base plate"
        self.sweep: list[tuple[float, float, float, float]] = []  # (x0, y0, x1, y1) a moving part covers over its travel

    def part(self, name: str, slot: str, shape=None, material: str = "printed", role: str = "", up=UP,
             section: bool = False, mesh: trimesh.Trimesh | None = None) -> str:
        """Add a part from a CadQuery shape (or a ready mesh); ``section`` also keeps its half away from the camera."""
        whole = mesh if mesh is not None else to_mesh(shape)
        cut = None
        if section:
            half = shape.cut(NEAR)
            cut = to_mesh(half) if half.Volume() > 1e-6 else trimesh.Trimesh()
        printed = material in ("printed", "tpu")
        return self.a.add(LG.Member(name, CadPart(name, slot, whole, material), role, up if printed else None, cut))

    def join(self, a: str, b: str, kind: LG.JointKind = "contact") -> None:
        self.a.join(a, b, kind)

    def servo(self, name: str, slot: str, role: str, at, shaft, body, on: str, horn: tuple | None = None,
              lying: bool = True) -> dict:
        """An MG90S in its printed mount on the part ``on``, with a horn of ``horn`` = (direction in the model frame,
        length): the parts' keys, the servo frame and the horn's tip."""
        pose = _frame(at, shaft, body)
        s = self.part(name, slot, material="servo", role=role, mesh=_moved(_servo_mesh(), pose))
        m = self.part("servo mount", slot, role="printed cradle: the servo's tabs screw to it",
                      up=tuple(-pose[:3, 1]) if lying else tuple(pose[:3, 2]), mesh=_moved(_mount_mesh(lying), pose))
        k = self.part("screws", slot, material="metal", role="M2 and M3: every part is screwed down, none glued",
                      mesh=_moved(_screws_mesh(lying), pose))
        self.join(s, m)
        self.join(m, on)
        self.join(k, m, "joint")
        self.join(k, s, "joint")
        out = {"servo": s, "mount": m, "pose": pose}
        if horn is not None:
            d, length = np.asarray(horn[0], float), horn[1]
            ang = math.atan2(d @ pose[:3, 1], d @ pose[:3, 0])
            turn = np.eye(4)
            turn[:3, :3] = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])[:3, :3]
            h = self.part("horn", slot, role="on the servo's spline: turns with the shaft", mesh=_moved(_horn_mesh(length), pose @ turn))
            self.join(h, s, "joint")
            mid = HORN[1] - HORN[2] / 2
            out |= {"horn": h, "tip": _pt(pose, (length * math.cos(ang), length * math.sin(ang), mid)),
                    "top": float(_pt(pose, (0, 0, HORN[1]))[2])}
        return out

    def link(self, name: str, slot: str, p0, p1, ends: tuple[str, str], axis, role: str, r: float = 0.9,
             eye: float = 2.2, material: str = "metal") -> str:
        """A rod between two pins along ``axis``, an eye on each pin, joined to the parts at its ends."""
        shape = _rod(p0, p1, r).fuse(_cyl(eye, 3.0, tuple(p0), tuple(axis)), _cyl(eye, 3.0, tuple(p1), tuple(axis)))
        k = self.part(name, slot, shape, material, role)
        for e in ends:
            self.join(k, e, "joint")
        return k

    # -------------------------------------------------- the tract, the head and the glottis
    def tract(self) -> None:
        tr = self.body.parts["tract"]
        back = -10 * self.r.length + (6.0 if tr.type == "larynx_tube" else 0.0)  # a bellows takes 6 mm at rest
        tube = _cyl(RO, -4 - back, ((back - 4) / 2, 0, 0), (1, 0, 0)).fuse(_cyl(RO + 5, 3.0, (-5.5, 0, 0), (1, 0, 0)))
        tube = tube.cut(_cyl(R, -back + 2, (back / 2, 0, 0), (1, 0, 0)))
        for lo, hi, z in self.slots:
            tube = tube.cut(_box((lo, -2.2, z - 3), (hi, 2.2, z + 3)))
        for liner in self.liners.values():
            tube = tube.cut(_liner(*liner))
        self.cast = self.part("cast tract", "tract", tube, "silicone", "cast silicone, bore 24 mm: the articulators squeeze it "
                              "from outside", section=True)
        w = RO + 10
        face = (cq.Workplane("YZ").center(0, (ZP + self.face_top) / 2).rect(2 * w, self.face_top - ZP).extrude(4)
                .edges("|X").fillet(6.0).translate(V(-4, 0, 0)).val())
        foot = _box((0, -w, ZP), (14, w, ZP + 4), 1.0)
        rib = cq.Workplane("XZ").polyline([(14, ZP + 4), (0, ZP + 4), (0, ZP + 18)]).close().extrude(4).val()
        face = face.fuse(foot, rib.translate(V(0, -w + 4, 0)), rib.translate(V(0, w, 0))).clean()
        face = face.cut(_cyl(R, 10, (-2, 0, 0), (1, 0, 0)))
        self.face = self.part("face plate", "", face, role="the head's frame: holds the cast at the teeth and the "
                              "lips in front", up=(1, 0, 0), section=True)
        self.join(self.cast, self.face)
        self.join(self.face, self.base)
        if tr.type == "larynx_tube":
            self.larynx(back)
        else:
            self.glottis(back, ZP, self.base, self.cast)

    def glottis(self, g: float, floor: float, on: str, front: str) -> None:
        """The reed housing with its front face at the glottis ``g``, standing on ``floor``: its air chamber, the
        reed over the glottal slit, and the air fitting and hose behind."""
        body = _cyl(RO + 5, 18, (g - 9, 0, 0), (1, 0, 0))
        stand = _box((g - 17, -RO, floor), (g - 1, RO, -RO), 1.0).cut(_cyl(RO + 5, 20, (g - 9, 0, 0), (1, 0, 0)))
        housing = body.fuse(stand).clean()
        housing = housing.cut(_cyl(R - 2, 11, (g - 8.5, 0, 0), (1, 0, 0)))  # air chamber
        housing = housing.cut(_box((g - 3.1, -6, -1.5), (g + 0.1, 6, 1.5)))  # glottal slit
        housing = housing.cut(_cyl(1.6, 6, (g - 16, 0, 0), (1, 0, 0)))  # inlet
        k = self.part("reed housing", "tract", housing, role="the glottis: pumped air buzzes the reed over a slit "
                      "into the tract", up=(1, 0, 0), section=True)
        kr = self.part("reed", "tract", _box((g - 3.5, -7, -7), (g - 3.0, 7, 7)), "metal", "brass: buzzes in the pumped air, the voice "
                       "source", section=True)
        barb = _cyl(2.6, 8, (g - 22, 0, 0), (1, 0, 0)).cut(_cyl(1.4, 10, (g - 22, 0, 0), (1, 0, 0)))
        kb = self.part("air fitting", "tract", barb, "metal", "barbed inlet: the pump's air enters here")
        x0, bend = g - 22.0, 12.0
        c45 = bend * (1 - math.cos(math.pi / 4))
        path = cq.Wire.assembleEdges([
            cq.Edge.makeLine(V(x0, 0, 0), V(x0 - 8, 0, 0)),
            cq.Edge.makeThreePointArc(V(x0 - 8, 0, 0), V(x0 - 8 - bend * math.sin(math.pi / 4), 0, -c45),
                                      V(x0 - 8 - bend, 0, -bend)),
            cq.Edge.makeLine(V(x0 - 8 - bend, 0, -bend), V(x0 - 8 - bend, 0, -bend - 18)),
        ])
        ring = cq.Workplane(cq.Plane(V(x0, 0, 0), V(0, 1, 0), V(-1, 0, 0))).circle(4.0).circle(2.4)
        hose = ring.sweep(cq.Workplane().add(path), transition="round").val()
        kh = self.part("air hose", "tract", hose, "rubber", "from the air pump, off the bench")
        self.join(k, front)
        self.join(k, on)
        self.join(kr, k)
        self.join(kb, k)
        self.join(kh, kb, "joint")
        self.housing = k

    def larynx(self, back: float) -> None:
        """A clamp holds the cast's back end; a bellows joins it to the reed housing, which rides a lead-screw slide
        the larynx servo turns: lowering the larynx lengthens the pharynx by the ``larynx`` control."""
        ext = 10 * self.c.get("larynx", 0.0)
        # a ring on an arm from a post beside the bench: the space under the cast is the tongue sled's travel
        yp = 38.0
        clamp = _cyl(RO + 4, 6, (back + 3, 0, 0), (1, 0, 0)).fuse(
            _box((back, RO + 2, -4), (back + 6, yp + 6, 4), 1.0), _box((back, yp, ZP + 3), (back + 6, yp + 6, 4), 1.0),
            _box((back - 4, yp - 1, ZP), (back + 10, yp + 9, ZP + 4), 1.0))
        clamp = clamp.cut(_cyl(RO, 8, (back + 3, 0, 0), (1, 0, 0))).cut(  # flat underneath, over the follower's guide
            _box((back - 1, -RO - 10, -RO - 12), (back + 7, RO + 10, -RO - 2)))
        kc = self.part("pharynx clamp", "tract", clamp, role="holds the cast's back end from a post beside the "
                       "tongue's travel", up=(1, 0, 0), section=True)
        self.join(kc, self.cast)
        self.join(kc, self.base)
        length = 6.0 + ext
        n = max(3, int(length // 3))
        pitch = length / n
        rings = [_cyl(RO if i % 2 == 0 else R + 1.5, pitch + 0.01, (back - (i + 0.5) * pitch, 0, 0), (1, 0, 0))
                 for i in range(n)]
        bel = rings[0].fuse(*rings[1:]).cut(_cyl(R, length + 2, (back - length / 2, 0, 0), (1, 0, 0)))
        kb = self.part("larynx bellows", "tract", bel, "silicone", "stretches as the larynx lowers",
                       section=True)
        self.join(kb, self.cast, "joint")
        self.join(kb, kc)
        g = back - length
        rest = back - 6.0
        travel = 10 * self.r.controls["larynx"][1]
        zs = ZP + FOOT_DOWN - 2.0  # the slide's top: its screw lies on it, on the servo's axis
        lo, hi = rest - travel - 30, rest - 1
        ks = self.part("larynx slide", "tract", _box((lo, -10, ZP), (hi, 10, zs), 1.0), "metal",
                       "40 mm lead-screw slide, on the base")
        self.join(ks, self.base)
        kca = self.part("slide carriage", "tract", _box((g - 20, -12, zs), (g + 2, 12, zs + 6), 1.0), "metal",
                        "carries the reed housing")
        self.join(kca, ks)
        kscr = self.part("lead screw", "tract", _rod((lo - 1.5, 0, zs + 2), (g - 20, 0, zs + 2), 2.0), "metal",
                         "turned by the larynx servo: drives the carriage")
        self.join(kscr, kca, "joint")
        self.join(kscr, ks)
        sv = self.servo("larynx servo", "tract", "turns the lead screw: lowers the larynx", (lo - 6.5, 0, zs + 2),
                        (1, 0, 0), (0, 1, 0), self.base)
        kcp = self.part("shaft coupler", "tract", _cyl(4.0, 6.0, (lo - 3.5, 0, zs + 2), (1, 0, 0)), "metal",
                        "joins the servo's spline to the lead screw")
        self.join(kcp, sv["servo"], "joint")
        self.join(kcp, kscr, "joint")
        self.glottis(g, zs + 6, kca, kb)

    # -------------------------------------------------- the tongue
    def tongue(self) -> None:
        to = self.body.parts["tongue"]
        if to.type == "pneumatic_tongue":
            return self.pneumatic()
        c = self.c
        travel, rail_mm = RAIL_MM[to.type]
        start, stroke = 10 * to.params["rail_start"], 10 * to.params["stroke"]
        xp, lift = -10 * c["tongue_place"], 10 * c["tongue_height"]
        yr = 12.0  # the rail's centre line, under the cam servo
        mid = min(-start - travel / 2, -5.0 - rail_mm / 2)
        k_rail = self.part("rail", "tongue", _box((mid - rail_mm / 2, yr - 3.5, ZP), (mid + rail_mm / 2, yr + 3.5, ZP + 5),
                                                    0.4), "metal", "linear rail: the tongue's place along the tract")
        self.join(k_rail, self.base)
        car = _box((xp - 11, yr - 8.5, ZP + 1.5), (xp + 11, yr + 8.5, ZP + 8), 0.6).cut(
            _box((xp - 12, yr - 3.55, ZP - 1), (xp + 12, yr + 3.55, ZP + 5.05)))
        k_car = self.part("carriage", "tongue", car, "metal", "rides the rail, carries the sled")
        self.join(k_car, k_rail)
        top = ZP + 11.0
        e, rc = stroke / 2, 9.0
        sled = _box((xp - 25, -10, ZP + 8), (xp + 14, 36, top), 1.0).cut(  # a window the cam turns through
            _box((xp - min(e + rc + 1.0, 16.0), -3.0, ZP + 7), (xp + e + rc + 1.0, 3.0, top + 1)))
        k_sled = self.part("sled", "tongue", sled, role="carries the cam servo and the follower's guide along the rail")
        lo_p, hi_p = self.r.controls["tongue_place"]
        self.sweep.append((-10 * hi_p - 25, -10.0, -10 * lo_p + 14, 36.0))
        self.join(k_sled, k_car)
        # the cam servo lies on the sled; its horn and cam turn in the plane y = 0 under the tract
        zs = top + FOOT_DOWN

        def rise(phi: float) -> float:
            return -e * math.cos(phi) + math.sqrt(rc * rc - (e * math.sin(phi)) ** 2)

        phi = _inverse(lambda f: rise(f) - rise(0.0), min(lift, 2 * e), 0.0, math.pi)
        d = np.array([math.sin(phi), 0.0, -math.cos(phi)])
        sv = self.servo("cam servo", "tongue", "turns the cam: how far the tongue rises", (xp, 6.0, zs), (0, -1, 0),
                        (-1, 0, 0), k_sled, horn=(d, max(e, 4.0)))
        centre = np.array([xp, 0.0, zs]) + e * d
        k_cam = self.part("cam", "tongue", _cyl(rc, 4.0, (centre[0], 0.0, centre[2]), (0, 1, 0)),
                          role="eccentric, screwed to the horn: lifts the follower", up=(0, -1, 0))
        self.join(k_cam, sv["horn"], "joint")
        z_roll, z_end = zs + rise(phi) + 3.0, -R + lift
        follower = _cyl(3.0, 4.0, (xp, 0, z_roll), (0, 1, 0)).fuse(_rod((xp, 0, z_roll), (xp, 0, z_end), 1.5))
        k_fol = self.part("follower rod", "tongue", follower, "metal", "rides the cam on its roller and pushes up "
                          "through the cast's floor slot into the tongue pad")
        self.join(k_fol, k_cam)
        # the guide: an upright behind the cam and an arm over it, with a bushing around the rod
        zg = zs + e + rc + 6.0 + 1.5
        zk = ZP + FOOT_STAND - DOME[1] + HORN[1]  # the rail servo's horn top: the crank's underside
        upright = _box((xp - 21, -6, top), (xp - 16.5, 6, zg + 5), 1.0)
        arm = _box((xp - 21, -4, zg), (xp + 4, 4, zg + 5), 1.0).cut(_cyl(1.85, 12, (xp, 0, zg + 2.5), UP))
        tab = _box((xp - 21, -12, zk), (xp - 16.5, -5.5, zk + 3), 0.8)
        k_guide = self.part("follower guide", "tongue", upright.fuse(arm, tab).clean(), role="keeps the follower "
                            "upright; the connecting rod drives the sled by its pin", up=(0, -1, 0))
        self.join(k_guide, k_sled)
        self.join(k_fol, k_guide, "guide")
        self.join(k_fol, self.cast, "guide")
        pad = _bump(c["tongue_place"], c["tongue_height"], P.K_BODY, R, self.back).fuse(_liner(*self.liners["body"])).cut(
            _cyl(1.5, 40, (xp, 0, z_end - 20), UP))
        k_pad = self.part("tongue pad", "tongue", pad, "tpu", "soft TPU: bulges into the bore where the "
                          "follower lifts it", section=True)
        self.join(k_pad, k_fol, "joint")
        self.join(k_pad, self.cast)
        self.crank(xp - 18.75, -start - travel - 18.75, travel, zk, k_guide)
        if to.type == "tip_tongue":
            self.tip()

    def crank(self, x_pin: float, x_far: float, travel: float, zk: float, on: str) -> None:
        """The rail servo stands in front of the rail; a crank on its horn and a connecting rod slide the sled
        (an offset slider-crank: the crank turns half a turn for the rail's whole travel)."""
        a, y_pin, y_c = travel / 2, -8.75, -36.0
        b = (y_pin - y_c) + a + 12.0

        def x_of(th: float) -> float:
            return a * math.cos(th) + math.sqrt(b * b - (y_pin - y_c - a * math.sin(th)) ** 2)

        ths = np.linspace(0.6 * math.pi, 2.4 * math.pi, 721)
        xs = np.array([x_of(t) for t in ths])
        i0, i1 = sorted((int(np.argmin(xs)), int(np.argmax(xs))))
        x_c = round((x_far - xs.min() + 1.0) / LG.GRID) * LG.GRID
        th = _inverse(x_of, x_pin - x_c, ths[i0], ths[i1])
        u = np.array([math.cos(th), math.sin(th), 0.0])
        sv = self.servo("rail servo", "tongue", "swings the crank: where the tongue sits along the tract",
                        (x_c, y_c, ZP + FOOT_STAND), UP, (-1, 0, 0), self.base, horn=(u, 12.0), lying=False)
        crank = _slot(4.5, 3.2, a, 3.0).rotate(V(0, 0, 0), V(0, 0, 1), math.degrees(th)).translate(V(x_c, y_c, zk))
        k_crank = self.part("crank", "tongue", crank, role="screwed to the rail servo's horn")
        self.join(k_crank, sv["horn"], "joint")
        self.sweep.append((x_c - a - 5, y_c - a - 5, x_c + a + 5, y_c + a + 5))
        z = zk + 4.5
        tip = (x_c + a * u[0], y_c + a * u[1], z)
        self.link("connecting rod", "tongue", tip, (x_pin, y_pin, z), (k_crank, on), UP,
                  "printed link from the crank to the sled's pin", r=1.6, eye=3.2, material="printed")

    def tip(self) -> None:
        """Two servos on a bridge over the tract carry the tip blade through a five-bar linkage: together they set
        where it presses (x) and how deep (z).  The blade hangs under the palate on a stem through the cast's roof."""
        c = self.c
        xt, lift = -10 * c["tip_place"], 10 * c["tip_height"]
        zb = 44.0
        bridge = _box((-92, 10, zb - 4), (-4, 34, zb), 1.0).fuse(_box((-92, 28, zb - 14), (-4, 34, zb - 3), 1.0)).clean()
        k_br = self.part("palate bridge", "tongue", bridge, role="over the tract, cantilevered from the face plate: "
                         "carries the tip servos", up=(0, 1, 0))
        self.join(k_br, self.face)
        zs, pz = zb + FOOT_DOWN, 33.0 - lift  # the servos' axes; the crosshead
        xt = min(xt, X_STEM)  # the stem stays behind the cast's flange; the blade reaches forward of it
        p = np.array([xt, pz])
        ends = []
        for sx, a, b, elbow, y_end in ((-27.0, 12.0, 30.0, 1.0, 0.0), (-63.0, 24.0, 46.0, -1.0, -3.4)):
            s = np.array([sx, zs])
            h = _reach(s, p, a, b, elbow)
            sv = self.servo("tip servo", "tongue", "one of a pair: together they place the tip blade", (sx, ARM_Y, zs),
                            (0, -1, 0), (-1, 0, 0), k_br, horn=(np.array([h[0] - sx, 0.0, h[1] - zs]), a))
            ends.append((sv["tip"], sv["horn"], y_end))
        cross = _rod((xt, -5.4, pz), (xt, 1.6, pz), 1.0).fuse(_rod((xt, 0, pz), (xt, 0, R - lift), 1.5))
        k_stem = self.part("tip stem", "tongue", cross, "metal", "crosshead and stem: carry the blade through the "
                           "cast's roof slot")
        self.join(k_stem, self.cast, "guide")
        for tip_pt, horn, y_end in ends:
            self.link("pushrod", "tongue", tip_pt, (xt, y_end, pz), (horn, k_stem), (0, 1, 0),
                      "steel, with ball links: a horn drives it")
        blade = _bump(c["tip_place"], c["tip_height"], P.K_TIP, 5.0, self.back, from_top=True).fuse(
            _liner(*self.liners["tip"])).cut(
            _cyl(1.5, 40, (xt, 0, R - lift + 20), UP))
        k_bl = self.part("tip blade", "tongue", blade, "tpu", "presses down from the palate; seals "
                         "the alveolar ridge for /t/", section=True)
        self.join(k_bl, k_stem, "joint")
        self.join(k_bl, self.cast)

    def pneumatic(self) -> None:
        c = self.c
        k_m = self.part("valve manifold", "tongue", _box((-112, -16, ZP), (-24, 16, ZP + 16), 1.0), "board",
                        "three proportional valves: inflate the chambers")
        self.join(k_m, self.base)
        # one cast tongue lines the floor; its three chambers sit where it bulges (``P.rig``)
        liner = _liner(*self.liners["body"])
        lo = max(self.liners["body"][0], -10 * self.r.controls["tongue_place"][1] - 6.0)
        body = _bump(c["tongue_place"], c["tongue_height"], P.K_BODY, R, self.back)
        tip = _bump(c["tip_place"], c["tip_height"], P.K_TIP, 6.0, self.back)
        k_t = self.part("hydrostat tongue", "tongue", liner.fuse(body, tip).clean(), "silicone",
                        "one cast with three chambers: the valves swell its body and tip into the bore", section=True)
        self.join(k_t, self.cast)
        for x in np.linspace(lo + 12, -16, 3):
            k_p = self.part("air port", "tongue", _rod((x, 0, -R), (x, 0, -RO - 4), 2.0), "metal",
                            "barbed port through the cast's floor into a chamber")
            self.join(k_p, self.cast, "joint")
            self.join(k_p, k_t, "joint")
            line = _rod((x, 0, -RO - 3), (float(np.clip(x, -106, -30)), 0, ZP + 15), 2.2)
            k_l = self.part("air line", "tongue", line, "rubber", "from the manifold to its chamber")
            self.join(k_l, k_p, "joint")
            self.join(k_l, k_m, "joint")

    # -------------------------------------------------- the lips
    def lips(self) -> None:
        li = self.body.parts["lips"]
        if li.type == "round_lips":
            return self.round_lips()
        tube = 10 * P.JAW[li.type][1]
        rest_area = li.params["rest_area"]
        rest = 2 * math.sqrt(rest_area / math.pi) * 10

        def drop_of(jaw: float) -> float:
            return 2 * math.sqrt((rest_area + P.JAW[li.type][0] * jaw) / math.pi) * 10 - rest

        drop = drop_of(self.c.get("jaw", 0.0))
        lip = _cyl(RO, tube, (tube / 2, 0, 0), (1, 0, 0))
        k_up = self.part("upper lip", "lips", lip.intersect(_box((0, -RO, rest / 2), (tube, RO, RO))), "silicone",
                         "cast, bonded to the face plate", section=True)
        self.join(k_up, self.face)
        lower = lip.intersect(_box((0, -RO, -RO), (tube, RO, -rest / 2))).translate(V(0, 0, -drop))
        k_lo = self.part("lower lip", "lips", lower, "silicone", "rides down with the jaw", section=True)
        self.join(k_lo, self.face)
        zj = -RO - drop
        jaw = _box((0.3, -9, zj - 5), (tube + 5, 9, zj), 1.0).fuse(_box((0.3, -4, zj - 30), (4.3, 4, zj - 4), 0.6))
        k_jaw = self.part("jaw", "lips", jaw, role="carries the lower lip down its guide", up=(0, -1, 0))
        self.join(k_lo, k_jaw, "joint")
        self.join(k_jaw, self.face)
        gt = -RO - drop_of(self.r.controls["jaw"][1]) - 6.0  # below the jaw at its widest
        guide = _box((0, -9, gt - 26), (8.5, 9, gt), 1.0).cut(_box((-1, -4.3, gt - 34), (4.6, 4.3, -RO)))
        k_g = self.part("jaw guide", "lips", guide, role="vertical guide for the jaw, screwed to the face plate",
                        up=(1, 0, 0))
        self.join(k_g, self.face)
        self.join(k_jaw, k_g, "guide")
        s = np.array([round((tube + 34) / LG.GRID) * LG.GRID, ZP + FOOT_DOWN])
        pin_rest = np.array([tube + 5, -RO - 2.5])
        a = 12.0
        b = float(np.linalg.norm(pin_rest - s))
        pin = pin_rest - np.array([0.0, drop])
        h = _reach(s, pin, a, b, -1.0)
        sv = self.servo("jaw servo", "lips", "pulls the jaw open through the jaw link", (s[0], ARM_Y, s[1]), (0, -1, 0),
                        (1, 0, 0), self.base, horn=(np.array([h[0] - s[0], 0, h[1] - s[1]]), a))
        self.link("jaw link", "lips", sv["tip"], (pin[0], 0.0, pin[1]), (sv["horn"], k_jaw), (0, 1, 0),
                  "from the jaw servo's horn to the jaw", r=1.2, eye=2.6)
        if li.type == "spread_lips":
            x_c = s[0] + 35
            sc = self.servo("corner servo", "lips", "pulls both mouth corners apart", (x_c, 0, ZP + FOOT_STAND), UP,
                            (1, 0, 0), self.base, horn=(np.array([0.0, 1.0, 0.0]), 10.0), lying=False)
            z_h = sc["top"] - HORN[2] / 2
            horn2 = _slot(HORN[0], 2.0, 10.0, HORN[2]).rotate(V(0, 0, 0), V(0, 0, 1), -90).translate(
                V(x_c, 0, sc["top"] - HORN[2]))
            k_h2 = self.part("horn", "lips", horn2, role="on the servo's spline: turns with the shaft")
            self.join(k_h2, sc["horn"], "joint")
            self.join(k_h2, sc["servo"], "joint")
            zc = rest / 2 + 2.0
            yc = math.sqrt(RO * RO - zc * zc) + 1.5  # just outside the lip, at the mouth's corner
            for side, horn in ((1.0, sc["horn"]), (-1.0, k_h2)):
                clip = _box((tube / 2 - 2, side * yc - 2, zc - 2), (tube / 2 + 2, side * yc + 2, zc + 2), 0.5)
                k_cl = self.part("corner clip", "lips", clip, role="clip bonded to a mouth corner", section=True)
                self.join(k_cl, k_up, "joint")
                cord = _rod((x_c, side * 10.0, z_h), (tube / 2 + 1.5, side * (yc + 1.5), zc - 1.5), 0.5)
                k_co = self.part("corner cord", "lips", cord, "rubber", "braided: the corner servo pulls it",
                                 section=True)
                self.join(k_co, k_cl, "joint")
                self.join(k_co, horn, "joint")

    def round_lips(self) -> None:
        c = self.c
        reach = 10.0 + 10 * c["protrusion"]
        ring = 10 * max(c["lip_radius"], 0.0) + 2.0  # the drawstring closes the ring to a pucker
        outer = cq.Solid.makeCone(RO, ring + 4, reach, V(0, 0, 0), V(1, 0, 0))
        inner = cq.Solid.makeCone(R, ring, reach + 0.01, V(-0.005, 0, 0), V(1, 0, 0))
        k_sl = self.part("lip sleeve", "lips", outer.cut(inner), "silicone", "stretches as the "
                         "ring is pushed forward", section=True)
        self.join(k_sl, self.face)
        k_ring = self.part("lip ring", "lips", cq.Solid.makeTorus(ring + 2, 2.5, V(reach, 0, 0), V(1, 0, 0)),
                           "silicone", "its drawstring closes it to a pucker", section=True)
        self.join(k_ring, k_sl, "joint")
        zc = -(ring + 2) - 2.5
        k_clip = self.part("ring clip", "lips", _box((reach - 3, -2.5, zc - 7), (reach + 3, 6.5, zc + 1.5), 0.6),
                           role="where the pushrod and the drawstring hold the ring")
        self.join(k_clip, k_ring, "joint")
        self.join(k_clip, k_sl, "joint")
        zs = ZP + FOOT_DOWN
        # protrusion: a pushrod of fixed length from the horn to the clip
        s = np.array([50.0, zs])
        prot_hi = 10 * self.r.controls["protrusion"][1]
        pin_mid = np.array([10.0 + prot_hi / 2, -(10 * P.RING / 2 + 2) - 2.5 - 3.0])
        a, b = 14.0, float(np.linalg.norm(pin_mid - s))
        pin = np.array([reach, zc - 3.0])
        h = _reach(s, pin, a, b, -1.0)
        sv = self.servo("protrusion servo", "lips", "pushes the lip ring forward", (s[0], ARM_Y, s[1]), (0, -1, 0),
                        (1, 0, 0), self.base, horn=(np.array([h[0] - s[0], 0, h[1] - s[1]]), a))
        self.link("pushrod", "lips", sv["tip"], (pin[0], 0.0, pin[1]), (sv["horn"], k_clip), (0, 1, 0),
                  "steel, with ball links: a horn drives it")
        # the drawstring: the horn's angle sets how far the string is drawn
        lo, hi = self.r.controls["lip_radius"]
        ang = math.radians(150 - 100 * (c["lip_radius"] - lo) / (hi - lo))
        s2 = np.array([90.0, zs])
        sv2 = self.servo("drawstring servo", "lips", "draws the ring's string: rounds and closes the lips",
                         (s2[0], ARM_Y + 4.0, s2[1]), (0, -1, 0), (1, 0, 0), self.base,
                         horn=(np.array([math.cos(ang), 0.0, math.sin(ang)]), 12.0))
        k = self.part("drawstring", "lips", _rod(sv2["tip"], (reach + 1.5, 4.0, zc - 6.0), 0.5), "rubber",
                      "braided cord around the lip ring, drawn by its servo")
        self.join(k, sv2["horn"], "joint")
        self.join(k, k_clip, "joint")

    # -------------------------------------------------- the whole
    def build(self) -> LG.Assembly:
        to, tr = self.body.parts["tongue"], self.body.parts["tract"]
        self.face_top = 48.0 if to.type == "tip_tongue" else RO + 10.0
        self.back = -10 * self.r.length + (6.0 if tr.type == "larynx_tube" else 0.0)
        self.slots: list[tuple[float, float, float]] = []
        floor, roof = -(R + RO) / 2, (R + RO) / 2
        if to.type != "pneumatic_tongue":  # a follower through the floor; a tip stem through the roof
            for place in ("tongue_place", "tip_place"):
                if place in self.r.controls:
                    lo, hi = (10 * v for v in self.r.controls[place])
                    tip = place == "tip_place"
                    self.slots.append((-hi - 3, (min(-lo, X_STEM) if tip else -lo) + 3, roof if tip else floor))
        # the membranes the articulators press into the bore, set into pockets in the cast
        cast = LG.WALL["silicone"] if to.type == "pneumatic_tongue" else LINER_MM
        self.liners = {"body": (self.back + 2.0, -4.5, cast, False)}
        if to.type == "tip_tongue":
            span = 10 * math.sqrt(1.0 / P.K_TIP)
            self.liners["tip"] = (max(self.back + 2.0, -10 * self.r.controls["tip_place"][1] - span), -4.5, LINER_MM, True)
        self.a.members.append(LG.Member(self.base, CadPart(self.base, "", trimesh.Trimesh())))
        self.tract()
        self.tongue()
        self.lips()
        lo = np.min([m.part.mesh.bounds[0][:2] for m in self.a.members[1:]] + [s[:2] for s in self.sweep], axis=0)
        hi = np.max([m.part.mesh.bounds[1][:2] for m in self.a.members[1:]] + [s[2:] for s in self.sweep], axis=0)
        g = LG.GRID
        x0, x1 = math.floor((lo[0] - 8) / g) * g, math.ceil((hi[0] + 8) / g) * g
        y0, y1 = math.floor((lo[1] - 8) / g) * g, math.ceil((hi[1] + 8) / g) * g
        plate = (cq.Workplane("XY").box(x1 - x0, y1 - y0, 6.0).translate(((x0 + x1) / 2, (y0 + y1) / 2, ZP - 3))
                 .edges("|Z").fillet(8.0).edges(">Z").fillet(1.5).val())
        self.a.members[0] = LG.Member(self.base, CadPart(self.base, "", to_mesh(plate)),
                                      "everything stands on it, screwed down", UP)
        return self.a


def assembly(body: Body, ctl: dict[str, float] | None = None) -> LG.Assembly:
    """The talking robot at a posture (``ctl``, System 1's controls; by default ``posture``)."""
    return _Rig(body, ctl).build()


def sweep(body: Body) -> list[dict]:
    """The rig at each end of each control's range, the others at ``posture``: where parts float, interfere or come
    loose.  A clean posture can hide parts that meet at the end of a travel."""
    out = []
    for c, (lo, hi) in P.rig(body).controls.items():
        for v in (lo, hi):
            r = LG.audit(assembly(body, posture(body) | {c: v}), forms=False)
            if r["floating"] or r["undeclared"] or r["loose"]:
                out.append({"control": c, "value": v} | {k: r[k] for k in ("floating", "undeclared", "loose")})
    return out


VIEWS = [View("section", (0.3, -1.0, 0.35)), View("side", (0.0, -1.0, 0.0))]


def phonology(body: Body, ctl: dict[str, float] | None = None) -> CadModel:
    """The body as a bench instrument at a posture, in section so the bore shows (``worlds/cad.CAD``)."""
    return assembly(body, ctl).model(VIEWS, (460, 320))
