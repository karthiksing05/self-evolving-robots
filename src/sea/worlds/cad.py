"""CAD of each world's bodies, for pictures of a run (scripts/render_bodies.py, docs/evolve.md).

Kept out of the world modules on purpose: those are judge components whose source the harness
fingerprints (sea/harness.py), and a picture is not part of the ruler.  Each function reads the body
through its world's own model (the gripper's MuJoCo build, the mobile ``_spec``, the phonology ``rig``),
so what is drawn is what is simulated.  ``CAD`` maps a world's name to its function.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cadquery as cq
import mujoco
import numpy as np
import trimesh

from sea.control.executor import HOME_Q
from sea.design import fingers as F
from sea.design.cad import CadModel, CadPart, View, solid
from sea.evolve.core import Body
from sea.sim.build_model import build_model
from sea.worlds import gripper as G
from sea.worlds import mobile as M
from sea.worlds import phonology as P


# ---------------------------------------------------------------- the gripper
def gripper(body: Body, opening_mm: float = 80.0) -> CadModel:
    """The jaws as the simulator holds them, open to ``opening_mm`` (or as wide as the jaw goes): the
    SO-101's own CAD (its STL exports) for the servo, the fixed jaw and the moving finger, posed by the
    jaw part's joint, and the printed pads as the boxes ``design/fingers.py`` builds.  The slide's rods and
    bushings are bought items the simulator does not draw; the bill of materials lists them."""
    des = G.to_design(body)
    sim = build_model(des)
    m, d = sim.model, sim.data
    qa = HOME_Q.copy()
    qa[-1] = F.opening_for(des.fingers, opening_mm / 1000) or F.q_range(des.fingers)[1]
    sim.set_arm_q(qa)
    g = m.body("gripper").id
    rot, origin = d.xmat[g].reshape(3, 3), d.xpos[g]
    moving = {"moving_jaw_so101_v1", "jaw_coupler"}
    parts = []
    for i in range(m.ngeom):
        name = m.body(m.geom_bodyid[i]).name
        if m.geom_group[i] != 2 or name not in moving | {"gripper"}:
            continue
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[i]
            v0, nv, f0, nf = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
            mesh = trimesh.Trimesh(m.mesh_vert[v0:v0 + nv].copy(), m.mesh_face[f0:f0 + nf].copy(), process=False)
            label = m.mesh(k).name
        elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * m.geom_size[i])
            label = m.geom(i).name
        else:
            continue
        pose = np.eye(4)
        pose[:3, :3] = rot.T @ d.geom_xmat[i].reshape(3, 3)
        pose[:3, 3] = rot.T @ (d.geom_xpos[i] - origin)
        mesh.apply_transform(pose)
        mesh.apply_scale(1000.0)
        pad = "_pad" in label
        servo = label.startswith("sts3215")
        slot = "pads" if pad else ("jaw" if name in moving else "")
        parts.append(CadPart(label, slot, mesh, "tpu" if pad else ("servo" if servo else "printed")))
    # the gripper frame: x closes the jaws, the fingertips point down -z
    return CadModel(parts, [View("side", (0.0, -1.0, 0.0)), View("three-quarter", (0.55, -1.0, -0.45))])


# ---------------------------------------------------------------- the mobile explorer
CELL_MM = {"nimh_pack": (50.0, 14.5), "liion_pack": (65.0, 18.5)}  # cell length and diameter
MOTOR_MM = {"n20_gearmotor": (34.0, 12.0), "n20_encoder_gearmotor": (40.0, 12.0), "jga25_gearmotor": (60.0, 25.0)}
SPROCKET, IDLER = 18.0, 15.0  # mm: the tracks' drive sprocket radius, and how far the track stands above the idler axle


def _hull(circles: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Convex hull of circles (x, z, r): the outline of a belt around its wheels."""
    pts = sorted({(round(x + r * math.cos(a), 3), round(z + r * math.sin(a), 3))
                  for x, z, r in circles for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)})

    def half(ps: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in ps:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    return half(pts)[:-1] + half(pts[::-1])[:-1]


def mobile(body: Body) -> CadModel:
    """The body as solids, in millimetres (x forward, y left, z up, the ground at z = 0).  The geometry is
    the simulator's: the world's ``_spec`` gives the overall width, height and belly clearance; the catalogue's docs give
    the rest (wheel, sprocket and idler sizes; cells and motors at their datasheet sizes)."""
    V = cq.Vector
    s, ch = M._spec(body), body.parts["chassis"].params
    lo, dr, bt, se = (body.parts[k] for k in ("locomotion", "drive", "battery", "sensor"))
    w, ln, side = 10 * ch["width"], 10 * ch["length"], 5 * (s.width - ch["width"])
    belly, tub_h, p = 10 * s.belly, 10 * M.TUB_H, lo.params
    parts = []

    def cyl(r: float, length: float, at: tuple[float, float, float], axis: tuple[float, float, float]):
        a = V(*axis).normalized()
        return cq.Solid.makeCylinder(r, length, V(*at) - a * (length / 2), a)

    def wheel(x: float, z: float, r: float, mecanum: bool = False) -> None:
        for y in (w / 2 + side / 2, -w / 2 - side / 2):
            if mecanum:
                parts.append(solid("hub", "locomotion", cyl(0.62 * r, side, (x, y, z), (0, 1, 0))))
                for th in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                    c = (x + 0.82 * r * math.cos(th), y, z + 0.82 * r * math.sin(th))
                    ax = (-math.sin(th) * 0.707, 0.707, math.cos(th) * 0.707)
                    parts.append(solid("roller", "locomotion", cyl(0.17 * r, 0.9 * side, c, ax), "rubber"))
            else:
                tyre = cyl(r, side, (x, y, z), (0, 1, 0)).cut(cyl(0.62 * r, side * 1.1, (x, y, z), (0, 1, 0)))
                parts.append(solid("tyre", "locomotion", tyre, "rubber"))
                parts.append(solid("hub", "locomotion", cyl(0.62 * r, 0.8 * side, (x, y, z), (0, 1, 0))))

    if lo.type == "tracks":
        tl, ih = 10 * p["track_length"], 10 * p["idler_height"]
        rear, front = SPROCKET, SPROCKET + tl
        idler = (front + 0.8 * (ih - SPROCKET), ih, IDLER)
        outer = _hull([(rear, SPROCKET, SPROCKET), (front, SPROCKET, SPROCKET), idler])
        inner = _hull([(rear, SPROCKET, SPROCKET - 5), (front, SPROCKET, SPROCKET - 5), idler[:2] + (IDLER - 5,)])
        for y in (w / 2 + side, -w / 2):  # XZ workplanes extrude toward -y
            belt = (cq.Workplane("XZ").polyline(outer).close().extrude(side)
                    .cut(cq.Workplane("XZ").polyline(inner).close().extrude(side)).translate((0, y, 0)))
            parts.append(solid("track", "locomotion", belt, "rubber"))
            for x, z, r in ((rear, SPROCKET, SPROCKET - 6), (front, SPROCKET, SPROCKET - 7), (idler[0], ih, IDLER - 6)):
                parts.append(solid("wheel", "locomotion", cyl(r, 0.7 * side, (x, y - side / 2, z), (0, 1, 0))))
        tx, axles = (rear + front) / 2 - ln / 2, [rear]
    elif lo.type == "rocker_bogie6":
        r, bl = 10 * p["wheel_radius"], 10 * p["bogie_length"]
        xs = [r, r + bl, r + 2 * bl]
        piv_b = ((xs[0] + xs[1]) / 2, r + 20.0)
        piv_r = ((piv_b[0] + xs[2]) / 2, r + 35.0)
        for x in xs:
            wheel(x, r, r)
        for y in (w / 2 + side / 2, -w / 2 - side / 2):
            for (x1, z1), (x2, z2) in (((xs[0], r), piv_b), ((xs[1], r), piv_b), (piv_b, piv_r), (piv_r, (xs[2], r))):
                d = np.array([x2 - x1, 0.0, z2 - z1])
                parts.append(solid("link", "locomotion", cyl(4.0, float(np.linalg.norm(d)) + 8, ((x1 + x2) / 2, y, (z1 + z2) / 2),
                                                             tuple(d))))
            for x, z in (piv_b, piv_r):
                parts.append(solid("pivot", "locomotion", cyl(7.0, side + 6, (x, y, z), (0, 1, 0)), "metal"))
        tx, axles = piv_r[0] - ln / 2, xs
    else:
        r = 10 * p["wheel_radius"]
        tx = max(0.0, r - 10.0)
        axles = [tx + 10.0, tx + ln - 10.0]
        for x in axles:
            wheel(x, r, r, lo.type == "mecanum4")
    tub = (cq.Workplane("XY").box(ln, w, tub_h).edges("|Z").fillet(min(10.0, w / 4)).faces(">Z").shell(-2.0)
           .translate((tx + ln / 2, 0, belly + tub_h / 2)))
    parts.append(solid("tub", "chassis", tub))
    front, top, floor = tx + ln, belly + tub_h, belly + 2.0

    ml, md = MOTOR_MM[dr.type]
    for x in axles:
        for y in ((w / 2 - 3 - ml / 2, -w / 2 + 3 + ml / 2) if lo.type != "rocker_bogie6" else (w / 2 + side + 4 + ml / 2,
                                                                                                   -w / 2 - side - 4 - ml / 2)):
            z = floor + md / 2 if lo.type != "rocker_bogie6" else 10 * p["wheel_radius"]
            parts.append(solid("motor", "drive", cyl(md / 2, ml, (x, y, z), (0, 1, 0)), "metal"))
    cl, cd = CELL_MM[bt.type]
    n = M._cells(bt)
    rows = max(1, int((w - 2 * ml - 10) // (cd + 1)))
    cols = math.ceil(n / rows)
    x0, y0 = tx + ln / 2 - cols * (cl + 2) / 2 + cl / 2, -(min(n, rows) - 1) * (cd + 1) / 2
    for i in range(n):
        at = (x0 + (i // rows) * (cl + 2), y0 + (i % rows) * (cd + 1), floor + cd / 2)
        parts.append(solid("cell", "battery", cyl(cd / 2, cl, at, (1, 0, 0)), "cell"))

    zc = belly + tub_h / 2
    if se.type == "bump_switch":
        parts.append(solid("bumper", "sensor", cq.Workplane("XY").box(5, w, tub_h - 8).translate((front + 9.5, 0, zc))))
        for y in (w / 4, -w / 4):
            parts.append(solid("switch", "sensor", cq.Workplane("XY").box(7, 6, 10).translate((front + 3.5, y, zc)), "board"))
    elif se.type == "ultrasonic":
        parts.append(solid("sonar board", "sensor", cq.Workplane("XY").box(1.6, 45, 20).translate((front + 0.8, 0, zc)),
                           "board"))
        for y in (13.0, -13.0):
            parts.append(solid("transducer", "sensor", cyl(8.0, 12.0, (front + 7.6, y, zc), (1, 0, 0)), "metal"))
    elif se.type == "camera":
        parts.append(solid("mast", "sensor", cyl(3.0, 25.0, (front - 20, 0, top + 12.5), (0, 0, 1))))
        parts.append(solid("camera", "sensor", cq.Workplane("XY").box(12, 18, 18).translate((front - 20, 0, top + 34)), "board"))
        parts.append(solid("lens", "sensor", cyl(4.0, 6.0, (front - 11, 0, top + 34), (1, 0, 0)), "servo"))
    else:
        parts.append(solid("lidar base", "sensor", cyl(20.0, 12.0, (tx + ln / 2, 0, top + 6), (0, 0, 1)), "servo"))
        parts.append(solid("lidar head", "sensor", cyl(17.0, 28.0, (tx + ln / 2, 0, top + 26), (0, 0, 1)), "rubber"))
    return CadModel(parts, [View("three-quarter", (1.0, -1.25, 0.85)), View("side", (0.0, -1.0, 0.0))])


# ---------------------------------------------------------------- the talking robot
SERVO_MM = (23.0, 12.5, 29.0)  # an MG90S, mounting tabs left off


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


def phonology(body: Body, ctl: dict[str, float] | None = None) -> CadModel:
    """The body as solids, in millimetres, at a posture (``ctl``, System 1's controls; by default ``posture``).
    The tract runs along x from the glottis to the teeth (x = 0), the lips in front, on a printed base plate:
    the reed housing holds the glottis end, a face plate at the teeth holds the front, and every servo stands
    on the plate, rides a carriage or sits on the face plate's shelf.  The silicone tract, the lips and the
    face plate are cut on their mid-plane (a section view) so the bore shows; the mechanisms are whole, one
    MG90S block per servo the rig counts.  The squeezed bore follows ``tract``'s profile: the gap left at each
    point is the diameter of the model's radius there."""
    V = cq.Vector
    r, tr, to, li = P.rig(body), body.parts["tract"], body.parts["tongue"], body.parts["lips"]
    c = posture(body) | (ctl or {})
    R, RO = 10 * P.BORE, 10 * P.BORE + 4.0
    L = 10 * (r.length + c.get("larynx", 0.0))
    zp = -RO - 34.0  # top of the base plate
    cut = cq.Solid.makeBox(2000, 1000, 2000, V(-1000, -1000, -1000))  # the half toward the camera
    parts = []

    def cyl(rad: float, length: float, at: tuple[float, float, float], axis: tuple[float, float, float]):
        a = V(*axis).normalized()
        return cq.Solid.makeCylinder(rad, length, V(*at) - a * (length / 2), a)

    def rod(p0: tuple[float, float, float], p1: tuple[float, float, float], rad: float = 1.2):
        d = np.subtract(p1, p0)
        return cyl(rad, float(np.linalg.norm(d)), tuple(np.add(p0, p1) / 2), tuple(d))

    def block(lo: tuple[float, float, float], hi: tuple[float, float, float]):
        return cq.Solid.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], V(*lo))

    def servo(x: float, y: float, z0: float, name: str, slot: str, lying: bool = False) -> None:
        """An MG90S standing on z0 (or lying on its side, shaft along y)."""
        a, b, h = SERVO_MM
        dy, dz = (h, b) if lying else (b, h)
        parts.append(solid(name, slot, block((x - a / 2, y - dy / 2, z0), (x + a / 2, y + dy / 2, z0 + dz)), "servo"))

    def bump(place: float, height: float, k: float, half_w: float):
        """Silicone pressed into the bore from below by an articulator at ``place`` cm behind the teeth."""
        r0 = max(0.0, P.BORE - height)
        span = math.sqrt(1.0 / k)
        d = np.linspace(place - span, place + span, 41)
        rad = np.minimum(P.BORE, r0 + k * (P.BORE - r0) * (d - place) ** 2)
        top = np.clip(10 * (-P.BORE + 2 * (P.BORE - rad)), -R + 0.5, R - 0.01)  # at rest a pad still touches the bore
        pts = [(-10 * a, z) for a, z in zip(d, top, strict=True)] + [(-10 * d[-1], -R - 1), (-10 * d[0], -R - 1)]
        prof = cq.Workplane("XZ").polyline(pts).close().extrude(2 * half_w).translate((0, half_w, 0)).val()
        return prof.intersect(cyl(R, L + 2, (-L / 2, 0, 0), (1, 0, 0))).cut(cut)

    # tract: the cast tube between the reed housing and the face plate; a larynx bellows lengthens it
    base = 10 * r.length
    tube = cyl(RO, base, (-base / 2, 0, 0), (1, 0, 0)).cut(cyl(R, base + 2, (-base / 2, 0, 0), (1, 0, 0))).cut(cut)
    parts.append(solid("cast tract", "tract", tube, "silicone"))
    back = -base
    if tr.type == "larynx_tube":
        ext = L - base  # the bellows' stretch: the larynx control
        n = max(2, int((ext + 6) // 3))
        bellows = cyl(RO, 3.0, (back - 1.5, 0, 0), (1, 0, 0))
        for i in range(1, n):
            bellows = bellows.fuse(cyl(RO if i % 2 == 0 else R + 1.5, (ext + 6) / n, (back - (i + 0.5) * (ext + 6) / n, 0, 0),
                                       (1, 0, 0)))
        bellows = bellows.cut(cyl(R, ext + 10, (back - (ext + 6) / 2, 0, 0), (1, 0, 0))).cut(cut)
        parts.append(solid("larynx bellows", "tract", bellows, "silicone"))
        back -= ext + 6
        parts.append(solid("larynx slide", "tract", block((back - 40, -10, zp), (back + 10, 10, zp + 6)), "metal"))
        servo(back - 56, 0, zp, "larynx servo", "tract")
    parts.append(solid("reed housing", "", block((back - 16, -RO - 2, zp + (6 if tr.type == "larynx_tube" else 0)),
                                                 (back, RO + 2, RO + 2))))
    parts.append(solid("air hose", "", cyl(4.0, 30.0, (back - 31, 0, 0), (1, 0, 0)), "rubber"))
    face = block((-4, -RO - 8, zp), (0, RO + 8, RO + 8)).cut(cyl(R, 10, (-2, 0, 0), (1, 0, 0))).cut(cut)
    parts.append(solid("face plate", "", face))

    # tongue: under the tube; the rail on the plate, the carriage on the rail, the cam servo on the carriage
    if to.type == "pneumatic_tongue":
        parts.append(solid("body chamber", "tongue", bump(c["tongue_place"], c["tongue_height"], P.K_BODY, R), "silicone"))
        parts.append(solid("tip chamber", "tongue", bump(c["tip_place"], c["tip_height"], P.K_TIP, 6.0), "silicone"))
        for place in (c["tongue_place"], c["tip_place"]):
            parts.append(solid("air line", "tongue", rod((-10 * place, 0, -R), (-10 * place, 0, zp + 14), 2.0), "rubber"))
        parts.append(solid("valve manifold", "tongue", block((-110, -15, zp), (-20, 15, zp + 14)), "board"))
    else:
        travel = 10.0 if to.type == "long_rail_tongue" else 7.0
        start = to.params["rail_start"]
        xp = -10 * c["tongue_place"]
        parts.append(solid("tongue pad", "tongue", bump(c["tongue_place"], c["tongue_height"], P.K_BODY, R), "tpu"))
        parts.append(solid("rail", "tongue", block((-10 * (start + travel) - 12, -6, zp), (-10 * start + 12, 6, zp + 6)),
                           "metal"))
        parts.append(solid("carriage", "tongue", block((xp - 14, -11, zp + 6), (xp + 14, 36, zp + 12))))
        parts.append(solid("cam", "tongue", cyl(7.0, 14.0, (xp, 0, -RO - 15), (0, 1, 0)), "metal"))
        parts.append(solid("cam post", "tongue", block((xp - 3, -3, zp + 12), (xp + 3, 3, -RO - 15))))
        parts.append(solid("pad head", "tongue", block((xp - 15, -9, -RO - 8), (xp + 15, 9, -RO))))
        servo(xp, 21.5, zp + 12, "cam servo", "tongue", lying=True)
        servo(-10 * (start + travel) - 26, 0, zp, "rail servo", "tongue")
        if to.type == "tip_tongue":
            xt = -10 * c["tip_place"]
            parts.append(solid("tip blade", "tongue", bump(c["tip_place"], c["tip_height"], P.K_TIP, 5.0), "tpu"))
            for y, (sx, sy) in ((-4.0, (-18.0, 0.0)), (4.0, (-18.0, 14.0))):
                servo(sx, sy, zp, "tip servo", "tongue")
                parts.append(solid("pushrod", "tongue", rod((sx + 6, sy, zp + 29), (xt, y, -R + 1)), "metal"))

    # lips: in front of the face plate
    shelf_z = RO + 8
    if li.type == "round_lips":
        reach = 10.0 + 10 * c["protrusion"]
        ring = 10 * max(c["lip_radius"], 0.0) + 2.0  # the drawstring closes the ring to a pucker
        outer = cq.Solid.makeCone(RO, ring + 4, reach, V(0, 0, 0), V(1, 0, 0))
        inner = cq.Solid.makeCone(R, ring, reach + 0.01, V(-0.005, 0, 0), V(1, 0, 0))
        parts.append(solid("lip sleeve", "lips", outer.cut(inner).cut(cut), "silicone"))
        parts.append(solid("lip ring", "lips", cq.Solid.makeTorus(ring + 2, 2.5, V(reach, 0, 0), V(1, 0, 0)).cut(cut), "silicone"))
        parts.append(solid("servo shelf", "lips", block((-64, 0, shelf_z), (-2, 20, shelf_z + 4))))
        servo(-50, 10, shelf_z + 4, "protrusion servo", "lips", lying=True)
        servo(-22, 10, shelf_z + 4, "drawstring servo", "lips", lying=True)
        parts.append(solid("pushrod", "lips", rod((-10, 4, shelf_z + 10), (reach, 4, ring + 4.5)), "metal"))
    else:
        tube_mm = 10 * P.JAW[li.type][1]
        span = P.JAW[li.type][0]
        rest = 2 * math.sqrt(li.params["rest_area"] / math.pi) * 10
        drop = 2 * math.sqrt((li.params["rest_area"] + span * c.get("jaw", 0.0)) / math.pi) * 10 - rest
        lip = cyl(RO, tube_mm, (tube_mm / 2, 0, 0), (1, 0, 0))
        parts.append(solid("upper lip", "lips", lip.intersect(block((0, -RO, rest / 2), (tube_mm, RO, RO))).cut(cut), "silicone"))
        lower = lip.intersect(block((0, -RO, -RO), (tube_mm, RO, -rest / 2))).translate(V(0, 0, -drop))
        parts.append(solid("lower lip", "lips", lower.cut(cut), "silicone"))
        parts.append(solid("jaw", "lips", block((0, -8, -RO - 5 - drop), (tube_mm + 4, 8, -RO - drop))))
        servo(tube_mm + 20, 0, zp, "jaw servo", "lips")
        parts.append(solid("jaw link", "lips", rod((tube_mm + 2, 0, -RO - 5 - drop), (tube_mm + 14, 0, zp + 29), 2.0)))
        if li.type == "spread_lips":
            parts.append(solid("servo shelf", "lips", block((-40, 0, shelf_z), (-2, 20, shelf_z + 4))))
            servo(-22, 10, shelf_z + 4, "corner servo", "lips", lying=True)
            parts.append(solid("corner pull", "lips", rod((-10, 4, shelf_z + 10), (tube_mm, 4, RO - 2)), "metal"))

    xs = np.concatenate([p.mesh.bounds[:, 0] for p in parts])
    parts.append(solid("base plate", "", block((xs.min() - 8, -24, zp - 6), (xs.max() + 8, 42, zp))))
    return CadModel(parts, [View("section", (0.3, -1.0, 0.35)), View("side", (0.0, -1.0, 0.0))], tile=(420, 300))


CAD: dict[str, Callable[..., CadModel]] = {"gripper": gripper, "gripper_dense": gripper, "mobile": mobile,
                                              "phonology": phonology}
