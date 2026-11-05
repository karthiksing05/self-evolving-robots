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
from sea.design import pad_cad
from sea.design.cad import CadModel, CadPart, View, solid
from sea.evolve.core import Body
from sea.sim.build_model import build_model
from sea.worlds import graph_cad, rig_cad
from sea.worlds import gripper as G
from sea.worlds import mobile as M


# ---------------------------------------------------------------- the gripper
def gripper(body: Body, opening_mm: float = 80.0) -> CadModel:
    """The jaws as the simulator holds them, open to ``opening_mm`` (or as wide as the jaw goes): the
    SO-101's own CAD (its STL exports) for the servo, the fixed jaw and the moving finger, posed by the
    jaw part's joint, the jaw mechanism's links and pins as ``sim/jaws.py`` builds them, and the printed pads
    as the boxes ``design/fingers.py`` builds."""
    des = G.to_design(body)
    sim = build_model(des)
    m, d = sim.model, sim.data
    qa = HOME_Q.copy()
    qa[-1] = F.opening_for(des.fingers, opening_mm / 1000) or F.q_range(des.fingers)[1]
    sim.set_arm_q(qa)
    g = m.body("gripper").id
    rot, origin = d.xmat[g].reshape(3, 3), d.xpos[g]
    moving = {"moving_jaw_so101_v1", "jaw_coupler", "jaw_follower", "jaw_carriage"}
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
        elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=m.geom_size[i][0], height=2 * m.geom_size[i][1], sections=48)
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
        slot = "pads" if pad else ("jaw" if name in moving or label.startswith("jaw_") else "")
        steel = label.endswith("_pin") or label[:-1] == "jaw_rod"
        material = "tpu" if pad else "servo" if servo else "metal" if steel else "printed"
        parts.append(CadPart(label, slot, mesh, material))
    parts = pad_cad.dress(parts, des.fingers)  # the boxes become the pads as built, lands on the boxes' faces
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
            if mecanum:  # the rollers the simulator rolls on (sim/mobile_model.py, D-070), front-left and rear-right
                from sea.sim.mobile_model import (  # leaning one way, the other two the other
                    ROLLER_R,
                    roller_axes,
                )

                parts.append(solid("hub", "locomotion", cyl(0.62 * r, side, (x, y, z), (0, 1, 0))))
                lean = 1.0 if (x > 0) == (y > 0) else -1.0
                for c, ax in roller_axes(r, lean):
                    parts.append(solid("roller", "locomotion", cyl(ROLLER_R * r, 0.9 * side, tuple(np.array([x, y, z]) + c),
                                                                    tuple(ax)), "rubber"))
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
posture = rig_cad.posture


def phonology(body: Body, ctl: dict[str, float] | None = None) -> CadModel:
    """The body as a bench instrument at a posture (``ctl``, System 1's controls; by default ``posture``): every
    servo in its mount with the horn and linkage it drives, the cast tract in section so the bore shows (``rig_cad``)."""
    return rig_cad.phonology(body, ctl)


CAD: dict[str, Callable[..., CadModel]] = {"gripper": gripper, "gripper_dense": gripper, "mobile": mobile,
                                              "phonology": phonology, "mobile_kit": graph_cad.graph,
                                              "phonology_kit": graph_cad.graph}
