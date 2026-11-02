"""The design language (docs/design-language.md): pads drawn on the simulated faces, rigs nothing floats in."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from PIL import Image

from sea.contracts import FingerSpec
from sea.control.frames import quat_wxyz_to_mat
from sea.design import fingers as F
from sea.design import language as LG
from sea.design import pad_cad
from sea.design.build import _ray_hits
from sea.design.cad import CadPart
from sea.design.callouts import labelled
from sea.evolve.core import Body, Edit, apply
from sea.mechanism.geometry import winding
from sea.worlds import cad, rig_cad
from sea.worlds import gripper as G
from sea.worlds.phonology import PhonologyWorld

PADS = {
    "flat_tpu": dict(reach_mm=10.0, pad_angle_deg=0.0, groove_mm=0.0, lip_mm=0.0, pad_tpu_mm=2.0, width_mm=12.0),
    "grooved_tilted": dict(reach_mm=10.0, pad_angle_deg=40.0, groove_mm=1.5, lip_mm=0.0, pad_tpu_mm=1.0, width_mm=20.0),
    "hooked": dict(reach_mm=20.0, pad_angle_deg=20.0, groove_mm=1.5, lip_mm=3.0, pad_tpu_mm=1.5, width_mm=16.0),
    "bare_lip": dict(reach_mm=5.0, pad_angle_deg=0.0, groove_mm=0.0, lip_mm=2.0, pad_tpu_mm=0.0, width_mm=12.0),
}
JAWS = {"hooked": "parallelogram", "bare_lip": "prismatic"}
TOL = 0.1  # mm: a drawn land and its simulated face agree to this


def _gripper(name: str) -> Body:
    w = G.GripperWorld()
    cat = {p.name: p for p in w.catalogue()}
    edits = [Edit(slot="pads", type="pads", params=PADS[name])]
    if name in JAWS:
        edits.append(Edit(slot="jaw", type=JAWS[name], params={k: (p.lo + p.hi) / 2 for k, p in cat[JAWS[name]].params.items()}))
    return apply(w.stock(), edits, cat, "t")


def _faces(spec: FingerSpec, fr: pad_cad.PadFrame, t: float):
    """The simulated faces an object can touch, in the pad frame (mm): each face box's front, and the lip's front and
    its top in front of the land, as (centre, normal, (axis, half), (axis, half), which samples count).  A sample
    another box covers (the lip over a face's foot) is no contact face."""
    bx = pad_cad.boxes(spec, fr)

    def bare(i: int, n: np.ndarray):
        def keep(p: np.ndarray) -> np.ndarray:
            q = p + 0.05 * n
            covered = [np.all(np.abs((q - c) @ r) <= h, axis=1) for j, (c, h, r) in enumerate(bx) if j != i]
            return ~np.any(covered, axis=0) if covered else np.ones(len(p), bool)

        return keep

    out = []
    for i, (c, h, r) in enumerate(bx):
        out.append((c + r[:, 0] * h[0], r[:, 0], (r[:, 1], h[1]), (r[:, 2], h[2]), bare(i, r[:, 0])))
        if spec.lip_mm > 0 and i == len(bx) - 1:
            out.append((c + r[:, 1] * h[1], r[:, 1], (r[:, 0], h[0]), (r[:, 2], h[2]), lambda p: p[:, 0] > t + TOL))
    return out


@pytest.mark.parametrize("name", PADS)
def test_pad_frames_place_the_simulated_boxes(name):
    """``pad_cad`` draws on the same boxes ``fingers.pad_geoms`` hands the simulator."""
    spec = G.to_design(_gripper(name)).fingers
    geoms = F.pad_geoms(spec)
    mine = [(fr, b) for fr in pad_cad.frames(spec) for b in pad_cad.boxes(spec, fr)]
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
    assert len(mine) == len(geoms)
    for g, (fr, (c, h, r)) in zip(geoms, mine, strict=True):
        sim = 1000 * (np.asarray(g.pos) + (signs * np.asarray(g.half)) @ quat_wxyz_to_mat(g.quat).T)
        assert g.body == fr.body and np.allclose(sim, fr.origin + (c + (signs * h) @ r.T) @ fr.rot.T, atol=1e-6)


@pytest.mark.parametrize("name", PADS)
def test_pad_lands_are_the_simulated_contact_faces(name, monkeypatch):
    """Every face an object can touch is drawn where the simulator has it: coplanar within 0.1 mm over the face, no
    coplanar land past its edges by more than 0.1 mm, and the pad's lowest point on the boxes' bottom plane.  Only
    the part of a face inside the jaw (its socket, grown by the slip fit) or within 0.15 mm of it is not drawn."""
    body = _gripper(name)
    spec = G.to_design(body).fingers
    with monkeypatch.context() as m:
        m.setattr(pad_cad, "dress", lambda parts, spec: parts)
        fit = pad_cad.jaws(cad.gripper(body).parts, spec)
    t = 1000 * F.thickness(spec)
    for fr in pad_cad.frames(spec):
        host = fit[fr.body][1].copy().apply_transform(np.linalg.inv(pad_cad._pose(fr.rot, fr.origin)))
        drawn = [p.mesh for p in pad_cad.pad(spec, fr, host) if p.material != "metal"]
        sock = LG.tessellate(pad_cad.socket(host, fr.length + 1000 * F.COVER + 3.0))
        assert abs(min(float(d.vertices[:, 1].min()) for d in drawn)) < TOL  # the boxes' bottom is y = 0
        for centre, n, (a1, h1), (a2, h2), keep in _faces(spec, fr, t):
            u, v = np.meshgrid(np.linspace(-h1 + TOL, h1 - TOL, max(3, int(4 * h1))),
                               np.linspace(-h2 + TOL, h2 - TOL, max(3, int(4 * h2))))
            p = centre + u.reshape(-1, 1) * a1 + v.reshape(-1, 1) * a2
            p = p[keep(p)]
            path = np.vstack([p + s * n for s in (-0.15, 0.0, 0.05, 0.5, 1.0, 2.0, 3.0)])
            clear = (winding(sock, path) < 0.5).reshape(7, -1).all(axis=0)  # the face and the way to it lie outside the jaw
            assert clear.mean() > 0.25, (fr.kind, centre)
            p = p[clear]
            hit = np.min([_ray_hits(p + 3.0 * n, np.tile(-n, (len(p), 1)), d.triangles) for d in drawn], axis=0)
            assert np.all(np.abs(hit - 3.0) < TOL), (fr.kind, centre, float(np.max(np.abs(hit - 3.0))))
            for d in drawn:  # the exposed land in the face's plane (not where the TPU skin meets the shoe)
                flat = (d.face_normals @ n > np.cos(np.radians(1.0))) & np.all(
                    np.abs((d.triangles - centre) @ n) < 0.02, axis=1)
                out = d.triangles_center[flat] + 0.05 * n
                exposed = np.all([winding(o, out) < 0.5 for o in drawn], axis=0) if flat.any() else flat[flat]
                q = d.triangles[flat][exposed].reshape(-1, 3) - centre
                assert np.all(np.abs(q @ a1) <= h1 + TOL) and np.all(np.abs(q @ a2) <= h2 + TOL), (fr.kind, centre)


@pytest.mark.parametrize("name", ["flat_tpu", "hooked"])
def test_drawn_pads_are_screwed_on_and_nothing_floats(name):
    for asm in pad_cad.assemblies(cad.gripper(_gripper(name)).parts):
        r = LG.audit(asm, forms=False)
        assert (r["floating"], r["undeclared"], r["loose"]) == ([], [], []), (asm.base, r)
        assert {m.part.material for m in asm.members} >= {"printed", "metal"}


def test_pads_that_cannot_be_built_are_refused():
    ok = FingerSpec(**PADS["flat_tpu"])
    assert pad_cad.refusals(ok) == [] and pad_cad.refusals(FingerSpec()) == []
    assert any("groove" in why for why in pad_cad.refusals(ok.model_copy(update={"groove_mm": 3.0})))
    assert any("TPU" in why for why in pad_cad.refusals(ok.model_copy(update={"pad_tpu_mm": 0.5})))


# ---------------------------------------------------------------- the talking robot's rig
PHON = PhonologyWorld()


def _rich() -> Body:
    cat = {p.name: p for p in PHON.catalogue()}
    return apply(PHON.stock(), [Edit(slot=s, type=t, params={k: (p.lo + p.hi) / 2 for k, p in cat[t].params.items()})
                                for s, t in (("tongue", "tip_tongue"), ("lips", "round_lips"), ("tract", "larynx_tube"))],
                 cat, "t")


@pytest.mark.parametrize("which", ["stock", "rich"])
def test_rig_is_grounded_with_no_undeclared_interference(which):
    asm = rig_cad.assembly(PHON.stock() if which == "stock" else _rich())
    r = LG.audit(asm, forms=False)
    assert (r["floating"], r["undeclared"], r["loose"]) == ([], [], [])
    names = {m.part.name for m in asm.members}
    assert {"base plate", "cast tract", "face plate", "reed housing", "servo mount", "horn", "screws"} <= names
    assert all(m.role for m in asm.members)  # every part says what it does


def test_a_control_moves_its_mechanism_and_nothing_fixed():
    body = PHON.stock()
    base = rig_cad.posture(body)

    def at(**ctl) -> dict[str, tuple[str, trimesh.Trimesh]]:
        return {m.key: (m.part.name, m.part.mesh) for m in rig_cad.assembly(body, base | ctl).members}

    ref = at()
    fixed = {"base plate", "cast tract", "face plate", "rail", "rail servo", "reed housing", "upper lip", "jaw guide",
             "jaw servo"}
    for ctl, moves in (({"tongue_height": 0.2}, {"cam", "follower rod", "tongue pad"}),
                       ({"tongue_place": 4.0}, {"sled", "carriage", "cam servo", "tongue pad", "crank", "connecting rod"}),
                       ({"jaw": 0.0}, {"jaw", "lower lip", "jaw link"})):
        now = at(**ctl)
        assert set(now) == set(ref)
        moved = {ref[k][0] for k in ref if not np.allclose(ref[k][1].vertices.mean(0), now[k][1].vertices.mean(0), atol=1e-3)}
        assert moves <= moved and not moved & fixed, (ctl, moved)
    rod = next(m for m in rig_cad.assembly(body, base | {"tongue_place": 4.0}).members if m.part.name == "follower rod")
    assert rod.part.mesh.bounds[:, 0].mean() == pytest.approx(-40.0, abs=0.01)  # the follower stands at the place


def test_labelled_picture_numbers_every_component():
    asm = rig_cad.assembly(PHON.stock())
    png = labelled(asm.model(rig_cad.VIEWS, (460, 320)), {m.part.name: m.role for m in asm.members}, "stock",
                   size=(700, 350))
    im = Image.open(io.BytesIO(png))
    assert im.width == 700 + 2 * 48 and im.height > 350 + 2 * 48


def test_parts_may_touch_but_not_share_volume():
    """Interference is volume shared more than 0.5 mm deep: a block resting on the plate is a contact, a rod through
    it is not, and a declared joint is exempt."""
    def asm(*parts, joints=()):
        out = LG.Assembly(base="plate")
        for name, mesh in (("plate", trimesh.creation.box(extents=(40, 40, 4))), *parts):
            out.add(LG.Member(name, CadPart(name, "", mesh)))
        out.joints += list(joints)
        return out

    block = trimesh.creation.box(extents=(10, 10, 10)).apply_translation((0, 0, 7))
    rod = trimesh.creation.cylinder(radius=1.0, height=30).apply_translation((5, 5, 0))
    away = trimesh.creation.box(extents=(4, 4, 4)).apply_translation((0, 0, 30))
    assert LG.interference(asm(("block", block)))["undeclared"] == [] and LG.floating(asm(("block", block))) == []
    assert [p[:2] for p in LG.interference(asm(("rod", rod)))["undeclared"]] == [["plate", "rod"]]
    assert LG.interference(asm(("rod", rod), joints=[("rod", "plate", "joint")]))["undeclared"] == []
    assert LG.floating(asm(("block", block), ("away", away))) == ["away"]
    assert LG.interference(asm(("away", away), joints=[("away", "plate", "contact")]))["loose"][0][:2] == ["away", "plate"]


def test_form_laws_measure_edges_and_walls():
    cube = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    assert LG.sharp_edge_mm(cube) == pytest.approx(120.0)
    assert LG.sharp_edge_mm(trimesh.creation.icosphere(subdivisions=3, radius=5.0)) == 0.0
    assert LG.thinnest_wall(trimesh.creation.box(extents=(20.0, 20.0, 2.0))) == pytest.approx(2.0, abs=0.05)
    assert LG.overhang_fraction(cube, (0.0, 0.0, 1.0)) == 0.0  # its bottom lies on the bed
