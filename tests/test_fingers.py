"""The finger family and the real objects (D-038): pad geometry, capability, and YCB tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sea.compose.library import Library
from sea.compose.typecheck import typecheck
from sea.contracts import Design, FingerSpec, HardLimits
from sea.control.frames import quat_wxyz_to_mat
from sea.design import fingers as F
from sea.design.grip import STOCK_WIDEST_MM, basis, widest_mm
from sea.sim.build_model import build_model, pair_friction
from sea.sim.grasp_world import (
    load_config,
    object_spec,
    real_object,
    real_panels,
    regression_panels,
    table_panel,
)
from sea.sim.objects import catalog, load_asset

ROOT = Path(__file__).resolve().parents[1]
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits_grasp.yaml").read_text()))
PADS = FingerSpec(reach_mm=10, pad_angle_deg=30, pad_tpu_mm=2)


# ---------------------------------------------------------------- geometry
def test_all_zero_is_the_stock_jaws():
    assert F.is_stock(FingerSpec()) and F.pad_geoms(FingerSpec()) == [] and F.thickness(FingerSpec()) == 0
    assert set(F.RANGES) | {"jaw", "link_mm", "drive_mm_per_rad"} == set(FingerSpec.model_fields)
    assert not F.is_stock(FingerSpec(pad_tpu_mm=1)) and F.thickness(FingerSpec(pad_tpu_mm=1)) == pytest.approx(0.004)


def test_the_gap_grows_with_the_opening_and_opening_for_inverts_it():
    for spec in (FingerSpec(pad_tpu_mm=2), PADS, FingerSpec(reach_mm=30, pad_angle_deg=45, pad_tpu_mm=3)):
        qs = np.linspace(*F.Q_RANGE, 40)
        gaps = [F.gap(spec, q) for q in qs]
        assert all(b > a for a, b in zip(gaps, gaps[1:], strict=False))
        for w in (0.019, 0.043, 0.067):
            assert F.gap(spec, F.opening_for(spec, w)) == pytest.approx(w, abs=1e-6)
        assert F.opening_for(spec, 1.0) is None  # a metre is past the widest opening


def _face_normal(spec: FingerSpec, kind: str, q: float) -> np.ndarray:
    g = next(g for g in F.pad_geoms(spec) if g.kind == kind)
    n = quat_wxyz_to_mat(g.quat)[:, 0]  # pad +x points at the object
    return n if g.body == "gripper" else F.R0 @ (F._rz(q) @ n)


def test_the_tilted_pad_is_parallel_at_its_angle_and_both_pads_end_level():
    q = math.radians(PADS.pad_angle_deg)
    fixed, moving = _face_normal(PADS, "jaw_fixed", q), _face_normal(PADS, "jaw_moving", q)
    assert fixed @ moving == pytest.approx(-1.0, abs=1e-6)  # opposed faces
    assert fixed @ _face_normal(PADS, "jaw_moving", 0.0) > -0.9  # not parallel when closed
    low = {g.kind: min(F._corners(g)[:, 2]) if g.body == "gripper" else
           min(F.moving_to_gripper(q, p)[2] for p in F._corners(g)) for g in F.pad_geoms(PADS)}
    assert low["jaw_fixed"] == pytest.approx(low["jaw_moving"], abs=1e-3)
    assert F.below_tool(PADS, q) == pytest.approx(F.TOOL_Z - (F.FIXED_TIP_Z - 0.010), abs=1e-3)


def test_a_groove_and_a_lip_add_faces():
    plain, grooved = F.pad_geoms(PADS), F.pad_geoms(PADS.model_copy(update={"groove_mm": 3, "lip_mm": 2}))
    assert len(plain) == 2 and len(grooved) == 6  # per pad: two groove halves and a lip
    assert {g.name for g in grooved} >= {"jaw_fixed_pad0", "jaw_fixed_pad2", "jaw_moving_pad2"}


def test_the_pads_are_jaw_geoms_with_their_material_friction():
    from sea.sim.build_model import JAW_FRICTION

    sim = build_model(Design(id="p", fingers=PADS), table_panel("t", "train", 0, object_spec("box", 40, 40, 100, 0.6)))
    m = sim.model
    pads = [i for i in range(m.ngeom) if m.geom(i).name.startswith("jaw_") and "_pad" in m.geom(i).name]
    assert len(pads) == 2 and {sim.geom_kind[i] for i in pads} == {"jaw_fixed", "jaw_moving"}
    names = {(m.geom(int(a)).name, m.geom(int(b)).name): k for k, (a, b) in enumerate(zip(m.pair_geom1, m.pair_geom2,
                                                                                          strict=True))}
    k = next(v for (a, _), v in names.items() if a == "jaw_fixed_pad0")
    assert m.pair_friction[k, 0] == pytest.approx(pair_friction(F.TPU_FRICTION, 0.6))
    stock = next(v for (a, _), v in names.items() if "_pad" not in a)
    assert m.pair_friction[stock, 0] == pytest.approx(pair_friction(JAW_FRICTION, 0.6))


# ---------------------------------------------------------------- capability
def test_capability_is_one_instrument_for_stock_and_pads():
    stock = Design(id="s")
    assert {s: widest_mm(stock, s) for s in STOCK_WIDEST_MM} == STOCK_WIDEST_MM and "table" in basis(stock)
    pads = Design(id="p", fingers=FingerSpec(pad_angle_deg=40, groove_mm=3, pad_tpu_mm=3, width_mm=20))
    assert "measured on these fingers" in basis(pads)
    # tilted pads hold wider cans and boxes, and spheres 5 mm narrower than the stock (D-054)
    assert all(widest_mm(pads, s) > STOCK_WIDEST_MM[s] for s in ("cylinder", "box"))
    assert widest_mm(pads, "sphere") < STOCK_WIDEST_MM["sphere"]
    lib = Library.seed()
    wide = object_spec("cylinder", 75, 80, 58, 0.7)
    assert any(e.code == "aperture" for e in typecheck(lib.prims["pick_up"].body, stock, lib.prims, LIMITS, control=wide))
    assert typecheck(lib.prims["pick_up"].body, pads, lib.prims, LIMITS, control=wide) == []
    scan = real_object("tennis_ball")  # its bounding width is not a grip width: the trial decides
    assert typecheck(lib.prims["pick_up"].body, stock, lib.prims, LIMITS, control=scan) == []


# ---------------------------------------------------------------- real objects
def test_every_catalog_object_has_an_asset_a_class_and_fits_under_the_arm():
    cat, classes = catalog(), load_config()["real"]
    assert set(cat) == set(classes) and len(cat) == 21
    for name, o in cat.items():
        a = load_asset(name)
        assert a.parts and all(p.shape[1] == 3 for p in a.parts) and len(a.visual_f) <= 3000
        assert np.allclose(a.extents_mm, [o["width_mm"], o["depth_mm"], o["height_mm"]], atol=0.1)
        assert o["height_ok"] and o["height_mm"] <= 85 and o["mass_g"] > 0


def test_real_tables_carry_the_scan_and_are_seeded():
    c = real_object("tomato_soup_can")
    assert c.asset == "tomato_soup_can" and c.shape == "sphere" and c.mass_g == 349 and c.size_mm == (67.7, 67.6)
    names = ["golf_ball", "cup"]
    a = real_panels("train", 4, names, seed=3)
    assert a == real_panels("train", 4, names, seed=3) and a != real_panels("val", 4, names, seed=3)
    assert [p.controls[0].asset for p in a] == names * 2 and a[0].id == "rtr000_golf_ball"
    assert [p.id for p in regression_panels(names)] == ["reg_golf_ball", "reg_cup"]
    sim = build_model(Design(id="s"), a[0])
    sim.step(500)
    assert sim.control_q("obj") == pytest.approx(0.0, abs=2e-3)  # a scan rests on the table


SLIDE = FingerSpec(jaw="prismatic", drive_mm_per_rad=40)
LINKS = FingerSpec(jaw="parallelogram", link_mm=70)


def test_jaw_mechanisms_d042():
    """The moving jaw's joint is a part: the slide neither turns nor rises, the parallelogram turns not but
    rises on its circle, the hinge does both; every gap still grows with the servo angle."""
    tip = np.array([F.MOVING_FACE_X, F.MOVING_TIP_Y, F.MOVING_MID_Z])
    for spec in (FingerSpec(), SLIDE, LINKS, SLIDE.model_copy(update={"pad_tpu_mm": 2, "reach_mm": 10})):
        qs = np.linspace(*F.q_range(spec), 40)
        gaps = [F.gap(spec, q) for q in qs]
        assert all(b > a for a, b in zip(gaps, gaps[1:], strict=False)), spec
        assert F.gap(spec, F.opening_for(spec, 0.06)) == pytest.approx(0.06, abs=1e-6)
    assert F.tip_rise(1.2, SLIDE) == pytest.approx(0.0) and F.tip_rise(1.2, FingerSpec()) > 0.035
    assert F.tip_rise(1.2, LINKS) == pytest.approx(0.070 * (1 - np.cos(1.2)), abs=1e-9)
    assert np.allclose(F.finger_rot(0.9, LINKS), F.R0) and not np.allclose(F.finger_rot(0.9, FingerSpec()), F.R0)
    assert F.gap(SLIDE, 1.0) - F.gap(SLIDE, 0.0) == pytest.approx(0.040)
    assert F.opening_for(SLIDE, 0.080) is not None and F.opening_for(SLIDE, 0.090) is None  # 15.8 + 40 x 1.745 mm
    assert F.squeeze_n(SLIDE, 0.5, 1.62) == pytest.approx(40.5) and F.squeeze_n(FingerSpec(), 0.0, 1.62) < 25
    assert not F.has_pads(SLIDE) and F.thickness(SLIDE) == 0 and F.pad_geoms(SLIDE) == []
    assert F.below_tool(SLIDE, 1.0) == pytest.approx(F.below_tool(SLIDE, 0.0), abs=1e-9)
    pads = F.pad_geoms(LINKS.model_copy(update={"pad_tpu_mm": 2}))
    assert {g.body for g in pads} == {"gripper", "jaw_coupler"}
    assert F.moving_to_gripper(0.0, tip, LINKS) == pytest.approx(F.moving_to_gripper(0.0, tip))
