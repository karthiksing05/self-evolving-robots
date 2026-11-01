"""Reach envelope grid (spec 7.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sea.contracts import HardLimits
from sea.control.frames import approach_axis
from sea.control.reach import DIRS, GRID_SHAPE, ReachGrid, design_key, reach_for

ROOT = Path(__file__).resolve().parents[1]
GRID = ReachGrid.load(ROOT / "tests" / "fixtures" / "reach_stock.npz")
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
POINTS = yaml.safe_load((ROOT / "tests" / "fixtures" / "ik_points.yaml").read_text())["points"]


def test_fixture_shape_and_bounds():
    assert GRID.occ.shape == (len(DIRS),) + GRID_SHAPE
    np.testing.assert_allclose(GRID.lo, LIMITS.workspace_box[0].as_list())
    np.testing.assert_allclose(GRID.hi, LIMITS.workspace_box[1].as_list())
    assert set(np.unique(GRID.occ)) <= {0.0, 1.0}
    assert [len(a) for a in GRID.axes()] == list(GRID_SHAPE)


def test_lookup_on_grid_nodes_and_outside():
    xs, ys, zs = GRID.axes()
    for di, name in enumerate(DIRS):
        for ix, iy, iz in [(0, 0, 0), (4, 4, 3), (8, 8, 6), (3, 5, 2)]:
            p = np.array([xs[ix], ys[iy], zs[iz]])
            assert GRID.occupancy(p, name) == pytest.approx(GRID.occ[di, ix, iy, iz])
    assert GRID.occupancy(np.array([0.5, 0.0, 0.0]), "down") == 0.0
    assert GRID.occupancy(np.array([0.2, 0.0, -0.2]), "45") == 0.0


def test_trilinear_interpolation():
    g = ReachGrid(np.zeros((3, 2, 2, 2)), np.zeros(3), np.ones(3))
    g.occ[0, 1, :, :] = 1.0  # occupied on the x = 1 face only
    assert g.occupancy(np.array([0.25, 0.5, 0.5]), "down") == pytest.approx(0.25)
    assert g.occupancy(np.array([0.75, 0.1, 0.9]), "down") == pytest.approx(0.75)
    assert not g.reachable(np.array([0.4, 0.5, 0.5]), np.array([0.0, 0.0, -1.0]))
    assert g.reachable(np.array([0.6, 0.5, 0.5]), np.array([0.0, 0.0, -1.0]))
    assert not g.reachable(np.array([0.6, 0.5, 0.5]), np.array([1.0, 0.0, 0.0]))  # other bin is empty


def test_grid_agrees_with_appendix_c_inside_the_box():
    """Grid lookups are coarse; points on/near grid nodes must agree with the IK fixture."""
    agree = 0
    total = 0
    for p in POINTS:
        pos = np.array([p["x"], p.get("y", 0.0), p["z"]])
        if np.any(pos < GRID.lo) or np.any(pos > GRID.hi):
            continue
        total += 1
        agree += GRID.reachable(pos, approach_axis(p["approach"], pos)) == (p["expected"] == "reachable")
    assert total >= 12 and agree / total >= 0.75


def test_save_load_and_cache(tmp_path):
    path = tmp_path / "g.npz"
    GRID.save(path)
    back = ReachGrid.load(path)
    np.testing.assert_array_equal(back.occ, GRID.occ)

    class FakeIK:
        urdf = ROOT / "assets" / "so101" / "so101_new_calib.urdf"

    key = design_key(FakeIK.urdf, None, LIMITS)
    assert key == design_key(FakeIK.urdf, np.eye(4), LIMITS)  # identity tip = stock
    assert key != design_key(FakeIK.urdf, None, LIMITS.model_copy(update={"max_force_n": 10.0}))
    GRID.save(tmp_path / f"reach_{key}.npz")
    got = reach_for(FakeIK(), LIMITS, None, tmp_path)  # type: ignore[arg-type]
    np.testing.assert_array_equal(got.occ, GRID.occ)


def test_compute_small_grid():
    from sea.control.executor import stock_ik

    g = ReachGrid.compute(stock_ik(), LIMITS, shape=(2, 1, 2), seeds=1)
    assert g.occ.shape == (3, 2, 1, 2)
