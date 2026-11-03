"""The printed pad shoe in the simulator (D-066): what is drawn is what is simulated."""

import mujoco
import numpy as np
import pytest

from sea.contracts import ArmSpec, Design, FingerSpec, SensorSuite
from sea.design import fingers as F
from sea.design import language as L
from sea.sim import shoe
from sea.sim.build_model import build_model


def _sim(**kw):
    return build_model(Design(id="t", fingers=FingerSpec(**kw), arm=ArmSpec(), sensors=SensorSuite(jaw_pads=True)))


def test_the_shoe_keeps_the_design_languages_fits():
    assert shoe.SLIP == L.CLEARANCE and shoe.EDGE == L.EDGE_R["printed"]


def test_stock_fingers_have_no_shoe():
    m = _sim().model
    assert not [i for i in range(m.ngeom) if "shoe" in m.geom(i).name or "_pad_screw" in m.geom(i).name]


@pytest.mark.parametrize("kw", [dict(pad_angle_deg=30, pad_tpu_mm=0.5),
                                dict(jaw="prismatic", drive_mm_per_rad=47, sector=True, reach_mm=20, pad_tpu_mm=0.5),
                                dict(groove_mm=3, pad_tpu_mm=1.5, reach_mm=5)])
def test_each_pad_has_a_shoe_that_wraps_its_finger_behind_the_land(kw):
    spec = FingerSpec(**kw)
    bare, padded = _sim(jaw=spec.jaw, drive_mm_per_rad=spec.drive_mm_per_rad, sector=spec.sector).model, _sim(**kw).model
    for kind, body in (("jaw_fixed", "gripper"), ("jaw_moving", F.moving_body(spec))):
        pieces = [i for i in range(padded.ngeom) if padded.geom(i).name.startswith(f"{kind}_pad_shoe")]
        assert pieces and all(padded.geom_contype[i] for i in pieces)  # they collide, as jaw geoms
        assert len([i for i in range(padded.ngeom) if padded.geom(i).name.startswith(f"{kind}_pad_screw")]) == 2
        added = padded.body(body).mass[0] - bare.body(body).mass[0]
        assert 0.0015 < added < 0.006  # a shoe, its skin and two M2 screws: a few grams
    # every piece stops behind its land's face, so the land (with the skin's friction) is what an object meets
    t = 1000 * F.thickness(spec)
    for fr, sh in shoe.shoes(spec, _hosts(spec)):
        for p in sh.pieces:
            local = (fr.rot.T @ (1000 * p - fr.origin).T).T
            assert local[:, 0].max() <= t - shoe.FRONT_MM + 1e-6


def _hosts(spec: FingerSpec):
    from sea.control.frames import STOCK_MJCF
    from sea.sim.build_model import _finger_hosts, _jaw_joint

    s = mujoco.MjSpec.from_file(str(STOCK_MJCF))
    _jaw_joint(s, spec)
    return _finger_hosts(s)


def test_the_shoe_wall_clears_the_fingertip():
    """The socket holds the stock tip: no tip vertex lies outside the shoe's outside below its top."""
    spec = FingerSpec(pad_angle_deg=0, pad_tpu_mm=0.5)
    fr, sh = next((f, s) for f, s in shoe.shoes(spec, _hosts(spec)) if f.kind == "jaw_fixed")
    from scipy.spatial import Delaunay

    outside = Delaunay(np.vstack([(fr.rot.T @ (1000 * p - fr.origin).T).T for p in sh.pieces]))
    v = next(np.frombuffer(b).reshape(-1, 3) for n, b, _ in _hosts(spec) if n == "gripper")
    local = (fr.rot.T @ (1000 * v - fr.origin).T).T
    tip = local[(local[:, 1] >= 0.5) & (local[:, 1] <= fr.length + 1000 * F.COVER - 0.5)
                & (np.abs(local[:, 0]) < 40) & (np.abs(local[:, 2]) < 40)]
    assert len(tip) > 20 and (outside.find_simplex(tip) >= 0).mean() > 0.97
