"""A guide's rods as beams through every bushing, held at one end or both (D-080)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from sea.contracts import FingerSpec
from sea.mech import build
from sea.mech.graph import Bought, Box
from sea.mech.patch import Op, apply_patch
from sea.sim import jaws as J
from sea.worlds.gripper import _guide_errors, _module, _rod_numbers


def _long_slide(bushings, far: bool):
    g = _module("long_slide", {"sector_radius_mm": 50})
    rod = next(p for b in g.bodies if b.name == "gripper" for p in b.parts if getattr(p, "role", "") == "rod")
    front = rod.pos[0] + rod.half_length
    ops = [Op(op="set", path="guides/jaw_carriage/bushings", value=list(bushings))]
    ops += [Op(op="add", path="bodies/jaw_carriage/parts", value={"kind": "bought", "item": "lm3uu", "kg": J.BUSHING_KG})
            for _ in range(2 * len(bushings) - 2)]
    if far:
        ops += [Op(op="set", path="guides/jaw_carriage/far_end", value=front),
                Op(op="add", path="bodies/gripper/parts", value={
                    "kind": "box", "name": "far_clamp", "pos": [front + 0.004, rod.pos[1], rod.pos[2]],
                    "half": [0.004, 0.005, 0.01], "role": "clamp", "piece": "far_clamp"})]
    return apply_patch(g, ops)


def test_one_bushing_on_cantilevers_is_d076():
    for a in (0.01, 0.05, 0.12):
        kc, kb = J.guide_beam((a,), (0.0,), None)
        assert np.allclose(kc, 2 * J.ROD_EI / a**3 * np.array([[12, -6 * a], [-6 * a, 4 * a * a]]), rtol=1e-12)
        assert np.allclose(2 * kb, kc, rtol=1e-12)  # each rod carries half


def test_holding_both_ends_and_spreading_bushings_unload_them():
    """The long slide at full opening: holding the far ends halves the rack's lift, and two bushings per rod
    spread wider carry less each; with one bushing the squeeze's couple across its 10 mm sets its load."""
    q = 1.4
    rows = {k: _rod_numbers(FingerSpec(jaw="built", mechanism=_long_slide(bs, far).model_dump_json()), q, q)
            for k, (bs, far) in {"cant": ((0.0,), False), "both": ((0.0,), True), "12": ((-0.012, 0.012), True),
                                 "25": ((-0.025, 0.025), True)}.items()}

    lift = {k: (v["lift_mm"] - v["bend_mm"]) for k, v in rows.items()}
    load = {k: v["bushing_n"] for k, v in rows.items()}
    assert lift["both"] < 0.6 * lift["cant"] and lift["25"] < lift["12"] < lift["both"]
    assert load["both"] == pytest.approx(load["cant"], rel=0.02)
    assert load["25"] < load["12"] < 0.6 * load["both"]


def test_the_trials_rods_load_the_beam():
    from sea.control.frames import STOCK_MJCF

    g = _long_slide((-0.012, 0.012), True)
    spec = mujoco.MjSpec.from_file(str(STOCK_MJCF))
    build.compile(spec, g)
    m = spec.compile()
    rods = J.RodSprings(m)
    assert rods.beam is not None and rods.beam[2] == (-0.012, 0.012)
    d = mujoco.MjData(m)
    u = {"jaw_rod_dy": 1e-4, "jaw_rod_yaw": 2e-3}  # across the rods sideways: the teeth push only in the other plane
    for n, v in u.items():
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_forward(m, d)
    rods.apply(d)
    s = float(d.qpos[m.jnt_qposadr[m.joint("jaw_slide").id]])
    far, _, bs = rods.beam
    kc, _ = J.guide_beam(J.bushing_overhangs(rods.centre, s, rods.clamp, bs, far), bs, far - rods.clamp)
    f = kc @ np.array([u["jaw_rod_dy"], u["jaw_rod_yaw"]])
    got = [d.qfrc_applied[m.jnt_dofadr[m.joint(n).id]] for n in ("jaw_rod_dy", "jaw_rod_yaw")]
    assert got == pytest.approx(list(-f), rel=1e-9)
    assert not rods.broken()


def test_catalogue_slides_keep_d076s_rods():
    from sea.control.frames import STOCK_MJCF

    for name, p in (("prismatic", {"pinion_teeth": 30}), ("long_slide", {"sector_radius_mm": 40})):
        spec = mujoco.MjSpec.from_file(str(STOCK_MJCF))
        build.compile(spec, _module(name, p))
        m = spec.compile()
        assert all(m.numeric(i).name != "jaw_guide" for i in range(m.nnumeric)) and J.RodSprings(m).beam is None


def test_a_built_guide_must_be_buildable_as_drawn():
    good = _long_slide((-0.012, 0.012), True)
    assert _guide_errors(good, good.guides[0]) == []
    overlap = _long_slide((-0.004, 0.004), True)
    assert "overlap" in _guide_errors(overlap, overlap.guides[0])[0]
    short = apply_patch(good, [Op(op="remove", path="bodies/jaw_carriage/parts/lm3uu")])
    assert any("carries 3 LM3UU" in e for e in _guide_errors(short, short.guides[0]))
    loose = apply_patch(good, [Op(op="remove", path="bodies/gripper/parts/far_clamp")])
    assert any("nothing holds the rods' far end" in e for e in _guide_errors(loose, loose.guides[0]))
    assert isinstance(next(p for b in good.bodies for p in b.parts if getattr(p, "name", "") == "far_clamp"), Box)
    assert sum(isinstance(p, Bought) for b in good.bodies for p in b.parts) == 4
    assert math.isfinite(build.rod_end(good))
