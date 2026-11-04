"""The seed library's machine elements do what a machine-elements text says (D-096), on an empty world."""

from __future__ import annotations

import math

import mujoco
import pytest

from sea.mech import build
from sea.mech.base import WORLD
from sea.mech.elements import ELEMENTS, fragment, merge
from sea.mech.graph import Body, Box, Gearmotor, Graph, Servo
from sea.mech.kinematics import _residual

BASE = Graph(name="base", bodies=(Body(name="frame", parent="world", pos=(0.0, 0.0, 0.5), parts=(
    Box(name="frame_box", pos=(0.0, 0.0, -0.1), half=(0.05, 0.05, 0.02)),)),))


def _model(frag: Graph, *acts, gravity=False):
    g = merge(BASE, frag).model_copy(update={"actuators": acts})
    spec = WORLD.model()
    if not gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.timestep = 0.001
    build.compile(spec, g, WORLD)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _q(m, d, name):
    return float(d.qpos[m.jnt_qposadr[m.joint(name).id]])


def _servo(joint, kp=200.0, stall=20.0):
    return Servo(name="drive", joint=joint, stall_nm=stall, kp=kp, kv=2.0, armature=0.001)


def _settle(m, d, target, secs=1.5):
    d.ctrl[m.actuator("drive").id] = target
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)


@pytest.mark.parametrize("name", sorted(ELEMENTS))
def test_each_element_compiles_at_its_midrange_and_its_loops_close(name):
    mids = {"four_bar": {"crank_m": 0.05, "coupler_m": 0.15, "rocker_m": 0.12, "ground_m": 0.16, "start_rad": 1.0},
            "slider_crank": {"crank_m": 0.05, "rod_m": 0.15, "start_rad": 1.0}}.get(name)
    m, d = _model(fragment(name, "e", "frame", (0.0, 0.0, 0.0), params=mids))
    res = _residual(m, d)
    assert res.size == 0 or abs(res).max() < 1e-5


def test_a_lever_s_end_travels_its_arm_times_its_angle():
    m, d = _model(fragment("lever", "e", "frame", params={"arm_m": 0.2, "swing_rad": 1.0}), _servo("e_q"))
    _settle(m, d, 0.5)
    assert _q(m, d, "e_q") == pytest.approx(0.5, abs=0.02)
    tip = d.xpos[m.body("e_arm").id] + d.xmat[m.body("e_arm").id].reshape(3, 3) @ [0.2, 0.0, 0.0]
    assert tip[2] - 0.5 == pytest.approx(-0.2 * math.sin(_q(m, d, "e_q")), abs=1e-3)


def test_a_four_bar_stays_closed_and_its_rocker_rocks():
    g = fragment("four_bar", "e", "frame", params={"crank_m": 0.05, "coupler_m": 0.15, "rocker_m": 0.12, "ground_m": 0.16,
                                                   "start_rad": 1.0})
    m, d = _model(g, _servo("e_q"))
    seen, rock = [], []
    for target in (0.0, 1.5, 3.0, 4.5, 6.0):
        _settle(m, d, target, 1.0)
        seen.append(abs(_residual(m, d)).max())
        rock.append(_q(m, d, "e_q3"))
    assert max(seen) < 2e-3 and max(rock) - min(rock) > 0.3  # the crank turns fully: Grashof, shortest link driving
    with pytest.raises(ValueError, match="cannot assemble"):
        fragment("four_bar", "e", "frame", params={"crank_m": 0.01, "coupler_m": 0.02, "rocker_m": 0.02, "ground_m": 0.3,
                                                   "start_rad": 0.0})


def test_a_slider_crank_strokes_twice_its_crank():
    g = fragment("slider_crank", "e", "frame", params={"crank_m": 0.05, "rod_m": 0.15, "start_rad": 1.0})
    m, d = _model(g, _servo("e_q"))
    s = []
    for k in range(13):  # around the whole turn: the dead centres are not at round angles from the start
        _settle(m, d, k * math.pi / 6, 0.5)
        s.append(_q(m, d, "e_s"))
    assert max(s) - min(s) == pytest.approx(0.10, abs=0.005)


def test_a_rack_moves_the_pitch_radius_per_radian():
    m, d = _model(fragment("rack_pinion", "e", "frame", params={"pitch_r_m": 0.02, "width_m": 0.01, "travel_m": 0.1}), _servo("e_q"))
    _settle(m, d, 1.0)
    assert abs(_q(m, d, "e_s")) == pytest.approx(0.02 * _q(m, d, "e_q"), rel=0.02)


def test_a_gear_train_reverses_by_its_ratio_and_trades_speed_for_torque():
    frag = fragment("gear_train", "e", "frame", params={"driver_r_m": 0.01, "follower_r_m": 0.04, "width_m": 0.01})
    m, d = _model(frag, _servo("e_q"))
    _settle(m, d, 2.0)
    assert _q(m, d, "e_q2") == pytest.approx(-_q(m, d, "e_q") * 0.25, rel=0.02)


def test_a_cam_lifts_its_follower_twice_its_eccentricity():
    m, d = _model(fragment("cam", "e", "frame", params={"disc_r_m": 0.03, "eccentric_m": 0.01}), _servo("e_q"), gravity=True)
    low = []
    for k in range(13):
        _settle(m, d, k * math.pi / 6, 0.5)
        low.append(_q(m, d, "e_s"))
    assert max(low) - min(low) == pytest.approx(0.02, abs=0.004)


@pytest.mark.parametrize("r,teeth", [(0.03, 8), (0.05, 12), (0.02, 6), (0.08, 16), (0.015, 6), (0.1, 16)])
def test_a_ratchet_turns_one_way_only(r, teeth):
    frag = fragment("ratchet", "e", "frame", at=(0.0, 0.0, 0.1), params={"wheel_r_m": r, "teeth": teeth})  # clear of the frame
    m, d = _model(frag, Gearmotor(name="drive", joint="e_q", stall_nm=10.0 * r, no_load_rad_s=5.0))  # 10 N at the teeth
    d.ctrl[m.actuator("drive").id] = 1.0
    for _ in range(3000):
        mujoco.mj_step(m, d)
    assert _q(m, d, "e_q") > 3 * 2 * math.pi / teeth  # free: it has turned past the pawl, tooth after tooth
    mujoco.mj_resetData(m, d)
    d.ctrl[m.actuator("drive").id] = -1.0
    for _ in range(3000):
        mujoco.mj_step(m, d)
    assert abs(_q(m, d, "e_q")) < 2 * math.pi / teeth + 0.1  # locked: within a tooth


def test_a_latch_holds_until_released_and_a_guide_carries_in_a_line():
    m, d = _model(fragment("latch", "e", "frame", params={"arm_m": 0.1, "swing_rad": 1.5}), gravity=True)
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert abs(_q(m, d, "e_q")) < 1e-3
    d.eq_active[m.equality("e_latch").id] = 0
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert abs(_q(m, d, "e_q")) > 0.05
    m, d = _model(fragment("guide", "e", "frame", params={"rod_m": 0.2, "rod_gap_m": 0.04, "travel_m": 0.08}), _servo("e_s"))
    _settle(m, d, 0.05)
    assert _q(m, d, "e_s") == pytest.approx(0.05, abs=0.005)


def test_composition_refuses_a_second_use_of_a_name_and_a_body_that_is_not_there():
    g = merge(BASE, fragment("lever", "a", "frame"))
    with pytest.raises(ValueError, match="another prefix"):
        merge(g, fragment("lever", "a", "frame"))
    with pytest.raises(ValueError, match="to mount on"):
        merge(g, fragment("lever", "b", "nowhere"))
    with pytest.raises(ValueError, match="outside"):
        fragment("lever", "b", "frame", params={"arm_m": 9.0})
