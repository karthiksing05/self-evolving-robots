"""The graph's machine elements behave as their datasheet lines say (D-087), each on a robot built whole from a
graph on the empty base."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from sea.mech import build
from sea.mech.base import WORLD
from sea.mech.graph import (
    Band,
    Body,
    Box,
    Cylinder,
    Encoder,
    Gearmotor,
    Graph,
    Joint,
    Latch,
    LimitSwitch,
    Piston,
    RangeFinder,
    Servo,
    TorsionSpring,
    Wheel,
)
from sea.mech.runtime import Machine


def _model(g: Graph, gravity: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData, Machine]:
    spec = WORLD.model()
    if not gravity:
        spec.option.gravity = [0.0, 0.0, 0.0]
    build.compile(spec, g, WORLD)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, Machine(m, d, g)


def _rotor(*acts, joint=Joint(name="q", type="hinge", axis=(0.0, 0.0, 1.0)), **kw) -> Graph:
    disc = Cylinder(name="disc", pos=(0.0, 0.0, 0.0), radius=0.05, half_length=0.005)
    return Graph(name="rotor", bodies=(Body(name="rotor", parent="world", pos=(0.0, 0.0, 0.5), joints=(joint,),
                                            parts=(disc,)),), actuators=acts, **kw)


def test_a_gearmotor_gives_its_stall_torque_at_rest_and_runs_up_to_its_no_load_speed():
    m, d, mc = _model(_rotor(Gearmotor(name="m", joint="q", stall_nm=1.2, no_load_rad_s=16.8)))
    mc.command("m", 1.0)
    mujoco.mj_forward(m, d)
    assert d.actuator_force[0] == pytest.approx(1.2)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.qvel[0] == pytest.approx(16.8, rel=0.01)
    mc.command("m", -0.5)  # half duty backward: half the stall torque, against the spin
    mujoco.mj_forward(m, d)
    assert d.actuator_force[0] == pytest.approx(-0.6)  # and the back-EMF is the joint's damping
    assert -d.qfrc_passive[0] == pytest.approx(1.2 / 16.8 * d.qvel[0], rel=1e-6)


def test_a_piston_pushes_pressure_times_area_and_the_needle_valve_limits_its_speed():
    bore, p, v = 0.75 * 0.0254, 6.9e5, 0.3
    g = _rotor(Piston(name="p", joint="q", bore_m=bore, pressure_pa=p, valve_m_s=v),
               joint=Joint(name="q", type="slide", axis=(1.0, 0.0, 0.0)))
    m, d, mc = _model(g, gravity=False)
    mc.command("p", 1.0)
    mujoco.mj_forward(m, d)
    assert d.actuator_force[0] == pytest.approx(p * math.pi * (bore / 2) ** 2)
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.qvel[0] == pytest.approx(v, rel=0.01)


def test_a_torsion_spring_falls_linearly_over_its_swing():
    g = _rotor(TorsionSpring(name="trap", joint="q", torque_nm=(0.45, 0.20), span_rad=math.pi))
    m, d, _ = _model(g)
    for q, want in ((0.0, 0.45), (math.pi / 2, 0.325), (math.pi, 0.20)):
        d.qpos[0] = q
        mujoco.mj_forward(m, d)
        assert d.actuator_force[0] == pytest.approx(want)


def test_a_band_pulls_k_per_metre_past_its_rest_length_and_never_pushes():
    g = _rotor(Band(name="band", a="world", a_point=(-0.1, 0.0, 0.5), b="rotor", b_point=(0.0, 0.0, 0.0),
                    k_n_m=40.0, rest_m=0.089), joint=Joint(name="q", type="slide", axis=(1.0, 0.0, 0.0)))
    m, d, _ = _model(g, gravity=False)
    for x, want in ((0.0, 40.0 * (0.1 - 0.089)), (0.05, 40.0 * (0.15 - 0.089)), (-0.02, 0.0)):
        d.qpos[0] = x
        mujoco.mj_forward(m, d)
        assert -d.qfrc_passive[0] == pytest.approx(want, abs=1e-9)


def test_a_latch_holds_its_joint_until_released():
    arm = Box(name="arm", pos=(0.1, 0.0, 0.0), half=(0.1, 0.01, 0.01))
    g = Graph(name="pendulum", bodies=(Body(name="arm", parent="world", pos=(0.0, 0.0, 0.5),
                                            joints=(Joint(name="q", type="hinge", axis=(0.0, 1.0, 0.0)),),
                                            parts=(arm,)),), latches=(Latch(name="trigger", joint="q"),))
    m, d, mc = _model(g)
    for _ in range(250):
        mujoco.mj_step(m, d)
    assert abs(d.qpos[0]) < 1e-3 and mc.held("trigger")
    mc.release("trigger")
    for _ in range(250):
        mujoco.mj_step(m, d)
    assert d.qpos[0] > 1.0  # it fell


def test_sensors_read_the_world():
    block = Box(name="block", pos=(0.0, 0.0, 0.0), half=(0.02, 0.02, 0.02))
    plate = Box(name="plate", pos=(0.0, 0.0, 0.0), half=(0.05, 0.05, 0.005))
    g = Graph(name="bench", bodies=(
        Body(name="bench", parent="world", pos=(0.0, 0.0, 0.3), parts=(plate,)),
        Body(name="block", parent="world", pos=(0.0, 0.0, 0.33), joints=(Joint(name="drop", type="free"),),
             parts=(block,))),
        sensors=(LimitSwitch(name="switch", body="bench", pos=(0.0, 0.0, 0.005), half=(0.03, 0.03, 0.003)),
                 RangeFinder(name="ir", body="bench", pos=(0.04, 0.04, -0.006), quat=(0.0, 1.0, 0.0, 0.0))))
    m, d, mc = _model(g)
    for _ in range(500):
        mujoco.mj_step(m, d)
    kg = m.body("block").mass[0]
    assert mc.read("switch") == pytest.approx(kg * 9.81, rel=0.05)
    assert mc.read("ir") == pytest.approx(0.3 - 0.006, abs=1e-6)  # the floor, from the sensor under the plate


def test_a_servo_holds_its_target_and_a_chassis_rolls_on_its_wheels():
    m, d, mc = _model(_rotor(Servo(name="s", joint="q", stall_nm=1.0), sensors=(Encoder(name="enc", joint="q"),)))
    mc.command("s", 0.7)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert d.qpos[0] == pytest.approx(0.7, abs=1e-3) and mc.read("enc") == pytest.approx(d.qpos[0])
    r = 0.03
    wheels = [Body(name=f"w{k}", parent="cart", pos=(0.0, y, 0.0), joints=(Joint(name=f"w{k}", type="hinge",
                                                                                axis=(0.0, 1.0, 0.0)),),
                   parts=(Wheel(name=f"tread{k}", pos=(0.0, 0.0, 0.0), radius=r, half_width=0.008,
                                quat=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),)) for k, y in enumerate((-0.08, 0.08))]
    skid = Cylinder(name="skid", pos=(-0.12, 0.0, -0.02), radius=0.01, half_length=0.005)
    cart = Body(name="cart", parent="world", pos=(0.0, 0.0, r + 0.001), joints=(Joint(name="root", type="free"),),
                parts=(Box(name="deck", pos=(-0.05, 0.0, 0.0), half=(0.08, 0.06, 0.005)), skid))
    g = Graph(name="cart", bodies=(cart, *wheels),
              actuators=tuple(Gearmotor(name=f"m{k}", joint=f"w{k}", stall_nm=0.3, no_load_rad_s=10.0) for k in (0, 1)))
    m, d, mc = _model(g)
    for k in (0, 1):
        mc.command(f"m{k}", 1.0)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    v = np.linalg.norm(d.qvel[:2])
    assert v == pytest.approx(10.0 * r, rel=0.1)  # it rolls: the wheels' speed times their radius
