"""Sensing (docs/gripper-experiment.md "Sensing"): servo telemetry, touch, IMU and the payload estimate."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from sea.contracts import Design, SensorSuite
from sea.control.executor import HOME_Q, TICK_STEPS
from sea.sim.build_model import build_model
from sea.sim.grasp_probe import pick, scene_object
from sea.sim.sensors import Sensors, load_config, manifest

CAN_H = 0.06


def _can_scene(mass_g: float, d_mm: float = 30.0):
    return build_model(Design(id="stock"), objects=[scene_object("cylinder", d_mm, mass_g)], table_z=0.0)


def _settle(sim, q, ticks: int = 20) -> None:
    for _ in range(ticks):
        sim.data.ctrl[:] = q
        sim.step(TICK_STEPS)


def test_idle_arm_feels_only_its_own_weight():
    sim = _can_scene(50)
    sim.set_arm_q(HOME_Q)
    _settle(sim, HOME_Q)
    f = Sensors(sim, seed=0).read()
    assert f.static and abs(f.payload_g) < 25
    assert np.linalg.norm(f.wrist_acc) == pytest.approx(9.81, abs=0.1)  # an IMU at rest reads gravity
    assert f.jaw_force_n.max() < 0.5 and f.jaw_force_n.min() >= 0
    assert f.link_force_n is None  # link skin is a sim-only option, off by default
    assert f.current_a[1] > 0.3 > f.current_a[0]  # shoulder_lift holds the arm up; the pan joint idles
    assert f.kinetic_j < 1e-4


@pytest.fixture(scope="module")
def held():
    """(mass_g -> (frame, can height)) after picking a 30 mm can of each mass."""
    out = {}
    for mass in (50, 400):
        sim = pick(scene_object("cylinder", 30.0, mass), Design(id="skin", sensors=SensorSuite(link_skin=True))).sim
        out[mass] = (Sensors(sim, seed=1).read(), float(sim.data.xpos[sim.objects["obj"]][2]))
    return out


def test_payload_jaws_and_current_track_the_held_mass(held):
    for mass, (f, z) in held.items():
        assert z > CAN_H / 2 + 0.03  # actually lifted (ground truth)
        assert f.static and f.payload_g == pytest.approx(mass, abs=40)
        assert f.jaw_force_n.min() > 10  # both pads squeeze
        assert set(f.link_force_n) == {"gripper", "moving_jaw_so101_v1"}  # only the jaws touch anything
        assert f.current_a[-1] > 2.5  # the gripper servo holds at stall while it squeezes
    assert held[400][0].current_a[1] > held[50][0].current_a[1] + 0.5  # shoulder works harder for 400 g


def test_sensors_are_seeded_and_quantized():
    sim = _can_scene(50)
    sim.set_arm_q(HOME_Q)
    _settle(sim, HOME_Q, 5)
    a, b = Sensors(sim, seed=7).read(), Sensors(sim, seed=7).read()
    assert np.array_equal(a.load, b.load) and np.array_equal(a.wrist_acc, b.wrist_acc)
    exact = Sensors(sim, seed=None).read()
    res = Sensors(sim).ch["q"]["resolution"]
    assert np.allclose(exact.q, np.round(sim.arm_q() / res) * res)
    assert not np.array_equal(Sensors(sim, seed=8).read().load, a.load)


def test_a_heavier_arm_than_the_self_model_reads_as_phantom_payload():
    # Injected fault: 80 g more at the gripper than the robot's own model knows about.  Nothing is
    # held, so the payload estimate is the robot noticing that its arm no longer matches its model.
    sim = _can_scene(50)
    sim.model.body_mass[sim.model.body("gripper").id] += 0.08
    sim.set_arm_q(HOME_Q)
    _settle(sim, HOME_Q, 40)
    f = Sensors(sim, seed=0).read()
    assert f.payload_g == pytest.approx(80, abs=30) and f.jaw_force_n.max() < 0.5


def test_the_judges_never_read_sensors():
    # D-031: verdicts, reward and acceptance come from ground truth, so sensor noise cannot move a pass.
    src = Path(__file__).resolve().parents[1] / "src" / "sea"
    judges = [*(src / "verify").glob("*.py"), src / "compose" / "checks.py", src / "spec" / "posture.py"]
    for path in judges:
        assert not re.search(r"\bsensors\b", path.read_text()), path.name


def test_unmounted_parts_read_as_absent_and_the_manifest_says_so():
    # D-033: the body decides which channels exist, and the agent is told exactly that.
    bare = Design(id="bare", sensors=SensorSuite(jaw_pads=False))
    sim = build_model(bare, objects=[scene_object("cylinder", 30.0, 50.0)], table_z=0.0)
    sim.set_arm_q(HOME_Q)
    _settle(sim, HOME_Q, 5)
    f = Sensors(sim, seed=0).read()
    assert f.jaw_force_n is None and f.link_force_n is None and f.wrist_acc is not None and f.static
    on = {p["part"]: p["mounted"] for p in manifest(bare)}
    assert on == {"servo_telemetry": True, "wrist_imu": True, "jaw_pads": False, "link_skin": False}
    cfg = load_config()
    toggles = {n for n, part in cfg["parts"].items() if part["toggle"]}
    assert toggles == set(SensorSuite.model_fields)  # every toggleable part has a switch on the design
    owned = [c for part in cfg["parts"].values() for c in part["channels"]]
    assert sorted(owned) == sorted(cfg["channels"])  # every channel belongs to exactly one part
    assert cfg["parts"]["wrist_imu"]["priority"] == "high" and not cfg["parts"]["link_skin"]["bench"]

