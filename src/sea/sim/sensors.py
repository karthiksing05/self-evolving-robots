"""What the robot can feel about itself and its contacts (docs/gripper-experiment.md "Sensing").

``Sensors.read`` returns one ``SensorFrame``: servo telemetry (position, speed, signed load,
current), jaw pressure pads, per-link contact force, and a wrist IMU, each with the noise and
quantization of the part named in ``configs/sensors.yaml``.  Derived signals (joint torque,
payload, kinetic energy, momentum) are computed from those noisy channels and the robot's *own*
nominal model (``SelfModel``), never from ground truth, so a fault injected into the simulated
arm shows up as a residual the robot can notice.

The verifier never reads a ``SensorFrame``: verdicts come from ground-truth sim state.  Sensor
noise can therefore change what the robot decides, but never what counts as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sea.contracts import ARM_JOINTS, Design
from sea.control.frames import MJCF_TOOL_SITE
from sea.sim.build_model import STALL_TORQUE, Sim, build_model

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sensors.yaml"
G = 9.81
STATIC_QD = 0.05  # rad/s: below this on every joint the arm counts as still (payload is valid)
N_ARM = len(ARM_JOINTS) - 1  # joints that carry the payload (the gripper joint carries the squeeze)
JAW_KINDS = ("jaw_fixed", "jaw_moving")


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


@dataclass
class SensorFrame:
    t: float
    q: np.ndarray  # rad, ARM_JOINTS order
    qd: np.ndarray  # rad/s
    load: np.ndarray  # signed fraction of stall torque
    current_a: np.ndarray  # A
    jaw_force_n: np.ndarray  # (fixed, moving) normal force on the jaw pads
    link_force_n: dict[str, float]  # robot body -> contact normal force from anything but the robot
    wrist_acc: np.ndarray  # m/s^2, gripper frame, includes gravity (reads +g up when still)
    wrist_gyro: np.ndarray  # rad/s, gripper frame
    torque: np.ndarray = field(default_factory=lambda: np.zeros(len(ARM_JOINTS)))  # N*m, from load
    payload_g: float = 0.0  # meaningful only when ``static``
    kinetic_j: float = 0.0
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))  # kg*m/s, arm linear momentum
    static: bool = False


class SelfModel:
    """The robot's model of its own arm: nominal design, no scene, no injected faults."""

    def __init__(self, design: Design):
        sim = build_model(design)
        self.model, self.data = sim.model, sim.data
        self.qadr, self.dadr = sim.arm_qadr, sim.arm_dadr
        self.site = sim.site_id
        self.root = self.model.body("base").id
        self.stall = np.array([STALL_TORQUE[float(design.arm.servo_voltage.get(j, 7.4))] for j in ARM_JOINTS])

    def _set(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0.0
        if qd is not None:
            self.data.qvel[self.dadr] = qd
        mujoco.mj_forward(self.model, self.data)

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Joint torques that hold the bare arm still at ``q``."""
        self._set(q)
        return self.data.qfrc_bias[self.dadr].copy()

    def tool_jacobian(self, q: np.ndarray) -> np.ndarray:
        """3 x n_joints positional Jacobian of the tool site."""
        self._set(q)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site)
        return jacp[:, self.dadr]

    def energy_momentum(self, q: np.ndarray, qd: np.ndarray) -> tuple[float, np.ndarray]:
        self._set(q, qd)
        mq = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, mq, self.data.qvel)
        mujoco.mj_subtreeVel(self.model, self.data)
        momentum = self.model.body_subtreemass[self.root] * self.data.subtree_linvel[self.root]
        return 0.5 * float(self.data.qvel @ mq), momentum.copy()

    def payload_g(self, q: np.ndarray, torque: np.ndarray) -> float:
        """Grams hanging at the tool point that explain the torque the arm is not using on itself."""
        r = (torque - self.gravity(q))[:N_ARM]
        jac = self.tool_jacobian(q)[:, :N_ARM]
        w, *_ = np.linalg.lstsq(jac.T, r, rcond=None)
        return float(w[2] / G * 1000.0)


class Sensors:
    """Reads a ``Sim`` the way the robot would: noisy, quantized, and through its own model."""

    def __init__(self, sim: Sim, seed: int | None = 0, self_model: SelfModel | None = None):
        self.sim = sim
        self.cfg = load_config()
        self.ch = self.cfg["channels"]
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.self_model = self_model or SelfModel(sim.design)
        servo = self.cfg["servo"]
        volts = [float(sim.design.arm.servo_voltage.get(j, 7.4)) for j in ARM_JOINTS]
        self.stall = np.array([STALL_TORQUE[v] for v in volts])
        self.stall_current = np.array([float(servo["stall_current_a"][v]) for v in volts])
        self.idle_current = float(servo["idle_current_a"])
        self.imu_site = sim.model.site(MJCF_TOOL_SITE).id
        self.robot_bodies = {int(sim.model.geom_bodyid[g]) for g in sim.geom_kind}

    def _measure(self, name: str, x: np.ndarray) -> np.ndarray:
        spec = self.ch[name]
        x = np.asarray(x, dtype=float)
        if self.rng is not None and spec["noise"] > 0:
            x = x + self.rng.normal(0.0, spec["noise"], x.shape)
        x = np.round(x / spec["resolution"]) * spec["resolution"]
        return np.maximum(x, 0.0) if name.endswith("_force_n") else x  # pads and skin cannot read negative

    def _contacts(self) -> tuple[np.ndarray, dict[str, float]]:
        m, d = self.sim.model, self.sim.data
        jaw = np.zeros(2)
        links: dict[str, float] = {}
        fr = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            k1, k2 = self.sim.geom_kind.get(int(c.geom1)), self.sim.geom_kind.get(int(c.geom2))
            if (k1 is None) == (k2 is None):  # robot-robot or scene-scene: nothing the skin feels
                continue
            g = int(c.geom1) if k1 is not None else int(c.geom2)
            kind = k1 if k1 is not None else k2
            mujoco.mj_contactForce(m, d, i, fr)
            f = abs(float(fr[0]))
            if kind in JAW_KINDS:
                jaw[JAW_KINDS.index(kind)] += f
            body = m.body(int(m.geom_bodyid[g])).name
            links[body] = links.get(body, 0.0) + f
        return jaw, links

    def _imu(self) -> tuple[np.ndarray, np.ndarray]:
        m, d = self.sim.model, self.sim.data
        mujoco.mj_rnePostConstraint(m, d)  # fills cacc; reads state, changes nothing
        acc, vel = np.zeros(6), np.zeros(6)
        mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_SITE, self.imu_site, acc, 1)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.imu_site, vel, 1)
        return acc[3:], vel[:3]

    def read(self) -> SensorFrame:
        sim = self.sim
        d = sim.data
        q_true = d.qpos[sim.arm_qadr].copy()
        qd_true = d.qvel[sim.arm_dadr].copy()
        tau_true = np.array([d.actuator_force[sim.model.actuator(j).id] for j in ARM_JOINTS])
        load = self._measure("load", np.clip(tau_true / self.stall, -1.0, 1.0))
        current = self.idle_current + np.abs(load) * (self.stall_current - self.idle_current)
        jaw, links = self._contacts()
        acc, gyro = self._imu()
        q = self._measure("q", q_true)
        qd = self._measure("qd", qd_true)
        torque = load * self.stall
        kinetic, momentum = self.self_model.energy_momentum(q, qd)
        static = bool(np.all(np.abs(qd[:N_ARM]) < STATIC_QD))
        return SensorFrame(
            t=float(d.time), q=q, qd=qd, load=load, current_a=self._measure("current_a", current),
            jaw_force_n=self._measure("jaw_force_n", jaw),
            link_force_n={k: float(self._measure("link_force_n", np.array(v))) for k, v in sorted(links.items())},
            wrist_acc=self._measure("wrist_acc", acc), wrist_gyro=self._measure("wrist_gyro", gyro),
            torque=torque, payload_g=self.self_model.payload_g(q, torque) if static else 0.0,
            kinetic_j=kinetic, momentum=momentum, static=static,
        )
