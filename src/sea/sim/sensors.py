"""What the robot can feel about itself and its contacts (docs/gripper-experiment.md "Sensing").

``Sensors.read`` returns one ``SensorFrame``: servo telemetry (position, speed, signed load,
current), jaw pressure pads, per-link contact force, and a wrist IMU, each with the noise and
quantization of the part named in ``configs/sensors.yaml``.  Derived signals (joint torque,
payload, kinetic energy, momentum) are computed from those noisy channels and the robot's *own*
nominal model (``SelfModel``), never from ground truth, so a fault injected into the simulated
arm shows up as a residual the robot can notice.

The verifier never reads a ``SensorFrame``: verdicts come from ground-truth sim state.  Sensor
noise can therefore change what the robot decides, but never what counts as a pass.

Which channels exist depends on the body: ``Design.sensors`` mounts the toggleable parts in the
config's ``parts`` table, and an unmounted part's channels read ``None``.  ``manifest`` is what the
agent is told it can feel (D-033).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sea.contracts import ARM_JOINTS, Design, SensorTrace
from sea.control.frames import MJCF_TOOL_SITE
from sea.sim.build_model import STALL_TORQUE, Sim, build_model

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "sensors.yaml"
G = 9.81
STATIC_QD = 0.05  # rad/s: below this on every joint the arm counts as still (payload is valid)
IMPACT_ACC = 10.0  # m/s^2 beyond gravity (1 g): no normal move reaches it; a collision does (D-036)
STALL_LOAD = 0.95  # fraction of stall torque: a joint this loaded is at its limit
STALL_LAG_RAD = 0.05  # a stalled joint this far behind its command is losing, not just straining
LOAD_ABORT_S = 0.5  # stalled and losing this long while holding something: the payload is too heavy
LOG_EVERY = 5  # ticks between logged samples: 10 Hz at 50 Hz ticks
SETTLE_TICKS = 5  # the first ticks after a reset are the servos settling, not the robot feeling anything
PAD_GRIP_N, PAD_LOST_N = 5.0, 1.0  # pads summed: holding something / holding nothing
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
    jaw_force_n: np.ndarray | None  # (fixed, moving) normal force on the jaw pads; None if not mounted
    link_force_n: dict[str, float] | None  # robot body -> contact force from anything but the robot
    wrist_acc: np.ndarray  # m/s^2, gripper frame, includes gravity (reads +g up when still)
    wrist_gyro: np.ndarray  # rad/s, gripper frame
    torque: np.ndarray = field(default_factory=lambda: np.zeros(len(ARM_JOINTS)))  # N*m, from load
    payload_g: float = 0.0  # meaningful only when ``static``
    kinetic_j: float = 0.0
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))  # kg*m/s, arm linear momentum
    static: bool = False


def mounted_parts(design: Design) -> set[str]:
    """Sensor parts on this body: every part that cannot be toggled, plus the toggled-on ones."""
    return {name for name, part in load_config()["parts"].items()
            if not part["toggle"] or getattr(design.sensors, name)}


def manifest(design: Design) -> list[dict]:
    """What the agent is told it can feel: one entry per part, mounted or not, with its channels."""
    cfg = load_config()
    on = mounted_parts(design)
    return [{"part": name, "mounted": name in on, **{k: part[k] for k in ("priority", "toggle", "bench", "mass_g",
                                                                         "mount", "tells")},
             "channels": {c: {k: cfg["channels"][c][k] for k in ("unit", "resolution", "noise")}
                          for c in part["channels"]}}
            for name, part in cfg["parts"].items()]


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


_SELF_MODELS: dict[str, SelfModel] = {}


def self_model_for(design: Design) -> SelfModel:
    """One nominal self-model per design (building one compiles a model, about 0.1 s)."""
    key = design.model_dump_json(exclude={"id", "parent"})
    if key not in _SELF_MODELS:
        _SELF_MODELS[key] = SelfModel(design)
    return _SELF_MODELS[key]


def dynamic_acc(frame: SensorFrame) -> float:
    """Wrist acceleration beyond gravity, orientation-free: | |a| - g |."""
    return abs(float(np.linalg.norm(frame.wrist_acc)) - G)


class SensorLog:
    """Turns one ``SensorFrame`` per tick into a ``SensorTrace``: 10 Hz samples plus features that
    every tick contributes to (peaks, impacts, a lost grip)."""

    def __init__(self, parts: set[str]):
        self.out = SensorTrace(parts=sorted(parts))
        self.n = 0
        self.gripping = False
        self.last_impact = -1
        self.stall_run = 0  # consecutive ticks with an arm joint at STALL_LOAD or more
        self.stall_joint = ""

    def add(self, t: float, f: SensorFrame, grip_closed: bool, carrying: bool = False, tick_s: float = 0.02) -> None:
        """``grip_closed``: the jaws are commanded shut and not opening (a pad drop then means a slip).
        ``carrying``: the program has lifted what it holds, so a still arm's payload is the object's weight."""
        o, self.n = self.out, self.n + 1
        if self.n <= SETTLE_TICKS:
            return
        acc = dynamic_acc(f)
        arm = np.abs(f.load[:N_ARM])
        if acc > o.peak_acc:
            o.peak_acc = round(acc, 3)
        if acc > IMPACT_ACC and self.n - self.last_impact > LOG_EVERY:
            o.impacts += 1
            self.last_impact = self.n
        if float(arm.max()) > o.peak_load:
            o.peak_load = round(float(arm.max()), 3)
            o.peak_load_joint = ARM_JOINTS[int(arm.argmax())]
        if arm.max() >= STALL_LOAD:
            self.stall_run += 1
            self.stall_joint = self.stall_joint or ARM_JOINTS[int(arm.argmax())]
            if self.stall_run * tick_s > o.stall_s:
                o.stall_s, o.stall_joint = round(self.stall_run * tick_s, 3), self.stall_joint
        else:
            self.stall_run, self.stall_joint = 0, ""
        pads = None if f.jaw_force_n is None else float(f.jaw_force_n.sum())
        if pads is not None:
            o.peak_pads_n = round(max(o.peak_pads_n or 0.0, pads), 2)
            if pads >= PAD_GRIP_N:
                self.gripping = True
            elif self.gripping and pads < PAD_LOST_N and grip_closed and o.pad_drop_t is None:
                o.pad_drop_t = round(t, 3)
        if f.static and carrying:
            o.payload_held_g = round(f.payload_g, 1)
        if self.n % LOG_EVERY == 0:
            o.t.append(round(t, 3))
            o.acc_dyn.append(round(acc, 3))
            o.load_max.append(round(float(arm.max()), 3))
            o.grip_load.append(round(abs(float(f.load[-1])), 3))
            if pads is not None:
                o.pads_n.append(round(pads, 2))
            o.payload_g.append(round(f.payload_g, 1) if f.static else None)


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
        self.mounted = mounted_parts(sim.design)
        self._prev: tuple[float, np.ndarray] | None = None  # (time, wrist velocity) at the last reading

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
        """Accelerometer (specific force, reads +g up at rest) and gyro, in the gripper frame.

        Between readings the accelerometer reports the mean acceleration since the last one, as an
        IMU's low-pass filter does; sampling the 500 Hz contact chatter instead reads several m/s^2
        of vibration while the jaws hold at stall.  A first reading, or one after a reset, is instantaneous."""
        m, d = self.sim.model, self.sim.data
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.imu_site, vel, 0)
        rot = d.site_xmat[self.imu_site].reshape(3, 3)
        t = float(d.time)
        if self._prev is not None and t - self._prev[0] > 1e-6:
            acc = rot.T @ ((vel[3:] - self._prev[1]) / (t - self._prev[0]) - m.opt.gravity)
        else:
            mujoco.mj_rnePostConstraint(m, d)  # fills cacc; reads state, changes nothing
            acc6 = np.zeros(6)
            mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_SITE, self.imu_site, acc6, 1)
            acc = acc6[3:]
        self._prev = (t, vel[3:].copy())
        return acc, rot.T @ vel[:3]

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
            jaw_force_n=self._measure("jaw_force_n", jaw) if "jaw_pads" in self.mounted else None,
            link_force_n={k: float(self._measure("link_force_n", np.array(v))) for k, v in sorted(links.items())}
            if "link_skin" in self.mounted else None,
            wrist_acc=self._measure("wrist_acc", acc), wrist_gyro=self._measure("wrist_gyro", gyro),
            torque=torque, payload_g=self.self_model.payload_g(q, torque) if static else 0.0,
            kinetic_j=kinetic, momentum=momentum, static=static,
        )
