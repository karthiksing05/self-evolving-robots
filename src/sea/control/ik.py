"""Placo inverse kinematics on the (generated) URDF (spec 8.1).

``IK(urdf_abs_path).solve(pos, axis, tool_tip, seeds=5)`` aligns the tool-frame z axis with
``axis`` (an axis-align task, not a full 6-D frame task) and places the tool tip at ``pos``.
A mounted tool moves the tool point: the tip transform (Appendix B, relative to
``gripper_frame_link``) becomes a fixed ``tool_tip`` link in a URDF variant, so placo solves for
the tip directly.  ``track`` is the warm-started variant the motor primitives call at 50 Hz; it can
also align the tool x axis (roll) for hooks, blades and knob turning.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import yaml

from sea.contracts import ARM_JOINTS, Pose
from sea.control.frames import ROOT, URDF_TOOL_FRAME, angle_deg, quat_wxyz_to_mat

TIP_FRAME = "tool_tip"
SEEDS_PATH = ROOT / "configs" / "ik_seeds.yaml"
CACHE_DIR = ROOT / ".cache" / "urdf"

CONVERGED_MM = 1.0
CONVERGED_DEG = 3.0
MAX_ITERS = 100
# Position errors are in metres and axis errors in radians; at equal weights the 5-DOF arm trades
# ~10 mm of position for ~1 deg of an off-plane approach axis.  30 keeps every Appendix C point and
# makes position dominate (docs/decisions.md D-025).
POS_WEIGHT = 30.0
# The tool point sits off the wrist-roll axis, so with wrist flex at its limit a position-dominant
# tracker can spin the roll to fix millimetres; when no roll target is given, a soft prior keeps the
# roll where it is (D-025).
ROLL_HOLD_WEIGHT = 0.05


def load_seeds(path: Path = SEEDS_PATH) -> list[np.ndarray]:
    data = yaml.safe_load(path.read_text())
    return [np.asarray(s, dtype=float) for s in data["seeds"]]


def tip_matrix(tool_tip: Pose | np.ndarray | None) -> np.ndarray:
    if tool_tip is None:
        return np.eye(4)
    if isinstance(tool_tip, Pose):
        t = np.eye(4)
        t[:3, :3] = quat_wxyz_to_mat(tool_tip.quat_wxyz)
        t[:3, 3] = tool_tip.pos.as_list()
        return t
    return np.asarray(tool_tip, dtype=float)


def _rpy(r: np.ndarray) -> tuple[float, float, float]:
    sy = -r[2, 0]
    pitch = float(np.arcsin(np.clip(sy, -1.0, 1.0)))
    if abs(sy) < 0.999999:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:  # gimbal lock
        roll = 0.0
        yaw = float(np.arctan2(-r[0, 1], r[1, 1]))
    return roll, pitch, yaw


def urdf_with_tip(urdf_abs_path: Path, tip: np.ndarray, cache_dir: Path = CACHE_DIR) -> Path:
    """Write (once) a URDF variant with a fixed ``tool_tip`` link under the tool frame.

    Mesh paths are rewritten to absolute paths so the variant can live in the cache directory.
    """
    src = Path(urdf_abs_path)
    text = src.read_text()
    text = re.sub(r'filename="(?!/)([^"]+)"', lambda m: f'filename="{(src.parent / m.group(1)).resolve()}"', text)
    roll, pitch, yaw = _rpy(tip[:3, :3])
    xyz = " ".join(f"{v:.9g}" for v in tip[:3, 3])
    extra = (
        f'  <link name="{TIP_FRAME}"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1e-9"/>'
        '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>\n'
        f'  <joint name="{TIP_FRAME}_joint" type="fixed"><origin xyz="{xyz}" rpy="{roll:.9g} {pitch:.9g} {yaw:.9g}"/>'
        f'<parent link="{URDF_TOOL_FRAME}"/><child link="{TIP_FRAME}"/><axis xyz="0 0 0"/></joint>\n'
    )
    text = text.replace("</robot>", extra + "</robot>")
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{src.stem}_{key}.urdf"
    if not out.exists():
        out.write_text(text)
    return out


class IK:
    """Placo IK for one URDF and one tool tip."""

    def __init__(self, urdf_abs_path: str | Path, seeds: list[np.ndarray] | None = None,
                 tip: Pose | np.ndarray | None = None):
        path = Path(urdf_abs_path)
        if not path.is_absolute():
            raise ValueError("placo needs an absolute URDF path (spec 5.1)")
        self.urdf = path
        self.seeds = seeds if seeds is not None else load_seeds()
        self.tip = tip_matrix(tip)
        self._variants: dict[str, IK] = {}
        if np.allclose(self.tip, np.eye(4)):
            self.frame = URDF_TOOL_FRAME
            self._urdf_used = path
        else:
            self.frame = TIP_FRAME
            self._urdf_used = urdf_with_tip(path, self.tip)
        self._setup()

    # ---------------------------------------------------------------- placo plumbing
    def _setup(self) -> None:
        import placo

        self.robot = placo.RobotWrapper(str(self._urdf_used), placo.Flags.ignore_collisions)
        names = list(self.robot.joint_names())
        self._idx = [7 + names.index(j) for j in ARM_JOINTS]
        self.limits = np.array([self.robot.get_joint_limits(j) for j in ARM_JOINTS])
        s = placo.KinematicsSolver(self.robot)
        s.mask_fbase(True)
        s.enable_joint_limits(True)
        s.mask_dof("gripper")
        self.pos_task = s.add_position_task(self.frame, np.zeros(3))
        self.pos_task.configure("pos", "soft", POS_WEIGHT)
        self.axis_task = s.add_axisalign_task(self.frame, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
        self.axis_task.configure("axis", "soft", 1.0)
        self.roll_task = s.add_axisalign_task(self.frame, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.roll_task.configure("roll", "soft", 0.0)
        self.roll_hold = s.add_joints_task()
        self.roll_hold.set_joints({"wrist_roll": 0.0})
        self.roll_hold.configure("roll_hold", "soft", 0.0)
        s.add_regularization_task(1e-3)
        self.solver = s

    def set_q(self, q: np.ndarray) -> None:
        state = self.robot.state.q.copy()
        state[self._idx] = np.asarray(q, dtype=float)[: len(ARM_JOINTS)]
        self.robot.state.q = state
        self.robot.update_kinematics()

    def get_q(self) -> np.ndarray:
        return np.asarray(self.robot.state.q)[self._idx].copy()

    def fk(self, q: np.ndarray | None = None) -> np.ndarray:
        """World transform of the tool point (tip) for joint vector ``q`` (ARM_JOINTS order)."""
        if q is not None:
            self.set_q(q)
        return np.asarray(self.robot.get_T_world_frame(self.frame))

    def frame_fk(self, frame: str, q: np.ndarray | None = None) -> np.ndarray:
        if q is not None:
            self.set_q(q)
        return np.asarray(self.robot.get_T_world_frame(frame))

    def _error(self, pos: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        t = np.asarray(self.robot.get_T_world_frame(self.frame))
        return float(np.linalg.norm(t[:3, 3] - pos) * 1000.0), angle_deg(t[:3, 2], axis)

    def _iterate(self, pos: np.ndarray, axis: np.ndarray, max_iter: int, tol_mm: float,
                 tol_deg: float) -> tuple[float, float]:
        res, ang = self._error(pos, axis)
        for _ in range(max_iter):
            if res < tol_mm and ang < tol_deg:
                break
            self.solver.solve(True)
            self.robot.update_kinematics()
            res, ang = self._error(pos, axis)
        return res, ang

    def _set_targets(self, pos: np.ndarray, axis: np.ndarray, roll_axis: np.ndarray | None,
                     roll_weight: float) -> None:
        self.roll_hold.configure("roll_hold", "soft", 0.0)  # track() re-enables it when no roll is asked
        self.pos_task.target_world = np.asarray(pos, dtype=float)
        self.axis_task.targetAxis_world = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        if roll_axis is None:
            self.roll_task.configure("roll", "soft", 0.0)
        else:
            self.roll_task.targetAxis_world = np.asarray(roll_axis, dtype=float) / np.linalg.norm(roll_axis)
            self.roll_task.configure("roll", "soft", roll_weight)

    # ---------------------------------------------------------------- public API
    def for_tip(self, tool_tip: Pose | np.ndarray | None) -> IK:
        tip = tip_matrix(tool_tip)
        if np.allclose(tip, self.tip):
            return self
        key = hashlib.sha256(np.round(tip, 9).tobytes()).hexdigest()
        if key not in self._variants:
            self._variants[key] = IK(self.urdf, self.seeds, tip)
        return self._variants[key]

    def solve(self, pos, axis, tool_tip: Pose | np.ndarray | None = None, seeds: int = 5,
              tol_mm: float = CONVERGED_MM, tol_deg: float = CONVERGED_DEG,
              max_iter: int = MAX_ITERS) -> tuple[np.ndarray, float, float]:
        """Solve from up to ``seeds`` seeds (zero pose first); first converged wins.

        Returns ``(q, residual_mm, axis_err_deg)``; when nothing converges, the best attempt.
        """
        ik = self.for_tip(tool_tip)
        pos = np.asarray(pos, dtype=float)
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        ik._set_targets(pos, axis, None, 0.0)
        candidates = [np.zeros(len(ARM_JOINTS))] + list(ik.seeds)
        best: tuple[np.ndarray, float, float] | None = None
        for seed in candidates[: max(1, seeds)]:
            ik.set_q(seed)
            res, ang = ik._iterate(pos, axis, max_iter, tol_mm, tol_deg)
            q = ik.get_q()
            if res < tol_mm and ang < tol_deg:
                return q, res, ang
            if best is None or res + ang < best[1] + best[2]:
                best = (q, res, ang)
        assert best is not None
        return best

    def track(self, pos, axis, q0: np.ndarray, roll_axis: np.ndarray | None = None,
              roll_weight: float = 1.0, max_iter: int = 12, tol_mm: float = 0.2,
              tol_deg: float = 1.0) -> tuple[np.ndarray, float, float]:
        """Warm-started solve from ``q0`` for one control tick (optionally aligning tool x)."""
        pos = np.asarray(pos, dtype=float)
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        self._set_targets(pos, axis, roll_axis, roll_weight)
        if roll_axis is None:
            self.roll_hold.set_joints({"wrist_roll": float(np.asarray(q0)[ARM_JOINTS.index("wrist_roll")])})
            self.roll_hold.configure("roll_hold", "soft", ROLL_HOLD_WEIGHT)
        self.set_q(q0)
        res, ang = self._iterate(pos, axis, max_iter, tol_mm, tol_deg)
        if roll_axis is not None:  # roll is redundant for pos+axis; keep iterating until aligned
            for _ in range(max_iter):
                x = np.asarray(self.robot.get_T_world_frame(self.frame))[:3, 0]
                if angle_deg(x, roll_axis) < 2.0:
                    break
                self.solver.solve(True)
                self.robot.update_kinematics()
            res, ang = self._error(pos, axis)
        q = self.get_q()
        q[-1] = q0[-1] if len(q0) == len(ARM_JOINTS) else 0.0  # gripper is not solved
        return q, res, ang
