"""Frame conventions shared by IK (URDF/placo) and simulation (MJCF/MuJoCo).

Spec 5.1: the URDF tool frame ``gripper_frame_link`` uses +z as the approach direction; the MJCF
site ``gripperframe`` sits at the same point but uses +x as the approach direction.  Body names
differ between the two files.  Everything else in the repo expresses tool geometry in the URDF
convention ("tool frame": origin at the stock jaw tip, +z = approach).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets" / "so101"
STOCK_URDF = ASSETS / "so101_new_calib.urdf"
STOCK_MJCF = ASSETS / "so101_new_calib.xml"

URDF_TOOL_FRAME = "gripper_frame_link"
MJCF_TOOL_SITE = "gripperframe"

# MJCF body -> URDF link
BODY_TO_LINK: dict[str, str] = {
    "base": "base_link",
    "shoulder": "shoulder_link",
    "upper_arm": "upper_arm_link",
    "lower_arm": "lower_arm_link",
    "wrist": "wrist_link",
    "gripper": "gripper_link",
    "moving_jaw_so101_v1": "moving_jaw_so101_v1_link",
}
LINK_TO_BODY = {v: k for k, v in BODY_TO_LINK.items()}

# Columns are the URDF tool-frame axes expressed in the MJCF site frame:
# urdf x = -site z, urdf y = site y, urdf z (approach) = site x.
R_SITE_URDF = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
R_URDF_SITE = R_SITE_URDF.T

APPROACH_DIRS: dict[str, np.ndarray] = {
    "down": np.array([0.0, 0.0, -1.0]),
    "45": np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5)]),
    "horizontal": np.array([1.0, 0.0, 0.0]),
}


def approach_axis(name: str, pos: np.ndarray | None = None) -> np.ndarray:
    """Unit approach direction for a named bin.  Horizontal components point radially away
    from the base through ``pos`` (so lateral targets keep the arm-plane geometry)."""
    base = APPROACH_DIRS[name]
    if pos is None or name == "down":
        return base.copy()
    r = np.hypot(pos[0], pos[1])
    c, s = (1.0, 0.0) if r < 1e-9 else (pos[0] / r, pos[1] / r)
    return np.array([base[0] * c, base[0] * s, base[2]])


def nearest_bin(axis: np.ndarray) -> str:
    """Nearest of the three approach bins by elevation angle (Section 7.3)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    elev = np.degrees(np.arcsin(np.clip(-a[2], -1.0, 1.0)))  # 90 = straight down
    return min(("down", 90.0), ("45", 45.0), ("horizontal", 0.0), key=lambda kv: abs(kv[1] - elev))[0]


def site_to_urdf_rot(site_xmat: np.ndarray) -> np.ndarray:
    """World rotation of the URDF tool frame given the MJCF site's world rotation."""
    return np.asarray(site_xmat).reshape(3, 3) @ R_SITE_URDF


def quat_wxyz_to_mat(q: np.ndarray | tuple[float, ...]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def transform(pos: np.ndarray, rot: np.ndarray | None = None) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = pos
    if rot is not None:
        t[:3, :3] = rot
    return t


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def perpendicular(axis: np.ndarray) -> np.ndarray:
    """A deterministic unit vector perpendicular to ``axis``."""
    a = np.asarray(axis, dtype=float)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(a, ref)
    return p / np.linalg.norm(p)
