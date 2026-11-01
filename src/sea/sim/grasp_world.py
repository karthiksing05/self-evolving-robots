"""The grasp world (docs/gripper-experiment.md section 1): one object on a table per panel.

Each object is a ``ControlSpec`` of class ``grasp`` whose coordinate is its height above rest, so
the panel pipeline (typecheck, System 1, verdicts, memory, the loop) runs on it unchanged: a lift
of ``travel`` held for ``hold_s`` passes; a lift that falls back reads ``wrong_behavior``; a grip
that never lifts reads ``stuck``.  Judge tier, like ``panels.py``: this is the task distribution.

- ``grasp_panels(split, n, seed)``: phase A objects (inside the stock envelope V1 measured).
- ``phase_b_panels()``: the fixed incompatible objects, one known failure each.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from sea.contracts import GRASP_CLASS, ControlSpec, ObjectShape, PanelSpec, Pose, Rules, Vec3

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "grasp.yaml"
OBJECT_ID = "obj"
BOX_DEPTH_MM = 30.0  # box extent along the approach; the width is across the jaws


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def table_half() -> tuple[float, float]:
    t = load_config()["table"]
    return float(t["half_x"]), float(t["half_y"])


def _rules() -> Rules:
    task = load_config()["task"]
    return Rules(banned_contacts=[], force_cap_by_class={GRASP_CLASS: float(task["squeeze_cap_n"])},
                 time_limit_s=float(task["time_limit_s"]))


def object_spec(shape: ObjectShape, width_mm: float, height_mm: float, mass_g: float, friction: float,
                dx: float = 0.0, dy: float = 0.0) -> ControlSpec:
    task = load_config()["task"]
    if shape == "sphere":
        height_mm = width_mm
    return ControlSpec(id=OBJECT_ID, cls=GRASP_CLASS, pose_panel=Pose(pos=Vec3(x=dx, y=dy, z=0.0)),
                       size_mm=(round(width_mm, 1), round(height_mm, 1)), stiffness=0.0,
                       travel=float(task["lift_mm"]) / 1000, friction=round(friction, 3),
                       success=f"held_for($target, {float(task['hold_s'])})", shape=shape,
                       mass_g=round(mass_g, 1))


def table_panel(pid: str, split: str, seed: int, obj: ControlSpec) -> PanelSpec:
    t = load_config()["table"]
    return PanelSpec(id=pid, seed=seed, split=split, pose_world=Pose(pos=Vec3(x=t["x"], y=t["y"], z=t["z"])),  # type: ignore[arg-type]
                     tilt_rad=0.0, controls=[obj], rules=_rules())


def _rng(split: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"grasp:{split}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def grasp_panels(split: str, n: int, seed: int = 0) -> list[PanelSpec]:
    """``n`` phase-A tables for ``split`` (train, val or test), alternating cylinders and boxes."""
    cfg = load_config()
    rng = _rng(split, seed)
    shapes = sorted(cfg["phase_a"])
    out = []
    for i in range(n):
        shape = shapes[i % len(shapes)]
        r = cfg["phase_a"][shape]
        obj = object_spec(shape, rng.uniform(*r["width_mm"]), rng.uniform(*r["height_mm"]), rng.uniform(*r["mass_g"]),
                          rng.uniform(*cfg["friction"]), rng.uniform(*cfg["placement"]["dx"]),
                          rng.uniform(*cfg["placement"]["dy"]))
        out.append(table_panel(f"g{split[:2]}{i:03d}", split, seed, obj))
    return out


def phase_b_panels() -> list[PanelSpec]:
    """The incompatible objects, centred on the table at friction 0.8."""
    return [table_panel(f"gb_{o['id']}", "phase_b", 0, object_spec(o["shape"], o["width_mm"], o["height_mm"],
                                                                   o["mass_g"], 0.8))
            for o in load_config()["phase_b"]]
