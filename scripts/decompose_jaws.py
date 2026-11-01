"""Convex-decompose the stock jaw meshes with CoACD (run once; output is vendored).

MuJoCo collides meshes by their convex hull.  The fixed jaw is part of the wrist-roll follower
bracket, whose hull is ~6 cm across, so presses near other geometry collide with empty space.
This writes ``assets/so101/collision/<mesh>.npz`` with arrays ``part_<i>`` (vertices, mesh frame,
meters), which ``sim/build_model.py`` uses in place of the hull geoms (docs/decisions.md D-013).
"""

from __future__ import annotations

import sys
from pathlib import Path

import coacd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
MESHES = ["wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"]


def main(threshold: float = 0.08) -> None:
    out_dir = ROOT / "assets" / "so101" / "collision"
    out_dir.mkdir(parents=True, exist_ok=True)
    coacd.set_log_level("error")
    for name in MESHES:
        mesh = trimesh.load(ROOT / "assets" / "so101" / "assets" / f"{name}.stl", force="mesh")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=16, seed=0)
        arrays = {f"part_{i}": np.asarray(v, dtype=np.float64) for i, (v, _) in enumerate(parts)}
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        print(name, len(arrays), "parts")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.08)
