"""Measure the stock reach grid and the Appendix C fixture points (spec 7.3, Appendix C).

Writes ``tests/fixtures/reach_stock.npz`` and prints the Appendix C table so changes in placo or
the model can be re-recorded in docs/decisions.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from sea.contracts import HardLimits
from sea.control.frames import STOCK_URDF, approach_axis
from sea.control.ik import IK
from sea.control.reach import REACH_DEG, REACH_MM, ReachGrid

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    limits = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
    ik = IK(STOCK_URDF)
    points = yaml.safe_load((ROOT / "tests" / "fixtures" / "ik_points.yaml").read_text())["points"]
    for p in points:
        pos = np.array([p["x"], p.get("y", 0.0), p["z"]])
        _, res, ang = ik.solve(pos, approach_axis(p["approach"], pos), tol_mm=REACH_MM, tol_deg=REACH_DEG)
        got = res < REACH_MM and ang < REACH_DEG
        print(f"{p['x']:.2f} {p.get('y', 0.0):+.2f} {p['z']:+.2f} {p['approach']:>10s} expected={p['expected']:<11s} "
              f"got={'reachable' if got else 'unreachable'} res={res:.2f}mm ang={ang:.2f}deg")
    t = time.time()
    grid = ReachGrid.compute(ik, limits)
    grid.save(ROOT / "tests" / "fixtures" / "reach_stock.npz")
    print(f"grid {grid.occ.shape} reachable fraction per bin {grid.occ.mean(axis=(1, 2, 3)).round(3)} "
          f"in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
