"""Reach envelope (spec 7.3): a 9x9x7 occupancy grid per approach bin, cached by design hash.

For each grid point over the workspace box (origin ``base_link``) and each of three approach
directions (down, 45 deg, horizontal; non-vertical bins point radially away from the base), IK
runs from up to 5 seeds and stops at the first that converges (residual < 2 mm, axis error < 5 deg,
<= 100 iterations).  Lookup is trilinear interpolation of occupancy in the nearest bin;
reachable if >= 0.5.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from sea.contracts import HardLimits
from sea.control.frames import approach_axis, nearest_bin
from sea.control.ik import IK, POS_WEIGHT

GRID_SHAPE = (9, 9, 7)
DIRS = ("down", "45", "horizontal")
REACH_MM = 2.0
REACH_DEG = 5.0


class ReachGrid:
    def __init__(self, occ: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        self.occ = np.asarray(occ, dtype=np.float32)  # (3, nx, ny, nz)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.occ.shape[1:])  # type: ignore[return-value]

    def axes(self) -> list[np.ndarray]:
        return [np.linspace(self.lo[i], self.hi[i], n) for i, n in enumerate(self.shape)]

    @classmethod
    def compute(cls, ik: IK, limits: HardLimits, tip=None, seeds: int = 5, shape=GRID_SHAPE) -> ReachGrid:
        lo = np.array(limits.workspace_box[0].as_list())
        hi = np.array(limits.workspace_box[1].as_list())
        solver = ik.for_tip(tip)
        occ = np.zeros((len(DIRS),) + tuple(shape), dtype=np.uint8)
        xs, ys, zs = (np.linspace(lo[i], hi[i], n) for i, n in enumerate(shape))
        for di, name in enumerate(DIRS):
            for ix, x in enumerate(xs):
                for iy, y in enumerate(ys):
                    for iz, z in enumerate(zs):
                        p = np.array([x, y, z])
                        _, res, ang = solver.solve(p, approach_axis(name, p), seeds=seeds, tol_mm=REACH_MM,
                                                   tol_deg=REACH_DEG)
                        occ[di, ix, iy, iz] = int(res < REACH_MM and ang < REACH_DEG)
        return cls(occ, lo, hi)

    def occupancy(self, pos: np.ndarray, bin_name: str) -> float:
        g = self.occ[DIRS.index(bin_name)]
        f = (np.asarray(pos, dtype=float) - self.lo) / (self.hi - self.lo) * (np.array(self.shape) - 1)
        if np.any(f < -1e-9) or np.any(f > np.array(self.shape) - 1 + 1e-9):
            return 0.0
        i0 = np.clip(np.floor(f).astype(int), 0, np.array(self.shape) - 2)
        t = f - i0
        val = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[0] if dx else 1 - t[0]) * (t[1] if dy else 1 - t[1]) * (t[2] if dz else 1 - t[2])
                    val += w * float(g[i0[0] + dx, i0[1] + dy, i0[2] + dz])
        return val

    def reachable(self, pos: np.ndarray, axis: np.ndarray) -> bool:
        """Is ``pos`` reachable with approach ``axis`` (nearest of the three bins)?"""
        return self.occupancy(pos, nearest_bin(axis)) >= 0.5

    def save(self, path: Path) -> None:
        np.savez_compressed(path, occ=self.occ.astype(np.uint8), lo=self.lo, hi=self.hi)

    @classmethod
    def load(cls, path: Path) -> ReachGrid:
        data = np.load(path)
        return cls(data["occ"], data["lo"], data["hi"])


def design_key(urdf: Path, tip: np.ndarray | None, limits: HardLimits) -> str:
    h = hashlib.sha256()
    h.update(Path(urdf).read_bytes())
    h.update(np.round(np.eye(4) if tip is None else np.asarray(tip), 9).tobytes())
    h.update(limits.model_dump_json().encode())
    h.update(f"pos_weight={POS_WEIGHT}".encode())  # solver weights change what counts as reachable
    return h.hexdigest()[:16]


def reach_for(ik: IK, limits: HardLimits, tip: np.ndarray | None, cache_dir: Path) -> ReachGrid:
    """Cached grid for one (URDF, tool tip, limits) combination."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"reach_{design_key(ik.urdf, tip, limits)}.npz"
    if path.exists():
        return ReachGrid.load(path)
    grid = ReachGrid.compute(ik, limits, tip)
    grid.save(path)
    return grid
