"""Real objects for the grasp world (configs/ycb_objects.yaml, D-038): YCB scans as sim assets.

``load_asset(name)`` reads ``assets/ycb/<name>.npz``, written by ``scripts/import_ycb.py``: the
convex parts the simulator collides with, a decimated scan it draws, and the scan's inertia at the
object's published mass (Calli et al. 2015, Table II).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "configs" / "ycb_objects.yaml"
ASSETS = ROOT / "assets" / "ycb"


@dataclass(frozen=True)
class Asset:
    parts: list[np.ndarray]
    visual_v: np.ndarray
    visual_f: np.ndarray
    com: np.ndarray
    inertia: np.ndarray
    rgba: np.ndarray
    extents_mm: np.ndarray  # width (y, across the jaws), depth (x), height (z)


@cache
def load_asset(name: str) -> Asset:
    d = np.load(ASSETS / f"{name}.npz")
    parts = [d[k] for k in sorted((k for k in d.files if k.startswith("part_")), key=lambda s: int(s.split("_")[1]))]
    return Asset(parts, d["visual_v"], d["visual_f"], d["com"], d["inertia"], d["rgba"], d["extents_mm"])


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    return yaml.safe_load(CATALOG.read_text())["objects"]
