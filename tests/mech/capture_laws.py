"""Record the gripper's mechanism-law output on the catalogue's jaws (D-078), before the laws read the graph:
the golden file they must reproduce.  python tests/mech/capture_laws.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sea.evolve.core import Part
from sea.mechanism import gripper as MG
from sea.worlds.gripper import GripperWorld

sys.path.insert(0, str(Path(__file__).parent))
from capture_jaws import GRID  # noqa: E402

OUT = Path(__file__).with_name("golden_laws.json")


def bodies():
    w = GripperWorld()
    for jaw, grid in GRID.items():
        for ps in grid:
            b = w.stock()
            yield f"{jaw} {json.dumps(ps, sort_keys=True)}", b.model_copy(update={"parts": {**b.parts, "jaw": Part(type=jaw, params=ps)}})


def capture() -> dict:
    out = {}
    for key, b in bodies():
        g = MG.geometry(b)
        out[key] = {"laws": MG.laws(b), "fastened": sorted(g["fastened"]), "loose": sorted(g["loose"]),
                    "clashes": sorted(g["clashes"]), "carried": {k: v["worst_gap_mm"] for k, v in g["carried"].items()}}
    return out


if __name__ == "__main__":
    OUT.write_text(json.dumps(capture(), indent=1))
    print(OUT)
