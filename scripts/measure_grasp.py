"""Measure the stock gripper's capability table (docs/gripper-experiment.md, V1).

    uv run python scripts/measure_grasp.py            # measure and print the diff against the stored table
    uv run python scripts/measure_grasp.py --write    # re-record after a reviewed change

Runs the fixed case list through ``sea.sim.grasp_probe.pick`` (about 0.3 s per pick, one core) and
writes ``tests/golden/stock_grasp.json``: one row per pick plus the envelope that phase A, the
typecheck aperture rule and the phase-B object table rely on.  Like the golden verdicts, never
re-record without reading the diff and writing a decision entry.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "stock_grasp.json"
CASES: list[tuple[str, float, float, float, float]] = [  # shape, width mm, mass g, friction, sphere_past_mm
    *[("cylinder", w, 50.0, 0.8, 0.0) for w in (20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90)],
    *[("cylinder", 30, m, 0.8, 0.0) for m in (100, 200, 400, 800, 1200)],
    # friction: if a failure does not move with friction, it is geometry, not slip
    *[("cylinder", 30, m, 0.3, 0.0) for m in (200, 800)],
    *[("cylinder", 50, 50.0, mu, 0.0) for mu in (0.3, 1.5)],
    *[("box", w, 50.0, 0.8, 0.0) for w in (20, 30, 40, 50, 60, 70, 80)],
    *[("sphere", w, 50.0, 0.8, past) for w in (20, 30, 40, 50, 60, 70, 80, 90) for past in (5.0, 12.0)],
    *[("sphere", 40, 50.0, mu, 12.0) for mu in (0.3, 1.5)],
]


def measure() -> dict[str, Any]:
    from sea.sim.grasp_probe import PROBE, pick, scene_object

    rows = []
    for shape, w, mass, mu, past in CASES:
        probe = replace(PROBE, sphere_past_mm=past) if shape == "sphere" else PROBE
        row = pick(scene_object(shape, w, mass, mu, probe), probe=probe).row()  # type: ignore[arg-type]
        if shape == "sphere":
            row["sphere_past_mm"] = past
        rows.append(row)
    return {"probe": asdict(PROBE), "picks": rows, "envelope": envelope(rows)}


def envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the stock gripper can do, read off the rows (default friction 0.8, 50 g unless stated)."""

    def widest(shape: str, **match: float) -> float | None:
        held = [r["width_mm"] for r in rows if r["shape"] == shape and r["held"] and r["friction"] == 0.8
                and r["mass_g"] == 50 and all(r.get(k) == v for k, v in match.items())]
        return max(held) if held else None

    heavy = [r for r in rows if r["shape"] == "cylinder" and r["width_mm"] == 30 and r["friction"] == 0.8]
    squeeze = sorted(r["squeeze_n"] for r in rows if r["held"])
    return {
        "cylinder_widest_held_mm": widest("cylinder"),
        "box_widest_held_mm": widest("box"),
        "sphere_widest_held_mm": {str(p): widest("sphere", sphere_past_mm=p) for p in (5.0, 12.0)},
        "cylinder_30mm_heaviest_held_g": max((r["mass_g"] for r in heavy if r["held"]), default=None),
        "held_squeeze_n_median": squeeze[len(squeeze) // 2] if squeeze else None,
        "held_grip_current_a": max((r["grip_current_a"] for r in rows if r["held"]), default=None),
    }


def diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    key = ("shape", "width_mm", "mass_g", "friction", "sphere_past_mm")
    before = {tuple(r.get(k) for k in key): r for r in old.get("picks", [])}
    out = []
    for r in new["picks"]:
        o = before.get(tuple(r.get(k) for k in key))
        if o is None:
            out.append(f"new case {r['shape']} {r['width_mm']} mm {r['mass_g']} g mu {r['friction']}: held={r['held']}")
        elif o["held"] != r["held"] or abs(o["hold_mm"] - r["hold_mm"]) > 5 or abs(o["jaw_q"] - r["jaw_q"]) > 0.02:
            out.append(f"{r['shape']} {r['width_mm']} mm {r['mass_g']} g mu {r['friction']}: held {o['held']}->{r['held']}"
                       f" hold_mm {o['hold_mm']}->{r['hold_mm']} jaw_q {o['jaw_q']}->{r['jaw_q']}")
    if old.get("envelope") != new["envelope"]:
        out.append(f"envelope {old.get('envelope')} -> {new['envelope']}")
    return out


def main(argv: list[str]) -> int:
    t = time.time()
    got = measure()
    old = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    for r in got["picks"]:
        print(f"{r['shape']:>8s} {r['width_mm']:5.1f} mm {r['mass_g']:6.0f} g mu {r['friction']:.1f} "
              f"held={str(r['held']):5s} hold {r['hold_mm']:5.1f} mm jaw_q {r['jaw_q']:+.3f} "
              f"squeeze {r['squeeze_n']:5.1f} N pads {r['pads_n']} I_grip {r['grip_current_a']:.2f} A "
              f"payload {r['payload_g']} g")
    print(json.dumps(got["envelope"], indent=1))
    changes = diff(old, got)
    print(f"{len(got['picks'])} picks in {time.time() - t:.1f}s; "
          f"{len(changes)} change(s) against {GOLDEN.relative_to(ROOT)}")
    for c in changes:
        print("  " + c)
    if "--write" in argv:
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
