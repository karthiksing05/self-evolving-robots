"""Measure a gripper on every real object (configs/ycb_objects.yaml, D-038).

    uv run python scripts/measure_real.py                          # stock jaws
    uv run python scripts/measure_real.py --fingers '{"reach_mm": 10, "pad_tpu_mm": 2}' --out results/x.json

Runs ``pick_up`` with System 1's parameter search and confirmation re-runs, as the loop does, on each
object centred on the table and at ``--placements`` seeded offsets.  An object is held when every
placement passes.  The stock run splits the objects into phase A (held: System 1 learns them) and
phase B (not held: System 2's targets) by measurement, not by a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    from sea.compose.library import Library
    from sea.contracts import Design, FingerSpec, HardLimits, SensorSuite
    from sea.design.build import build
    from sea.sim.env import SuiteConfig, attempts, run_panel
    from sea.sim.grasp_world import real_panels, regression_panels
    from sea.sim.objects import catalog

    ap = argparse.ArgumentParser()
    ap.add_argument("--fingers", default="{}", help="FingerSpec fields as JSON (default: stock jaws)")
    ap.add_argument("--objects", default="", help="comma-separated names (default: the whole catalog)")
    ap.add_argument("--placements", type=int, default=2, help="seeded off-centre placements per object")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    limits = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits_grasp.yaml").read_text()))
    fingers = FingerSpec(**json.loads(args.fingers))
    built = build(Design(id="measure", fingers=fingers, sensors=SensorSuite(jaw_pads=True)), limits)
    cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=limits, seed=0)
    names = args.objects.split(",") if args.objects else list(catalog())
    t0 = time.time()
    rows: dict[str, dict] = {}
    for name in names:
        panels = regression_panels([name]) + real_panels("val", args.placements, [name], seed=7)
        runs = [r for p in panels for r in run_panel(p, cfg)]
        sim_s = sum(r.trace.sim_seconds for r in runs if r.trace is not None)
        rows[name] = {
            "held": all(r.result.verdict == "pass" for r in runs),
            "verdicts": [r.result.verdict for r in runs],
            "notes": [r.trace.note if r.trace is not None else "" for r in runs],
            "attempts": sum(attempts(r) for r in runs),
            "sim_s": round(sim_s, 1),
            "params": [r.params for r in runs],
        }
        print(f"{name:18s} {'HELD' if rows[name]['held'] else 'not held':9s} {rows[name]['verdicts']} "
              f"{rows[name]['notes']} attempts={rows[name]['attempts']}", flush=True)
    out = {"fingers": fingers.model_dump(), "placements": 1 + args.placements, "objects": rows,
           "held": sorted(n for n, r in rows.items() if r["held"]),
           "not_held": sorted(n for n, r in rows.items() if not r["held"]),
           "wall_s": round(time.time() - t0, 1)}
    print(f"held {len(out['held'])}/{len(rows)} in {out['wall_s']} s")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
