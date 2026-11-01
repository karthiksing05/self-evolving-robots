"""B3 battery (docs/iteration.md §5): stock arm + seed library on fixed panels, per-class pass share.

    uv run python scripts/battery.py                    # measure and compare with the saved baseline
    uv run python scripts/battery.py --save             # make this measurement the new baseline
    uv run python scripts/battery.py --n 6 --rack hook:14,6,45 --rack pad:16,16,5,0.9

Panels are fixed (``panel_seed``, splits and counts below), System 1 runs with its normal budget,
and every class reports its pass share with a 95% bootstrap interval.  A class whose interval no
longer overlaps the baseline's is flagged.  The output names the harness judge key; a baseline
recorded under a different judge key, or with different panels, rack or System 1 budget, is
refused (re-baseline instead).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "golden" / "battery_baseline.json"
TEMPLATE_ARGS = {"pad": ("w", "h", "thickness", "friction"), "hook": ("depth", "throat", "length"),
                 "knob_cup": ("inner_diameter", "depth", "ribs"), "probe": ("diameter", "length"),
                 "slot_inserter": ("blade_thickness", "width", "length"), "soft_tip": ("radius", "softness")}


def _ci(x: list[float], seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(a, size=(1000, a.size), replace=True).mean(axis=1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def measure(n: int, splits: list[str], rack: list[str], m: int) -> dict:
    from sea.compose.library import Library
    from sea.contracts import Design, HardLimits
    from sea.control.executor import stock_ik
    from sea.control.reach import reach_for
    from sea.design.build import build
    from sea.design.space import make_tool_spec
    from sea.harness import fingerprint, judge_key
    from sea.sim.env import SuiteConfig, run_suite
    from sea.sim.panels import generate, reference_panel

    limits = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
    tools = []
    for r in rack:
        name, _, vals = r.partition(":")
        tools.append(make_tool_spec(name, **dict(zip(TEMPLATE_ARGS[name], map(float, vals.split(",")), strict=False))))
    built = build(Design(id="battery", tool_rack=tools), limits)
    if not built.ok:
        raise SystemExit(f"rack does not build: {built.errors}")
    cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=limits, seed=0,
                      m=m, confirm=False, reach=reach_for(stock_ik(), limits, None, ROOT / ".cache" / "reach"))
    out: dict = {"judge_key": judge_key(fingerprint()), "n": n, "rack": rack, "m": m, "splits": {}}
    for split in splits:
        panels = [reference_panel()] if split == "reference" else generate(split, n, 1)
        suite = run_suite(panels, cfg)
        by_cls: dict[str, list[float]] = {}
        verdicts: dict[str, int] = {}
        for r in suite.results:
            by_cls.setdefault(r.cls, []).append(1.0 if r.verdict == "pass" else 0.0)
            verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
        out["splits"][split] = {"by_class": {c: _ci(v) for c, v in sorted(by_cls.items())},
                                "all": _ci([x for v in by_cls.values() for x in v]), "verdicts": verdicts}
    return out


def compare(new: dict, old: dict) -> list[str]:
    flags = []
    for split, s in new["splits"].items():
        o = old.get("splits", {}).get(split)
        if o is None:
            continue
        for cls, (m, lo, hi) in s["by_class"].items():
            if cls not in o["by_class"]:
                continue
            om, olo, ohi = o["by_class"][cls]
            if hi < olo or lo > ohi:
                flags.append(f"{split}/{cls}: {om:.2f} [{olo:.2f}, {ohi:.2f}] -> {m:.2f} [{lo:.2f}, {hi:.2f}]")
    return flags


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="panels per split")
    ap.add_argument("--splits", default="reference,train,val")
    ap.add_argument("--rack", action="append", default=[], help="template:p1,p2,... (repeatable)")
    ap.add_argument("--m", type=int, default=2, help="System 1 local-search trials")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)
    new = measure(args.n, args.splits.split(","), args.rack, args.m)
    for split, s in new["splits"].items():
        m, lo, hi = s["all"]
        print(f"{split}: pass {m:.2f} [{lo:.2f}, {hi:.2f}]  verdicts {s['verdicts']}")
        for cls, (cm, clo, chi) in s["by_class"].items():
            print(f"   {cls:16s} {cm:.2f} [{clo:.2f}, {chi:.2f}]")
    if BASELINE.exists() and not args.save:
        old = json.loads(BASELINE.read_text())
        if old.get("judge_key") != new["judge_key"]:
            print(f"baseline judged by {old.get('judge_key')}, now {new['judge_key']}: re-baseline (--save)")
            return 1
        setup = {k: (old.get(k), new[k]) for k in ("n", "rack", "m") if old.get(k) != new[k]}
        if setup:
            print(f"baseline measured with a different setup {setup}: not comparable")
            return 1
        flags = compare(new, old)
        print("\n".join(["moved beyond the baseline interval:", *flags]) if flags else "no class moved beyond its interval")
    if args.save:
        BASELINE.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
