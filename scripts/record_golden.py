"""Record (or check) the golden verdicts: stock arm + seed library on the reference panel.

    uv run python scripts/record_golden.py            # print the diff against the stored file
    uv run python scripts/record_golden.py --write    # re-record after a reviewed change

A golden diff is how a change to a fundamental shows its effect on a fixed instrument
(docs/iteration.md).  Never re-record without reading the diff and writing a decision entry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "reference_verdicts.json"


def record() -> dict:
    from sea.compose.library import Library
    from sea.contracts import Design, HardLimits
    from sea.design.build import build
    from sea.sim.env import SuiteConfig, run_suite
    from sea.sim.panels import reference_panel

    limits = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
    built = build(Design(id="stock"), limits)
    cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=limits, seed=0,
                      confirm=False, m=0)
    suite = run_suite([reference_panel()], cfg)
    return {r.result.control_id: {"cls": r.result.cls, "verdict": r.result.verdict, "program": r.program}
            for r in suite.runs}


def main(argv: list[str]) -> int:
    got = record()
    old = json.loads(GOLDEN.read_text())["verdicts"] if GOLDEN.exists() else {}
    changed = {k: (old.get(k, {}).get("verdict"), v["verdict"]) for k, v in got.items()
               if old.get(k, {}).get("verdict") != v["verdict"]}
    for k, (a, b) in sorted(changed.items()):
        print(f"{k} ({got[k]['cls']}): {a} -> {b}")
    if not changed:
        print("golden verdicts unchanged")
    if "--write" in argv:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"verdicts": got}, indent=1, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
