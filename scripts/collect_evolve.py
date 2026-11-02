"""Gather an evolution run into one tracked record (docs/evolve.md).

    uv run python scripts/collect_evolve.py runs/evolve/gripper_s2_s0 [more runs ...] --out results/evolve

For each run directory, writes ``<out>/<run>.json``:
- the summary;
- per iteration: the held groups, failing mechanisms, and every proposal with its move, edit,
  reasoning, predicted and actual gains, resources and result;
- System 2's call records;
- the final body's bill of materials against the stock body's.

Prompts and full trial outcomes stay in the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def collect(run: Path) -> dict:
    from sea.evolve.core import Body
    from sea.evolve.loop import make_world

    summary = json.loads((run / "summary.json").read_text())
    world = make_world(summary["world"])
    iters = []
    for line in (run / "iterations.jsonl").read_text().splitlines():
        rec = json.loads(line)
        mech = Counter(o["mechanism"] for o in rec.get("failing", {}).values())
        props = []
        for t in rec.get("trials", []):
            p = t["proposal"]
            props.append({
                "k": t["k"], "move": t.get("move"), "declared_move": p["move"], "edit": t.get("edit"),
                "mechanism": p["mechanism"], "principle": p["principle"], "hypothesis": p["hypothesis"],
                "predicted_gains": p.get("gains", []), "risks": p.get("risks", []), "idea": p.get("idea"),
                "result": t.get("result"), "gained": t.get("gained", []), "lost": t.get("lost", []),
                "prediction": t.get("prediction"),
                "resources": {k: t["resources"][k] for k in ("usd", "lead_days", "fab_minutes", "lines", "errors")}
                if t.get("resources") else None,
                "still_failing": t.get("still_failing", []),
            })
        iters.append({"iteration": rec["iteration"], "body": Body.model_validate(rec["body"]).describe(),
                      "held": rec["held"], "failing_mechanisms": dict(mech.most_common()),
                      "failing": {k: {"mechanism": v["mechanism"], "detail": v["detail"][:400]}
                                  for k, v in rec.get("failing", {}).items()},
                      "proposals": props, "accepted": rec.get("accepted"), "notes": rec.get("notes", ""),
                      "llm": rec.get("llm", {}), "seconds": rec.get("seconds"), "stop": rec.get("stop")})
    final = Body.model_validate(summary["final_body"])
    calls = []
    if (run / "llm" / "calls.jsonl").exists():
        calls = [json.loads(x) for x in (run / "llm" / "calls.jsonl").read_text().splitlines()]
    harness = json.loads((run / "harness.json").read_text()) if (run / "harness.json").exists() else {}
    return {"run": run.name, "summary": summary, "final_body": final.describe(),
            "final_bom": [b.model_dump() for b in world.bom(final)],
            "stock_bom": [b.model_dump() for b in world.bom(world.stock())],
            "judge_key": harness.get("judge_key"), "iterations": iters, "llm_calls": calls}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/evolve"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        rec = collect(run)
        (args.out / f"{run.name}.json").write_text(json.dumps(rec, indent=1) + "\n")
        s = rec["summary"]
        print(f"{run.name}: train {len(s['final_held_train'])}/{s['train_groups']}, test final "
              f"{len(s['test']['final_held'])} vs stock {len(s['test']['stock_held'])}; {rec['final_body']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
