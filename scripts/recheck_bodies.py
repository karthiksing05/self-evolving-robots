"""Re-solve runs' final bodies on today's judge, under several seeds (D-097): what a recorded score is worth now.

    uv run python scripts/recheck_bodies.py runs/evolve/gripper_s2rods_s0 [more runs ...] --seeds 0 1 2 --out recheck.json

A run that crashed has no summary: its final body is the last iteration's body with the trial it kept.  The stock body
is always included.  Writes, per body and seed, the groups held on train and test and each failing task's mechanism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def final_body(run: Path, world):
    from sea.evolve.core import Body, Edit, Idea
    from sea.evolve.loop import apply

    summary = run / "summary.json"
    if summary.exists():
        return Body.model_validate(json.loads(summary.read_text())["final_body"])
    rec = [json.loads(x) for x in (run / "iterations.jsonl").read_text().splitlines()][-1]
    body = Body.model_validate(rec["body"])
    kept = next((t for t in rec.get("trials", []) if t.get("result", "").startswith("KEPT")), None)
    if kept is None:
        return body
    prop = kept["proposal"]
    idea = (prop.get("idea") or {})
    if idea.get("mechanism"):
        body = world.build_idea(body, Idea.model_validate(idea), new_id="final")
    edits = [Edit.model_validate(e) for e in prop.get("edits") or []]
    return apply(body, edits, {p.name: p for p in world.catalogue()}, new_id="final") if edits else body


def main(argv: list[str] | None = None) -> int:
    from sea.evolve.loop import Evaluator, held_groups, make_world

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--world", default="gripper")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-stock", action="store_true", help="leave out the stock body, already measured")
    args = ap.parse_args(argv)
    world = make_world(args.world)
    tasks = world.tasks()
    train, test = [t for t in tasks if t.split == "train"], [t for t in tasks if t.split != "train"]
    bodies = {**({} if args.no_stock else {"stock": world.stock()}), **{r.name: final_body(r, world) for r in args.runs}}
    out: dict = {}
    for name, body in bodies.items():
        out[name] = {"body": body.model_dump(), "seeds": {}}
        for seed in args.seeds:
            ev = Evaluator(world, args.world, workers=args.workers, seed=seed)
            try:
                got = ev.solve(body, tasks)
            finally:
                ev.close()
            row = {"train": sorted(held_groups(train, got)), "test": sorted(held_groups(test, got)),
                   "failing": {t.id: got[t.id].mechanism for t in tasks if not got[t.id].success},
                   "tasks": sum(o.success for o in got.values()), "n": len(tasks)}
            out[name]["seeds"][seed] = row
            print(f"{name} seed {seed}: train {len(row['train'])}, test {len(row['test'])}, tasks {row['tasks']}/{row['n']}", flush=True)
            args.out.write_text(json.dumps(out, indent=1))
    groups = sorted({t.group for t in tasks})
    for name, r in out.items():
        rows = list(r["seeds"].values())
        always = set.intersection(*(set(x["train"]) & set(x["test"]) for x in rows))
        print(f"{name}: held on train and test under every seed: {len(always)}/{len(groups)}; never: {sorted(set(groups) - always)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
