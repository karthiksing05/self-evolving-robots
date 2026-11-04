"""Run the evolution core on one world (docs/evolve.md, D-041).

    uv run python scripts/evolve.py --world phonology --arm s2 --iterations 10
    uv run python scripts/evolve.py --world gripper --arm s2 --iterations 12 --workers 8
    uv run python scripts/evolve.py --world mobile --arm random --seed 1

Arms: ``s2`` (System 2 on the Claude plan through the Claude Code CLI, never API credits), ``random``
(blind catalogue edits under the same screen and trials) and ``s1`` (System 1 only: the stock body).
Writes ``config.json``, ``manifest.json``, ``iterations.jsonl``, ``llm/`` (every prompt and answer) and
``summary.json`` to ``--out`` (default ``runs/evolve/<world>_<arm>_s<seed>``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    from sea.evolve.loop import RunConfig, run

    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--arm", choices=("s2", "random", "s1"), default="s2")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tune-trials", type=int, default=RunConfig.model_fields["tune_trials"].default,
                    help="trials per iteration spent fitting a promising swap's parameters (D-044); 0 turns it off")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--tools", action="store_true", help="System 2 may question the harness while it proposes (D-086)")
    ap.add_argument("--requirements", action="store_true", help="System 2 keeps a measured requirements list (D-089)")
    ap.add_argument("--research", action="store_true", help="System 2 researches the web and keeps the notebook (D-090)")
    ap.add_argument("--concepts", action="store_true", help="the concept set, functions, comparison, failure modes (D-091)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or ROOT / "runs" / "evolve" / f"{args.world}_{args.arm}_s{args.seed}"
    if out.exists():  # a run's record is never overwritten (two were lost that way, D-082)
        ap.error(f"{out} exists: choose another --out")
    cfg = RunConfig(world=args.world, arm=args.arm, iterations=args.iterations, seed=args.seed, workers=args.workers,
                    tune_trials=args.tune_trials, model=args.model, effort=args.effort, tools=args.tools,
                    requirements=args.requirements, research=args.research, concepts=args.concepts)
    summary = run(cfg, out, log=lambda m: print(m, flush=True))
    keep = {k: summary[k] for k in ("world", "arm", "final_held_train", "test", "spent", "llm", "wall_s", "sim_s")}
    print(json.dumps(keep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
