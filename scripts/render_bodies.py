"""Pictures of a run's bodies: the CAD of the stock body and of each body the run kept (docs/evolve.md).

    uv run python scripts/render_bodies.py runs/evolve/gripper_s2v3_s0 [more runs ...] --out results/evolve/bodies

For each run, renders the CAD of every body in order (``sea.worlds.cad``), with cameras fitted to all of them so
they share a scale, and writes ``<out>/<run>/<k>_<body id>.png`` and ``<out>/<run>/frames.json``: per
picture, the iteration that kept the body, the groups it held, its change from the previous body (per
slot), what the change gained and cost, and System 2's reason for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _g(v: float) -> str:
    return f"{v:.3g}"


def changes(old: dict, new: dict) -> list[str]:
    """The change from one body to the next, slot by slot, in words."""
    out = []
    for slot in sorted(set(old["parts"]) | set(new["parts"])):
        a, b = old["parts"].get(slot), new["parts"].get(slot)
        if a == b:
            continue
        if a is None or b is None or a["type"] != b["type"]:
            ps = ", ".join(f"{k} {_g(v)}" for k, v in sorted(b["params"].items())) if b else ""
            out.append(f"{slot}: {a['type'] if a else 'none'} → {b['type'] if b else 'removed'}" + (f" ({ps})" if ps else ""))
        else:
            ps = [f"{k} {_g(a['params'][k])} → {_g(v)}" if k in a["params"] else f"{k} {_g(v)}"
                  for k, v in sorted(b["params"].items()) if a["params"].get(k) != v]
            out.append(f"{slot}: " + ", ".join(ps))
    return out


def _kept(rec: dict, body_id: str) -> dict | None:
    for t in rec.get("trials", []):
        if f"b{rec['iteration']:02d}{t.get('k', '')}" == body_id:
            return t
    return None


def _why(rec: dict, trial: dict) -> list[str]:
    """System 2's reasons for a kept change: its proposal's, or each merged proposal's."""
    if trial.get("merged"):
        parts = [_kept(rec, b) for b in trial["merged"]]
        return [(p.get("proposal") or {}).get("hypothesis") or p.get("hypothesis", "") for p in parts if p]
    return [(trial.get("proposal") or {}).get("hypothesis") or trial.get("hypothesis", "")]


def frames(run: Path) -> tuple[str, list[dict], list[dict]]:
    """The world, and each distinct body along the run with the record of the change that made it."""
    summary = json.loads((run / "summary.json").read_text())
    recs = [json.loads(x) for x in (run / "iterations.jsonl").read_text().splitlines()]
    out, bodies, prev = [], [], None
    seq = [(r["iteration"], r["body"], len(r["held"])) for r in recs]
    if not seq or summary["final_body"]["parts"] != seq[-1][1]["parts"]:
        seq.append((len(recs), summary["final_body"], len(summary["final_held_train"])))
    for it, body, held in seq:
        if prev is not None and body["parts"] == prev["parts"]:
            continue
        f = {"iteration": it, "id": body.get("id") or "final", "held": held, "groups": summary["train_groups"],
             "changes": changes(prev, body) if prev else [], "body": body}
        if prev is not None:
            rec = next((r for r in recs if r["iteration"] == it - 1), None)
            t = _kept(rec, f["id"]) if rec else None
            if t is not None:
                res = t.get("result", "")
                f |= {"move": t.get("move", ""), "gained": t.get("gained", []), "lost": t.get("lost", []),
                      "cost": res.split("costs ", 1)[1] if "costs " in res else "", "why": _why(rec, t)}
        out.append(f)
        bodies.append(body)
        prev = body
    return summary["world"], out, bodies


def main(argv: list[str] | None = None) -> int:
    from sea.design.cad import render
    from sea.evolve.core import Body
    from sea.worlds.cad import CAD

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for run in args.runs:
        if not (run / "summary.json").exists():
            continue
        world_name, fs, bodies = frames(run)
        if world_name not in CAD:
            print(f"{run.name}: the {world_name} world has no CAD", file=sys.stderr)
            continue
        models = [CAD[world_name](Body.model_validate(b)) for b in bodies]
        pngs = render(models)
        d = args.out / run.name
        d.mkdir(parents=True, exist_ok=True)
        for k, (f, png, mdl) in enumerate(zip(fs, pngs, models, strict=True)):
            f["file"] = f"{k:02d}_{f['id']}.png"
            f["views"] = [v.name for v in mdl.views]
            (d / f["file"]).write_bytes(png)
        (d / "frames.json").write_text(json.dumps({"run": run.name, "world": world_name, "frames": fs}, indent=1) + "\n")
        print(f"{run.name}: {len(fs)} bodies -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
