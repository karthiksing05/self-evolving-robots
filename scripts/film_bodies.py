"""Films of each change a run kept: the body before it failing and the body after it holding, on the task
groups the change gained (docs/evolve.md, D-050).

    uv run python scripts/film_bodies.py runs/evolve/gripper_s2v3_s0 [more runs ...] --out results/evolve/films

For each kept change, up to ``--per-change`` gained groups are filmed, distinct failure mechanisms first.
Each film replays one train task twice with the controls the run recorded (``sea.worlds.film``): the
previous body with System 1's best failing controls, then the new body with the controls that passed.
Nothing is searched again.  The two play side by side (in turn, when they have sound) and end on their
verdicts.  Writes ``<out>/<run>/<k>_<body id>_<group>.mp4`` with a JPEG poster, and ``films.json``: per
film, the change, the task, and each side's verdict on the replay beside the run's own.

A run recorded when any harness component its world's replay reads differed from today's (``film.READS``),
or with no harness.json to tell, is skipped: the replay would show something the run never saw.  Replays
run under the run's seed, and a film whose replay disagrees with the run's record is not written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from render_bodies import _kept, frames


def _human(s: str) -> str:
    return s.replace("_", " ")


def main(argv: list[str] | None = None) -> int:
    from sea.evolve.core import Body
    from sea.evolve.loop import make_world
    from sea.harness import fingerprint
    from sea.worlds.film import FILM, READS, encode, pair

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-change", type=int, default=3)
    args = ap.parse_args(argv)
    now = fingerprint()
    for run in args.runs:
        if not (run / "summary.json").exists():
            continue
        if not (run / "harness.json").exists():
            print(f"{run.name}: no harness.json, so what it was judged by is unknown; not filmed", file=sys.stderr)
            continue
        world_name, fs, _ = frames(run)
        if world_name not in FILM:
            print(f"{run.name}: the {world_name} world has no films", file=sys.stderr)
            continue
        then = json.loads((run / "harness.json").read_text()).get("components", {})
        moved = [c for c in READS[world_name] if then.get(c, {}).get("hash") != now[c]["hash"]]
        if moved:
            print(f"{run.name}: recorded before {', '.join(moved)} changed; not filmed", file=sys.stderr)
            continue
        tasks = {t.id: t for t in make_world(world_name).tasks()}
        seed = int(json.loads((run / "summary.json").read_text()).get("seed", 0))
        recs = {r["iteration"]: r for r in map(json.loads, (run / "iterations.jsonl").read_text().splitlines())}
        out, d = [], args.out / run.name
        for k, f in enumerate(fs):
            rec = recs.get(f["iteration"] - 1)
            trial = _kept(rec, f["id"]) if rec else None
            if trial is None:
                continue
            chosen, seen = [], set()
            fails = [(tid, o) for tid, o in rec["failing"].items() if tid in tasks and tasks[tid].split == "train"]
            for distinct in (True, False):  # one film per failure mechanism first, then the rest
                for g in trial.get("gained", []):
                    hit = next(((tid, o) for tid, o in fails if tasks[tid].group == g), None)
                    if hit and g not in {c[0] for c in chosen} and (not distinct or hit[1]["mechanism"] not in seen):
                        chosen.append((g, *hit))
                        seen.add(hit[1]["mechanism"])
            for g, tid, before in chosen[: args.per_change]:
                after = trial.get("outcomes", {}).get(tid, {})
                prev_body = Body.model_validate(fs[k - 1]["body"])
                b = FILM[world_name](prev_body, tasks[tid], before.get("controls", {}), seed)
                a = FILM[world_name](Body.model_validate(f["body"]), tasks[tid], after.get("controls", {}), seed)
                if b.ok or a.ok != bool(after.get("success")):
                    print(f"{run.name} {f['id']} {tid}: the replay ({b.verdict} / {a.verdict}) disagrees with the "
                          f"record; not filmed", file=sys.stderr)
                    continue
                if not b.ok and before.get("mechanism"):
                    b.verdict = f"{_human(before['mechanism'])} ({b.verdict})" if world_name.startswith("gripper") else b.verdict
                title = tasks[tid].facts if len(tasks[tid].facts) <= 110 else tasks[tid].facts[:107] + "…"
                film, audio = pair(b, a, title, (f"before · {fs[k - 1]['id']}", f"after · {f['id']} (this change)"))
                name = f"{k:02d}_{f['id']}_{g}.mp4"
                encode(film, d / name, audio)
                out.append({"frame": k, "id": f["id"], "group": g, "task": tid, "file": name,
                            "poster": name.replace(".mp4", ".jpg"), "seconds": round(len(film) / 25, 1),
                            "sound": audio is not None,
                            "before": {"ok": b.ok, "verdict": b.verdict, "recorded": before.get("mechanism", "")},
                            "after": {"ok": a.ok, "verdict": a.verdict, "recorded": after.get("success")}})
                print(f"{run.name} {name}: before {'pass' if b.ok else 'fail'}, after {'pass' if a.ok else 'fail'}")
        if out:
            d.mkdir(parents=True, exist_ok=True)
            (d / "films.json").write_text(json.dumps({"run": run.name, "world": world_name, "films": out}, indent=1,
                                                     ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
