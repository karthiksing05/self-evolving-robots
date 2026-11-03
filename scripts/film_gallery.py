"""A gallery of a run: the stock body and the final body side by side, one film per task group, held or not.

    uv run python scripts/film_gallery.py runs/evolve/mobile_s2mj_s0 [more runs ...] --out results/evolve/gallery

For each task group (its first train task) both bodies are solved again by System 1 under the run's seed, and
those attempts are filmed with ``sea.worlds.film`` (``pair``).  A film whose verdict on either side disagrees
with the run's record (the first iteration's held groups for the stock, the final ones for the evolved body)
is not written, nor one whose drawn robot goes more than ``film.CLIP_MM`` into
what it touches; a run judged by a harness component the replay reads that has changed since is skipped
(``film.READS``), as in ``film_bodies.py``.  Writes ``<out>/<run>/<group>.mp4`` with a JPEG poster, and
``gallery.json``: per film, the group, its split, the verdict and the run's record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from sea.evolve.core import Body
    from sea.evolve.loop import make_world
    from sea.harness import fingerprint
    from sea.worlds.film import CLIP_MM, FILM, READS, encode, pair

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    now = fingerprint()
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        world_name = summary["world"]
        then = json.loads((run / "harness.json").read_text()).get("components", {}) if (run / "harness.json").exists() else {}
        moved = [c for c in READS.get(world_name, ()) if then.get(c, {}).get("hash") != now[c]["hash"]]
        if world_name not in FILM or moved or not then:
            print(f"{run.name}: not filmed ({'judged before ' + ', '.join(moved) + ' changed' if moved else 'no record'})",
                  file=sys.stderr)
            continue
        world, seed = make_world(world_name), int(summary.get("seed", 0))
        body = Body.model_validate(summary["final_body"])
        held = set(summary["final_held_train"])
        stock_held = set(json.loads((run / "iterations.jsonl").read_text().splitlines()[0])["held"])
        stock = world.stock()
        first: dict[str, object] = {}
        for t in world.tasks():
            if t.split == "train" and t.group not in first:
                first[t.group] = t
        out, d = [], args.out / run.name
        for g, t in first.items():
            clips = [FILM[world_name](b, t, world.solve(b, t, seed).controls, seed) for b in (stock, body)]
            wrong = [n for n, c, h in zip(("stock", "final"), clips, (stock_held, held), strict=True) if c.ok != (g in h)]
            if wrong:
                print(f"{run.name} {g}: the {' and '.join(wrong)} replay disagrees with the record; not filmed",
                      file=sys.stderr)
                continue
            deep = max(c.clipping_mm for c in clips)
            if deep > CLIP_MM:
                print(f"{run.name} {g}: a drawn part goes {deep:.1f} mm in; not filmed", file=sys.stderr)
                continue
            title = t.facts if len(t.facts) <= 110 else t.facts[:107] + "…"
            frames, audio = pair(*clips, title, ("stock body", f"evolved body ({body.id or 'final'})"))
            encode(frames, d / f"{g}.mp4", audio)
            out.append({"group": g, "task": t.id, "file": f"{g}.mp4", "poster": f"{g}.jpg", "clipping_mm": round(deep, 2),
                        **{n: {"ok": c.ok, "verdict": c.verdict, "headline": c.headline}
                           for n, c in zip(("stock", "final"), clips, strict=True)}})
            print(f"{run.name} {g}: stock {'pass' if clips[0].ok else 'fail'}, final {'pass' if clips[1].ok else 'fail'}",
                  flush=True)
        if out:
            (d / "gallery.json").write_text(json.dumps({"run": run.name, "world": world_name, "body": body.id,
                                                        "films": out}, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
