"""Gather one run directory into a single JSON (D-039): per iteration the coverage, escalations, targets,
every proposal with its hypothesis and screening errors, the paired trial, the decision and the LLM
call records; then the designs, the final summary and per-table verdicts, timing and the goal log.

    uv run python scripts/collect_run.py runs/gripper_live_s1 results/experiments/gripper_live_s1.json
"""
import json
import sys
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])


def load(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


iters = []
for d in sorted(run.glob("iter_*")):
    it = int(d.name[-3:])
    rep = load(d / "report.json", {})
    prop = load(d / "proposal.json", {})
    dec = load(d / "decision.json")
    calls = jsonl(d / "llm_calls.jsonl")
    res = jsonl(d / "results.jsonl")
    cands = []
    for i, c in enumerate(prop.get("candidates", [])):
        p = c["proposal"]
        cands.append({"i": i, "kind": p["kind"], "rung": p["rung"],
                      "fingers": (p.get("design_delta") or {}).get("fingers", {}),
                      "reparam": p.get("reparam"), "hypothesis": p["hypothesis"], "rationale": p["rationale"],
                      "ruled_out": p.get("ruled_out", []), "prediction": p["prediction"],
                      "errors": c["errors"], "vpc": c["vpc"], "cost": c["cost"]})
    iters.append({
        "iteration": it, "coverage_reward": rep.get("coverage_reward"), "reward": rep.get("reward"),
        "design": rep.get("design_id") or rep.get("design"),
        "verdicts": {r["panel_id"]: r["verdict"] for r in res},
        "escalations": prop.get("escalations", []), "targets": prop.get("targets", []),
        "candidates": cands, "chosen": prop.get("chosen"), "trial": prop.get("trial"), "decision": dec,
        "llm": [{k: c.get(k) for k in ("input_tokens", "output_tokens", "usd", "n_proposals", "billing", "error",
                                       "attempts")} for c in calls],
        "cost": load(d / "cost.json"), "promoted": prop.get("promoted"),
    })
designs = {p.stem: load(p) for p in sorted(run.glob("design_*.json"))}
final = load(run / "final" / "summary.json")
final_results = jsonl(run / "final" / "results.jsonl")
timing = load(run / "timing.json")
goals = jsonl(run / "goals.jsonl")
out.write_text(json.dumps({"run": run.name, "iterations": iters, "designs": designs, "final": final,
                           "final_results": [{"panel": r["panel_id"], "verdict": r["verdict"],
                                              "trials": r.get("s1_trials")} for r in final_results],
                           "timing": timing, "goals": goals, "final_design": load(run / "final" / "design.json")},
                          indent=1))
print(out, len(iters), "iterations")
