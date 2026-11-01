"""``sea report``: plots and a markdown summary from run logs alone (spec 8.10, Section 11).

Works on one run directory or on a directory of runs (several proposers/seeds); runs are grouped by
``spec.proposer`` and reported as means with 95% bootstrap intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sea.contracts import CONTROL_CLASSES, ControlResult

# Reference categorical palette (light surface), fixed order; never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VERDICT_ORDER = ["pass", "stuck", "wrong_behavior", "no_actuation", "unreachable", "rule_violation",
                 "no_applicable_primitive", "inconclusive"]


def _runs(path: Path) -> list[Path]:
    if (path / "spec.yaml").exists():
        return [path]
    return sorted(p for p in path.iterdir() if (p / "spec.yaml").exists())


def _iters(run: Path) -> list[dict[str, Any]]:
    out = []
    for d in sorted(run.glob("iter_*")):
        rep = d / "report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            out.append({"iteration": r["iteration"], "coverage_reward": r["coverage_reward"], "reward": r["reward"],
                        "print_minutes": r["cost"]["print_minutes"],
                        "verdicts": [x["verdict"] for x in r["results"]]})
    return out


def bootstrap_ci(x: np.ndarray, seed: int = 0, n: int = 1000) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_report(path: str | Path, allow_mixed: bool = False) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    path = Path(path)
    runs = _runs(path)
    plots = path / ("final/plots" if len(runs) == 1 and runs[0] == path else "plots")
    plots.mkdir(parents=True, exist_ok=True)
    keys = {json.loads((r / "harness.json").read_text())["judge_key"] for r in runs if (r / "harness.json").exists()}
    if len(keys) > 1 and not allow_mixed:
        raise ValueError(f"runs were judged by different harness versions {sorted(keys)}; "
                         "re-run them or pass allow_mixed=True (docs/iteration.md)")
    groups: dict[str, list[Path]] = {}
    for r in runs:
        spec = yaml.safe_load((r / "spec.yaml").read_text())
        groups.setdefault(spec.get("proposer", "s2"), []).append(r)
    lines = [f"# Report: {path.name}", "", f"Judge version: {', '.join(sorted(keys)) or 'unknown'}", ""]

    # coverage reward vs iteration, one line per proposer with a 95% bootstrap band
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style(ax)
    for i, (name, rs) in enumerate(sorted(groups.items())):
        series = [[it["coverage_reward"] for it in _iters(r)] for r in rs]
        n = max((len(s) for s in series), default=0)
        if n == 0:
            continue
        stats = [bootstrap_ci(np.array([s[k] for s in series if len(s) > k])) for k in range(n)]
        m, lo, hi = (np.array(v) for v in zip(*stats, strict=True))
        color = SERIES[i % len(SERIES)] if i < len(SERIES) else INK2
        ax.fill_between(range(n), lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(range(n), m, color=color, linewidth=2, label=f"{name} (n={len(rs)})")
        ax.annotate(name, (n - 1, m[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("iteration", color=INK2)
    ax.set_ylabel("coverage reward", color=INK2)
    ax.set_ylim(0, 1)
    ax.set_title("Coverage reward by iteration (mean, 95% bootstrap CI)", color=INK, fontsize=11, loc="left")
    if len(groups) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(plots / "coverage_vs_iteration.png", dpi=150)
    plt.close(fig)

    # verdict mix per iteration (first run of the first group)
    first = runs[0]
    its = _iters(first)
    if its:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        _style(ax)
        bottom = np.zeros(len(its))
        for j, v in enumerate(VERDICT_ORDER):
            share = np.array([it["verdicts"].count(v) / max(len(it["verdicts"]), 1) for it in its])
            if share.sum() == 0:
                continue
            ax.bar(range(len(its)), share, bottom=bottom, color=SERIES[j], edgecolor=SURFACE, linewidth=1.5,
                   label=v, width=0.7)
            bottom += share
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("iteration", color=INK2)
        ax.set_ylabel("share of controls", color=INK2)
        ax.set_title(f"Verdict mix by iteration ({first.name})", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(1.0, 1.0), loc="upper left")
        fig.tight_layout()
        fig.savefig(plots / "verdict_mix.png", dpi=150)
        plt.close(fig)

    # final per-class pass share
    finals = [r / "final" / "results.jsonl" for r in runs if (r / "final" / "results.jsonl").exists()]
    by_cls: dict[str, list[float]] = {c: [] for c in CONTROL_CLASSES}
    for f in finals:
        for line in f.read_text().splitlines():
            res = ControlResult.model_validate_json(line)
            by_cls[res.cls].append(1.0 if res.verdict == "pass" else 0.0)
    if finals:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        names = [c for c in CONTROL_CLASSES if by_cls[c]]
        vals = [float(np.mean(by_cls[c])) for c in names]
        order = np.argsort(vals)
        ax.barh([names[k] for k in order], [vals[k] for k in order], color=SERIES[0], height=0.6)
        for y, k in enumerate(order):
            ax.text(vals[k] + 0.01, y, f"{vals[k]:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("pass share (final evaluation)", color=INK2)
        ax.set_title("Final evaluation: pass share by control class", color=INK, fontsize=11, loc="left")
        fig.tight_layout()
        fig.savefig(plots / "final_by_class.png", dpi=150)
        plt.close(fig)

    lines += ["| proposer | runs | final coverage reward (mean [95% CI]) | final reward | iterations |",
              "|---|---|---|---|---|"]
    for name, rs in sorted(groups.items()):
        summ = [json.loads((r / "final" / "summary.json").read_text()) for r in rs
                if (r / "final" / "summary.json").exists()]
        cov = bootstrap_ci(np.array([s["coverage_reward"] for s in summ]))
        rew = bootstrap_ci(np.array([s["reward"] for s in summ]))
        n_it = [s.get("iterations", 0) for s in summ]
        lines.append(f"| {name} | {len(rs)} | {cov[0]:.3f} [{cov[1]:.3f}, {cov[2]:.3f}] | {rew[0]:.3f} | "
                     f"{np.mean(n_it) if n_it else 0:.1f} |")
    lines += ["", "Plots: " + ", ".join(sorted(p.name for p in plots.glob("*.png")))]
    out = plots.parent / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out
