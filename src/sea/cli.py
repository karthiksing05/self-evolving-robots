"""``sea`` command line (spec 8.10)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Self-evolving arm (sim)")


@app.command()
def panels(split: str = typer.Option("train"), n: int = typer.Option(200), seed: int = typer.Option(1),
           out: Path | None = typer.Option(None, help="write the panels as JSON lines here")) -> None:
    """Generate panels and print a per-class summary."""
    from sea.sim.panels import generate, reference_panel

    ps = [reference_panel()] if split == "reference" else generate(split, n, seed)
    counts: dict[str, int] = {}
    for p in ps:
        for c in p.controls:
            counts[c.cls] = counts.get(c.cls, 0) + 1
    if out is not None:
        out.write_text("".join(p.model_dump_json() + "\n" for p in ps))
    typer.echo(json.dumps({"split": split, "panels": len(ps), "controls_by_class": counts}, indent=1))


@app.command()
def run(spec_path: Path, run_dir: Path | None = typer.Option(None, help="defaults to runs/<spec stem>_s<seed>"),
        seed: int | None = typer.Option(None, help="override the spec seed")) -> None:
    """Run the loop for one experiment spec."""
    from sea.loop import evolve, load_spec

    spec = load_spec(spec_path)
    if seed is not None:
        spec = spec.model_copy(update={"seed": seed})
    run_dir = run_dir or Path("runs") / f"{spec_path.stem}_s{spec.seed}"
    out = evolve(spec, run_dir, log=typer.echo)
    typer.echo(json.dumps({k: out[k] for k in ("coverage_reward", "reward", "stop_reason", "iterations")}))


@app.command("eval")
def eval_(run_dir: Path, split: str = typer.Option("test")) -> None:
    """Evaluate a finished run's final design and library on another split."""
    from sea.loop import evaluate

    out = evaluate(run_dir, split)
    typer.echo(json.dumps({k: out[k] for k in ("split", "coverage_reward", "reward", "control_coverage")}))


@app.command()
def reward(run_dir: Path) -> None:
    """Recompute the reward from the logs alone (standalone; must match the loop)."""
    from sea.verify.reward import main

    typer.echo(json.dumps(main(run_dir), indent=1))


@app.command()
def replay(run_dir: Path, iter_: int = typer.Option(0, "--iter"), panel: str = typer.Option(...),
           control: str = typer.Option(...), viewer: bool = typer.Option(True)) -> None:  # pragma: no cover - GUI
    """Re-run one control from a logged iteration (opens the MuJoCo viewer)."""
    from sea.compose.library import Library
    from sea.contracts import Design, PrimitiveDef
    from sea.loop import load_spec
    from sea.sim.panels import generate, reference_panel

    spec = load_spec(run_dir / "spec.yaml")
    design = Design.model_validate_json((run_dir / f"design_{iter_:03d}.json").read_text())
    prims = [PrimitiveDef.model_validate(d) for d in json.loads((run_dir / f"library_{iter_:03d}.json").read_text())]
    split = panel.split("_p")[0]
    pool = [reference_panel()] if panel.startswith("reference") else generate(split, spec.splits.get(split, 200),
                                                                             spec.panel_seed)
    p = next(x for x in pool if x.id == panel)
    from sea.design.build import build
    from sea.sim.env import PanelRunner, SuiteConfig, run_control

    built = build(design, spec.limits)
    cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library({d.name: d for d in prims}),
                      limits=spec.limits, seed=spec.seed, confirm=False)
    runner = PanelRunner(p, cfg, repeat=0)
    c = next(x for x in p.controls if x.id == control)
    if viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(runner.sim.model, runner.sim.data) as v:
            from sea.control import executor

            orig = executor.Episode.tick

            def tick(self, q):  # type: ignore[no-untyped-def]
                orig(self, q)
                v.sync()

            executor.Episode.tick = tick  # type: ignore[method-assign]
            out = run_control(runner, c)
            executor.Episode.tick = orig  # type: ignore[method-assign]
    else:
        out = run_control(runner, c)
    typer.echo(out.result.model_dump_json(indent=1))


@app.command()
def report(run_dir: Path, allow_mixed: bool = typer.Option(False, help="pool runs judged by different "
                                                                       "harness versions")) -> None:
    """Plots and a markdown summary for one run (or a directory of runs)."""
    from sea.report import write_report

    path = write_report(run_dir, allow_mixed=allow_mixed)
    typer.echo(str(path))


if __name__ == "__main__":  # pragma: no cover
    app()
