# Self-Evolving Arm (sim)

A simulated SO-101 arm runs coverage tests on procedurally generated appliance panels: buttons,
knobs, levers, doors, drawers, sliders and slots. When it fails, two kinds of reasoning take over:

- **System 1** is fast and local: scripted motor primitives, guarded contact, a fixed retry rule,
  and CMA-ES over a primitive's parameters.
- **System 2** is deliberate: Claude proposes typed programs, tools or body changes, each with a
  prediction. An escalation ladder applies, so it reaches for the cheapest fix that could work.

A Cobweb concept memory learns what controls look like and how they behave. That memory supplies
expectations, surprise and uncertainty, and it lets System 2 retrieve past fixes. A resource
ledger prices every change. A posture parameter β trades boldness against caution.

The full specification is [PLAN.md](PLAN.md). The theory is in [docs/theory.md](docs/theory.md).

## The harness is the product

The proposers search, and the harness judges. "Harness" here means the verifier, the reward,
the hard limits, the ledger, paired acceptance and the run logs. Nothing a proposer outputs can
change how it is judged.

- [docs/harness.md](docs/harness.md) covers the trust boundary, its invariants and the tests
  that pin them, and what a run directory contains.
- [docs/iteration.md](docs/iteration.md) covers how we change the fundamentals themselves:
  - the tiers (judge, world, skill, policy);
  - the observe → localize → predict → change → measure → record cycle;
  - per-fundamental playbooks;
  - the measurement batteries;
  - a change-record template.
- [docs/decisions.md](docs/decisions.md) records every place the code departs from or fills in
  the spec.
- [docs/gripper-experiment.md](docs/gripper-experiment.md) is the headline experiment for rungs 4
  and 5: a stock gripper learns, meets objects it cannot handle, and is changed through CAD. It
  also covers the validation ladder, sensing, the planned arm tool server (MCP) and plan billing.
- [docs/goals.md](docs/goals.md) is the goal and sub-goal design: System 2 as a planner over a
  hierarchical task network that assigns sub-goals to System 1, grounded in HTN planning, Soar,
  TOTE and hierarchical task analysis.

## Quick start

```bash
uv sync                                          # Python 3.12; MuJoCo, placo, CadQuery, Cobweb
uv run sea panels --split train --n 200 --seed 1
uv run sea run configs/exp/smoke.yaml            # offline: scripted System 2, 3 iterations
uv run sea reward runs/smoke_s1                  # standalone reward recompute (must match)
uv run sea report runs/smoke_s1                  # plots + markdown summary
export ANTHROPIC_API_KEY=...                     # then a real System 2 run:
uv run sea run configs/exp/e1_s2.yaml
uv run sea eval runs/e1_s2_s1 --split compose    # frozen design + library on another split
uv run sea replay runs/e1_s2_s1 --iter 5 --panel train_p012 --control c3   # MuJoCo viewer
```

Tests:

```bash
uv run pytest -m "not slow and not llm" --cov=sea --cov-branch --cov-report=json
uv run python scripts/check_package_coverage.py  # per-package gates (Section 12.1)
uv run pytest -m slow                            # verifier suite on real sim, golden, loop integration
```

## Layout

```
src/sea/
  contracts.py      Pydantic data contracts shared by every package (Section 6)
  sim/              MuJoCo model composition, control dynamics, panel generator, suite runner
  control/          IK (placo), reach grid, motor primitives, executor, System 1 local search
  compose/          program DSL, check grammar, typecheck, primitive library
  verify/           verdicts, coverage, standalone reward
  design/           tool templates (CadQuery), build checks, design space
  memory/           Cobweb concept memory (4 trees) and embeddings
  propose/          System 2 prompt, LLM clients (Anthropic, Mock, Scripted), ladder, targets, baselines
  spec/             posture (beta) and the resource ledger
  loop.py           the evolve loop and run log
  harness.py        component fingerprints (which judge produced a run)
  report.py, cli.py
configs/            limits, panels, primitives, IK seeds, experiments (configs/exp)
scripts/            jaw decomposition, reach measurement, golden verdicts, coverage gates
tests/              unit tests, verifier suite (tests/verifier), golden verdicts, LLM fixtures
```

## Status

| Milestone | State |
|---|---|
| M1–M3: contracts, sim, panels, verdicts, coverage, reward, primitives, executor, DSL, typecheck, System 1 | Implemented |
| M4: tools, build, ledger, posture | Implemented |
| M5: Cobweb memory, embeddings, targets | Implemented |
| M6: System 2 (Claude Code plan, Anthropic API, Mock, Scripted), escalation, ladder, baselines | Implemented |
| M7: library promotion and pruning | Implemented |
| M8: experiment configs and `sea report` | Implemented; experiments not yet run |
| M9: L1 fingers, L2 kinematics | In progress: scene objects, stock capability table (V1), sensing and the sensor manifest built; plan in docs/gripper-experiment.md, goal layer in docs/goals.md |
| M10: world model | Optional, not started |

Known limitations (wide doors, knob cup and the knob force-cap question, toggles, slot blade
alignment, high-tilt reach) are listed in [docs/harness.md](docs/harness.md) under "Where the
harness is still thin", with the evidence in [docs/decisions.md](docs/decisions.md).
