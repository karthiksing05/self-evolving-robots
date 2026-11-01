# The harness

The harness is everything that decides what is true about a run: the simulator, the verifier,
the reward, the hard limits, the ledger, the acceptance test and the logs. System 1, System 2
and the baselines are search processes that run *inside* it. They can propose anything, but
they can't change how their proposals are judged.

That split is the design. A self-improving system is only as good as the thing that tells it
whether it improved. If a proposer can move the ruler, the loop will learn to move the ruler.
So the harness is built so that its numbers can be trusted without trusting any proposer.
[iteration.md](iteration.md) covers the other half: how *we* change the harness itself.

## What runs where

```
 panels (seeded)          System 1 (per control)                     judge (verifier)
 ───────────────►  typecheck ─► executor ─► fixed retry ─► CMA-ES  ──► verdict + measurements
                                 ▲ hard limits enforced here                     │
                                 │                                                ▼
   commit ◄── accept ◄── paired trial ◄── screen + price ◄── proposer ◄── escalation + targets
     │      (posture)   (same DR draws,   (ladder, posture,  (S2 / baseline)  (memory: surprise,
     │                   reference panel)  typecheck, build,                   familiarity)
     ▼                                     ledger, value/cost)
 run log ──► sea reward (standalone recompute) ──► must equal the in-loop reward
```

| Stage | Module | Trusted? | Notes |
|---|---|---|---|
| Panel generation | `sim/panels.py` | judge | Seeded; splits and holdouts are fixed per `panel_seed` |
| Physics | `sim/build_model.py`, `sim/controls.py` | world | Domain randomization seeded from (seed, panel, repeat) |
| Primitive execution | `control/executor.py`, `control/motor.py` | enforces limits | Force, speed and workspace limits are checked every tick |
| Verdicts | `verify/verdicts.py` | judge | Section 8.2 precedence; tested as a product (`tests/verifier/`) |
| Reward | `verify/coverage.py`, `verify/reward.py` | judge | `reward.py` imports only `contracts` and `coverage` |
| Screening | `propose/ladder.py`, `compose/typecheck.py`, `design/build.py` | judge | Runs before any simulation or spend |
| Costing | `spec/ledger.py` | judge | Cost before action; 20% reserve; estimates vs actuals refit |
| Acceptance | `spec/posture.py` `accept` | judge | Paired bootstrap on per-control deltas; any reference regression vetoes |
| Proposals | `propose/*` | **untrusted** | Typed `Proposal` objects only; never read as limits or code |

## Invariants

Each invariant names where it is enforced and which test holds it in place. If you change
any of these files, run the listed test before anything else.

1. **Proposers can't touch the judges.**
   - Hard limits come from the run spec, never from LLM output. `HardLimits` is not part of
     any proposal schema.
   - Check expressions are parsed by a hand-written grammar; there is no `eval` anywhere in
     `src/sea`.
   - Parameter ranges above a hard limit are rejected by typecheck, and again at run time by
     the executor.
   - Tests: `test_checks.py` (no-`eval` grep), `test_typecheck.py`.
2. **The reward can be recomputed from the logs alone.**
   - `sea reward runs/<id>` reads `spec.yaml`, `design_<k>.json`, `iter_<k>/results.jsonl`
     and `cost.json`, and must reproduce `iter_<k>/report.json` bit for bit.
   - Test: `test_reward_audit.py`.
3. **Runs are deterministic.**
   - Every random draw comes from a seeded generator: panels, domain randomization, CMA-ES,
     bootstrap, Cobweb insertion order, the context encoder's whitening fit, and the
     baselines' samplers. Library code that reads numpy's global RNG runs inside a
     seed-and-restore block (decisions.md D-026).
   - The same spec and seed give identical logs.
   - Test: `test_loop_integration.py`.
4. **Comparisons are paired.**
   - A trial and its base run on the same panels and the same randomization draws.
   - The acceptance interval comes from per-control deltas, not from two independent
     averages.
5. **Regressions veto.**
   - A reference-panel control that passed in the base and fails in the trial blocks
     acceptance, whatever the mean says.
   - The E7 ablation (`reference_regression: false`) exists to show what happens without this
     veto.
6. **Cost before action.**
   - Every proposal is priced before it runs.
   - A proposal that would eat into the 20% reserve is refused. The reserve is only spent by
     the final evaluation.
   - Test: `test_ledger.py`.
7. **The verifier is tested like a product.**
   - The planted-fault panels in `tests/verifier/` each have a declared verdict.
   - Fault-free twins must pass at least 95% of the time over 5 seeds.
   - Golden verdicts on the reference panel catch silent drift: `tests/golden/`,
     `scripts/record_golden.py`.
8. **No live LLM in tests.**
   - `MockLLM` replays fixtures keyed by the prompt hash.
   - `ScriptedLLM` is an offline System 2 that follows the ladder. It can record new fixtures.
9. **Every deviation from the spec is written down** in [decisions.md](decisions.md), with
   the reason.
10. **Every run says which harness judged it.**
    - `runs/<id>/harness.json` holds a per-component fingerprint plus a `judge_key`.
    - `sea report` refuses to pool runs with different judge keys (`--allow-mixed` overrides
      this, on purpose).

## The run directory is the source of truth

```
runs/<id>/
  spec.yaml                 the RunSpec (limits, posture, budgets, proposer, seeds)
  harness.json              component fingerprints + judge_key (sea/harness.py)
  design_<k>.json           design in effect during iteration k (k = N: final)
  library_<k>.json          primitive library in effect during iteration k
  iter_<k>/
    results.jsonl           one ControlResult per control (verdict, measurements, surprise, escalation)
    report.json             coverage, pass rate, coverage reward, reward, cumulative cost at report time
    cost.json               {cumulative, iteration, end, remaining}
    proposal.json           escalations, targets, every candidate with its screening errors, cost, value, vpc
    decision.json           the acceptance decision (rule, delta and interval, regressions, prediction score)
    llm_calls.jsonl         prompt hash, model, tokens, USD per System 2 call
    traces/*.npz            per-control time series (joint q, tip force, control q, contact flags)
  memory/                   Cobweb trees (dump_json) + sidecar.json (labels, payloads)
  final/                    report.json, results.jsonl, cost.json, design.json, library.json, summary.json
  timing.json               wall-clock seconds (never in report.json or cost.json)
```

Anything the loop decided can be reconstructed from these files. `sea replay` re-runs one
control from any iteration in the MuJoCo viewer.

## Running it

```bash
uv sync
uv run sea run configs/exp/smoke.yaml          # offline: scripted System 2, 3 panels, 3 iterations
uv run sea reward runs/smoke_s1                # recompute the reward from the logs
uv run sea report runs/smoke_s1                # plots + markdown summary
uv run sea run configs/exp/e1_s2.yaml          # live Claude (needs ANTHROPIC_API_KEY)
```

## Where the harness is still thin

These are known gaps. Each one makes a verdict less trustworthy in a specific way.

| Gap | Effect | Tracked in |
|---|---|---|
| Hook pull on 80–90° doors stalls near 77° | Wide doors read `stuck` even with a good hook | decisions.md D-013 |
| Knob cup barely turns knobs | Knobs read `stuck` or `no_actuation` | decisions.md D-013 |
| Knob class cap bounds total tip force | A cup turning a knob by its pointer needs more lateral force than the 10 N cap, so it reads `rule_violation`; whether rotary caps should bound only the axial push is an open judge-tier decision | decisions.md D-015 |
| Slot blade alignment | The blade rarely seats in train-panel slots, so a good `slot_inserter` + program is usually rejected in trial | decisions.md D-013 |
| 5-DOF reach on tilted panels | Above about 30° tilt most controls are `unreachable` for the stock arm; only L2 kinematics can change that | decisions.md D-025 |
| Toggle reaches about 2/3 of its throw | Toggles read `stuck` or `wrong_behavior` | decisions.md D-013 |
| L1 fingers and L2 kinematics | Proposals are rejected until `design/arm.py` lands (M9) | decisions.md D-019 |
| World model (3.6) | Not built (M10, optional) | — |

Fixing physics gaps changes the *world* tier, so it moves the golden verdicts. Follow the
procedure in [iteration.md](iteration.md) §4.2.
