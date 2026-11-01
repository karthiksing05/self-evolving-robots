# Iterating on fundamentals

There are two loops in this project.

- **The inner loop** is automatic. Under fixed fundamentals, System 1 tunes parameters and
  System 2 proposes programs, tools and body changes. The harness accepts or rejects each
  one ([harness.md](harness.md)).
- **The outer loop** is us. We change the fundamentals the inner loop stands on: the
  verifier, the physics, the primitives, the tool templates, the panel distributions, the
  reward, the posture table and the prompt.

The outer loop decides how far the system can go. A better System 2 can't beat a verifier
that calls the wrong things a pass, or primitives too weak to express the fix. But the outer
loop is also where results get quietly corrupted, because it is the loop that can move the
ruler. This document is the discipline for running it.

## 1. Principles

1. **Separate the robot from the ruler.**
   - Every change is either to *what is measured* (judge tier) or to *what is measuring*
     (everything else).
   - A judge-tier change invalidates comparisons with earlier runs. `harness.json` records
     a `judge_key`, and `sea report` won't pool runs with different keys.
   - After a judge change, re-run the baselines; never compare across it.
2. **One fundamental per change.**
   - Change one tier at a time and measure it alone.
   - If a physics fix and a primitive fix land together and coverage rises 8 points, nobody
     knows which one did it, or whether one of them hurt.
3. **Measure on frozen instruments, not on the evolving loop.**
   - Judge a fundamental by its effect on fixed batteries (§5): golden verdicts, the
     planted-fault suite, and E0 on fixed-seed panels.
   - The inner loop adds variance (proposals, acceptance), which hides small effects and
     invents large ones.
4. **Write the prediction before the change.**
   - Write what should move, by how much, and what might regress, the same way System 2's
     `Prediction` names `flips_to_pass` and `may_regress`.
   - A change that helps for a reason other than the predicted one isn't understood yet.
5. **Make the failure legible before fixing it.**
   - Find the layer (§3) before touching anything.
   - Most "the robot can't do X" bugs this project has hit were in the world or skill tier,
     not where they first looked.
6. **Let the loop discover; lift with evidence.**
   - Hand-written fundamentals (a new primitive, a new template) should be rare.
   - Prefer letting System 2 compose existing pieces and letting library learning promote
     what recurs (at least `k_promote` controls across 2 panels).
   - Lift a pattern into a fundamental only when the loop keeps rediscovering it, or
     provably can't express it.
7. **Paired, seeded, with intervals.**
   - Claims need 5 seeds and 95% bootstrap intervals.
   - Single-seed numbers are for debugging only.
   - A change shows its effect on paired draws (same panels, same randomization).
8. **Every change is reversible and recorded.**
   - One commit per change, plus an entry in [decisions.md](decisions.md) using the template
     in §6.
   - Re-recorded fixtures (golden verdicts, reach grid, IK points) get committed in the same
     commit as the change that moved them.

## 2. The tiers

| Tier | Fundamentals | Files | Blast radius | Minimum evidence |
|---|---|---|---|---|
| **judge** | verdict rules and thresholds, success checks, reward and cost terms, hard limits, panel and grasp-world task distributions and splits, acceptance rule, build checks and print pricing, contract defaults | `verify/`, `compose/checks.py`, `spec/posture.py`, `design/build.py`, `contracts.py`, `configs/panels.yaml`, `configs/grasp.yaml`, `configs/limits.yaml`, `configs/limits_grasp.yaml` | Every past number; comparability | B0–B4, and re-baseline all compared proposers |
| **world** | contact model, control dynamics, jaw and tool collision geometry, scene objects, sensor models, IK, reach | `sim/`, `configs/sensors.yaml`, `control/ik.py`, `control/reach.py`, `control/frames.py` | Which tasks are possible at all | B0–B3, golden diff explained control by control |
| **skill** | motor primitives, seed programs, program language, executor behaviour, tool templates | `control/motor.py`, `control/executor.py`, `compose/dsl.py`, `configs/primitives.yaml`, `design/tools.py`, `design/space.py` | What the robot can do before learning | B0–B3; no reference regressions |
| **policy** | posture table, ledger prices, escalation thresholds, target scoring, prompt, LLM client, memory encoding, library promotion | `spec/ledger.py`, `propose/`, `memory/`, `compose/library.py` | How efficiently the loop searches | B0, B4, and E1/E8 slices over 5 seeds |

`sea/harness.py` lists the files behind each component, and each component's hash is
written into every run.

## 3. The cycle

```
observe ─► localize ─► predict ─► change one thing ─► measure on batteries ─► record ─► keep or revert
   ▲                                                                                        │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

**Observe.**
- Start from logs, not intuition:
  - `sea report` (verdict mix by iteration, pass share by class);
  - `iter_<k>/proposal.json`, which says why candidates were rejected;
  - the escalation reasons in `results.jsonl`.
- Then open the traces.
- `sea replay --iter k --panel P --control C` shows one control in the viewer.

**Localize.** Use the symptom to pick the layer before forming a fix.

| Symptom | Likely layer | How to check |
|---|---|---|
| Verdict disagrees with what the replay shows | judge (verifier) | Build a synthetic trace of the case; add it to `tests/test_verdicts.py`; the precedence in Section 8.2 is usually the culprit |
| A control flips pass/fail across randomization draws | world (stiff contact, guard thresholds) | Run the same control over 5 DR seeds and plot tip force vs control q; look for force spikes and stalls |
| Force spikes far above the cap on first contact | skill (guarded motion step size) or world (contact stiffness) | Timeline of commanded vs actual tip and force per tick (`guarded_step`) |
| Unrelated controls are touched (`contacts` names another control) | skill (approach path, tool swap pose) | List contact geom pairs per tick; check the path from the previous pose |
| A tool can never engage a handle | skill (engage point) or design (tool parameters) | Dump world bounding boxes of tool parts and handle geoms at engagement |
| System 2 proposals keep failing typecheck | policy (prompt or signatures) or skill (missing primitive) | Read `proposal.json` errors; if the needed effect is inexpressible, it's a skill gap |
| Escalations are mostly `unfamiliar` late in a run | policy (memory encoding, `n_fam`) | Check `concept_n` against how many same-class results exist; is the context text discriminative? |
| Reward rises while reference controls fail more | judge (reward or regression check) | Compare with the E7 runs; check that the regression set is the full panel |
| Accepted changes don't hold on re-run | policy (acceptance rule too loose) | Prediction scores in `decision.json`; rerun with `lcb` |
| Coverage stalls with budget left | policy (value per cost, stop rule) or skill | Are candidates rejected (screening) or tried and rejected (acceptance)? |

**Predict.** Write the change record (§6) *before* running anything: expected movement,
possible regressions, and which battery will show it.

**Change one thing.** One tier, one fundamental. If the fix needs two, land and measure them
separately, the enabling change first.

**Measure.** Run the batteries for that tier (§5). Compare with the last recorded baseline on
the same `judge_key` (for judge changes, re-baseline instead).

**Record.** Fill in the change record's results. Commit the code, the re-recorded fixtures and
the decision entry together.

**Keep or revert.**
- Keep when the predicted movement shows up with an interval that excludes zero and nothing
  in "may regress" did.
- Revert when the change helped for an unpredicted reason. Re-diagnose before trying again.

## 4. Playbooks

Each playbook covers when to change the fundamental, how to change it safely, and what has to
be regenerated afterwards.

### 4.1 Verifier rules (judge)
- **When.** A replay clearly contradicts a verdict, or a planted fault is misjudged.
- **Do.**
  1. Write the failing synthetic trace first, in `tests/test_verdicts.py`.
  2. Change the smallest rule that fixes it, keeping the Section 8.2 precedence.
  3. If you add a fault class, add a planted panel to `tests/verifier/` with its declared verdict.
- **Evidence.** The verifier suite passes 10 of 10; fault-free twins pass at least 95% over
  5 seeds; golden diff explained.
- **Regenerate.** `scripts/record_golden.py --write`. Re-run every baseline you intend to compare.

### 4.2 Physics, contact and control dynamics (world)
- **When.** Behaviour that is physically implausible: penetration, jitter, stiffness spikes,
  latches that never release, controls that move without contact.
- **Do.**
  1. Reproduce on one control with a timeline (commanded vs actual tip, force, contact pairs).
  2. Change the model, not the controller. Never tune physics until a particular program
     passes, because that fits the world to the robot.
  3. Prefer changes with a physical justification (a real stiffness, a real geometry) and
     write it down.
- **Evidence.** `test_controls.py` (detent dips, latch thresholds, push-latch at 2 mm) and
  `test_model_consistency.py`. Golden diff explained per control. E0 coverage by class with
  intervals.
- **Regenerate.** Golden verdicts. If geometry near the tool frame changed, the reach grid
  and IK fixtures (`scripts/measure_reach.py`).

### 4.3 Kinematics and reach (world)
- **When.** `unreachable` verdicts that the viewer shows are reachable, or the reverse; a
  placo or model update.
- **Do.** Re-measure with `scripts/measure_reach.py`. It prints the Appendix C table and
  rewrites `tests/fixtures/reach_stock.npz`.
- **Evidence.** `test_ik.py` and `test_reach.py` against the Appendix C points. If a point
  changes, write a decision entry; that is the spec's own rule.
- **Regenerate.** Reach grids under `.cache/reach` (keyed by URDF, tip and limits hash, so they
  refresh automatically).

### 4.4 Motor primitives, seed programs, executor (skill)
- **When.**
  - A failure mode recurs across classes and System 2 can't express the fix with existing
    primitives (typecheck errors in `proposal.json`, or rung-1 proposals that never pass).
  - Or the executor does something no program asked for (path collisions, tool swap poses).
- **Do.**
  1. Before writing a new primitive, check whether a composition of existing ones works,
     with a short script that calls `control.executor.run_program` on the failing control.
  2. If it does, add a seed *program* to `configs/primitives.yaml`, not a primitive.
  3. A new primitive needs:
     - declared `requires`, `provides` and params with ranges;
     - guarded force (stop at the cap, at travel, or at a joint limit);
     - speed under the hard limit, and half speed within 10 mm of contact;
     - its extent defaulting to the control's travel (D-009).
- **Evidence.** `test_motor.py` (cap within 0.5 N, speed limits, 30 mm approach offset) and
  `test_typecheck.py`. E0 by class: the target class improves and no class regresses beyond
  its interval.
- **Regenerate.** Golden verdicts. Integration fixtures if the prompt's library section changed
  (§4.10).

### 4.5 Tool templates (skill) and build checks (judge)
- **When.** System 2 asks for a geometry no template can express, or build checks reject
  physically fine tools (or accept bad ones).
- **Do.**
  1. Extend a template's parameters before adding a template.
  2. Keep the 20 × 20 mm mount and the Appendix B tip transforms.
  3. Keep collision parts convex and analytic (B-002), so hooks keep their throats open.
- **Evidence.** `test_build.py` at min, mid and max of every template. Build a tool that
  should fail (thin wall, over mass) and see it rejected.
- **Regenerate.** Nothing. Tool builds are cached by spec hash. A change to `design/build.py` is
  judge tier (D-028): it re-keys runs, so re-run B3 and re-baseline.

### 4.6 Panel distributions and splits (judge)
- **When.** The train distribution misses a real appliance pattern, or a split leaks
  (compose holdouts appear in train).
- **Do.** Change `configs/panels.yaml`. Holdout combinations live in `compose_holdouts`, and
  the test split widens upper bounds by 25%.
- **Evidence.** `test_compose_split.py`. Class balance at n ≥ 30 (`sea panels --split train`).
- **Regenerate.** Everything comparative. A new distribution is a new benchmark: re-run E0 and
  all baselines. Never change the test or compose splits after looking at results on them.

### 4.7 Reward and cost terms (judge)
- **When.** E7 shows degenerate designs (probe length or mass pinned at the maximum), or the
  reward ranks runs against judgement.
- **Do.** Change `verify/coverage.py` and keep `reward.py` standalone (it imports only
  `contracts` and `coverage`).
- **Evidence.** `test_coverage.py`, `test_reward_audit.py`, and the E7 pair before and after.
- **Regenerate.** Re-baseline every run you compare. Old `report.json` values aren't
  comparable.

### 4.8 Posture table and acceptance (policy, or judge for acceptance)
- **When.** Too many reversed accepts (optimist) or a stalled search (pessimist) at equal
  budget.
- **Do.**
  - Change `TABLE` in `spec/posture.py`, or override per run with explicit `Posture` fields
    first (they take precedence over β).
  - Move a table row only after a sweep shows it.
- **Evidence.** E8: β in {−2, −1, 0, 1, 2}, fixed and annealed, 5 seeds. Report final coverage,
  reversed accepts and regressions. `test_posture.py`.
- **Regenerate.** Nothing; but the acceptance rule is judge-tier, so its change bumps `judge_key`.

### 4.9 Ledger prices and budgets (policy)
- **When.** Estimate vs actual error grows (`ledger.history`), or value per cost picks
  obviously wrong rungs.
- **Do.**
  - Prefer refitting constants from logged actuals (the ledger refits print time every
    10 proposals).
  - Change heuristics such as `cost_est` with a sweep (E9).
- **Evidence.** `test_ledger.py`; E9 slices.

### 4.10 System 2 prompt and schema (policy)
- **When.** Proposals ignore the ladder, cite evidence that isn't in the prompt, or keep
  proposing what `proposal.json` shows was already rejected.
- **Do.**
  1. Change `propose/prompt.py`. The prompt hash keys `MockLLM` fixtures, so every prompt
     change invalidates recorded fixtures.
  2. Re-record them by running with `llm: scripted` and `llm_fixtures: <dir>`, or with
     `llm: anthropic` and `llm_fixtures: <dir>` (each live answer is written as a fixture),
     then review the diff.
- **Evidence.** `test_llm_mock.py`, `test_escalation.py`; E1-s2 against E1-s2_nomem over 5 seeds;
  rejection rate by reason in `proposal.json`.
- **Regenerate.** `tests/fixtures/llm/`.

### 4.11 Memory encoding (policy)
- **When.** Surprise fires on routine results, familiarity never accumulates, or retrieved
  fixes are irrelevant.
- **Do.** Change the context text (`context_text`), the whitening dimension, or `n_min` and
  `n_fam`. Never feed measurements into the context tree; that's what the full tree is for.
- **Evidence.** `test_memory.py` (cluster separation, z-scores); E2 and E4 slices.

## 5. Batteries

| Battery | Command | Runtime | Used by tiers |
|---|---|---|---|
| B0 unit gates | `uv run pytest -m "not slow and not llm" --cov=sea --cov-branch --cov-report=json && uv run python scripts/check_package_coverage.py` | under 10 min | all |
| B1 golden diff | `uv run python scripts/record_golden.py` | about 1 min | judge, world, skill |
| B2 verifier suite | `uv run pytest tests/verifier -m slow` | minutes | judge, world |
| B3 E0 slice | `uv run python scripts/battery.py` (16 fixed panels per split, per-class pass share with intervals, diffed against `tests/golden/battery_baseline.json`; refuses a baseline from another judge key, panel count, rack or System 1 budget); the full E0 is `sea run configs/exp/e0.yaml` plus `sea eval --split …` | about 15 s for the script; 5–30 min for the full E0 | judge, world, skill |
| B4 loop audit | `uv run sea run configs/exp/smoke.yaml && uv run sea reward runs/smoke_s1` and `pytest tests/test_loop_integration.py -m slow` | minutes | judge, policy |
| B5 policy slice | the relevant E-configs × 5 seeds (`sea run ... --seed k`), then `sea report runs/<dir>` | hours | policy |

Keep the latest B1 and B3 outputs next to the decision entry they justify. They are what the
next change gets compared against.

## 6. Change record (copy into decisions.md)

```markdown
## F-<nnn> <one-line title>
Tier: judge | world | skill | policy.  Component(s): <names from sea/harness.py>.
Observed: <symptom, with run id / iteration / control, or test name>.
Localized to: <layer and the evidence that ruled out the others>.
Prediction: <what moves, by how much, on which battery>.  May regress: <...>.
Change: <what changed and why it is physically or logically justified>.
Result: <battery numbers before → after, with intervals; golden diff, control by control>.
Decision: keep | revert.  Re-recorded: <fixtures>.  Re-baselined: <runs, if judge tier>.
```

## 7. Anti-patterns

- **Tuning the verifier until the run looks good.** If a verdict is wrong, prove it with a
  synthetic trace and a replay first.
- **Fitting the world to the robot.** Physics changes need a physical justification, not
  "now the hook passes."
- **Solving one control with a new primitive.** A primitive has to earn its place across
  classes or panels. Otherwise it's a program, and the loop can find programs.
- **Peeking at test or compose splits** before freezing a change.
- **Comparing across judge keys,** or reporting a single seed as a result.
- **Two tiers in one commit.**
- **Re-recording fixtures without reading the diff.** A re-record is a claim that every
  changed line is intended.

## 8. Worked examples from this repo

These came up while building the seed system, and they show the cycle on real failures.

**The hook that hit the lever at mount time (skill tier).**
- *Observed:* the tab-door trace showed 11–14 N of force in the first half second, before the
  hook had gone near the door.
- *Localized:* the contact listing named `ctl_c3_0_bar`, the *lever* next door. The tool was
  mounted at the home pose, where a 45 mm hook pokes into the lever.
- *Change:* tool swaps happen at a raised swap pose (`executor.SWAP_TIP`, covered by the swap's
  5 s). Unrelated contact forces disappeared from every tool-mounted trace.
- *Lesson:* look at which bodies are touching before tuning force guards.

**The tab the hook couldn't catch (skill tier, and a design signal).**
- *Observed:* the hook stopped at 3 N on the tab, and the door never unlatched.
- *Localized:* world-space bounding boxes at engagement. The 11 mm pocket under the tab lip
  can't fit a 3 mm foot plus an 8 mm barb, and the engage point placed the barb top flush
  against the lip.
- *Change:* the engage point now puts the foot 1 mm above the plate and centres the barb in
  the pocket.
- *Result:* hooks with throat ≤ 6 mm open the tab door to about 90%; throat 8 mm still fails.
- *Lesson:* the remaining failure is *correct*. It depends on a design parameter the loop can
  change (rung 3, `hook.throat`), so it belongs to the inner loop, not to us.

**Hook pull stalling near 77° (world/skill, left open).**
- A fixed-orientation hook binds against the fin once the door passes about 70°.
- Rotating the tool with the door (tried) and adding a radial pull component (tried) both made
  it worse.
- It is recorded as a known limitation (D-013) instead of being tuned around. A program that
  re-grips or pushes the door from inside is a rung-1 fix the loop can discover.
