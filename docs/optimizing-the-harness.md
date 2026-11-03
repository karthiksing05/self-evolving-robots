# Optimizing the harness

How to make the loop find better bodies with fewer trials, LLM calls and dollars, without moving the
ruler. Each lever names what it changes, how to tell whether it helped, and what we have seen so far.
The rules for changing a fundamental ([iteration.md](iteration.md)) apply to every lever here:
- one lever per change;
- measured on frozen worlds;
- compared against the `random` and `s1` arms.

## What to measure

| Metric | Why |
|---|---|
| Groups held at the end, train and **test** | The result; the test split catches a body fitted to the train placements |
| Groups gained per System 2 call, per sim-hour, per dollar | What reasoning buys, against `random` under the same screen and trials |
| Refusals before trial, by reason | Wasted proposals: catalogue errors, unbuildable bodies, unaffordable or unprovided parts |
| Rejected trials, by reason (regression vs no gain) | Whether System 2 understands the trade-offs, or only the targets |
| Prediction accuracy (predicted gains vs actual) | Whether the reasoning is calibrated, independent of luck |
| Moves used (params, swap, idea) and when | Whether escalation follows evidence |

## Lever 1: what System 2 sees (information design)

The largest gains so far came from showing System 2 the right facts, not from a larger model.
- **Mechanisms with numbers, not symptoms.** "squeezed_down_out: moving jaw 28 N at +10 mm from the
  centre, pushing 48° down" names the fix; "slip" does not.
- **Computed facts about the current body** (the brief). These are numbers System 2 would otherwise
  guess, such as the fingertip rise at each opening or the reachable formant range.
- **Why things were refused or rejected.** In D-039, naming the regressed objects changed what
  System 2 tried next. In the first smoke run, refusals it could not see stalled it.
- **What parameter moves have not fixed, per slot.** This is the evidence that makes a redesign the
  reasonable next move.
- **Ideas to try:** the cost of each candidate before it is proposed (the catalogue's availability
  line already does this per part type), and predicted-vs-actual history, so System 2 can see its own
  calibration.

## Lever 2: what System 2 can do (action-space design)

- **Parts, not solutions.** The catalogue offers general parts per slot (joint types, a pad family,
  servo grades), including ones that cost and do not help. A catalogue that contains the answer as
  one obvious entry measures reading, not design.
- **Parameters that are physical and nearly independent**, with ranges that always build. Every
  sampled point is valid, so no trial is spent on a body the ranges could have excluded (D-038).
- **A move for what the catalogue lacks.** An `idea` is logged with its needs. Recurring ideas tell
  the outer loop which part to add next.
- **Ideas to try:** let System 2 combine moves across slots in one proposal (already allowed), and
  measure whether combined moves beat sequential ones under the ratchet.

## Lever 3: analysis before payment

Every refusal before a trial is a trial saved. The screen runs, in order: catalogue, build check,
resources, no-op. Ideas to try:
- **Analytic predictions per held group.** For example, the gripper brief's table already shows
  whether the fingertip reaches below an object's top at its opening. A cheap predicted-loss check
  could warn System 2 before a paid trial.
- **A dry-run tool.** One simulated attempt on a chosen task, with the diagnosis, so an agentic
  System 2 can test a hypothesis for one attempt's cost instead of a whole trial (the `sea-arm`
  analyze intent, [gripper-experiment.md](gripper-experiment.md) §5).

## Lever 4: the evaluation economy

- **Held groups first, stop at the first loss.** Most rejected candidates are rejected within a few
  tasks.
- **Cache by body content** (`Body.key`). The same body is never re-solved, even under another id.
- **Parallel workers** (`--workers`). Worlds are rebuilt per process from the registry.
- **Confirmation seeds** inside System 1. A lucky pass is not a skill, and a failure must repeat
  before it counts.
- **Ideas to try:** sample train placements per iteration, as the panel loop's `iter_sample` does, and
  keep the full set for acceptance.

## Lever 5: credit assignment

A trial's record names:
- the groups gained and lost;
- the mechanism of each loss;
- the mechanisms still failing.

That is the difference between "rejected" and "rejected because the cube now slips out of a 40°
pad". Ideas to try: attribute a gain to the slot that changed when a proposal edits several slots, by
trialling the edits separately, but only when the combined proposal is kept.

## Lever 6: task validity and calibration

Before trusting any run in a world:
- the stock body fails some groups for physical reasons;
- a gold body from the catalogue, within the budget, holds every group;
- no params-only body holds everything, if the world is meant to need a redesign.

The world tests check all three. A world that fails them measures nothing.

## Lever 7: pacing and budgets

- Iterations, dollars, lead days and machine minutes are separate budgets, each reported.
- A run stops when every group is held.
- On the Claude plan, the limit is the usage window. `PlanLimitReached` stops a run cleanly, and run
  seeds sequentially.

## Lever 8: protecting innovation

A swap's parameters are a guess, while the current body's have been tuned for several iterations.
Judging the two head to head makes the loop conservative for the wrong reason. The tools:
- the frontier (repair a promising body with `base`);
- the tuning allowance (`tune_trials` full trials fitting the slots a promising swap changed).

Measure the allowance with `--tune-trials 0` against the default, under the same seeds.
- *Seen so far:* no effect on the random arm in the mobile world. Over 30 seeds at 10 and 20
  iterations, outcomes were identical (6.47 and 7.43 of 10 groups) for about 4% more trials. Blind
  swaps rarely gain a group while breaking a held one, which is what triggers tuning. The allowance is
  for directed swaps; System 2 runs measure it there.

## Evidence so far (2 November 2026)

The frozen campaign `results/evolve/c1/` ran one commit (`7efc6b9`) on two worlds. Each world had
three System 2 seeds, 30 random-arm seeds and the stock body:

| World | System 2: groups held, train / test | Random: train / test (solved) | Stock | System 2 cost |
|---|---|---|---|---|
| mobile (10 groups) | 10/10, 10/10 on 3 of 3 seeds | 6.5, 6.3 (0 of 30) | 4 | 2 calls, 5–8 trials per seed |
| phonology (13 groups) | 13/13, 13/13 on 3 of 3 seeds | 11.0, 10.2 (7 of 30) | 6 | 1 call, 4 trials per seed |

- **Information design (lever 1)** carried most of it. In the mobile world, System 2 ruled out wheels
  across their whole range from the brief ("r = 4.5 cm climbs about 3.35 cm"). It ruled out the
  rocker-bogie from the sofa clearance, and chose tracks with an idler between the porch edge and
  the sofa.
- **Merging (D-045)** turned three kept-eligible single-slot proposals into one body in both worlds.
  In phonology the union also gained a group neither part gained alone.
- **Tuning (D-044)** has not yet produced a keeper in a live run. In 7 dimensions, perturbing every
  parameter at once broke a different held group each trial, so steps are now sparse.
- **Task validity (lever 6)** mattered most for the gripper. The diagnosis's own numbers, a 50 g plum
  sliding out at 30 N, exposed soft-contact creep in the simulator (D-048). It had made round objects
  fail for a reason no design could fix.
- **The gripper** under the same loop with the frontier, full swap trials and tuning
  (`gripper_s2v2_s0`, old physics) went from 6 to 19/21 in 4 iterations. The first loop
  (`gripper_s2_s0`) reached 13/21 in 10.
- **On the corrected physics** (commit `61129fd`), all three System 2 seeds held 21/21 train groups.
  They took 4, 2 and 11 iterations and 4, 2 and 11 calls, and reached 19, 20 and 21 of 21 on test.
  The stock body held 13–15 on test. The random arm, given the same screen and trials, ran three seeds
  of 14 iterations: 17, 20 and 16 train, and 15, 20 and 15 test. Its 20 came from a seed that
  happened onto the slide. The gripper's catalogue is small, so blind search can find the redesign.
  What reasoning bought was reaching all 21 every time, in 2–11 iterations, and about 3 more test
  groups on average.
  - Every seed arrived at the same redesign, a slide with flat TPU pads.
  - In seed 0, System 2 built from a tuned frontier body. It also read the tuning trials as a
    sensitivity map ("potted meat held at reach ≤ 5 mm, lost at ≥ 8; the mustard bottle needed ≥ 8")
    and chose 6.5 mm.
  - None of the tuning trials was kept, but they were the cheapest experiments in the run.
- **Generalization:** seed 0 fitted its fingertip reach to the train placements. It lost the potted
  meat can and the tuna can at one test placement each: a force limit, and a jaw landing on top.
  `gripper_dense` (four train placements per object, same test) measured whether denser training
  closes that gap. With one seed it is inconclusive:
  - it reached 21/21 on all four placements by iteration 8, with 8 calls;
  - it held 20/21 on test, missing the mustard bottle at one placement;
  - that is inside the 19–21 range of the two-placement seeds.

  Separating the two would take several seeds of each.
- **The harder worlds, under the mechanism laws** (`results/evolve/mobile_randmj.json`,
  `phonology_randsnd.json`; 30 random seeds each, at most 10 iterations, same screen and trials):
  - mobile, 13 routes in MuJoCo: random held 8.8 train (7–12) and 8.3 test, solving 0 of 30. Patio
    held on no seed, porch and sandbox on 7. System 2 held 13/13 on all three seeds in 2–3
    iterations (test 13, 12, 13). The stock body holds 6.
  - phonology, 17 sounds: random held 12.8 train (7–17) and 11.7 test, solving 2 of 30; /ta/ held
    on 7. System 2 held 17, 16 and 17 in 9, 10 and 2 iterations, 15 test on every seed. The stock
    holds 6.
  - The gap is widest on the routes and sounds that need several slots changed together: the patio
    (tracks, pack and clearance) and /ta/.

## Failure modes to watch

- **Repetition:** the same refused or rejected change proposed again. The record is in the prompt;
  count repeats.
- **Fitting the train placements:** compare train and test groups for the final body.
- **Escalating too early:** swaps before parameter moves have been tried on a slot. Count moves per
  slot before the first swap.
- **Moving the ruler:** any edit to a world's tasks, physics or catalogue re-keys its runs.
  Re-validate with the gold check before comparing.
