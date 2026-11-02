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

## Failure modes to watch

- **Repetition:** the same refused or rejected change proposed again. The record is in the prompt;
  count repeats.
- **Fitting the train placements:** compare train and test groups for the final body.
- **Escalating too early:** swaps before parameter moves have been tried on a slot. Count moves per
  slot before the first swap.
- **Moving the ruler:** any edit to a world's tasks, physics or catalogue re-keys its runs.
  Re-validate with the gold check before comparing.
