# Design philosophy

The ideas the harness is built on, each with where it lives in the code and the evidence that made
it a rule. [harness.md](harness.md) covers the trust boundary and [iteration.md](iteration.md) the
discipline for changing fundamentals; this is the short list underneath both.

## 1. The ruler is not the robot

What counts as success or cost (the judge tier: verifier, reward, limits, task distributions, resource
check) is separate from what searches (System 1, System 2, baselines).
- A proposer can suggest anything and change nothing about how it is judged.
- A change to the judge re-keys every comparison (`harness.judge_key`).
- *Where:* `harness.COMPONENTS`, `verify/`, `evolve/core.py` and `resources.py`, `worlds/*`.

## 2. Validate the task before the agent

A task must have a stock failure for a stated reason and a gold solution that passes. If the gold
fails, the fault is in physics or skills, and no proposer result means anything (SWE-bench's
PASS_TO_PASS / FAIL_TO_PASS discipline).
- *Evidence:* the potted meat can started every episode inside the robot (D-040). The pad opening
  fell back to fixed angles (D-038). A guessed TPU stiffness made grippy pads hold worse than bare
  plastic (D-038). Each one silently failed tasks for reasons no design could fix.

## 3. Diagnose the mechanism, not the symptom

"Slip" is a symptom. "The moving jaw touches 10 mm above the box's centre and pushes 48° down" is a
mechanism, and it names what to change.
- Each world reports a mechanism label with its numbers, read from the simulator after the attempt:
  a twin's analysis.
- The diagnosis explains; it never decides. The verdict comes from ground truth by fixed rules, so a
  better diagnosis cannot move a score.
- *Where:* `World.solve` → `Outcome.mechanism/detail`; `worlds/gripper.py: diagnose`.

## 4. Know the body you have

Before proposing, System 2 gets computed facts about the current body (the brief): its ranges and its
kinematic limits. On the stock gripper, these include:
- what the jaw can reach;
- how far its fingertip rises at each opening;
- the squeeze it can exert.

These are facts, never advice, so the reasoning is System 2's and is checkable against numbers.
- *Where:* `World.brief`; `fingers.fingertips`, `tip_opening`, `squeeze_n`.

## 5. The cheapest sufficient change, chosen from evidence

The ladder runs cheapest first:
1. retune the controls (System 1);
2. change the current parts' parameters;
3. swap a part type, which is a redesign;
4. propose a part that does not exist yet (an idea).

Escalation is earned by evidence, not by a schedule. A redesign is the natural next move when the
brief shows the current part cannot reach the task, and the record shows parameter changes on that
part keep failing on the same mechanism.
- *Where:* `evolve/core.py` (moves, `edit_rung`), `prompt.py` (`_exhaustion`).

## 6. History first

A change is kept only if it gains something and loses nothing the robot already does. The held set
only grows: a ratchet. Held tasks are trialled first, so a regression ends a trial early and its cause
reaches the next prompt.
- *Evidence:* in the first live gripper run, System 2 kept steepening a pad that won the gelatin box
  and lost the banana and the cube (D-039). The rule caught it every time; naming what broke is what
  changed the reasoning.

## 7. Protect a new structure until it is tuned

A redesign is judged against a body whose parameters have had many iterations of tuning, so the new
part's first guess is an unfair comparison. Two mechanisms give it a fair trial:
- a body that gains more than it breaks stays on a frontier to be repaired;
- the loop spends a fixed, reported allowance fitting the new part's parameters before judging it.

The ratchet never weakens: what is kept still loses nothing. This is NEAT's speciation in miniature.
- *Where:* `evolve/loop.py` (`frontier`, `tune`, `RunConfig.tune_trials`); D-043, D-044.

## 8. Resources are part of the design

Every part has a bill of materials. The bench's manifest says what is on hand, what can be bought
(price, lead time) and what each machine minute costs.
- A design that needs an item the bench does not provide is refused before trial, and the refusal
  says what is missing.
- An idea outside the catalogue is recorded with what it needs, so new ideas are welcome and their
  cost is acknowledged before anyone builds them.
- *Where:* `evolve/resources.py`, `World.bom`, `World.manifest`; sensors as resources (D-033) are the
  same principle for sensing.

## 9. Predict, then let the trial decide

Every proposal states its mechanism, the principle it relies on, the groups it should gain and the
held groups it risks. The simulator decides. Predictions are logged next to outcomes, so the reasoning
can be scored, not just the result.
- *Where:* `evolve.core.Proposal`; `iterations.jsonl`.

## 10. Generality through contracts

The loop does not know what a gripper, a vocal tract or a rover is. A world supplies the following,
and the same loop, prompt sections, screen, ratchet and records run on all of them:
- tasks, parts, physics and a brief;
- a bill of materials and a manifest;
- System 1.

If an idea only works in one world, it belongs in that world's module.
- *Where:* `evolve/core.World`, `docs/evolve.md`.

## 11. Measure on frozen instruments

Change one fundamental at a time, and judge it by fixed batteries and gold checks, not by the evolving
loop. A number that moved because the ruler moved is not a result.
- *Where:* `scripts/battery.py`, `tests/golden/`, [iteration.md](iteration.md).

## 12. Count in units of reality

Each proposal and trial is charged as the real build would be:
- sim seconds;
- machine minutes;
- dollars and days of lead time;
- LLM tokens at list price, even on plan billing.

A design that is cheap in simulation and impossible on the bench is not a result.

## 13. Every claim has a record

Every prompt, answer, screen, trial, decision, purchase and idea is written down. A run reads back
as a case study, and a reviewer can check any number without trusting the loop's summary.
- *Where:* `runs/evolve/<run>/`, `results/`.
