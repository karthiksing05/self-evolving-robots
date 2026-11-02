# The evolution core: one loop, many worlds

`sea.evolve` changes a robot's body until it can do every task, in any world that implements one
contract (D-041). The panel loop (`sea.loop`, [harness.md](harness.md)) predates it and still runs
experiments E0–E11. The core is the general form of that loop's body-change half:
- the program composition, memory and posture machinery stay in the panel loop;
- what the core adds is the part catalogue, resource grounding and a redesign move.

## What a world supplies

| Method | Returns | Rule |
|---|---|---|
| `tasks()` | `Task`s with a `group` (an object, a phoneme, a destination) and a `split` | A group is held when every one of its train tasks passes |
| `catalogue()` | `PartType`s: a slot, physical parameters with ranges and docs, what the part provides | Offer general parts, never a solution to a task; include parts that cost and do not help |
| `stock()` | The body the robot starts with: one part per slot | |
| `solve(body, task, seed)` | An `Outcome`: success, a 0–1 score, a **mechanism** label and the numbers behind it | Runs System 1 (`search.local_search`, or the world's own); success must survive confirmation seeds |
| `brief(body, tasks)` | Computed facts about this body: ranges, kinematic limits, reachable regions | Facts, never advice |
| `principles()` | The world's physics in sentences | |
| `bom(body)` | `BomLine`s per slot: manifest items, or fabricated material (`print:PLA` in grams) | |
| `manifest()` | Budget, fabrication rates, and items with what is on hand, price and lead time | An item with no price cannot be bought |
| `check(body)` | Reasons the body cannot be built | |

The mechanism comes from the simulator after the attempt: a twin's analysis, like a dry run. It
explains a failure to System 2. It never decides one; `success` does.

## The moves

System 2 answers with a `ProposalBatch` of one to three proposals. Each proposal names the
mechanism, the principle it relies on, a hypothesis, the groups it should gain and the held groups
it puts at risk. There are three moves, cheapest first:

1. **`params`**: new values for the current parts' parameters.
2. **`swap`**: another part type in a slot, which redesigns that part. Its parameters default to the
   middle of their ranges.
3. **`idea`**: a part the catalogue does not have. It names a slot and a bill of materials
   (`Need`: an item and quantity, with System 2's estimated price and lead time for any item the
   manifest lacks). `resources.cost_idea` costs it against the bench, as for a part, and says whether
   it fits what is left (D-047). It is never trialled, because nothing can build it in the simulator.
   Ideas are the outer loop's input: candidates for new catalogue parts, each with a price.

The loop records the move a proposal actually makes (`edit_rung`), not the one it declares.

## Screening, before any simulation

1. **Catalogue:** the slot exists, the type fills it, and every parameter exists and is in range.
2. **Build:** `world.check(body)`.
3. **Resources** (`resources.check`): the candidate's bill of materials against the current body's.
   - Items the current body already has are reused.
   - The bench's pool is the manifest's spares plus what the run owns: the stock body's items and
     every purchase, in `Spent.owned`. A spare is counted once (D-046).
   - Shortfalls are bought if they have a price and the budget allows (dollars, lead time).
   - Anything else is refused as "not provided".
   - Fabricated material is charged only for slots whose part changed: a changed part is a new
     print.
4. **No-op:** an edit that leaves the body unchanged is refused.

Refusal reasons go into the next prompt's record, so System 2 learns what the bench provides.

## Trial and acceptance: the ratchet

A screened candidate solves the held groups first and stops at the first loss. Then it solves the
failing ones. A swap is always scored on every group, so the record shows what a redesign would gain
as well as what it broke (`full_trials` does the same for params moves).

A candidate is **kept-eligible** if it gains a group, or at least a task, and loses none. Among
eligible candidates the loop keeps:
1. the one gaining the most groups;
2. then the most tasks;
3. then the cheapest (dollars plus machine hours);
4. then a params move over a swap.

The held set only grows, the PASS_TO_PASS rule: the robot's history comes first. Purchases and
machine time of the kept candidate are charged to the run. A body already trialled is refused with
its earlier result, so no body is paid for twice.

## Merging gains in different slots

Kept-eligible proposals that change disjoint slots are not alternatives. After the proposals' trials,
the loop builds one more body: the best candidate, plus every other kept-eligible candidate whose
changed slots it does not touch. If that joins at least two, the union is screened, costed and
trialled once, and kept if it scores higher (D-045).

In `phonology_s2_s0`, rounded lips and a tongue tip were each kept-eligible on their own. Their union
also produced /ɝ/, which neither gained alone, and held all 13 groups in one iteration.

## Redesigns: the frontier and the tuning allowance

A redesign usually breaks something before it is better. Two mechanisms let one survive that phase
without weakening the ratchet:
- **The frontier (D-043).** A rejected body that gains more groups than it loses is kept on a
  frontier. The prompt lists it with what it would gain and what it broke. A proposal can build on it
  with `base`, and the result must still keep every held group.
- **The tuning allowance (D-044).** A swap is System 2's first guess at a new part's parameters,
  judged against a body tuned over many iterations. The swap this iteration that gained a group but
  broke a held one gets up to `tune_trials` more trials (default 4):
  - a Gaussian search over the parameters of every slot it changed, in each parameter's range,
    centred on the best tuning so far;
  - each trial is screened, costed and scored like a proposal;
  - it stops at the first kept-eligible body;
  - it runs only if the swap gained more groups than the best kept-eligible candidate, since otherwise
    it could not change the decision.
  This is NEAT's protection of structural innovation, applied to parts: a new structure competes
  only once its own parameters are fitted. The random arm gets the same allowance.

## The prompt

It has the same sections in every world (`prompt.py`):
1. The harness's seven principles (`SYSTEM`).
2. The world and the iteration.
3. The current body, and its brief.
4. The world's physics principles.
5. The catalogue. Each part type shows:
   - its parameters;
   - what it needs, with items the current body already has marked reused;
   - for another type, the true marginal cost of swapping it in now, or why it would be refused;
   - its computed facts on this body (`part_facts`), when the world gives them.
6. The resources left.
7. The held groups, each with its task facts.
8. The failing tasks, each with its mechanism, its numbers and System 1's best attempt.
9. The record of past proposals and tuning trials: what each gained, lost (with the numbers) or was
   refused for.
10. A per-slot summary of what parameter-only trials have not fixed.
11. System 2's calibration: its predicted gains and named risks against what the trials found.
12. The frontier, and the ideas recorded so far with their costing.

That summary and the brief are what make a redesign arise rather than be ordered. When parameter
moves on a slot keep failing on one mechanism, and the brief shows the part cannot reach what the
task needs, the cheapest move left is a swap.

## Arms and records

- `s2`: System 2 on the Claude plan through the Claude Code CLI (D-029), never API credits.
- `random`: blind catalogue edits through the same screen and trials. It measures what reasoning
  buys.
- `s1`: the stock body.

`scripts/evolve.py --world <name> --arm <arm>` writes `config.json`, `manifest.json`,
`iterations.jsonl`, `llm/` and `summary.json`:
- `iterations.jsonl` has every failing outcome and every proposal with its screen, resources, trial
  outcomes and decision;
- `llm/` has every prompt and answer, with tokens and list-price cost;
- `summary.json` has the curve, the final body, train and test groups for the final and stock
  bodies, spending, ideas and LLM usage.

The test split is scored for both the final body and the stock body: other placements or seeds of
the same groups.

`scripts/collect_evolve.py` gathers a run into `results/evolve/`, with `--aggregate` for many seeds of
a baseline arm. `scripts/ideas.py` lists every idea recorded across runs, with its costing and the
mechanisms open when it was proposed. That list is the outer loop's input, for example
`results/evolve/ideas_gripper.md`.

## Adding a world

1. Write `src/sea/worlds/<name>.py` with a class satisfying `World`, and register it in
   `sea/worlds/__init__.py`.
2. Add it as a judge-tier component in `harness.py`: its tasks and physics are the ruler for its
   runs.
3. Validate it before running a proposer on it: the stock body fails some groups for physical
   reasons, and a gold body built from the catalogue within the budget holds every group.
   Otherwise a result means nothing.

Worlds so far:
- `gripper`: the SO-101 and 21 YCB objects (D-042).
- `phonology`: a mechanical talking robot.
- `mobile`: a ground robot exploring a house and garden. It has ten routes and five slots
  (locomotion, chassis, drive, battery, sensor), modelled with quasi-static vehicle mechanics:
  - edge climbing, trenches, openings, ramps, Winkler sand, overhangs, dark corridors and energy;
  - stock holds 4/10 groups;
  - a gold body holds 10/10 for $16.50: tracks, a narrower tub, more cells and a sonar;
  - no parameter setting of the stock parts holds everything.
