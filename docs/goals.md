# Goals and sub-goals: a task network between System 2 and System 1

**Status:** design (D-034). The contracts and the degenerate network are built (D-037); the
agentic System 2 is not. The build order is in [gripper-experiment.md](gripper-experiment.md)
section 8.

System 2 should not be a proposal generator that runs once per escalation. It should work like a
planner that talks to System 1. It breaks a goal into sub-goals, gives the executable ones to
System 1, reads what came back, and revises: refine, re-decompose, look closer, or change the body.
The one-shot proposer stays as the fast path for problems that memory already knows how to solve.

This document grounds that structure in cognitive science and AI planning, then maps each idea to
something concrete in the harness. Most of the pieces already exist in the code under other names.
What is new is the goal layer above a single control, and the protocol that lets System 2 and
System 1 go back and forth over it.

## 1. What we borrow, and what each idea decides

| Source | Idea | What it decides in the harness |
|---|---|---|
| HTN planning (Erol, Hendler & Nau 1994; SHOP2, Nau et al. 2003) | Tasks are primitive (directly executable) or compound. A *method* decomposes a compound task into subtasks under preconditions. | The shape of the network. Leaves are System 1 skills. Methods are library programs whose `requires` are preconditions and whose `provides` are effects. |
| Soar (Laird, Newell & Rosenbloom 1987) | An *impasse* (the agent cannot proceed) creates a subgoal to resolve it. The result is *chunked* into a rule, so the impasse does not recur. | When System 2 gets involved: only at an impasse, and the impasse type picks the sub-goal type. Library promotion is chunking. |
| TOTE units (Miller, Galanter & Pribram 1960) | Behaviour is nested Test–Operate–Test–Exit loops. Each unit tests its goal, acts until the test passes, and exits. | Every node carries a success test, and the judge evaluates it. A node exits on its test, never on someone's claim that it is done. |
| ICARUS (Langley & Choi 2006) | Skills are indexed by the goal concepts they achieve. Execution failure triggers problem solving, and the solution is learned as a new skill. | Memory indexes methods by the goal they achieve, so retrieval comes before deliberation. Our Cobweb memory (Fisher 1987) comes from the same line of work. |
| Means–ends analysis (Newell & Simon, GPS) | Pick the operator that most reduces the measured difference between the state and the goal. | How System 2 chooses a design edit. The difference is measurable: for example, the V1 capability table says the stock gripper holds up to 45 mm, so a 60 mm can is 15 mm beyond it. |
| Options and MAXQ (Sutton, Precup & Singh 1999; Dietterich 2000) | Temporally extended actions have an initiation set, a policy and a termination condition. Value decomposes over the hierarchy. | A sub-goal handed to System 1 is an option: preconditions, program plus parameter ranges, and a stop condition (test or budget). Each method keeps its own success-per-cost record. |
| Hierarchical task analysis (Annett & Duncan 1967) | Re-describe a goal into sub-goals only while P × C (chance of failure × cost of failure) is unacceptable. | When to stop decomposing: a leaf is executable when System 1's measured pass rate on that method, at the posture's confidence bound, makes failure cheap enough. |
| HTN method learning (HTN-MAKER, Hogg, Muñoz-Avila & Kuter 2008) | Learn methods from solved plan traces. | A decomposition that succeeds repeatedly is promoted into the library as a new method. |

One idea is ours and has no direct precedent in those sources: **the body is part of the plan.**
When no method's preconditions can hold with the current body, the sub-goal becomes "change the
body so that one does". A design edit is a compound task whose effect is to make a precondition
true. The finger change in phase B is exactly this. The typecheck's aperture rule fails, so
System 2 creates a Design sub-goal whose test is "the aperture rule passes *and* the object is
held *and* the regression set still passes".

## 2. The pieces we already have

| HTN / cognitive concept | Already in the code | Where |
|---|---|---|
| Primitive task (operator) | Motor primitive with parameter ranges | `configs/primitives.yaml`, `control/motor.py` |
| Method body | DSL program: `seq`, `guard`, `repeat`, `branch`, `with`, `bind` | `contracts.Program`, `compose/dsl.py` |
| Method preconditions / effects | `PrimitiveDef.requires` / `provides`; the typecheck | `compose/typecheck.py` |
| TOTE test | Guard caps and success checks, evaluated on ground truth | `compose/checks.py`, `verify/verdicts.py` |
| Impasse | Escalation with a reason: `structural`, `s1_exhausted`, `surprise` | `loop.escalations` |
| Chunking | Library promotion and pruning | `loop.maybe_promote`, `compose/library.py` |
| Indexing by goal | Cobweb concepts and fix-tree retrieval | `memory/concepts.py`, `FixRecord` |
| Prediction to corroborate | `Prediction` on every proposal, scored after the trial | `contracts.Prediction`, `Decision.prediction_score` |

What is missing:
1. **A goal level above one control.** System 2 sees failing controls, not goals and sub-goals.
2. **More than one exchange.** System 2 cannot say "first find out X, then try Y" and see X's
   result before choosing Y.
3. **Budget per sub-goal.** Budgets are per run, so nothing stops one sub-goal from spending
   everything.

## 3. The design

### 3.1 The goal language is judge tier

A goal's test is a predicate over ground-truth sim state. It is composed from a fixed vocabulary
that the verifier implements:
- the existing checks, such as `moved_at_least`;
- the grasp-world predicates that arrive with grasp world v0: `lifted(obj, mm)`,
  `held_for(obj, s)`, `placed(obj, region)`;
- `still_passes(set)` for regression sets.

System 2 composes tests from this vocabulary. It cannot invent one. If a planner could write its
own success test, it could declare its sub-goals met, which is reward hacking one level up. So the
vocabulary is fingerprinted with the judge, and the decomposition logic is policy.

### 3.2 The network

```
Goal      id, parent, test (judge vocabulary), why (System 2's rationale), slice (budget), status
Task      one of:
  Skill     leaf for System 1: method or primitive, parameter ranges, sensor guards
  Analyze   a question answered by sense or analyze tools; never changes the arm or the body
  Design    a body change: delta family, Prediction; goes through acceptance like any proposal
  Network   a method: ordered subtasks
Method    name, achieves (goal pattern), requires (preconditions), body (subtasks); learned or seed
Report    System 1 back to System 2: goal id, verdict from the judge, impasse type, trace summary
          including sensor features, resources spent
```

The network for a run is stored in `runs/<id>/goals.jsonl`, one line per node event. Like the
tool-call log, it makes a session replayable.

### 3.3 The protocol

1. **Retrieve first.** For each goal, memory looks for a method that achieves it and whose
   preconditions hold, and ranks candidates by measured success per cost. If one exists,
   System 1 runs it and no LLM is called. This is Soar's chunk firing, and it is most of phase A.
2. **One-shot when the fix is known.** If the retrieved method fails with an impasse that the fix
   tree has seen before, the one-shot `s2` proposer handles it. This is the fast, rote path.
3. **Deliberate when it is new.** Otherwise the agentic System 2 takes the goal. It decomposes,
   allocates a budget slice to each sub-goal, and dispatches leaves to System 1. It reads each
   Report and revises.
4. **The impasse type picks the next sub-goal:**

   | Impasse (from System 1's Report) | Sub-goal it creates |
   |---|---|
   | `s1_exhausted`: the search ran out without passing | Re-decompose with another method, or Analyze why |
   | `structural`: no method's preconditions can hold with this body | Design, with the difference measured against the capability table |
   | `surprise`: a sensor residual or a failed prediction | Analyze before acting again; a possible fault (phase C) |
   | Reflex stop (`stopped_by`): `load_limit`, `slip`, `force_limit` (D-036) | Analyze first; never retry blindly. `load_limit` points at the body (the payload is past the arm's stall; phase C), `slip` at the grip (another method, or the fingers) |
   | Slice exhausted | Return to the parent, which re-allocates or gives up |

5. **Exit on the test.** A node closes when the judge says its test holds, when its slice runs
   out, or at the depth limit. A parent never closes because a child says it is done.
6. **Learn.** A decomposition that passed `k_promote` times is promoted as a method, with
   preconditions and effects taken from its nodes. A method that keeps failing is pruned, as
   programs are today.

### 3.4 Resources: what the agent is told it has

Every decomposition is an allocation. So at each node the agent sees four things:
1. **The node's remaining slice:** sim-seconds, System 1 iterations, plan calls, print minutes,
   grams, approvals. A child's slices sum to at most its parent's, and the ledger charges actual
   spend to the node.
2. **The sensor manifest:** which sensors are mounted on the current body, what each channel
   measures, its resolution and noise, and whether it exists on the bench arm (D-033). A plan that
   depends on a sensor the body does not carry is rejected before dispatch.
3. **The capability table:** what the current body can do (V1), and after a finger change, the
   new body's computed capabilities (finger family rule 3).
4. **The design levels the run allows,** and the regression set a change must not break.

### 3.5 What stays fixed

- The judge decides success at every node, from ground truth. Sensors inform decisions but never
  decide outcomes (D-031).
- Reward is computed at the root only. Sub-goal outcomes are diagnostics and credit for ranking
  methods, never reward terms. A pseudo-reward on sub-goals is the classic way for a hierarchy to
  learn to satisfy easy sub-goals instead of the task.
- A body change happens only through `design.submit`, so it gets paired trials and the
  reference and regression vetoes.
- On the real arm, act tools need approval once per session, and the server enforces the hard
  limits on every call regardless (D-033).

### 3.6 How the existing loop fits

The current loop is the degenerate network:
- one root goal (coverage of the training panels);
- one leaf per control, served by System 1 with retrieved programs;
- one-shot System 2 at each impasse.

Expressing the loop that way first means the network can be built without changing any
experiment's numbers. The battery is the check. Agentic depth is then an extra capability, not a
rewrite.

## 4. What it lets us measure

| Question | Comparison |
|---|---|
| Does decomposition beat one-shot proposals? | `s2` (one-shot) against `s2_agent` (task network) against System 1 alone: success, plan calls per solved goal, sim-hours, design edits per success |
| Does retrieval (chunking) cut deliberation? | `s2_agent` with and without method retrieval: LLM calls per solved goal, first half of the run against the second (as E11 does for the library) |
| Does the impasse type pick the right sub-goal? | Per phase-B object, the first sub-goal type against the declared escalation in the object table |
| Is decomposition depth well-calibrated? | Network depth and leaves per goal against System 1's measured pass rate (the P × C stopping rule) |

## 5. Where it goes in the code

| Module | Tier | Contents |
|---|---|---|
| `compose/checks.py` (extended) | judge | Per-control goal tests: the check atoms, with `held_for` from grasp world v0 (built) |
| `verify/goals.py` | judge | **Built (D-037).** Goal tests above one control: `pass_share(SET, x)`, `still_passes(SET)`, `accepted()` |
| `plan/network.py` | policy | **Built (D-037).** `Goal`, `Task`, `Method`, `GoalReport`; slice arithmetic; validation (tests drawn from the vocabulary, slices that fit, leaves that are executable, sensors that are mounted); the impasse-to-sub-goal table; `goals.jsonl`; the degenerate network |
| `plan/agent.py` | policy | The agentic System 2 over `sea-arm`: `goal.decompose`, `goal.assign`, `goal.report`, `goal.close`, `resources.manifest` |
| `sea-arm` server | skill / world | Sense, analyze, act and design tools (section 5 of the gripper doc), plus the goal tools above |
