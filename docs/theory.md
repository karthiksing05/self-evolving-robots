# Theory

Copied from PLAN.md Section 3 (spec v2.1). The code must reflect this; deviations are in [decisions.md](decisions.md).

## 3.1 Primitives

A **primitive** is a typed, parameterized unit with preconditions and effects. There are three kinds.

| Kind | Examples | Parameters | Executed by |
|---|---|---|---|
| Motor | `approach`, `contact`, `push`, `rotate`, `hook_pull`, `slide`, `insert`, `release`, `retreat` | Target frame, direction, force cap, travel or angle, speed | System 1 controller |
| Body | Tool templates (`pad`, `hook`, `knob_cup`, `soft_tip`, `slot_inserter`, `probe`), finger parts, link scale modules | Geometry (mm), material contact parameters | Build pipeline: CadQuery to MJCF |
| Check | `moved_at_least(joint, d)`, `latched(control)`, `force_below(cap)`, `no_banned_contact()`, `detent_crossed(n)` | Thresholds | Verifier, from sim state |

Each primitive declares a **type signature**:
- `requires`: a contact type (`pad`, `hook`, `cup`, `probe`, `edge`), minimum reach, and minimum force.
- `provides`: an effect on a control class, for example `actuates: push_button` or `opens: door[handle=push_latch]`.

## 3.2 Composition

Programs are trees built from primitives with these operators:

| Operator | Meaning | Example |
|---|---|---|
| `Seq(a, b, ...)` | Run in order; stop on failure | `Seq(approach($target), push($target, 0.003), retreat())` |
| `Guard(a, until=check, cap=check)` | Run `a` until `until` is true or `cap` becomes false | `Guard(push($target), until=detent_crossed($target, 1), cap=force_below(8.0))` |
| `Repeat(a, n \| until)` | Repeat up to n times, or until a check holds | Turn a knob 3 detents |
| `Branch(check, a, b)` | Pick a or b from the sim state | Open the door if it's latched, otherwise skip |
| `With(tool_index, program)` | Mount a rack tool, run, unmount | `With(1, hook_pull($target))` where rack slot 1 holds a hook |
| `Bind(primitive, params)` | Fix some parameters, leave the rest free | A partly applied primitive becomes a new primitive |

**Typing rule.** A program type-checks if:
- every motor primitive's `requires` is met by the mounted tool (or the stock gripper);
- the reach of every target is inside the current body's reach envelope (Section 7.3);
- no hard limit can be exceeded by any parameter in its declared range.

Programs that don't type-check are rejected before simulation, and the type error is returned to System 2.

**Conventions.** All numeric literals are SI (meters, newtons, radians); no unit suffixes. `$target` refers to the control the program is run on; check functions resolve symbols like `travel` from that control's `ControlSpec`. The exact grammar and per-operator arity are in Appendix A.

**Composition is the cheap path.** Changing a program costs no print minutes. Changing the body does. That ordering drives the escalation ladder in Section 3.4.

## 3.3 System 1 and System 2

| | System 1 | System 2 |
|---|---|---|
| Role | Fast, reactive, trial and error | Slow, deliberate decision maker |
| Runs | Inside an episode, every control step (500 Hz sim) and between attempts | Between episodes, on escalation |
| Implemented by | Scripted motor primitives, guarded contact, fixed retry rules, local parameter search (CMA-ES over a primitive's continuous parameters, at most `m` trials) | An LLM (Claude) producing typed programs or design changes with predictions. Optional world model (3.6) to screen candidates. |
| Inputs | Sim state, force and torque, joint state | Coverage report, failure traces, concept-memory matches, primitive library, design records, posture, budgets |
| Outputs | Joint targets; parameter values | `Proposal`: a new program, a library promotion, or a body change, each with a prediction |
| Can change | Continuous parameters inside declared ranges | Program structure, which primitives and tools are used, body parameters (via build checks) |
| Can't change | Hard limits, the verifier, the reward | Hard limits, the verifier, the reward |

**Escalation rule (metacognition).** System 1 hands a control to System 2 when any of these holds:
1. The verdict is `unreachable`, `rule_violation`, or `no_applicable_primitive`. These are structural failures that parameter tuning can't fix.
2. System 1's local search used its `m` trials (default 8) without a pass.
3. Surprise is above τ, which depends on posture: τ = 3σ conservative, 5σ optimistic. The outcome contradicts memory, so deliberate diagnosis is needed.
4. The control is unfamiliar: its **familiarity count** `concept_n` is under `n_fam` = 5. `concept_n` is the number of payload results under the categorized leaf's parent. It's separate from the expectation ancestor in Section 8.5, which always has at least 20 results once the tree is big enough.

## 3.4 Escalation ladder

System 2 must consider fixes in this order. It moves down only if the cheaper rung can't plausibly fix the failure, and must write why in the proposal.

| Rung | Change | Typical cost |
|---|---|---|
| 0 | Re-parameterize (hand back to System 1 with new ranges) | Sim seconds |
| 1 | Recompose existing primitives (new program) | Sim seconds + 1 LLM call |
| 2 | Use or adapt an existing tool from the library | Print minutes only if adapted |
| 3 | New tool from a template (L0) | Print minutes, grams |
| 4 | Finger change (L1) | More print minutes |
| 5 | Kinematic change (L2) or actuator change (L3) | Most expensive; full regression |

The resource ledger (Section 10) prices every rung, so the value-per-cost rule usually enforces the ladder automatically. The explicit order is a guard against an LLM that jumps to body changes.

**Mechanical check.** `Proposal.kind` maps to a rung: `reparam`=0, `program`=1, `use_tool`=2, `new_tool`=3, `finger`=4, `arm`=5, `actuator`=5. `promote` has no rung. A proposal at rung k must have, for every i < k, an entry in `ruled_out` that starts with `rung<i>:` and has a non-empty reason. Otherwise it's rejected before costing.

## 3.5 Library learning (growing the repertoire)

A composed program is **promoted** to a named primitive when all of these hold:
- it passed on at least `k_promote` distinct controls across at least 2 panels (`k_promote` comes from the posture, Section 9; 3 at β = 0);
- its parameters can be abstracted: the values that differed between uses become arguments, and the rest are fixed;
- it isn't a duplicate. Its embedding (program text plus outcome signature) isn't within distance `d_dup` of an existing primitive in the program-concept tree (Section 8.5).

On promotion, System 2 writes:
- a name, for example `open_push_latch_door`;
- a one-line docstring;
- the type signature.

The new primitive is added to the library and shown to System 2 in later prompts. This follows the LILO/DreamCoder idea of compressing successful programs into reusable abstractions, with Cobweb doing the clustering.

**Pruning.** The library has a size budget (default 40 primitives, Section 10). A primitive unused for 10 iterations, or dominated by another (same `provides`, lower cost, equal or higher pass rate), is retired. Retired primitives stay in the log.

## 3.6 Optional world model (after milestone 8)

The user may want a JEPA-style world model as part of System 2.

**What it is.** A small latent model trained on logged transitions: (context embedding, program embedding) → predicted outcome embedding plus pass probability. It plays the same role as action-conditioned world models such as V-JEPA 2-AC.

**How it's used.** Only to **rank** System 2's candidate proposals before simulation ("imagination"). It never replaces the verifier.

**Requirements:**
- PyTorch, under 5M parameters, trained on CPU from the run logs.
- Acceptance: on held-out logs, ranking with the model raises the top-1 proposal pass rate by at least 10 points over random ordering. If it doesn't, the feature stays off.

## 3.7 Concept memory's role

Cobweb organizes everything the robot learns into concepts that people can read. There are four trees; three are listed here, and the fourth ("full", used only to score surprise) is defined in Section 8.5.
- **Context tree:** what controls look like (embeddings) → how they behave. It supplies expectations, surprise, and uncertainty.
- **Fix tree:** text embeddings of root causes → the fixes that worked.
- **Program tree:** embeddings of programs and their outcome signatures → clusters of similar strategies. It's used for de-duplication and to suggest abstractions.

Uncertainty from the context tree feeds the posture rule (Section 9) and target selection (Section 8.4).

