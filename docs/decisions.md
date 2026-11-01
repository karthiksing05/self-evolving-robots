# Decisions log

Every deviation from `PLAN.md` (spec v2.1), every dependency beyond Section 5, and every API that
differed from the spec is recorded here with the fix chosen (spec Section 0, rule 8).  Entries are
append-only; supersede an entry by adding a new one that names it.

## D-001 Toolchain versions (Oct 31, 2026)
Python 3.12.15 via `uv` (pinned in `.python-version`).  Resolved: mujoco 3.17.0, placo 0.10.2,
cadquery 2.8.0, torch 2.15.0, transformers 5.22.1, cobweb-language-embedding @ 2f80775.  All import
on macOS arm64.  `uv.lock` pins the rest.

## D-002 Claude API surface differs from Section 8.6
- **Temperature 0.2 dropped.** `claude-opus-5-5` (and every model under API version 2026-10-21)
  rejects `temperature`/`top_p`/`top_k` with a 400.  Determinism for System 2 comes from the request
  hash cache (Section 0, rule 4), not sampling settings.
- **No forced tool choice.** `tool_choice: {type: "tool"}` / `"any"` return a 400 on Opus 5.5 and
  Sonnet 5.5.  `AnthropicLLM` sends `tool_choice: auto`, names the `propose` tool in the instruction,
  validates the tool input with `ProposalBatch.model_validate`, and retries once with the validation
  error (the spec's retry rule) when the call is missing or invalid.
- **Thinking.** Opus 5.5 always thinks; effort is set explicitly (`output_config.effort`).
- **Refusal fallbacks.** Server-side fallbacks are on by default (`betas=["server-side-fallback-2026-07-01"]`,
  `fallbacks="default"`), so a policy decline is re-run on a fallback model inside the same call.
- **Pricing for the ledger** (USD per MTok in/out): opus-5-5 4/20, sonnet-5-5 2/10, haiku-5-5 0.1/0.5.

## D-003 Cobweb API at 2f80775 matches Section 8.5
Checked `CPPCobwebTree.__init__/ifit/categorize/log_prob/dump_json/load_json` and
`PCAICAWhiteningModel.fit/transform`.  One fact the wrapper relies on that the spec does not state:
on insert, a leaf that receives a non-matching instance is *kept* as an object and becomes a child
of a new internal node (`cobweb()` clones it upward).  Leaf objects therefore stay leaves, so
payloads are attached to the leaf object returned by `ifit` and never need to be moved.
`PCAICAWhiteningModel` needs scikit-learn (already a transitive dependency; 1.9.1).

## D-004 `ControlResult` gains `ik_residual_mm`, `program_id`, `s1_trials`
The full tree's measurement vector (Section 8.5) includes the IK residual, which `ControlResult`
did not carry.  The fields are additive with defaults, so old JSON still validates.

## D-005 `StepTrace` gains executor bookkeeping
`aborted`, `expected_samples`, `detents_crossed`, `latched_final`, `push_phase`, `in_contact`,
`success_time_s`, `tool_swaps`, `sim_seconds`, `servo_heat_j`, `note`.  `judge()` (Section 8.2)
needs them to decide `inconclusive` and to read the success check without re-simulating.

## D-006 Defaults on `Design`, `ArmSpec`, `FingerSpec`, `BuildEstimate`
Stock values are the defaults (finger fields 0 = stock jaw), so `Design(id="d000")` is the stock arm.

## D-007 Positional primitive arguments
`PrimRef.args` is a dict, but Appendix A allows positional arguments.  The DSL stores positional
arguments under `_0`, `_1`, ... and prints them first, so text and JSON round-trip exactly.

## D-008 Contact typing
Typecheck enforces `requires.contact` of each motor primitive intersected with the Appendix B
"contacts each class accepts" table, against the mounted tool's contact type (stock gripper
provides `edge` and `pinch`).  Banned contacts are *not* rejected by typecheck: the verifier flags
them at runtime.  This is what the M4 acceptance requires (stock gripper on a tab door with no-pinch
must give `rule_violation`, not `no_applicable_primitive`).  Slots additionally require the
`slot_inserter` template.

## D-009 Primitive extents default to the control
`push`, `rotate`, `hook_pull`, `slide`, `insert` take their extent (depth, angle, distance) from the
target's `ControlSpec.travel` plus a margin when the argument is omitted, so seed programs need no
per-control constants.

## D-010 Harness fields on `RunSpec`
`llm` (anthropic | mock), `llm_fixtures`, `eval_split`, `embedder` (fake | real).  They select
harness plumbing, not science, and default to the spec's behaviour.

## D-011 Force cap per control (Section 9 table)
`force_cap_fraction` interpolates 0.6 (beta=-1) to 1.0 (beta=+1) of the *class* cap from `Rules`,
and the result is always clamped to `HardLimits.max_force_n`.  Reading "up to the hard limit" as
"above the class cap" would make the optimist guarantee `rule_violation` on stuck buttons.

## M-001 `ConceptMemory.learn` takes the context vectors
Spec 8.5 lists `learn(results, confirmed_failures)`, but a `ControlResult` carries no context vector.
The wrapper is `learn(results, confirmed_failures, contexts, iteration=0)`, where `contexts` maps
`ControlResult.key` ("panel:control") to the 64-d context vector and `iteration` prefixes payload ids.
A newer result for a key replaces its pending (unconfirmed) entry; a pass drops it.

## M-002 Acuity (prior variance) of the text trees
Cobweb's default prior variance 1/(2e*pi) suits the whitened context (per-dimension variance ~1) but
swamps L2-normalized 384-d text vectors (per-dimension variance ~1/384), which would make the fix
and program trees unable to split.  Those two trees use prior_var = 1/(2e*pi) / dim.  Vector spaces
and the d_dup = 0.15 Euclidean threshold are unchanged.

## M-003 Rank-limited whitening
`ContextEncoder.fit` requires >= 64 rows and calls `PCAICAWhiteningModel.fit`.  When the raw
matrix has numerical rank r < 64 (identical descriptions, e.g. under `FakeEmbedder`), PCA with 64
components would divide noise by ~0 variance; the encoder fits with pca_dim = r and zero-pads the
output to 64 dims (constant dims carry no information for Cobweb).  Before fitting, `encode` uses a
fixed seeded Gaussian projection 768 -> 64.  Without a renderer, the image half of the context uses
`geometry_surrogate` (a procedural 32x32 crop from class, size, recess, handle and tilt).

## M-004 Surprise details
z = (measured - concept mean) / max(concept sd, floor) with floors peak_force 0.05 N,
travel_reached 1e-4, detents 0.25, time_s 0.05 s, ik_residual_mm 0.1 mm (a concept whose results
are all identical would otherwise give infinite z).  With no learned results, z = 0 and nll = 0.
The full-tree NLL uses `CPPCobwebTree.log_prob(x, zeros, 32, greedy=False)`; its running log-sum-exp
underflows to a non-finite value for points far from every concept, in which case the wrapper
falls back to the best single-node log-likelihood.

## M-005 Retrieval and categorization in the wrapper
- Categorize-to-leaf: `categorize(x, use_best=True, max_nodes=64)` (bounded best-first search, no
  `retrieve_k`), then greedy descent by `log_prob_class_given_instance` to a leaf.
- `similar_fixes` / `similar_programs` rank payload vectors under the leaf's parent by Euclidean
  distance (spec), widening to further ancestors only when that subtree holds fewer than k payloads.
  `similar_programs(program, outcome=None)` uses an all-zero outcome signature.
- RNG for Cobweb tie-breaks is seeded per call from (memory seed, tree, tree insert count), and
  queries use a fixed seed, so reads never change later inserts.  `load(path)` is an instance method
  that replaces the memory's state and returns `self`.

## P-001 Posture edit counting and relative change (spec 9 "Edit size")
`ladder.count_edits` counts each reparam range, each `design_delta.arm/fingers/servo_voltage` entry,
and each `rack_add` tool once (a tool is one template instantiation; counting its 3-4 params would
make every tool proposal illegal under the conservative max_edits = 1).  The `max_param_delta` bound
is `|new - old| / max(|old|, scale)` with per-field scales (`ladder.FIELD_SCALE`, e.g. 20 mm for
`wrist_offset_mm`, 0.1 m for `riser_m`) because several stock values are 0.

## P-002 Approvals are a rate, not a reserve
`Ledger` treats approvals as `approvals_per_hour` per *sim*-hour (sim clock = cumulative charged
`sim_seconds`); `would_break_reserve` fails a body change when the trailing hour is full.  The
reserve fraction is not applied to approvals.  `remaining()["approvals"]` is measured against the
implied total `approvals_per_hour * sim_hours` so annealing and scarcity weights can see it.
Every body-change kind (`use_tool`, `new_tool`, `finger`, `arm`, `actuator`) costs one approval;
print time and grams are charged only when parts are printed.  `remaining()` also reports
`design_iterations` (via `tick_iteration`) so `anneal` uses the scarcest budget including iterations;
scarcity weights cover only the cost-bearing `RESOURCES`.

## P-003 Posture details
- Integer rows round half-up, so beta = 0 gives max_rung 4 (3.5), max_edits 2, repeats 2, k_promote 3.
- `anneal` changes only the global beta; `per_area` overrides stay fixed (spec Sec 17 default:
  force conservative, others annealed).
- `accept` pairs per-control deltas by randomization repeat when trial and base have the same number
  of repeats, else compares with the per-key base mean; keys present on one side only are ignored;
  no overlapping keys means reject.  Any regression vetoes acceptance.  `prediction_score` is the
  fraction of `flips_to_pass` that actually flipped (0 when none were predicted).

## P-004 AnthropicLLM request shape (extends D-002)
`client.beta.messages.create(model, max_tokens=16000, system, messages, tools=[propose],
tool_choice={"type": "auto"}, output_config={"effort": "medium"}, cache_control={"type": "ephemeral"},
betas=["server-side-fallback-2026-07-01"], fallbacks="default")`, anthropic SDK 2.3.0.  The tool is
not `strict` (the `Node` schema is recursive); the input is validated with `ProposalBatch`.  Haiku
requests omit `fallbacks` (no server-side fallback for Haiku 5.5).  Retries append the assistant turn
unchanged (thinking blocks included) and a user turn with an `is_error` tool_result per tool call
(or a text note when no tool was called).  Cost is priced by the model that served the response and
includes cache writes (x1.25) and reads (x0.1).  Only the live client constructor is excluded from coverage.

## P-005 Target selection details (spec 8.4)
`pick_targets` ranks escalated controls by score (ties by control key), keeps the top
`max_targets` controls, then groups them by concept; a group's score is its best member's.
`cost_est` is a documented heuristic table of expected fix cost per verdict as fractions of the
default budgets (structural verdicts expect a ~30 min tool and one approval; stuck and
wrong_behavior expect an LLM call and sim retries).

## B-001 Tool geometry conventions (design/tools.py)
Every template has a 20 x 20 x 12 mm dovetail block at z in [0, 12] mm of the tool frame
(`gripper_frame_link`, +z = approach); working geometry starts at its top face.  `length`
parameters (hook, slot_inserter, probe) are measured from the jaw tip, so Appendix B tip
transforms hold unchanged.  Choices the spec leaves open: the hook is a 20 mm wide extruded J
(shaft on the z axis, foot to x = -depth at z = length, barb of height `throat` rising from the
foot end; material thickness clip(0.3 depth, 1.5, 3) mm, barb clipped to keep 2 mm clear of the
mount), so the throat gap depth - 1.5 x thickness is open for every parameter in range.  The
knob_cup wall is a 16-gon annulus (inner apothem = inner_diameter/2, wall 2 mm) on a base cylinder
over z in [0, 12]; ribs are 1.5 x 1.5 mm bars.  slot_inserter's base widens to max(20, width) so
the blade never overhangs.  soft_tip prints as two pieces: PLA mount, TPU stalk + sphere.

## B-002 Analytic collision decomposition is the default
Templates are unions of convex pieces, so `build(decompose="analytic")` returns those pieces as
MuJoCo-ready parts (exact, fast, keeps the hook throat open).  `"hull"` (one convex hull) is the
fast-test stub Section 12.1 mentions; `"coacd"` runs CoACD on the STL and is exercised only in
`@pytest.mark.slow` tests.

## B-003 Print orientation, overhang and support
Per template: pad printed face-down (needs support when w or h < 20 mm, the mount ledge
overhangs), hook on its side, knob_cup / probe / slot_inserter upright, soft_tip upright with
support (sphere underside).  Overhang = down-facing area steeper than 50 deg from vertical, off the
bed; tolerance 1 mm^2.  Supported prints add 25% to printed volume for time and material.

## B-004 Mass and added mass
Mass = exact CAD volume x density (PLA 1.24, TPU 1.21 g/cm^3) x infill factor 0.5 (walls dominate
parts this small).  `BuildEstimate.added_mass_g` = heaviest rack tool (only one is mounted at a
time) + finger extensions.  Finger print estimates are a box approximation (jaw_len x 10 x 6 mm per
jaw + tip sphere); L2/L3 part estimates belong to `design/arm.py`.

## B-005 Wrist torque check constants
Measured once with MuJoCo on the stock MJCF at zero pose: wrist_flex axis to `gripperframe`
0.1594 m; wrist subtree (wrist + gripper + jaw) 0.178 kg with its COM 0.067 m from the axis.
Worst case = arm horizontal, tool mounted, nominal 5 N tip load (half the 10 N class cap) at the
tool tip; must be <= 0.8 x stall (1.62 N*m at 7.4 V, 2.94 at 12 V).  Consequence: probes longer
than ~74 mm fail at 7.4 V and pass at 12 V, which couples rung 3 tools to rung 5 actuators.

## B-006 Minimum-wall check without rtree/embree
trimesh's ray intersector needs rtree, which is not a dependency.  `build.wall_thickness` uses a
vectorized Moller-Trumbore caster instead (2,000 seeded area-weighted samples, inward rays).  A
tool fails when more than 0.5% of samples are thinner than `min_wall_mm`.  OCC tessellations of
curved faces leave duplicate seam vertices; meshes are welded (6 decimals) before checks.
`build()` also checks link scale (Section 9 lists build.py among the hard-limit checkers).

## D-012 Latches are joint limits
A latched door or drawer is modelled as the control joint's upper limit at rest; the force the
robot applies along the joint axis is read from the `efc_force` of the `LIMIT_JOINT` constraint
row, and the limit is lifted when it exceeds `latch_threshold`.  Penalty springs at 500 Hz were
unstable at the stiffness needed to hold 12 N.  Push-latches release at 2 mm of inward travel and
then pop out on a spring.

## D-013 Physics status and known limitations (Oct 31)
Seed library on the reference panel, stock gripper or a fixed rack (hook 14/6/45, knob_cup,
pad, slot_inserter):
- push button, slider, lever (pad): pass.
- recessed button (stock): `stuck` as expected; first contact can spike above the class cap.
- tab door / tab drawer with a hook: the hook catches the lip only when foot + barb fit the 11 mm
  pocket under it (throat <= 6 mm at 3 mm material); throat 8 fails.  This is a real design
  dependency, left for the loop (rung 3 `hook.throat`).
- doors over ~75 deg: a fixed-orientation hook binds against the fin near 77 deg (reaches
  1.25-1.35 rad of 1.386).  Rotating the hook with the door and adding a radial pull component
  both made it worse; left open as a rung-1 problem (re-grip / push from inside).
- knob cup turns knobs only slightly; toggle reaches about 2/3 of its throw.
- slot insertion: in the smoke run (panel_seed 32) the scripted slot_inserter + insert program
  never seated the blade on train slots (lateral misalignment at the slot mouth), so every such
  trial was rejected.  Left open as a skill-tier problem (compliant search at the mouth).
- High-tilt train panels (> ~45 deg) put many controls outside the stock reach.
Engage-point fix for tabs: the hook foot sits 1 mm above the plate with the barb centred in the
pocket (`motor._hook_geometry`).  Tool swaps happen at a raised swap pose (`executor.SWAP_TIP`)
because a tool mounted at the home pose can intersect neighbouring controls.

## D-014 Tool geoms compile with collisions on
MuJoCo computes body-level broad-phase filters at compile time; tool geoms compiled with
`contype=0` never collide after being enabled.  All rack tool geoms compile with collisions on and
`Sim.mount()` toggles `contype`/`conaffinity` (and alpha) per geom.

## D-015 Knob cup torsional friction
The cup's torsional friction is `(0.02 + 0.01 x ribs) x inner_diameter / 20 mm` m (MuJoCo condim 4
torsion), standing in for rib engagement that a convex-part decomposition cannot represent; it
scales with the bore like a contact radius.

Open judge-tier question: the cup turns a knob by pushing on its pointer at a few mm radius, so
the lateral contact force needed for even 0.05 N*m exceeds the 10 N knob class cap, and the
verifier (which caps total tip contact force) calls it `rule_violation`.  Whether the class cap
should bound only the axial push for rotary controls is a verifier change (docs/iteration.md 4.1),
not a physics tweak, and is left for an explicit decision.

## D-016 Suite runner (sim/env.py)
- Candidates are `applicable_programs` (C-005 order).  The first gets the whole System 1 budget
  (attempt, fixed retry, CMA-ES); the others get one attempt each if System 1 did not pass.
- Each repeat rebuilds the panel model with its own domain-randomization draw seeded from
  `(run seed, panel id, repeat)`; trial and base runs use the same draws.
- A failure is confirmed when `repeats_before_fail - 1` re-runs on fresh draws (repeat indices
  1000+) repeat its verdict; unconfirmed failures are neither learned nor escalated.
- Traces keep the final attempt only, so System 1 trials and confirmation re-runs are charged at
  the final attempt's duration (an estimate).
- `trace_path` is relative to the run directory so two runs of the same spec are byte-comparable.

## D-017 Offline System 2 and fixture recording (extends D-010)
`RunSpec.llm` gains `scripted`: `propose/baselines.ScriptedLLM`, a rule-based System 2 that follows
the ladder (missing contact -> reuse a known tool at rung 2 or a template tool at rung 3;
stuck / wrong behaviour -> rung 0 range widening of the failing primitive).  With `llm_fixtures`
set it writes `MockLLM` fixtures; `RecordingLLM` does the same around the live model.
`RunSpec` also gains `s1_trials` (m) and `trial_repeats` (3) so tests and smoke runs can shrink
them; defaults are the spec's.

## D-018 Baselines and the ladder
The ladder and posture checks guard against an LLM that jumps rungs; the body-search baselines
(`random`, `cma`, `bo`) only pass structure checks and fill `ruled_out` mechanically.  They
propose a whole rack (DesignSpace vector -> rack) as one `new_tool` proposal; CMA-ES and TPE are
told the trial's `delta_mean`.  `s1_only` proposes rung-0 range widening only.

## D-019 Design levels in this build
L0 (tools) and L3 (servo voltage, applied as actuator force range in `build_model`) are
supported.  L1 (fingers) and L2 (kinematics) proposals are rejected at screening with a reason
until `design/arm.py` lands (milestone M9); `test_model_consistency.py` skips its L2 half.

## D-020 Loop details the spec leaves open (loop.py)
- Surprise and familiarity are scored against memory *before* the iteration's results are learned.
- Only confirmed failures escalate; passes never do.  Without memory (`s2_nomem`) each class is one
  concept with uncertainty 1.
- `report.json` uses the cumulative cost at report time (before this iteration's proposal);
  `cost.json` stores it as `cumulative` (plus `iteration`, `end`, `remaining`), so `sea reward`
  reproduces it exactly.
- A suite run whose actual sim time crosses the reserve is charged as spent (the simulation already
  ran) and stops the loop with `stop_reason = "sim budget reserve"`.
- Trial = target panels (all controls) + reference panel (full, or `regression_sample_n` controls
  when the posture says `sample`), `trial_repeats` (+1 for `ucb`) paired draws.  Base runs are cached
  per (state version, panel, repeat, keys).  A regression is a reference control with base pass
  share >= 0.5 and trial share < 0.5.
- `value = confidence x sum(gain_est(verdict))` over predicted flips;
  `cost_delta = lambda_print x print_min / budget + lambda_mass x d(added mass) / max mass`.
- Promotion names are `<target_cls>_<hash6>` with the originating proposal's hypothesis as the
  docstring (the mock and scripted proposers do not write names); the promoted primitive keeps the
  originating program's `provides` (handle filters included).
- The context encoder is whitened on the train split's controls at start-up; seed trees from E0
  logs (`ConceptMemory.build_seed`) are not loaded by default.

## D-021 Harness fingerprint
`sea/harness.py` hashes the files behind each component (verifier, reward, limits, panels,
acceptance, physics, kinematics, primitives, tools, ledger, proposer, memory, loop) and tags
each with a tier (judge / world / skill / policy).  Runs write `harness.json`; `sea report`
refuses to pool runs with different judge keys unless `--allow-mixed`.  See docs/iteration.md.

## V-001 Verdict thresholds and measurements (verify/verdicts.py)
Section 8.2 order, with these readings.  Rule 1: banned = `Rules.banned_contacts` and
`HardLimits.banned_contacts`, counted only for contacts with the target control (`contact_types`
plus the `contacts` log); "force exceeded the class cap" compares max(tip force, peak force on the
control) with `Rules.force_cap_by_class` (hard limit when the class is missing).  Rule 4 "force
reached the cap" means peak tip force >= `trace.force_cap` - 0.5 N (guarded moves stop within
0.5 N, test_motor), using the class cap when no guard cap was recorded.  Rule 6 "moved" means
max |q - q_rest| >= 30% of travel in either direction, mirroring rule 4, so force-at-cap traces
split cleanly into stuck (< 30%) and wrong_behavior (>= 30%).  Rule 7 counts samples with
`push_phase` true and `in_contact` false.  Measurements: `peak_force` is the peak force on the
control (not the summed tip force), `travel_reached` = max |q - q_rest| / travel (unclipped),
`time_s` = first success time for a pass, else the episode's sim seconds.

## V-002 Verdict fallback
When no rule applies (control pushed with >= 0.2 N, moved < 30%, force below the cap, contact
kept), the verdict is `stuck` (the program stopped short), unless the control was pushed
backwards by more than 5% of travel and further than forwards, which is `wrong_behavior`.

## V-003 Abort semantics in the executor
`trace.aborted` (verdict 0, inconclusive) now means a simulation failure only: `Episode.tick`
raises `Abort("sim_unstable")` when MuJoCo counts a BADQACC/BADQVEL/BADQPOS warning during the
episode (MuJoCo silently resets the state) or qpos/qvel go non-finite; `run_program` sets
`aborted` and skips the back-off.  `force_limit` and `time_limit` aborts are evidence, judged
normally.  A `workspace` refusal (target outside the box) is `unreachable` only if the control was
never touched (peak control force < 0.2 N); after contact the trace is judged on what happened.
`ik_residual_mm` in the trace and result stays the measured value (no sentinel).

## V-004 Planted-fault suite (tests/verifier/)
Ten single-control panels at the reference pose, beta = 0 guard cap (0.8 x class cap), seed
library programs, declared verdicts:

| Case | Fault | Tool | Declared |
|---|---|---|---|
| stiff_button | stiffness x5 (k 1800 N/m, travel 3.3 mm) | stock | stuck |
| no_detent | detent removed (success needs a detent) | stock | wrong_behavior |
| latch_above_cap | pull-handle door, latch 6 N x2.5 = 15 N > 12 N guard cap | hook | stuck |
| deep_recess | 10 mm button in a 22 mm recess | stock | rule_violation |
| tab_door_no_pinch | tab door (62 deg) under no-pinch | stock | rule_violation |
| knob_over_torque | knob stiffness x200 | knob_cup | stuck |
| reversed_slider | joint axis reversed | stock | wrong_behavior |
| jammed_lever (extra) | joint range collapsed | pad | stuck |
| out_of_reach (extra) | panel at x = 0.42 m, past the workspace box | stock | unreachable |
| mislocalized (extra) | perceived surface 15 mm proud of the real one | stock | no_actuation |

Fault-free twins remove the fault (deep_recess: 5 mm recess; tab door: the rack hook, depth 14,
throat 6, length 45, which is also the M4 acceptance case; the hook passes 62 deg doors reliably
and stalls near 77 deg, so the door travel is 62 deg).  The deep recess is declared
`rule_violation` because the stock jaw jams in the 14 mm opening and the summed contact force
jumps past the 10 N class cap in one tick, which is what the physics does.  Each case runs two
ways: a fast synthetic 1-D guarded actuator against the real `ControlDynamics` with the same
faults (M2 acceptance), and the seed library on the compiled model (`@slow`, M3 acceptance).
Twins run over DR seeds 0-4.  Real sim, Oct 31: 9/10 faults give the declared verdict, 45/45 twin
runs pass (knob excluded, V-005).

## V-005 Physics shortfalls found by the verifier suite
- Knob: `rotate` with the knob cup presses the knob above its 10 N class cap (peak 12 N) even
  fault-free, so every knob is `rule_violation`; the real-sim knob case is `xfail`.
- Unreachable targets inside the workspace box: the executor keeps moving with the best-effort IK
  solution and the arm crashes into the panel (20-70 N, `rule_violation` before `unreachable` in
  the precedence).  The approach should stop when the contact-waypoint residual exceeds 5 mm.
- Tip force is the sum of contact normal magnitudes; wedging (a jaw in a recess opening) inflates
  it well above the net force a wrist sensor would read.
- Tab drawer + hook: after the latch releases, the drawer snaps to its end stop while the hook is
  still engaged, and the force exceeds the hard limit (`force_limit`, `rule_violation`) even though
  success was reached first.

## V-006 Button caps stand proud of the board
The panel board is a solid box, so a cap that starts 1 mm (recessed) or 4 mm (push button) above it
could not be pressed further than that: every recessed button with travel > 1 mm was impossible,
and so were test-split push buttons with long travel.  Cap top is now max(1 or 4 mm, 1.3 x travel +
2 mm), the joint's full range plus 1 mm, and recess walls grow by the same amount, so the cap
still sits (recess - 1 mm) below the rim.  Buttons with travel <= 1.54 mm are unchanged.  The
reference recessed button (8 mm recess) now passes with the stock gripper.

## V-007 Reward audit layout
`verify/reward.py` reads `spec.yaml`, `design_<k>.json` (iteration k uses the newest design with
index <= k, so the loop writes a design file only when the design changes), `iter_<k>/results.jsonl`
and `iter_<k>/cost.json["cumulative"]`, plus optional `final/`.  `coverage_reward` is computed as
passes / N, which equals control_coverage x pass_rate without float drift, so the loop and the audit
agree bit for bit when both call `coverage.reward`.

## D-022 Pre-simulation `unreachable`
When no library program type-checks for a control *only* because of the reach check, the suite
runner emits `unreachable` (with the stock IK's best residual at the contact point) instead of
`no_applicable_primitive`.  Both are structural and escalate the same way, but System 2 needs to
know whether the gap is reach (tool length, kinematics) or a missing contact/primitive.
`no_applicable_primitive` is still emitted when programs fail typecheck for any other reason.

## D-023 `approach` refuses unreachable contact waypoints
`approach` aborts (`note = "unreachable"`) when the multi-seed IK residual at the contact waypoint
exceeds 5 mm.  The trace keeps that residual, so the verdict is `unreachable` (rule 2) instead of
the `rule_violation` produced when the best-effort pose was driven into the panel (V-005).

## D-024 matplotlib for `sea report`
Section 5 lists no plotting library, but `sea report` must produce plots.  matplotlib (already a
transitive dependency) is declared explicitly; plots are static PNGs with a fixed categorical
order per verdict so a verdict keeps its colour across runs.

## C-001 `StepTrace.failed_primitive` (contract change, extends D-005)
System 1 searches "the failing primitive" (spec 8.1), which the trace did not record.  The executor
sets `failed_primitive` to the first motor primitive that returned False, to the primitive running
when a guard's `cap` check failed, or to the one running at an abort (force, workspace, time limit).
Empty when no primitive failed (the program ran to the end but the success check did not hold); then
System 1 searches the first actuating primitive (one with `provides`) that has parameters.

## C-002 Typecheck rules (spec 8.7, Section 9)
`typecheck` returns `TypeCheckError(code, message, path)` with codes `structure`, `unknown`, `arg`,
`range`, `hard_limit`, `tool`, `contact`, `provides`, `ref`, `reach`.  Beyond D-008:
- Force arguments (`force`, `normal_force`) above `max_force_n`, `speed` above `max_tip_speed_mps` and
  `force_below(x)` caps above `max_force_n` are `hard_limit` errors, also when they fall inside the
  primitive's declared range (limits can be tighter than the seed ranges).  Promoted composites are
  checked over their whole argument range (both ends of `$p<i>`), and `reparam_errors` checks System 2
  `reparam` ranges the same way.
- Some actuating primitive acting on the target must `provide` its class and handle (`provides`).
- Without a `control`, contact typing uses the program's `target_cls` (any handle for doors/drawers).
- Primitive targets other than `$target` are STRING arguments (`push("c3")`, Appendix A); check
  atoms use bare control ids.  Unknown ids are `ref` errors when a panel is given.
- Guard caps are not required to bound force: `cap` is any check (e.g. `~latched(c1)`).

## C-003 Reach check in typecheck
The target point is the control's contact point (the handle point for tab/pull handles) at rest,
computed from `PanelSpec`/`ControlSpec` with the same frames as `build_model` (tested against the
sim).  With a plain `ReachGrid` (stock tool frame), a mounted tool is handled by moving the query back
along the approach axis by the tip's z offset: exact for z-only tips (pad, cup, probe, blade,
soft_tip); a hook's lateral `depth` offset (<= 20 mm) is ignored.  `ReachProvider(limits)` gives exact
per-tip grids (`control/reach.reach_for`, ~6 s each, cached in `.cache/reach`) and can be passed
instead.

## C-004 Tool choice for seed programs (`wrap_with_tool`)
Candidates: the stock gripper, then rack slots in order; the first whose contact satisfies the
program's requirements on the control wins.  A candidate whose only usable contacts are banned (stock
`pinch` on a tab or knob under no-pinch) is kept as a fallback and used only when nothing else fits,
so the verifier still reports `rule_violation` when it is the only option (M4: stock gripper on the
tab door gives `rule_violation`; with a rack hook the hook is chosen).  Programs that already contain
`With` are returned unchanged when they type-check.

## C-005 Program order in `applicable_programs`
Programs are ranked by the smoothed pass rate `(passes + 1) / (uses + 2)`, then name, so an untried
program ranks above one that keeps failing.  Each returned `PrimitiveDef` carries the runnable,
tool-wrapped program in `body`.

## C-006 System 1 details (`control/local_search.py`)
- The fixed retry is not counted toward `m`; `s1_trials` counts local-search trials only.  Escalation
  rule 2 is `S1Outcome.escalate == "s1_exhausted"` (trials used up, or nothing left to search: no free
  parameters, `max_param_delta` = 0, or `m` = 0).
- Retry offset: a fresh N(0, 1 mm) perception draw in the panel plane (only when the attempt had
  perception noise; noise-free runs stay deterministic) plus 1 mm toward the control centre (projected
  on the panel plane; none when the contact point is the centre).
- Free parameters: the failing primitive's declared parameters not fixed by the program (named,
  positional numeric or `Bind`).  x0 is the value the primitive actually used: an existing override,
  the motor function's numeric default, or the control-derived default (D-009; force = the episode's
  cap).  Parameters with a control-dependent budget and no fixed default (`hook_pull.distance`) are not
  searched.  Bounds: declared range (or a `reparam` range: keys `prim.param`, or bare `param` for the
  searched primitive) intersected with x0 +/- `max_param_delta` x the declared width; force bounds stop
  at the episode cap (`motor._cap` clamps to it anyway), speed at the hard limit.
- CMA-ES (`cma` 4.5) on the normalized box, sigma0 0.3, popsize min(m, 4 + 3 ln n), seed `seed + 1`.
  Objective: pass first, then the largest travel fraction; structural verdicts in a trial are
  penalized.  The returned trace/result is the best attempt.

## C-007 `reset_episode` resets control state
`ControlDynamics.reset()` restores detent sides and count, latch and travel extremes; `reset_episode`
calls it for every control.  Before this, a second attempt on the same sim inherited detent crossings
and latch state (an immediate false `pass` on detented buttons).

## C-008 Guarded motion and speed limits (test_motor acceptance)
Measured fixes so the 12.2 motor checks hold; reference-panel outcomes are unchanged (c0, c3, c6, c8
pass with the same tools; door/drawer/knob/toggle travel within run-to-run noise):
- `guarded_step`: the first step into an unmeasured contact is a 20 um probe; the measured contact
  stiffness is remembered (decay 0.9 per tick, so a bounce or stiction does not erase it); a step may
  add at most half the remaining margin; in contact steps grow at most 1.5x per tick.  The guard
  stops within 0.5 N of the cap for buttons up to ~12 kN/m (20x the reference button); at 30 kN/m
  the `contact` primitive itself (0.01 m/s, 0.5 N threshold) overshoots by ~4 N.
- `move_to`: trapezoidal speed (1 m/s^2) and the wrist roll turned toward its target at <= 2 rad/s
  (a one-tick 83 deg roll jump whipped the tool point to 2.8x the speed limit); per-tick IK tolerance
  0.05 mm (0.2 mm alternated and jittered the tip speed by +/-30%).
- `Episode.unload()` ramps the command back to the measured tip before `retreat`/`finish`; `retreat`
  leaves a still-loaded contact at 0.01 m/s for its first 5 mm; `finish` backs off only when the tip
  is within the approach offset of the control (a second 5 cm back-off ran the slider retreat out of
  reach).  Measured tip speed now stays under 1.15x the limit on the stock press, slide and toggle.
- Known: toggle and knob contact events still produce tip-force spikes (13.9 N toggle, 11.4 N knob)
  above their 10 N class cap.

## C-009 Test plumbing
Hypothesis tests use `derandomize=True` (12.1).  `scripts/check_package_coverage.py` measures each
package as coverage.py does with branches on: (covered lines + covered branches) / (statements +
branches); top-level modules other than `contracts.py` (`loop.py`, `cli.py`, ...) count only toward
the overall gate.  `test_model_consistency` skips the L2 half until `design/arm.py` exists (M9).

## D-025 IK position weight 30 (world tier; extends Section 8.1)
Observed: the smoke run on train panels gave almost only `unreachable`; on 40 train panels the
stock arm reached 30% of controls on panels tilted < 15 deg, although the down-approach envelope
covers that whole region.  Localized: placo position (m) and axis-align (rad) tasks at equal
weights; when the requested approach axis has a small out-of-plane component (a 5-DOF arm cannot
yaw the tool), the solver gives up 8-25 mm of position for ~1 deg of axis.
Change: `POS_WEIGHT = 30` on the position task (axis task stays 1.0).
Result (40 train panels, residual < 5 mm): tilt 0-15 deg 0.30 -> 1.00, 15-30 0.12 -> 0.48,
30-45 0.15 -> 0.31, 45-60 0.04 -> 0.07; Appendix C: 16/16 unchanged (weight 300 broke 2 points,
so it was rejected).  Reach grids are now keyed on the weight.
Two latent skill-tier errors surfaced once positions were accurate, both fixed:
- Stock pinch engage points (knob, tab fin) put the fixed-jaw tip 0.2 mm *below* the plate;
  `motor.PINCH_HEIGHT` now keeps the jaw tips 2 mm above it.
- Actual tip speed reached 1.25x the limit while commanded speed sat exactly at it (servo lag and
  catch-up); `move_to` now cruises at `CRUISE_FRACTION = 0.85` of the hard limit.
- With position dominant, warm-started tracking could spin the wrist roll (0 -> 2.8 rad in 0.14 s)
  when wrist flex sat at its limit, because the tool point is off the roll axis; the planted
  lever twin then stopped on a wrist-roll limit hit.  `track` now adds a soft joints task holding
  the current wrist roll (`ROLL_HOLD_WEIGHT = 0.05`) whenever no roll target is given; `solve`
  (reach, waypoint residuals) is unaffected.  A split weight (30 for solve, 1-3 for track) was
  tried first and rejected: tracking at weight 3 still spun the roll.
Reference panel before -> after: unchanged verdict classes (button, lever, slider, recessed button
pass; toggle partial; knob over the cap; door 1.14 rad; tab drawer near travel); verifier suite
10/10 cases as before (knob xfail), twins 45/45; golden verdicts unchanged.

## D-026 Seed the encoder's whitening fit (judge-adjacent determinism)
Observed: two `evolve` runs with the same spec in one process wrote different System 2 prompts,
so MockLLM replay raised `FixtureMissing`.  Results, confirmations and ledger matched; escalation
uncertainty did not (train_p008:c0 1.378 vs 1.447).  Localized by refitting the context encoder
three times on identical rows: the context vectors differed, the Cobweb trees did not.
Cause: `PCAICAWhiteningModel.fit` builds `PCA(n_components=k)` without `random_state`.  For a wide
matrix (72 x 768, k = 45 < 0.8 * 72) sklearn's `auto` solver takes randomized SVD and draws from
numpy's global RNG, which earlier sim code had advanced.
Change: `ContextEncoder.fit` seeds numpy's global RNG from the encoder seed for the duration of the
fit and restores the previous state afterwards (same pattern as `memory.concepts._seeded`).
Test: `test_encoder_fit_ignores_global_rng` (wide, rank-40 input, two different global seeds, equal
encodings, global state restored); `test_loop_integration` record/replay now passes.

## D-027 One-parameter CMA-ES searches get a padding coordinate (skill tier)
Observed: `scripts/battery.py --n 16` crashed in System 1: pycma 4.5 `tell` raised
`ValueError: not yet initialized (dimension needed)`.  The search had one free parameter and both
candidates scored the same objective.
Cause: on flat fitness pycma grows sigma; once a coordinate's std exceeds `maxstd` it calls
`sigma_vec._init_(N).set_i(...)`, and `DiagonalDecoding.set_i` treats any size-1 scaling vector as
uninitialized, so N = 1 always raises there.  pycma documents 1-D search as unsupported.
Change: `run_system1` appends one ignored coordinate (x0 = 0.5) when only one parameter is free and
reads back only the real coordinates.  Searches with two or more parameters are unchanged; 1-D
searches draw different samples than before for the same seed.
Test: `test_one_free_parameter_on_a_flat_objective` (six seeds, previously raised).

## D-028 Every module is fingerprinted; contracts and build checks are judge tier
Observed: `contracts.py`, `compose/dsl.py`, `compose/library.py` and `propose/llm.py` were in no
harness component, so editing them left `harness.json` unchanged.  `design/build.py` was skill tier
in `harness.py` but listed as judge ("Screening") in harness.md.
Cause: components were listed by hand when M1-M8 landed.
Change: new judge components `contracts` (HardLimits and Budgets defaults, verdict and contact enums,
coverage bins) and `build` (`design/build.py` applies the hard limits to a body and prices its print,
and the judge decides what counts as cost).  `dsl.py` joins `primitives`, `library.py` joins `memory`,
`llm.py` joins `proposer`.  `harness.TOOLING` names the three modules that only read runs.
The judge key moved, so the B3 baseline was re-recorded; every number in it is unchanged.
Test: `test_every_module_is_fingerprinted` fails when a new module under `src/sea` has no component.

## D-029 System 2 through the Claude Code CLI, billed to the claude.ai plan
Context: live runs should use the Claude subscription, not API credits.
Change: `ClaudeCodeLLM` runs `claude -p --output-format json --json-schema <ProposalBatch>` with
`--tools ""`, `--safe-mode`, `--strict-mcp-config` and `--no-session-persistence`, from an empty
directory, with `ANTHROPIC_API_KEY` removed from the child environment (a key outranks the plan
login).  Before the first call it reads `claude auth status` and raises `PlanBillingError` unless
`authMethod` is `claude.ai`.  `--bare` is not used: it accepts only an API key.  The CLI's strict
schema check rejects `prefixItems`, so `cli_schema` folds tuple ranges into `items`; the answer is
still validated against the pydantic model.  A usage-limit error raises `PlanLimitReached` instead of
recording an empty batch.  `RunSpec.llm` now defaults to `claude_code`; `anthropic` stays available.
Calls log `billing: plan` and the CLI's list-price `total_cost_usd`, so the ledger charges a proposer
the same whichever way it was billed and proposer comparisons stay fair.
Measured (2026-11-01, Max plan, Opus 5.5, effort medium): one full System 2 prompt took 25 s,
6.1k input and 2.3k output tokens, $0.095 at list price, two valid proposals.
Test: `test_llm_mock.py` ClaudeCodeLLM cases (fake CLI runner; no live calls in tests).

## D-030 Scene objects and the stock gripper's grasp envelope (world tier)
Change: `build_model(objects=[SceneObject...], table_z=...)` adds free rigid bodies (or fixed probes)
and a table.  Objects use contype 4 / conaffinity 3 so they rest on the table (contype 2) and touch
the robot; reusing the control bits made them fall through the table.
Measured (top-down pick, jaws along world y, close to -0.17 rad, lift 50 mm, hold 1 s, friction 0.8):
cylinders of 20-40 mm hold at 50-800 g; 50 mm lifts then slips out during the hold; 60 and 70 mm
never lift.  Squeeze is about 22 N per jaw at the 7.4 V stall torque.  So width, not mass, is where
the stock gripper stops, and heavy objects fail at the arm (a rung-5 case), not at the jaws.
Follow-up: record this as the stock gripper fixture table (docs/gripper-experiment.md V1).
Superseded in part by D-032: these numbers were taken with the jaws at MuJoCo's default friction
1.0, which hid the object's friction.  The width limit stands; the 20 mm result does not.

## D-031 Sensing: the robot reads sensors, the judge reads ground truth (world tier)
Change: `sim/sensors.py` reads servo telemetry (position, speed, signed load, current), jaw pressure
pads, per-link contact force and a wrist IMU, with noise and quantization from `configs/sensors.yaml`,
where every channel names the part that would measure it on a real SO-101.  Torque, payload, kinetic
energy and momentum are derived from those channels and the robot's own nominal model (`SelfModel`),
never from ground truth.  The verifier does not read sensors, so sensor noise can change decisions
but never what counts as a pass; that is why the component is world tier, not judge.
Measured: payload estimate 50 / 200 / 398 g for 50 / 200 / 400 g held cans, -2 g idle; an 80 g mass
fault injected at the gripper reads as an 80 g phantom payload with empty jaws.
Test: `tests/test_sensors.py`.

## D-032 Jaw-object friction mixes both surfaces; the stock gripper capability table (world tier)
Problem: the stock jaw geoms carry MuJoCo's default friction 1.0, and MuJoCo gives a contact the
larger of its two geoms' frictions.  So every object was gripped at 1.0 whatever its own friction,
grippier than bare printed PLA, and a TPU pad (finger family, `pad_tpu_mm`) could never have
helped.  The probe showed it: a 30 mm can gave identical results at friction 0.3, 0.5 and 0.8.
Change: when a scene has objects, each jaw-object geom pair is an explicit contact pair with
friction `sqrt(jaw * object)` (the geometric mean, as Box2D mixes friction), the jaw face is bare
PLA at 0.45 (`JAW_FRICTION`), and objects use contype 8 / conaffinity 10 with bit 8 added to the
non-jaw robot geoms, so no jaw-object contact is generated twice.  Models without objects are
unchanged (no pairs, robot conaffinity 1), so the panel world and the battery are untouched.
Instrument: `sim/grasp_probe.py` is a fixed scripted pick (open, descend beside the object, close to
-0.17 rad, lift 50 mm, hold 1 s; held = still 30 mm up).  `scripts/measure_grasp.py` runs 34 cases
in about 10 s into `tests/golden/stock_grasp.json`.
Measured (stock gripper, 7.4 V):
- cylinders 25-45 mm and boxes 20-50 mm hold; cylinders of 50 mm and wider, and 60 mm boxes, do not;
- the 50 mm failure is identical at friction 0.3, 0.8 and 1.5, so it is geometry (the splayed
  faces squeeze the can out), not slip: support for the `pad_angle_deg` hypothesis;
- spheres gripped at the equator all slip out; 20-40 mm spheres hold only with the tips 12 mm below
  it, at any friction, and creep upward during the hold: a profile problem, not a material one;
- the 30 mm can holds to 800 g even at friction 0.3; at 1200 g the shoulder stalls and the payload
  reading saturates near 800 g (a rung-5 case, as the scenario predicts);
- a 20 mm can now topples out of the tips at friction 0.8 (it held at the hidden 1.0);
- squeeze is 40-60 N total (about 23 N per jaw); held objects read their mass to within 40 g.
Consequence: spheres move from the phase-A object set to phase B (gold design: V-groove or concave
pads), and the phase-A set is cylinders 25-45 mm and boxes 20-50 mm, 50-800 g.
Test: `tests/test_grasp_table.py` (scenario claims from the table, a re-measure of three picks, the
contact pairs).

## D-033 Sensors are resources the agent is told about; IMU first, jaw pads toggleable (world, judge key)
Decision (user, 2026-11-01): the wrist IMU is the high-priority sensor (integrated first, built
first on the bench); the FSR jaw pads are toggleable; link skin stays a sim-only research option.
The sensors a body carries are a resource in the same sense as print minutes and plan calls, so
the agent is told exactly what it can feel.
Change: `configs/sensors.yaml` gains a `parts` table (channels, mount, priority, toggle, bench,
mass, what each part tells the robot).  `Design.sensors` (`SensorSuite`: `jaw_pads` on,
`link_skin` off) mounts the toggleable parts, so the suite evolves with the body.  An unmounted
part's channels read `None`, never zero, because "no reading" and "no force" are different facts.
`sensors.manifest(design)` lists every part, and the System 2 prompt gains a "Sensors on this body"
section rendered from it.
Re-key: `SensorSuite` lives in `contracts.py`, so the judge key moved d9408f12e242 -> c64218777cbc.
The battery gave identical numbers under the new key before it was re-saved (D-028 procedure).
Not yet: part mass is listed but not added to the MJCF model (3 g IMU, 1 g pads); a `DesignDelta`
that toggles sensors comes with the finger family, because pads mount on the finger faces.
Also decided: on the real arm, act tools need approval once per session (the server enforces
the hard limits on every call regardless), and the configs/exp suite E0-E11 runs after the gripper
work.
Test: `tests/test_sensors.py::test_unmounted_parts_read_as_absent_and_the_manifest_says_so`.

## D-034 System 2 becomes a planner over a task network; one-shot stays the fast path (policy, design)
Decision (user, 2026-11-01): System 2 should decompose goals into sub-goals, assign them to
System 1 repeatedly, and revise from what comes back, grounded in hierarchical task networks and
task decomposition from cognitive science.  The one-shot proposer remains for rote cases.
Design: docs/goals.md.  The main choices:
- Goal tests are written in a judge-tier vocabulary, so the planner cannot grade itself.
- Retrieval comes first (Soar chunking, ICARUS skill indexing): a known method runs without an
  LLM.  A known impasse with a known fix goes to one-shot `s2`.  Only new problems get agentic
  decomposition.
- The impasse type picks the sub-goal type.
- Every node carries a budget slice and the sensor manifest.
- Reward stays at the root; sub-goal outcomes rank methods and never enter the reward.
- A body change is a sub-goal whose effect is to make a method's precondition true.
Build: the current loop is expressed first as the degenerate network (one root, one leaf per
control, one-shot System 2 at impasses), and the battery checks that nothing moved.  Agentic depth
comes with the `sea-arm` server.

## D-035 Grasp world v0: objects are controls whose coordinate is their height (judge, world, skill)
Change: the grasp world runs through the panel pipeline unchanged instead of beside it.
- **Tasks:** `configs/grasp.yaml` and `sim/grasp_world.py` (judge tier) generate tables with one
  object each:
  - phase A: cylinders 25-45 mm and boxes 20-50 mm, 50-400 g, friction 0.5-1.0, placed within
    4-5 cm of the table centre;
  - phase B: five fixed objects, each failing for one known reason.
- **Contract:** a `grasp` object is a `ControlSpec` with a shape and a mass. Its success check is
  the new `held_for($target, 1.0)`: at least its 30 mm travel above rest, continuously for 1 s.
  A lift that falls back is therefore `wrong_behavior`, and a grip that never lifts is `stuck`.
- **Sim:** `build_model` makes each object a top-level free body (MuJoCo allows free joints only
  there) with object collision bits, and mixes its friction into the jaw contacts (D-032). Its
  control coordinate is the free joint's z, so `control_q`, `q_rest`, travel and every verdict
  rule apply unchanged. A floor 0.4 m below the table catches objects squeezed off the edge.
- **Executor:** once `grasp` starts closing, the jaws' squeeze on the target is left out of the
  hard force limit and counted against the grasp class cap. The servo bounds it: about 46 N static
  at stall, closing peaks measured at 54-83 N, cap 120 N. Every other contact, including a jaw
  landing on the object during the descent, stays under the 15 N limit (`configs/limits_grasp.yaml`,
  which also allows the pinch).
- **Skill:**
  - the `grasp` motor primitive has 7 System 1 parameters: depth, side clearance, close target,
    close time, lift, hold and speed;
  - the seed program is `pick_up = grasp($target)`;
  - the typecheck's aperture rule (`requires: {aperture: width}`) rejects an object wider than
    `design/grip.py`'s widest stable grip, which is the V1 numbers for the stock jaws (cylinder
    45 mm, box 50 mm, sphere 40 mm with a deep grip). An object too wide for every program is
    `no_applicable_primitive`, which is structural and escalates before any simulation.
- **Battery:** `scripts/battery.py --world grasp` (17 s) reports pass share per shape into its own
  baseline.

Measured (stock gripper, System 1 at m = 2):
- phase A train 15/16 and val 16/16. The one failure is a tall, heavy can (34 mm wide, 77 mm tall,
  388 g) that cleared its travel and slid out during the hold, which is a grip phase A should learn;
- phase B 0/5, each with its declared failure: the 55 and 65 mm cans and the 60 mm box are
  `no_applicable_primitive` (aperture); the 30 and 40 mm balls are `wrong_behavior` and `stuck`. With
  m = 8, System 1 still does not find the deep sphere grip that V1 shows exists, so the balls
  escalate as `s1_exhausted`.

Panel world: `grasp` is a `ControlClass` but not in `CONTROL_CLASSES`, so panel generation and the
memory encoding are untouched, and the seed program `pick_up` applies to grasp objects only. Judge
key c64218777cbc -> 27ac5d986ca4 (contracts, grasp world, grasp limits). The panel battery's
numbers and the golden verdicts were identical before the baseline was re-saved.
Not yet: memory's class one-hot has no `grasp` slot (grasp results share the zero vector), and the
loop cannot yet be pointed at the grasp world (the phase A->B runner, build step 5).
Test: `tests/test_grasp_world.py`.


## D-036 Sensors in the loop: traces, memory, System 2's summary and two holding reflexes (judge, policy, skill)
Change: what the robot feels now reaches every consumer except the judge.
- **IMU model:** the accelerometer reports the mean acceleration since the last reading (the
  difference in wrist velocity over the tick, minus gravity, in the gripper frame), as a real
  IMU's low-pass filter does. Sampling the 500 Hz contact chatter instead read 0.8-4 m/s² of
  "vibration" while the jaws held still at stall; the median is now 0.03 m/s².
- **Trace:** `StepTrace.sensors` is a `SensorTrace`:
  - 10 Hz samples: dynamic wrist acceleration, the largest arm-joint load, gripper load, pad
    force, and payload while the arm is still;
  - per-tick features: `peak_acc`; `impacts` (over 1 g of dynamic acceleration, 0.1 s apart);
    `peak_load` and `stall_s` (the longest run of an arm joint at 95% of stall or more), each with
    its joint; `peak_pads_n`; `pad_drop_t` (the pads lost a grip with the jaws commanded shut and not
    opening); `payload_held_g` (the last still reading after the lift).
- **Calibration (measured over both batteries):**

  | World | Peak wrist acceleration p50 / p95 / max (m/s²) | `stall_s` p50 / max (s) |
  |---|---|---|
  | Grasp | 1.9 / 3.7 / 4.7 | 1.0 / 2.1 |
  | Panels | 4.2 / 8.5 / 8.9, plus 16.8 on one `force_limit` drawer | 1.2 / 2.3 |

  The impact threshold is therefore 1 g (10 m/s²), which no normal move reaches. Brief saturation
  is normal servo behaviour (accelerating, or holding a squeeze), so neither peak load nor stall
  time alone is a signal.
- **Reflexes (System 1, `Episode._holding_reflexes`).** They read `SensorFrame`s only, and only
  while the jaws hold a grasp target:
  - `load_limit`: an arm joint at 95% of stall or more falls more than 0.05 rad behind its command
    for 0.5 s. While holding, normal picks lag at most 0.003 rad, the 700 g can included; 900 g and
    1200 g cans lag 0.17-0.27 rad at the shoulder.
  - `slip`: the pads read 5 N or more, then under 1 N, with the jaws still commanded shut. It needs
    the pads. Without them a ball's slip goes unnoticed, so the jaw-pad toggle has a cost the agent
    can read in the manifest.
  - An impact is a measurement, not a reflex. In the sim the force guard stops every contact first,
    so an IMU reflex at 1 g would only race it.
- **Results and memory:** after the verdict, `env._judge` copies `peak_acc` and `stall_s` into
  `ControlResult`. `MEASUREMENTS` gains both, so memory's expectations include them and a sensor
  residual counts as surprise, which is the phase C signal. SD floors are 0.2 m/s² and 0.04 s.
- **System 2:** `TraceSummary` gains `felt`, `stopped_by` (the trace note), `peak_wrist_acc`,
  `stall_s` with its joint, `payload_held_g` and `pad_drop_t`. The failing-controls section shows
  them on a `felt:` line, and an attempt without sensors adds no line.
- **Seeds:** sensor noise is seeded by the panel draw (`PanelRunner.seed`).

Sensing never changes the physics: a test runs a pick blind and sensed and compares them tick for
tick. The reflexes only stop an attempt early, and the verdict still comes from ground truth.
Measured on the stock gripper with pads:

| Object | Verdict | Stopped by | Felt |
|---|---|---|---|
| Phase A picks, 700 g can | pass | - | payload within 10 g of the true mass |
| 900 g and 1200 g cans | `wrong_behavior`, `stuck` | `load_limit` | shoulder stalled and behind its command |
| 50 mm can, 30 and 40 mm balls | `wrong_behavior` | `slip` | pads lost the grip |

Both batteries' numbers, the golden verdicts and the V1 table are identical: the reflexes change
the diagnosis, never the verdict. Judge key 27ac5d986ca4 -> da5003023c10 (contracts).
Open:
- System 1's guards in the panel primitives still read ground-truth contact force
  (`Episode.last_force`), an ideal force sensor the real arm does not have. Before V5 they move to
  felt signals (servo load, pads, IMU). That changes the panel world's numbers and needs its own
  re-baseline.
- The sensor suite is not yet part of memory's context, so concepts do not separate bodies with
  and without pads.

Test: `tests/test_reflexes.py`.

## D-037 Task network contracts; the loop is recorded as the degenerate network (judge, policy)
Change: the goal layer of D-034 exists as contracts, and every run writes its network.
- **Judge (`verify/goals.py`):** goal tests above one control, read only from what the judge has
  already decided:
  - `pass_share(SET, x)`: at least a share x of a named set's controls pass;
  - `still_passes(SET)`: every control of SET that passed before the change still passes;
  - `accepted()`: the paired trial passed the acceptance rule.

  Tests combine with `&`. A goal on one control keeps its check expression (`compose/checks.py`).
  The planner composes tests from these two vocabularies and cannot add to either.
- **Policy (`plan/network.py`):**
  - `Goal`: test, target control, the goals it serves, rationale, slice, and the task that serves it.
  - `Task`: skill, analyze, design or network, with the sensor parts it needs.
  - `Method`: HTN's decomposition one level above a composite program.
  - `GoalReport`: the judge's verdict, impasse, reflex stop, program and spend. It is not called
    `Report`, which is already the iteration report.
  - `IMPASSE_SUBGOAL`: structural makes a design goal; s1_exhausted and unfamiliar make a network
    goal; surprise and reflex make an analyze goal; an exhausted slice returns to the parent.
- **Slices:** a slice is a `CostVector` over the ledger's budgeted resources. Servo heat is priced,
  not sliced. A goal without a slice shares its parent's. The root gets every budget less the
  reserve, and the explicit slices of a goal's children must fit inside its own.
- **Validation:** `validate` refuses a network before dispatch for any of these: no single root, a
  dangling parent or served goal, a cycle, a test outside the vocabulary or naming an unknown set,
  a root with no slice, children whose slices exceed their parent's, a skill naming a program that
  is not in the library, or a task that needs an unmounted sensor.
- **The degenerate network (docs/goals.md 3.6):** `evolve` records the loop as it runs.
  - The root's test is `pass_share(train, 1)`, served by the `coverage` method.
  - Each control System 1 serves is a leaf with that control's success check. It gets one report
    per iteration: verdict, escalation reason as the impasse, reflex stop, program, and sim time at
    the ledger's estimate (D-016, now `env.attempts`).
  - Each System 2 proposal is a design node serving that iteration's targets. It closes on
    `accepted()`, and its spend is the ledger's difference across the trial.
  - At the end the root closes `passed` if its test holds, `exhausted` otherwise.

  `goals.jsonl` holds one event per line (open, report, close), and `replay` reads it back.

Recording changes nothing: a smoke run before and after has 72 byte-identical output files
(results, reports, costs, proposals, decisions, designs, libraries, memory, traces), with
`goals.jsonl` the only new one. Both batteries and the golden verdicts are identical. Judge key
da5003023c10 -> b1706b1856a4 (`verify/goals.py` joins the verifier).
Not yet: `plan/agent.py`, the agentic System 2 that writes networks deeper than one level. It
needs the `sea-arm` tool server, because an analyze goal needs tools to answer it.
Test: `tests/test_network.py`.
