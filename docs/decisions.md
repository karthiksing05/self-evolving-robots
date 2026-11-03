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

## D-038 The finger family and 21 real objects; three faults found while measuring them (judge, world, skill, policy)
Change: System 2 can now change the fingers (L1), and the gripper experiment runs on real objects.
- **Finger family (skill, `design/fingers.py`):** `FingerSpec` is `reach_mm`, `pad_angle_deg`,
  `groove_mm`, `lip_mm`, `pad_tpu_mm` and `width_mm`, all zero for the stock jaws (ranges in
  docs/gripper-experiment.md section 3).
  - One pad box goes on each jaw, over the bottom 10 mm of the stock face. The stock geometry
    constants come from the SO-101 jaw meshes.
  - The moving pad is tilted so the two faces are parallel at a jaw opening equal to the pad
    angle. It is lengthened by its tip's rise, so both pads end level there.
  - `build_model` adds the pads as jaw geoms, so they pinch and pair with objects. Their friction is
    PLA 0.45 or TPU 0.9, combined with the object's by the geometric mean (D-032).
  - `finger_estimate` prices the pad volume: print minutes, grams of PLA and TPU, added mass.
  - The loop enables L1. Screening refuses a finger proposal when the run has no print budget.
- **Grasping with pads (skill, `control/motor.py`; world, `sim/grasp_probe.py`):** for a pad
  design, the grasp primitive and the V1 probe:
  - set the fixed pad's face, not the stock face, beside the object;
  - limit depth by the lowest pad point at the object's jaw opening;
  - open wide enough for the pads to clear the object.

  With stock jaws both reduce exactly to their old settings. The V1 table re-measures with 0
  changes in 34 picks.

  Before descending, `grasp` now rises to the approach height, but only when the jaws would pass
  lower than the object's top. An unconditional rise moved the stock heavy pick 3 mm and reoriented
  the wrist at home, which changed the joint that stalls. The conditional rise leaves both batteries
  identical.
- **Real objects (judge and world):** 21 YCB scans (Calli et al. 2015, CC BY 4.0) with their
  published masses and material frictions.
  - `scripts/import_ycb.py` writes `configs/ycb_objects.yaml` and `assets/ycb/*.npz`: convex
    collision pieces, a decimated visual mesh, and the inertia at the published mass. The scan
    meshes are processed by an agent-written script.
  - Each object rests in a pose no taller than 85 mm. A straight-down SO-101 tool point tops out
    near 95 mm, so a taller object cannot be lifted 30 mm. That ceiling is the arm's, not the
    harness's.
  - New code: `sim/objects.py`, and `real_object`, `real_panels` and `regression_panels` in
    `grasp_world.py`.
  - `RunSpec` gains `world: panels | grasp`, `objects` and `regression_objects`. The loop takes its
    train, eval and regression sets from `world_panels` and `regression_panels`.
- **Capability (skill, `design/grip.py`):** one instrument measures every design: the V1 probe.
  - The stock jaws read its stored table. A finger design is probed on demand and cached by its
    parameters and the probe's source.
  - The typecheck's aperture rule now skips scanned objects. A scan's bounding width is not the
    width the jaws close on: the stock jaws hold the banana (69.6 mm bounding width) and the cup
    (57 mm), which the table would have called structural.
- **System 2's prompt (policy):** each failing control now shows:
  - the object, its size and mass;
  - the contacts the judge accepts on it.

  The design section adds the finger family with its ranges and the current design's capability. A
  new section, "Rejected before trial", lists the last six proposals that screening refused, with
  the reasons.

  In the first live smoke run every proposal of all three iterations was screened out ("no rack
  tool provides the contacts"), and System 2 was never told why.

Measured (`scripts/measure_real.py`: `pick_up` with System 1's search and confirmation re-runs, at
three placements):
- **Stock jaws:** hold 5/21 objects: banana, cup, foam brick, marker, cube. These are phase A and the
  regression set. The other 16 are phase B.
- **Gold check (`results/gold/`):** two designs hold 9/21 and lose nothing:

  | Design | Objects gained |
  |---|---|
  | reach 10, 20° tilt, 2 mm TPU | golf ball, lemon, plum, strawberry |
  | reach 15, 30° tilt, 2 mm TPU | gelatin box, golf ball, plum, strawberry |

  Steeper tilt holds more round objects and loses small ones: 40° with a groove holds 11/21 without
  the cube, marker and banana. Five objects nothing holds: sugar box and mustard bottle (too wide),
  pudding box and tuna can (stall the arm), dice (16.8 mm).

Found while measuring, and fixed:
1. **`opening_for` bisected up to 1.7 rad.** Past about 1.2 rad the moving pad swings toward
   horizontal and the face-line gap goes negative, so the bracket check failed. It returned None,
   and every pad grasp fell back to fixed jaw angles. It is now bounded to the openings the skills
   use. Before the fix the best design was +4 objects and -3.
2. **TPU had a contact time constant of 0.04 s.** That was an unmeasured guess at the skin's
   compliance, and it made TPU pads drop small objects that bare PLA pads held: 0/5 against 3/5 on
   the small and round set. The skin is now modelled by its friction only.
3. **The aperture table rule** would have made two objects the stock jaws hold structural (above).

Numbers unchanged:
- both batteries (identical splits under the new key);
- the golden verdicts;
- the V1 stock table.

Judge key b1706b1856a4 -> 4ffb19f19f33: contracts, `configs/grasp.yaml`, `grasp_world.py` and
`sim/objects.py` changed, so the baselines are re-keyed.

Tests: `tests/test_fingers.py`, plus the prompt tests in `tests/test_llm_mock.py`.

## D-039 Three gaps the first live gripper run exposed in the harness (policy)
The first live gripper run (`configs/exp/gripper_live.yaml`, results in `results/experiments/`) showed three
places where the harness, not System 2, was the problem.
- **Edits are changes, not keys.** In iteration 4, a finger proposal restated two fields at their
  current values (`lip_mm: 0`, `pad_tpu_mm: 2`). Screening counted 5 edits against the posture's 3
  and refused it, though only 3 values changed. `ladder.count_edits` now takes the current design
  and does not count a field restated at its current value. Without a design it counts keys as
  before.
- **A past fix names what it broke.** In iterations 1, 2 and 6, System 2 steepened the pad tilt to
  reach the 73 mm gelatin box. Each time, the banana and the wood cube regressed and the trial was
  rejected. Its "Similar past fixes" section showed only `regressions 2`. A rejected fix now adds a
  `broke:` line naming up to six regressed controls.

- **Finger changes are priced.** The live run's ledger shows 0 print minutes, although it accepted
  a finger change: screening added print volume only for new rack tools, and the trial charged
  screening's estimate. Screening now adds both pads whenever the fingers change. `build.finger_print`
  gives the volume, PLA and TPU of the new pads: a changed pad is a new part. The accepted design
  (25° tilt, 3 mm groove, 2 mm TPU) costs about 14 print minutes, 0.6 g PLA and 0.4 g TPU. D-038's
  "priced by the ledger" held for a design's estimate, not for the charge.

None of these changes touches the judge or the world. The batteries do not use System 2 and are
unchanged. The live run they come from used the harness before this entry (its `harness.json`).
Tests: `tests/test_ladder.py`, `tests/test_llm_mock.py`.

## D-040 The grasp world started some episodes inside the object (judge validity)
Diagnosing why the potted meat can failed every gripper design showed a 1002 N contact at t = 0.
`HOME_Q` holds the tool 80 mm up at x = 0.18 m, and the 83.5 mm can, 101 mm deep and centred at
0.22 m, reached under it. The episode started in collision and tripped `force_limit` before any
motion: a broken task, not a gripper limit. SWE-bench's rule applies: validate the task before you
judge the agent on it.
- `executor.GRASP_HOME_Q` starts grasp-world episodes with the arm's lowest point 112 mm above the
  table over the object area. That clears the tallest object, which is 85 mm. `reset_episode` picks
  the home pose from the panel. Every other world keeps `HOME_Q`.
- With the stock jaws, the potted meat can is now held at all three placements. The stock jaws hold
  6/21 objects: phase A is banana, cup, foam brick, large marker, potted meat can and wood cube
  (`results/real_stock.json`; the old file is in `results/superseded/start_collision/`).
- Both batteries are identical: panels 3/5/12 passes, grasp 15/16/0. Their verdicts did not move, so
  they are re-keyed under the new judge key with D-041 and D-042.

## D-041 A world-agnostic evolution core: parts, resources, and redesign as a move (judge + policy)
The user asked for redesigns to arise from the harness, not to be hand-made, and to be general across
domains. They also asked that every new part be grounded in what the bench actually provides.
`sea.evolve` is a second loop alongside the panel loop. It is built on four ideas:
- **A body is one part per slot.** Each part type has physical parameters with ranges and docs.
  System 2 has three moves, cheapest first:
  - `params`: new values for the current parts;
  - `swap`: another part type in a slot, a redesign of that part;
  - `idea`: a part no catalogue has. It is recorded with what it needs and never trialled.
  
  The loop labels each trial with the move it actually made, whatever the proposal claimed.
- **Resources ground every move** (`resources.py`). A world's manifest lists items on hand, items that
  can be bought (price, lead time) and machine time per fabricated material. The check for each
  candidate:
  - reuses whatever the current body already has;
  - buys shortfalls within the run's budget;
  - charges fabrication only for slots whose part changed;
  - refuses anything the bench cannot provide, and says so. That refusal reaches System 2 in the next
    prompt, so it learns what is provided.
- **Diagnosis drives escalation.** Each world returns a mechanism label and the numbers behind it for
  every failure, and a `brief`: computed facts about the current body, never advice. The prompt has
  the same sections in every world: the harness's seven principles; the world, its brief, physics,
  catalogue and resources; the held and failing groups; and the record of past proposals. The record
  includes a per-slot summary of what parameter-only trials have not fixed. A swap is then the
  natural next step when that summary and the brief show the current part cannot reach.
- **A ratchet.** A candidate is kept only if it gains a task group and loses none. The held set only
  grows, the PASS_TO_PASS rule. Held groups are trialled first, so a regression stops a trial early.

Arms: `s2` (System 2 on the Claude plan through the Claude Code CLI, never API credits), `random`
(blind catalogue edits through the same screen and trials) and `s1` (the stock body).
- Every prompt, answer, screen, trial and decision is written to the run directory.
- The final body and the stock body are both scored on the held-out test split.
- Worlds register by name (`sea/worlds/__init__.py`). `scripts/evolve.py` runs one.

Tests: `tests/test_evolve.py`, a toy world covering every path: a ratchet rejection, a kept params
move, a refusal because an item is not provided, a recorded idea and a swap that buys a part.

## D-042 The moving jaw's joint is a part; the gripper world (world + judge)
For the 21 YCB objects, the diagnosis (`worlds/gripper.py: diagnose`, which replays each failed
attempt and reads the contacts) separates three causes:
- **Wide, low boxes** (gelatin, tuna, pudding, sugar, mustard): on the hinge, the moving fingertip
  rises 14 mm at 75 mm and 35 mm at 95 mm. The finger presses the box's top edge or top face
  (`moving_jaw_on_top_face`, `squeezed_down_out` with the contact 10 mm above centre, pushing 48°
  down).
- **Round objects:** both contacts sit above the equator and push down. On the lift, the object is
  squeezed out (`squeezed_down_out`).
- **The potted meat can:** start collision (D-040).

No pad on a hinged jaw fixes the first cause for a box 30 mm tall. The pad would have to hang 30 mm
below the fixed fingertip, which then hits the table on a 17 mm die.

So the catalogue offers the moving jaw's joint as a part, chosen from the basic lower pairs rather
than designed for any one object. `design/fingers.py` and `sim/build_model.py` build each:
- `revolute`: the stock hinge;
- `parallelogram`: a four-bar with link length `link_mm`. The finger keeps its orientation but still
  rises by `link (1 - cos q)`. It is a coupler body held at minus the hinge angle by a joint equality,
  with jaw-jaw contact excluded as for the hinge.
- `prismatic`: a rack-and-pinion slide, `drive_mm_per_rad` 20–60. The gripper joint becomes a slide.
  The actuator's gear is 1/radius, so the servo still takes radians and its torque arrives as force.
  Damping and rotor inertia are reflected. `Sim.arm_q`, `servo_ranges` and the sensors report the
  servo angle. The executor clips commands in servo units; it had clipped a slide's command at
  0.096 "rad", its range in metres.

`fingers.fingertips` and `tip_opening` give the fingertip gap and the moving fingertip's height
relative to the fixed one. The brief and the diagnosis read those; the face-line `gap` of D-038 says
where faces are parallel, and stays.

The gripper world (`worlds/gripper.py`):
- **Tasks:** 21 groups. Each has a centred table and one seeded placement to train on, and two other
  placements to test on.
- **Slots:** `jaw` (the three joints), `pads` (none or the D-038 family) and `arm_servos` (7.4 V, or
  12 V on the lifting joints).
- **Bills of materials:** printed grams at 5 min/g PLA and 8 min/g TPU, steel rods, bushings,
  bearings, servos and a 12 V supply. The budget is $40, 7 days of lead time and 600 machine minutes.
  The 12 V upgrade costs $68, so it cannot be bought within the run as configured. The prompt says
  so.
- **System 1:** `run_panel` with its search, so verdicts match the panel loop.

A measurement, not a gold design: with the slide at 55 mm/rad and no pads, `measure_real` holds the
gelatin box, tuna can, pudding box, wood cube and tennis ball at both placements tried.

Revolute behaviour is unchanged: `has_pads` replaces `is_stock` where the code meant "has pads", and
every hinge function returns what it did. The batteries are identical. The judge key changes through
`contracts.py` (three `FingerSpec` fields) and the new judge components (`evolve`, `world_gripper`).
Tests: `tests/test_fingers.py` (`test_jaw_mechanisms_d042`), `tests/test_world_gripper.py`.

## D-043 What the first evolve run on the gripper showed the core was missing (policy + judge)
Iteration 0 of `runs/evolve/gripper_s2_s0` (System 2 on the plan) proposed three things at once:
- pads tilted 20°: kept, gaining the golf ball and the screwdriver;
- a 60 mm/rad slide with flat TPU pads: rejected, it lost the potted meat can;
- an 80 mm parallelogram: rejected, it lost the banana.

So the jaw redesigns arose from the brief without any prompt to make them. The record exposed four
gaps, fixed for the next runs (that run keeps the code it started with):
- **A redesign was judged only on what it broke.** The trial stopped at the first loss, so nobody
  learned what the slide would have gained. A swap is now always scored on every group, and
  `full_trials` does the same for params moves.
- **There was nowhere to stand while repairing a redesign.** A body that gains more groups than it
  loses goes on a **frontier**. The prompt lists it with what it would gain and why it lost, and a
  proposal can build on it with `base`. The ratchet still applies: the result must keep every held
  group. That is the usual shape of a redesign: break one thing, then repair it.
- **Part types showed their docs, not their numbers.** System 2 assumed the slide had the hinge's
  servo range. `World.part_facts` (optional) now gives each part type fitted to the current body at
  the ends of its parameter ranges. For the gripper that is its servo range, widest fingertip gap,
  and face tilt, fingertip rise and squeeze at 35, 55 and 85 mm.
- **A body could be paid for twice, and a run recorded no fingerprint.** A body already trialled is
  refused with its earlier result. `harness.json` is written at the start of every run. Each trial
  scores its predicted gains against the actual ones (`prediction`), and `summary.json` aggregates
  moves, refusals, regressions and mean prediction precision.

Tests: `tests/test_evolve.py` (`test_frontier_lets_a_redesign_be_repaired`, repeat refusal,
prediction scoring) and `tests/test_world_gripper.py` (`test_part_facts_show_each_joint_on_this_body`).

## D-044 A new part is tuned before it is judged; System 2 sees held tasks, loss numbers and swap costs (policy)
A swap's parameters are System 2's first guess. A swap is judged against a body whose parameters have
had several iterations of tuning, so a good mechanism can lose to a well-tuned bad one. NEAT protects
structural innovations for the same reason (Stanley & Miikkulainen 2002): a new structure needs its
weights fitted before it competes.

The loop now gives each iteration a **tuning allowance** (`tune_trials`, default 4):
- **What gets tuned:** the swap this iteration that gained a group but lost a held one, taking the
  one with the best net gain.
- **How:** a Gaussian search over the parameters of every slot that body changes. It works in each
  parameter's own range, centres on the best tuned body so far, and its step shrinks. Each tuning
  trial is screened, costed and scored on every group, like a proposal.
- **Objective:** kept-eligible first, then net groups, then tasks newly passed, then partial credit.
- **When it stops:** at the first kept-eligible body.
- **Logging:** tuning trials appear in the record as `tune` rows, and the summary counts them.
- **Same for every arm:** the random arm gets the same allowance, so the comparison still measures
  what reasoning buys.
- **What System 2 is told:** "judge a part type by what it can do across its range".

Three prompt gaps, found while building the mobile world:
- **Held groups:** shown with their task facts. Before, System 2 could not see the 9 cm sofa clearance
  until a body broke it.
- **Rejections:** show each lost task's numbers, not only its mechanism.
- **Catalogue costs:**
  - Each catalogue part now shows its true marginal cost from the current body ("swapping it in now
    costs $14.00, 4 days, 57 machine min"), or why it is refused.
  - Items the current body already has show as reused, not bought.
- **Calibration:** a section compares System 2's predicted gains and named risks with what its trials
  found.

The two formatting helpers (`availability`, `manifest_text`) moved from `resources.py` (judge) to
`prompt.py` (policy). What System 2 is shown is policy. Only `check` decides admissibility.

Tests: `tests/test_evolve.py` (`test_tuning_protects_a_new_part`: a swap that breaks a group at its
midpoint is tuned into a keeper; with `tune_trials=0` it is not).

## D-045 Kept-eligible proposals in different slots are merged (policy)
In the first mobile run's iteration 0, System 2 made three proposals and each was kept-eligible:
- a narrower tub with more cells, gaining the lawn;
- a sonar, gaining the hallway;
- tracks, gaining the garden, the sandbox and the lawn.

The loop kept only the tracks, and the sonar waited an iteration. Gains in different slots are not
alternatives. After its trials, the loop now merges kept-eligible candidates:
- It takes the best candidate, then adds each other candidate whose changed slots are disjoint from
  those already taken.
- If that joins at least two, it trials the union once, screened and costed like any proposal. The
  union is kept if it scores higher.
- The ratchet still decides: the merged body must keep every held group.

`RunConfig.merge` (default on) turns it off, and the record shows `merge` rows. The mobile seed 0 run
predates this change. Seeds started later carry it in `harness.json`.

Tests: `tests/test_evolve.py` (`test_disjoint_gains_are_merged`).

## D-046 A spare is used once: the run keeps a ledger of what it owns (judge)
The phonology world's author found the problem. `resources.check` counted the manifest's on-hand
spares plus the current body's items as available. After a kept change used two spare cells, those
cells were in the current body and still counted as spares. Every later change therefore bought less
than it should.

The run now keeps `Spent.owned`, which starts as the stock body's items. Each purchase of a kept
change is added to it (`resources.start`, `resources.charge`, `ResourceReport.buy`). A candidate's
pool of an item is the manifest's spares plus what the run owns:
- a spare is counted once, whichever body holds it;
- going back to an earlier part still costs nothing new.

A check with no ledger keeps the old single-step behaviour.

The gripper manifest listed the kit's six 7.4 V servos as on hand as well as in the stock arm, so it
now lists none spare. No gripper body was affected: the only servo change replaces three of them with
12 V servos, which is a different item. The mobile and phonology manifests already listed spares
only.

All three worlds and `resources.py` change, which re-keys the judge. Runs started before this commit
(gripper `s2v2_s0`, the first mobile seeds) carry the earlier key in `harness.json`.

Tests: `tests/test_evolve.py` (`test_a_spare_is_used_once`).

Amendment (after `phonology_s2_s0`): tuning now runs only if the promising swap gained more groups than
the best kept-eligible candidate that iteration. In that run, a merge already gained seven groups
while the loop spent four tuning trials on a 15.5 cm cast that gained one and broke four. A tuned
body can be kept only if it outscores the best candidate. A swap that already gains fewer groups is
unlikely to, so the allowance goes where it can change the decision.

## D-047 An idea is a costed bill of materials (policy + judge)
An `idea` used to carry its needs as free text, matched loosely against the manifest's item names.
That fell short of the rule that every new part must acknowledge whether the bench provides it. An
idea now names a slot and a bill of materials (`Need`: item, quantity, and for an item the manifest
lacks, System 2's estimated price and lead time).

`resources.cost_idea` costs it the same way `check` costs a part, against the same ledger:
- provided;
- bought at the manifest's price;
- cannot be bought;
- not on the manifest, at System 2's own estimate;
- machine minutes for printed grams.

It also says whether the whole idea fits what the run has left. The record, the prompt's idea list
and `summary.json` carry that costing. Nothing is charged, because an idea is never built.

So an idea is a proposal for the outer loop with a price on it. For example, a shoulder spring that
needs a $3 spring and 20 g of PLA is a cheap catalogue addition. A 12 V arm is a budget decision.

Also noted: the phonology world's gold path costs $24, not $16, once the spare servo is counted once
(D-046).

Tests: `tests/test_evolve.py` (the jet-pack idea: a $4 rail from the manifest, a $50 jet at System 2's
estimate, 30 machine minutes, over budget).

## D-048 Friction that does not creep: impratio 10 in grasp scenes (world + judge)
In `gripper_s2v2_s0` a 50 g plum, squeezed at about 30 N by each jaw, "slipped under load". With
friction near 0.5 that is about 60 times the force needed to hold it. The same body and tasks were
run at MuJoCo's default `impratio` of 1 and at 10:
- at 1, the lemon, the plum and the apple each lost a placement to `slipped_under_load`;
- at 10, all 16 tasks held.

MuJoCo's soft contacts let friction yield under a steady load unless frictional impedance is raised.
The XML reference describes `impratio` > 1 as preventing slip without raising the friction
coefficient. MuJoCo Menagerie's gripper scenes use 10. So the slip was the simulator, not the
design, and no proposal could have fixed it (rule 2).

`build_model` now sets `impratio = GRASP_IMPRATIO = 10` in every scene where the jaws pair with a
free object. Panel scenes are unchanged, and the B3 panel battery is identical at 3/5/12. Every
instrument built on the old physics was re-measured, and the old records are kept:

| Instrument | Before | After |
|---|---|---|
| V1 stock table (`tests/golden/stock_grasp.json`) | cans to 45 mm, boxes to 50, spheres to 40 (tips 12 mm low) | cans to 55 mm, boxes to 60, spheres to 50 |
| V1 hold height of held picks | scattered from −13 to +73 mm | 39 to 55 mm, the commanded 45 mm lift |
| Grasp battery (train / val / phase B) | 15/16, 16/16, 0/5 | 16/16, 16/16, 1/5 |
| Stock jaws on the 21 real objects (`measure_real`) | 6/21 | 13/21 |

Two readings of the old table were wrong:
- *"The 50 mm can fails at every friction, so it is geometry."* A failure that does not move with
  friction is also the mark of creep. The can now holds at friction 0.3, 0.8 and 1.5 alike.
- *"Hold height noise."* The scatter was objects creeping out mid-hold.

What changed with the physics:
- **Typecheck:** the aperture rule reads the re-measured widths, `grip.STOCK_WIDEST_MM` = 55 / 60 /
  50 mm.
- **Phase B:** its wide objects moved just past the new envelope. `wide_can_55` became
  `wide_can_60`, and `wide_box_60` became `wide_box_70`.
- **The ball:** System 1 alone now holds `ball_30` with a below-equator grip. So phase B has four
  objects the stock fails, not five. The analytic world is left at that rather than redesigned.
- **Stock record:** `results/real_stock.json` is re-measured, and the old one is in
  `results/superseded/impratio1/`.
- **Stock failures that remain** are the wide boxes and the bottle (the splayed stock faces), the
  tuna can (an arm stall) and three marginal fruit placements.

Two gripper evolve runs predate this change, and their `harness.json` records the old code:
- `gripper_s2_s0` (13/21);
- `gripper_s2v2_s0`, stopped at iteration 7 with 19/21 held. Its last two groups were failing on the
  creep.

Tests: `tests/test_grasp_table.py` and `tests/test_grasp_world.py` (updated to the new envelope);
the golden files re-recorded from reviewed diffs.

Amendment (after `gripper_s2v3_s0` iteration 1): a tuning step now moves one or two of the swap's
parameters, not all of them. Tuning the slide with pads perturbed all seven parameters at once, and
each of its four trials broke a different held group. A swap's values are System 2's reasoned guess,
so a sparse step around them is the better search. Parts with a single parameter are unaffected.

## D-049 Pictures of a run come from CAD, outside the ruler (report)
A run's record said what changed in words: `jaw -> prismatic (drive_mm_per_rad=56)`. People reading
a run need to see the part. `scripts/render_bodies.py` renders the CAD of every body a run kept,
from `sea/worlds/cad.py`:
- **Gripper:** the SO-101's own CAD (its STL exports) for the servo, the fixed jaw and the moving
  finger, posed by the jaw part's joint in the compiled MuJoCo model, and the pads as the boxes
  `design/fingers.py` builds. The slide's rods and bushings are bought items the simulator does not
  model, so they are not drawn; the bill of materials lists them.
- **Mobile explorer:** CadQuery solids from the world's `_spec` (overall width, height, belly) and
  the catalogue's stated sizes (wheel, sprocket and idler; cells and motors at datasheet size).
- **Talking robot:** a section view of the cast tract at a posture: System 1's controls, or by default
  each articulator at the end of its travel. The squeezed bore follows `tract`'s radius profile. The
  tract hangs between the reed housing and a face plate at the teeth, on a base plate. Every servo
  stands on the plate, rides the tongue carriage or sits on the face plate's shelf, one MG90S block per
  servo the rig counts.

`sea/design/cad.py` renders models with VTK offscreen: orthographic, one camera per view fitted to
all of a run's bodies, so they share a scale. Each picture outlines what its change added or altered (`cad.changed`): a
part of a slot with no twin of the same name, place and size in the previous body.

The CAD sits outside the world modules on purpose. Those are judge components hashed whole into the
harness fingerprint, and a picture method there would have changed the judge key of every later run
without changing the ruler. Both CAD modules are listed in `harness.TOOLING`, the modules that read runs
but never shape one. System 2 never sees the pictures.

Tests: `tests/test_cad.py`. Every part is a closed shape in a slot of the body, and a swap changes its
slot's CAD. The slide's finger shifts along the closing axis without turning. Renders share one camera.
The script records each kept change with its reason.

## D-050 Films of each kept change, replayed from the run's records (report)
A picture shows what a body is; it does not show the body failing and then working. For each change a
run kept, `scripts/film_bodies.py` films the previous body failing and the new body passing, side by
side. It films up to three of the groups the change gained, distinct failure mechanisms first.

Nothing is searched again. Each side replays one train task with the controls the run recorded:
- before: System 1's best failing controls, from `failing` in the iteration record;
- after: the controls the kept trial passed with.

A replay must also reproduce everything else the verdict was judged on, which the controls alone do not
carry:
- **The run's seed.** The gripper's domain-randomization draw depends on it. Replayed under seed 0, a
  seed-2 run's grooved slide slips on the screwdriver it held.
- **System 1's fixed retry.** System 1 records `{}` both for its first attempt and for the retry that
  re-localizes the object by 1 mm. So for `{}` the film runs both and keeps the one System 1 kept: the
  retry if it passed or stopped structurally, otherwise the one with the better objective.

Each world has one filmer in `sea/worlds/film.py`:
- **Gripper:** the simulator itself. The pick is replayed in MuJoCo with a camera following the object.
  The verdict shown is the judge's on that replay, beside the run's diagnosis of the failure.
- **Talking robot:** the CAD (D-049) at the recorded posture, the formants the simulator computes for
  it, and for a vowel the sound. The sound is a pulse train at about 120 Hz, shaped by the simulator's
  own transfer function, the glottal source slope and the lips' radiation. A stop is silent. With
  sound, the two sides play in turn so each can be heard.
- **Mobile explorer:** an animation of the simulator's verdict, not physics. The world is quasi-static,
  so the CAD is posed over terrain built from the route's features. A small solver rests the body with
  its centre of mass lowest wherever its wheels or tracks touch. The robot stops at the feature where
  the replayed attempt failed, at the route's nominal dimensions, and the frame says why.

A replay is only honest if what it reads has not changed since the run. `film.READS` names the harness
components each world's replay depends on, and a run recorded when any of them differed is skipped. For
example, the gripper runs before D-048's friction change are not filmed. A run with no harness.json is
skipped too, because what it was judged by is unknown. The frozen mobile and phonology campaign (c1)
differs from today only in gripper components, so it is filmed.

The script also checks each film against the record: a film whose replay disagrees with the run (a
"before" that passes, or an "after" whose verdict differs) is not written. Before the seed and the retry
were replayed, this check caught 2 of 20 gripper films. Now every film agrees: 44 films across 10 runs.

The module is harness tooling (`harness.TOOLING`), like the CAD.

Tests: `tests/test_film.py`.
- The stock hinge on the tuna can, with the recorded controls, stalls as recorded.
- A seed-2 success replays under seed 2 and not under seed 0.
- A `{}` success that came from the retry replays as held.
- The stock tract's /ɔ/ posture misses as recorded (F1 +24%), and the evolved body's passes.
- The mobile film stops at the porch step for wheels and reaches the end with tracks.
- The voice's energy sits at the tract's resonance.
- Silent films play together, and sounding ones in turn.
- The script films a kept change from a run record, and skips a run with no harness.json.


## D-051 The mechanism audit: what breaks physical law in the kept bodies (tooling)
**Observed.** The films (D-050) show things no real robot can do:
- Every slide jaw System 2 kept floats free of the gripper.
- The jaws sink into objects they hold.
- The mobile robot drives through walls.

The user asked for a harness that mandates physical law: power transmitted from motor to motion, and proof that it all works. The plan has six phases:
- an Assembly per body (parts, joints with bearings or rails, motor → transmission → joint, power);
- laws checked before any trial;
- an audit inside every attempt;
- worlds rebuilt on the Assembly in the order gripper → mobile → talking robot;
- then every campaign re-run.

This entry is P0. It measures what the kept bodies break, before any law decides anything.

**Method.** `scripts/audit_mechanism.py` with `sea.mechanism` (`geometry.py`, `audit.py`):
- **Gripper geometry:**
  - The SO-101's own STL parts and the pads are posed at 9 jaw positions across the range. Geometry runs through coal (hpp-fcl) BVHs, with a winding-number test for a part sealed inside another. The CoACD hulls are never used: they overlap by 3.4 mm where the real meshes clear.
  - Per pair of parts on different bodies: the surface crossings, the size of their region, and its distance from the jaw axis.
  - Per moving body: its clearance to its parent through the range.
- **Gripper replays:** each film's replay (D-050), with the deepest penetration per kind of contact pair recorded every 20 ms tick.
- **Mobile and talking robot:** these simulators carry no geometry, so the audit reads the facts their own models give.

**Measured** (`results/mechanism_audit/`):
- **Stock hinge:** the jaw always touches its parent. Its overlap with the servo lies 1.6–10 mm from the jaw axis: the jaw's hub on the servo horn.
- **Every slide body (10 across 5 runs, D-042's `prismatic`):**
  - At the open end of the range the jaw is **70.6–84.5 mm from any material that could carry it**. No rail, rack or pinion exists; `_jaw_joint` only retypes the hinge and sets the actuator gear to 1/r.
  - When closed, it overlaps the servo body over about 22 × 39 × 20 mm, up to 39 mm from the axis (well past the horn), and overlaps the fixed jaw over up to 16 × 46 × 23 mm.
  - The pads pass through each other at full close.
- **Hinge-with-pads bodies (old physics):** at full close the pads pass into the fixed jaw and each other. The joint range lets the jaw close through them.
- **Contact:** in 16 of the 18 replays that held, the object sits more than 1 mm into the jaws at the 95th percentile, from 0.74 mm (sugar box) to 5.78 mm (screwdriver). Stock failures sink 0.7–5.5 mm. Soft contact is part of why these grasps hold.
- **Mobile:** voltage is not in the model.
  - Under series wiring the stock pack is 2.4 V and kept packs are 6–11.1 V, while every motor is simulated at the same rated stall torque and speed.
  - N20 stall current (0.75 A) is within the TB6612's 1.2 A per channel in every kept body.
- **Talking robot:**
  - No kept body places an articulator past the end of its 13.5 cm tract.
  - The tip tongue's tip and pad share 3.25–5.0 cm of travel, so both can occupy the same section.
  - Articulators remain free variables, with no mechanism.

**Decision.** Nothing in `sea.mechanism` decides a verdict yet, so it is tooling (`harness.TOOLING`) and the judge key is unchanged. It joins the judge as the component `mechanism` when its laws refuse bodies (P2).

These numbers are the baseline the later phases must remove:
- slide jaws carried by a rail, with the rack and pinion as real bodies (P1, P3);
- rigid-contact penetration of 1 mm or less at the 95th percentile (P1);
- voltage and driver ratings in the mobile model (P4);
- articulators driven through linkages (P5).

**Tests.** `tests/test_mechanism.py`:
- Overlap and clearance.
- A part sealed inside another is caught.
- The winding number.
- A floating part is found.
- Distance from an axis.
- The penetration record.
- The stock jaw rides its horn while a slide jaw floats more than 50 mm free.

## D-052 The force on the target is per robot body, as a vector sum (skill; read by the judge)
**Observed.** The class cap (`verify/verdicts.py`, rule 1) compares a force with the cap. That force was `peak_control_force`, the sum of the absolute normal force over every robot–target contact (`control/executor.py: _process_contacts`). In a grasp this double-counts:
- the two jaws push in opposite directions, so a squeeze counts once per jaw;
- a jaw meets the object in several convex pieces (CoACD, D-013), so it counts once per contact point.

V-005 recorded the same inflation for wedged tools. While working on D-054, stiffer contacts (the physical change it records) turned stock grasps into `rule_violation`. One banana pick read 222 N of "force" while each single contact carried at most 54 N.

**Change.** Per robot body, the target force is the magnitude of the vector sum of that body's contact forces on the target. The largest such body is the force the robot applies. Contact forces are rotated from the contact frame to the world, with their sign set by which geom is the robot's.
- For a single pushing tool, this equals the old sum when its contacts are aligned.
- For a squeeze, it is the force one jaw applies.
- The hard limit on every other contact (`total`) is unchanged.

**Prediction.** Under the current physics no verdict moves: the planted faults, their twins, the golden verdicts and the grasp tests stay as they are. The measure only matters once contacts are stiff (D-054).

**Measured.** Under the current physics everything holds: the 10 planted faults keep their declared verdicts (the knob xfail as before) and the twins pass; the golden verdicts are unchanged; the grasp and reflex tests pass.

## D-053 The class cap reads the squeeze the servo sustains, not an impact (skill; read by the judge)
**Observed.** D-052 made the target force the per-body vector sum of contact forces, taken at every 2 ms physics step. A closing jaw meets the object with the momentum of the servo's rotor: armature 0.028 kg·m², the identified value in the SO-101 model. How hard that momentum lands depends on the compliance between rotor and object, ω·√(k·J). With the contacts made rigid (D-054), the only spring is the finger, and a stock pick read 129–277 N for one to five steps. The squeeze those picks then held was 25–50 N.

The stiffness that decides the impact is mostly the gear train's, and the STS3215 has none published: its datasheet gives only a backlash of 0.5° or less, and no measured value was found. A verdict should not turn on a number nobody has measured.

**Change.** The class cap reads the force held on the target for a whole control tick: the largest, over the run, of the smallest force within any window of 10 physics steps (20 ms, one 50 Hz servo command). A shorter pulse is an impact. The servo does not command it, and the episode keeps it separately as `peak_ctl_impact` for the mechanism audit's laws (P2). The hard limit on every other contact is unchanged. `no_actuation` now needs force held for a tick, so a graze is not actuation.

**Prediction.** Under the committed physics (soft contacts), a contact absorbs an impact over about 20 ms, so the sustained force and the per-step peak differ by a few newtons. No verdict moves: the planted faults keep their declared verdicts, the golden verdicts and the B-battery are unchanged, and the grasp tests pass.

**Measured.** Under the committed physics, the suite passes: 609 tests, including the 10 planted faults with their declared verdicts (the knob xfail as before), their twins, the golden verdicts and the grasp and reflex tests. No battery class moved beyond its interval.

One verdict did move, against the prediction. The validation slider `val_p011 c1` went from `rule_violation` to `pass`, which moves the val sliders from 0.30 to 0.40:
- its push peaked at 11.3 N, over the slider's 10 N cap, for less than a tick;
- the force it held for a tick was 9.0 N.

A soft contact's overshoot had decided that verdict, and an overshoot is not a push the robot commanded. The audit's impact law (P2) is where such a pulse will be judged, against the energy it carries rather than an unmeasured stiffness.

## D-054 Grasp contacts are rigid, and the finger bends on its own joint (world: physics)
**Observed.** The mechanism audit (D-051) found held objects sunk into the jaws: in 16 of 18 held grasps the 95th percentile of penetration was above 1 mm, with peaks of 7.7–10.2 mm. MuJoCo's contact is a spring per unit of effective mass that relaxes over the time constant `solref[0]` (20 ms by default). So how deep an object sinks depends on its mass, not its material. On the 21 stock picks at the default:

| Object | Mass | Sinks into the jaws (p95) |
|---|---|---|
| Die | 5 g | 8.5 mm |
| Cup | 13 g | 2.9 mm |
| Sugar box | 514 g | 0.3 mm |

Objects set down on the table sank 5–10 mm before the contact caught them.

Three fixes failed on the way, each for a reason:
- **One calibrated compliance** (impedance 0.97 to 0.99 at 20 ms, about 3e4 N/m on the 50 g probe): light objects still sank 2–8 mm.
- **A time constant scaled by the square root of the object's mass:**
  - Light objects sank 1.3 mm or less, but heavy ones sank 3 mm, because the arm adds about 1.9 kg⁻¹ to the effective inverse mass at a jaw (read from `efc_diagA`).
  - A slow contact also let a 370 g can sink 13 mm into the table on landing.
- **Rigid contacts alone** (time constant two timesteps): every object stayed under 0.8 mm. But the closing jaw now stopped dead within 2 ms, and a stock pick read 211–277 N on its target. The compliance the old contact had lent the jaw has to live somewhere physical.

**Change.**
- **Contacts.** In grasp scenes every contact has a time constant of two timesteps (4 ms), the stiffest MuJoCo integrates stably, with impedance 0.97 to 0.99 over 1 mm (`GRASP_SOLREF`, `GRASP_SOLIMP` in `sim/build_model.py`).
- **The finger bends on its own joint.** `jaw_flex` sits on the gripper joint's axis, in the same body, so the rotor drives `gripper` and the finger deflects on `jaw_flex`. The servo's encoder still reads `gripper`.
  - Its stiffness is the stock jaw's bending stiffness in the closing direction: 1.1e3 N·m/rad about the hinge, or 2.2e5 N/m at a load 70 mm out. It is integrated over the jaw mesh's sections with solid PLA (E 3.5 GPa). The section's second moment runs from 2e4 mm⁴ at the root to 90 mm⁴ at the tip.
  - The gear train, which has no published stiffness, is rigid, so this is the stiff bound on an impact.
  - The rebound is taken as lost in the gear train's tooth friction and backlash (damping ratio 1). The servo's damping was identified in free motion and does not include that loss.
  - The joint's inertia, k·dt², keeps its mode at one radian per timestep.
- **The gold.** It is re-recorded with `scripts/measure_grasp.py --write`, with cases now past the envelope: cans to 90 mm, boxes to 80 mm, spheres to 90 mm. `STOCK_WIDEST_MM` (`design/grip.py`) follows it.

**Prediction** (from the rigid-contact measurements, before the flex joint):
- Every object's p95 penetration stays under 1 mm.
- The closing impact reaches the stiff bound, ω·√(kJ) = 1.8 rad/s × √(1100 × 0.028) ≈ 10 N·m, or about 140 N at 70 mm. It lasts less than a tick, so D-053's measure reads the squeeze.
- The stock envelope does not shrink.

**Measured.**
- **Penetration, on the 21 stock picks** (one replay each, with System 1's fixed retry). Under the default, p95 into the jaws exceeded 1 mm on 12 of 21 picks, worst 8.5 mm, and objects sank up to 9.9 mm into the table. Now:
  - into the jaws: p95 at most 0.77 mm (the die), median 0.16 mm;
  - into the table: at most 0.48 mm.
  - 14 of the 21 replays pass, against 13 before.
- **The closing impact** on a phase A can is 129 N for about 10 ms, then a sustained squeeze of 25–33 N. No bad-state warnings appear.
- **The stock envelope widens:**

  | | Before | Now |
  |---|---|---|
  | Widest can | 55 mm | 70 mm |
  | Widest box | 60 mm | 70 mm |
  | Spheres, tips 12 mm past the equator | to 50 mm | to 80 mm |
  | Spheres, tips 5 mm past the equator | none | to 80 mm |

  The 80 and 90 mm cans, the 80 mm box and the 90 mm sphere lift 0 mm. Unchanged: the heaviest 30 mm can held (800 g; 1200 g lifts 2.7 mm), the grip current (2.691 A), and the median squeeze (43.3 → 43.4 N).
- **Tilted pads** (40°, 3 mm groove) hold cans and boxes to 85 mm and spheres to 75 mm. That is 15 mm wider than the stock on cans and boxes, and 5 mm narrower on spheres; why is not yet explained.

**Reading.**
- **Old failures were artifacts.** The old stock failures (cans past 55 mm, balls near the equator) came from the simulator. A soft contact let them sink and turn out of the jaws.
- **Phase B no longer escalates.** Phase B's objects (D-048: cans of 60 and 65 mm, a 70 mm box, balls of 30 and 40 mm) are now inside the envelope, and `tests/test_grasp_world.py` records that the stock picks the 60 mm can and the 40 mm ball. Re-deriving phase B is a change to the judge and gets its own entry (D-055).
- **Test cases re-pinned.** Cases near the old threshold were draw-dependent and are now pinned:
  - the lost-grip reflex: a 50 mm, 800 g can at friction 0.2 lifts 10 mm and slips;
  - the lift that falls back: a 60 mm, 800 g can at friction 0.25;
  - the seed replay compares tick 30, because the perception draw moves the approach from about tick 15.
  - The recorded tuna-can failure now replays as `no_actuation`, because the jaw only grazes the 85 mm can.
- **Open items.**
  - A load–deflection test on a real STS3215 horn would replace the rigid gear train.
  - The gold's payload cases (a 30 mm can at 800–1200 g) are denser than lead, which the density law in P2 will refuse.
  - Films and campaign results were judged under the old physics. `film.READS` marks the films stale, and P6 re-runs the campaigns.

## D-055 Phase B re-derived on rigid contacts, with objects of real density (judge: world_grasp)
**Observed.** Phase B (`configs/grasp.yaml`) holds objects that each fail the stock gripper for one known reason. Its five were chosen on soft contacts (D-048): cans of 60 and 65 mm, a 70 mm box, and balls of 30 and 40 mm. On D-054's physics the stock envelope reaches 70 mm for cans and boxes and 80 mm for spheres. The grasp battery's phase B then passed 3 of 5: the 60 mm can and both balls. The 65 mm can and the 70 mm box still failed with System 1's search. Train and val passed 16/16 each, as before.

**Change.** Phase B is four objects, each checked with a pick forced past the aperture rule:

| Object | Size | Mass | Density | Forced pick | Reason |
|---|---|---|---|---|---|
| `wide_can_80` | 80 mm across, 80 tall | 150 g | 0.37 g/cm³ | lifts 0 mm | wider than the stable opening |
| `wide_can_90` | 90 mm across, 85 tall | 200 g | 0.37 g/cm³ | lifts 0 mm | wider than the stable opening |
| `wide_box_80` | 80 mm across, 60 tall | 150 g | 1.0 g/cm³ | lifts 1 mm | wider than the stable opening |
| `weight_1kg` | 48 mm across, 70 tall | 1000 g | 7.9 g/cm³ | lifts 5 mm, arm stalls | heavier than the stock arm lifts |

- The three wide objects' tips close on air, and the aperture rule refuses them before simulation.
- `weight_1kg` is a steel calibration weight. The jaws hold it at a 28 N squeeze while the arm stalls (`load_limit`), which the L3 arm-servo option addresses.
- Spheres are gone. No ball under the 85 mm height limit fails the stock jaws. A 90 mm ball is refused by the aperture rule (the V1 probe's fixed pick fails at 90 mm), yet System 1's pick lifted it 40 mm and held it. The aperture rule is conservative there, and P2 revisits it.

**Prediction.** The stock gets `no_applicable_primitive` on the three wide objects and `stuck` (`load_limit`) on the weight: phase B passes 0/4. Train and val are unchanged.

**Measured.**
- **Grasp battery:**
  - train and val: 16/16 each;
  - phase B: 0/4, with 3 `no_applicable_primitive` and 1 `stuck`.
- **Baselines.** Both baselines are re-saved under the new judge key (061a17f63ee6). The panel baseline changes only in its key. Its val sliders read 0.40, the value recorded before D-052: D-052's force measure had moved `val_p011 c1` into `rule_violation` (0.30), and D-053 moved it back.
- **Not yet re-measured.** The real-object phase split (`scripts/measure_real.py`, `docs/gripper-experiment.md`) was measured on soft contacts. It needs System 1's full search on the 21 YCB objects at three placements, a longer run that waits for a go-ahead.

## D-056 The servo horn rides the jaw (world: physics)
**Observed.** The SO-101 model draws the STS3215's horn as part of the gripper servo's mesh. The stock jaw is screwed to that horn and turns about it, so in the mechanism audit (D-051) the jaw cut through the horn at all 9 jaw positions: a disc 20 mm across and 3.5 mm thick around the hinge axis. Every film showed a finger turning through a part that should turn with it.

**Change.**
- **The servo.** The gripper body carries the model's own horn-less servo mesh (`sts3215_03a_no_horn_v1`).
- **The horn.** The horn rides the moving jaw. It is extracted as the faces of the servo mesh that the horn-less mesh lacks: 1,908 faces, 20 × 20 × 3.9 mm, coaxial with the hinge. It is drawn only (`gripper_horn`, no contact).
- **The slide's horn stays put.** On a slide the horn turns a pinion, which the simulator does not have yet.
- **Collision.** The servo's collision hull shrinks from 40.16 to 39.81 cm³.

**Prediction.** The stock grasps do not move, and the stock's remaining overlaps with the jaw are its bearing seats at the axis.

**Measured.**
- The gold does not move: 0 of 46 picks changed (`scripts/measure_grasp.py`).
- On the stock, the horn now meets the servo only at the spline (a 6 mm region within 3 mm of the axis).
- The jaw meets the servo only at the rear hub, 3 mm deep and within 10 mm of the axis, the same at every angle. That is where the jaw's back plate bears on the servo's rear boss, the contact the plan's interference law allows across a revolute joint.

## D-057 The parallelogram jaw is a real four-bar (world: physics)
**Observed.** D-042's parallelogram was an ideal mechanism. A coupler body was held at minus the hinge angle by a joint equality, with no links, pins or follower, and the whole stock finger mesh, hub included, was drawn on it. In the films a finger translated with nothing holding it. System 2's kept parallelogram bodies use long links: 15 of the 31 recorded `link_mm` values are 89.1 mm, and none is under 72 mm.

**Change.** `sim/jaws.py` builds the jaw from parts (docs/harness.md, invariant 11). Every solid is drawn and takes part in contact. Masses come from volume at 1,240 kg/m³ (PLA).
- **Driver.**
  - Two 4 mm PLA plates, 9 mm wide with round ends, lie in the stock finger's own wall planes: one on the horn's face, one on the rear hub.
  - A steel pin through their lower ends ties them together.
  - 5.6–9.2 g.
- **Follower.**
  - The same pair of plates, on a fixed pin held by a printed bracket on the servo's end.
  - The pin sits 22 mm from the hinge at 45° up from the closing axis, so the transmission angle stays at or above 35° over the jaw's range.
- **Coupler.**
  - It carries the stock finger's blade: the jaw below z = −55 mm in the gripper frame, where the finger is under 13 mm thick. The blade weighs 10.6 g.
  - Two hubs span between the plates, joined by a web.
  - When the links outreach the finger, a strut carries the blade from the lower pin. This applies when the pin's hub would overlap the blade by less than 1 mm, which happens above about 78 mm.
  - 23–25 g.
- **Closure.**
  - A connect constraint joins the follower's lower pin to the coupler.
  - The finger stays upright because the links are parallel; no constraint holds its angle.
  - `Sim.set_arm_q` poses the dependent joints (`Sim.linked`).
  - Excludes apply only between bodies pinned to each other.
- **Drawing.** `worlds/cad.py` draws the follower, the pins and the bracket. `jaws.py` joins the `physics` component.
- **Unchanged.** The finger's kinematics: it moves on a circle of radius `link_mm` and keeps its orientation, so `design/fingers.py` does not change.

**Prediction.** This was made before the picks were run and is entered here with their results.
- The verdicts are the same as the ideal linkage's.
- The jaw angle moves by no more than 0.02 rad.
- Squeeze changes by no more than 25%.
- The reason: the finger's path is unchanged, and only the path the force takes through the mechanism changes.

**Measured.**
- **Geometry audit** (`scripts/audit_mechanism.py`, link 50, 70, 80 and 90 mm, 9 jaw angles):
  - Every moving body is carried, with gaps of 0.3 mm or less (the pin clearances).
  - The only material shared across a joint is the servo's spline in the horn and driver plate (within 3 mm of the axis, 0.3 mm deep) and its rear pin (within 2.6 mm, 0.6 mm deep). These are the bearing seats.
  - The first build put the plates inside the servo and the blade. Their planes had been measured from the servo's mid-plane rather than the jaw frame's origin, which sits 18.8 mm away along the axis. The audit caught it.
- **Loop closure** during held picks: 0.009–0.061 mm (the plan's limit is 0.1 mm).
- **Picks against the ideal linkage at HEAD:** 12 picks (links of 50, 70 and 90 mm, each on a 30 mm can at 50 g and 400 g, a 50 mm can and a 40 mm box).
  - All held, as before.
  - The jaw angle moved by 0.013 rad or less.
  - Squeeze changed by −8% to +23%. The largest change was at 90 mm on the 30 mm can, from 36.8 to 45.3 N.
- **The gold** has no parallelogram picks, so it does not move.

**Consequence, open (judge tier, P3).** The finger stays where the stock puts it, so links longer than about 78 mm hang the coupler's pins below the fingertip. At System 2's 89 mm, the lower pin is 7 mm below the tip and its hub reaches 13 mm below. The pins meet the table or the object first. Two judge-tier options remain for P3: make the finger's mount on the coupler a part parameter, or narrow `link_mm`. Until then, the simulator builds the long links as they are, with this cost.

## D-058 The slide jaw is a real rack and pinion on two rods (world: physics)
**Observed.**
- **The old slide was an ideal joint.** D-042's slide turned the gripper hinge itself into a slide joint and set the actuator gear to 1/r. It had no pinion, rack, rods or carriage: the stock finger's whole mesh, hub included, slid along x.
- **Its catalogue entry names real parts.** The entry reads "a printed rack and pinion on two steel rods", and the bill of materials lists two 3 × 100 mm steel rods and two LM3UU bushings.
- **System 2 always took a large pinion.** Every recorded slide body uses 52–60 mm/rad (134 records), a pinion 104–120 mm across.

**Change.** `sim/jaws.slide` builds the mechanism from those parts.
- **The gripper joint stays the servo's output**, a hinge in radians, for every jaw. `Sim.grip_scale` is gone.
- **Pinion.**
  - A printed spur gear on a hub screwed to the horn, 6 mm thick.
  - The hub reaches through the frame's cut-out around the horn (clear to 12 mm), so the gear runs outboard of the frame at y = −28.5 to −34.5 mm.
  - Whole teeth at about module 1: the tooth count nearest 2r/1 mm, with the module that makes the pitch radius exactly `drive_mm_per_rad`. Backlash is 0.6 mm.
- **Rack.**
  - In the pinion's plane, with its pitch line on the pinion's pitch circle.
  - Long enough to stay in mesh over the whole travel, with a tooth centred under the pinion at q = 0.
- **Carriage.**
  - A printed block under the rack with a square bore for each rod (the bushings: 3 mm bore, 0.1 mm clearance).
  - An arm across to the stock finger's blade (the same blade as D-057).
  - Its frame is the jaw's at q = 0, so the blade and pads sit on it as on the stock jaw. `fingers.moving_body` names it. The finger's kinematics in `design/fingers.py` do not change: x moves by r·q.
- **Rods.**
  - Two 3 × 100 mm steel rods on the gripper. Their front end is 2 mm past the carriage fully open.
  - A printed clamp holds them at their back end. A bracket runs from the clamp to the frame's horn side, crossing the gear plane behind the pinion.
- **The teeth are a joint equality**, slide = r·q. Tooth contact itself is too fine for a 2 ms step. Its impedance (0.999–0.9999) puts the mesh compliance near printed teeth: about 0.07 mm at 110 N, against an estimated 0.1 mm at 160 N for PLA teeth on a 6 mm face.
- **Flex.** The finger's flex spring (D-054) moves from the gripper joint to the carriage, in series with the slide.
- **Masses** come from volume (PLA 1,240 kg/m³, steel 7,850 kg/m³) plus the bushings' datasheet mass.
  - Parts fixed to the gripper join its mass, because the SO-101's gripper inertial ignores geom masses: rods and clamp here, and D-057's bracket and pin hub.
  - A 30 mm/rad pinion weighs 24 g and the carriage 28 g.

**Prediction.**
- **Picks.** The same verdicts and the same squeeze within 5%. The finger's path and the force ratio τ/r are unchanged; only the reflected inertia (now the carriage's real mass through the pinion) and the teeth's compliance are new.
- **Geometry.** The catalogue's 100 mm rods cannot hold System 2's slides: travel 1.9 r against 100 mm of rod less the carriage and clamp.

**Measured.**
- **Picks against the ideal slide at HEAD.** 15 picks: 20, 30 and 40 mm/rad, each on 30 mm cans at 50 g and 400 g, a 50 mm can, a 40 mm box and a 70 mm can.
  - Every verdict is the same.
  - Squeeze is within 4%.
  - Rack-to-pinion error is 0.05–0.10 mm under 70–160 N.
- **Geometry audit** (9 jaw angles).
  - The audit now also checks the jaw's parts against the wrist over the wrist's whole roll range (13 rolls). A film found this first: a 60 mm/rad pinion jammed against the wrist at 161 N.
  - **20 mm/rad:**
    - The carriage is carried on its rods at every angle, with 0.1 mm gaps (the bushing clearance).
    - The only material shared across a joint is the servo's spline in the horn and hub (within 3 mm of the axis).
    - After the tooth phase and backlash were set, the teeth mesh with no interference.
  - **25 and 30 mm/rad:** clean except that the pinion meets the wrist's roll housing when the wrist rolls past 2.4 rad (both) or −2.7 rad (30 mm/rad).
  - **40 mm/rad:**
    - The carriage runs into the rods' clamp when closed.
    - The bracket's path past the pinion is gone.
  - **b021's 56 mm/rad:** all three failures, with a pinion of 112 mm and 76 g. This prediction held.
- **The real envelope** is about 22 mm/rad: a pinion about 44 mm across, about 42 mm of travel, and a widest opening of about 54 mm. The wrist sets it. The rods would allow about 35 mm/rad.
- **One film test changed case.** `test_film`'s retry case was gripper_s2v3_s2's 60 mm/rad slide holding the mustard bottle. It held only on the ideal slide: on the real one, its 120 mm pinion jams against the wrist. The test now uses the stock body on the banana, which slips on the first attempt and holds on System 1's retry.

**Consequence, open (judge tier, P2/P3).**
- Every slide body System 2 kept (52–60 mm/rad) is a mechanism the simulator now draws as unbuildable. Its successes so far rest on parts that cannot be assembled.
- Turning these audit findings into refusals before trial is the mechanism law of P2.
- Making the catalogue mechanical (module, teeth and a stocked rod length, with the servo's full turn as an option only if its datasheet supports it) is P3.

## D-059 The gripper's mechanism laws refuse a body before trial; the slide's part is its pinion's teeth (judge: world_gripper, mechanism)
**Observed.** D-057 and D-058 build every jaw from its parts, and the audit shows what cannot be assembled. Every slide System 2 kept is one of these. But nothing refused them: `GripperWorld.check` ran only the build rules, so an unbuildable body was still trialled and could still be kept.

**Change.**
- **`sea.mechanism.gripper` holds the laws.** It is a new judge component, `mechanism`, together with `mechanism/geometry.py`. `check` returns the build rules' errors plus the laws' violations, and the violations reach System 2 through the existing "refused before trial" channel. The laws read the simulator's own build on the real meshes, at 9 jaw angles, and for the wrist at 13 roll angles.
  - **Carried:** every moving body stays within 0.5 mm of the body that carries it (a pin's or a bushing's clearance).
  - **No interference:** no two parts share material, except where the hinge body bears on the servo within 10 mm of the hinge axis (the stock's own rear-hub seat, D-056). A shared region thinner than 0.1 mm is two faces in contact, not interference.
  - **Transmission:** a four-bar's links stay at least 30° from toggle.
  - **How violations are reported:** each names the parts, the size of the shared region and the jaw and roll angles, merged per pair of named parts.
- **The audit is faster.** The script's geometry moves into the module. When two surfaces do not cross, one part is either wholly inside the other or wholly outside, so 9 vertices decide containment instead of 400. That cut a law check from about 30 s to 4–9 s. Results are cached per body.
- **The slide's catalogue parameter is mechanical.** `drive_mm_per_rad` (20–60 mm/rad) becomes `pinion_teeth`: 17–60 whole teeth on the module-1 pinion that `sim/jaws.py` builds, so the pitch radius is teeth/2 mm. Seventeen is the fewest teeth a 20° gear cuts without undercut.
- **Not a law:** tooth bending. No measured allowable stress exists for printed PLA gear teeth. A Lewis check at the servo's stall torque gives about 35 MPa at the root of a 40-tooth, 6 mm-wide tooth, against printed PLA's 40–60 MPa tensile strength. Deferred with beam stress (plan, "Deferred").

**Prediction** (from D-058's measured envelope, before the laws ran):
- The stock and every four-bar (50–90 mm) pass.
- Slides up to about 42 teeth (22 mm/rad) pass.
- From about 44 teeth the pinion meets the rolled wrist.
- b021's 56 mm/rad (112 teeth, outside the new range) breaks interference (clamp and carriage, bracket and pinion, pinion and wrist) and carried.

**Measured.**
- The stock and four-bars at 50, 89 and 90 mm: no violations.
- Slides at 17 and 40 teeth: none.
- 44 and 46 teeth: one, the wrist and the pinion share material when the wrist rolls past +2.38 rad.
- 50 and 60 teeth: also the wrist's servo.
- 112 teeth: the pinion with the wrist and the wrist servo; the clamp bracket with the pinion and its hub; the clamp with the carriage; and the carriage carried 0.9 mm from the gripper when closed.
- The prediction held.
- Tests: `test_mechanism.py`, with each refusal paired with a passing twin and a planted toggle for the transmission law, and `test_world_gripper.py`.

**Consequence.** The gripper campaign's slide bodies cannot be proposed again as they were, and runs judged before this key cannot be pooled with runs after it (invariant 10). Re-running the gripper campaign under this ruler is the next step, after asking.

## D-060 The fingers stop on each other (world: physics)
**Observed.** The first iteration of the D-059 gripper campaign refused System 2's pads (32° and 45°) because the moving pad met the gripper frame at the closed end of the jaw's range. Checking the simulator showed a deeper problem. The gripper and its jaw are parent and child in MuJoCo, and on a four-bar the gripper and coupler are excluded. MuJoCo drops contacts between those pairs, so a jaw closing on nothing went through the other finger to its joint limit. With 32° pads, that meant 12.7 mm through it. A film of an empty close would show it.

**Change.** Explicit contact pairs join the fixed-finger geoms (the `jaw_fixed` parts and pads) to the moving-finger geoms (the `jaw_moving` parts and pads) that can meet. Pairs bypass the parent filter. A pair is added when its two geoms come within 1 mm of each other over the jaw's range, widened by 0.05 rad at each end for a jaw pressed past its soft limit. The check runs once per finger spec: bounding spheres first, then the hull distance at 256 jaw positions (`build_model._stops`). The friction is the two faces' combined friction, and the contacts are rigid like grasp contacts (D-054). The pairs are robot-robot contacts, so the jaw sensors and the force limit ignore them, as before.

**Prediction.**
- A padded jaw closing on nothing stops where the faces meet.
- The stock's fingers never meet: they stay 1.7 mm apart at the limit. So the gold does not move.
- Grasps do not change, because the fingers meet only with nothing between them.

**Measured.**
- Closing on nothing:
  - Stock: the limit, q = −0.17, with a 1.7 mm gap.
  - 32° pads: stops at q = −0.049, gap 0.0 mm.
  - Four-bar (70 mm) with pads: stops at −0.083, gap 0.0 mm.
  - 20 mm/rad slide with pads: the limit, with a 2.4 mm gap.
- Pairs added:
  - stock: 2;
  - 32° and 45° pads: 5;
  - four-bars: 3–5;
  - 20 mm/rad slides: 0.
- Cost of a stock pick (median of 5):
  - all 256 fixed×moving pairs: 0.203 s;
  - no finger pairs: 0.164 s;
  - the 2 that can meet: 0.176 s.
- The pairs are in every model, including the panel world's, since a padded jaw stops on its other finger there too. The stock fingers never meet, so the golden verdicts hold (`tests/golden/` passes unchanged).
- The stock pick is unchanged.
- Test: `test_mechanism.py::test_a_jaw_closing_on_nothing_stops_on_the_other_finger`.

## D-061 The jaw closing on its other finger is its stop, not interference (judge: mechanism)
**Observed.** D-059's interference law swept the jaw over its whole joint range. A finger that meets the other only below the angle where the faces touch was refused, even though the jaw stops there (D-060). Three of System 2's first three proposals were refused this way: pads at 32° and 45°, 12 and 20 mm wide.

**Change.** A shared region between a moving finger part (the stock jaw, the blade, the moving pads) and a fixed one (the gripper frame, the fixed pads) is the jaw's stop when it occurs only over the closed end of the swept range: the first jaw positions, with no gap. The same parts meeting with the jaw further open are still interference.

**Prediction.** The three refused pad proposals pass. Nothing that passed before is refused. The other laws do not change.

**Measured.**
- The three proposals now pass. Their only finger contacts were at q = −0.175, the first sample.
- Padded four-bars and slides pass.
- Test: `test_mechanism.py::test_the_fingers_meeting_as_the_jaw_closes_is_its_stop_not_interference`, which also checks the open-jaw twin.
- The aborted campaign start (seed 0, iteration 0, three refusals) was set aside and the campaign restarted under this key.

## D-062 The pack's voltage drives the motors (judge: world_mobile)
**Observed.** The mobile world had no voltage. The stock pack, two AA NiMH cells (2.4 V), drove 6 V N20 motors at their rated torque, and seed 2's three Li-ion cells (11.1 V) drove the same 6 V motors at the same torque (audit, D-051).

**Change.**
- A pack's cells are in series. Nominal 1.2 V per NiMH cell and 3.7 V per Li-ion cell; 1.4 V and 4.2 V charged.
- The motor table holds at the winding's rated voltage: 6 V for the N20s, 12 V for the JGA25. A DC motor's stall torque and no-load speed are both proportional to its supply, so the drive's stall force and no-load speed scale by pack volts / rated volts. Stall current scales the same way in the pack-current check.
- A new check: the pack must give at least half the winding's rated voltage, and at most 1.5 times it when charged. The 1.5 is a hobby rule of thumb, not a datasheet limit (a thin spot): above it, windings and brushes wear fast.
- The stock pack becomes four NiMH cells (9.6 Wh, 4.8 V), because two cells (2.4 V) fall below a 6 V winding's 3 V floor.
- The catalogue's docs and the brief state the voltages.

**Prediction.**
- The old stock body is refused (2.4 V nominal, under 3 V). The new stock pushes 0.8 times the old force at 0.8 times the speed and carries 54 g more. On the analytic ruler it holds 3 or 4 of the 10 train routes, against 4 before; the 5.5 cm porch step and the lawn are still out of reach.
- Seed 0's final body (five NiMH cells, 6.0 V) is unchanged. Seed 1's (six cells, 7.2 V) is 1.2 times stronger. Seed 2's (three Li-ion cells, 11.1 V, 12.6 V charged) is refused.

**Measured.**
- The old stock body is refused: "2 nimh_pack cells in series give 2.4 V (2.8 V charged); the 6 V n20_gearmotor winding takes 3.0-9.0 V".
- The new stock: 4.8 V, 12.8 N of stall force, 0.75 m/s at no load, 0.43 kg. It holds **5** train routes, not the predicted 3–4: kitchen, living room, under the sofa, the shed, and now the lawn. The miss is the energy. Four cells double the pack (9.6 Wh), and that, not the force, decides the 5 km survey.
- Seed 0's final body is unchanged (13.3 N). Seed 1's pushes 16.0 N. Seed 2's is refused, as predicted.
- A side effect: at 4.8 V, six rocker-bogie motors draw 3.6 A at stall, under the NiMH pack's 4 A, so that check no longer refuses the rocker-bogie on the stock pack. At 7.2 V they draw 5.4 A and are refused.
- Tests: `test_world_mobile.py::test_the_pack_voltage_drives_the_motors_within_the_winding_range`; three tests updated for the new stock body.

## D-063 Steps, trenches, ramps and openings are driven in MuJoCo (judge: world_mobile)
**Observed.** The mobile world walked every obstacle with closed-form vehicle mechanics. Its films were animations of the verdict. When a robot clipped a door jamb, the film drew the jamb through the robot (the user saw it), and the tracks floated beside the hull. The world's results could not be called physics.

**Change.**
- `src/sea/sim/mobile_model.py` (component `world_mobile`) builds the body in MuJoCo from the world's numbers in the CAD's layout:
  - wheels on driven hinges;
  - the rocker-bogie as real rockers and bogies on pivots, with the two rockers tied by a differential (q_left = −q_right);
  - tracks as a dense row of rollers along the bottom run and up the sloped front run, each coupled to the drive sprocket at equal surface speed by a stiff joint equality (the belt is drawn only).
- Each driven joint has a general actuator on the DC motor's torque-speed line at the pack's voltage (D-062), clamped at stall, with the rotor's inertia reflected through the gearbox. Rolling resistance is the world's C_rr as joint friction. Contacts are stiff (4 ms on a 1 ms step) with elliptic cones.
- Every terrain edge is rounded 3 mm. A sharp MuJoCo box edge takes its contact normal from one face: a track met a step's corner with the normal pointing straight back, and was pushed off it.
- An episode starts the robot 2 cm short of the obstacle, measured from its real extent, at the speed its motors hold on the flat at that duty. It ends when the rear is past, it is stuck (under 5 mm in 0.8 s), it tips past 70°, or time runs out.
- What MuJoCo decides:
  - **step and trench**, at the approach duty: past; the belly on the edge (`belly_hits_step_edge`); stalled below full duty (`stalled_at_step`, control-level) or at it (`motor_torque_short_on_step`); slipping (`step_higher_than_climbable`, or `gap_longer_than_support` in a trench); thrown back (`bounced_off_step`); tipped over (`tipped_over`, new);
  - **ramp**, at the cruise duty: slipping (`slips_on_ramp`) or stalled;
  - **opening**: the robot starts at its line's error and drives straight at two jambs, and a robot that touches one is stopped by it (`clipped_gap_edge`). A chassis wider than the opening is still refused geometrically.
- Search sees the same physics. Each episode's arguments are quantised (duty 0.05, friction 0.03, heights 1 mm, line 2.5 mm) and cached, so attempts that differ by less share one episode.
- The edge a body climbs (in the brief, and for climbing out of its own rut in sand) is measured in MuJoCo at full duty by bisection.
- Sand (Winkler), the low overhang, the dark corridor and energy stay as they were: MuJoCo's soft contact is not sinkage, and the other three are geometry and bookkeeping.

**Prediction.**
- The stock body still holds kitchen, living room, under the sofa, the shed and the lawn, and still fails the porch on the step. MuJoCo lets its 3 cm wheels climb more than the closed form's 1.97 cm (about 2.5 cm on 0.7 friction), which is still short of 5.5 cm.
- Seeds 0 and 1's track bodies (idlers 7.0 and 6.5 cm) still hold all 10 routes, but only near full approach duty on the porch. In the prototype a 6.5 cm idler climbed 5.5 cm at duty 1.0 and stalled at 0.5.
- A solve takes seconds instead of milliseconds. A campaign seed runs 20–60 minutes instead of 2.

**Measured.**
- **Stock:** the same 5 routes, solved in 2.6 s, against milliseconds before. Its failures are now physical. On the porch it reaches the 5.5 cm edge at 0.89 duty, hops it, and grounds its 2 cm belly on the edge. In the garden it drops into the trench and grounds there. Both were `step_higher_than_climbable` and `gap_longer_than_support`.
- **Seeds 0–2's track bodies:** 10/10 each, in 12–24 s. The prediction that the porch needs near-full duty was wrong: once the edges were rounded (below), a 6.5 cm idler climbs 5.5 cm at 0.4 duty and stalls at 0.2–0.3.
- **Climb limits come from dynamics as well as statics.** The closed form allowed a 3 cm wheel 1.97 cm on a 0.7 edge; MuJoCo gives about 3.6 cm from a run-up. Two effects the closed form did not have:
  - an impact's friction impulse lifts a wheel that meets an edge at speed (μ v up, here about 1 m/s);
  - four N20s at stall from rest put about 0.38 N·m on a 0.43 kg chassis. That equals its weight's moment about the rear axle, so the nose lifts about 3 cm (a wheelie), and the front wheels land on top of an edge they could not roll up.

  With near-zero friction, no wheel climbs any edge from rest (checked), so the lift is not a contact artifact.
- **Found while building, not predicted: three contact artifacts, each fixed before measuring.**
  - The motor rotor's inertia on all 26 coupled rollers of a track made it about 50 times harder to spin than the robot was to push. Only the sprocket now carries J × ratio².
  - Measured from the tub, a track's front run started 2.7 cm inside the step and was launched backwards. Placement now uses the compiled model's real extent.
  - A sharp box corner took its contact normal from one face: a wheel met a step corner with a normal of (0.41, −0.91) instead of the true (−0.94, 0.33), and was lifted by it. Every edge is now two boxes tangent to a 3 mm rounded nosing.
  - Under load, the track couplings slip at most 0.64 mm of belt over a climb.
- **What parameters alone now reach** (`test_world_mobile.py::test_stock_parts_reach_the_porch_garden_and_sandbox_by_parameters_but_never_the_dark_hallway`):
  - 3.67 cm wheels at 50:1 with a 4 cm belly hop the porch and pop out of the sandbox rut;
  - a tub with a 1 cm belly slides over the garden trench on its belly;
  - no setting of the stock parts holds the dark hallway, which needs a sensor.

  The closed form said four routes needed new parts; physics says one. The redesigns System 2 found (tracks, a sonar) remain one way through. The mobile campaign is re-run under this key.
- **Mecanum:** its wheels are cylinders with the world's reduced friction, not rollers. A thin spot under invariant 11, and the reason the mecanum distractor now matches the stock wheels instead of losing to them.
- **Tests:** `test_world_mobile.py` rewritten from the closed-form rules to these outcomes (the climb, the stalled track, the trench, the distractors, the stock's mechanisms).
- **Amendment, found by filming the replays: the ramp lay flat.** Its tilt went into the MJCF as radians, but MuJoCo reads `euler` in degrees. So a 22° ramp was built 0.38° off flat, hovering 19 cm up, and every robot drove under it. Every ramp verdict above was wrong. Fixed. Re-measured: stock 5/10 and the three track bodies 10/10, the same outcomes, now climbing the face: each robot's centre rides 5 cm above the slope (`test_world_mobile.py::test_a_ramp_is_climbed_on_its_surface_at_its_slope`). Lesson: a judge built on physics needs a picture of every episode type before any of its numbers are believed.
- **Campaign under this key** (`results/evolve/mobile_s2mj_s{0,1,2}.json`; 3 minutes a seed, as the cached physics is shared across search):
  - every seed went from the stock 5 to 10/10 train routes in two iterations, and 10/10 test against stock 5;
  - every seed chose a sonar and tracks with a 6.5–7 cm idler on a narrower tub;
  - seed 1 chose two Li-ion cells (7.4 V), inside D-062's range.
  - No seed used the parameter-only routes physics opened (big fast wheels, a sliding belly). System 2 reasoned to the redesign the failures named.
  - The films replay the MuJoCo episodes; the deepest a robot part sits in the ground is 0.84 mm.

## D-064 A long slide: a gear sector on the horn drives the rack on 150 mm rods (judge: world_gripper)

**Why.** Under the mechanism laws every gripper seed stopped at 17/21. The four groups still lost are the pudding box (90 mm), sugar box (92), mustard bottle (96) and tuna can (85). The hinged jaw's fingertip rises 28–33 mm at those widths and tilts 32–42°, and it opens at most 91 mm. The catalogue slide does not rise, but a full pinion is capped at 60 teeth (it would otherwise clear neither the rod clamp's bracket nor the wrist), so it stops at 58 mm. All three seeds recorded the same need as an idea: a straight slide with about 90 mm of travel. Seed 0 (iteration 11) proposed the form adopted here: a printed module-1 gear *sector* of 42 mm pitch radius on the servo horn, with teeth only over the arc the jaw sweeps, driving the catalogue rack and carriage on longer rods. A step-up gear stage (seeds 0 and 1) needs a second shaft on bearings beside the gripper frame. Every place to carry it there is taken by the gears' own planes, the rods or the wrist, so the sector is the buildable form of the same request. The redesign stays System 2's: the catalogue only gains the part.

**Change.**
- New catalogue part `long_slide` (jaw slot), with param `sector_radius_mm` from 30 to 50. The finger travels that many mm per servo radian, and squeeze is servo torque / radius.
- The sector carries teeth over the jaw's range plus two teeth either side, on a web to the horn hub. The rack and carriage are the catalogue slide's. The rods are 3 × 150 mm (new manifest item `steel_rod_3x150`, $2.00).
- The same laws apply: carried, no interference through the jaw's range and the wrist's roll, and the stop.

**Prediction.**
- A sector of 40–48 mm passes every law. Above about 48 mm the sector or the rack meets the wrist or the clamp's bracket, and is refused with numbers.
- At 45 mm the widest opening is about 95 mm, and the fingertip stays within ±3 mm of level at every width.
- The campaign (three seeds, as before): System 2 picks the long slide within two iterations and gains the tuna can and the pudding and sugar boxes. The mustard bottle (96 mm) is marginal. Small objects (dice, 17 mm) may be lost to the coarser resolution: 45 mm per radian puts a 1° servo error at 0.8 mm. I expect 19–21/21 train.

**Measured, before the campaign.**
- Laws: sectors of 30–47 mm and 49 mm pass. At 48 and 50 mm the sector's last tooth meets the wrist rolled to −2.74 rad, by 1.3–1.6 mm, and is refused with those numbers. That boundary is not monotonic because whole teeth move the sector's ends. At 47 mm the fingertips open 98 mm and the moving fingertip stays +1 mm of level at every width; squeeze is 34 N.
- My first layout hung the rods under the rack, 12–24 mm above the fingertips at a 47 mm sector, and the lower rod pressed on the tuna can's rim (the force limit). Re-laid, before any campaign: the rods run outboard of a 102 mm deep object (y = −56 mm) and above the rack's pitch line. The carriage block sits beside the gear plane, and the clamp's bracket crosses that plane between the rack's teeth and the sector's sweep, to the fixed finger's outer face. The lowest moving part is now about 25 mm above the fingertips.
- Single trials at 47 mm:
  - **Bare PLA faces** held the pudding box, sugar box, mustard bottle, tuna can and dice. The potted meat can (102 mm deep) failed because the carriage pressed 20 N on its top edge.
  - **0.5 mm TPU pads** held the potted meat can, but the pads close the opening to 91 mm, which loses the sugar box and the mustard bottle.

  That is a real trade-off, and it is left to System 2.
- **The force-limit diagnosis now names the part.** It used to read "a contact exceeded the force limit". It now says, for example, "the carriage pressed 20 N on its side … +42 mm from the object's centre (top at +42)", so a redesign can move the part that pressed.
- **Campaign under this key** (`results/evolve/gripper_s2long_s{0,1,2}.json`, 14 iterations, 31–44 min a seed): **the prediction missed.**
  - Every seed ended at 17/21 train, the same as before, on the revolute jaw with retuned TPU pads (30°, 14° and 23°). Test: 17 of 21 for each seed, against stock 14, 16 and 14.
  - Every seed did pick the long slide in iteration 0 or 1, as predicted: 10, 12 and 15 long-slide trials, of which 4, 2 and 8 were refused by the wrist-interference law at 48–50 mm.
  - The best long-slide trial in each seed gained 3–4 of the 4 wide groups (gelatin, pudding, sugar and tuna; the mustard bottle never). Each lost as many: the apple, banana, tomato soup can and potted meat can. The no-regression rule rejected every one.
  - **Why it lost them** (the diagnosis now names the part): with nothing above the fingertips but 25 mm of finger, the rack presses on the tops of tall or round objects as the jaw closes. That happened in 56 force-limit outcomes across the seeds, plus 5 by the carriage. Pads long enough to reach below it (reach 22–30 mm) then land on the tops of small objects instead.
  - **What System 2 asked for next, unprompted**:
    - A **drop finger**: a rigid finger hanging 60–95 mm below the rack and rods, with a matching extension on the fixed jaw (seeds 0 and 1).
    - An **outboard rack**: so nothing spans the gap between the fingers (seeds 0 and 1).
    - A **self-aligning pivot pad**: on the moving jaw (all three seeds; this is decision 2's pad shoe).

    These are recorded as untested ideas in each record, and they become D-065 and D-066.

## D-065 The pick comes down from above its lowest part, and System 1 searches its grip depth (skill: primitives)

**Why.** Replaying D-064's rejected long-slide trials showed two faults in System 1's pick (`control/motor.py::grasp`). Neither is a fault of the body.
- **The approach point was a fixed 40 mm above the grip.** The arm runs in a straight line from its start to that point, then descends.
  - With short pads, the jaws' lowest part ends that line just above the object's top. The stock reaches 23.5 mm below its tool point, and 17 + 6.5 + 10 is under 40.
  - With 28 mm pads it ends 23 mm below the top. So the line dragged the fixed pad sideways into the object's shoulder.
  - For example, the tennis ball: the fixed pad came down 21 mm from the ball's axis on a 33 mm radius, at 13 N, and hit the force limit. The same happened on the stock hinged jaw with the same pads.
  - Every long-reach pad System 2 tried in any gripper campaign (reach 20–30 mm) lost the round objects this way. The diagnosis honestly said "a pad pressed down on its top", but that was the pick's fault, not the pad's.
- **The grip depth ignored the parts above the fingers.**
  - The pick puts the tool point 17 mm below the object's top. Grip depth is not a control System 1 searches.
  - The slide's rack spans the gap between the fingers 22.5 mm above the tool point (a 46 mm sector). So on a tall object it came down to within 5 mm of the top, and the object rising in the squeeze closed that. A pick that knows its jaw keeps what spans the gap clear.

**Change** (as made; the prediction below was written for a first form, kept as written).
- The approach point is `max(40 mm, depth + below + PICK_CLEAR)` above the grip, so the lowest jaw part passes `PICK_CLEAR` (10 mm) over the object's top.
- System 1 searches the grasp's `depth`, which `configs/primitives.yaml` always declared (2–35 mm). Its start point is the pick's default, `pick_depth`. It had never been searched: `local_search.default_value` returned no number for a default computed per object, and parameters without one are skipped.
- **First form, rejected on measurement.** It capped depth below the slide's rack (`fingers.gap_ceiling`, the rack's underside less 10 mm).
  - Over the 21 groups (one train placement each, seed 0), the bare long slide fell from 17 to 15. A ball's top passes under the rack, which runs 28.5–34.5 mm to the side of the fingers' centre line. The cap made the slide grip lemons and plums above their equators.
  - The class model cannot say which objects reach the rack: the soup can is classed a sphere but lies on its side, flat-topped along the rack.
  - Searching depth lets each object find its own depth.

**Prediction** (written before the change was made in the tree; the patch was tried on replays first, below).
- Stock gold, the phase A/B sets and every test that uses the stock or short pads: unchanged, because neither rule binds there.
- Long-reach pads on any jaw stop landing on the round objects.
- The campaign (three seeds): System 2 tries the long slide again within two iterations, now with pads of reach 15–30 mm. It keeps the apple, banana, soup can and tennis ball, and gains the tuna can and the pudding box. Pads close the opening to about 91 mm, so the sugar box and mustard bottle stay out unless it goes bare. I expect 19/21 train in at least two seeds.

**Measured on replays, with the change patched in before it was made.**
- Long slide at 46 mm with 28 mm, 2 mm TPU pads, System 1's default first attempt:
  - Approach fix alone: tennis ball and cup are now held. The soup can and apple still fail, with the rack pressing 27 N and 8 N on their tops.
  - The stock hinged jaw with the same pads now holds the tennis ball as well.
- **Over the 21 groups** (`GripperWorld.solve`, System 1's full search, on each group's first train placement, seed 0):

  | body | before | after |
  |---|---|---|
  | stock | 16 | **17** (+gelatin box) |
  | evolved hinged jaw, 30° 0.5 mm TPU pads (s0's b050) | 17 | 17 |
  | long slide 47 mm, bare | 17 | **20**: all but the banana |
  | long slide 47 mm, 20 mm 0.5 mm TPU pads | 16 | 18 |

  - **The prediction missed on the stock:** depth is searched for every body, and the stock now grips the 30 mm gelatin box deeper, 20.5 mm (the table cap), so it holds it. That gelatin box is the group every seed's evolved pads had gained, so the evolved hinged bodies are now no better than the stock.
  - The bare long slide gains the apple, soup can and potted meat can (a shallower grip keeps the rack and carriage off their tops) and keeps every wide box.
  - `test_world_gripper::test_solve_reads_the_failure` now reads the stock's failure on the pudding box instead.


## D-066 The pad's printed shoe is simulated (world: physics)

**Why.** A pad was simulated as its land boxes alone. The pad as built (`design/pad_cad.py`) is a PLA shoe that slides over the stock jaw's tapered tip, with the land in front, a TPU skin and two M2 cap screws behind:
- its socket is the tip's convex hull grown by the slip fit;
- its walls are at least 1.6 mm thick round the tip.

The pictures drew the shoe, but the physics and the mechanism laws never saw it: a "thin spot" in harness.md, and against invariant 11 (what is drawn is what is simulated).

**Change.**
- New `sim/shoe.py` (component `physics`) holds the shoe's sections, socket and screw positions; `pad_cad` now draws from them, so the picture and the simulation share one set of numbers.
- The simulator adds, on the jaw body that carries each pad:
  - the shoe's outside as convex pieces between successive sections, colliding like the pads (`jaw_*_pad_shoe*`, PLA friction). They stop 0.3 mm behind the land's face, so an object still meets the land and the skin's friction.
  - the two screw heads, as cylinders (`jaw_*_pad_screw*`). Fastened to the finger, they are finger parts for the stop rule (D-061).
  - the mass of the shoe (its loft less the socket), the skin and both screws: about 2 g a finger.
- `test_shoe.py`:
  - the shoe keeps the design language's slip fit and edge round;
  - stock fingers have none;
  - every padded finger has pieces, two screws and 1.5–6 g;
  - every piece stops behind its land;
  - the socket holds the tip: over 97 % of the tip's vertices lie inside the shoe's outside.

**Prediction** (written before the measurement below; the laws were run on six bodies while building it).
- Over the 21 groups (System 1's full search, each group's first train placement, seed 0), stock, the hinged jaw with 30° pads, the bare long slide and the long slide with 20 mm pads each change by at most one group. The shoe sits behind and beside the land, and adds 2 g against a 30–500 g object.
- The bodies that passed the laws still pass. The four-bar at 89 mm with pads was already refused (its coupler meets the fixed pad); its shoe adds a second line to that refusal.

**Measured.**
- Over the 21 groups, every body holds exactly what it held under D-065, group for group: stock 17, the hinged jaw with 30° pads 17, the bare long slide 20, the long slide with 20 mm pads 18. The prediction held.
- Every test passed, including the golden grasps.
- Model builds take 0.03–0.06 s, with the shoes cached per finger spec.
- What the shoe changes is what the films show and what the laws check: the shoe and screws, no longer boxes alone.

## D-067 The pads are priced as the shoe they are printed as (judge: build)

**Why.** `design/build.finger_estimate` priced a pad's print from its land boxes. That is about 0.3 cm³ a pad at 30° with 0.5 mm TPU. The part printed is the shoe of D-066, about 1.1 cm³ of PLA, with the skin as its TPU. The bill of materials, the print minutes and the added mass System 2 reads all under-counted the pads.

**Change.**
- `finger_estimate` and `finger_print` read the PLA and TPU volumes of both shoes (`sim/build_model.shoe_volumes`, cached per finger spec), not the boxes.
- The screws were already on the bill (four M2).

**Prediction.**
- PLA grams for a pair of pads rise about 3× (for example 0.7 g to about 2.4 g at 30°).
- Print minutes rise by a few minutes.
- No grasp outcome changes; this is a price.
- Tests that pin pad grams or minutes move, and are re-derived here.

**Measured.**

| pads (both) | PLA before → after | TPU | print minutes |
|---|---|---|---|
| 30°, 0.5 mm TPU | 0.55 → 1.29 g | 0.09 g | 13.7 → 15.6 |
| reach 20 mm, 0.5 mm TPU | 1.34 → 3.71 g | 0.22 g | 16.0 → 22.2 |
| reach 30, groove 3, lip 4, 1.5 mm TPU | 2.17 → 3.47 g | 1.06 → 0.87 g | 20.4 → 23.3 |

- PLA rose 1.6–2.8×, a little under the predicted 3×. The boxes over-counted deep grooves and long lips, which the shoe's land carries in less material.
- Every test passed unchanged: none pinned pad grams.

## D-068 A force limit is named from the step that pressed hardest (judge: world_gripper)

**Why.** In the D-065 campaign, seed 0's final body failed two placements with "a contact exceeded the force limit at t=0.02 s". That reads as a start-of-episode collision, and it was not one: a replay with every contact logged showed the moving pad pressing 17 N on the sugar box's top edge at 2.06 s.
- The diagnosis replay samples contacts every 10 simulation steps (20 ms), but a force limit can trip between two samples.
- With no sample showing a contact, the fallback took the first row: t = 0.02 s, no part.
- System 2 was told something false about when and what pressed.

**Change.**
- The replay still keeps a row every 10 steps.
- It now also keeps the step where the robot pressed hardest on the object.
- The force-limit diagnosis reads that step.
- Verdicts are untouched: only the text System 2 reads changes.

**Prediction.** No verdict or held count moves. Every force-limit diagnosis now names a part and a time inside the attempt, except a limit tripped by contact with the table alone.

**Measured.**
- Seed 0's final body (D-065 campaign):
  - `rva000_sugar_box`: "a pad pressed 12 N down on its top at t=2.07 s".
  - `rva000_potted_meat_can`: "a pad pressed 25 N down on its top at t=0.89 s".
  - Both were "t=0.02 s" before.
- `test_world_gripper` gains the case.

## D-069 Harder routes: garden stairs, a power cord, a threshold strip and a hose (judge: world_mobile)

**Why.** Under MuJoCo (D-063) every seed reached 10/10 in two iterations with the same answer: tracks, a sonar and a narrower tub. The routes test one edge at a time, with nothing between edges and nothing round. A house has:
- flights of steps whose treads are shorter than a robot;
- cords and hoses lying across the floor;
- bevelled threshold strips where carpet meets tile.

These test different things: pitch and tipping across several nosings, a round obstacle against belly clearance, and a ramp up and down.

**Change.**
- Three new obstacle kinds, driven in MuJoCo like the step and the trench:
  - `stairs`: `n` steps of rise `h` and tread `run`, each nosing rounded like every edge;
  - `bump`: a cable or hose of diameter `d` across the path, a capsule;
  - `threshold`: a strip `h` high and `w` wide, with 20° bevels each side.
- Three new routes (each with two test variants at ±5 %, as every route has):
  - `patio`: 2 m of concrete, three 7.5 cm steps on 18 cm treads, 1 m of concrete; 60 s (first written as 5 cm on 24 cm: see below).
  - `study`: 2 m of wood, a 1.2 cm power cord, 1 m of wood, a 1.4 cm threshold strip onto the rug, 2 m of carpet; 45 s.
  - `greenhouse`: 2 m of grass, a 2.5 cm garden hose, 2.5 m of gravel; 60 s.
- System 1's approach-duty control now applies at every edge kind.

**Prediction.**
- The stock (four 3 cm wheels, 2 cm belly) fails the patio and the greenhouse hose and passes the study, so it holds 6 of 13 train routes.
- The three D-063 final bodies (tracks with 6.5–7 cm idlers on a narrow tub) hold the study and the greenhouse. They may fail the patio: a 20 cm tub on 24 cm treads pitches nose-down onto the second nosing.
- A new three-seed campaign reaches 13/13 in at least two seeds, and its redesigns name the stairs (a longer track or a higher idler) if the patio fails.

**Measured, before the campaign** (System 1's full search, each route's train placement).
- **At first draft (5 cm steps on 24 cm treads):**
  - The stock failed the patio ("its running gear slipped on the 3 4.9 cm steps") and grounded its 2 cm belly on the 2.5 cm hose. It passed the study. As predicted.
  - All three D-063 final bodies passed all three routes: the patio did not trip them. A route that no evolved body fails tests nothing new, so it was steepened before any campaign.
- **Steeper patio, probed on several bodies** (7.5 cm steps on 18 cm treads, the treads shorter than the robots):
  - Seeds 0 and 1's final bodies slip on it (idlers 6.5–7 cm on 24 cm tracks).
  - Tracks with an 8 cm idler on 30 cm passed, with or without a 4 cm ride height.
  - A rocker-bogie with 6 cm wheels did not, even on 25 mm metal gearmotors.
  - So the patio needs a redesign the catalogue can build.
- **A 3.5 cm hose** stopped no tracked body, so the greenhouse keeps the 2.5 cm hose, which tests the stock's belly.
- **No catalogue body holds every route now.**
  - Climbing 7.5 cm stairs takes an 8 cm idler, and the track stands 1.5 cm above it.
  - The sofa leaves 9 cm, so the most it allows is a 7.3 cm idler.
  - Measured: tracks 8 cm × 30 cm on a 26 cm tub hold 12 of 13, all but the sofa ("the robot is 9.5 cm tall; the overhang is 9.0 cm"). The old gold body, 6.6 cm × 22 cm, holds 12 of 13, all but the patio. Seven sofa-height bodies tried slipped on the stairs.
  - The steps were not lowered to remove the conflict. At 7.0 cm every D-063 final already climbs them, and the test would add nothing.
  - A body that does both needs a mechanism the catalogue does not have. As with the gripper's long slide (D-064), the catalogue gains one only when System 2 asks for it.
- **Tests:**
  - `test_world_mobile` now counts 13 routes and 26 test tasks;
  - the stock also holds the study, and fails the patio and greenhouse on its belly;
  - the gold test records the conflict;
  - a new test drives the cord, threshold, hose and stairs.

## D-070 Mecanum wheels roll on their rollers (judge: world_mobile)

**Why.** In the MuJoCo episodes (D-063) a mecanum wheel was a plain cylinder. Its rollers were two factors: friction × 0.7 everywhere, and × 0.6 more at an edge ("rollers spin free at an edge"). The CAD drew ten rollers on 45° axes that the physics never saw (invariant 11).

**Change.**
- Each mecanum wheel is a driven hub carrying ten rollers on free hinges, as the CAD draws them:
  - centres at 0.82 r, capsules of radius 0.18 r reaching the wheel's radius;
  - axes at 45° across the rim, in the X pattern: front-left and rear-right one way, the other two the other way, so driving straight their side forces cancel through the chassis.
- A roller passes force only across its own axis, so the traction factor and the edge factor are deleted. Loose ground (rollers clog) and rolling resistance keep their factors: neither is a contact MuJoCo models.
- The CAD reads the roller layout from `sim/mobile_model.py`.

**Prediction.**
- On flat ground the push a mecanum body reaches is within 15 % of before: cos 45° is 0.71, the factor was 0.7.
- At edges it climbs a little higher than the 0.6 factor allowed, but still below plain wheels of the same radius: a roller meets an edge on a 0.18 r capsule, not the tyre's full radius.
- No held route moves for the stock, the gold body or the D-063 finals (none uses mecanum).
- The distractor test still holds: mecanum wheels gain nothing over the stock's routes.

**Measured** (the highest edge climbed at full duty, MuJoCo bisection; and the train routes held).

| body | wood edge, before → after | concrete edge | routes held |
|---|---|---|---|
| stock wheels, 3 cm | 3.14 (unchanged) | 3.61 | 6 |
| mecanum, 3 cm | 2.34 → 3.05 cm | 2.53 → 3.42 | 5 → 5 |
| mecanum, 4 cm | 3.31 → 4.00 | 3.44 → 3.81 | 5 → 5 |

- Mecanum wheels climb higher than the 0.6 edge factor allowed, and still below plain wheels of the same radius. As predicted.
- Driving straight at a 3 cm step, the X pattern's side forces cancel: 0.0 cm of drift across it.
- No held route moved, and every mobile, film and CAD test passed. The mecanum wheels still gain nothing over the stock (they lose the gravel ramp, where loose ground clogs their rollers).

## D-071 The talking robot's drives obey the laws of the linkages that move them (judge: world_phonology)

**Why.** The phonology world gives System 1 position controls (`phonology.rig`): a tongue place, a cam lift, a lip radius. The bench rig's CAD (`worlds/rig_cad.py`, tooling) draws a servo and a linkage behind each control, but nothing checked that the linkage can make the motion. A probe of the CAD's linkages at the ends of every control range, for bodies at the catalogue's corners, found two parts that cannot be built as described:
- **Drawstring (every `round_lips`):** a 12 mm horn turning 100° takes up 18 mm of cord. Closing the ring from 1.45 cm radius to shut takes up 91 mm, its change in circumference.
- **Larynx slide (every `larynx_tube`):** a 180° servo turns the slide's 8 mm lead screw half a turn, 4 mm. The part promises 10–30 mm.

Two more faults:
- **A pad that leaves the tract:** a 10 cm rail starting 2.5–4.0 cm behind the teeth carries the pad to 12.5–14.0 cm, while a 12.5 cm cast ends at 12.5.
- **A crank short of the rail:** the rail crank's servo was rounded onto the base plate's grid the wrong way, 2 mm short of the rail's back end.

The plan's P5 asks for exactly this: articulators driven through linkages, with reachable ranges and checks.

**Change.**
- **`worlds/linkages.py`** (new, in the `world_phonology` component) is the mechanism between each servo and its control. It holds:
  - the rail's offset slider-crank;
  - the tongue cam's eccentric and roller follower;
  - the tip blade's two-servo five-bar;
  - the jaw link;
  - the protrusion pushrod;
  - the drawstring's spool;
  - the larynx's pinion and rack.

  `rig_cad` draws from it, so the drawn linkage is the checked one.
- **Pre-trial laws.** `PhonologyWorld.check` refuses a body whose drives break a law at any of 21 postures per control (a 21 × 21 grid for a coupled pair: tip place × height, lip radius × protrusion). The laws:
  - **reach:** every rod spans the gap from its arm to its pin;
  - **travel:** every arm turns at most the MG90S's 180°;
  - **transmission:** where a rod drives a slide, its angle to the slide's normal is at least 30°; the five-bar's two rods meet the crosshead at 30° or more;
  - **no dead centre:** a horn and its rod stay at least 10° from lying in line;
  - **draw:** the spool winds the cord the ring needs, and the pinion moves its rack the slide's whole travel, within the servo's turn;
  - **clear of the glottis:** the tongue's sled reaches 2.5 cm behind its pad. It must stop short of the reed housing's stand, or of the larynx slide's carriage 3 mm further forward. The rail crank and its rod must pass clear of the stand.
- **A posture law, inside every attempt.** The tip blade presses from the palate and the body pad from the floor. Where both narrow one section, the gaps they leave must sum to the bore. The concentric tract model let each pass through the other, so `phonology.clash` measures the crossing. A posture crossing by more than 0.5 mm fails its attempt, whatever it sounds like, and its detail says so. System 1's map scores those postures as failures, and the diagnosis leaves them out.
- **New drives for two parts.** Each part's controls, ranges and acoustics are unchanged:
  - **`round_lips`:** both ends of the drawstring wind onto a 16 mm spool on the servo. The cord runs in a sheath fixed to the ring clip, so protrusion draws none. Closing the ring takes 163° of turn.
  - **`larynx_tube`:** a pinion (module 0.5, 40 teeth, 10 mm pitch radius) on the servo drives a rack on the slide's carriage. 30 mm takes 172°.
- **Harness geometry the CAD audit found wrong:**
  - **Rail crank:** the servo moves onto the grid on the side that reaches the rail's back end. The crank is 2 mm longer than half the travel, so the stroke between its dead centres leaves a grid step to place the servo in. The rod is 12 mm longer, so it meets the sled at 39.9° or more (30.1° before).
  - **Reed housing's stand:** it becomes a bracket on the +y side. The rail crank's arm swings under the cast's centreline, and the old full-width stand stood in its path.
  - **Pharynx clamp:** its ring opens underneath, so the follower's guide passes under the cast's back end at full stroke.

**Prediction** (written before `check` called the laws).
- The stock passes.
- So does every final body of the frozen c1 phonology campaign (three System 2 seeds, all `short_cast` + `tip_tongue` + `round_lips`). So do 29 of its 30 random-search finals; the 30th (random s12, a long rail on a 12.94 cm larynx tube) is refused.
- Of the 468 bodies at the catalogue's corners, the 48 with a long rail on a 12.5 cm tract are refused, and no others.
- The tightest drives are the jaw link (30.4° at full opening) and the second tip servo, whose horn comes within 18° of a dead centre.
- **Three-seed re-run:** 13/13 train groups on each seed, as in c1. The acoustics, the controls and the stock are unchanged. The only difference a run can see is the refusal of long rails on short casts, which costs a proposal, not a sound.

**Revised before any campaign.** A CAD audit of law-passing corner bodies at their control ends found two more faults:
- the sled and the rail's rod running into the glottis's hardware;
- the blade and pad pressing through each other.

The sled and crank law and the clash law above were added, the stand and clamp redrawn, and the prediction revised:
- The stock and the c1 System 2 finals still pass, and still hold the same tasks under System 1: the stock 18 of 39 train and test tasks, the c1 final all 39, before and after.
- Of the 30 random finals, 3 are refused, all for the sled:
  - s12 and s13, long rails on larynx tubes;
  - s16, a cam tongue reaching 10.94 cm on a 12.65 cm larynx tube.
- At the catalogue's corners, 156 of 468 bodies are refused, every one for the sled:
  - every long rail on a cast shorter than 15 cm, and one starting 4 cm back on a cast shorter than 16.5 cm;
  - a 7 cm rail starting 4 cm back on a 12.5 cm cast;
  - the same on any larynx tube shorter than 13.8 cm.

  No other law refuses a corner, so the drives are built for their parts' ranges.
- The clash law turns 14–21 % of a tip tongue's 600 sampled postures into failures. That is how many of the postures System 1 could reach before were ones the parts cannot take.
- **The three-seed re-run:** 13/13 train groups on each seed. The c1 final holds every sound without a clashing posture. A run that proposes a long rail on a short cast loses that proposal.

**Measured** (`results/evolve/phonology_s2link_s{0,1,2}.json`).
- **Three seeds:** each seed held 13/13 train groups at iteration 1 and 26/26 test tasks. Each kept the same body as c1 (`short_cast` 13.5 + `tip_tongue` 3.25/1.3 + `round_lips` 1.6) after 1 System 2 call and 4 proposals.
- **Refusals:** no proposal was refused, because none reached for a long rail.
- **Stock on the test split:** 12 or 13 of 26 tasks, as before.

As predicted. Under real linkages, a sled clear of the glottis, and a blade that cannot press through the pad, the c1 talker still says every sound. The 21 % of its sampled postures that clash were never the ones it needed.
