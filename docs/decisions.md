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


**The campaign** (three seeds, 14 iterations; `results/evolve/gripper_s2depth_s{0,1,2}.json`).

| seed | train groups by iteration | final train | test groups (stock) | kept body |
|---|---|---|---|---|
| s0 | 16 → 17 at 4 | 17/21 | 17 (15) | long slide, 46 mm sector; 15 → 12 mm reach, 2 mm grooved pads, 20 mm wide |
| s1 | 16 → 17 at 1 | 17/21 | 17 (16) | revolute; 10° pads, 2 mm TPU, 12 mm wide |
| s2 | 16 → 20 by 8 | **20/21** | 18 (15) | revolute; 42° pads, 10 mm wide, no reach |

- **The prediction missed:** 19/21 or better in one seed, not two.
- **The long slide.** System 2 tried it again within two iterations in s0, as predicted, and kept it at iteration 3. It never tried it bare. It kept pads and then nudged their reach and width one task at a time for ten iterations, without a new group.
  - The slide's sector stops the fingertips 90 mm apart, so the pudding box and sugar box, at about 90 mm, slip through unsqueezed (`fixed_jaw_missed`) or meet a pad on their top.
  - The bare slide held 20 of the 21 groups on one placement each in the replays above, but never in a campaign. A search that adds pads first never finds it.
- **s2's 20/21** is a hinged jaw with 42° pads 10 mm wide. The angled pads lean into a box's side instead of meeting its top edge.
- **The mustard bottle is lost on every seed** with `moving_jaw_on_top_face`: the hinged jaw's arc, or the slide's carriage, comes down on the bottle's shoulder. No kept body grips below it.
- **Stock:** the first iteration holds 16 groups. That is one fewer than the replay's 17, because the campaign needs both placements of each group.
- **Films:** `results/evolve/films/gripper_s2depth_s*` holds 6 before→after films of the changes the seeds kept. `results/evolve/gallery/gripper_s2depth_s*` holds 60 films of the stock beside each final body, one per group, replayed under each run's seed. A replay that disagreed with its run's record was not filmed (s2's stock gelatin box).

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

**The campaign under D-069 and D-070** (three seeds, 10 iterations allowed; `results/evolve/mobile_s2routes_s{0,1,2}.json`).

| seed | train groups by iteration | final train | test routes (stock) | kept body |
|---|---|---|---|---|
| s0 | 6 → 9 → 13 at 2 | **13/13** | 13 (6) | tracks 7 × 26 cm on a 26 × 10 cm tub, 4 cm ride; N20 100:1; Li-ion 2 cells; sonar |
| s1 | 6 → 11 → 12 → 13 at 3 | **13/13** | 12 (6), lost the patio | tracks 7 × 28 cm on a 24 × 10 cm tub, 4.5 cm ride; N20 100:1; Li-ion; sonar |
| s2 | 6 → 11 → 12 → 13 at 3 | **13/13** | 13 (6) | tracks 6.5 × 24 cm on a 22 × 12 cm tub, 3 cm ride; encoder N20 50:1; Li-ion; sonar |

- **D-069's prediction held:** 13/13 on every seed, not just two. System 2 named the patio steps from iteration 0 ("belly_hits_step_edge on … the 7.5 cm patio steps"), and every seed answered with tracks, a Li-ion pack and a sonar.
- **D-069's measured conflict was wrong.** It said no catalogue body holds every route: an 8 cm idler climbs the stairs but will not pass under the sofa. All three finals pass under the sofa with idlers of 6.5–7 cm, and climb the stairs.
- **Why: the patio verdict is a knife edge.** Each body was replayed on the patio's train placement under five search seeds (System 1's full search each time):

  | body | patio climbed |
  |---|---|
  | D-063 seed 0 final (7 × 24, NiMH) | 2 / 5 |
  | D-063 seed 1 final (6.5 × 24, Li-ion), measured as slipping in D-069 | 3 / 5 |
  | this campaign's s0 | 4 / 5 |
  | this campaign's s1 | 4 / 5 |
  | this campaign's s2 | 3 / 5 |

  - Every tracked body climbs on some seeds and slips on others. D-069's probe was one seed per body, so it read 2–3 in 5 as "slips".
  - A run needs its own seed to succeed (with `confirm=2` on fresh noise inside it), so the 13/13 is partly the search's luck. s1's test placement, 5 % different, slipped.
  - The longer finals (26–28 cm tracks, 4 of 5) are steadier than the 24 cm ones (2–3 of 5), as D-069 expected from treads shorter than the robot. Steadier is not reliable.
- **Open fault, for the next judge-tier change:** a group held on one search seed is not "held" when its verdict flips on half of them. The fix to weigh is to score a route by its success rate over several search seeds, with a threshold, at the cost of several times the trial time per body. Until then the patio's 13/13 reads "climbs on most seeds".
- **Films:** `results/evolve/films/mobile_s2routes_s*` holds before→after films of the changes the seeds kept. `results/evolve/gallery/mobile_s2routes_s*` holds the stock beside each final on every route, replayed in MuJoCo under each run's seed.

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

## D-072 Harder sounds: the fricatives /s/ and /ʃ/, and the syllables /ta/ and /ka/ (judge: world_phonology)

**Why.** The talking robot's thirteen sounds are steady postures: ten vowels and three closures. System 2 solves them in one call on every seed (D-071). Speech has two things they leave out:
- **Turbulence:** a fricative is a narrow channel whose noise the cavity in front of it filters. The world already names the channel: narrower than a vowel (0.15 cm²), wider than a closure (0.05 cm²).
- **Time:** a stop is heard as a stop only when the release reaches the vowel fast. The servos and linkages now bound how fast that can be (D-071).

**Change.**
- **Fricatives.** /s/ (as in *see*) and /ʃ/ (as in *she*): the narrowest point lies in its place, and its area is at least 0.05 and below 0.15 cm². The airway elsewhere is open (above 0.2 cm²). The front cavity's first resonance, the peak of the frication noise, lies in its band:
  - /s/: the alveolar ridge, 0–1.5 cm behind the teeth; a peak of 4–8 kHz;
  - /ʃ/: behind it, 1.5–3.5 cm; a peak of 2–4 kHz.

  The front cavity runs from the constriction to the lips. Its peak comes from the same lossy chain model as the vowels, with the constriction as its closed end, on a grid reaching 10 kHz.

  The bands are the published ranges. For /s/, energy above 4 kHz, with a major peak reported at 3.5–5 kHz (Behrens & Blumstein 1988) and at 6–8 kHz (Jongman, Wayland & Wong 2000). For /ʃ/, 2–4 kHz (Hughes & Halle 1956; Behrens & Blumstein 1988). All as summarized by Maniwa, Jongman & Wade (2009); the original tables were not reachable.
- **Syllables.** /ta/ and /ka/: a closure posture that passes the stop, a posture that passes /ɑ/, and between them no servo turns more than 30°. An MG90S turns 60° in 0.1 s at 4.8 V, so the release reaches the vowel in 50 ms. Formant transitions shorter than about 50 ms are heard as stops, and longer ones as glides (Liberman, Delattre, Gerstman & Cooper 1956).
  - The servo angles come from each drive in `linkages`: crank, cam, five-bar, jaw link, pushrod, spool, pinion.
  - System 1 searches both postures together, starting from pairs in its map, with servo noise on both.
  - The tip blade's clash with the pad (D-071) applies to each posture.
- **/pa/ is left out.** No lip part in the catalogue can release a bilabial closure that fast. On the best sampled pair, the round lips' spool turns 77° (129 ms) from the seal to /ɑ/, and the jaw lips never seal. A task no catalogued part can reach gives System 2 nothing to climb.
- **Test splits:** the task groups go from 13 to 17. Each new group has a train task and two test tasks, under other noise, with the fricative bands and the /ɑ/ targets moved 3 %, as for the other sounds.
- **The film and diagnosis read the new kinds.** A fricative's film plays the front cavity's filtered noise. A syllable's film glides from the closure through the burst to the vowel over the time its servos take.

**Prediction** (from a sampled-posture probe, before the change):
- **The stock holds none of the four new groups.** Its single cam pad reaches neither the ridge nor a narrow channel behind it. It holds no /k/, so it holds no /ka/.
- **The c1 final** (`tip_tongue` + `round_lips`, 13.5 cm) holds /ʃ/, /ta/ and /ka/. /s/ is uncertain: the rounded lips' 1 cm tube puts most of its alveolar channels' peaks below 4 kHz, and 9 of 21 sampled channels lie in the band.
- **Bodies with spread or jaw lips** hold /s/ easily (44–48 of 47–48 sampled channels in the band) but lose /uw/, /uh/, /ao/ and /p/, which need rounding.
- **Three-seed campaign:** 16 or 17 of 17 train groups on each seed, from more than one System 2 call. /s/ against the rounded vowels is the conflict the catalogue may not resolve.

**Measured, System 1 on the train tasks** (before the campaign):
- **Stock:** 6/17, none of the new groups. /s/ and /ʃ/ fail with `no_channel_at_*`; /ta/ and /ka/ fail on their closures. As predicted.
- **The c1 final:** 16/17. It holds /s/, /ʃ/ and /ka/. /ta/ fails as `transition_too_slow`: the nearest clean pair is 53° apart on the drawstring servo.
  - The prediction was wrong on both halves: /s/ was the uncertain one, and /ta/ was expected to hold.
- **Tip tongue with jaw or spread lips:** 14 and 12 of 17. Each loses /p/ and the rounded vowels, as predicted, and also /ta/ or /ka/ by a few degrees.
- **A 17/17 body exists in the catalogue:** the c1 body with its rail starting 3.5 cm back. It holds the syllables on train and one test task. The /ɑ/ target moved up 3 % takes more than 30° from either closure. So the gold test requires every train group, and every test group but the syllables.

**Measured, the campaign** (`results/evolve/phonology_s2sounds_s{0,1,2}.json`; films and galleries in `results/evolve/{films,gallery}/phonology_s2sounds_s*`).

| seed | train groups by iteration | System 2 calls | final body | test groups (stock) |
|---|---|---|---|---|
| s0 | 6 → 16 → … → 17 at 9 | 9 | c1, cast 13.2 cm | 15 (6), loses /ta/, /ka/ |
| s1 | 6 → 16, then 16 through 10 | 10 | c1 | 15 (6), loses /ta/, /ka/ |
| s2 | 6 → 15 → 17 at 2 | 2 | c1 with rail 3.0, cast 13.0 cm | 15 (6), loses /ta/, /ɝ/ |

- **As predicted:** 16 or 17 of 17 train groups per seed, from more than one call.
- **The conflict was not the predicted one.** /s/ was held on every seed. The last group was /ta/ each time, at the 30° edge, and the syllables are what the test split loses.
- **System 2 found the lever:** the cast's length and where the rail starts, so /ɑ/ reaches its F1 with the tip blade held higher. s0 and s2 found settings that work. s1 tried rail starts of 3.5–4.0 cm on 13.25–13.4 cm casts, and less protrusion. Each missed /ta/ by a few degrees or lost /ka/. It never tried 3.5 cm on the 13.5 cm cast, which holds all 17.
- **A harness fault:** in s0's second call, System 2 wrote three proposals whose values were only in their prose (rail_start 3.25 → 4.0, a shorter cast), with an empty `edits` list. The loop refused each as "the edit leaves the body unchanged" and the iteration was lost. The refusal should name the fault, no edits given, so the next call can correct it.

## D-073 A fourth world: the ME 2110 competition robot, every task in one 40 s match (world: me2110)

**Why.** The three worlds each test one capability: grasping, crossing ground, shaping a tract. A real design brief asks one machine to do several things at once, sharing its actuators, its envelope, its budget and its time. The ME 2110 "Pirates of the Caribbean" brief (`ME2110_Fall_2026_Final_Project_v1.3.pdf`, used here as an example brief, not a class) is such a brief. Six scored tasks share one 40 s match. The kit supplies 2 gearmotors, 2 pneumatic pistons, 4 mousetraps and 5 rubber bands. The robot starts inside a 12 × 24 × 18 in box, and the parts bought must cost under $120.

**Change.**
- **A new world, `me2110`** (`src/sea/worlds/me2110.py`, simulator `src/sea/sim/me2110_model.py`, judge component `world_me2110`). The arena is built in MuJoCo from the brief's figures, in inches:
  - the start line, the 30 in dock, and the quadrant's diagonals;
  - the fortress: 24 in across and 13 in tall, turning clockwise at a constant speed, with four cannon ports. Each port is a 4 in gap in the wall, its barrier at the colour's tab height (6, 7, 8 or 8.75 in), and a receptacle behind it;
  - the 4 in square flagpole up to 72 in, and the tentacles: 14 in across, up to 24 in;
  - 10 coins (2.5 × 0.5 in) and 5 jewels (1.5 in) on the fortress top's rim;
  - the mermaid and merman (3.5 and 3.7 in bases, 7 in tall), 20 in apart and 20 in out;
  - the chest (4.7 × 3.2 × 3.6 in), its lid propped open, 19.9 in out, with the heart (1.35 × 2.5 in) inside 0.5 in below the lip;
  - the gold and jewel mounds (3.5 × 4 in tops, a 1 in scoring border).
- **The robot is real mechanisms on kit actuators.**
  - **Gearmotors** run on a DC torque-speed line.
  - **Pistons** push P·A at up to 7 bar, through pinned closed linkages (a cylinder, its rod, the lever it drives). Their air comes from one tank, and each stroke spends it.
  - **Mousetraps** follow their spring's torque-angle line. **Rubber bands** are springs.
  - **Latches** are equalities released at a time, **magnets and clips** are equalities that break above their holding force, and **the chest's prop** gives way when the chest is jostled.
  - The slots and their stock parts:
    - frame: a plywood frame;
    - drive: two gearmotors on wheels;
    - mermaid: a push bar;
    - release: latches tripped when the drive arrives;
    - cannon, flag, heart, treasure: none.
  - The catalogue's alternatives:
    - drive: a mousetrap car, or a static base;
    - mermaid: two arms swung by mousetraps;
    - cannon: a gated chute on a piston escapement, timed by the IR sensor seeing the port's tab;
    - flag: a piston-raised boom, or a telescoping mast extended by rubber bands;
    - heart: a hook boom dropped by gravity and raised by a piston;
    - treasure: a peel blade on a fold-out trough with a size sieve and two side troughs to the mounds;
    - release: a cam timer on a gearmotor.
- **The brief's rules become laws.** Before trial:
  - the folded envelope fits the box, in either orientation;
  - kit actuators within the kit's counts; every stored-energy part has a release;
  - the parts bought under $120, prints at $0.05 per gram;
  - the pistons' air within the tank;
  - no magnets on the robot.

  In the episode:
  - projectiles fly at most 12 in unsupported and never upward;
  - the robot never deliberately leaves its quadrant;
  - at 40 s it is static, its pistons are retracted and its motors are off;
  - any part inside the fortress's cylinder at 40 s voids the cannonballs.
- **Seven groups.** A task is achieved when it scores at least half its maximum points:
  - activate: leave the dock, 11 → 6;
  - gold and jewels: 175 → 88;
  - flag: affixed above 24 in, 40 → 20;
  - cannonballs: 36 → 18;
  - mermaids: 24 → 12;
  - Davy Jones' heart: 61 → 31;
  - the match: all six achieved in one 40 s run.

  Train tasks are two port colours at 3 rpm. Test tasks are the other two colours at 2 and 4.5 rpm: the brief says only that the speed differs between tracks.
- **System 1** places the robot in the dock (any lateral position, as the brief allows) and chooses where it parks, its drive duty and each mechanism's timing (local search, as in the mobile world).
- **Assumed, because the brief does not give them**, each a constant in the module:
  - the kit's gearmotor (12 V, about 1.2 N·m stall, 160 rpm);
  - the piston (3/4 in bore, 4 in stroke) and the 1 L tank;
  - the mousetrap's spring;
  - a rubber band's stiffness;
  - the masses of the arena's pieces;
  - the magnet's and the clip's holding forces;
  - that the flagpole turns with the fortress.

**Prediction.**
- **The stock holds activate and mermaids (2 of 7).** It drives out and pushes one merfolk over with its bar, starting 10 in to one side. It cannot reach the others: it has no mechanism for them.
- **System 2 reaches each of the six tasks on some seed.** Gold and jewels and the heart come last: each needs a fold-out mechanism with several parameters placed within an inch.
- **The match is the hard group.** Each task wants the robot somewhere different:
  - the troughs want its centre over the mounds' line, 14 in out;
  - the flag wants it near the fortress;
  - the heart wants it at the chest and then back in the dock.

  The peel blade left over the fortress at 40 s voids the cannonballs. Four pistons' worth of mechanisms compete for two pistons. At most one seed in three reaches the match.
- **A solve takes about a minute per task** (40 s episodes at a 2 ms step take about a second, plus the search). A campaign seed takes hours, not minutes, so each campaign is asked for before it runs.

**Building the world: what reachability exposed** (2026-11-03, before any campaign; System 1 on hand-made bodies, never kept as results).
Each task was checked for one body that achieves it, so that no group measures an impossible goal. Each failure that came from the model, not the design, was fixed:
- **The piston booms swung the wrong way.** The hinge's range had the wrong sign, so the flag boom swung down through the deck and the hook boom swung back over the robot. Now the stroke is q ∈ [0, θ0 − θ1].
- **The booms sat near toggle.** With the cylinder's pin straight behind the pivot, a boom folded at 175° moved 0.5 rad for 0.2 in of rod. The pin now sits 90° beyond the middle of the swing, and a new law refuses a swing whose rod meets the boom below 20°.
- **The flag boom's fold is a design parameter** (`fold_deg`, 100–175°).
- **The needle valve rang at the 2 ms step.** As a damper on the 70 g rod it has a 0.1 ms time constant. Acting through the closed loop, that made the boom chatter and spiked the clip to 6 N. It is now c (dl/dq)² on the boom's hinge: the same power, integrated implicitly. The swing is smooth at 0.28 m/s of rod, and the clip carries 0.04 N.
- **A clip or magnet now lets go on shear or pull, not on the push** that presses the flag onto its mount or onto the pole: a face takes that by contact.
- **A timed flag boom catches only when it arrives with a face square.** That is about 0.5 s in each 5 s quarter-turn, and it cannot carry over to other speeds. The brief allows bought sensors, so a new part, `sensed_boom`, fires the boom a delay after a $15 long-range IR sensor sees a face square. It fixes the flag at 40 points on 16 of 36 settings on train, 9 of 36 at 2 rpm and 2 of 36 at 4.5 rpm.
- **The peel chute could not have scored:**
  - its side troughs were fixed at 25°, under the 27° friction angle of 0.5;
  - its blade was mirrored for a clockwise top;
  - its 3.2 in mouth dropped coins on the trough's wall;
  - its junction did not scale with the frame's width;
  - the deck had no slot for the trough;
  - a slot across the floor caught coins' rims.

  Now:
  - the side slope and the trough width are parameters;
  - the blade leads at −y and crosses the rim over the trough;
  - a cross trough and a funnel run out to the side troughs;
  - the deck is slotted;
  - the sieve is two rails along the trough, below a V floor that centres each piece;
  - the outer trough folds 15° forward of vertical, so it falls forward even when short.
- **No sliding route fits at a friction of 0.5.** The fortress top is 11.5 in above the mounds, and the route from the rim to a mound's border is about 25 in. So the chute is acrylic-lined ($8 bought) and the pieces are plastic, about 0.3 between them; the top (0.4), mounds and floor are unchanged. With that, a legal 16 × 9.3 in body scored 99 and 79 points on two seeds, putting 7–9 coins on the gold mound. Its jewels, short cylinders, roll away when they tip onto their sides.
- **Two mousetraps, not four.** The brief gives two; `KIT` had four.
- **Each task reached by some body:**

  | task | the body that reached it |
  |---|---|
  | activate | the stock |
  | mermaids | a 24 in bar at 5.5 in, or the trap arms |
  | cannon | the gate chute (27 points) |
  | flag | the sensed boom (40) |
  | heart | the hook boom, 1 of 16 random bodies (61) |
  | gold and jewels | the peel chute (99 once) |

**Open before any campaign.**
- **The match may be impossible as the kit stands.** The cannon, the flag boom and the heart each need a piston, and the kit has two. The brief says only "a set of pneumatic and electrical actuators"; two pistons is this world's assumption. The other flag, the band mast, has not yet fixed the flag on any of 64 bodies and controls tried: at full extension it is rigid, and the pole's corners knock the flag. Either the kit count or a flag mechanism that needs no piston decides whether the match group is valid.

## D-074 The ME 2110 match is judged on points, not on all six tasks (judge: world_me2110)

**Why.** The brief gives two pistons. The cannon, the flag boom and the heart hook each need one, and the band mast, the one flag that needs none, has not yet fixed a flag (D-073). So "all six tasks in one match" may be impossible, and a group no body can achieve measures nothing. A competition is won on points anyway. The six task groups stay as they are; only the match changes:
- it is achieved when one 40 s run scores at least half the points the six tasks offer (174 of 347) and breaks no rule;
- its score is the run's points over 347 (plus progress shaping, as before), so a higher-scoring match is always better;
- each run's points are in its detail; failing, the mechanism names the tasks that scored nothing.

**Prediction** (before the change):
- the stock still fails the match: it scores 11 + 5 = 16 points;
- the six task groups' verdicts do not move (the change touches only the match);
- in a campaign, System 2's best match will come from a body that gives up one piston task, most likely the heart (61 points but a return trip), for the cannon and the flag; the treasure's 175 decides who passes, so a body without a working peel chute does not reach 174.

**Measured** (System 1 on the stock, before any campaign). The stock fails the match on every variant: 11 points on train (it parks back at x = -6.6 in, short of the merfolk) and 16 on both tests (11 + 5 for a merfolk moved). The prediction said 16; System 1's 10-sample search on train settled on a parking spot that misses the merfolk. The six task groups are untouched.
- **No hand-written aim** (before any campaign). System 1's flag and treasure searches had started from a parking spot computed from the body's geometry (where the raised flag or the chute's tip lands). That was a designer's help, so it is gone: System 1 starts from the same default for every body, and finding the spot is the harness's job.

## D-075 ME 2110 parts cannot share space (judge: world_me2110)

**Why.** A physics review before the first campaign found that the robot's parts never meet. The robot class has contype 2 and conaffinity 1, so robot geoms never collide with one another. Most fixed mechanisms (the chute, the push bar, the sensor post, the tray) are geoms on the one robot body anyway, and MuJoCo never collides geoms on the same body. A push bar could stand inside the cannon chute, or a boom swing through a trough, and `check` would pass it. Once System 2 starts stacking mechanisms onto one base, every match point would be suspect. The campaign was stopped at iteration 1 (no result kept) and restarted after this change. The law:
- every robot geom is tagged with its mechanism (frame, drive, mermaid, cannon, flag, heart, treasure; the flag plate rides its mount);
- each mechanism is swept from stowed to deployed in 12 steps, with the others all stowed and then all deployed. Piston loops are closed analytically: the cylinder turns and the rod slides to keep the pin on the boom;
- two mechanisms' parts may not overlap by more than 0.1 in anywhere;
- a mechanism is fastened to the frame, so against the frame only the growth beyond its mounted overlap counts. A boom's pivot sits on the deck; a boom swinging down through the deck does not;
- a piston's cylinder and rod may pass through the frame's plates (a slot is cut for them), never through another mechanism;
- this is measured with `mj_geomDistance` on the compiled model, at 10–40 ms per body.

The review also found two places where the model itself broke the law, through no choice of System 2's (world tier, in the same commit because the law exposed them):
- the mousetrap's base was modelled inside the swinging arm and rotated through the frame's corner post. It is now screwed to the frame; only the bail swings;
- the peel chute's side troughs stood folded with their walls toward the frame, 0.79 in through the deck. They now hinge 1.2 in outboard (`SIDE_OUT`), and the cross trough and funnel run out to them. `envelope` now counts the folded side troughs in the robot's width; it had left them out.

**Prediction** (before the measurements below):
- the stock and every hand-checked body that fits the box pass the law;
- planted clashes are refused with both parts named: a push bar at the cannon's height, trap arms that swing through a side trough, and a hook boom whose tray meets the peel chute;
- the peel chute's reachability body, honestly 13.2 in wide folded, no longer fits the box. An 8 in frame does fit, and there it still sorts.

**Measured.** All three predictions held:
- **Hand-checked bodies:** the stock, the trap arms, the cannon, the sensed boom and the peel chute body pass the law.
- **Planted clashes:**
  - push bar against the cannon chute's mount: 0.90 in;
  - trap arms against a side trough: 0.22 in;
  - hook tray against the peel chute: 0.38 in.
- **Peel chute:** on an 8 in frame (the test body's width, 9.3 to 8 in) it scores 99 points (9 coins) with the same controls.
- **Still idealised (noted, not changed):**
  - parts of different mechanisms still do not collide in the dynamic sim. The sweep covers each mechanism's designed path, not a boom knocked off it;
  - latches release at exact times;
  - the magnet welds within 25° and 0.12 in.

**Campaign under D-075** (three System 2 seeds, 10 iterations each, 4 workers each; `results/evolve/me2110_s2_s{0,1,2}.json`):
- **Final bodies:** every seed held 2 of 7 train groups (activate and mermaids) from iteration 1 to the end. Each final body is the stock with a wider, higher push bar (22–23.5 in, at 6.5–7.5 in). The best match on every seed was 35 of 347 points.
- **Test:** seed 0 held both groups on test; seeds 1 and 2 held activate only (the merfolk on the test placements). Test tasks held: 4, 3 and 3 of 14, against the stock's 2.
- **Proposals:** 84 in total. 34 were refused before trial: at least 23 for the fold-up box and 7 for the new interference law. 46 were rejected after trial; 4 were kept-eligible. All 4 kept-eligible proposals were the push bar.
- **Cost:** 10 System 2 calls per seed, 52–68 min wall time each.
- **D-074's prediction failed.** No body gave up a piston task to score the cannon and flag. No piston or chute mechanism ever held.
- **Why** (from the prompts and answers, not a guess):
  - The first, free win (the 22–24 in bar) fills the box's 24 in side. Every catalogue mechanism mounts on the deck's front edge and needs room ahead of the frame, so none fits. Seed 1 wrote this itself in iteration 5: "No reach above about 0.8 fits".
  - The keep rule never trades a held group (the merfolk's 24 points) for room to try the treasure (175).
  - Further limits on the harness and the brief, which are the levers to test next, one at a time:
    - each refusal costs one of three proposals in an iteration;
    - a swap that scores some points but does not pass is thrown away rather than tuned;
    - the brief gives computed facts only for parts on the body, never the feasible ranges of a candidate part (the hook boom passes only at drops of 32° or less);
    - the brief never says where our cannon port is.
  - The match has no verified gold body under D-075 (lever 6), so it is not yet shown that 174 points is reachable with two pistons inside the box.

## D-076 The gripper's slide rides real rods; fixed parts are mounted; held objects stay outside the jaws (judge: mechanism, world_gripper)

**Why.** A render of b020 showed three things: an oversized sector gear, rods that seemed to pass through the fixed jaw, and a carriage that looked unsupported. A physics review then measured what the laws never checked:
- **Parts on the gripper's own body were never checked against each other.** The rods, clamp, bracket, frame, servo and pads all sit on that body. No fixed part had to be mounted; the short slide's clamp bracket stood 4.7 mm off the frame, so its rods hung from nothing.
- **The carriage rode an ideal slide.** It was rigid in its other five directions, so the 3 mm rods never bent.
- **The rack and gear were a joint equality.** It carried the teeth's tangential force, but never pushed the rack off the gear.
- **The interference sweep sampled nine jaw positions.** That is 11 mm of travel per step on the long slide.
- **Contact penetration was measured, never enforced.**

The long slide reached 20 of 21 groups (D-066) on exactly this idealisation.

**Change.**
- **Rods (simulator, `sim/jaws.py`).** The carriage gets four joints on its rods: across them in y and z, yaw and pitch. `RodSprings` loads them each step with two cantilevers' stiffness at the carriage's current overhang a from the clamp: 2EI/a³ [[12, -6a], [-6a, 4a²]] per plane, with EI = 0.795 N·m² for 3 mm steel. The teeth push the rack off the gear with tan 20° of the force the coupling carries. An armature of k·dt² keeps each mode stable, as for the finger's flex (D-054).
  - An attempt is unphysical if, held for 20 ms, the rack lifts more than half a module off the gear (the teeth skip).
  - It is also unphysical if a bushing carries more than 52.9 N. No LM3UU datasheet gives a rating; this is the static rating of the 3 mm miniature LMS3UU (37.4–52.9 N static, 48.9–56.1 N dynamic, motion.com). The bushing load is each rod's share of the force, plus the couple across its 10 mm length.
- **Fixed parts (law, `mechanism/gripper.py`).** Parts on the gripper's body may share material only where one is fastened to another:
  - a rod inside its clamp's bore;
  - a pad seated on the finger it wraps, at most 2 mm;
  - a printed bracket screwed to the frame or servo, at most 1 mm;
  - otherwise only touching faces.

  Every fixed part must reach the frame through parts that touch it, within 0.5 mm.
- **Brackets (`sim/jaws.py`).** Both slides' clamp brackets now end on the frame mesh's face over their whole footprint, found by casting rays at the mesh. They had used hand-set faces: one 4.7 mm short, one 1.3 mm in, and wrong for any other gear size.
- **Sweep.** The jaw's range starts at nine steps. A step is halved, up to four times, while some pair's clearances at its two ends add to less than the distance a point moves between them. Pairs in contact at both ends, at a seat, are left alone.
- **Held objects (executor).** In grasp scenes, an attempt in which the robot stays more than 1 mm inside the object for 0.2 s is unphysical. Panel worlds keep their contact.
- **Brief.** For a slide, each opening states:
  - the rods' overhang;
  - how far the squeeze lifts the rack, under the teeth's force and the squeeze's couple at the pad, worst sign;
  - each bushing's load against its rating.

  The same facts appear for candidate slides in the part facts.
- **The lower arm is not added to the static sweep.** It is no parent of the jaw's bodies, so the simulator's contacts already stop a jaw that meets it, in the poses the arm actually takes. Requiring clearance over the whole wrist-flex range refused an 89 mm four-bar for poses System 1 never uses.

**Prediction** (written with the change, before the full measurements below):
- the stock, four-bar and revolute bodies are unchanged;
- the long slide that reached 20/21 fails most picks on bushing load or tooth skip;
- the short slide's clamp, now mounted, passes the laws;
- held objects already sit under 1 mm, so the sinking guard changes no verdict.

**Measured.**
- **Long slide (`sector_radius_mm` 46, gripper_s2depth_s0's final pads):** 0 of 42 train picks. 20 failed on bushing load (59–174 N against 52.9) and 10 on tooth skip (the rack lifted 0.6–1.4 mm at 63–97 mm of overhang, against a 0.5 mm limit).
  - The same body with near-rigid rods (EI × 1000, the checks off) held 36 of 42. So the new joints change nothing else: the bending is the difference.
- **Statics:** the brief says 153 N per bushing at the long slide's 35 mm opening. A 40-tooth short slide sees 299 N at 25 mm. No slide in the catalogue's ranges stays within the bushing's rating: one bushing per rod cannot carry a squeeze applied 30–40 mm off the rods. A slide now needs a better guide, such as two bushings per rod or a miniature rail; that is a part for System 2 to propose as an idea, not one added here.
- **Contact:** the stock and long-slide picks sink 0.13–0.41 mm into the object (95th percentile), so the guard changes no current verdict.
- **Brackets:** both slides' brackets now seat 0.3–0.7 mm into the frame face for every gear size tried (17–60 teeth, sector 30–50 mm).
- **Laws:**
  - a 112-tooth pinion's clamp, bracket and rods are now refused as mounted on nothing; before, the bracket was drawn through the pinion;
  - a 50 mm sector and a 60-tooth pinion still meet the rolled wrist;
  - sweeps take 13–27 s per body, cached.
- **Not modelled** (recorded):
  - the overhung load of a large gear on the servo's spline: Feetech publishes no radial rating for the STS3215;
  - tooth bending (D-059);
  - belt or bushing friction.

**Campaign** (System 2, three seeds, 14 iterations, 4 workers, `gripper_s2rods_s{0,1,2}`; results in `results/evolve/`):

| Seed | Train groups | Test tasks (final vs stock) | Final pads |
|---|---|---|---|
| 0 | 19/21 (38/42 tasks) | 18 vs 15 | angle 31°, TPU 2 mm, reach 17 mm, width 12 mm |
| 1 | 17/21 (34/42) | 17 vs 16 | angle 30°, TPU 1.5 mm, reach 0, width 12 mm |
| 2 | 18/21 (38/42) | 16 vs 15 | angle 32.7°, TPU 1.5 mm, reach 8 mm, width 9.05 mm |

- **Every seed kept the stock revolute jaw and changed only the pads.** The best, 19/21, is one group below the long slide's 20/21 under the idealised slide (D-066), now reached with a mechanism that passes every law.
- **Proposals:** 144 in all. 9 kept, 77 rejected, 38 refused before trial and 17 recorded as ideas.
  - The slides' refusals were interference, for example the pinion against the rolled wrist or one pad shoe against the other.
  - The slides that were trialled lost most held groups: each squeeze dents a bushing or skips the rack.
- **System 2 named the fix this entry predicted, without being told it.** Seeds 0, 1 and 2 each proposed a braced or bridged long slide (about $10.60, 300 machine min). Seed 2 added a rack-preload roller.
- **It converged on one further mechanism, over and over:** a moving pad that stays parallel to the fixed one as the jaw turns. On the three seeds it appeared as:
  - a flexure-mounted or pivoting self-aligning pad (4 times);
  - a link-driven or tie-rod levelled pad (3);
  - a convex rocker pad (3).
- **None of these ideas could be tried:** an idea is recorded, not built, because the catalogue has no part for it. This is the case for the design-creativity work: ideas built from real links, joints and guides rather than catalogue entries.
## D-077 The mobile robot's parts cannot pass into each other; the talking robot is checked as built (judge: world_mobile, world_phonology)

**Why.** After D-076 closed the gripper, the request was that every world's physics be resolved. Two worlds still had a gap.
- **Mobile.** The MuJoCo model (D-061) gave the robot's parts no contact with each other. A geometry that put a wheel through a rocker link, or a roller into the tub, drove as if the parts were ghosts.
- **Talking robot.** Its CAD sweep (`rig_cad.sweep`) found floating, loose and unjoined overlapping parts, but only the pictures used it; `check` never ran it. The sweep itself was wrong in one place: the tip blade (from the roof) and the tongue pad (from the floor) close the same section for /t/, and the sweep called their meeting a fault and their parting "loose".

**Change.**
- **Mobile (`sim/mobile_model.py`, `worlds/mobile.py`).**
  - The robot's wheeled parts collide with each other in the simulator, through a separate contact class. A track's rollers overlap by design, so tracks keep the old classes.
  - Before trial, `self_clashes` builds the robot off the ground at rest, and with each rocker and bogie at the ends of its range. `check` refuses a body whose parts pass more than 0.5 mm into each other, naming the parts, the depth and the pose.
  - Rollers and linkage geoms are named so the refusal can say which parts.
  - **Two model faults found by the law, fixed.**
    - **The rocker-bogie was rigid.** Its joint ranges were written as radians (±0.6 and ±0.7) in a model that reads degrees, so since D-061 the rocker could turn only ±0.6° and each bogie ±0.7°. They are now ±34° and ±40°, as the catalogue says ("the bogie can tilt about 40 degrees").
    - **The links ran through the wheels.** The rocker and bogie links lay in the wheels' own plane, so a 6 cm wheel's disc held its rocker's pivot. A real rocker-bogie carries its wheels on stub axles beside the links, so the links now run 6 mm outboard of the wheels.
- **Talking robot (`worlds/phonology.py`, `worlds/rig_cad.py`, `design/language.py`).**
  - `check` runs the assembly sweep once the linkage laws pass, and reports every floating, loose or unjoined overlapping part as "assembly: …", with the control and value at which it happens. About 50 s, cached per body.
  - A new joint kind, `meets`, covers two soft parts that may press on each other at some postures and be apart at others. The tip blade and its stem each meet the tongue pad.
- **Recorded, not modelled:**
  - **Tracks:** a dense row of driven rollers stands in for an inextensible belt (D-061). That is fair for the step, trench and stairs, where the belt's own sag does not decide the verdict.
  - **Sonar cone:** the HC-SR04's 15° cone already covers the corridor's edges from 0.76 m, inside its 2 m range, so the cone does not bind any route.
  - **Motor heating:** neither the N20 nor the JGA25 publishes thermal constants, so the routes, all under 90 s, are not judged on heat.
  - **Talking robot actuation force:** closing a section takes about 4–5 N (≈1.1·E·L·t³/R² for the silicone tract, E ≈ 0.6 MPa). An MG90S on a 9 mm crank gives about 24 N, so force does not bind.

**Prediction** (written with the change; the mecanum and rocker-bogie corners had already been probed when the law was written, so only the kept-body lines are a true prediction):
- the stock bodies of both worlds pass;
- every kept mobile body (all tracked) and every kept talking-robot body (s2sounds, s2link, three seeds each) passes, so no result moves;
- in mobile, catalogue corners with big wheels on short linkages are refused: a large mecanum wheel's rollers reach into the tub, and large rocker-bogie wheels on a short bogie reach into each other.

**Measured.**
- The stock bodies pass. All six kept mobile bodies are tracked and pass; all six kept talking-robot bodies pass the assembly sweep.
- **Mobile refusals at catalogue corners:**
  - mecanum, wheel radius 4 cm: each wheel's rollers pass 1.8 mm into the tub; 3 cm passes;
  - rocker-bogie, wheel 6 cm and bogie 10 cm: neighbouring wheels pass 20.0 mm into each other, at rest and with the rocker swung;
  - rocker-bogie with wheels 6/14, 6/18, 5/10, 4/12 and 3/8 (wheel cm / bogie cm): passes.
  - Before the links moved outboard, wheel 6 / bogie 14 (the tests' distractor), 6/18 and 5/10 were refused for a link inside a wheel, 7.8–15.3 mm. That refusal was the model's fault, not the body's.
- No kept result is judged differently, so no campaign is re-run.

## D-078 A mechanism is a graph of parts; the gripper's jaws are its seed modules (world: physics; judge: mechanism)

**Why.** On every seed of the D-076 campaign, System 2 asked for mechanisms that nothing could build. It proposed a braced or bridged slide three times, and a moving pad that stays parallel to the fixed one ten times. Both were recorded as ideas and never tried.

The theory says competence grows by composing typed primitives and promoting recurring compositions into a library (theory.md §3.1–3.5). That holds today for control programs (`compose/`) but not for bodies. Each gripper jaw was described four times over, by hand:
- its MJCF builder (`jaws.four_bar`, `jaws.slide`);
- its kinematics (`design/fingers.py`'s branches per jaw type);
- the name tables the laws and the CAD read (`HAND`, `BODIES`, `_assembly`, `_allowed_mm`);
- its bill of materials (`worlds/gripper.py`).

Nothing System 2 wrote could enter any of these.

**Change.** This is the first step (C0) of the plan for compositional mechanisms. It changes no behaviour.
- **`sea/mech/graph.py`: a mechanism is plain data.**
  - **Bodies** carry parts and joints. A body's inertia follows a declared rule (`Inertial`: slender, disc or lumped). The hand builders' approximations are now visible data on the seeds, not constants inside code.
  - **Parts** are things with a function: a link plate, a hub, a web, a gear, the rack's teeth, a rod, a clamp, a bracket that seats on the frame, the finger's blade, and bought parts. Each part declares its material, its role to the laws (structure, pin, rod, clamp, bracket, finger) and the printed piece it belongs to.
  - **Constraints:** a loop (a closing pin), a mesh (teeth as a joint equality), and a guide (a carriage on two clamped rods that bend, D-076).
- **`sea/mech/build.py`: one builder compiles any graph.** It takes mass from volume and density (bought parts from their datasheets), seats brackets on the frame mesh, compiles loops as pins and meshes as equalities, and gives a guide its four bending joints.
- **`sea/mech/seed/gripper.py`: the catalogue's four jaws are now graphs.** The four jaws are revolute, parallelogram, prismatic and long slide. The robot's model supplies the gripper's site data: the servo's range and the jaw's orientation. `jaws.four_bar` and `jaws.slide` are deleted, and `jaws.py` keeps only the SO-101's measurements and the rods' physics.
- **The laws read the graph (C0b).**
  - The bodies to sweep, which fixed parts may share material (a rod in a clamp's piece; a bracket or clamp on the stock frame) and the transmission angle all come from the graph.
  - The transmission angle is measured at every loop's pin in the simulator, not computed from the parallelogram's hard-coded follower direction. A loop System 2 builds is then judged by the same law.

**Prediction** (written before C0b and C0c were measured):
- the graph compiles the same model as the deleted builders, at the ends and middle of every parameter range: every body, joint, solid, constraint and mass within 1e-6 (relative and absolute);
- every law gives the same output on those 13 bodies;
- the full suite stays green with the same verdicts;
- no result moves, and no campaign is re-run.

**Measured.**
- **Model equality.** The graph compiles the same model as the deleted builders on all 13 configurations: the revolute jaw, and the parallelogram, prismatic and long slide each at four values spanning their ranges. Bodies, joints, solids (positions, sizes, meshes), equalities, excluded pairs, the rods' data and masses all agree within 1e-6 (`tests/mech/test_seed_equivalence.py`). Planting a PLA density 0.1 % too high fails it.
- **Law equality.** The laws reading the graph give the same output on all 13 bodies as they did reading name tables: the refusals (the 60-tooth pinion against the wrist, the 50 mm sector against the rolled wrist), the fastened pairs, the loose parts, the clashes and the carried gaps (`tests/mech/test_laws_equivalence.py`).
- **The measured transmission angle** at the parallelogram's loop pin equals the old formula's within 0.1°. A follower pin moved onto the closing axis is now caught as a real geometric fault: the links come within about 10° of toggle. The old test planted that fault by overriding the formula.
- **Two things are deferred, with reasons:**
  - **The catalogue jaws' bill of materials stays the hand estimate** (`worlds/gripper.py`: PLA grams by formula, screws and bearings the graph does not yet hold). Taking it from the graph's volumes would change prices, and so the keep rule, so it needs its own measured decision. Mechanisms System 2 builds (C1) take theirs from the graph from the start.
  - **The catalogue's kinematic facts stay `design/fingers.py`'s**, for the same reason. A built idea's facts come from the simulator.
- **Full suite:** 710 passed, 2 skipped, 1 xfailed (the 683 before, plus the 27 equivalence tests).

## D-079 System 2's ideas can be mechanisms: built from a module and a patch, closed, priced and judged (policy: prompt, schema; world: physics; judge: built jaws only)

**Why.** Under D-076, the mechanisms System 2 asked for were recorded and never tried: a braced slide three times, and a pad that stays parallel ten times. An idea was free text. The catalogue was the only thing that could be built, and the laws bound on variables that no catalogue part exposed (the slide's single bushing).

D-078 made every jaw a graph. This is the second step (C1): an idea can now carry a graph, and that graph is trialled exactly like a catalogue swap.

**Change.**
- **`Idea.mechanism`** (`evolve/core.py`) names a module, its parameters and a patch. A patch is a list of `add`/`set`/`remove` operations addressed by names, never by position (`mech/patch.py`), for example `bodies/jaw_follower/parts/jaw_follower_horn/length` or `loops/jaw_follower/point`.
  - A jaw built this way is `Part(type="built", mechanism=<graph JSON>)`.
  - Its key and description carry the graph.
- **Closure kinematics** (`mech/kinematics.py`). A general linkage does not move its joints in proportion to the servo, so the graph's free joints are solved at 61 servo angles across the range. The solver is least squares on the simulator's own equality residuals, stepping out from the closed jaw. The results are kept as tables that the model's `linked` joints follow.
  - A loop that cannot close somewhere in the range is reported, with the servo angle where it jams.
- **The finger's motion is measured from the compiled model** (`kinematics.finger_motion`) for built jaws: position, tilt and squeeze per newton-metre. The grasp skills read it in place of `design/fingers.py`'s per-type formulas. Catalogue jaws keep those formulas (D-078 deferral).
- **The world** (`worlds/gripper.py`):
  - `build_idea` builds the module, applies the patch, and refuses with the reason when a name or range does not apply;
  - `check` refuses a built jaw that does not compile, carries no finger or does not close, and only then runs the same laws as every jaw;
  - the bill of materials comes from the graph: PLA by volume, rods by length, bought parts by item, two MR63 bearings per pinned joint or loop, two M3 screws per clamp or bracket piece.
- **The loop** (`evolve/loop.py`) builds an idea that carries a mechanism with `world.build_idea`, scores it whole (move `idea`), and keeps it on the same rule as a swap. An idea without a mechanism is recorded as before. One the world cannot build is refused before the trial, with the reason.
- **The prompt** gains a "Mechanisms" section: the part kinds and fields, the patch grammar, and every module's graph at its parameters' midpoints. Rule 3 says that an idea with a mechanism is built and trialled.
- **`iteration.md` principle 8:** a law that binds must expose the variable it binds on, and no idea is ruled out before it is built and trialled.

**Prediction.**
- *Mechanics (measured while building, so a check rather than a forecast):* a catalogue jaw rebuilt through `build_idea` with no patch moves its finger as the catalogue's formula says; a patched four-bar closes and tilts its finger; a loop that cannot close is refused before the laws.
- *Campaign (forecast, not yet run):*
  - on at least 2 of 3 seeds, System 2 writes a mechanism that builds and reaches a trial;
  - at least one of those is refused by a law that names a variable the patch can change;
  - no claim on groups held.
- *Catalogue jaws:* the D-078 golden models and laws are unchanged; the full suite stays green.

**Measured (tests).**
- **Closure solver.** With the parallelogram's ratios removed, the solver recovers ±1 to 1e-6 and closes to 1e-8. The finger of a built parallelogram, and of a built prismatic jaw, matches the catalogue formula to the table's interpolation (under 10 µm, tilt under 0.01°) (`tests/mech/test_kinematics.py`).
- **A patched four-bar** (follower links shortened from 70 to 60 mm, pivot moved to match):
  - it closes to 7e-11 and its finger tilts +2.5°, +7.6° and +15.9° at a servo of 0.5, 1.0 and 1.5 rad, where the parallelogram's stays at 0;
  - its bill of materials is 34 g of PLA, 6 MR63 bearings and 2 M3 screws;
  - the laws refuse it for real reasons: the moved follower shares material with the wrist and the driver links, and its loop comes within 10° of toggle (`tests/mech/test_built_jaw.py`).
- **A loop pin placed 40 mm off** is refused with the one closure error, before the laws.
- **The loop on the toy world:**
  - an idea with a mechanism is built, trialled and kept;
  - one without a mechanism is recorded;
  - one naming a module that does not exist is refused with its name (`tests/test_evolve.py`).
- **Not yet built, next:**
  - C1b: a guide with several bushings per rod (the braced slide);
  - C1c: a pad mounted on any body (the levelled pad).
  
  Until they exist, the two ideas System 2 asked for most still cannot be written; the campaign waits for them.
- **Catalogue jaws unchanged:** their briefs are kept byte-for-byte (a parallelogram's face still reads "-0 deg"), and the D-078 golden models and laws pass.
- **Full suite:** 726 passed, 2 skipped, 1 xfailed (710 before, plus 16 new tests).

## D-080 A guide's rods can be held at both ends, and its carriage can ride several bushings per rod (world: physics; judge: built jaws only)

**Why.** Under D-076, System 2 asked three times for a "braced" or "bridged" long slide. Each time it named the same two changes:
- a printed bridge that clamps the rods' far ends, so they stop overhanging 94 mm;
- two bushings per rod, spaced to resist the squeeze's couple.

The bushing law and the tooth-skip law bound on exactly those variables, and no graph could set them: a `Guide` held one bushing per rod and cantilevered its rods (`mech/build.py` refused anything else). Under principle 8, the variable a law binds on has to be exposed.

**Change.**
- **`Guide.far_end`** (gripper x where a second clamp holds the rods' far ends; none means cantilevered) and **`Guide.bushings`** (each bushing's offset along the rods from the carriage's middle).
- **The rods as beams.** Each rod is a beam clamped at one end, or at both, with a node at every bushing. The rigid carriage imposes each bushing's deflection and slope. That is the D-076 model, in which a bushing's 10 mm length takes a couple, applied at every bushing.
  - The carriage's stiffness and each bushing's load (its force plus its couple over its length) come from the assembled beam.
  - With one bushing and cantilevered rods the matrices are D-076's, and the code keeps D-076's exact path, so catalogue slides are unchanged to the bit.
- **Checks on a built guide, before the laws:**
  - bushings at least a bushing's length apart;
  - a far end beyond the clamp, held by a clamp part there;
  - the carriage carrying two LM3UU per bushing position (so the price follows the physics);
  - in trial, a bushing past the rods' end leaves them.

**Prediction** (written before the code; static loads at the long slide's grip):
- catalogue slides: golden models, laws and every D-076 number unchanged;
- rods held at both ends, one bushing per rod: the rack's lift falls below the 0.5 mm tooth-skip limit across the travel, but each bushing still carries roughly D-076's 138 N, because the squeeze's couple across one 10 mm bushing dominates;
- held at both ends with two bushings per rod at ±12 mm: per-bushing load about halves and stays above the 52.9 N rating;
- about ±25 mm or wider is needed to get under the rating.

So System 2's bridged slide is predicted to fix the teeth but need wide bushing spacing (or a lower squeeze) to pass the bushing law. The prediction is falsifiable in both directions.

**Measured** (static, the long slide at sector radius 50 mm and the grip's torque, `tests/mech/test_guide.py`), as worst bushing load / rack lift at a servo of 0.6, 1.0 and 1.4 rad:

| rods | 0.6 rad | 1.0 rad | 1.4 rad |
|---|---|---|---|
| cantilevered, one bushing per rod (as the trial loads it) | 146 N / 0.1 mm | 153 N / 0.9 mm | 161 N / 2.3 mm |
| held at both ends, one bushing | 146 N / 0.2 mm | 153 N / 0.6 mm | 161 N / 1.2 mm |
| held at both ends, bushings ±12 mm | 71 N / 0.1 mm | 84 N / 0.3 mm | 87 N / 0.7 mm |
| held at both ends, bushings ±25 mm | 53 N / 0.0 mm | 64 N / 0.1 mm | 65 N / 0.4 mm |

Against the prediction:
- **Unchanged as predicted:**
  - the catalogue slides emit no beam data and keep D-076's exact code path;
  - with both ends held and one bushing per rod, each bushing still carries the couple across its own length (146–161 N against the 52.9 N rating);
  - two bushings at ±12 mm roughly halve that and stay over the rating.
- **Wrong in two places:**
  - *Holding the far ends alone does not stop the teeth skipping.* The lift only halves (2.3 to 1.2 mm at 1.4 rad), because one bushing per rod gives the carriage little resistance to turning, and the teeth sit 20 mm or more from the carriage's middle. The teeth hold only with two bushings per rod as well (0.7 mm at ±12 mm, 0.4 mm at ±25 mm).
  - *±25 mm is not wide enough.* At 53–65 N it is still over the rating, so a wider spacing or a lower squeeze is needed.
  
  So the bridged slide System 2 described (both ends held, about 30 mm between bushings) is predicted to keep its teeth, with its bushings near or over their rating. A trial will tell.
- **The catalogue's static fact reads 8–14 % low** (138 N against 146–161 N): `rack_lift_mm` leaves out the teeth's separating force times its lever to the carriage's middle, which the trial's `RodSprings` includes. Facts for built guides use the full beam (`guide_static`). The catalogue's fact text is kept as it was, because changing it changes the prompt. This is a known understatement, to fix in its own change.
- **The trial's rods apply the assembled beam** (checked in the y plane, where the teeth add nothing): `RodSprings` forces equal −K·u to 1e-9.
- **A built guide is refused before the laws** if its bushings overlap, if it carries the wrong number of LM3UU, or if no clamp part holds a declared far end.
- **Full suite:** 731 passed, 2 skipped, 1 xfailed (726 before, plus 5 new tests).

## D-081 Pads can mount on any body of a mechanism (world: physics; judge: built jaws only)

**Why.** Ten of System 2's D-076 ideas asked for a moving pad that stays parallel to the fixed one. The most concrete was the "levelled moving pad": a pad carrier pinned at the stock fingertip, with a levelling link to the gripper that makes a parallelogram. Pads could sit only on the body carrying the blade, so a graph could build the carrier but could not put the pad on it.

**Change.**
- **A `PadMount` part** marks the body the pads ride on, with the same offset convention as the blade (where the jaw's frame at q = 0 sits in the body's frame). It has no solid and no mass.
- **The finger is the body the pads ride on:** the mount's body if there is one, otherwise the blade's.
  - Its motion is measured as before (`kinematics.finger_motion`).
  - The grasp skills, facts and pad placement follow it.
- **Refused before the laws:**
  - a mount whose body is not axis-aligned with the jaw at q = 0 (the pads are drawn in the jaw's axes);
  - a body with a mount and no pads.
- **A graph that keeps the stock jaw's blade** (no `remove_stock_mesh` of the jaw) has a finger. Before this, an edit of the empty revolute module was refused as "no finger".

**Prediction** (written before the code):
- catalogue jaws are unchanged (no graph has a mount);
- a levelling parallelogram built from the revolute module keeps the mount's body within 0.1° of its closed orientation across the servo's range, while the stock jaw turns 100° or more, and its pads ride on the carrier;
- that fixture is likely refused by the interference law, because its pads sit on the blade's face while the blade turns away. That is a real fault the design would have to answer, not a bug.

**Measured** (`tests/mech/test_pad_mount.py`). The levelling parallelogram built from the revolute module (a 60 mm link from a pin on the gripper, offset as the parallelogram's follower is, to a carrier pinned on the stock jaw):
- **Kinematics, as predicted:**
  - it closes to 8e-13 over the servo's whole range (-0.17 to 1.75 rad), while the stock jaw turns 110°;
  - the carrier keeps its orientation to 0.000°;
  - the pads are placed on the carrier.
- **Refused by the laws, as predicted, but for a different fault than predicted:**
  - the carrier's hub and the levelling link pass through the stock jaw's mesh (12 × 8 × 12 mm and 24 mm of shared material);
  - the link's body sits 0.6 mm off the pin that should carry it.

  These are the faults a real version would have to design around. The laws read the graph and do not trust the simulator's contact exclusions.
- **Refused before the laws:**
  - the same body with no pads;
  - a carrier turned 30° from the jaw's axes.
- **Full suite:** 733 passed, 2 skipped, 1 xfailed (731 before, plus 2 new tests).

## D-082 Two laws that refused pads the jaw could build, a failure text that hid the part that jammed, and a run that died on one slow answer (judge: mechanism; world: physics, gripper; policy: loop)

**Why.** The first C1 campaign seed (`runs/evolve/gripper_c1_s0`) crashed at iteration 5. A System 2 answer that wrote a mechanism took more than the CLI's 900 s; the calls had grown from 300 to 625 s and from 30k to 60k output tokens. Reading that seed's record turned up three faults in the harness itself.
- **The stop rule.** Under D-061, fingers that meet only over the closed end of the range are the jaw's stop, but only if they overlap from the very first sample at −0.17 rad. Thin pieces of a pad's shoe slide past each other in the swept geometry and stop overlapping by −0.17 rad, so they "start" at −0.16 rad. Their 0.3 mm overlaps, all below the angle where the fingers first meet (about −0.01 rad, where the jaw stops), were refused as interference.
  - Counting refusals whose first law is this one: 2 of 33 proposals on D-076 seed 0, **16 of 28 on seed 1**, 5 of 27 on seed 2, and 3 of 9 in the crashed seed.
  - These are proposals ruled out without evidence. The pad lever that carried D-076 was mostly closed on seed 1.
- **Grooved shoes.** A V-grooved pad's shoe is built for contact as two convex halves, one each side of the groove. Neither half includes the middle of the shoe's back, so a wedge of the back is missing where the screws sit.
  - Every grooved pad's two screws then "touch nothing" (2.2 mm off), and the mounted law refuses the pad.
  - The shoe's mass is understated by that wedge too.
- **The failure text.** The trialled built slide (the outboard-rack double-clamped slide) failed 8 tasks with "fixed jaw: no contact; moving jaw: no contact". A replay shows the rack on its carriage pressing the banana and jamming the jaw at 1.36 rad. A part other than the jaws on the object was counted but never named, so System 2 could not see what to move.

**Change.**
- *Judge (mechanism):* the jaw's stop is the most open angle at which a moving finger part meets a fixed finger part, over a range of overlap that runs from the closed end. A finger-on-finger overlap at angles no more open than that is unreachable, so it is the stop, not interference.
- *World (physics):* each half of a grooved shoe includes the profile's points on the groove's midplane, so the back is whole.
- *World (gripper):* a failure's detail names any other robot part pressing the object while the jaws close (`also on the object while closing: the rack 6 N`). The label is unchanged.
- *Policy (loop):* the CLI waits 1800 s and asks once more after a timeout. A call that still gives nothing usable leaves that iteration with no proposals instead of ending the run. Plan limits and billing errors still stop it.

**Prediction** (written before the two law changes; the loop and failure-text changes were already written and tested):
- the stop rule passes the refused pads above (the 30° pad with 12 mm reach, 2 mm groove) and changes no law output on the 13 catalogue jaws without pads (`tests/mech/test_laws_equivalence.py`);
- a real finger clash in mid-range is still refused;
- the whole grooved shoe reaches its screws (no loose screws on grooved pads) and gains a few percent of mass;
- golden verdicts are unchanged, since no golden body has a grooved pad (to check);
- the next campaign trials more pad bodies than D-076 did.

**Measured.**
- **The refused pads, re-checked on the body each was proposed for**: 16 proposals from D-076 and the crashed seed whose first refusal was a finger-on-finger overlap.
  - With the stop rule as first written (overlap below the stop angle is the stop): 9 pass, 6 are still refused and 1 no longer applies.
  - All 6 overlap by millimetres up to an open angle of +0.08 rad.
- **The prediction was wrong about the rule, so it was revised before any campaign.** A closing jaw stops at its fingers' first contact, the most open angle of any finger-on-finger overlap, so in a static sweep such an overlap is always past the stop. The simulator gives every finger pair that can meet a contact pair (D-060), so a trial stops the jaw where it would really stop. Where that is (a wide stop cannot grip a small object) is the trial's to measure.
  - The rule is therefore: a moving finger part on a fixed one is the jaw's stop, wherever they meet. The prediction's "a real finger clash in mid-range is still refused" is withdrawn.
  - Under this rule, **13 of the 16 pass**. 2 are refused by another law, the fixed pad sinking 2.4–2.5 mm into the gripper frame against its 2 mm seat (a pad cannot be printed into the finger it wraps; kept). 1 no longer applies. The 13 merged proposals refused the same way are not re-checked (no proposal record).
- **The grooved shoe:**
  - the 30° pad with 12 mm reach and a 2 mm groove now has no loose screw, and passes every law and the build check;
  - varying its parameters one at a time showed the groove alone caused the loose screws.
- **The failure text:** in the replay of the banana on the outboard-rack slide, `jaw_rack` is the only robot part on the object, for 40 samples, while the jaw holds at 1.36 rad; the detail now names it.
- **Full suite:** 735 passed, 2 skipped, 1 xfailed; golden verdicts and the catalogue jaws' law outputs unchanged.

## D-083 A mechanism is built together with the proposal's edits to other slots (policy: loop, prompt)

**Why.** The restarted campaign's first seed (`runs/evolve/gripper_c1b_s0_unbuilt`) wrote a mechanism in 13 proposals over 7 iterations, every one a long slide with its rods held at both ends and its bushings spread. None was built. Each came with an edit to the pads, and the loop built an idea only when the proposal had no edits (D-079), so all 13 were recorded as untested ideas. The campaign was stopped.

**Change.**
- An idea that carries a mechanism is built, then the proposal's edits to other slots are applied, and the whole is trialled.
- An edit to the slot the mechanism builds is refused as ambiguous.
- The prompt's rule 3 says so.

**Measured before this entry, so a check rather than a forecast.** Built now with their pad edits (no trial), the 13 meet the guide's checks and the laws instead of the record:
- 8 carry 2 LM3UU where their two bushing offsets need 4;
- 5 hit the wrist with the 50 mm sector's pinion (0.2 mm at the wrist rolled −2.74 rad, the same refusal the catalogue's 50 mm long slide gets);
- none reaches a trial as written.

**Prediction for the campaign:** seeing these refusals, System 2 adds the bushings or shrinks the sector, and on at least 2 of 3 seeds a mechanism it wrote reaches a trial.

**Open question for the user (no change made):** the interference law sweeps the wrist's whole roll range (±2.74 rad). Whether the robot ever rolls that far while gripping, and whether a roll limit should be a variable System 2 can set, decides whether the 50 mm sector is outlawed.

## D-084 The brief describes a built jaw (world: gripper), and what the first C1 campaign showed before it crashed

**Why.** The C1 campaign (`runs/evolve/gripper_c1c_s{0,1,2}`, D-079 to D-083) crashed on every seed on the iteration after a built jaw was kept: the brief named a jaw's kind from a table with no entry for `built`. The crashes came at iteration 13 on seeds 0 and 1 (of 0–13) and iteration 6 on seed 2, so no seed reached its test-split evaluation. The paths that run after a keep had no test.

**Change.** The brief names a built jaw by its mechanism's name. A test builds the brief for a built body, and the part facts, bill of materials and check were run on a kept one.

**What the campaign showed** (train only; no test split):

| seed | mechanisms written (long slide / revolute / parallelogram) | refused | trialled | kept | groups |
|---|---|---|---|---|---|
| 0 | 21 (14 / 6 / 1) | 14 | 7 | 1 | 16 → 19 |
| 1 | 25 (11 / 11 / 3) | 12 | 13 | 1 | 16 → 18 |
| 2, to iteration 6 | 9 (4 / 4 / 1) | 5 | 4 | 1 | 16 → 18 |

- **D-083's prediction is met.** On every seed a mechanism System 2 wrote reached a trial, and on every seed one was kept.
- **System 2 answered the refusals it saw.** It added the missing bushings and moved the bushings apart. For the wrist it first shrank the sector to 48 mm, then trimmed the sector's arc, which passed the laws and reached trial.
- **Every long slide that reached a trial lost most groups**, to the bushing and tooth-skip laws and to the rack jamming on the object. None was kept.
- **The three kept mechanisms are all one new variable.** Each is the stock hinged jaw with a `pad_mount` moved 0.5–3 mm: a pad whose face sits off its usual place on the finger.
  - Seed 0: a "standoff" 3 mm along the pad's own 33° axis, which it described as a printed spacer.
  - Seed 1: raised 1 mm.
  - Seed 2: dropped 3 mm.

  Each gained one task (the tuna can on seeds 1 and 2). Checked against the model:
  - the shoe is regenerated around the real finger in the moved pad frame (`sim/shoe.frames`, `_finger_hosts`), so it still wraps the finger (no gap, the same ~12 mm socket overlap as a stock-mounted pad);
  - its screws sit as they do on a stock-mounted pad (the drawn heads 2.0–2.4 mm off the finger in both cases);
  - the pad is printed as one piece and priced by its volume.

  So these are real, printable pads with a variable the catalogue never exposed (where the land sits on the finger), found through D-081's pad mount.
- **The wrist-roll refusal was examined (no change yet).** All 84 stock picks roll the wrist between −1.95 and 0 rad (train −1.95 to 0, test −1.69 to 0). The sector's 0.2–0.4 mm clash is at −2.74 rad, the joint's limit. The user's decision: the law's roll range is to be derived from the IK per body, and the sim limited to it (next).

## D-085 The wrist's roll is limited to what the body's grasps need, derived from the IK (judge: mechanism; world: physics, gripper)

**Why.** The interference law swept the wrist through its whole joint range (±2.74 rad), so every long slide with a large sector was refused for clipping the wrist by 0.2–0.4 mm at −2.74 rad (D-083, D-084). No grasp rolls that far: the stock picks stay within −1.95 to 0 rad. A real SO-101's roll can be limited in its servo's firmware. The user's decision: the roll range is to be derived from the IK, per body, and the simulator held to it, so the law checks exactly the poses a trial can reach.

**Change.**
- For each body, the world computes the roll its grasps need. For every task (train and test) it takes the grasp poses `motor.grasp` commands (above the object, at the grip, lifting), aligned with the object's roll direction, solves them with the IK from the home pose for the body's tool, and records the wrist roll.
- The range runs from home (roll 0) to the furthest roll needed, widened by `ROLL_MARGIN` = 0.2 rad for the tracking that moves the arm between those poses. It is clipped to the joint's limits.
- The design carries this range. The simulator sets the wrist roll joint's limits to it, so a trial cannot roll past what the law checked, and the interference law's sweep, which reads the model's joint range, follows it.

**Prediction** (written before the code):
- for the stock body the range is about −2.13 to +0.2 rad (IK −1.93 to 0, measured −1.95 to 0);
- golden verdicts are unchanged, since no pick rolls outside the range;
- the law's output changes only where a part met the wrist beyond the range: the catalogue's 50 mm long slide no longer meets the rolled wrist (the D-078 law equivalence test is re-recorded for that body, and the change is listed);
- the trimmed and full 50 mm sectors System 2 wrote in `gripper_c1c` pass the interference law, unless something else stops them.

**Measured.**
- **The stock body's range is −2.13 to +0.20 rad**, as predicted. Each body's range comes from its own grasp poses: a 33° pad with 12 mm reach gives −2.16. The scenes take 6 s to build once per process, then 0.14 s per body.
- **The simulator's wrist joint takes that range**, and the interference law's sweep follows it, because it reads the model's joint range.
- **Of the D-078 law goldens, two change, both as predicted:** the catalogue's 50 mm long slide and its 60-tooth pinion jaw no longer meet the rolled wrist. Both met it only at −2.74 rad. The D-078 gates (graph against the deleted builders) now run at the joint's whole range, under a fixture, so their goldens stand unchanged and they still test what they were for. `tests/mech/test_roll_range.py` tests the limit.
- **The five `gripper_c1c` mechanisms refused for the wrist, re-checked:**
  - 2 now pass;
  - 1 is refused by another fault (its clamp bracket and the pinion share 7 mm);
  - 2 still meet the wrist inside their range (0.7 × 0.3 × 0.2 mm at −2.13 rad, and 3.1 × 3.4 × 1.5 mm at −2.15 rad, within the margin), so they stay refused.

  The prediction "pass unless something else stops them" holds for 3 of 5. The other two meet the wrist within what the grasps need plus the margin.
- **`motor.grasp`'s poses are now one function** (`grasp_poses`), used by the grasp and by the range.
- **Full suite:** 741 passed, 2 skipped, 1 xfailed. Two D-059/D-064 tests that asserted the wrist refusal now check both sides: the body passes within the roll the grasps need, and is refused over the joint's whole range.

## D-086 System 2 can question the harness while it proposes (policy: loop, prompt, backends)

**Why.** In `gripper_c1c`, 32 of System 2's 94 proposals (34%) were refused before any trial: an overloaded bushing, a sector meeting the wrist, a bracket against a pinion, a pad sunk into the frame. Each refusal reached System 2 one iteration later, at the cost of one of its three proposal slots. Only 0.75 of its mechanisms per call reached a trial. The wrist question (D-085) was the plainest case: nothing let it ask what roll a body's grasps need before it drew a sector. The user's decision: give System 2 tools to question the harness mid-iteration, so that questions like this are part of how it designs.

**Change.**
- **A read-only MCP server**, `sea` (`evolve/tools.py`), speaks the subset of the protocol `claude -p` uses: newline-delimited JSON-RPC over stdio. It adds no dependency. Its four tools each take a draft written as a `Proposal`:

  | Tool | What it returns |
  |---|---|
  | `check` | the loop's own screen, word for word: "would trial" with the cost, or every refusal |
  | `measure` | the world's brief of the drafted body and its cost. For the gripper, also the wrist roll each train task's grasp needs, the range the simulator holds the wrist to, and how far a built mechanism's loops are from closing |
  | `module` | a module's graph at given parameters |
  | `simulate` | one train task on the drafted body (test tasks are refused); one task is not a trial |

- **One path from proposal to trial.** The tools answer from the same code the loop runs: `loop.candidate` (the base, `build_idea`, edits, the ambiguous-slot refusal) and `loop.screen` (the laws, the resources, the record), both lifted out of the loop unchanged.
- **The budget is in the context, not in the arguments.** The loop writes each iteration's context (`tools/iterNN.json`: the body, the promising bodies, the record of trialled bodies, the resources, the train task ids). The server enforces 40 calls per iteration, of which 3 may simulate, counted from the call log, so a retried CLI call does not reset it. Every call goes to `tool_calls.jsonl`; each iteration's record and the summary count them.
- **Flags.** `--safe-mode` drops every MCP server, ours included (a smoke test: the server was never started). With tools, the CLI therefore gets the isolation flags themselves: `--tools ""`, `--setting-sources ""`, `--disable-slash-commands`, `--strict-mcp-config`, and `--allowedTools` limited to the four `mcp__sea__` tools. In the smoke test the model saw no CLAUDE.md, memory or skill. Without tools, the command is unchanged, flag for flag.
- **The ablation switch** is `RunConfig.tools`, off by default, so earlier runs reproduce. When it is on, the prompt lists the tools and their budget, and says that only the trial of the answer keeps a body.
- **The judge key is unchanged** (ce7cf7da9f79): the tools change what System 2 can learn before it answers, not what is accepted.

**Prediction** (written before any campaign, against `gripper_c1c`'s 34% refused and 0.75 mechanisms trialled per call, its median call 480 s and 47k output tokens):
- with tools on, fewer proposals are refused before trial: under 20%;
- more mechanisms reach a trial per call: over 1.0;
- calls take longer and write more tokens; the time is measured, not bounded;
- no claim on groups.

**Measured** (the build):
- **The server answers on the gripper.** `check` takes 10–30 s, since it runs the interference sweep: a pad edit is refused for its known 2.4 mm overlap with the frame, and the 40 mm long slide would be trialled. `measure` takes 7 s. `simulate` takes 14 s per task; on the long slide it reports the rack lifting 1.0 mm off the gear on the gelatin box.
- **`tests/test_tools.py`:**
  - each draft's check equals the loop's refusal text;
  - the budget holds across a new server process, and arguments cannot raise it;
  - test tasks are refused;
  - the stdio protocol (with the client's `server/discover` probe answered as an unknown method);
  - a built jaw's wrist range (−2.13..+0.20 rad on the stock body);
  - the flags with and without tools;
  - the loop's per-iteration context and counts.
- **Full suite:** 747 passed, 2 skipped, 1 xfailed.
- **One live call** (Opus, effort high) on `gripper_c1c_s0`'s iteration 8, where all three of System 2's proposals had been refused before trial. It took 525 s (the c1c median is 480 s), 15 turns, 41k output tokens and 377k input tokens, mostly cache reads.
  - **Tool use:** 13 calls (5 `measure`, 2 `module`, 3 `simulate`, 3 `check`), 97 s of tool time, none over budget.
  - **The simulations changed its explanation.** The potted meat can failed at reach 20 mm (106 N on its top) and succeeded at reach 8 mm. System 2 wrote that this "overturns my iteration 6 explanation", which had blamed face tilt.
  - **It checked mechanisms before rejecting them.** It measured the catalogue parallelogram and the long slide, and read their graphs. It judged that neither fits without a bracket it had not designed or checked, and that the long slide's trials had overloaded its bushings.
  - **Result:** it answered with three pad-reach moves bracketing the value it found. All three would be trialled, against none of the original three.

  One iteration is no evidence for the prediction, and it raises a question the campaign must answer: whether the tools make System 2 more conservative, choosing parameter moves it can verify over mechanisms.

## D-087 The mechanism graph belongs to no robot: a base names the robot, and the graph gains the machine elements other worlds need (world: physics; F0 of the general framework)

**Why.** The general framework (docs/design-process.md, plan of 2026-11-04) asks that every world's bodies be graphs of real parts, so one builder, one set of laws and one library serve them all. That covers the ME 2110 robot first, then the mobile explorer and the talking robot. The graph was written for the gripper:
- its builder posed bodies up to a body named `gripper`;
- its kinematics drove everything from a joint named `gripper`, on the SO-101's model, with the stock moving jaw as the default finger;
- it had no wheels, motors, pistons, springs, bands, latches or sensors.

**Change.**
- **A `Base` names the robot a graph is added to:** its root body, its driving joint, its stock model and its default finger. The world passes it in; the graph stays plain data, so no recorded graph's JSON changes. `so101_gripper` is the gripper's base, and the default.
- **New machine elements**, each general and none an assembly:
  - **wheel:** a rolling cylinder, given its grip on the ground;
  - **plywood:** a material for plates;
  - **free root joint:** for a chassis;
  - **actuators:**
    - a position servo with its stall torque;
    - a gearmotor, as a torque–speed line from stall torque and no-load speed;
    - a piston, as pressure times bore area on a slide, with a needle valve's speed limit as a damper;
    - a torsion spring, as a torque–angle line (a mousetrap);
    - a band, as a linear spring between two points with its rest length and largest stretch;
  - **latch:** holds a joint until released;
  - **sensors:** a limit switch (touch), a range finder (IR or ultrasonic) and an encoder.

  Each compiles to the simulator's own elements, with its numbers from the part's datasheet, which the world's catalogue supplies as data.

**Prediction** (before the code):
- the gripper's models, laws and verdicts are unchanged: the D-078 equivalence goldens, the law goldens and the gold and stock tests all pass untouched;
- each new element compiles and behaves as its datasheet line says, within the simulator's discretisation (tests: stall torque at zero speed and no-load speed at zero torque; piston force P·A; spring torque at its two angles; a band's force k·stretch; a latch that holds until released; a range reading against geometry);
- no world uses the new elements yet (F6 does).

**Measured.**
- **The gripper is unchanged.** The D-078 equivalence goldens, the law goldens and the gold, stock and planted-fault tests pass untouched (753 passed, 2 skipped, 1 xfailed, with the fingerprint test re-run after the new modules joined the physics component).
- **Every element behaves as its datasheet line says** (`tests/mech/test_elements.py`, on the empty `world` base):

  | Element | Measured |
  |---|---|
  | Gearmotor | 1.2 N·m at rest; runs up to 16.8 rad/s within 1% |
  | Piston | P·A = 196 N; settles at the valve's 0.3 m/s within 1% |
  | Mousetrap spring | 0.45, 0.325 and 0.20 N·m at 0, 90° and 180° |
  | Band | k·stretch while stretched, nothing when slack |
  | Latch | holds a pendulum level, which falls once released |
  | Limit switch | reads the weight resting on it within 5% |
  | Range finder | reads the floor's distance exactly |
  | Encoder | reads its joint |
  | Servo | reaches its target |
  | Two-wheeled cart on a free root | rolls at the wheels' speed times their radius, within 10% |

- **One finding about stability.** Speed-dependent forces are joint damping, which the simulator integrates implicitly, because as explicit actuator forces they were unstable: the motor's back-EMF on a 56 g wheel, and the needle valve on a light rod. This is how `me2110_model` already treats the valve. A servo's gearbox likewise puts damping and reflected inertia on its joint.

## D-088 Every world declares instruments, and System 2 can read, sweep and look up facts on any draft (judge: world contracts, no verdict change; policy: tools, prompt; F1 of the general framework)

**Why.** The general framework puts the data an engineer gathers on the harness's side, and the knowledge on System 2's. Until now each world's numbers about a body reached System 2 only as the brief's prose, and only for the current body (D-086's `measure` printed the brief of a draft). An engineer measures what the requirement needs, on the candidate in hand, and sees how it moves with a parameter.

**Change.**
- **An `Instrument` contract** (`evolve/core.py`): a named reading with a unit and its arguments, on a body or on a body facing one task. A world may declare `instruments()`, `read()` and `task_facts()`. `evolve/instruments.py` adds the readings every world has: cost, fabrication minutes, lead days, and the number of rules and laws that refuse the body. It also adds a task's numbers as facts, and a safe evaluator for arithmetic over readings and facts (System 2's requirement metrics, F2). Its sandbox allows arithmetic, named readings, facts and a few math functions, and nothing else.
- **Each world's existing computations became instruments,** with no new physics.
  - **Gripper (21):** servo range, widest opening, and by width: the servo angle, the face's tilt, the fingertip's rise, the squeeze, the rack's lift and the bushing load. Also tooth skip, bushing rating, pads and friction, the lifting stall, the shoulder hold for a load, the wrist roll allowed and the roll each task's grasp needs (D-085, now `gripper.roll_needed`), and the loop closure.
  - **Mobile (18):** mass and dimensions, stall force, speed, the edge it climbs by surface (driven in MuJoCo), traction and stall slopes, ground pressure and sinkage, energy per metre, steering error and sensing.
  - **Phonology (12):** read off System 1's map of sampled postures: tract length, servos, the F1–F3 ranges, the distance from given formants to the nearest sampled vowel, closure area by place, and fricative postures and their hiss range.
  - **ME 2110 (8):** the folded envelope and box fit, kit actuators used and left, spend, and print mass. F6 rebuilds this world from kit primitives.
- **Tools.** `sea` gains `read` (one instrument on a draft), `sweep` (up to 6 readings as one parameter takes up to 12 values) and `facts` (a train task's numbers). The prompt's tools section lists every instrument.

**Prediction** (written after the instruments were written, before they were checked: an inversion of order, recorded):
- every brief is unchanged byte for byte;
- every instrument reads on every world's stock body;
- verdicts and goldens are unchanged.

**Measured.**
- **Briefs:** the gripper's brief is byte-identical on the stock body, the long slide, the prismatic slide and a braced built guide (hashes compared against the previous commit). The rods' prose now formats numbers from `_rod_numbers`, which the instruments also read.
- **Instruments:** every instrument reads on each stock body. Mobile takes 0.2 s for all of them, ME 2110 0.0 s and phonology 14 s (building its map of sampled postures).
- **Task facts:** the gripper's come from the object catalogue (width, depth, height, mass, friction, round). The others are flattened from `Task.data`; ME 2110's carry only a variant index until F6.
- **Tests:** `tests/test_tools.py` covers reading, refusals, the evaluator's sandbox, and the read, sweep and facts tools on a toy world.

## D-089 A requirements list System 2 writes and the harness measures, with coverage and sensitivity (judge: the answer's contract; policy: loop, prompt, tools; F2 of the general framework)

**Why.** Designers begin from a requirements list (Pahl and Beitz; Ulrich and Eppinger): every need a metric with a target, a demand or a wish, traced to its source, and checked against every candidate. System 2 saw tasks only as pass/fail groups and failures one at a time. The general framework splits the work:
- **System 2** decides what the needs are: it knows them from the tasks, the brief and its research;
- **the harness** measures them, on the current body and on any draft, from the instruments of D-088.

LLM-written requirement lists leave most needs out (Massoudi and Fuge 2025 measured coverage under 20%), so the harness reports the groups no requirement covers.

**Change.**
- **The answer's contract.** `ProposalBatch.requirements` is the whole list, revised in any answer; an empty list keeps the last one. Each `Requirement` has an id, a statement, a metric (arithmetic over instrument calls and task facts), a sense, a marginal value, an optional ideal value, the task groups it is measured on, and its source.
- **Measuring** (`evolve/requirements.py`). For each requirement on a body, the harness takes its worst value over its groups' train tasks and says whether it meets the ideal, the marginal value, neither, or could not be measured, and why. A requirement is a **demand** if it guards a held group, else a **wish**. **Coverage** lists the train groups no requirement names.
- **The loop** keeps the list, writes it to `requirements.json` each iteration, shows the table in the prompt, and records on every trial which rows the candidate moved.
- **Tools.**
  - `measure` adds the draft's values beside the current body's.
  - A new `sensitivity` tool moves each parameter of the current parts by a tenth of its range, up and down. It reports how each requirement's metric moves, and names a parameter that helps one requirement while hurting another (a coupling, Suh; a contradiction, Altshuller).

**Prediction** (before the code):
- with the `requirements` switch off (the default), every run reproduces exactly; with it on and no list written yet, the prompt gains only the empty section's one line;
- in a toy world a planted list is measured correctly (worst task, demand or wish), coverage lists the unnamed groups, and a trial records the rows it moved;
- the sensitivity tool finds a planted coupling;
- no claim on groups until F9's campaigns.

**Measured** (`tests/test_requirements.py`, a toy world with instruments):
- **A planted list is measured as written:** the reach requirement at its worst task (−0.25 m short at the 0.45 m target, "no"); the cost requirement at 0, which meets its marginal value. A requirement guarding a held group is a demand; one on the body alone is a wish.
- **Coverage** names the groups a list leaves out. A metric that reads a missing instrument is "unknown", with the reason.
- **In the loop,** with the switch on, the first prompt shows the empty list's line and the second the measured list. A trial records the rows it moved (R1 −0.25 → −0.15), and the list is written to `requirements.json`. With the switch off, the prompt has no section.
- **Sensitivity** finds the planted coupling: lengthening the stick helps reach and hurts "the tip stays low", and shortening it does the reverse. `measure` shows the draft's value beside the current body's.
- **Tests:** 33 passed across the tools, requirements, loop and fingerprint tests.

## D-090 System 2 researches what it needs and keeps an engineering notebook (judge: a world's blocked topics; policy: research, tools, loop, prompt; F3 of the general framework)

**Why.** The general framework gives System 2 the knowledge side of design, so the harness must give it a way to *acquire* knowledge as an engineer does: read handbooks, datasheets and papers, then write down what it found and where. The world must never carry that knowledge in its place (the user, 2026-11-04: "part of our harness should encode conducting appropriate research"). ME 2110 is a benchmark of the harness, so pages about that course or this brief must stay out of reach. Otherwise the score would measure retrieval of past teams' solutions.

**Change.**
- **Search and fetch.** With `RunConfig.research` on, System 2's call also has the CLI's `WebSearch` and `WebFetch`.
- **A guard on every search and fetch** (`evolve/research.py`). A `PreToolUse` hook passed through `--settings`:
  - refuses a query, URL or fetch prompt that matches the world's `research_blocked` patterns (ME 2110 declares the course, its title and the brief's distinctive terms; other worlds declare none);
  - refuses calls beyond this iteration's research budget;
  - logs every call, allowed or refused, to `research.jsonl`.

  A `PostToolUse` hook records the URLs each call returned or read.
- **The notebook.** A `note` tool on the `sea` server records a finding in a persistent notebook (`library/notebook.jsonl` by default): its claim, its source URL, a quote, its numbers, the range where it holds, and what it applies to.
  - A note must cite a URL this run actually searched or fetched; a citation it never read is refused.
  - Each note keeps its provenance: world, run and iteration. Notes carry over to later runs and other worlds.
- **Credit.** The prompt shows the notebook, most credited first. A kept change whose proposal cites a note's id credits that note.

**Prediction** (before the code):
- with research off, every run reproduces exactly;
- the guard refuses planted course queries and URLs, and passes ordinary engineering ones;
- the budget stops the next call;
- a note citing an unread URL is refused;
- the notebook persists across runs, and a kept citation earns credit;
- a live smoke test with the real CLI confirms the hook blocks, and that WebSearch works under the plan login.

**Measured.**
- **The first live smoke test failed, and that was the finding.** The blocked search ran. A malformed test config had crashed the hook, and the CLI blocks a call only on exit code 2, so the guard failed open. It now fails closed: any failure in the pre hook refuses the call (tested by running it with a missing config).
- **Live, with the real CLI on the plan login:** "ME 2110 Georgia Tech final project" was refused and the model saw the reason; "LM3UU linear bushing dynamic load rating" ran, and its 10 URLs were logged for citation. `--settings` hooks do run under `--setting-sources ""`.
- **Guard tests** (`tests/test_research.py`): 9 planted course queries and URLs refused (including `me2110`, `ME-2110`, the course's title, the brief's terms), and 5 engineering queries passed (including "how to sort coins by size mechanism": the guard blocks the course, not the engineering). The budget stops the next call.
- **Notebook tests:** a note citing an unread page is refused. The same finding from a later run is recorded once. A kept proposal citing a note's id credits it, and another run's prompt shows the note.
- **A known limit:** a hook cannot edit what a search returned, so an innocent query can still surface course pages in its results. The post hook counts these as `leaks` in each iteration's record, and F9 reports the count.

## D-091 Concepts are developed before they are judged: a set with allowances, functions before form, computed comparisons, failure modes predicted (judge: the answer's contract, a module's `self`; policy: concepts, loop, prompt, tools; F4 of the general framework)

**Why.** In `gripper_c1c`, System 2 wrote the long slide about ten times, each a fresh, undeveloped concept judged against a tuned hinged jaw. It needed three developments at once and lost to whichever was missing, and no seed developed one concept across iterations. BesiegeField observed the same failure in LLM mechanism design: agents stuck editing their first idea or abandoning it. The classical practices (docs/design-process.md) answer this:
- **Set-based design** (Sobek, Ward and Liker): keep several concepts, give each a development budget, and narrow on evidence.
- **Function before form** (Pahl and Beitz): state the functions, then the options for each (a morphological chart).
- **Computed comparison** (Pugh, without its guessed ratings): every candidate against the incumbent on the requirement metrics.
- **Parallel drafts** (Dow et al., a controlled experiment): several drafts before any feedback beat serial iteration.
- **FMEA:** predict failure modes before the test, and score them after.

**Change** (`RunConfig.concepts`, off by default).
- **The concept set** (`evolve/concepts.py`).
  - A new structure (a built mechanism, or a swapped part type, so the worlds without modules have concepts too) that reaches a trial or is refused for a law becomes a concept, with its id, its body and its lineage. A bad patch or a refusal for money or a repeat opens nothing.
  - Up to 3 are alive at once. Each has an allowance of 4 trials and 6 refusals.
  - System 2 develops one with `base` set to its id; a mechanism's module `self` means the base body's own graph, so a patch continues the last development.
- **Narrowing, always on computed evidence.** The harness drops a concept when:
  - its allowance is spent without gaining a group or improving a requirement row by more than 1% (with progress, the allowance renews);
  - after 2 developments, another live concept is as good on every requirement row, better on one, and holds every group it gained (a Pareto test);
  - it has sat undeveloped for 5 iterations. This third ground is added to the plan's two: without it, a set of concepts nobody develops would lock out new ones.

  The protection before dominance is the point of the set: a first draft measured against a developed concept loses, and that is how the slide lost. A dropped concept's reason stays in the record, and the same structure may open again as a new concept: nothing is outlawed. The ratchet still decides what is kept.
- **Functions.** `ProposalBatch.functions` is System 2's function structure: each function with the options it considers, and a proposal's `options` says which it takes. The harness marks the options that are neither in the catalogue nor the modules, and those no concept has tried.
- **Comparison.** The prompt shows each live concept against the incumbent on every requirement, marked +, 0 or −; a `compare` tool does this for up to 3 drafts.
- **Parallel drafts.** `simulate` on a draft that opens a new concept is refused until two distinct drafts have been checked or measured this iteration.
- **Failure modes.** A proposal may list `failure_modes` (mode, cause, effect, and a signature: a phrase its failure would show as, in the words of a law or a diagnosis). After its trial or refusal the harness scores whether the signature showed, rates detection from where it showed (the laws, or only the trial) and severity from the held groups lost. System 2's calibration section reports modes predicted and failures not predicted. This is never a gate.

**Prediction** (before the code):
- with the switch off, a scripted run's trials reproduce exactly (the answer's schema does grow, as it did for D-089, so a live System 2 sees the new optional fields);
- on a toy world: a refused or trialled mechanism enters the set; `self` continues its graph; a fourth concept is refused while three live; allowance, dominance and idleness each drop a concept with the reason recorded; a dropped concept cannot be built on; the matrix's signs follow the requirement's sense; the parallel-draft rule refuses and then allows; predicted modes are scored.
- no claim on groups until F9.

**Result (tests).** The prediction held on the toy world (`tests/test_concepts.py`, 14 tests). The full suite: 777 passed, 1 xfailed, 2 skipped, 1 failed. The failure is `test_real_embedders_shapes`, a slow nightly test that downloads models: it hung in a socket connect until its 300 s timeout, twice, and `memory/embed.py` is untouched by this work. It passed at F0. This was the first full run since F0, and it found a break of my own: D-088 renamed `_built_rods` to `_rod_numbers` and left `tests/mech/test_guide.py` importing the old name, so the suite could not collect from `5dbae24` until the repair here. The subsets I ran for D-088 to D-090 did not include that file. No claim on groups until F9.

## D-092 System 1 carries out System 2's ideas: controls from the graph, operating plans, and whether the operation or the body failed (skill: mech program, search; world: states, plans; policy: operation, loop, prompt; F5 of the general framework)

**Why.** A new mechanism is only as good as what operates it. ME 2110's controls are written by hand per part (`me2110._space`), so a mechanism System 2 composes has no controls unless someone writes them; System 1 starts each search from defaults that know nothing of what System 2 intended; and a failed trial does not say whether the mechanism could not do the task or System 1 did not find how. The mobile world already separates those for itself (`not_found_by_search`, D-041's probe). This makes that general.

**Change** (`RunConfig.plans`, off by default).
- **Controls from the graph** (`mech/program.py`). Every actuator, latch and sensor exposes typed controls: a servo's time and target (its joint's range), a motor's start, run and duty, a piston's time and hold, a latch's release, a sensor's threshold. `Program` plays any such set of values on a `Machine`. An actuator may be triggered by a sensor, and its time then counts from the sensor firing. No world's code names a part.
- **The operating plan** (`core.OperatingPlan`, `Proposal.plan`). Starting values by control name, the sensor triggers, and intents: a state the body should reach, with its bounds. It travels with the body and counts in its key, so a plan alone can be proposed and trialled. A name the world does not control is refused, naming those it does.
- **System 1 starts from it.** `local_search` tries the plan first and records that first attempt's score (`Outcome.nominal`). This reaches the worlds that search continuous controls (mobile, ME 2110). The gripper picks among programs and the talking robot starts from its babbling map, so they take intents only, for now.
- **Operation or body** (`evolve/operation.py`). Each failed task of a trial is labelled from the outcome's states (`Outcome.states`): `operation_not_found` if an intended state was not reached (or the world's probe says the body could), `body_cannot` if every intended state was reached and the task still failed, `unobserved` if there is no intent or the world does not read that state. ME 2110 reads each task's points and progress; every other world reads none yet, and says `unobserved` rather than guess.
- **A new structure gets a larger search.** A plan's `search` multiplies System 1's attempts. Under the concept set, a new concept searches twice as long (philosophy §7: a new structure is protected until its own parameters are fitted). Its developments inherit that with the plan, so it costs simulation for as long as that lineage lives; dropping it on a kept body would change the body's key and risk losing what was held.
- `simulate` also reports states and the plan's first score, which covers the deferred `experiment` tool.

**Prediction** (before the code):
- with the switch off and no plan, scripted runs reproduce exactly and bodies keep their keys;
- on a compiled graph, `Program` plays a motor's window, a piston's stroke, a latch's release and a sensor-triggered actuator at the commanded times, with no part named in the code;
- on a toy world: System 1's first attempt uses the plan; a plan-only proposal is trialled; an unknown control is refused; the three labels come out as defined; a new concept's body searches twice as long;
- no claim on groups until F9.

## D-093 ME 2110 is scored as the brief scores it: points in one match, on held-out tracks; the kit as the brief lists it (judge: ME 2110's numbers, the loop's objective; F6 of the general framework, first part)

**Why.** The user's goal for this world is the highest robot score, and D-074's `match` group is a fixed bar of 174 points the brief never states: a change from 35 to 60 points is invisible to the ratchet. Revision 1.1 (p. 23) and the inspection list (p. 26) allow 4 mousetraps, where D-073 read 2 from pp. 17 and 21. The held-out tracks cover two of the four port colours.

**Change.**
- **The objective is generic.** `World.objective = "points"` (other worlds keep groups). An outcome may carry `points`; the loop keeps a change that raises the mean points over the tasks that carry them, unless one of the parts in that total (`points.*`) drops by more than `World.points_noise`. Groups are still computed and shown, and a change that raises the total while losing a part goes on the frontier.
- **The score.** A `match` outcome's points are its total, or 0 if it broke any rule, as a disqualification would. Each iteration and the summary report the mean over the variants, on train and on held-out, against the brief's sprint bars 60, 150 and 215.
- **Noise was first set to 5 points,** the smallest partial score in the brief; D-094 measured it and set 9.
- **The kit.** 4 mousetraps, the later text of the brief's own contradiction (pp. 17 and 21 say 2; p. 23's revision and p. 26's inspection list say 4). A lone robot's flag scores as the highest flag, which is what the model already gives.
- **Tracks.** `Task.data` carries each track's numbers, and held-out tracks now cover all four port colours (6, 7, 8 and 8.75 in), fortress speeds from 2 to 5 rpm and floor friction from 0.4 to 0.6. The first two keep their numbers.

**Prediction** (before the code):
- other worlds' scripted runs reproduce exactly;
- a toy world with points keeps a rise, refuses a rise with a part lost beyond noise and puts it on the frontier, and refuses a rise within nothing gained; a broken rule scores 0;
- D-075's best body, re-scored on the held-out tracks, will score below its 35 train points, since those tracks were not searched and include speeds it has never met. This is the baseline F9 is compared with.

**Result (tests, D-093).** The toy-world predictions held (`tests/test_points.py`), and ME 2110's world tests pass with 35 tasks. The re-scoring of D-075's best body on the held-out tracks costs several matches per track, so it runs in F9 as the baseline, and its prediction stays open until then.

## D-094 ME 2110 from the kit: a robot is one graph in the arena, operated by its program (world: me2110_kit; judge: its laws; skill: graph kinds; F6 of the general framework, second part)

**Why.** The ME 2110 catalogue gives System 2 one task-specific mechanism per slot, each with its hand-written controls: the design is largely done before System 2 starts, and the D-075 score of 35 measures that catalogue as much as the process. The user asked for the world stripped to what a student is given and rebuilt from kit primitives.

**Change.** A new world, `me2110_kit`, beside the old one, which stays as the baseline for F9.
- **Same arena, scoring and tracks.** `sim/me2110_model.Match` is split so its arena, events and scoring run with any robot; `sim/kit_match.KitMatch` compiles a graph into that arena and plays `mech.program` on it, with the tank's pressure falling at each stroke as before.
- **The body is one part:** `robot`, a graph whose root is a free body. The only starting point is a plain plate chassis with the controller on it. It has no drive and scores nothing, as a student starts from a kit and no robot. Machine elements come in F8.
- **New graph kinds:** capsule, sphere, a joint's own spring and friction, and `carried` (an item that starts on the robot: the balls, the flag on its clip). Patches may add actuators, latches, sensors and carried items.
- **Laws, all general:** a free root; kit counts (the brief's p. 26 list, sensors included); the go/no-go box at the start pose; parts of a body must touch (nothing floats); a part's mass comes from its material's volume (the fixture alone may state it); parts may not pass through each other over every driven joint's range, with closed loops solved at each pose; the budget. An `exclude` waives a pair (a slot cut in a plate), and each one is counted, since it can hide a real collision.
- **Kit mass** (motors, tank, valves, traps) is added to the root by what the graph uses.
- **The equivalence fixture** (`tests/framework/`, never a seed). Each old catalogue robot is converted automatically from its compiled simulator model into a graph. Tests require the same bodies, joints, geoms, masses and constraints to 1e-6. They then replay the old controller's timeline through `Program` and compare each task's points.

**Prediction** (before the code):
- geometry and mass reproduce to 1e-6 on every fixture robot;
- actuators are not the same code (the graph's piston damps its rod where the old one reflected the valve onto the boom, and a timeline has no encoder feedback), so points may differ; I predict each task's points within the 5-point noise on the bodies where the old controller scores, and I will report any that are not rather than loosen the test;
- a simple kit robot written in the test (two gearmotors, wheels, a caster) leaves the dock and scores 11 from its program alone; the plain chassis scores 0; each law refuses a planted fault.

**Measured while testing the fixture (D-093's noise, D-094's replay prediction).** The old model's own match is chaotic. On the cannon body, moving the start 0.1 in, the park 0.1 in or the duty by 0.01 moves the cannon score between 18 and 36, in steps of one ball (9 points), and two of the eight runs broke a rule. The graph robot's replay scored 18 against the old 27: inside that spread, so my prediction of "within 5 points on one match" asked for something the old model does not do with itself. I did not loosen it silently. The test now compares means over nearby starts, within one ball. Two consequences:
- `points_noise` is 9 (one ball, the measured step), not the 5 I reasoned from the brief.
- A kit world score is the mean of 3 matches at System 1's controls, each with the start jittered, so a lucky match cannot win a change. The old catalogue's D-075 score of 35 was a single match, so the F9 baseline has to be re-scored the same way before the two are compared.

**Result (tests, D-094).** Bodies, joints, solids, masses and friction reproduce to 1e-6 on all four fixture robots. Matches with no loose items replay within a point (activate 11 and mermaids 24 both match), and the chaotic cannon replay agrees on the mean over six nearby starts, within one ball. The kit robot written for the test leaves the dock from its program alone and scores 11, the plain chassis scores 0, each law refuses its planted fault, and that robot goes through the whole loop: it opens a concept, is scored and is kept. Not shown: System 2 composing anything, or any score above 11 from the kit. Full suite: 809 passed, 2 skipped, 1 xfailed, with the model-download test deselected.

## D-095 Every world is stripped to its brief: design knowledge comes from research or not at all (judge: each world's rules; policy: prompt, requirements; F7 of the general framework)

**Why.** The user's direction: strip to the brief, then measure what the general process recovers. Each world's `principles()` is mostly engineering knowledge written by hand: gripper 7 of 7, mobile 8 of 8, talking robot about 8 of 10, ME 2110 about 7 of 11. It tells System 2 what a good answer looks like, which is teaching the test, and it is exactly what the notebook should now supply from sources.

**Change** (`RunConfig.stripped`, off by default, so the current loop stays as the arm to compare with).
- Each world gets `brief_rules()`: what a student or engineer is told about the task (rules, scoring, limits). The prompt then shows `# Rules of this world`, numbered, in place of `# Physics principles`. `principles()` stays for the comparison arm.
- The split is by one test: would a rule change what counts as a success or what may be built? Gripper and mobile have none beyond their descriptions. The talking robot keeps the repeat-on-fresh-noise and 50 ms rules, which define when a sound counts. ME 2110 keeps the box, the kit, the 40 s state and the fortress rule.
- Requirement coverage now traces rules as well as groups: a requirement citing `rule 2` in its source covers it, and uncovered rules are listed (the part D-089 left for this phase).

**Prediction** (before the code):
- no prompt with `stripped` contains any principle's text, in any world; the off arm's prompt is unchanged;
- on campaigns, F9 only: stripped without research should do worse than the current loop, and research should recover part of the gap. If stripped with research does as well as the current loop, the hand-written principles were not needed.

## D-096 A library of mechanical-engineering skills, reused over time: machine elements to start from, and what earns its place (skill: mech elements; policy: library; F8 of the general framework)

**Why.** The user asked for a library of mechanical-engineering skills reused over time, with the architecture kept compositional. With F6 and F7, System 2 starts from a plate and a parts list. A real engineer starts from machine elements, which is general knowledge valid in any world, and from what their own past work has proven.

**Change** (`RunConfig.library`, off by default).
- **Seeds are machine elements only** (`mech/elements.py`): lever, four-bar, slider-crank, rack and pinion, gear train, cam and follower, ratchet and pawl, latch, guide and carriage. Each is built from real parts, joints and transmission, and mounts on any body. A test drives each one and checks its textbook relation (a gear train's ratio, a four-bar's closure, a cam's lift, a ratchet's one way). No seed is task-specific.
- **Composition.** A mechanism may list `uses`: elements, each with its parameters, the body it mounts `on`, where (`at`, `quat`) and a `prefix` for its names. A patch then connects them. To let contact-driven elements work, a graph may declare `contacts`, pairs of solids that touch each other.
- **Three stores, each kept across runs in `library/`:**
  - **Mechanisms** (`mechanisms.json`). When a body is kept, the bodies it added since its parent are entered under a structural signature (their tree, joints, part kinds and actuators, with lengths left out), with its provenance. It is promoted once seen in 2 or more (world, mount) contexts; numbers that differ between its contexts become its arguments. System 2's name for the concept names it.
  - **Notes** (`notebook.jsonl`). As before, now also counting the rejected proposals that cited a note, so a note is checked against later trials and not only credited.
  - **Methods** (`methods.json`). Each requirement metric System 2 wrote, kept with its worlds and runs. A metric used in more than one run is promoted and shown wherever its instruments exist.
- **Pruning.** An unpromoted mechanism unseen for 10 runs is removed. A test asserts every stored mechanism has a provenance trial.

**Prediction** (before the code):
- every seed compiles on the kit's arena and on an empty world, passes its relation, and passes the world's laws;
- a concept kept in two contexts is promoted with the varying lengths as arguments and can be instantiated at new values; one kept in one context is not offered; an unused entry is pruned;
- with the switch off, nothing is read or written;
- nothing about campaigns: whether the library is used at all is an F9 measurement.

**Outcome.** The prediction held, once these were fixed along the way:
- **The kit's fastening law had a hole, now closed (judge).** `_floating` looked only inside one body, so a bracket bolted to the robot with no joint could float 9 cm above the plate and pass, and a hinge could hang in mid-air. Bodies without a joint now count as one rigid piece with what they are bolted to, and a jointed body more than 1 cm from what it moves on is refused. No existing kit test or fixture robot depended on the hole. Earlier kit results were never campaign results, so nothing has to be re-scored.
- **The ratchet.** It first failed in reverse at r = 0.02 m and looked broken at r = 0.08 m, for two different reasons. At 0.08 m the wheel was inside the test's own fixed frame, which was a test error. At 0.02 m a tooth sank 4.6 mm into the pawl because MuJoCo's default contact is soft relative to a light wheel. Named contact pairs now use `solref=[0.004, 1]`. It is tested across its whole declared range, r = 0.015 to 0.1 m, with 10 N at the teeth. That load is what is tested and what its docstring states; above it the pawl's stop can give.
- **`mech/base.py`.** The empty world read joint ranges as degrees, a real D-087 bug that only the seeds' range tests exposed.
- **The four-bar.** Its ground bracket lay in the links' plane, so the coupler passed through it. The bracket now sits behind that plane. Its crank and rocker share a plane and are excluded from colliding with each other, which remains the open `excludes` item for F9.
- **The four-bar's solver** raised `ZeroDivisionError` when the crank's far pin landed on the rocker's pivot, which would have crashed a trial instead of refusing it. It now refuses with the reason.

Beyond the prediction: a promoted mechanism reaches a graph as patch operations (`LB.expand`), so a world needs no library code beyond calling `compose` for the seeds. The store is locked for runs in parallel. A kept element is entered like anything else, so a seed can also appear under its own signature once it has been kept in two contexts; that duplication is left in, and F9 can show whether it matters.

**Not yet shown.** No real System 2 call has used the library. Whether it uses the elements at all, and whether they help, are F9 measurements.

**Rechecking the C1 campaign against today's judge (2026-11-04, after D-096).** While putting `gripper_c1c_s{0,1,2}` on the results page, each kept gain was re-solved once on the current code, with the run's seed:
- seed 2's tuna can reproduces: the kept body holds it and the body it started from does not;
- seed 1's tuna can does not: the recorded controls fail on replay, and a fresh search of 9 attempts fails;
- seed 0's potted-meat can does not: a fresh search of 2 attempts stops at the force limit, as the body it started from does.

The cause is not established. D-085's wrist-roll limit is one candidate, a single re-solve's search luck is another. So D-084's 19, 18 and 18 are results under the judge of that time. F9's gripper arm measures on today's judge. Two long-slide failures do replay as recorded (apple: a bushing at 119 N against 52.9 N; banana: stuck).

## D-097 Verify the gripper on today's judge, then iterate with the full framework until it holds 21 of 21 (evaluation; F9 for the gripper)

**Why.** The user's direction: keep iterating until the gripper is thoroughly verified with the new harness, aiming for all 21 groups. Every gripper score on record predates a judge or physics change, and 1 of 3 rechecked gains reproduced, so there is no baseline to improve on. How much a verdict depends on the search seed has also never been measured.

**Change.** `scripts/recheck_bodies.py` re-solves a run's final body on every task, under several seeds, on today's code. No verdict or law changes. A score counts as verified only when the final body holds its groups under every seed of a recheck on the judge of that moment, on both train and held-out tasks. Redesigns come only from System 2, and no law is loosened to reach the target. If a law looks wrong it is examined on physical evidence and recorded on its own, never as a means to a score.

**Prediction** (before the run):
- stock holds 15 to 17 of 21 train groups under each seed, as recorded before;
- the rods-run and C1 best bodies each hold fewer groups than their records under at least one seed, by 1 to 3 groups: the records took a single seed's verdict and today's judge differs;
- at least 1 group flips between seeds on every body, so a one-seed score overstates what a body can do;
- for the campaign, the full framework reaches 19 or more verified on at least one seed. I do not predict 21: no body has reached it under the D-076 physics, and the last two groups have been the hardest throughout.

**Baseline measured (D-097's recheck, `results/evolve/recheck/d097_baseline.json`; 3 seeds, 42 train and 42 test tasks).**

| body | train groups, seeds 0/1/2 | test groups | held on train and test under every seed |
|---|---|---|---|
| stock | 16, 16, 16 | 15, 16, 15 | 13 |
| `gripper_s2rods_s0` | 19, 18, 18 | 18, 19, 19 | **18** |
| `gripper_c1c_s0` | 19, 19, 19 | 18, 19, 18 | **18** |
| `gripper_c1c_s1` | 17, 18, 17 | 17, 16, 16 | 16 |
| `gripper_c1c_s2` | 17, 18, 18 | 17, 17, 17 | 17 |

- Prediction on stock (15 to 17 train): held. Prediction that bodies fall below their records: wrong for `c1c_s0`, right for `s2rods_s0` on two seeds. Groups do flip between seeds, as predicted.
- This corrects the earlier single-task recheck, which overstated the doubt: at body level the C1 scores largely stand.
- **The verified best is 18 of 21.** Both best bodies never hold the same three groups: mustard bottle, potted-meat can, sugar box.
- **The live run** (`gripper_f_smoke`) ran the full framework end to end, and showed three process gaps: no web research at all though it was on; coverage calling every group uncovered, since System 2 named groups in `source` and not in `tasks`; 1.75M tokens in and 16 minutes for one System 2 call.

## D-098 A guide's rods and bushings are a choice of real parts (judge: mechanism; world: physics, gripper)

**Why.** Every long slide that reached a trial passed 0 of its 42 picks, mostly on bushing load and tooth skip. Both are set by the 3 mm rod and the LM3UU, which are global constants. A graph could draw a thicker rod and the physics would still bend a 3 mm one. This breaks the standing rule that a binding law must expose the variable it binds on: the slide family has never been fairly tried. Nothing here is a design hint; it adds parts a shop can buy, with their datasheet numbers.

**Change.** `Guide.bushing` names one of LM3UU, LM4UU, LM5UU, LM6UU, LM8UU (`jaws.BUSHINGS`). The rod's bending stiffness, the bushing's length and rating, and the mass follow it. Where sources differ the lowest rating listed is used, so a slide is never flattered. The rating is checked in the simulator, the part facts and the instruments alike. Laws: a built guide's rods must have that bushing's bore, carriage must carry that bushing, the carriage must have material around each bushing's outside (1 mm wall at least), rods far enough apart for two bushings. Modules `prismatic` and `long_slide` gain `rod_mm`, default 3.

**Prediction** (before the code):
- every existing golden is unchanged: 3 mm stays numerically identical;
- rod stiffness goes as d^4 (6 mm is 16 times 3 mm), a hand-checked number in a test;
- a long slide on 6 mm rods has static bushing load under its 265 N rating and rack lift under the skip limit, where 3 mm has neither at wide openings;
- it gets heavier by about 100 g, which is the cost the wrist and shoulder laws then judge;
- whether any slide then passes picks is not predicted. The rack pressing on objects and the sector's sweep are other causes, untouched here.

## D-099 System 2 researches before it simulates, and a group named as a requirement's source is covered (policy: tools, prompt, requirements)

**Why.** With research switched on, the live run made no search in 35 turns, so the framework's source of engineering knowledge was never used. "You may search" leaves out what the user asked the harness to encode: conducting appropriate research. Coverage called all 21 groups uncovered because System 2 traces needs through `source` ("reg_sugar_box simulation..."), and only `tasks` was counted.

**Change.** In a run's first iteration `simulate` is refused until the run has recorded 2 notes, or its searches are spent, in the same way as the parallel-draft rule. It says what to do, never what to find. The prompt asks for research first and again when a failure comes back. Coverage counts a group named in `tasks` or in `source`.

**Prediction:** the next live run makes searches and records at least 2 notes in iteration 0; the uncovered list shrinks to groups no requirement mentions. Whether notes are cited in kept changes is a campaign measurement.

**D-098 and D-099 measured.** The bare `long_slide` at a 50 mm sector, seed 0, on all 84 tasks (`results/evolve/recheck/d098_bare_slides.log`). No pad design, no tuning, no System 2:

| rod | tasks | train groups | test groups |
|---|---|---|---|
| 3 mm | 0 | 0 | 0 |
| 4 mm | 3 | 0 | 0 |
| 5 mm | 70 | 17 | 16 |
| 6 mm | 72 | 18 | 17 |
| 8 mm | 72 | 18 | 17 |

At 6 mm it holds mustard bottle and sugar box, which no hinged jaw has held under this physics, and it loses banana, potted-meat can and tomato soup can. So the law that refused slides was right and the fixed part behind it was what stopped them. D-098's numbers held. The prediction that other causes remain also held: what is left is the rack on the object's top, force limits and pads.

The second live run (`gripper_f_smoke2`) met D-099's prediction: 4 searches, 3 notes, each cited in a requirement's source, and 16 groups uncovered where it was 21. System 2 chose 6 mm rods itself, and its pad-and-slide proposals lost apple and soup can.

**The campaign** (`gripper_f_s{0,1,2}`): full framework, stripped to the brief, 14 iterations, then `recheck_bodies.py` under 3 seeds.

**Prediction:** at least 1 seed ends on a slide body; verified groups (train and test, every seed) reach at least 19 on at least 1 seed; the group most likely to stay out is the potted-meat can (friction 0.35, squeezed out) and I do not predict 21 on this first campaign.

**The first campaign seed, and why it does not count.** `gripper_f_s0` reached 21 of 21 train groups in 7 iterations, and on `recheck_bodies.py` it holds them on train under every seed, with test at 20 or 21. Its final body has a printed rack 1.5 x 2 mm in section and 2 mm-wide gear teeth, chosen to clear the objects. Printed parts are rigid in the simulator and nothing limits them. By hand, that rack sees over 500 MPa on PLA that fails near 48 MPa, and it buckles at about 7 N against 31 N; the earlier body's 5 x 6 mm rack deflects about 13 to 18 mm at 85 mm overhang under 11 N, against the 0.5 mm at which teeth skip. So this result, and the bare slides' 70 to 72 tasks above, measure a judge with a hole. The verified best remains the hinged jaw's 18. Seeds 1 and 2 were stopped.

## D-100 A printed rack and its teeth carry real loads (judge: mechanism, physics; world: gripper)

**Why.** D-059 left tooth bending and beam stress out because no measured allowable stress existed for printed PLA, and deferred them. System 2 has now exploited exactly that, as any optimiser would. Measured values exist: flat-printed PLA at 47.6 MPa flexural strength and 1808 MPa modulus (ISO 178, Prusa Mini; 90 deg builds reach 21.3 MPa). This does not rule out a slide: a deeper rack and a backing roller behind the mesh are how real long racks work, and both are exposed.

**Change.** In a trial, per step, at the force the teeth actually carry (the same 20 ms hold as the rods):
- **Teeth:** Lewis stress F / (b m Y), with Shigley's Y by tooth count, on the gear and the rack;
- **Rack:** a beam from where it leaves the carriage to the mesh. Its bend under the separating force adds to the lift that makes teeth skip, and its bending stress counts. A box running along the rack and touching it counts in the section;
- **Buckling:** when the mesh pushes the rack toward its support: Euler, K = 2 free, K = 1 with a roller;
- **A roller** (`Cylinder` with role `roller`, on the gripper, behind the rack at the mesh) takes the separating force. It is a bought bearing and must be mounted like any fixed part.

Allowable stress is 47.6 MPa over a 1.5 factor (31.7 MPa). Any stress or bend past its limit makes the attempt unphysical, as a bushing's load does. The same numbers are instruments, at stall.

**Prediction** (before the code):
- no hinged jaw changes, as they have no rack;
- hand numbers hold in a test: the 5 x 6 mm rack's I, its bend at 85 mm, and Lewis on a module-1, 2 mm tooth at 31 N;
- `gripper_f_s0`'s final body fails nearly every pick;
- the bare module slide at 6 mm rods falls far from its 72 tasks, because its rack flexes;
- a rack with a roller or a deep rib can pass.
Whether System 2 finds such a design under the laws is the next campaign's question; I do not predict it reaches 21.

**D-100 measured** (seed 0, all 84 tasks). `gripper_f_s0`'s thin rack: 0 tasks, every one unphysical, against 84 under the old judge. Bare 6 mm slide: 24 tasks, 6 train groups, against 72 and 18. The same slide with a roller placed behind the mesh, ignoring that two laws refuse that placing (clearance from the carriage, and a mount): 69 tasks, 17 train groups, with force limits as the main failure. So all three predictions held, and the slide is not ruled out; making that roller real is design work left to System 2.

Mechanisms in `library/` entered under the old judge are archived (`mechanisms.pre-d100.json`). The notes stay, since they are sourced findings and not results of the judge.

**The campaign** (`gripper_g_s{0,1,2}`): full framework, stripped, 20 iterations, then the recheck under 5 seeds. **Prediction:** at least 1 seed keeps a slide with a backed or deep rack; verified groups beat the hinged jaw's 18 on at least 1 seed; 21 is not predicted.

**D-100's source, read directly.** Muraleedharan et al., *Effect of build orientation angle on mechanical properties and fracture behaviour of FDM 3D printed PLA*, Engineering for Rural Development, Jelgava, 2026 (https://erdev.lbtu.lv/sites/default/files/proceedings/2026/TF152.pdf): ISO 178 specimens (80 x 10 x 4 mm), Prusa PLA on a Prusa Mini, 0 deg 1808 +/- 45 MPa and 47.6 +/- 2.2 MPa; 90 deg 1567 +/- 42 MPa and 21.3 +/- 0.5 MPa. **The limit of this:** those are the best orientation's numbers, for roads along the bending stress. A rack can be printed that way, flat on its long side, but a tooth's root is loaded across its section in other ways, so the Lewis allowance may be generous. One study and one printer set both numbers. The 1.5 factor is judgement. A verified result is good in simulation under these numbers, not a promise about any particular print.

**D-100's suite.** Full suite: 872 passed, 8 failed, all in `tests/mech/test_seed_equivalence.py`, each for the one new numeric `jaw_rack`. That record is what the judge reads of the rack, and the deleted builders never made it, so the comparison drops that key alone and `golden_jaws.json` is unchanged. All 13 configurations still match it at 1e-6.

**Hand audit of `gripper_g_s0`'s `long_slide_r52_farplane_roller` (20/21 train at iteration 4), before its recheck.** System 2 built it from the D-100 failure: 6 mm rods, LM6UU bushings, a 2 x 7.4 mm rack backed by a 3 mm roller, a 52 mm sector on a 45 mm hub. What holds up by hand at the 31.2 N the teeth carry at stall: tooth stress 11.6 MPa against 31.7; rack in tension when gripping (2 MPa), since the gear sits behind the carriage, so the buckling reading of none is right rather than a hole; roller arm stress 8.3 MPa; bushings 88 N against 265. What the judge does not see: it treats the roller as rigid and the sector's hub as rigid. Under the 11.3 N separating force the 6 x 6 mm PLA arm bends 0.35 mm over its 26 mm free length, and the hub about 0.07 mm. Added to the rods' 0.26 mm at 96 mm wide, the mesh opens about 0.68 mm against the judge's 0.5 mm limit, passing the limit above about 60 mm wide at stall. The limit is half a module of a 2.25 mm tooth, so this is a margin the design has not earned rather than a tooth that skips. Also unjudged: the sector puts about 2.8 N m of side moment on the servo's output, and no law reads the shaft's side load. **Next, once the campaign is stopped:** the mesh's lift sums every printed member in its load path, from the parts, and an instrument exposes the hinge's side moment. **Prediction:** this body loses its widest objects until the arm is thickened, and a 10 x 10 mm arm (0.05 mm) recovers them, with the same 20 groups.

**Campaign G, recheck (5 seeds, all 84 tasks).** `gripper_g_s2`, a far-pivot parallelogram System 2 built at iteration 3 (`far_pivot_L90_b40`: 90 mm link pairs on a printed bracket, a 55 mm drive rod, 1 mm TPU pads, no rack and no teeth), holds **19/21 on train and test under every seed**, the new verified best under the D-100 judge (18 before). Its run reported 21/21 train and stopped; per seed the rows are 20/19, 20/20, 21/20, 20/19, 20/20. Over the 5 seeds it fails 11 task runs: `mustard_bottle` 9 (7 of them `moving_jaw_on_top_face`) and `potted_meat_can` 2. `gripper_g_s0`'s roller slide reported 20 and holds 17 throughout. Seeds 0 and 1 were stopped at iterations 5 and 4 to change the harness.

**Hand audit of `far_pivot_L90_b40` at stall (1.62 N m):** servo and rocker arms 27 to 31.5 MPa against 31.7; drive rod up to 141 N against an Euler load of 186 N (margin 1.3, at a 35 degree transmission angle); link plates margin 3.7; bracket 4 MPa; pins 3 MPa. Nothing is over, but the judge reads none of these, so the arms' margin is luck, not a result. The interference law sweeps the parts' solids and does not read `excludes`, so the exclusions hide no clash.

**What this exposes.** A run ends when train holds under one seed, so nothing rewards a body that holds under others: 21 reported, 19 verified. **Next:** a run does not stop at full train until the body holds under more seeds, and tasks lost there go back to System 2 as failures; and printed members in the load path are judged at the force they carry.

## D-101 (policy: loop): a run ends when the body holds under several seeds, and it can continue from a built body

**Why.** `gripper_g_s2` stopped at 21/21 under its one seed and verifies at 19. Every seed-dependent loss in the record so far (`reg_mustard_bottle` and the rest) was invisible to the run that made it, and the record of reported against verified is 21/19, 20/17, 19/18, so the gap is the rule rather than bad luck.

**What.** `RunConfig.robust` (`--robust N`): when the body first holds every train group, the evaluator widens to the run's seed plus N others, a task counts as solved only if it holds under all of them, and a loss returns to System 2 as an ordinary failure, naming the seed. Those seeds are `seed + 100 k`, outside the 0 to 4 of `scripts/recheck_bodies.py`, so the recheck stays held out. Solving under several seeds is only paid for after full train. `RunConfig.start` (`--start summary.json`) starts from a run's final body, so System 2 keeps its own redesign rather than rediscovering it. No law moved, and no knowledge about grippers.

**Prediction.** From `far_pivot_L90_b40` with N = 4: the widened train starts at 19 or 20 of 21, with `mustard_bottle` among the losses. The failures are mainly `moving_jaw_on_top_face`, so System 2 has a named mechanism to work on, and I expect its gain to come from fingertip height or pad reach rather than a new concept. Reaching a verified 21 is possible, not likely on the first campaign: I would put it near one in three.

## D-102 (skill: gripper, measured first): the same mechanism, operated in more than one way

**Why.** A failure is charged to the mechanism, but in the gripper world every grasp comes straight down with the jaws across the narrow side, and System 1 tunes only depth, clearance and timing. So an error in how a mechanism is used reads as an error in the mechanism. `far_pivot_L90_b40`'s mustard failure is the case: the moving jaw presses 19 mm above the bottle's centre, 65 degrees down, and the fixed jaw never touches, with the moving fingertip 5 mm higher than the fixed one (the user's observation, 2026-11-04).

**Measurement, in a worktree, before any change to the harness.** `grasp` gains `tilt` (about the object's long axis), `lean` (about the closing axis) and `along` (along the object). Each of the 11 (task, seed) pairs that body failed in its recheck is re-solved under 9 fixed modes with System 1 tuning the rest as usual. The wrist's limit is opened to the arm's own for this measurement, since D-085 limits it to what the straight-down grasps need.

**Prediction.** Tilt rescues at least half of the 7 `moving_jaw_on_top_face` failures, in one tilt direction only, the one that lowers the moving fingertip. `along` helps the bottle a little, and hurts by moving the grip off its centre of mass. `lean` helps least. **What would prove this wrong:** the 5-joint arm cannot hold the tilted pose, or a tilted jaw reaches the table first; then the failures stay, and they belong to the mechanism after all.

**D-102 measured (99 solves, the 11 failed pairs under 9 modes).** Rescued, of 11: straight 0, tilt -0.2 rad 8, along -30 mm 7, lean -0.2 rad 6, tilt -0.4, along +30 and lean +0.2 2 each, tilt +0.2 and +0.4 none. Some mode rescues 10 of 11; only `rva000_mustard_bottle` under seed 0 stays stuck. Against the prediction: tilt, one direction only, right. **`along` and `lean` were both wrong**: I expected them to help little, and they rescue 7 and 6. So these failures were mostly errors of operation, charged to the mechanism. Campaign H (iterations 1 to 3, no gain) was stopped, since it was asking for hardware to fix them. **Not yet checked:** that the arm really reaches these poses as commanded, and that they pass under the wrist's own limit, since this measurement opened it.

**D-102, corrected: the first measurement misread itself.** Checking the IK showed the 5-joint arm cannot hold `tilt` at these placements. Its axis error equals the whole commanded 0.2 rad, so the jaw stays nearly vertical and sits about 7 mm off. Those rescues came from the tool point going 10 mm lower, past the 35 mm limit on `depth` in `configs/primitives.yaml`, not from a tilt. The table clearance in `grasp_poses` allowed about 45 mm on this bottle, so a search bound nobody chose for this was failing tasks and charging them to the mechanism.

**Second measurement, only what the arm can do (the same 11 pairs).** Rescued: straight 0; deeper (`depth` up to 60 mm) 7; `along` -30 mm 7, +30 mm 2; `lean` -0.2 6, +0.2 2; `yaw` -0.2 1, +0.2 3, +/-0.4 none; deeper with `along` -30 mm 9; some mode 10. `rva000_mustard_bottle` under seed 0 is still stuck under every one. **The harness's answer to "what does this arm's orientation allow" is narrow:** it can lean in its own plane and turn its wrist; it cannot roll about the object's long axis here.

**What.** `depth` searches to 60 mm, the table clearance still bounding it. `grasp` gains `lean`, `yaw` and `along`. `PrimitiveDef.modes` lists other ways to use a primitive, and `env.other_ways` searches each in turn when the plain one fails, with System 1's usual search within it, stopping at the first pass. `tilt` is not offered, and a test requires every offered mode to be a pose the arm holds. The wrist limit (D-085) now covers all of them. A failed task's detail says how many ways were tried. System 2 and the judge are untouched; only operation changes, so every earlier score is on the old operation and has to be rechecked to be compared.

**Prediction.** `far_pivot_L90_b40` rises from 19 to 20 held throughout, not 21: `rva000_mustard_bottle` under seed 0 failed everything above, and it decides the group. Stock also rises from 13, because some of its failures were the same error, to about 15 or 16. Its share of failures labelled as the mechanism's becomes more trustworthy, since each has now failed 7 ways.

**D-102 measured.** Recheck, 84 tasks per seed, on seeds 0 to 4: stock holds **16/21** throughout (13 before); `far_pivot_L90_b40` holds **20/21** (19 before), failing one run of 420. Both as predicted. Those seeds chose this change, though, so they no longer test it. On unseen seeds: 5 to 9, 420 of 420 and 21/21; 10 to 29, 1678 of 1680. **Over all 30 seeds: 2517 of 2520 runs, 27 of 30 seeds clean, 20 objects never fail, and `mustard_bottle` fails 3 of its 120 runs (2.5%)**: `rva000` under seed 0, `rte000` under seeds 11 and 19. So this is 20 solid and a 21st at about 97.5%, not a 21/21. Stopping at seeds 5 to 9 would have overstated it. Audit under seed 1: 84 of 84; 83 by the plain grasp and 1 by `along`; three grips go past the old 35 mm; no jaw or arm part touches the table. So on that seed the search bound did most of the work and the other ways little. Suite: 4 tests had fixed what this changes (the stock wrist limit, now -2.31 rad, which by hand is -1.93 - 0.2 - 0.2; the pudding box's label; a slide's reported failure). The test that rods stop a pick stays strict by checking the plain grasp apart. Full suite before those four test edits: 887 passed, 4 failed. Those four and their files pass after them; no source changed.

## D-103 (judge: mechanism; world: physics): printed members that carry a pin's force are judged at that force, and a roller's arm bends

**Why.** The judge reads a printed rack and teeth (D-100) and treats every other printed part as rigid. By hand `far_pivot_L90_b40`'s arms reach 31.5 MPa against 31.7 and its drive rod has a margin of 1.3 against buckling, and `long_slide_r52_farplane_roller`'s mesh opens 0.68 mm against 0.5 once its roller arm bends. Neither was a result of the judge, and the thin rack showed what System 2 does with a part the judge calls rigid.

**What.** Each step reads the force at every loop's pin. A web or link plates spanning a pin from the body's origin is a strut: its axial load per plate against stress and, in compression, Euler buckling on its weakest axis. A web from a hub out to a pin is a crank: its bending at the hub at the force across it. Each is judged held for 20 ms, like the rack. A roller's arm is a beam from its support to the mesh, so its bend joins the lift and its stress is judged. All from the graph's own parts: no part is named, no rule is about grippers. **Not read:** members away from a loop's pin (a rocker's link plates, by symmetry the follower's), the hub's own flex, and the servo shaft's side load, which has no sourced limit.

**Prediction.** `far_pivot_L90_b40` keeps its scores, on 30 seeds, within a run or two, since by hand nothing is over; if transients exceed the arm's narrow margin it loses wide objects, and that margin was never real. `long_slide_r52_farplane_roller` loses its widest objects, and a 10 x 10 mm roller arm recovers them. A rod of half the depth fails by buckling, showing the law binds.

**D-103 examined: the load past the servo's torque is real.** Under the law the far pivot read 74 MPa and its rod 248 N, where the servo's stall gives 106 N. Held still, the rod's force times its lever about the hinge's axis is the stall torque to within 10% (a test now holds this), so the reading is true. The rest is the jaw arriving at the object at the servo's full 1.92 rad/s. A rotor of 0.028 kg m2 stopped through a stiffness k adds a torque of w sqrt(I k). The simulator's peak implies about 47 N m/rad. By hand, the 6 x 6 mm arm in bending alone is about 29 N m/rad, so a real one adds about 1.7 N m against the simulator's 2.2: the same order, overstated by about 30%, and softer pads or gears would lower it further. The law is therefore left as it is, reading the simulator's peak. The remedy is an operating one, not a weaker judge: the close time's bound goes from 0.2..1.0 s to 0.2..3.0 s, since a real controller can close slowly. Peak on the dice by close time: 76 MPa at 0.2 s, 43 at 1 s, 35 at 3 s, 33 at 5 s. Even at 5 s the rocker arm is over 31.7 by 4%. My hand audit's 31.5 was optimistic, and the margin was never real.

**D-103 measured (seed 0 only, 84 tasks).** The prediction's main branch was wrong and its fallback right, more severely than written:
- `far_pivot_L90_b40`: **8 of 84** (it held 20/21 groups before), 43 refused for its printed arms.
- The same mechanism with its two crank arms changed from 6 x 6 to 9 x 9 mm, a hand check that the law binds on a variable System 2 can change, not a result and not adopted: **82 of 84**, failing only `rva000_mustard_bottle` and `rte000_mustard_bottle`.
- `long_slide_r52_farplane_roller`: **51 of 84, 9/21 groups** (17 before). It lost far more than its widest objects, mostly to a rack lifting about 1.0 mm off its gear. The 10 x 10 mm roller arm of the prediction has not been tried.
- A rod of half the depth buckles, as predicted.
- Stock, which the law does not read and only the wider close time touches: 68 of 84, 17/21 on this one seed.

No body has yet been rechecked on several seeds under this law; the 2517 of 2520 of D-102 was measured without it and no longer stands as this body's score.


## D-104 (judge: mechanism): a printed strut may carry its Euler load over the print factor

**Why.** D-103 compares a strut's load, and D-100 a rack's, with the ideal Euler load. That is the load of a perfectly straight column on concentric pins, which a printed part is not, while the same parts' stress carries a factor of 1.5. System 2's `deep_cranks` replays at 192 N on a drive rod whose Euler load is 186 N: its arms have a margin of four and its rod has none. Design references say a working load must stay well below the theoretical one, without giving a number for printed struts, and I found no printed-PLA knockdown for solid struts. So this uses the factor the stress already uses, 1.5, as a judgement, not a measured value.

**What.** A strut or rack in compression may carry its Euler load over `PRINT_FACTOR`. The readings stay physical, and the refusal names both numbers.

**Prediction.** On seed 0, `deep_cranks` passes at least 75 of 84, losing runs to its rod (124 N allowed, 106 N steady), with System 1 closing slower. A campaign from it deepens the rod within a few iterations.

**Not judged: the servo's shaft.** The drive rod puts its full force on the horn: about 106 N steady, six or so times what the stock jaw does. No radial rating is published for the STS3215; ROBOTIS rates the comparable XM430 and XH430 at 40 N, 10 mm from the horn. That is another servo's number, so it does not become a law here, and only a bench test on the real servo can settle it. It is the largest open doubt about any jaw of this family as a real part.

**Campaign I, under D-103, before this.** Three seeds started from `far_pivot_L90_b40`, which held 1 group, and System 2 found the law's variable on the first iteration: `stout_arms`, `stout_cranks` and `deep_cranks` each widened the crank arms in plane to about 15 x 7 mm and rose to 20 groups. `deep_cranks`, with 6 mm pads, reached 21 on train under four seeds on iteration 2. On unseen seeds under D-103 it passed **2099 of 2100 runs over 25 seeds, 24 of them clean**, failing one mustard-bottle run. Its worst held arm stress over one seed's 84 tasks was 7.6 MPa against 31.7, and its drive rod reached 192 N against an Euler load of 186 N in a replay.

**D-104 measured.** `deep_cranks` under this law, seeds 2000 to 2004: 79, 77, 76, 72 and 75 of 84, so 379 of 420, about as predicted, and 5/21 groups held under every seed. Those seeds are one judge later than D-103's and the rest of the change is the factor. The other two seeds of campaign I were stopped at 20 groups, since they ran on the older law. Campaign J starts from `deep_cranks` under this law.

**Campaign J, first result.** Seed 1 widened the rod to 20 x 4.2 mm on its first iteration, arguing from Euler that the rod's depth is boxed in by the cranks and its weak-axis stiffness grows linearly with width: `wide_rod`, 21 on train under four seeds. Unseen seeds 3000 to 3004: **418 of 420**, 20/21 under every seed. Both failures are the mustard bottle on test: a seed's draw of friction, stiffness and perceived pose under which all 7 ways fail, the moving jaw 4 to 8 mm above the fixed one and pushing the bottle 33 to 39 degrees down, so the fixed jaw never touches it. The same tasks under other seeds pass in 1 to 3 attempts.
