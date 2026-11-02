# The gripper experiment: rungs 4 and 5 first

This is the headline experiment, and it comes before any other use of rungs 4 and 5.

1. **Phase A, learning.** A stock SO-101 gripper learns how it is used.
2. **Phase B, the change.** It meets objects it cannot handle, and the loop changes the gripper
   through CAD to handle them.
3. **Phase C, the arm.** We do the same for the arm: first as a body change (link lengths, servo
   voltage), then as a repair after injected arm faults.

The experiment has to show three things:
- the loop escalates for the right reason;
- the change it makes is a well-parameterized, space-efficient CAD edit;
- the change fixes the new case without breaking what the gripper already did.

Status markers: **built** means merged with tests; **design** means agreed but not built.
Decisions referenced as D-nnn are in [decisions.md](decisions.md).

## 1. The scenario

### Phase A: the gripper learns how it is used

The world is a table with rigid objects that the stock gripper can handle, as measured by the
stock capability table (V1, D-032): cylinders 25–45 mm wide and boxes 20–50 mm wide, weighing
50–800 g. The task is to pick one up, lift it 30 mm and hold it.

- System 1 tunes the grasp: grip depth, approach offset, close speed and grip target.
- The library promotes grasp programs that keep working.
- Memory records the usage profile: sizes, grip openings, jaw pressures, payloads.

When phase A ends, every object the stock gripper passes becomes the **regression set** for any
gripper change. This mirrors SWE-bench's PASS_TO_PASS tests: a change that fixes a new object
while breaking an old one is rejected. That is what "the gripper's history comes first" means in
the harness.

### Phase B: the incompatible object

Each phase-B object fails for one known reason and comes with a hand-made gold design that passes:

| Object | Why the stock gripper fails | Expected escalation | Gold design (to be confirmed in V2) |
|---|---|---|---|
| 50–70 mm can | Wider than the stable opening; fails identically at friction 0.3, 0.8 and 1.5, so it is geometry (D-032) | `structural`: typecheck aperture rule | Fingers with pads angled so the faces are parallel at 60 mm |
| 1.5 mm plate lying flat | Jaw tips cannot get under the edge | `s1_exhausted` | Fingers with a 1 mm lip |
| 20–40 mm sphere | Slips out of a grip at the equator at any friction; holds only with the tips 12 mm below it, and creeps upward (D-032) | `s1_exhausted` / `surprise` | V-groove or concave pads |
| 600 g block, far edge of the table | Shoulder near stall; the arm sags (wrist current, payload) | `surprise` on current | 12 V shoulder servo (L3), or a shorter forearm (L2) |
| Object 0.4 m out | Outside the reach grid | `structural`: reach | Longer links (L2) |

The world has no tool rack (`rack_slots: 0`), so rungs 2–3 are ruled out as unavailable and the
first body change is a finger (rung 4). The global ladder keeps its order, so panel experiments
E1–E11 stay comparable.

### The objects: 21 real ones (D-038)

The synthetic phase-B list above was the plan. The experiment now runs on real objects: 21 YCB
scans (Calli et al. 2015) with their published masses, imported by `scripts/import_ycb.py` into
`configs/ycb_objects.yaml` and `assets/ycb/` (convex collision pieces, a decimated visual mesh, the
scan's inertia at the published mass). Each rests in a pose no taller than 85 mm, because a
straight-down SO-101 tool point tops out near 95 mm and a taller object cannot be lifted 30 mm.

Phase A and phase B are split by measurement, not by a table. `scripts/measure_real.py` runs
`pick_up` with System 1's search and confirmation re-runs on each object at three placements:

- **Phase A, held by the stock jaws:** banana, cup, foam brick, large marker, wood cube. These five
  are the regression set.
- **Phase B, not held (16 objects):**
  - slip out, mostly round: golf ball, tennis ball, racquetball, lemon, plum, apple, strawberry,
    soup can on its side, screwdriver, dice;
  - stall the arm (`load_limit`): tuna can, pudding box;
  - too wide: sugar box (92 mm), mustard bottle (96 mm);
  - the gelatin box;
  - the potted meat can, which trips the force limit.

The stored result is `results/real_stock.json`.

**Re-measured with friction that does not creep (D-048).** At MuJoCo's default `impratio` of 1, soft
friction let a pinched object slide out under a steady load. A 50 g plum squeezed at 30 N fell, which
real TPU would not allow. With `impratio` 10 in grasp scenes, the stock jaws hold **13/21**:
- **phase A** is the five objects above, plus the potted meat can, tomato soup can, tennis ball,
  racquetball, golf ball, lemon, screwdriver and dice;
- **phase B** is now eight objects:
  - the gelatin, pudding and sugar boxes and the mustard bottle, wide flat sides on splayed faces;
  - the tuna can, an arm stall while a jaw presses its top;
  - the strawberry, plum and apple, each failing one placement.

The 6/21 split above is kept in `results/superseded/impratio1/`. So is everything built on it: the
gold check, and the first live run below.

The gold check (V2) runs the same script on finger designs (`results/gold/`). Two designs hold
9/21 objects and lose none of phase A:

| Design | Objects gained |
|---|---|
| reach 10 mm, pads tilted 20°, 2 mm TPU | golf ball, lemon, plum, strawberry |
| reach 15 mm, 30°, 2 mm TPU | gelatin box, golf ball, plum, strawberry |

Tilting further holds more round objects but costs the small ones. Pads at 40° with a V-groove,
3 mm TPU and 20 mm wide hold 11/21 but drop the cube, the marker and the banana. At a small
opening a steep pad makes a V that squeezes a small object out. That trade-off is the
experiment's design problem.

**First live run (D-039, `results/experiments/`):** 8 iterations, System 2 on the Claude plan
through the Claude Code CLI, fingers only.

- **Result:** on the 42 held-out test tables, coverage went from 0.286 for the System-1-only control
  to 0.429. The arm holds 8 objects at both placements against the control's 5, with no phase-A
  loss.
- **The accepted design** came from System 2's first proposal: 25° tilt, 3 mm V-groove, 2 mm TPU,
  costing about 14 print minutes. Measured like the stock jaws, it holds 8/21. It gains the golf
  ball, the soup can on its side and the screwdriver.
- **Every later finger proposal was rejected.** Four of them tilted the pads 38–40°. Two of those
  flipped the gelatin box, two flipped the tennis ball and racquetball, and all four broke the banana
  and the cube: the gold-check trade-off, found against the judge.
- **The run exposed three harness gaps**, fixed in D-039: edits counted by key, past fixes that did
  not name what they broke, and finger changes left unpriced.

### Phase C: the arm itself (on the radar)

Phase C injects arm faults into the simulator but not into the robot's own model:
- a weakened servo (torque derated 30–50%);
- added joint friction;
- a heavier link;
- an encoder offset.

The robot should notice each fault from its own sensors, as a payload or torque residual
(section 4). It should then propose the design edit that restores the task under the fault: a
12 V servo, a shorter link, a counter-mass. Detecting the fault is a sensing test, and repairing it
is a rung-5 test. D-031 already shows the detection half for one fault: 80 g of extra mass at the
gripper reads as an 80 g phantom payload with empty jaws.

## 2. Validation ladder

The rule we borrow from SWE-bench: **validate the task before validating the loop.** Every
incompatible object needs a stock run that fails with its declared verdict and a gold design that
passes. If the gold design fails, the fault is in physics or skills, and no proposer result means
anything until that is fixed.

| Layer | Rung 4 (fingers) | Rung 5 (arm, servos) | Cost |
|---|---|---|---|
| **V1: build and model** | Every point of the finger family builds: valid and watertight, walls of at least `min_wall`, no jaw-to-jaw interference over the full sweep, fits the jaw mount, under the mass limit. Checked at the corners plus 200 random interior samples. Stock gripper fixture table (opening, squeeze, envelope). | URDF and MJCF give the same forward kinematics (`test_model_consistency`, L2 half). Mass scales with volume. Torque and self-collision checks. Reach grid recomputed for each design. | Seconds |
| **V2: gold designs** | Gold passes at least 95% over 5 seeds; stock fails with its declared verdict; nothing in the phase-A regression set breaks. | Same test, with a gold arm or servo. | About 3 min |
| **V3: difficulty** | Random search and CMA-ES each try 200 finger designs per object type. The task should be neither trivial nor impossible, and the accepted design should sit on or near the capability-vs-cost front. | Same, over link scales and voltages. | About 1 h |
| **V4: the loop** | Correct escalation reason; the proposal is at rung 4 with rungs 0–3 ruled out; the failing cases now pass, the regression set still passes, and the cost is recorded. 6 proposers × 5 seeds, with memory on and off. | Same, with L2 and L3 enabled; phase C fault cases. | About 2 h per set |
| **V5: real** | Print the accepted fingers; run both object sets on a real SO-101; check that sim and real verdicts agree. | Servo swaps are cheap to test; link changes need reprinted parts. | Bench time |

## 3. The finger family: strong parameterization, efficient use of space

We want to show that, inside the architecture's limits, the loop can design a gripper with
precise parametric control over the CAD while using space efficiently. Four rules govern the
family. `design/fingers.py` builds it (D-038): one pad box per jaw, over the bottom 10 mm of the
stock jaw face, plus V-groove halves and a lip when those are set. The simulator collides with
the pads, and the ledger prices their print volume.

1. **Parameters are physical and mostly independent.** Each one changes one capability:

   | Parameter | Range | Capability it controls |
   |---|---|---|
   | `pad_angle_deg` | 0–45 | The jaw opening at which the two pad faces are parallel. The moving pad is lengthened by its tip's rise, so both pads end level there. Confirmed in V2: tilt is what holds round objects, and too much loses small ones. |
   | `reach_mm` | 0–30 | How far past the stock tip the pads extend: grasp depth and low objects. |
   | `groove_mm` | 0–6 | Depth of a V-groove across each face: it centres round objects. (A concave profile is not built.) |
   | `lip_mm` | 0–4 | An inward hook at the pad bottom, for an edge or a plate lying flat. |
   | `pad_tpu_mm` | 0–3 | A TPU skin: friction 0.9 instead of 0.45 for PLA. Only the friction is modelled (D-038); it adds a TPU print. |
   | `width_mm` | 8–24 | Pad width across the jaw (12 mm when 0). |

2. **Every point in the box builds.** The ranges come from the hard limits: minimum wall, maximum
   added mass, wrist torque, clearance over the jaw sweep, printer envelope. So a sampled design
   is never rejected for a reason the space could have excluded. Cheap analytic checks run before
   CAD, as the typecheck does for programs.

3. **Capability is computed, not guessed.** Each finger reports its derived numbers: maximum and
   minimum stable width, squeeze at the pad (servo torque over the lever arm), contact area, added
   mass and print minutes. The typecheck's width rule and the System 2 prompt both read these
   numbers.
   - As built, one instrument measures every design: the V1 probe, a fixed pick at 50 g and
     friction 0.8. The stock jaws read its stored table. A pad design is probed on demand and
     cached by its parameters and the probe's source (`design/grip.py`, about 6 s each).
   - Example: pads tilted 40° with a V-groove hold spheres to 65 mm and cylinders to 70 mm, against
     the stock jaws' 40 and 45.
   - The aperture rule applies only to the analytic shapes the probe uses. A scanned object's
     bounding width is not the width the jaws close on (a lying banana, a cup's rim), so its trial
     decides.
   - Added mass and print minutes come from the pad volumes.

4. **Space efficiency is measured.** The finger stays within the stock jaw's swept envelope plus
   its extension. V3 plots capability (objects graspable) against added grams and print minutes.
   The ledger's value-per-cost ranking already prefers the design nearest that front, and the
   report should show where the accepted design sits on it.

## 4. Sensing: more than vision

**Built (D-031):** `sim/sensors.py` and `configs/sensors.yaml`. The rule is **the robot reads
sensors, the judge reads ground truth.** Sensor noise can change what the robot decides, but
never what counts as a pass.

| Channel | Real part | What it tells the robot |
|---|---|---|
| Servo position, speed | STS3215 telemetry | Where the arm is, and whether it is still |
| Signed load, current | STS3215 load and current registers | Effort per joint; overload; a squeeze held at stall |
| Jaw pressure (2 pads) | FSR on each jaw face (added part) | Contact, squeeze, slip (pressure drop under constant command) |
| Contact force per link | Contact skin (research option) | Unintended collisions, and where they happen |
| Wrist accelerometer and gyro | 6-axis IMU on the gripper (added part) | Impacts, vibration, drops |
| Payload (derived) | Load registers plus the arm's own model | How much the arm is holding (50 / 200 / 398 g measured for 50 / 200 / 400 g) |
| Kinetic energy, momentum (derived) | Position and speed plus the arm's own model | How hard the arm is moving; impact risk |

Derived signals use the robot's own nominal model (`SelfModel`), so faults injected into the
simulator show up as residuals (phase C).

**Sensors are resources (D-033, built).** `configs/sensors.yaml` lists the mountable parts, each
with its channels, mount, priority, whether it can be toggled, whether it exists on the bench, and
its mass. `Design.sensors` says which toggleable parts are on this body, so the sensor suite
evolves with the gripper. An unmounted part reads `None`, not zero. The System 2 prompt includes
the manifest, so a plan never relies on a sensor the body does not carry.

| Part | Priority | Toggle | Bench | Default |
|---|---|---|---|---|
| Servo telemetry | core | fixed | yes | on |
| Wrist IMU | **high**: integrated and built first | fixed | yes | on |
| FSR jaw pads | optional | **toggleable** | yes | on |
| Link skin | research | toggleable | sim only | off |

**Integration (D-036, built).** What the robot feels now reaches every consumer except the judge:
- **Traces:** 10 Hz sensor samples plus per-attempt features: peak wrist acceleration, impacts
  over 1 g, stall time and joint, pad peak, pad drop, payload while carrying.
- **Memory:** peak wrist acceleration and stall time are measurements, so a sensor residual counts
  as surprise.
- **System 2:** the failing-controls section has a `felt:` line, including which reflex stopped the
  attempt.
- **System 1 reflexes while holding:** `load_limit` (a stalled joint falling behind its command)
  and `slip` (the pads lose the grip; needs the pads).

The reflexes change the diagnosis, never the verdict: both batteries are identical.

Still to do: the panel primitives' guards read ground-truth contact force, which the real arm
does not have. Moving them to felt signals is a separate, re-baselined change before V5.

## 5. The arm tool server (MCP) and the intent router (design)

The goal is to make the harness work more like a coding harness. In SWE-bench-style agents, the
model acts through a few well-designed tools (read, edit, run the tests), and tool design is much
of the performance (SWE-agent's agent-computer interface). Here the "repository" is the arm, so
everything that reads or drives it goes through one tool server, `sea-arm`, with two backends: the
MuJoCo sim and, later, the real SO-101 on its servo bus. Sim-to-real then means swapping the
backend behind the same tools.

| Intent | Tools | Effect | Cost and gating |
|---|---|---|---|
| **sense** | `arm.state`, `arm.touch`, `arm.imu`, `arm.payload`, `scene.objects`, `design.current` | Read only | Free |
| **analyze** | `sim.dry_run(program, design?)`, `program.typecheck`, `reach.check`, `memory.similar`, `trace.get`, `s1.tune` | Compute only; never changes the live arm or body | Charged in sim-seconds to the ledger |
| **act** | `arm.move_to`, `arm.grip`, `arm.run_program`, `arm.stop` | Moves the arm | Limits enforced in the server; on the real arm, each session needs approval |
| **design** | `design.build(delta)`, `design.submit(ProposalBatch)` | Builds a candidate; submitting starts the paid acceptance trial | Print and mass priced; acceptance stays with the harness |

**Trust boundary.**
- The server loads the hard limits from the run spec and checks every act call itself: workspace
  box, speed, force cap, banned contacts. Tool arguments can never carry limits.
- Tool results are data, never instructions.
- `design.submit` is the only path to a body change, and it still goes through paired trials and
  the reference-panel veto. So an agent with tools is no more trusted than one that writes a
  single proposal.
- Every call goes to `runs/<id>/tool_calls.jsonl` with its intent and cost, and is recorded for
  replay. An agentic session then replays as deterministically as `MockLLM` does today.

**System 2 as an agent.** The plan backend (D-029) already runs Claude Code, which speaks MCP
natively. So a `s2_agent` proposer runs `claude -p --mcp-config sea-arm.json`, allowed only sense,
analyze, design and goal tools. It is the target System 2 (D-034). It decomposes a goal into
sub-goals, assigns the executable ones to System 1, reads each report and revises: it can dry-run a
hypothesis, read the sensor summary, ask System 1 to tune a program, and submit a body change. The
task network it works over is in [goals.md](goals.md). The one-shot `s2` stays as the fast path for
impasses whose fix memory already knows. Comparing the two (does decomposition buy better results
per plan call and per sim-hour?) is one of the V4 measurements.

**The intent router.** Each tool call is routed by intent:
1. **Who serves it.** System 1 serves fast local skill execution and local search; the simulator
   serves dry runs; the real arm serves act calls on hardware.
2. **Which budget it charges.** Sim-hours, real-arm minutes, approvals or print.
3. **Whether a person must approve it.**
4. **What it corroborates.** A System 2 proposal already carries a hypothesis and a `Prediction`.
   The router links later sense and analyze results to that prediction. For example, "the opening
   is too small" predicts zero pad pressure at contact and a displaced object, and a dry run either
   confirms or refutes it before any paid trial. On the System 1 side, a typed intent such as
   (verb, target, contact type) resolves to library programs through the same lookup as
   `programs_for`, plus memory retrieval.

The router starts as a rule table (tool name → intent). A learned classifier comes later, and
only for free-form program arguments. The router never decides success.

## 6. LLM billing: the Claude plan, not API credits

**Built (D-029).** The default `llm: claude_code` runs System 2 through the logged-in Claude Code
CLI, so calls bill against the claude.ai plan. It refuses to run otherwise unless
`require_plan=False` is set explicitly. The ledger still charges each call's list-price cost, so
comparisons between proposers stay fair.

One measured call: 25 s, 6.1k input tokens, 2.3k output tokens, $0.095 at list price.

- On a plan, the limit becomes the usage window, not dollars. A limit error raises
  `PlanLimitReached` and stops the run cleanly. Resuming from the last iteration is on the build
  list.
- Run seeds sequentially or with few in parallel; parallel runs share one window.
- An agentic `s2_agent` uses several times more tokens per proposal than one-shot `s2`.
- If the CLI is not on PATH, set `SEA_CLAUDE_BIN`. The VS Code extension's bundled CLI is found
  automatically.

## 7. Compute and resources

Measured on an M4 Pro, single core:
- 0.22–0.75 s per simulated control;
- 0.45 s per CoACD build;
- 5.4 s per reach grid;
- about 0.5 s per pick in the grasp scene.

No GPU is needed. Projected:

| Work | Core-hours | Wall time on this laptop |
|---|---|---|
| V1 | seconds | Part of the fast tests |
| V2: 8 objects × stock and gold × 5 seeds | about 0.3 | 2–3 min |
| V3: 1,600 designs | about 11 | about 1 h |
| V4: headline gripper experiment, 30 runs | 15–25 | about 2 h, plus about 3 h of LLM time for the 10 System 2 runs (40 calls × 25 s each) |
| V4: rung-5 versions, 30 runs | 15–25 | about 2 h |
| Full suite, about 225 runs | up to about 250 | about a day, or about 4 h on 64 cores |

Hardware for V5:
- a real SO-101, plus 12 V STS3215 servos for L3;
- a printer with PLA and TPU, and calipers;
- the object set at the simulated sizes and masses;
- FSR pads and a 6-axis IMU, if the sensing channels are to exist on the bench;
- a force gauge or load cell, and a scale;
- a fixture plate that puts objects at known poses.

## 8. Build order

| Step | Status |
|---|---|
| Fingerprint covers every module (D-028) | built |
| Plan-billed System 2 backend (D-029) | built |
| Scene objects and grasp envelope (D-030) | built |
| Sensing v0 (D-031) | built |
| Stock gripper capability table (V1) and jaw-object friction pairs (D-032) | built |
| Grasp world v0: objects as controls on a table, `held_for` check, aperture typecheck rule, `grasp` primitive, `configs/limits_grasp.yaml` (pinch allowed), grasp battery (D-035) | built |
| Sensors as resources: parts manifest, toggles, System 2 prompt section (D-033) | built |
| Sensors in the loop: traces, memory measurements, System 2 `felt:` line, `load_limit` and `slip` reflexes (D-036) | built |
| Task network contracts (`plan/network.py`, `verify/goals.py`); the current loop recorded as the degenerate network in `goals.jsonl`, every number unchanged (D-034, D-037) | built |
| Finger family: pads built into the model, priced by the ledger, pad-aware grasp and probe (D-038) | built |
| Real objects: 21 YCB scans, phase A/B split by measurement, gold check (V2) over finger designs (D-038) | built |
| Live gripper experiment (`configs/exp/gripper_live.yaml`) and its System-1-only control | first run done (D-039) |
| Evolution core: the gripper as a world (jaw joints, pads and arm servos as parts with bills of materials), frontier, tuning, merging (D-040–D-047) | built; `gripper_s2_s0` 6→13/21, `gripper_s2v2_s0` 6→19/21 in 4 iterations (old physics) |
| Grasp-scene `impratio` 10, every grasp ruler re-measured (D-048) | built |
| **21/21 on the corrected physics** (`results/evolve/gripper_s2v3_s{0,1,2}.json`): all three System 2 seeds redesigned the jaw into the slide ($5.40 of rods and bushings) and tuned its pads; train 21/21 in 4, 2 and 11 iterations; test 19, 20 and 21 of 21 against stock 15, 14 and 13; the random arm at the same commit reached 17 train, 15 test | done |
| Finger CAD export (STEP/STL for printing), difficulty check (V3) | design |
| `sea-arm` MCP server over the sim: sense, analyze and goal tools, call log and replay, agentic `s2_agent` over the task network | design |
| `design/arm.py` (L2), actuator (L3), phase C fault injection | design |
| Live runs (V4); real prints and the SO-101 backend (V5) | later |

Additive judge-tier changes, such as new object classes or a new limits file, re-key every world.
The procedure is the one D-028 followed: re-run B3 and re-baseline only after showing that the old
world's numbers are identical.
