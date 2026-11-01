# Self-Evolving Arm (sim): implementation spec

Version 2.1, Oct 31, 2026 (reviewed; Appendices A–D are normative). Owner: Karthik Singaravadivelan. Companion to the MakeCU self-evolving hardware project, which shares this spec's data contracts and verifier.

This document is written to be handed to a coding agent. It has everything needed to build the repository: goals, the theory the code must reflect, exact dependencies, data contracts, module interfaces, algorithms, tests, coverage gates, and milestones with acceptance checks. Where a choice is left open, it's marked **OPEN** with a default the agent should use.

---

## 0. Instructions for the coding agent

1. Read the whole spec before writing code. Sections 3 (theory) and 6 (contracts) constrain everything else.
2. Build in the milestone order in Section 13. Don't start a milestone until the previous one's acceptance checks pass.
3. Every module gets unit tests as it's written. Coverage gates (Section 12) are enforced in CI from milestone 1.
4. All randomness goes through a seeded `numpy.random.Generator` or `torch.Generator` created from the run config. The Cobweb library uses Python's global `random` for tie-breaks, so every `ConceptMemory` call runs inside a context that sets `random.seed`, `torch.manual_seed`, and forces `device="cpu"`. Two runs with the same config and seed on the same platform must produce byte-identical `report.json` files. Wall-clock fields are kept out of `report.json`. Live LLM calls are the only other exception; they're cached by request hash.
5. Never call a live LLM in tests. Use `MockLLM` (Section 8.6), which replays fixtures from `tests/fixtures/llm/`.
6. Don't add dependencies beyond Section 5 without writing the reason in `docs/decisions.md`.
7. Hard limits (Section 9) are enforced in code paths the LLM can't reach. No LLM output may change a hard limit, the reward code, or the verifier.
8. If something in this spec is wrong or impossible (an API differs from what's described here), stop, write the issue in `docs/decisions.md` with the fix you chose, and continue.

---

## 1. Goal

The MakeCU loop changes the **product** until it passes its tests. This repo changes the **robot**, in simulation, until it can do its tasks.

A simulated SO-101 arm faces a suite of procedurally generated appliance panels: buttons, knobs, levers, doors, drawers, sliders, slots. It runs a coverage test on each panel. When it fails, two kinds of reasoning take over:

- **System 1:** fast, low-level control and trial and error. It tunes skill parameters and retries.
- **System 2:** a deliberate decision maker. It diagnoses the failure and resolves it by composing existing primitives in a new way. Only when composition can't solve it does it change the robot's body: a new tool, different fingers, a longer link.

Every change is scored by the same verifiable reward, and kept only if the reward rises with no regressions. Successful compositions are promoted into new primitives, so the robot's repertoire grows over time.

**Main question:** does a System 1 / System 2 loop that composes primitives, backed by a Cobweb concept memory, reach high coverage with fewer resources than standard optimizers? And do its compositions transfer to panels that need new combinations of known pieces?

**Hypotheses**

| ID | Hypothesis | Experiment |
|---|---|---|
| H1 | The S1/S2 loop reaches 90% coverage with fewer design iterations and print minutes than random search, CMA-ES, or Bayesian optimization over the same parameters | E1 |
| H2 | Choosing targets by concept-memory uncertainty beats choosing by coverage gaps alone | E2 |
| H3 | Designs found on train panels lose less than 10 coverage points on held-out panels | E3 |
| H4 | Stored design records make later fixes faster: iterations per fix fall over time | E4 |
| H5 | Annealing posture from optimistic to conservative as budgets run down beats every fixed posture at equal budget | E8 |
| H6 | Compositional generalization: on panels that need new combinations of known primitives, S2 composition solves at least 2× more controls zero-shot than an S1-only baseline, and needs no body change for at least half of them | E10 |
| H7 | Library learning: promoted primitives cut the S2 calls per solved control by at least 30% in the second half of a run | E11 |

---

## 2. Scope

**In scope:** MuJoCo simulation only, procedural panels, a parametric SO-101, parametric tools, scripted skills with guarded contact, a compositional program language, an LLM System 2, Cobweb memory, posture specs, a resource ledger, experiments, and plots.

**Out of scope for v1:** a real robot, real printing, learned visual policies, and multi-arm setups. The optional world model (Section 3.6) comes after milestone 8.

---

## 3. Theory the code must reflect

### 3.1 Primitives

A **primitive** is a typed, parameterized unit with preconditions and effects. There are three kinds.

| Kind | Examples | Parameters | Executed by |
|---|---|---|---|
| Motor | `approach`, `contact`, `push`, `rotate`, `hook_pull`, `slide`, `insert`, `release`, `retreat` | Target frame, direction, force cap, travel or angle, speed | System 1 controller |
| Body | Tool templates (`pad`, `hook`, `knob_cup`, `soft_tip`, `slot_inserter`, `probe`), finger parts, link scale modules | Geometry (mm), material contact parameters | Build pipeline: CadQuery to MJCF |
| Check | `moved_at_least(joint, d)`, `latched(control)`, `force_below(cap)`, `no_banned_contact()`, `detent_crossed(n)` | Thresholds | Verifier, from sim state |

Each primitive declares a **type signature**:
- `requires`: a contact type (`pad`, `hook`, `cup`, `probe`, `edge`), minimum reach, and minimum force.
- `provides`: an effect on a control class, for example `actuates: push_button` or `opens: door[handle=push_latch]`.

### 3.2 Composition

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

### 3.3 System 1 and System 2

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

### 3.4 Escalation ladder

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

### 3.5 Library learning (growing the repertoire)

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

### 3.6 Optional world model (after milestone 8)

The user may want a JEPA-style world model as part of System 2.

**What it is.** A small latent model trained on logged transitions: (context embedding, program embedding) → predicted outcome embedding plus pass probability. It plays the same role as action-conditioned world models such as V-JEPA 2-AC.

**How it's used.** Only to **rank** System 2's candidate proposals before simulation ("imagination"). It never replaces the verifier.

**Requirements:**
- PyTorch, under 5M parameters, trained on CPU from the run logs.
- Acceptance: on held-out logs, ranking with the model raises the top-1 proposal pass rate by at least 10 points over random ordering. If it doesn't, the feature stays off.

### 3.7 Concept memory's role

Cobweb organizes everything the robot learns into concepts that people can read. There are four trees; three are listed here, and the fourth ("full", used only to score surprise) is defined in Section 8.5.
- **Context tree:** what controls look like (embeddings) → how they behave. It supplies expectations, surprise, and uncertainty.
- **Fix tree:** text embeddings of root causes → the fixes that worked.
- **Program tree:** embeddings of programs and their outcome signatures → clusters of similar strategies. It's used for de-duplication and to suggest abstractions.

Uncertainty from the context tree feeds the posture rule (Section 9) and target selection (Section 8.4).

---

## 4. The loop (reference pseudocode)

```python
def evolve(cfg):
    state = init_state(cfg)                     # design, library, memory, ledger, posture
    train = generate("train", cfg.splits["train"], cfg.panel_seed)
    stall = 0
    for it in range(cfg.budgets.design_iterations):
        panels = sample_iteration_panels(train, state, cfg.iter_sample)   # failing panels first, then a fixed-seed sample
        report = run_suite(state.design, state.library, panels=panels)    # System 1 inside
        state.memory.learn(report)              # passes now; failures after confirmation (Sec 8.5)
        if report.coverage_reward >= cfg.stop.target_reward: break
        escalations = system1_escalations(report, state)          # Sec 3.3 rule
        if not escalations: continue
        targets = pick_targets(escalations, state.memory, state.ledger, state.posture)   # Sec 8.4
        proposals = system2_propose(targets, report, state)       # LLM; may return several candidates
        proposals = [p for p in proposals if typecheck(p, state) and build_checks(p, state)]
        if state.world_model: proposals = state.world_model.rank(proposals)
        p, vpc = best_by_value_per_cost(proposals, state.ledger, state.posture)   # Sec 10
        if p is None or vpc < cfg.stop.min_value_per_cost or state.ledger.would_break_reserve(p.cost):
            stall += 1
            if stall >= cfg.stop.patience: break
            continue
        stall = 0
        trial = run_suite(apply(state, p), state.library_with(p), panels=targets.panels + [REFERENCE])
        decision = accept(trial, report, state.posture)           # lcb | mean | ucb
        state.ledger.charge(p.cost_actual(trial))
        state.memory.record_fix(p, trial, decision)
        if decision.accepted:
            state = state.commit(p)
            maybe_promote(state)                # Sec 3.5
        state.posture = anneal(state.posture, state.ledger)       # Sec 10
        write_iteration_log(it, report, p, trial, decision, state)
    return final_eval(state, generate("test", cfg.splits["test"], cfg.panel_seed))
```

---

## 5. Environment and dependencies

- **Python:** 3.12. Manage with `uv`. Commit `uv.lock`.
- **OS:** Linux for CI; macOS on Apple Silicon must also work for development.
- **CI:** GitHub Actions, `ubuntu-latest`, CPU only.

| Package | Version | Use |
|---|---|---|
| mujoco | ≥ 3.2 (pin the exact version in uv.lock) | Physics; `mujoco.MjSpec` to compose models |
| placo | 0.10.2 | IK on generated URDF (0.9.x failed to import on Linux in our test) |
| cadquery | 2.8.x | Parametric tools and links; kernel checks |
| coacd | latest | Convex decomposition of tool meshes for collision |
| trimesh | latest | Mesh I/O, volume, mass |
| numpy, scipy | latest | Math |
| torch | ≥ 2.0 (CPU in CI) | Cobweb trees, embeddings, optional world model |
| cobweb-language-embedding | git commit `2f807757ad607b55e1f5ac258ca2ce5dda2f34fa` | Concept memory |
| sentence-transformers | ≥ 2.5 | Text embeddings (default `all-MiniLM-L6-v2`, 384-d) |
| open_clip_torch **or** transformers (DINOv2-small) | latest | Image embeddings of control crops. **OPEN**, default: DINOv2-small via transformers, 384-d CLS |
| pydantic | 2.x | Contracts |
| anthropic | latest | System 2 LLM |
| cma | latest | System 1 local search; CMA-ES baseline |
| optuna | latest | Bayesian optimization baseline |
| rerun-sdk | latest | Trace viewer (optional) |
| pyyaml, typer, rich | latest | Config, CLI, logs |
| pytest, pytest-cov, hypothesis, pytest-timeout | latest | Tests |
| ruff, mypy | latest | Lint, types |

Install the Cobweb package with:

```
uv add "cobweb-language-embedding @ git+https://github.com/Teachable-AI-Lab/cobweb-language-embedding@2f807757ad607b55e1f5ac258ca2ce5dda2f34fa"
```

### 5.1 Facts about the stock arm (use these, don't re-derive)

**Model files.** From `github.com/TheRobotStudio/SO-ARM100`, under `Simulation/SO101/`:
- `so101_new_calib.urdf` (for placo)
- `so101_new_calib.xml` (MuJoCo)
- `scene.xml`
- `assets/`

Vendor them into `assets/so101/` with the repo's license file.

**Structure.**
- Joints, base to tip: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- Tool frame in the URDF: link `gripper_frame_link`. Its local **z axis** is the approach direction (about 4.6° off the wrist-to-tip vector at zero pose).
- Tool frame in the MJCF: site `gripperframe` (so101_new_calib.xml, line 111). Its position matches the URDF frame to under 1 µm at zero pose, but its **x axis** is the approach direction. Define a fixed rotation between the two frames in `control/frames.py`.
- Body names differ between the files (`base` in MJCF vs `base_link` in URDF). Keep a name map in `control/frames.py`.
- Placo needs an **absolute** URDF path, or it can't find the meshes.

**Servos.** STS3215.
- 7.4 V: stall torque about 16.5 kg·cm (1.62 N·m).
- 12 V: about 30 kg·cm (2.94 N·m).
- The stock MJCF actuators declare `forcerange="-3.35 3.35"`. `build_model` must overwrite each actuator's forcerange from `ArmSpec.servo_voltage`: ±1.62 N·m at 7.4 V, ±2.94 N·m at 12 V. Default: 7.4 V on every joint.

**Reach, measured Oct 31.** Placo, stock URDF, no collisions, jaw tip as the tool point:
- Straight-down presses work 0.14–0.30 m in front of the base, from 5 cm below to 5 cm above it.
- Horizontal presses work only at 0.26 m or farther, and 10 cm or more above the base.
- 45° presses work 0.22–0.34 m out, from 5 cm below to 15 cm above the base.

Use these numbers to place panels so the stock arm can solve some of the controls, but not all.

---

## 6. Data contracts (`src/sea/contracts.py`)

All contracts are Pydantic v2 models with `model_config = ConfigDict(extra="forbid", frozen=True)`, except where noted. JSON schemas for LLM prompts are generated from these models. Units: meters, radians, newtons, seconds, grams, minutes, unless a field name says otherwise.

```python
class Vec3(BaseModel): x: float; y: float; z: float
class Pose(BaseModel): pos: Vec3; quat_wxyz: tuple[float, float, float, float]

ControlClass = Literal["push_button","toggle","knob","lever","door","drawer","slider","slot","recessed_button"]
ContactType  = Literal["pad","hook","cup","probe","edge","pinch"]
Verdict      = Literal["pass","stuck","no_actuation","wrong_behavior","unreachable",
                       "rule_violation","no_applicable_primitive","inconclusive"]

class ControlSpec(BaseModel):
    id: str; cls: ControlClass; pose_panel: Pose
    size_mm: tuple[float, ...]            # class-specific: (diameter,) or (w, h) ...
    stiffness: float                      # N/m or N·m/rad
    travel: float                         # m or rad to success
    detents: list[float] = []             # positions of detent bumps (m or rad)
    detent_peak: float = 0.0              # N or N·m
    friction: float
    latch_threshold: float | None = None  # doors/drawers
    handle: Literal["none","tab","pull","push_latch"] = "none"
    recess_depth_mm: float = 0.0
    success: "CheckExpr"                  # e.g. moved_at_least(joint, travel) & detent_crossed(1)

class PanelSpec(BaseModel):
    id: str; seed: int; split: Literal["train","val","test","compose","reference"]
    pose_world: Pose; tilt_rad: float; controls: list[ControlSpec]
    rules: "Rules"

class Rules(BaseModel):
    banned_contacts: list[ContactType] = ["pinch"]
    force_cap_by_class: dict[ControlClass, float]
    time_limit_s: float = 20.0

class ToolSpec(BaseModel):
    template: Literal["stock","pad","hook","knob_cup","soft_tip","slot_inserter","probe"]
    params: dict[str, float]              # template-specific, ranges in design/space.py
    contact: ContactType
class FingerSpec(BaseModel): jaw_len_mm: float; tip_radius_mm: float; softness: float
class ArmSpec(BaseModel):
    upper_scale: float = 1.0; fore_scale: float = 1.0; wrist_offset_mm: float = 0.0
    riser_m: float = 0.0; base_pitch_rad: float = 0.0
    servo_voltage: dict[str, Literal[7.4, 12.0]] = {j: 7.4 for j in ARM_JOINTS}   # per joint
class Design(BaseModel):
    id: str; parent: str | None; arm: ArmSpec; fingers: FingerSpec
    tool_rack: list[ToolSpec]             # ≤ rack_slots
    est: "BuildEstimate"

class BuildEstimate(BaseModel):
    print_minutes: float; grams_pla: float; grams_tpu: float; added_mass_g: float

# ---- programs ----
class PrimRef(BaseModel): name: str; args: dict[str, float | str] = {}
class Node(BaseModel):
    op: Literal["prim","seq","guard","repeat","branch","with","bind"]
    prim: PrimRef | None = None; children: list["Node"] = []
    check: "CheckExpr | None" = None; n: int | None = None
    tool_index: int | None = None         # index into Design.tool_rack for "with"
    cap: "CheckExpr | None" = None        # guard only: stop when this becomes false
    bind_args: dict[str, float] = {}      # bind only: fixed arguments
class Program(BaseModel): id: str; target_cls: ControlClass; root: Node; source: Literal["seed","s2","promoted"]
class PrimitiveDef(BaseModel):
    name: str; kind: Literal["motor","check","composite"]; doc: str
    requires: dict[str, str | float]; provides: dict[str, str]
    params: dict[str, tuple[float, float]]   # name -> (lo, hi)
    body: Program | None = None              # for composites
    uses: int = 0; passes: int = 0; created_iter: int = 0

# ---- execution and verdicts ----
class StepTrace(BaseModel):                 # one control attempt, not frozen while recording
    panel_id: str; control_id: str; program_id: str; attempt: int
    t: list[float]; q: list[list[float]]; tip_force: list[float]
    control_q: list[float]; contacts: list[tuple[float, ContactType, str]]
    s1_trials: int; ik_residual_mm: float; joint_limit_hits: list[str]
class ControlResult(BaseModel):
    panel_id: str; control_id: str; cls: ControlClass; verdict: Verdict
    peak_force: float; travel_reached: float; detents: int; time_s: float
    surprise_z: dict[str, float]; surprise_nll: float; concept_n: int
    escalated: bool; escalation_reason: str | None; trace_path: str
class Report(BaseModel):
    run_id: str; iteration: int; design_id: str; panels: list[str]
    results: list[ControlResult]
    control_coverage: float; pass_rate: float; coverage_reward: float
    cost: "CostVector"; reward: float       # coverage_reward minus cost terms (Sec 8.3)

# ---- System 2 ----
class Prediction(BaseModel): flips_to_pass: list[str]; may_regress: list[str]; confidence: float
class DesignDelta(BaseModel):
    arm: dict[str, float] = {}             # partial ArmSpec fields
    fingers: dict[str, float] = {}         # partial FingerSpec fields
    servo_voltage: dict[str, Literal[7.4, 12.0]] = {}
    rack_add: list[ToolSpec] = []; rack_remove: list[int] = []   # removal must be named when the rack is full
class Proposal(BaseModel):
    kind: Literal["reparam","program","use_tool","new_tool","finger","arm","actuator","promote"]
    rung: int; rationale: str; hypothesis: str; ruled_out: list[str]
    program: Program | None = None; design_delta: DesignDelta | None = None
    reparam: dict[str, tuple[float, float]] | None = None   # primitive param -> new range
    promote: PrimitiveDef | None = None; prediction: Prediction
class ProposalBatch(BaseModel): proposals: list[Proposal]   # 1-3 items; the LLM tool's input schema
class Decision(BaseModel):
    accepted: bool; rule: Literal["lcb","mean","ucb"]; delta_mean: float
    delta_lo: float; delta_hi: float; regressions: list[str]; prediction_score: float

# ---- spec, posture, resources ----
class HardLimits(BaseModel):
    max_force_n: float = 15.0; max_tip_speed_mps: float = 0.10
    workspace_box: tuple[Vec3, Vec3]; banned_contacts: list[ContactType] = ["pinch"]
    max_tool_mass_g: float = 40.0; max_link_scale: float = 1.3; min_wall_mm: float = 1.2
class Posture(BaseModel):
    beta: float = 0.0                      # other fields: None = derive from beta (Sec 8.9)
    max_param_delta: float | None = None; max_edits: int | None = None
    repeats_before_fail: int | None = None; accept_rule: Literal["lcb","mean","ucb"] | None = None
    surprise_tau: float | None = None; regression_panel: Literal["full","sample"] | None = None
    force_cap_fraction: float | None = None; k_promote: int | None = None
    anneal_with_budget: bool = True; per_area: dict[str, float] = {}   # area -> beta override
class Budgets(BaseModel):
    print_minutes: float = 240; grams_pla: float = 300; grams_tpu: float = 50
    design_iterations: int = 30; llm_usd: float = 20.0; approvals_per_hour: int = 6
    rack_slots: int = 4; library_size: int = 40; sim_hours: float = 40.0
    reserve_fraction: float = 0.2
class CostVector(BaseModel):
    print_minutes: float = 0; grams_pla: float = 0; grams_tpu: float = 0; llm_usd: float = 0
    approvals: int = 0; sim_seconds: float = 0; servo_heat_j: float = 0
class RunSpec(BaseModel):                    # the YAML file
    seed: int; limits: HardLimits; posture: Posture; budgets: Budgets
    levels: list[Literal["L0","L1","L2","L3"]] = ["L0"]
    proposer: Literal["s2","s2_nomem","random","cma","bo","s1_only"] = "s2"
    world_model: bool = False; llm_model: str = "claude-opus-5-5"; llm_cheap_model: str = "claude-sonnet-5-5"
    splits: dict[str, int] = {"train": 200, "val": 50, "test": 100, "compose": 60}
    panel_seed: int = 1                       # panels = generate(split, n, panel_seed)
    iter_sample: int = 40                     # train panels run per iteration (Sec 4)
    stop: "StopRule" = StopRule()
    ablations: "Ablations" = Ablations()

class StopRule(BaseModel):
    target_reward: float = 0.95; min_value_per_cost: float = 0.01; patience: int = 3
class Ablations(BaseModel):
    target_mode: Literal["gap","gap_uncertainty"] = "gap_uncertainty"   # E2
    use_fix_tree: bool = True                 # E4
    library_learning: bool = True             # E11
    cost_terms: bool = True                   # E7
    reference_regression: bool = True         # E7
    zero_shot: bool = False                   # E10: evaluate without evolving on the eval split
```

`CheckExpr` is a small expression language: `moved_at_least(control, d)`, `latched(control)`, `detent_crossed(control, n)`, `force_below(N)`, `no_banned_contact()`, combined with `&`, `|`, `~`. Store it as a string and parse it with a hand-written parser in `compose/checks.py`. Don't use `eval`.

---

## 7. Simulation

### 7.1 Model composition (`sim/build_model.py`)

- Load the stock MJCF with `mujoco.MjSpec.from_file`. Attach the panel, the tools, and the generated arm parts with `MjSpec` attach APIs. If `MjSpec` attach doesn't work for a case, fall back to XML templates and note it in `docs/decisions.md`.
- Timestep 0.002 s (500 Hz). Use `implicitfast` integrator, `elliptic` friction cone, and `condim=4` on tool–control contacts.
- Each control is a body with one joint (slide or hinge) inside the panel body.

### 7.2 Control dynamics (`sim/controls.py`)

At every step, before `mj_step`, write `data.qfrc_applied` on each control joint:

```
f = -stiffness * (q - q_rest) + detent_force(q) - friction_sign_term + latch_force(q)
detent_force(q) = -sum_i detent_peak * h((q - d_i) / w)    # h(u) = u * exp(0.5 - u^2/2) / 1 → peak |h| = 1 at u = ±1
w = 0.15 * detent spacing, or 0.15 * travel for a single detent
latch_force: holds the door or drawer closed until the applied force along its axis exceeds latch_threshold
push_latch: press-to-release: pushing inward by 2 mm releases the latch, then a spring opens 15°
```

**Ground-truth signals:**
- control joint position
- detents crossed (count sign changes of `q - d_i`, with hysteresis ±0.2·w)
- latch state
- per-contact geom pairs, mapped to `ContactType` by the tool geom's name tag

### 7.3 Reach envelope (`control/reach.py`)

For each design:
- Sample a fixed 9×9×7 grid over the workspace box (origin: `base_link`).
- For 3 approach directions (down, 45°, horizontal), solve IK with up to 5 seeds, stopping at the first that converges (residual < 2 mm and axis error < 5°, at most 100 solver iterations).
- Store an occupancy grid of 0/1 per direction, cached by design hash.
- Lookup: trilinear interpolation of occupancy, reachable if ≥ 0.5. A continuous approach direction uses the nearest of the 3 bins.
- Tests inject a precomputed grid (`tests/fixtures/reach_stock.npz`).

### 7.4 Panels (`sim/panels.py`)

- `generate(split, n, seed) -> list[PanelSpec]`. Each panel has 4–8 controls, with classes sampled so every class appears at least 20 times per split.
- Parameter ranges: `configs/panels.yaml`. The test split shifts every range by +25% at the top end.
- The **compose split** is built so each panel needs a combination never seen whole in train. Bin the continuous factors first: recess {0, 0–10, >10 mm}, tilt {0–20°, 20–45°, >45°}. Then hold out specific (control class × handle × recess bin × tilt bin) combinations; each component appears in train, but the combination doesn't. Examples:
  - a push-latch door inside a 20 mm recess (needs probe + push);
  - a knob on a 60° tilted face (needs cup + wrist re-orientation);
  - a slot behind a hinged flap (needs hook_pull, then insert).
- The **reference panel** is fixed (seed 0) and contains one control of each class.

---

## 8. Module specs

### 8.1 `control/` (System 1)

- `ik.py`: `IK(urdf_abs_path).solve(pos, axis, tool_tip: Pose, seeds=5) -> (q, residual_mm, axis_err_deg)`.
  - Placo: `solver.mask_fbase(True)`, `solver.enable_joint_limits(True)`, `add_position_task(frame, target)` (weight 1.0), `add_axisalign_task(frame, [0,0,1], target_axis_world)` (weight 1.0). Don't use `add_frame_task`; it's a full 6-D task.
  - Loop: up to 100 calls of `solver.solve(True)` + `robot.update_kinematics()`; converged at residual < 1 mm and axis error < 3°.
  - Seeds: zero pose, then 4 fixed configurations listed in `configs/ik_seeds.yaml`.
  - A mounted tool moves the tool point. Each tool template defines a tip transform relative to `gripper_frame_link` (Appendix B), used by IK, reach, and typecheck.
- `motor.py`: one function per motor primitive.
  - Each returns a generator of joint targets at 50 Hz.
  - The sim runs 10 steps per target.
  - Guarded contact: stop when tip force exceeds the cap, travel is reached, or a joint hits its limit.
  - Approach offset 30 mm along the control normal. Speed ≤ hard limit; half speed within 10 mm of contact.
- `executor.py`: `run_program(model, data, program, design, panel, control, limits) -> StepTrace`. Interprets the program tree: `seq`, `guard`, `repeat`, `branch`, and `with` (tool swap is instantaneous in sim but costs 5 s of sim time and counts a swap).
- `local_search.py`: System 1's parameter search. CMA-ES over the free continuous parameters of the failing primitive, at most `m` trials (default 8). Stops at the first pass, or when the posture's `max_param_delta` bounds are exhausted.
- **Fixed retry rule:** before local search, retry once with the target re-localized and offset 1 mm toward the control center.

### 8.2 `verify/`

- `verdicts.py`: `judge(trace, control, rules, limits) -> ControlResult`. The order matters:
  0. `inconclusive` if the trace is incomplete (episode aborted or under 50% of samples).
  1. `rule_violation` if a banned contact touched the control, or force exceeded the class cap.
  2. `unreachable` if the IK residual is over 5 mm at the contact waypoint.
  3. `pass` if the control's `success` check holds within the time limit.
  4. `stuck` if force reached the cap but travel is below 30% of required.
  5. `no_actuation` if peak force is under 0.2 N.
  6. `wrong_behavior` if it moved but the success check failed (wrong direction, no latch, or no detent).
  7. `inconclusive` if contacts were lost for more than 50% of the push phase.
  - `no_applicable_primitive` is never produced by `judge()`. The executor emits it before simulation when no library program type-checks for the control.
- `coverage.py`:
  - `control_coverage = (# controls with verdict not in {no_applicable_primitive, inconclusive}) / (# controls)`
  - `pass_rate = passes / covered`, and 0 when covered = 0
  - `coverage_reward = control_coverage × pass_rate`
- `reward.py`: **standalone script** with a `main(run_dir) -> dict` function. It recomputes `coverage_reward` and `reward` from `runs/<id>/spec.yaml`, `design_<k>.json`, and `iter_<k>/results.jsonl` + `cost.json` alone. It must not import anything from `sea` except `contracts` and `verify.coverage`. Tests call `main()` in-process so it counts toward coverage.

### 8.3 Reward

```
reward = coverage_reward − λ_print·cum_print_minutes/budget.print_minutes − λ_mass·design.est.added_mass_g/limits.max_tool_mass_g
defaults: λ_print = 0.05, λ_mass = 0.02; cum_print_minutes is cumulative over the run; reward can be negative
```

Acceptance uses `reward` on the trial panels **plus** the reference panel. Any reference-panel control that was `pass` and is now not `pass` is a regression.

### 8.4 Target selection (`propose/targets.py`)

For each escalated control:

```
score = gain_est + β_area · uncertainty − Σ_r w_r · cost_est_r
```

- `gain_est`: 1 for `unreachable` and `no_applicable_primitive`; 0.7 for `stuck` and `wrong_behavior`; 0.4 otherwise. Multiply by the number of controls in the same concept (fixing one may fix its siblings).
- `uncertainty`: the mean, over predicted attributes, of the concept's standard deviation divided by its global standard deviation, plus `1/sqrt(concept_n + 1)`.
- `w_r`: scarcity weight, `1 / max(remaining_fraction_r, 0.05)`, normalized.

Pick the top `max_targets` (default 3), grouped by concept.

### 8.5 `memory/` (Cobweb)

Wrap `cobweb_language_embedding`. These are the APIs we rely on, checked against commit `2f80775`:
- Import the tree directly: `from cobweb_language_embedding.CPPCobweb import CPPCobwebTree`. It isn't exported from the package root.
- `CPPCobwebTree(shape, device="cpu")`: `.ifit(tensor, label=None) -> node`.
- `.categorize(instance, label=None, use_best=True, greedy=False, max_nodes=inf, retrieve_k=None, return_visited=False)`. Don't use `retrieve_k`: it only collects nodes with a `sentence_id` and raises IndexError when fewer than k exist. Categorize to a single leaf and walk the tree ourselves.
- `.log_prob(instance, labels_vec, max_nodes, greedy)` has no defaults. `labels_vec` is a float tensor of length `tree.num_labels` (zeros for no label).
- `.dump_json(filename)` / `.load_json(filename)`: saves only count, mean, sum_sq, and label_counts per node. It drops the label-name map and any payload attributes, and loading needs a tree built with the same `shape`.
- Nodes have `.mean`, `.count`, `.parent`, `.children`, and `.mean_var_plabels()`.
- `PCAICAWhiteningModel.fit(x, pca_dim=...)` is a classmethod that returns a new model; use `pca_dim=64` and fit on at least 64 samples. `.transform(x)`.
- We don't use `CobwebRetriever`: it can't start empty, it picks CUDA when available, and it returns no distances. The fix tree is a plain `CPPCobwebTree` with our own payloads.

If a signature differs, adapt the wrapper and note it in `docs/decisions.md`.

| Tree | Instance vector | Payload on leaf | Used for |
|---|---|---|---|
| Context | Whitened `[image_emb(control crop, 384) ⊕ text_emb("<class> <handle> recess=<bin> tilt=<bin>", 384)]`, PCA+ICA to 64-d | IDs of `ControlResult`s | Expectations, surprise, uncertainty |
| Full | Context vector ⊕ z-scored measurement vector (peak force, travel reached, detents, time, IK residual), 69-d | — | Surprise as negative log likelihood (`surprise_nll`) |
| Fix | `text_emb(hypothesis + rationale)`, 384-d | IDs of `(Proposal, Decision)` records | Retrieving past fixes for System 2 |
| Program | L2-normalized `text_emb(program pretty-print)` ⊕ 0.5 × outcome signature (pass rate by control class, 9-d) | Program IDs | De-duplication (`d_dup` = 0.15 Euclidean), abstraction candidates |

**Persistence.** `ConceptMemory.save` calls `dump_json` per tree and also writes a sidecar JSON with the label map and the payloads, keyed by each node's index in a depth-first traversal (dump order is deterministic). `load` rebuilds trees with the same shape and re-attaches the sidecar. Distances for `similar_fixes` and `similar_programs` are computed by the wrapper: Euclidean distance between the query and the payload vectors stored under the categorized leaf and its parent.

**Wrapper API (`memory/concepts.py`)**

```python
class ConceptMemory:
    def expect(self, ctx_vec) -> Expectation          # mean, sd per measurement, concept_n, node_id
    def surprise(self, ctx_vec, measured) -> tuple[dict[str, float], float]   # z per attribute, nll
    def learn(self, results: list[ControlResult], confirmed_failures: set[str]) -> None
    def similar_fixes(self, text, k=5) -> list[FixRecord]
    def similar_programs(self, program, k=5) -> list[tuple[str, float]]
    def save(self, path) / load(self, path)
```

**Rules:**
- **Expectation:** categorize to a leaf, then walk up to the **deepest ancestor holding at least `n_min` = 20 payload results**. Mean and sd come from those results' measurements. This avoids biased small leaves; we saw a 3.7 N vs 2.9 N bias from a 5-instance leaf on Oct 31.
- **Learning:**
  - Passes are added immediately.
  - In sim, a failure counts as confirmed when its verdict repeats `posture.repeats_before_fail` times. It's then added with its verdict label. Unconfirmed failures are held in a queue.
- **Order effects:** build the seed trees from 3 shuffles and keep the one with the lowest mean NLL on validation.

### 8.6 `propose/` (System 2)

- `llm.py`:
  - `class LLM(Protocol): def propose(self, prompt: S2Prompt) -> list[Proposal]`.
  - `AnthropicLLM` uses tool-use with `ProposalBatch.model_json_schema()` as the tool's `input_schema` (a top-level object), temperature 0.2, at most 3 proposals. On a schema error, retry once with the error message.
  - `apply(state, proposal)`: `program` adds a program for the targets; `reparam` hands new ranges to System 1; `use_tool`/`new_tool`/`finger`/`arm`/`actuator` apply the `DesignDelta` to a copy of the design and rebuild; `promote` adds to the library.
  - `MockLLM` replays `tests/fixtures/llm/<prompt_hash>.json` and raises if no fixture exists.
  - Every call is logged with tokens and USD.
- `prompt.py`: builds `S2Prompt`. Sections:
  1. hard limits, as unchangeable facts
  2. posture and remaining budgets
  3. the escalation ladder
  4. the primitive library: signatures and docstrings only
  5. the current design summary
  6. failing controls with trace summaries (peak force, travel, contact types, IK residual, joint limit hits, surprise)
  7. top-5 similar past fixes and their outcomes
  8. output schema and rules: up to 3 proposals, each with the `ruled_out` entries the ladder requires, a prediction, and evidence cited by field name
- `baselines.py`: `random`, `cma`, `bo`, `s1_only`, `s2_nomem`. They share the ledger and acceptance code. The `s1_only` baseline can't change program structure or the body.
- `world_model.py` (optional): see Section 3.6.

### 8.7 `compose/`

- `dsl.py`: `Node` constructors, pretty-printer, parser (round-trip tested), and hashing.
- `typecheck.py`: `typecheck(program, design, library, limits, reach) -> list[TypeError]`.
- `library.py`: the primitive store, seeded from `configs/primitives.yaml`. Implements `promote()` (Section 3.5) and `prune()`.
- `checks.py`: the `CheckExpr` parser and evaluator.

**Seed library** (in `configs/primitives.yaml`):
- the 9 motor primitives
- 5 checks
- 6 seed programs: `press_button`, `flip_toggle`, `turn_knob`, `pull_lever`, `open_door_pull`, `slide_slider`

### 8.8 `design/`

- `space.py`: parameter ranges per level, and encode/decode to a vector for the CMA and BO baselines.
- `tools.py`: CadQuery templates. Each takes params, returns a solid, and has a `contact` type and a mount interface: a 20 × 20 mm dovetail at the jaw.

| Template | Parameters |
|---|---|
| pad | w, h (8–30 mm), thickness (3–8), friction (0.4–1.2) |
| hook | depth (4–20), throat (3–12), length (20–60) |
| knob_cup | inner diameter (8–40), depth (4–15), rib count (0–8) |
| soft_tip | radius (2–8), softness (0–1) → MuJoCo solref/solimp |
| slot_inserter | blade thickness (0.6–2), width (10–40), length (20–60) |
| probe | diameter (3–8), length (20–80) |

- `arm.py`: generates URDF and MJCF from `ArmSpec`. It scales link origins and visual and collision meshes along the link axis, and offsets the wrist. Masses scale with volume.
- `build.py`: `build(design) -> BuiltDesign`.
  - CadQuery solids → STL → CoACD convex parts → MJCF.
  - Checks:
    - validity
    - minimum wall thickness (sample 2,000 surface points, ray-cast inward)
    - overhang under 50° unless the template marks support
    - mass ≤ limit
    - wrist torque at full reach ≤ servo limit × 0.8
    - self-collision over 500 random configurations
  - Print estimate: `print_minutes = volume_cm3 × 1.6 + 6`, and `grams = volume_cm3 × 1.24 × infill_factor`. Refit these constants from any real data later.

### 8.9 `spec/`

- `posture.py`:
  - `derive(posture, area=None) -> EffectiveParams`. It turns β (clamped to [−2, 2]) into concrete settings by linear interpolation between the β = −1 and β = +1 columns of the table in Section 9, clamped beyond them. Integers are rounded. **Precedence:** a field set explicitly in the YAML overrides the β-derived value. Unset fields are `None` in `Posture` and come from `derive`.
  - `score()` (Section 8.4) and `accept(trial, base, eff) -> Decision`.
  - **Where the variation comes from:** MuJoCo is deterministic, so each repeat and re-run draws domain randomization from a seeded Generator: friction ×U(0.8, 1.2), stiffness ×U(0.9, 1.1), target pose noise N(0, 1 mm).
  - **Bounds:** run the trial panels 3 times with different randomization seeds; compute the per-control reward delta (trial minus base); bootstrap the mean over the pooled deltas (1,000 resamples, seeded) for a 90% interval. `lcb` needs `delta_lo > 0`; `mean` needs `delta_mean > 0`; `ucb` needs `delta_hi > 0` and charges one extra trial run.
  - `anneal(posture, ledger)`: `β = β0 − (β0 − β_min)·(1 − r)`, with β_min = −1 and r the minimum remaining budget fraction. Only applied if `anneal_with_budget` is set.
  - Per-area overrides: areas are `force`, `shape`, `kinematics`, `library`.
- `ledger.py`:
  - `estimate(proposal) -> CostVector`, `would_break_reserve(cost) -> bool`, `charge(cost)`, `remaining() -> dict`.
  - Logs estimate against actual per proposal, and refits linear print-time constants every 10 proposals.
  - Spend can never be negative, and a charge that would cross the reserve raises unless it's the final evaluation.
  - **Servo heat:** integrate `τ²·R` per joint with R = 1 Ω-equivalent. Cool exponentially with a 60 s time constant. Pause an episode (sim time advances, no motion) if any joint's heat exceeds 60 J until it falls below 30 J. This makes energy-hungry designs costly.
  - Wall-clock seconds are logged to `timing.json`, never to `report.json` or `cost.json`.

### 8.10 `logging` and CLI

```
runs/<run_id>/
  spec.yaml  design_000.json ... library_000.json ...
  iter_000/ results.jsonl  traces/*.npz  cost.json  proposal.json  decision.json  llm_calls.jsonl
  memory/ context.json full.json fix.json program.json  sidecar.json
  final/ report.json  plots/*.png
```

**CLI** (typer, `sea`):
- `sea panels --split train --n 200 --seed 1`
- `sea run configs/exp/e1_s2.yaml`
- `sea eval runs/<id> --split test`
- `sea replay runs/<id> --iter 5 --panel p012 --control c3` (opens the MuJoCo viewer)
- `sea reward runs/<id>` (standalone recompute; must match)
- `sea report runs/<id>` (plots and a markdown summary)

---

## 9. Hard limits and posture (conservative or optimistic)

**Hard limits** (`HardLimits`) are checked in `executor.py` (force, speed, workspace), `typecheck.py` (parameter ranges), and `build.py` (mass, wall, link scale). They're never read from LLM output.

**Posture** sets how boldly the system acts inside the limits. One knob ties it together: `score = gain + β·uncertainty − cost`.

| Decision | Conservative (β = −1) | Optimistic (β = +1) |
|---|---|---|
| Force cap per control | 60% of the class cap | Up to the hard limit |
| New primitives or tools | Only tried templates; rung ≤ 2 unless forced | All rungs |
| Edit size | One parameter, `max_param_delta` 0.3 | Up to 3 parameters, delta 1.0 |
| Fail verdict | After 3 consistent repeats | After 1 |
| Accept rule | `lcb` (β ≤ −0.34) | `ucb` (β ≥ 0.34); `mean` in between |
| Target choice | Lowest uncertainty first | Highest uncertainty first |
| Regression | Full reference panel | A sample of 4 controls |
| Escalation τ | 3σ | 5σ |
| Promotion | `k_promote` = 4 | `k_promote` = 2 |

**Annealing:** start at β0 and move toward conservative as budgets fall (Section 8.9).

**Per-area posture:** for example, conservative on `force`, optimistic on `shape`.

**Example spec** (`configs/exp/e1_s2.yaml`):

```yaml
seed: 1
limits: {max_force_n: 15, max_tip_speed_mps: 0.10,
         workspace_box: [{x: 0.10, y: -0.20, z: -0.10}, {x: 0.40, y: 0.20, z: 0.25}],
         banned_contacts: [pinch], max_tool_mass_g: 40, max_link_scale: 1.3, min_wall_mm: 1.2}
posture: {beta: 1.0, anneal_with_budget: true, per_area: {force: -1.0}}   # accept_rule etc. derived from beta
budgets: {print_minutes: 240, grams_pla: 300, grams_tpu: 50, design_iterations: 30,
          llm_usd: 20, rack_slots: 4, library_size: 40, sim_hours: 40, reserve_fraction: 0.2}
levels: [L0]
proposer: s2
```

---

## 10. Resource management

| Resource | Unit | Source of truth |
|---|---|---|
| Print time | minutes | `build.py` estimate; refit from actuals |
| Material | grams PLA, TPU | `build.py` |
| Design iterations | count | Loop counter |
| LLM spend | USD, tokens | `llm_calls.jsonl` |
| Approvals | count | Simulated approver: each body change needs one; at most `approvals_per_hour` per sim-hour |
| Sim compute | sim-seconds | Executor. Default budget 40 sim-hours per run: 30 iterations × 40 sampled panels × ~6 controls × ~10 s average, plus trials and regressions |
| Servo heat | J (proxy) | Ledger thermal model |
| Rack slots | count | Tools mounted at once |
| Library size | count | `compose/library.py` |

**Rules, all enforced in `ledger.py` and the loop:**
- **Cost before action:** each proposal carries an estimated `CostVector` and is refused if it breaks a budget or eats into the reserve.
- **Reserve:** 20% of each budget is held for the final evaluation and regressions.
- **Value per cost:** `value(p) = prediction.confidence × Σ gain_est over flips_to_pass` (gain_est from Section 8.4). `vpc(p) = value(p) / max(Σ_r w_r·cost_r / budget_r, 0.002)`. The floor keeps near-free program proposals from dividing by zero.
- **Reuse first:** before any `new_tool` proposal, `similar_fixes` and the tool library are searched. A known tool within parameter tolerance is reused at zero print cost.
- **Stop rule:** stop when the best value per cost is below `stop.min_value_per_cost` (0.01) for `stop.patience` (3) consecutive iterations.
- **Honest estimates:** log estimated against actual cost; refit every 10 proposals.
- **Limited rack:** with `rack_slots` = K, adding a tool beyond K requires choosing one to remove. System 2 must name it. This pushes toward a small set of general tools.

**Every report includes:** coverage reward per print hour, per gram, per LLM dollar, per approval, and per sim-hour.

---

## 11. Experiments

All experiments run 5 seeds and report means with 95% bootstrap confidence intervals. Each also reports LLM USD, sim-hours, and print-minute estimates.

| ID | Setup | Primary metric |
|---|---|---|
| E0 | Stock arm, seed library, no evolution; all splits | Coverage by control class; failure verdict mix |
| E1 | L0; proposers `random`, `cma`, `bo`, `s1_only`, `s2_nomem`, `s2`; 30 iterations | Coverage reward vs iterations and vs print minutes (H1) |
| E2 | `s2` with target selection gap-only vs gap + uncertainty | Coverage vs iterations (H2) |
| E3 | Freeze final designs and libraries from E1; run test split | Train − test coverage drop (H3) |
| E4 | Gaps presented in sequence, with vs without the fix tree | Iterations per fix over time (H4) |
| E5 | Add L1, then L2, then L3 | Coverage at fixed budget; search cost |
| E6 | Controller ablation: scripted System 1 vs System 1 with PPO for one contact primitive (optional) | Whether body changes still help |
| E7 | Reward hacking: remove cost terms; remove the reference regression | Rate of degenerate designs (probe length at max, mass at max) |
| E8 | Posture sweep β ∈ {−2, −1, 0, 1, 2}, fixed vs annealed | Final coverage at equal budget; reversed accepts; regressions (H5) |
| E9 | Budget sweep: print minutes {60, 240, 960} × rack slots {2, 4, 8} | Coverage at budget; library size; reuse rate |
| E10 | Compose split: `s2` vs `s1_only`, zero-shot (no evolution on compose panels) | Controls solved; share solved with no body change (H6) |
| E11 | Library learning on vs off | S2 calls per solved control, second half vs first half (H7) |

**Example success:**
- E1: `s2` reaches ≥ 0.90 coverage reward within 30 iterations, while the best non-LLM baseline is under 0.80.
- E10: `s2` solves ≥ 2× the `s1_only` count.
- Accepted changes that regress the reference panel stay under 5%.

---

## 12. Tests and coverage gates

### 12.1 Code coverage

CI command:

```
MUJOCO_GL=egl uv run pytest -m "not slow and not llm" --cov=sea --cov-branch --cov-report=term-missing --cov-report=json --cov-fail-under=85
uv run python scripts/check_package_coverage.py   # per-package minimums below
```

| Package | Minimum branch coverage |
|---|---|
| `sea.verify`, `sea.spec`, `sea.compose` | 95% |
| `sea.contracts`, `sea.memory`, `sea.propose` | 90% |
| `sea.control`, `sea.design`, `sea.sim` | 80% |
| Overall | 85% |

`check_package_coverage.py` reads `coverage.json` and fails if any package is below its minimum.

**Excluded from coverage** (via `[tool.coverage.run] omit` or `# pragma: no cover`): `propose/world_model.py` until M10, and the network call inside `AnthropicLLM` (its parsing is tested from fixtures).

**Test rules:** hypothesis tests use `derandomize=True`. Tests use a `FakeEmbedder` (deterministic hash → vector) unless marked `slow`; real embedding models and offscreen rendering run only in the nightly job. CoACD runs only in `slow` tests; fast tests use a stub that returns the convex hull.

**Test markers:**
- `slow`: more than 10 s; runs nightly.
- `llm`: live API; manual only.
- Default CI must finish in under 10 minutes.

### 12.2 Required tests (by file)

| Test file | Must check |
|---|---|
| `test_contracts.py` | Every model round-trips JSON. `extra="forbid"` rejects unknown fields. Schema generation works for `list[Proposal]`. |
| `test_checks.py` | The `CheckExpr` parser handles all primitives and `& \| ~` with precedence. Invalid strings raise. No `eval` (static grep test). |
| `test_dsl.py` | Pretty-print and parse round-trip on 200 random programs (hypothesis). Hashing is stable. |
| `test_typecheck.py` | A hook primitive without a hook tool is rejected. An out-of-reach target is rejected. A parameter range exceeding a hard limit is rejected. A valid seed program passes. |
| `test_library.py` | Promotion requires `k_promote` uses across 2 panels. Duplicate detection blocks near-identical programs. Pruning respects the size budget and dominance. |
| `test_controls.py` | Button detent: force–travel curve has a dip at each detent within 10%. Latch holds below threshold and releases above. Push-latch releases at 2 mm. |
| `test_model_consistency.py` | For the stock design and 5 random L2 designs, the MJCF site `gripperframe` and the URDF `gripper_frame_link` agree within 1 mm on 100 random configurations, and URDF z matches MJCF site x within 1°. |
| `test_ik.py` | For each point in `tests/fixtures/ik_points.yaml` (Appendix C: x, z from the `base_link` origin, direction, expected), the solve result matches `expected`; reachable points have residual under 2 mm. |
| `test_motor.py` | Guarded contact stops within 0.5 N of the cap. Commanded tip speed never exceeds the limit; measured speed stays under 1.2× the limit. Approach offset is 30 mm. |
| `test_verdicts.py` | One synthetic trace per verdict that `judge()` can produce gives exactly that verdict. The precedence order in Section 8.2 holds when several conditions are true. The executor emits `no_applicable_primitive` when no program type-checks. |
| `test_coverage.py` | Hand-computed examples: 0 controls, all pass, all inconclusive (pass_rate = 0), mixed. `coverage_reward` is in [0, 1] and equals passes / N. |
| `test_reward_audit.py` | After a 2-iteration run, `sea reward` output equals the in-loop reward exactly (float equality on the JSON values). |
| `test_memory.py` | Two well-separated synthetic context clusters (seeded): 60 buttons with force ~ N(2.9, 0.4) N and 40 levers ~ N(6, 0.8) N. Expectation uses an ancestor with at least 20 payloads. A button at exactly the cluster mean has z < 1; a bound button (8 N) has z > 5. Save and load (with sidecar) give identical expectations and payloads. Unconfirmed failures aren't learned. Same seed gives identical trees. |
| `test_targets.py` | Scarcity weights rise as a budget falls. With β < 0, the less uncertain target ranks first; with β > 0, the more uncertain one does. |
| `test_posture.py` | `derive` matches the table at β = ±1 and interpolates between; explicit fields override. `accept` is monotone: if `ucb` rejects, `mean` and `lcb` reject. Annealing moves β toward −1 as budgets fall, from any β0. Per-area overrides apply. |
| `test_ledger.py` | (hypothesis) Spend is never negative. The reserve is never crossed outside the final evaluation. The estimator refit lowers error on synthetic data. Thermal model: heat rises under load and decays with no load. |
| `test_build.py` | Every tool template at min, mid, and max parameters builds a valid solid. A thin-wall case is rejected. An over-mass case is rejected. Print estimate is monotone in volume. |
| `test_llm_mock.py` | `MockLLM` returns fixtures and raises on a missing fixture. `AnthropicLLM` parses a recorded response fixture. A schema error triggers one retry. |
| `test_escalation.py` | Each of the 4 escalation conditions triggers. A rung-k proposal missing a `rung<i>:` entry in `ruled_out` for some i < k is rejected; one with all entries passes. |
| `test_loop_integration.py` (slow) | 3 iterations, 3 panels, `MockLLM`: produces all log files; reward is recomputable; deterministic across 2 runs with the same seed. |
| `test_golden.py` (slow) | Stock arm on the reference panel matches stored verdicts in `tests/golden/reference_verdicts.json`. |
| `test_compose_split.py` | Every compose-split panel has at least one control whose (class × handle × recess × tilt) combination is absent from train. Every component appears in train. |

### 12.3 Hardware-style coverage tests (the verifier itself)

The verifier is the product, so it gets its own acceptance suite (`tests/verifier/`).

**Planted-fault panels.** 10 hand-built panels, each with a known fault:
- stiff button (stiffness ×5)
- no detent
- latch threshold above the force cap
- recessed button deeper than the stock tip
- door with a tab handle under the no-pinch rule
- knob with torque above the wrist limit
- slider with reversed direction
- and 3 more of the agent's choice

Each fault has a declared expected verdict.

**Acceptance:**
- The verifier gives the declared verdict for 10 of 10 faults.
- On the matching fault-free panels, it gives `pass` for at least 95% of controls over 5 seeds.
- `control_coverage` and `pass_rate` match hand counts.

---

## 13. Milestones and acceptance

| # | Deliverable | Acceptance |
|---|---|---|
| M1 | Repo skeleton, contracts, CI with coverage gates; stock SO-101 loads; IK; one button panel | `test_contracts`, `test_ik`, `test_model_consistency` (stock only) pass; a scripted press flips the button |
| M2 | All control classes with dynamics; panel generator with all splits; verdicts; coverage; standalone reward | `test_controls`, `test_verdicts`, `test_coverage`, `test_compose_split` pass; `reward.main()` matches hand counts on synthetic `results.jsonl`; the verifier suite (12.3) passes on synthetic traces |
| M3 | Motor primitives, executor, DSL, typechecker, seed library; System 1 local search | E0 runs end to end; `test_dsl`, `test_typecheck`, `test_motor` pass; the verifier suite passes on real sim runs |
| M4 | Tool templates, build pipeline, L0 design changes; ledger; posture | On the planted tab-handle door with no-pinch: the stock gripper gives `rule_violation` (its jaws count as `pinch`), and a rack hook gives `pass`. `test_build`, `test_ledger`, `test_posture` pass |
| M5 | Cobweb memory (all 4 trees), embeddings, targets | `test_memory`, `test_targets` pass; seed trees built from E0 logs |
| M6 | System 2 with `MockLLM` and `AnthropicLLM`; escalation; ladder; baselines | `test_escalation`, `test_llm_mock`, `test_loop_integration`, `test_reward_audit` pass; E1 runs with all proposers |
| M7 | Library learning: promotion and pruning; program tree | `test_library` passes; E11 runs |
| M8 | Experiments E1–E4, E8–E10; plots; report | Plots for H1–H6 generated by `sea report` |
| M9 | L1–L3 (fingers, arm, actuators); E5, E7 | `test_model_consistency` passes for L2 designs; E5 and E7 done |
| M10 (optional) | World model (3.6); E6 | World model meets its acceptance or is disabled with a written result |

---

## 14. Repository layout

```
self-evolving-arm/
  pyproject.toml  uv.lock  README.md  LICENSE
  .github/workflows/ci.yml
  assets/so101/                 # vendored URDF, MJCF, meshes + license
  configs/
    panels.yaml  primitives.yaml  limits.yaml  ik_seeds.yaml
    exp/ e0.yaml e1_*.yaml ... e11.yaml
  docs/ decisions.md  theory.md (copy of Section 3)
  scripts/ check_package_coverage.py  measure_reach.py
  src/sea/
    contracts.py
    sim/        build_model.py  controls.py  panels.py  env.py
    control/    frames.py  ik.py  reach.py  motor.py  executor.py  local_search.py
    compose/    dsl.py  checks.py  typecheck.py  library.py
    design/     space.py  tools.py  arm.py  build.py
    verify/     verdicts.py  coverage.py  reward.py
    memory/     concepts.py  embed.py
    propose/    llm.py  prompt.py  targets.py  baselines.py  world_model.py
    spec/       posture.py  ledger.py
    loop.py  cli.py
  tests/
    fixtures/llm/  golden/  verifier/
    test_*.py
  runs/                         # git-ignored
```

---

## 15. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| MuJoCo contact doesn't match real clicks and slips | Designs that only work in sim | Clicks come from explicit detent models; randomize friction and stiffness per seed; later, print the top 3 tools and test them on the real SO-101 (the link to MakeCU) |
| The LLM games the reward or jumps to body changes | Huge probes; wasted print budget | Hard limits outside LLM reach; cost terms; reference regression; ladder enforcement; E7 |
| L2 changes break IK or self-collide | Wasted iterations | Build checks; the FK consistency test; reach grid per design |
| LLM cost | Slow, expensive experiments | Cache by prompt hash; cheap model for re-parameterization; USD budget in the ledger |
| Cobweb API differs from Section 8.5 | Wrapper breaks | Wrapper isolates it; note fixes in `docs/decisions.md` |
| Too broad | Nothing finished | M1–M8 with L0 alone answer H1–H7; L1–L3 and the world model are extensions |

---

## 16. Related work (for the writeup)

**Body and tool co-design**
- Sims (1994), *Evolving Virtual Creatures*.
- Gupta et al. (2021), *Embodied intelligence via learning and evolution* (DERL).
- Bhatia et al. (NeurIPS 2021), *Evolution Gym*.
- Liu, Tian, Guo, Liu, Wu (CoRL 2023), *Learning to Design and Use Tools for Robotic Manipulation*. The closest prior work for L0.
- Ringwald et al. (2022), *Towards Task-Specific Modular Gripper Fingers*.

**LLM-driven robot design**
- Ringel et al. (2024), *Text2Robot* (arXiv 2406.19963).
- Qiu et al. (2024), *RoboMorph* (arXiv 2407.08626).
- *Lang2Morph* (2025, arXiv 2509.18937).

**Composition and library learning**
- Ellis et al. (2021), *DreamCoder*.
- Grand et al. (ICLR 2024), *LILO: Learning Interpretable Libraries by Compressing and Documenting Code* (arXiv 2310.19791).

**World models**
- Meta (2025), *V-JEPA 2*, including the action-conditioned V-JEPA 2-AC for robot planning.

**Concept memory**
- Fisher (1987), *Knowledge acquisition via incremental conceptual clustering* (Cobweb).
- Gupta, Singaravadivelan, Wang (2025), *Hierarchical Semantic Retrieval with Cobweb* (arXiv 2510.02539).
- Singaravadivelan, Gupta, Wang, MacLellan (2026), *CobwebTM* (arXiv 2604.14489).

**Dual-process framing**
- Kahneman (2011), System 1 and System 2.

**How this repo differs:**
- Changes are driven by measured failures on a coverage suite, not by a single task reward.
- System 2 composes primitives before touching the body, under an explicit cost ladder.
- Successful compositions become new primitives.
- A Cobweb memory built on neural embeddings decides where to explore and remembers what worked.
- One verifiable reward scores both the product (MakeCU) and the robot (this repo).

---

## 17. Open questions (defaults in brackets)

- Image embedding model for control crops [DINOv2-small].
- Should β be one number or one per area? [Per area: force conservative, others annealed.]
- Library size budget [40].
- Is MJX worth it for E9's sweeps? [No; CPU multiprocessing first.]
- When to move to real prints [after E3, top 3 tools only].

---

## Appendix A. Program and check grammar (normative)

```ebnf
program   := node
node      := prim | seq | guard | repeat | branch | with | bind
prim      := NAME "(" [arg ("," arg)*] ")"
arg       := NAME "=" (NUMBER | "$target" | STRING) | NUMBER | "$target"
seq       := "Seq(" node ("," node)+ ")"
guard     := "Guard(" node "," "until=" check "," "cap=" check ")"
repeat    := "Repeat(" node "," ("n=" INT | "until=" check) ")"
branch    := "Branch(" check "," node "," node ")"      # (if-true, if-false)
with      := "With(" INT "," node ")"                    # rack slot index
bind      := "Bind(" NAME "," "{" NAME ":" NUMBER ("," NAME ":" NUMBER)* "}" ")"
check     := term ("|" term)*
term      := factor ("&" factor)*
factor    := "~" factor | "(" check ")" | atom
atom      := "moved_at_least(" ref "," value ")" | "latched(" ref ")"
           | "detent_crossed(" ref "," INT ")" | "force_below(" NUMBER ")"
           | "no_banned_contact()"
ref       := "$target" | CONTROL_ID
value     := NUMBER | "travel"          # "travel" resolves from the referenced ControlSpec
NUMBER    := SI float, no units
```

| Op | Children | Required fields |
|---|---|---|
| `prim` | 0 | `prim` |
| `seq` | ≥ 2 | — |
| `guard` | 1 | `check` (until), `cap` |
| `repeat` | 1 | `n` or `check` |
| `branch` | 2 (true, false) | `check` |
| `with` | 1 | `tool_index` |
| `bind` | 0 | `prim` (name), `bind_args` |

The JSON form (`Node`) and the text form must round-trip.

## Appendix B. Tools, tip transforms, and contact types

- **Mount:** every tool attaches at a 20 × 20 mm dovetail on the fixed jaw. The tool's tip transform is given relative to `gripper_frame_link`. Translation is along that frame's z (approach) axis unless stated.

| Template | Contact type | Tip transform |
|---|---|---|
| stock (no tool) | `pinch` if both jaws touch the same control body within 50 ms, otherwise `edge` | identity |
| pad | `pad` | +z by `thickness` + 12 mm |
| hook | `hook` | +z by `length`, then −x by `depth` |
| knob_cup | `cup` | +z by `depth` + 12 mm |
| soft_tip | `pad` | +z by `radius` + 15 mm |
| slot_inserter | `edge` | +z by `length` |
| probe | `probe` | +z by `length` |

- **Geom naming:** `build_model` names every tool collision geom `tool_<contact>_<i>`. Stock jaws become `jaw_fixed_<i>` and `jaw_moving_<i>`. The verifier maps contacts by these prefixes.
- **Contacts each class accepts:** button and recessed button: `pad`, `probe`, `edge`. Toggle: `pad`, `edge`. Knob: `cup`, `pinch`. Lever: `pad`, `hook`. Door and drawer, by handle: `tab` accepts `pinch` or `hook` (a hook catches the tab's back edge), `pull` needs `hook`, `push_latch` needs `pad` or `probe`. Slider: `pad`, `edge`. Slot: `edge` (slot_inserter only).

## Appendix C. IK fixture points (stock arm, y = 0, from the `base_link` origin)

Measured Oct 31 with placo 0.10.2. Store these in `tests/fixtures/ik_points.yaml`.

| x (m) | z (m) | Approach | Expected |
|---|---|---|---|
| 0.14 | 0.05 | down | reachable |
| 0.22 | 0.00 | down | reachable |
| 0.30 | −0.05 | down | reachable |
| 0.34 | 0.00 | down | unreachable |
| 0.22 | 0.10 | down | unreachable |
| 0.26 | 0.15 | horizontal | reachable |
| 0.30 | 0.10 | horizontal | reachable |
| 0.34 | 0.00 | horizontal | reachable |
| 0.22 | 0.15 | horizontal | unreachable |
| 0.26 | 0.05 | horizontal | unreachable |
| 0.22 | 0.00 | 45° | reachable |
| 0.30 | 0.15 | 45° | reachable |
| 0.14 | 0.10 | 45° | unreachable |
| 0.18 | 0.05 | 45° | unreachable |

Lateral check: (0.22, ±0.06, 0.00), down, reachable. If placo or model updates change a result, regenerate the table with `scripts/measure_reach.py` and record it in `docs/decisions.md`.

## Appendix D. Defaults

**Panel ranges** (`configs/panels.yaml`). These are train ranges; test widens each upper bound by 25%.

| Class | size_mm meaning | Size range | Stiffness | Travel | Detents | Other |
|---|---|---|---|---|---|---|
| push_button | (diameter,) | 8–20 | 300–1500 N/m | 1–4 mm | 0–1, peak 0.5–2 N | — |
| recessed_button | (diameter,) | 6–12 | 300–1500 N/m | 1–3 mm | 0–1 | recess 5–25 mm |
| toggle | (w, h) | 6–15, 10–25 | 0.01–0.05 N·m/rad | 30–50° | 1, peak 0.02–0.06 N·m | — |
| knob | (diameter,) | 10–40 | 0.005–0.05 N·m/rad | 45–270° | 0–12 | — |
| lever | (length,) | 30–80 | 0.05–0.3 N·m/rad | 20–60° | 0 | — |
| door | (w, h) | 60–120, 60–120 | 0.02–0.1 N·m/rad | 60–90° | 0 | latch 2–12 N; handle tab, pull, or push_latch |
| drawer | (w, h) | 60–120, 30–60 | 20–100 N/m | 30–80 mm | 0 | latch 2–10 N |
| slider | (length,) | 20–60 | 50–300 N/m | 10–40 mm | 0–4 | — |
| slot | (w, gap) | 20–40, 1–3 | — | insert 10–25 mm | 0 | needs an edge contact |

**Other defaults:**
- Panel tilt: 0–60°.
- Force caps by class: 10 N for buttons, toggles, sliders, slots; 15 N for levers, doors, drawers.
- `n_min` = 20 (expectation ancestor); `n_fam` = 5 (escalation); `m` = 8 (System 1 trials); `max_targets` = 3.
- `d_dup` = 0.15; library size 40; retire after 10 unused iterations.
- Thermal: pause at 60 J, resume at 30 J, cooling time constant 60 s.
- LLM: temperature 0.2; at most 3 proposals per call; retry once on schema error.
- Embeddings: text `all-MiniLM-L6-v2`; image DINOv2-small CLS; crops rendered offscreen at 224 × 224 from a fixed panel camera; `FakeEmbedder` in fast tests.