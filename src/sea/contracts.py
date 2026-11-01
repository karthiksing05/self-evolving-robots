"""Data contracts shared by every package (spec Section 6).

All models are Pydantic v2, ``extra="forbid"`` and frozen unless noted.  Units are SI
(meters, radians, newtons, seconds) plus grams and minutes, unless a field name says
otherwise.  ``CheckExpr`` values are strings in the grammar of Appendix A and are parsed by
``sea.compose.checks``; nothing here evaluates them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# ---- Appendix D defaults that are not owned by a single model ----
N_MIN = 20  # payload results required at the expectation ancestor
N_FAM = 5  # familiarity count below which a control escalates
S1_TRIALS = 8  # System 1 local-search trials (m)
MAX_TARGETS = 3
D_DUP = 0.15
RETIRE_AFTER = 10  # iterations unused before a library primitive is retired
LAMBDA_PRINT = 0.05
LAMBDA_MASS = 0.02
MEASUREMENTS: tuple[str, ...] = ("peak_force", "travel_reached", "detents", "time_s", "ik_residual_mm")

CheckExpr = str

ControlClass = Literal[
    "push_button", "toggle", "knob", "lever", "door", "drawer", "slider", "slot", "recessed_button"
]
CONTROL_CLASSES: tuple[str, ...] = (
    "push_button", "toggle", "knob", "lever", "door", "drawer", "slider", "slot", "recessed_button"
)
ContactType = Literal["pad", "hook", "cup", "probe", "edge", "pinch"]
Verdict = Literal[
    "pass",
    "stuck",
    "no_actuation",
    "wrong_behavior",
    "unreachable",
    "rule_violation",
    "no_applicable_primitive",
    "inconclusive",
]
Handle = Literal["none", "tab", "pull", "push_latch"]
ProposalKind = Literal["reparam", "program", "use_tool", "new_tool", "finger", "arm", "actuator", "promote"]

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class _Model(BaseModel):
    model_config = _FROZEN


# ---------------------------------------------------------------- geometry
class Vec3(_Model):
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class Pose(_Model):
    pos: Vec3
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------- panels
class Rules(_Model):
    banned_contacts: list[ContactType] = ["pinch"]
    force_cap_by_class: dict[ControlClass, float]
    time_limit_s: float = 20.0


class ControlSpec(_Model):
    id: str
    cls: ControlClass
    pose_panel: Pose
    size_mm: tuple[float, ...]  # class-specific: (diameter,) or (w, h) ...
    stiffness: float  # N/m or N*m/rad
    travel: float  # m or rad to success
    detents: list[float] = []  # positions of detent bumps (m or rad)
    detent_peak: float = 0.0  # N or N*m
    friction: float
    latch_threshold: float | None = None  # doors/drawers
    handle: Handle = "none"
    recess_depth_mm: float = 0.0
    success: CheckExpr


class PanelSpec(_Model):
    id: str
    seed: int
    split: Literal["train", "val", "test", "compose", "reference"]
    pose_world: Pose
    tilt_rad: float
    controls: list[ControlSpec]
    rules: Rules


# ---------------------------------------------------------------- design
class ToolSpec(_Model):
    template: Literal["stock", "pad", "hook", "knob_cup", "soft_tip", "slot_inserter", "probe"]
    params: dict[str, float]  # template-specific, ranges in design/space.py
    contact: ContactType


class FingerSpec(_Model):
    jaw_len_mm: float = 0.0  # 0 = stock jaw
    tip_radius_mm: float = 0.0
    softness: float = 0.0


class ArmSpec(_Model):
    upper_scale: float = 1.0
    fore_scale: float = 1.0
    wrist_offset_mm: float = 0.0
    riser_m: float = 0.0
    base_pitch_rad: float = 0.0
    servo_voltage: dict[str, Literal[7.4, 12.0]] = {j: 7.4 for j in ARM_JOINTS}


class BuildEstimate(_Model):
    print_minutes: float = 0.0
    grams_pla: float = 0.0
    grams_tpu: float = 0.0
    added_mass_g: float = 0.0


class Design(_Model):
    id: str
    parent: str | None = None
    arm: ArmSpec = ArmSpec()
    fingers: FingerSpec = FingerSpec()
    tool_rack: list[ToolSpec] = []  # <= rack_slots
    est: BuildEstimate = BuildEstimate()


# ---------------------------------------------------------------- programs
class PrimRef(_Model):
    name: str
    args: dict[str, float | str] = {}


class Node(_Model):
    op: Literal["prim", "seq", "guard", "repeat", "branch", "with", "bind"]
    prim: PrimRef | None = None
    children: list[Node] = []
    check: CheckExpr | None = None
    n: int | None = None
    tool_index: int | None = None  # index into Design.tool_rack for "with"
    cap: CheckExpr | None = None  # guard only: stop when this becomes false
    bind_args: dict[str, float] = {}  # bind only: fixed arguments


class Program(_Model):
    id: str
    target_cls: ControlClass
    root: Node
    source: Literal["seed", "s2", "promoted"]


class PrimitiveDef(_Model):
    name: str
    kind: Literal["motor", "check", "composite"]
    doc: str
    requires: dict[str, str | float] = {}
    provides: dict[str, str] = {}
    params: dict[str, tuple[float, float]] = {}  # name -> (lo, hi)
    body: Program | None = None  # for composites
    uses: int = 0
    passes: int = 0
    created_iter: int = 0


# ---------------------------------------------------------------- execution and verdicts
class StepTrace(BaseModel):
    """One control attempt.  Not frozen: the executor appends while recording."""

    model_config = ConfigDict(extra="forbid")

    panel_id: str
    control_id: str
    program_id: str
    attempt: int
    t: list[float] = []
    q: list[list[float]] = []
    tip_force: list[float] = []
    control_q: list[float] = []
    contacts: list[tuple[float, ContactType, str]] = []
    s1_trials: int = 0
    ik_residual_mm: float = 0.0
    joint_limit_hits: list[str] = []
    # Executor bookkeeping (Section 8.2 needs these to judge a trace).
    aborted: bool = False
    expected_samples: int = 0
    detents_crossed: int = 0
    latched_final: bool | None = None
    push_phase: list[bool] = []  # per sample: inside a push/guard phase
    in_contact: list[bool] = []  # per sample: any robot-control contact
    success_time_s: float | None = None  # first sample where the success check held
    tool_swaps: int = 0
    sim_seconds: float = 0.0
    servo_heat_j: float = 0.0
    note: str = ""
    force_cap: float = 0.0  # guard cap in force for this attempt
    peak_control_force: float = 0.0  # robot -> target control, max over 500 Hz steps
    contact_types: list[ContactType] = []
    q_rest: float = 0.0
    program_ok: bool = False
    failed_primitive: str = ""  # motor primitive that returned False or was running at an abort (C-001)


class ControlResult(_Model):
    panel_id: str
    control_id: str
    cls: ControlClass
    verdict: Verdict
    peak_force: float
    travel_reached: float
    detents: int
    time_s: float
    surprise_z: dict[str, float] = {}
    surprise_nll: float = 0.0
    concept_n: int = 0
    escalated: bool = False
    escalation_reason: str | None = None
    trace_path: str = ""
    # Added (docs/decisions.md D-004): the full tree's measurement vector needs it.
    ik_residual_mm: float = 0.0
    program_id: str = ""
    s1_trials: int = 0

    @property
    def key(self) -> str:
        return f"{self.panel_id}:{self.control_id}"


class CostVector(_Model):
    print_minutes: float = 0
    grams_pla: float = 0
    grams_tpu: float = 0
    llm_usd: float = 0
    approvals: int = 0
    sim_seconds: float = 0
    servo_heat_j: float = 0


class Report(_Model):
    run_id: str
    iteration: int
    design_id: str
    panels: list[str]
    results: list[ControlResult]
    control_coverage: float
    pass_rate: float
    coverage_reward: float
    cost: CostVector
    reward: float  # coverage_reward minus cost terms (Sec 8.3)


# ---------------------------------------------------------------- System 2
class Prediction(_Model):
    flips_to_pass: list[str] = []
    may_regress: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class DesignDelta(_Model):
    arm: dict[str, float] = {}  # partial ArmSpec fields
    fingers: dict[str, float] = {}  # partial FingerSpec fields
    servo_voltage: dict[str, Literal[7.4, 12.0]] = {}
    rack_add: list[ToolSpec] = []
    rack_remove: list[int] = []  # removal must be named when the rack is full


class Proposal(_Model):
    kind: ProposalKind
    rung: int
    rationale: str
    hypothesis: str
    ruled_out: list[str] = []
    program: Program | None = None
    design_delta: DesignDelta | None = None
    reparam: dict[str, tuple[float, float]] | None = None  # primitive param -> new range
    promote: PrimitiveDef | None = None
    prediction: Prediction


class ProposalBatch(_Model):
    proposals: list[Proposal] = Field(min_length=1, max_length=3)


class Decision(_Model):
    accepted: bool
    rule: Literal["lcb", "mean", "ucb"]
    delta_mean: float
    delta_lo: float
    delta_hi: float
    regressions: list[str] = []
    prediction_score: float = 0.0


class FixRecord(_Model):
    """A past (Proposal, Decision) pair retrieved from the fix tree."""

    id: str
    iteration: int
    proposal: Proposal
    decision: Decision
    distance: float = 0.0


# ---------------------------------------------------------------- spec, posture, resources
class HardLimits(_Model):
    max_force_n: float = 15.0
    max_tip_speed_mps: float = 0.10
    workspace_box: tuple[Vec3, Vec3]
    banned_contacts: list[ContactType] = ["pinch"]
    max_tool_mass_g: float = 40.0
    max_link_scale: float = 1.3
    min_wall_mm: float = 1.2


class Posture(_Model):
    beta: float = 0.0  # other fields: None = derive from beta (Sec 8.9)
    max_param_delta: float | None = None
    max_edits: int | None = None
    repeats_before_fail: int | None = None
    accept_rule: Literal["lcb", "mean", "ucb"] | None = None
    surprise_tau: float | None = None
    regression_panel: Literal["full", "sample"] | None = None
    force_cap_fraction: float | None = None
    k_promote: int | None = None
    anneal_with_budget: bool = True
    per_area: dict[str, float] = {}  # area -> beta override


class Budgets(_Model):
    print_minutes: float = 240
    grams_pla: float = 300
    grams_tpu: float = 50
    design_iterations: int = 30
    llm_usd: float = 20.0
    approvals_per_hour: int = 6
    rack_slots: int = 4
    library_size: int = 40
    sim_hours: float = 40.0
    reserve_fraction: float = 0.2


class StopRule(_Model):
    target_reward: float = 0.95
    min_value_per_cost: float = 0.01
    patience: int = 3


class Ablations(_Model):
    target_mode: Literal["gap", "gap_uncertainty"] = "gap_uncertainty"  # E2
    use_fix_tree: bool = True  # E4
    library_learning: bool = True  # E11
    cost_terms: bool = True  # E7
    reference_regression: bool = True  # E7
    zero_shot: bool = False  # E10: evaluate without evolving on the eval split


class RunSpec(_Model):
    """The YAML file for one run."""

    seed: int
    limits: HardLimits
    posture: Posture = Posture()
    budgets: Budgets = Budgets()
    levels: list[Literal["L0", "L1", "L2", "L3"]] = ["L0"]
    proposer: Literal["s2", "s2_nomem", "random", "cma", "bo", "s1_only"] = "s2"
    world_model: bool = False
    llm_model: str = "claude-opus-5-5"
    llm_cheap_model: str = "claude-sonnet-5-5"
    splits: dict[str, int] = {"train": 200, "val": 50, "test": 100, "compose": 60}
    panel_seed: int = 1  # panels = generate(split, n, panel_seed)
    iter_sample: int = 40  # train panels run per iteration (Sec 4)
    stop: StopRule = StopRule()
    ablations: Ablations = Ablations()
    # Harness knobs that are not part of the science (docs/decisions.md D-010).
    llm: Literal["claude_code", "anthropic", "mock", "scripted"] = "claude_code"  # claude_code: plan billing (D-029)
    llm_fixtures: str | None = None  # mock: fixtures to replay; scripted: directory to record fixtures into
    eval_split: Literal["train", "val", "test", "compose"] = "test"
    embedder: Literal["fake", "real"] = "real"
    s1_trials: int = S1_TRIALS  # m, System 1 local-search trials (Appendix D)
    trial_repeats: int = 3  # randomization repeats per acceptance trial (Sec 8.9)


# ---------------------------------------------------------------- binning helpers
def recess_bin(recess_mm: float) -> str:
    """Section 7.4 recess bins: {0, 0-10, >10 mm}."""
    if recess_mm <= 0.0:
        return "0"
    if recess_mm <= 10.0:
        return "0-10"
    return ">10"


def tilt_bin(tilt_rad: float) -> str:
    """Section 7.4 tilt bins: {0-20, 20-45, >45 deg}."""
    import math

    deg = math.degrees(tilt_rad)
    if deg <= 20.0:
        return "0-20"
    if deg <= 45.0:
        return "20-45"
    return ">45"


PROPOSAL_RUNG: dict[str, int | None] = {
    "reparam": 0,
    "program": 1,
    "use_tool": 2,
    "new_tool": 3,
    "finger": 4,
    "arm": 5,
    "actuator": 5,
    "promote": None,
}
