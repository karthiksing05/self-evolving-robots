"""Design parameter ranges per level, and a [0, 1]^d encoding for the CMA-ES / BO / random baselines.

Levels (spec 8.8 / 13): L0 tool rack, L1 fingers, L2 arm kinematics, L3 servo voltage.
Tool parameters are in mm (friction and softness unitless), as in the Section 8.8 table.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sea.contracts import ARM_JOINTS, ArmSpec, BuildEstimate, Design, FingerSpec, HardLimits, ToolSpec
from sea.design.fingers import RANGES as FINGER_FAMILY

TOOL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "stock": {},
    "pad": {"w": (8.0, 30.0), "h": (8.0, 30.0), "thickness": (3.0, 8.0), "friction": (0.4, 1.2)},
    "hook": {"depth": (4.0, 20.0), "throat": (3.0, 12.0), "length": (20.0, 60.0)},
    "knob_cup": {"inner_diameter": (8.0, 40.0), "depth": (4.0, 15.0), "ribs": (0.0, 8.0)},
    "soft_tip": {"radius": (2.0, 8.0), "softness": (0.0, 1.0)},
    "slot_inserter": {"blade_thickness": (0.6, 2.0), "width": (10.0, 40.0), "length": (20.0, 60.0)},
    "probe": {"diameter": (3.0, 8.0), "length": (20.0, 80.0)},
}

TEMPLATE_CONTACT: dict[str, str] = {
    "stock": "edge",
    "pad": "pad",
    "hook": "hook",
    "knob_cup": "cup",
    "soft_tip": "pad",
    "slot_inserter": "edge",
    "probe": "probe",
}

RACK_TEMPLATES: tuple[str, ...] = ("pad", "hook", "knob_cup", "soft_tip", "slot_inserter", "probe")
SLOT_CHOICES: tuple[str, ...] = ("none",) + RACK_TEMPLATES  # L0 selector values
SLOT_PARAMS = 4  # normalized parameter dims per rack slot

FINGER_RANGES: dict[str, tuple[float, float]] = dict(FINGER_FAMILY)  # design/fingers.py RANGES
VOLTAGES: tuple[float, float] = (7.4, 12.0)


def arm_ranges(limits: HardLimits) -> dict[str, tuple[float, float]]:
    return {
        "upper_scale": (0.8, limits.max_link_scale),
        "fore_scale": (0.8, limits.max_link_scale),
        "wrist_offset_mm": (-20.0, 40.0),
        "riser_m": (0.0, 0.10),
        "base_pitch_rad": (-0.3, 0.3),
    }


# ---------------------------------------------------------------- tool specs
def mid_params(template: str) -> dict[str, float]:
    return {k: (lo + hi) / 2 for k, (lo, hi) in TOOL_RANGES[template].items()}


def params_at(template: str, frac: float) -> dict[str, float]:
    """Parameters at a fraction of every range (0 = all minimum, 1 = all maximum)."""
    return {k: lo + frac * (hi - lo) for k, (lo, hi) in TOOL_RANGES[template].items()}


def tool_errors(spec: ToolSpec) -> list[str]:
    """Range and contact errors for a tool spec (empty list = valid)."""
    ranges = TOOL_RANGES.get(spec.template)
    if ranges is None:  # pragma: no cover - the contract's Literal already rejects this
        return [f"unknown template {spec.template!r}"]
    errs: list[str] = []
    if spec.contact != TEMPLATE_CONTACT[spec.template]:
        errs.append(f"{spec.template}: contact must be {TEMPLATE_CONTACT[spec.template]!r}, got {spec.contact!r}")
    for k in spec.params:
        if k not in ranges:
            errs.append(f"{spec.template}: unknown parameter {k!r}")
    for k, (lo, hi) in ranges.items():
        if k not in spec.params:
            errs.append(f"{spec.template}: missing parameter {k!r}")
        elif not lo - 1e-9 <= spec.params[k] <= hi + 1e-9:
            errs.append(f"{spec.template}: {k}={spec.params[k]:g} outside [{lo:g}, {hi:g}]")
    return errs


def make_tool_spec(template: str, **params: float) -> ToolSpec:
    """A validated ``ToolSpec``; missing parameters take the middle of their range."""
    if template not in TOOL_RANGES:
        raise ValueError(f"unknown template {template!r}")
    full = mid_params(template)
    full.update({k: float(v) for k, v in params.items()})
    spec = ToolSpec(template=template, params=full, contact=TEMPLATE_CONTACT[template])  # type: ignore[arg-type]
    errs = tool_errors(spec)
    if errs:
        raise ValueError("; ".join(errs))
    return spec


def tools_equivalent(a: ToolSpec, b: ToolSpec, tol: float = 0.1) -> bool:
    """Same template and every parameter within ``tol`` x its range width (reuse-first rule, Sec 10)."""
    if a.template != b.template:
        return False
    for k, (lo, hi) in TOOL_RANGES[a.template].items():
        if abs(a.params.get(k, lo) - b.params.get(k, lo)) > tol * (hi - lo) + 1e-12:
            return False
    return True


# ---------------------------------------------------------------- design space
def _norm(v: float, lo: float, hi: float) -> float:
    return 0.5 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _denorm(u: float, lo: float, hi: float) -> float:
    return float(lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo))


class DesignSpace:
    """Flat [0, 1]^dim encoding of the levels being searched.

    Layout, in level order: L0 = ``rack_slots`` x (selector, 4 params); L1 = finger fields;
    L2 = arm scalars; L3 = one voltage selector per joint (< 0.5 -> 7.4 V, else 12 V).
    The selector maps to ``SLOT_CHOICES`` by ``floor(u * len)``; "none" leaves the slot empty.
    """

    def __init__(self, levels: Sequence[str], rack_slots: int, limits: HardLimits):
        self.levels = [lv for lv in ("L0", "L1", "L2", "L3") if lv in levels]
        self.rack_slots = rack_slots
        self.limits = limits
        self._arm = arm_ranges(limits)

    def names(self) -> list[str]:
        out: list[str] = []
        if "L0" in self.levels:
            for s in range(self.rack_slots):
                out.append(f"rack{s}.template")
                out.extend(f"rack{s}.p{j}" for j in range(SLOT_PARAMS))
        if "L1" in self.levels:
            out.extend(f"fingers.{k}" for k in FINGER_RANGES)
        if "L2" in self.levels:
            out.extend(f"arm.{k}" for k in self._arm)
        if "L3" in self.levels:
            out.extend(f"servo.{j}" for j in ARM_JOINTS)
        return out

    @property
    def dim(self) -> int:
        return len(self.names())

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.dim)

    def encode(self, design: Design) -> np.ndarray:
        out: list[float] = []
        if "L0" in self.levels:
            rack = [t for t in design.tool_rack if t.template != "stock"]
            for s in range(self.rack_slots):
                if s < len(rack):
                    t = rack[s]
                    out.append((SLOT_CHOICES.index(t.template) + 0.5) / len(SLOT_CHOICES))
                    vals = [_norm(t.params.get(k, lo), lo, hi) for k, (lo, hi) in TOOL_RANGES[t.template].items()]
                else:
                    out.append(0.5 / len(SLOT_CHOICES))
                    vals = []
                out.extend(vals + [0.5] * (SLOT_PARAMS - len(vals)))
        if "L1" in self.levels:
            f = design.fingers.model_dump()
            out.extend(_norm(f[k], lo, hi) for k, (lo, hi) in FINGER_RANGES.items())
        if "L2" in self.levels:
            a = design.arm.model_dump()
            out.extend(_norm(a[k], lo, hi) for k, (lo, hi) in self._arm.items())
        if "L3" in self.levels:
            out.extend(0.75 if design.arm.servo_voltage.get(j, 7.4) == 12.0 else 0.25 for j in ARM_JOINTS)
        return np.array(out, dtype=float)

    def decode(self, vec: np.ndarray, base: Design, new_id: str) -> Design:
        u = np.clip(np.asarray(vec, dtype=float), 0.0, 1.0)
        if u.shape != (self.dim,):
            raise ValueError(f"expected a vector of length {self.dim}, got {u.shape}")
        i = 0
        rack = list(base.tool_rack)
        fingers, arm = base.fingers, base.arm
        if "L0" in self.levels:
            rack = []
            for _ in range(self.rack_slots):
                choice = SLOT_CHOICES[min(int(u[i] * len(SLOT_CHOICES)), len(SLOT_CHOICES) - 1)]
                params = u[i + 1 : i + 1 + SLOT_PARAMS]
                i += 1 + SLOT_PARAMS
                if choice == "none":
                    continue
                vals = {k: _denorm(params[j], lo, hi) for j, (k, (lo, hi)) in enumerate(TOOL_RANGES[choice].items())}
                rack.append(make_tool_spec(choice, **vals))
        if "L1" in self.levels:
            fvals = {k: _denorm(u[i + j], lo, hi) for j, (k, (lo, hi)) in enumerate(FINGER_RANGES.items())}
            i += len(FINGER_RANGES)
            fingers = FingerSpec(**fvals)
        arm_update: dict[str, object] = {}
        if "L2" in self.levels:
            arm_update.update({k: _denorm(u[i + j], lo, hi) for j, (k, (lo, hi)) in enumerate(self._arm.items())})
            i += len(self._arm)
        if "L3" in self.levels:
            arm_update["servo_voltage"] = {j: (12.0 if u[i + k] >= 0.5 else 7.4) for k, j in enumerate(ARM_JOINTS)}
            i += len(ARM_JOINTS)
        if arm_update:
            arm = ArmSpec(**{**arm.model_dump(), **arm_update})
        return Design(id=new_id, parent=base.id, arm=arm, fingers=fingers, tool_rack=rack, est=BuildEstimate())
