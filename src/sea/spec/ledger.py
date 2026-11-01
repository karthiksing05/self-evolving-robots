"""Resource ledger and servo thermal model (spec Sections 8.9 and 10).

The ledger prices proposals before action, refuses spending that would eat into the reserve
held for the final evaluation, logs estimates against actuals and refits the print-time
constants.  Wall-clock time never enters it.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from sea.contracts import Budgets, CostVector, Proposal

RESOURCES: tuple[str, ...] = ("print_minutes", "grams_pla", "grams_tpu", "llm_usd", "approvals", "sim_seconds")
BODY_KINDS = frozenset({"use_tool", "new_tool", "finger", "arm", "actuator"})
DENSITY_G_CM3 = {"pla": 1.24, "tpu": 1.21}
SCARCITY_FLOOR = 0.05
VPC_FLOOR = 0.002
SIM_HOUR_S = 3600.0


class ReserveError(RuntimeError):
    """A charge would cross the reserve held for the final evaluation."""


class Ledger:
    def __init__(self, budgets: Budgets, print_a: float = 1.6, print_b: float = 6.0, refit_every: int = 10):
        self.budgets = budgets
        self.print_a = print_a
        self.print_b = print_b
        self.refit_every = refit_every
        self.spent = CostVector()
        self.approval_times: list[float] = []  # sim clock (s) at each approved body change
        self.history: list[dict[str, Any]] = []  # estimate vs actual per proposal
        self.iterations = 0

    # ------------------------------------------------------------ budgets
    def budget(self, resource: str) -> float:
        b = self.budgets
        if resource == "sim_seconds":
            return b.sim_hours * SIM_HOUR_S
        if resource == "approvals":
            return float(b.approvals_per_hour) * b.sim_hours  # implied total at the rate limit
        return float(getattr(b, resource))

    def remaining(self) -> dict[str, float]:
        """Fraction of each budget left, in [0, 1] (plus design iterations)."""
        out: dict[str, float] = {}
        for r in RESOURCES:
            cap = self.budget(r)
            used = float(getattr(self.spent, r))
            out[r] = 1.0 if cap <= 0 else max(0.0, min(1.0, 1.0 - used / cap))
        n = self.budgets.design_iterations
        out["design_iterations"] = 1.0 if n <= 0 else max(0.0, 1.0 - self.iterations / n)
        return out

    def scarcity_weights(self) -> dict[str, float]:
        """``w_r = 1 / max(remaining_r, 0.05)``, normalized to sum to 1 (Sec 8.4)."""
        rem = self.remaining()
        raw = {r: 1.0 / max(rem[r], SCARCITY_FLOOR) for r in RESOURCES}
        total = sum(raw.values())
        return {r: v / total for r, v in raw.items()}

    def tick_iteration(self) -> None:
        self.iterations += 1

    # ------------------------------------------------------------ pricing
    def print_minutes_for(self, volume_cm3: float, n_parts: int = 1) -> float:
        if volume_cm3 <= 0 and n_parts <= 0:
            return 0.0
        return self.print_a * volume_cm3 + self.print_b * max(n_parts, 1)

    def estimate(self, proposal: Proposal, *, added_volume_cm3: float = 0.0,
                 material: Literal["pla", "tpu"] = "pla", llm_usd: float = 0.0, sim_seconds: float = 0.0,
                 n_parts: int = 0, grams: float | None = None) -> CostVector:
        """Cost before action.  Body changes add print time, material and one approval."""
        body = proposal.kind in BODY_KINDS
        printed = body and (added_volume_cm3 > 0 or n_parts > 0)
        minutes = self.print_minutes_for(added_volume_cm3, n_parts) if printed else 0.0
        mass = 0.0
        if printed:
            mass = grams if grams is not None else added_volume_cm3 * DENSITY_G_CM3[material]
        return CostVector(
            print_minutes=minutes,
            grams_pla=mass if material == "pla" else 0.0,
            grams_tpu=mass if material == "tpu" else 0.0,
            llm_usd=llm_usd,
            approvals=1 if body else 0,
            sim_seconds=sim_seconds,
        )

    def _cap(self, resource: str, final: bool) -> float:
        cap = self.budget(resource)
        return cap if final else cap * (1.0 - self.budgets.reserve_fraction)

    def _approvals_in_window(self, clock: float) -> int:
        return sum(1 for t in self.approval_times if t > clock - SIM_HOUR_S)

    def would_break_reserve(self, cost: CostVector) -> bool:
        """True if charging ``cost`` would dip into the reserve or exceed the approval rate."""
        for r in RESOURCES:
            if r == "approvals":
                continue
            amount = float(getattr(cost, r))
            if amount > 0 and float(getattr(self.spent, r)) + amount > self._cap(r, final=False) + 1e-12:
                return True
        if cost.approvals > 0:
            clock = self.spent.sim_seconds + cost.sim_seconds
            if self._approvals_in_window(clock) + cost.approvals > self.budgets.approvals_per_hour:
                return True
        return False

    def charge(self, cost: CostVector, *, final: bool = False) -> None:
        for r in (*RESOURCES, "servo_heat_j"):
            if float(getattr(cost, r)) < 0:
                raise ValueError(f"negative charge for {r}: {getattr(cost, r)}")
        if not final and self.would_break_reserve(cost):
            raise ReserveError(f"charge {cost.model_dump()} would cross the reserve")
        update = {name: getattr(self.spent, name) + getattr(cost, name) for name in CostVector.model_fields}
        self.spent = CostVector(**update)
        self.approval_times.extend([self.spent.sim_seconds] * cost.approvals)

    def value_per_cost(self, value: float, cost: CostVector) -> float:
        """``value / max(sum_r w_r * cost_r / budget_r, 0.002)`` (Sec 10)."""
        w = self.scarcity_weights()
        denom = 0.0
        for r in RESOURCES:
            cap = self.budget(r)
            if cap > 0:
                denom += w[r] * float(getattr(cost, r)) / cap
        return value / max(denom, VPC_FLOOR)

    # ------------------------------------------------------------ honest estimates
    def record(self, estimate: CostVector, actual: CostVector, volume_cm3: float) -> None:
        """Log estimate vs actual; refit the print-time constants every ``refit_every`` records."""
        self.history.append({
            "estimate": estimate.model_dump(),
            "actual": actual.model_dump(),
            "volume_cm3": float(volume_cm3),
            "abs_error_min": abs(actual.print_minutes - estimate.print_minutes),
        })
        if len(self.history) % self.refit_every == 0:
            self.refit()

    def refit(self) -> bool:
        """Least-squares ``minutes = a * volume + b`` over records with printed parts."""
        pts = [(h["volume_cm3"], h["actual"]["print_minutes"]) for h in self.history
               if h["volume_cm3"] > 0 and h["actual"]["print_minutes"] > 0]
        vols = np.array([p[0] for p in pts], dtype=float)
        if len(pts) < 2 or np.ptp(vols) <= 0:
            return False
        mins = np.array([p[1] for p in pts], dtype=float)
        a, b = np.linalg.lstsq(np.stack([vols, np.ones_like(vols)], axis=1), mins, rcond=None)[0]
        self.print_a, self.print_b = float(a), float(b)
        return True

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "budgets": self.budgets.model_dump(),
            "print_a": self.print_a,
            "print_b": self.print_b,
            "refit_every": self.refit_every,
            "spent": self.spent.model_dump(),
            "approval_times": list(self.approval_times),
            "history": list(self.history),
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ledger:
        led = cls(Budgets(**data["budgets"]), data["print_a"], data["print_b"], data["refit_every"])
        led.spent = CostVector(**data["spent"])
        led.approval_times = [float(t) for t in data["approval_times"]]
        led.history = list(data["history"])
        led.iterations = int(data["iterations"])
        return led


class ServoThermal:
    """Per-joint heat proxy: integrate ``tau^2 * R``, cool with time constant ``tau_s``.

    Pauses (returns True) once any joint exceeds ``pause_j`` and stays paused until every joint
    is back under ``resume_j``.
    """

    def __init__(self, n_joints: int = 6, tau_s: float = 60.0, pause_j: float = 60.0, resume_j: float = 30.0,
                 r_ohm: float = 1.0):
        self.tau_s = tau_s
        self.pause_j = pause_j
        self.resume_j = resume_j
        self.r_ohm = r_ohm
        self.heat = np.zeros(n_joints, dtype=float)
        self.total_j = 0.0
        self.paused = False

    def step(self, torques: np.ndarray, dt: float) -> bool:
        tau = np.asarray(torques, dtype=float)
        power = tau * tau * self.r_ohm
        self.heat = self.heat * np.exp(-dt / self.tau_s) + power * dt
        self.total_j += float(power.sum() * dt)
        if not self.paused and float(self.heat.max()) > self.pause_j:
            self.paused = True
        elif self.paused and float(self.heat.max()) < self.resume_j:
            self.paused = False
        return self.paused
