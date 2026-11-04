"""What operates a composed mechanism (D-092): the controls a graph exposes, and playing them on a ``Machine``.

Every actuator, latch and sensor in a graph has typed controls, so anything System 2 composes can be operated
without code that names its parts: a servo's ``at`` and ``target``, a motor's ``at``, ``run``, ``duty`` and a reverse ``back_at`` and ``back_run``, a
piston's ``at`` and ``hold`` (its valve extends for that long, then retracts), a latch's ``at``, a sensor's
``threshold``.  An actuator with a trigger counts its times from the moment that sensor fires.
"""

from __future__ import annotations

import math

from sea.mech.graph import Encoder, Gearmotor, Graph, LimitSwitch, Piston, RangeFinder, Servo
from sea.mech.runtime import Machine

SWITCH_N = (0.0, 20.0, 0.5)  # a switch's force threshold: lo, hi, start
RAMP_S = 0.4  # the motor driver ramps its duty: a step pops a wheelie


def _joint_range(g: Graph, name: str) -> tuple[float, float]:
    for b in g.bodies:
        for j in b.joints:
            if j.name == name and j.range is not None:
                return j.range
    return (-math.pi, math.pi)


def controls(g: Graph, horizon: float) -> dict[str, tuple[float, float, float]]:
    """Every control of ``g``: its name, then lowest, highest and a neutral start, over a run of ``horizon`` s."""
    out: dict[str, tuple[float, float, float]] = {}
    for a in g.actuators:
        if isinstance(a, Servo):
            lo, hi = _joint_range(g, a.joint)
            out.update({f"{a.name}.at": (0.0, horizon, 0.0), f"{a.name}.target": (lo, hi, (lo + hi) / 2)})
        elif isinstance(a, Gearmotor):
            out.update({f"{a.name}.at": (0.0, horizon, 0.0), f"{a.name}.run": (0.0, horizon, horizon / 2),
                        f"{a.name}.duty": (-1.0, 1.0, 0.5), f"{a.name}.back_at": (0.0, horizon, horizon),
                        f"{a.name}.back_run": (0.0, horizon, 0.0)})
        elif isinstance(a, Piston):
            out.update({f"{a.name}.at": (0.0, horizon, 0.0), f"{a.name}.hold": (0.0, horizon, horizon / 2)})
    out.update({f"{lt.name}.at": (0.0, horizon, horizon / 2) for lt in g.latches})
    for s in g.sensors:
        if isinstance(s, LimitSwitch):
            out[f"{s.name}.threshold"] = SWITCH_N
        elif isinstance(s, RangeFinder):
            out[f"{s.name}.threshold"] = (0.0, s.max_m, s.max_m / 2)
        elif isinstance(s, Encoder):
            lo, hi = _joint_range(g, s.joint)
            out[f"{s.name}.threshold"] = (lo, hi, (lo + hi) / 2)
    return out


class Program:
    """One setting of a graph's controls, played against the clock.  ``triggers``: actuator or latch -> the sensor
    that starts its clock.  A control left out takes its neutral start."""

    def __init__(self, g: Graph, ctl: dict[str, float], horizon: float, triggers: dict[str, str] | None = None):
        self.g, self.horizon = g, horizon
        self.triggers = dict(triggers or {})
        self.ctl = {k: ctl.get(k, v[2]) for k, v in controls(g, horizon).items()}
        self.fired: dict[str, float] = {}

    def _fires(self, s, v: float) -> bool:
        thr = self.ctl[f"{s.name}.threshold"]
        return 0.0 <= v <= thr if isinstance(s, RangeFinder) else v >= thr  # a range past its reach reads -1

    def _clock(self, name: str, t: float) -> float | None:
        """Seconds on ``name``'s clock, or None while its trigger has not fired."""
        sensor = self.triggers.get(name)
        if sensor is None:
            return t
        return t - self.fired[sensor] if sensor in self.fired else None

    def step(self, m: Machine, t: float) -> None:
        """Set every command for time ``t``; call it before each simulator step."""
        for s in self.g.sensors:
            if s.name not in self.fired and self._fires(s, m.read(s.name)):
                self.fired[s.name] = t
        c = self.ctl
        for a in self.g.actuators:
            if not isinstance(a, Servo | Gearmotor | Piston):
                continue
            now = self._clock(a.name, t)
            if now is None or now < c[f"{a.name}.at"]:
                continue
            into = now - c[f"{a.name}.at"]
            if isinstance(a, Servo):
                m.command(a.name, c[f"{a.name}.target"])
            elif isinstance(a, Gearmotor):
                duty, back = c[f"{a.name}.duty"], now - c[f"{a.name}.back_at"]
                if 0.0 <= back < c[f"{a.name}.back_run"]:
                    m.command(a.name, -duty * min(1.0, back / RAMP_S))
                else:
                    m.command(a.name, duty * min(1.0, into / RAMP_S) if into < c[f"{a.name}.run"] else 0.0)
            else:
                m.command(a.name, 1.0 if into < c[f"{a.name}.hold"] else -1.0)
        for lt in self.g.latches:
            now = self._clock(lt.name, t)
            if now is not None and now >= c[f"{lt.name}.at"] and m.held(lt.name):
                m.release(lt.name)
