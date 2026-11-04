"""A compiled graph's actuators, latches and sensors at run time (D-087).

The simulator computes every actuator's force itself (``mech.build``: the motor's line, the piston's pressure
and needle valve, the spring's torque).  What changes during a run is what the robot's program commands: a
motor's duty, a valve's direction, a servo's target, a latch released.  ``Machine`` maps those names to the
model, and reads the sensors back by name.
"""

from __future__ import annotations

import mujoco
import numpy as np

from sea.mech.graph import Band, Graph, TorsionSpring


class Machine:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, g: Graph):
        self.model, self.data = model, data
        self.ctrl = {a.name: model.actuator(a.name).id for a in g.actuators
                     if not isinstance(a, Band | TorsionSpring)}
        self.latch = {lt.name: model.equality(lt.name).id for lt in g.latches}
        self.sense = {}
        for sn in g.sensors:
            i = model.sensor(sn.name).id
            self.sense[sn.name] = (int(model.sensor_adr[i]), int(model.sensor_dim[i]))

    def command(self, name: str, value: float) -> None:
        """A motor's duty or a valve's direction (-1..1), or a servo's target (its joint's units)."""
        self.data.ctrl[self.ctrl[name]] = value

    def release(self, name: str) -> None:
        self.data.eq_active[self.latch[name]] = 0

    def held(self, name: str) -> bool:
        return bool(self.data.eq_active[self.latch[name]])

    def read(self, name: str) -> float:
        """A sensor's reading: a switch's contact force (N), a range (m, -1 past its reach), a joint's position."""
        adr, dim = self.sense[name]
        return float(np.asarray(self.data.sensordata[adr: adr + dim])[0])
