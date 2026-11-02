"""What the gripper can hold (docs/gripper-experiment.md, finger family rule 3: capability is computed).

One instrument measures every design: the V1 probe (``sim/grasp_probe.py``), a fixed top-down pick at
50 g and friction 0.8.  For the stock jaws its numbers are stored as the V1 capability table
(tests/golden/stock_grasp.json, D-032); for a finger design (D-038) the same probe runs on demand,
cached by the finger parameters and the probe's source.  The typecheck's aperture rule reads the
result, so an analytic object wider than the gripper's widest stable grip is
``no_applicable_primitive`` (structural) before any simulation, and System 2 sees the reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

from sea.contracts import Design, FingerSpec

# Widest object held 30 mm up after a 1 s hold, at 50 g and friction 0.8 (V1, re-measured with grasp-scene
# impratio, D-048).  Spheres hold only with the jaw tips about 12 mm below the equator; System 1 has to find
# that grip.
STOCK_WIDEST_MM = {"cylinder": 55.0, "box": 60.0, "sphere": 50.0}
PROBE_WIDTHS_MM = tuple(range(20, 95, 5))
SRC = Path(__file__).resolve().parents[1]
CACHE = SRC.parents[1] / ".cache" / "grip"


def widest_mm(design: Design, shape: str) -> float:
    if design.fingers == FingerSpec():
        return STOCK_WIDEST_MM[shape]
    return measured(design.fingers.model_dump_json())[shape]


def basis(design: Design) -> str:
    return "V1 capability table" if design.fingers == FingerSpec() else "V1 probe, measured on these fingers"


def _key(fingers_json: str) -> str:
    h = hashlib.sha256(fingers_json.encode())
    for rel in ("design/fingers.py", "sim/grasp_probe.py", "sim/build_model.py", "control/motor.py"):
        h.update((SRC / rel).read_bytes())  # a change to the instrument re-measures
    return h.hexdigest()[:16]


@cache
def measured(fingers_json: str) -> dict[str, float]:
    """Widest width (mm) the V1 probe holds per shape, 0 if none; spheres at either tip depth."""
    path = CACHE / f"{_key(fingers_json)}.json"
    if path.exists():
        return json.loads(path.read_text())
    from sea.sim.grasp_probe import PROBE, pick, scene_object

    design = Design(id="probe", fingers=FingerSpec.model_validate_json(fingers_json))
    out: dict[str, float] = {}
    for shape in ("cylinder", "box", "sphere"):
        probes = [replace(PROBE, sphere_past_mm=p) for p in (5.0, 12.0)] if shape == "sphere" else [PROBE]
        held = [w for w in PROBE_WIDTHS_MM
                if any(pick(scene_object(shape, w, 50.0, 0.8, pr), design, pr).held for pr in probes)]  # type: ignore[arg-type]
        out[shape] = float(max(held, default=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out
