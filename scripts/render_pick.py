"""Filmstrip of one pick, for the before-and-after figures (D-039).

    uv run python scripts/render_pick.py '{"pad_angle_deg": 25, "groove_mm": 3, "pad_tpu_mm": 2}' golf_ball '{}' out.png

Runs ``pick_up`` on a real object centred on its table with the given fingers and System 1 parameters
(``{}`` = defaults), captures frames through the episode offscreen, and writes a horizontal strip plus
``out.json``: the judge's verdict, the reflex that stopped it, and each frame's time and object height.
Environment: FW/FH frame size, CD camera distance.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

from sea.compose.library import Library
from sea.contracts import Design, FingerSpec, HardLimits, SensorSuite
from sea.control import executor as X
from sea.design.build import build
from sea.sim.env import PanelRunner, SuiteConfig, _judge
from sea.sim.grasp_world import regression_panels

ROOT = Path(__file__).resolve().parents[1]
limits = HardLimits.model_validate(yaml.safe_load((ROOT / "configs/limits_grasp.yaml").read_text()))
fing = FingerSpec(**json.loads(sys.argv[1]))
name, params, out = sys.argv[2], json.loads(sys.argv[3]), Path(sys.argv[4])
n_frames = int(sys.argv[5]) if len(sys.argv) > 5 else 6
W, H = int(__import__("os").environ.get("FW", 360)), int(__import__("os").environ.get("FH", 300))
built = build(Design(id="x", fingers=fing, sensors=SensorSuite(jaw_pads=True)), limits)
cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=limits, seed=0, m=0,
                  confirm=False)
panel = regression_panels([name])[0]
obj = panel.controls[0]
runner = PanelRunner(panel, cfg, 0)
m = runner.sim.model
renderer = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption()
opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = float(__import__("os").environ.get("CD", 0.30)), 150, -14
captured = []
orig = X.Episode.tick


def tick(self, q):
    orig(self, q)
    if self.ticks % 5 == 0:
        body = self.sim.controls["obj"].body_id
        p = self.sim.data.xpos[body]
        cam.lookat[:] = [p[0], p[1], 0.05]
        renderer.update_scene(self.sim.data, cam, opt)
        captured.append((self.t, (self.sim.control_q("obj") - self.q_rest0["obj"]) * 1000, renderer.render().copy()))


X.Episode.tick = tick
prog = Library.seed().prims["pick_up"].body
tr = runner.attempt(obj, prog, params)
res = _judge(obj, panel, limits)(tr)
X.Episode.tick = orig
ts = np.array([c[0] for c in captured])
ups = np.array([c[1] for c in captured])
# key moments: start, above, contact (first rise), mid-lift, hold, end
picks = np.linspace(0, len(captured) - 1, n_frames).astype(int)
lift_i = int(np.argmax(ups > 3)) if (ups > 3).any() else None
if lift_i is not None and n_frames >= 6:
    picks = sorted({0, max(lift_i - 12, 1), lift_i, min(lift_i + 25, len(captured) - 1),
                    min(lift_i + 60, len(captured) - 1), len(captured) - 1})
frames, caps = [], []
for i in picks:
    t, up, img = captured[i]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 22, W, H], fill=(20, 20, 20))
    d.text((8, H - 18), f"t={t:.1f}s  object +{up:.0f} mm", fill=(240, 240, 240))
    frames.append(np.asarray(im))
    caps.append({"t": round(t, 2), "up_mm": round(float(up), 1)})
Image.fromarray(np.hstack(frames)).save(out)
json.dump({"verdict": res.verdict, "note": tr.note, "frames": caps, "max_up_mm": round(float(ups.max()), 1),
           "end_up_mm": round(float(ups[-1]), 1)}, open(out.with_suffix(".json"), "w"))
print(out, res.verdict, tr.note, f"max up {ups.max():.0f} mm, end {ups[-1]:.0f} mm")
