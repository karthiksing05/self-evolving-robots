"""Close-up of the jaws and finger pads (D-039).

    uv run python scripts/render_fingers.py '{"pad_angle_deg": 25, "groove_mm": 3, "pad_tpu_mm": 2}' out.png 0,0.44,0.9

Renders the stock jaws plus any pads (orange) from the side (top row) and a three-quarter view (bottom
row) at the given jaw openings, with the arm at home and no objects.
"""
import json
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sea.contracts import Design, FingerSpec
from sea.control.executor import HOME_Q
from sea.sim.build_model import build_model

fing = FingerSpec(**json.loads(sys.argv[1]))
out = sys.argv[2]
qs = [float(q) for q in (sys.argv[3] if len(sys.argv) > 3 else "0.0,0.45,0.9").split(",")]
sim = build_model(Design(id="c", fingers=fing), objects=[], table_z=-0.05)
m, d = sim.model, sim.data
W, H = 340, 340
r = mujoco.Renderer(m, H, W)
opt = mujoco.MjvOption()
opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
tiles = []
for q in qs:
    qa = HOME_Q.copy()
    qa[-1] = q
    sim.set_arm_q(qa)
    mujoco.mj_forward(m, d)
    gid = m.body("moving_jaw_so101_v1").id
    centre = (d.xpos[m.body("gripper").id] + d.xpos[gid]) / 2
    for az, el in ((90, 0), (135, -20)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = centre + [0, 0, -0.092]
        cam.distance, cam.azimuth, cam.elevation = 0.105, az, el
        r.update_scene(d, cam, opt)
        im = Image.fromarray(r.render().copy())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, H - 20, W, H], fill=(20, 20, 20))
        dr.text((8, H - 16), f"jaw {q:.2f} rad", fill=(240, 240, 240))
        tiles.append(np.asarray(im))
rows = [np.hstack(tiles[i::2]) for i in range(2)]
Image.fromarray(np.vstack(rows)).save(out)
print(out)
