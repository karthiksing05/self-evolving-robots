"""Films of a body at a task, for people reading a run (scripts/film_bodies.py, docs/evolve.md, D-050).

Each film replays one attempt with the controls the run recorded, so nothing is searched again:
- **Gripper:** the simulator itself. The pick is replayed in MuJoCo with System 1's recorded grasp
  parameters, and the verdict is the judge's on that replay. The film's copy of the scene is dressed for
  people (`_stage`): printed-part colours, a key and rim light with soft shadows, a table under the board
  and a studio backdrop, all visual only. The camera follows the jaws and the object; a level close-up on
  the moving fingertip draws MuJoCo's contact forces; stretches where nothing moves play faster, marked.
- **Talking robot:** the CAD at System 1's recorded posture, the formants the simulator computes for it,
  and the sound. The sound is a 120 Hz pulse train through the simulator's own transfer function. A
  stop is heard inside a vowel (/a·C·a/), with a release burst shaped by the cavity in front of the
  closure (`_utter`).
- **Mobile explorer:** an animation of the simulator's verdict, not physics. The world is quasi-static,
  so the CAD is posed over terrain built from the route's features, settled where its wheels or tracks
  rest. It stops where the replayed attempt, at the route's nominal dimensions, failed.

This module is kept out of the world modules, as the CAD is (sea/worlds/cad.py): a film is not part of
the ruler.  ``FILM`` maps a world's name to its function.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sea.evolve.core import Body, Task
from sea.worlds import cad as C
from sea.worlds import gripper as G
from sea.worlds import mobile as M
from sea.worlds import phonology as P

FPS = 25
SR = 22050  # audio sample rate
GOOD, BAD, INK, PAPER = (24, 140, 70), (200, 45, 40), (240, 240, 236), (22, 24, 28)


@dataclass
class Clip:
    frames: list[np.ndarray]  # H x W x 3, uint8
    ok: bool
    verdict: str  # what happened, in a line
    audio: np.ndarray | None = None  # mono, at SR, as long as the frames
    probe: Any = None  # what a step hook recorded over the filmed attempt (gripper; sea.mechanism.audit)
    clipping_mm: float = 0.0  # deepest a drawn robot part sits inside the object or the ground (CLIP_MM)


CLIP_MM = 1.5  # a film whose drawn robot goes deeper than this into what it touches is not written


def _robot_points(m, bodies: set[int], n: int = 3000) -> dict[int, np.ndarray]:
    """Points on the drawn robot surfaces (groups 0-2: meshes and pad boxes) of ``bodies``, per geom, in its frame."""
    import trimesh

    out = {}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bodies or m.geom_group[g] > 2:
            continue
        if m.geom_type[g] == 7:
            mid = m.geom_dataid[g]
            v, f = m.mesh_vertadr[mid], m.mesh_faceadr[mid]
            mesh = trimesh.Trimesh(m.mesh_vert[v:v + m.mesh_vertnum[mid]], m.mesh_face[f:f + m.mesh_facenum[mid]], process=False)
        elif m.geom_type[g] == 6:
            mesh = trimesh.creation.box(2 * m.geom_size[g])
        else:
            continue
        out[g] = mesh.sample(n, seed=1) if len(mesh.faces) else mesh.vertices
    return out


def _depth_hook(m, obj_body: int):
    """A step hook: how deep the drawn hand goes into the object's convex collision parts (m), every 5 ticks."""
    import trimesh

    names = {m.body(i).name for i in range(m.nbody)}
    hand = {m.body(b).id for b in ("gripper", "moving_jaw_so101_v1", "jaw_coupler", "jaw_follower", "jaw_carriage") if b in names}
    pts = _robot_points(m, hand)
    planes = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == obj_body and m.geom_group[g] == 3 and m.geom_type[g] == 7:
            mid = m.geom_dataid[g]
            v, f = m.mesh_vertadr[mid], m.mesh_faceadr[mid]
            hull = trimesh.Trimesh(m.mesh_vert[v:v + m.mesh_vertnum[mid]], m.mesh_face[f:f + m.mesh_facenum[mid]]).convex_hull
            planes.append((g, hull.face_normals, np.einsum("ij,ij->i", hull.face_normals, hull.triangles[:, 0])))
    state = {"n": 0, "deepest": 0.0}

    def hook(d):
        state["n"] += 1
        if state["n"] % 5:
            return
        for g, normal, off in planes:
            ro, to = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            for r, p in pts.items():
                local = (p @ d.geom_xmat[r].reshape(3, 3).T + d.geom_xpos[r] - to) @ ro
                state["deepest"] = max(state["deepest"], 1000 * float((off[None, :] - local @ normal.T).min(axis=1).max()))
    hook.state = state
    return hook


# ---------------------------------------------------------------- composing and encoding
@lru_cache(maxsize=8)
def _font(size: int):
    """DejaVu Sans, which matplotlib ships: unlike PIL's default it has the IPA a vowel's name is written in."""
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def caption(img: np.ndarray, top: str = "", bottom: str = "", colour: tuple[int, int, int] | None = None,
            size: int = 15) -> np.ndarray:
    """Text on dark bands at the top and bottom of a frame; ``colour`` tints the bottom band (a verdict)."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    pad = size // 2
    if top:
        d.rectangle([0, 0, w, size + 2 * pad], fill=(*PAPER, 200))
        d.text((pad + 2, pad - 1), top, fill=INK, font=_font(size))
    if bottom:
        lines = _wrap(bottom, w - 2 * pad, size)
        bh = len(lines) * (size + 3) + 2 * pad
        d.rectangle([0, h - bh, w, h], fill=(*(colour or PAPER), 255 if colour else 225))  # a verdict hides what's under it
        for i, line in enumerate(lines):
            d.text((pad + 2, h - bh + pad + i * (size + 3) - 1), line, fill=INK, font=_font(size))
    return np.asarray(im)


def _wrap(text: str, width: int, size: int) -> list[str]:
    font = _font(size)
    out, line = [], ""
    for word in text.split():
        if line and font.getlength(f"{line} {word}") > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*out, line][:4]


def _dim(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) * 0.45 + 255 * 0.55 * 0.25).astype(np.uint8)


def pair(before: Clip, after: Clip, title: str, labels: tuple[str, str], hold_s: float = 1.6) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Before and after side by side under a title, each ending on its verdict.  Clips with sound play in
    turn (the silent side dimmed), so each can be heard; silent clips play together."""
    from PIL import Image, ImageDraw

    def finish(c: Clip, label: str) -> list[np.ndarray]:
        tag = ("PASS · " if c.ok else "FAIL · ") + c.verdict
        body = [caption(f, label) for f in c.frames]
        end = caption(c.frames[-1], label, tag, GOOD if c.ok else BAD)
        return body, end

    (bf, be), (af, ae) = finish(before, labels[0]), finish(after, labels[1])
    hold = int(hold_s * FPS)
    if before.audio is not None or after.audio is not None:
        left = bf + [be] * hold + [_dim(be)] * len(af) + [be] * hold  # both verdicts at the end
        right = [_dim(af[0])] * (len(bf) + hold) + af + [ae] * hold
        silence = np.zeros(int(hold * SR / FPS), np.float32)
        audio = np.concatenate([before.audio if before.audio is not None else np.zeros(len(bf) * SR // FPS, np.float32),
                                silence,
                                after.audio if after.audio is not None else np.zeros(len(af) * SR // FPS, np.float32),
                                silence])
    else:
        n = max(len(bf), len(af))
        left = bf + [be] * (n - len(bf) + hold)
        right = af + [ae] * (n - len(af) + hold)
        audio = None
    h, w = left[0].shape[:2]
    bar = 34
    frames = []
    for lf, rf in zip(left, right, strict=True):
        canvas = np.full((h + bar, 2 * w + 6, 3), PAPER, np.uint8)
        canvas[bar:, :w], canvas[bar:, w + 6:] = lf, rf
        frames.append(canvas)
    head = Image.fromarray(frames[0][:bar])
    ImageDraw.Draw(head).text((10, 8), title, fill=INK, font=_font(16))
    head = np.asarray(head)
    for f in frames:
        f[:bar] = head
    return frames, audio


def encode(frames: list[np.ndarray], path: Path, audio: np.ndarray | None = None) -> None:
    """An H.264 MP4 (with AAC sound when there is audio), and a JPEG poster of the last frame beside it."""
    from PIL import Image
    from scipy.io import wavfile

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("films need ffmpeg on the PATH")
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    raw = np.stack([f[:h2, :w2] for f in frames]).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", str(FPS),
           "-i", "-"]
    with tempfile.TemporaryDirectory() as tmp:
        if audio is not None:
            wav = Path(tmp) / "a.wav"
            wavfile.write(wav, SR, (np.clip(audio, -1, 1) * 32000).astype(np.int16))
            cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "96k"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "medium", "-movflags", "+faststart",
                "-shortest" if audio is not None else "-an", str(path)]
        subprocess.run(cmd, input=raw, check=True)
    Image.fromarray(frames[-1]).save(path.with_suffix(".jpg"), quality=85)


# ---------------------------------------------------------------- the gripper: the simulator itself, dressed for people
PLA = (0.90, 0.88, 0.84, 1.0)  # printed parts, off-white as the CAD pictures draw them
SERVO = (0.14, 0.14, 0.15, 1.0)
BIRCH = (0.62, 0.50, 0.36, 1.0)  # the tabletop board
CYC = (np.array([238.0, 240.0, 242.0]), np.array([203.0, 206.0, 210.0]))  # studio backdrop: top, horizon
SLATE = (np.array([70.0, 76.0, 84.0]), np.array([38.0, 42.0, 48.0]))  # the close-up's darker backdrop
SS = 2  # supersampling: the backdrop goes in at twice the size, then the frame is filtered down
FAST = 4  # how much faster than real time a stretch where nothing moves plays
STILL = 0.004  # m/s: below this, fingertips and object are both still


def _stage(m, w: int, h: int) -> None:
    """Dress the film's own copy of the scene: colours, materials, its light and render quality.  Only visual
    fields change, none the physics reads, so the replay and its verdict are the judge's."""
    for i in range(m.nmat):
        servo = "sts3215" in m.mat(i).name
        m.mat_rgba[i] = SERVO if servo else PLA
        m.mat_specular[i], m.mat_shininess[i] = (0.5, 0.6) if servo else (0.15, 0.2)
    for g in range(m.ngeom):
        name, body = m.geom(g).name, m.body(m.geom_bodyid[g]).name
        if name == "floor":
            m.geom_rgba[g] = (*CYC[1] / 255, 1.0)
        elif name == "panel_board":
            m.geom_rgba[g] = BIRCH
        elif body.startswith("ctl_") and m.geom_group[g] <= 2:
            m.geom_rgba[g, :3] = m.geom_rgba[g, :3] ** 0.8  # a scan's mean texture colour, lifted a little
            if m.geom_type[g] == 7 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                _shrinkwrap(m, g)
        elif m.geom_matid[g] < 0 and np.allclose(m.geom_rgba[g, :3], 0.5) and m.geom_group[g] <= 2:
            m.geom_rgba[g] = PLA  # printed linkage parts left at the default grey
    m.light_diffuse[:], m.light_specular[:], m.light_castshadow[:] = 0.12, 0.0, False  # the studio rig lights it
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:], m.vis.headlight.specular[:] = 0.32, 0.30, 0.05
    m.vis.quality.shadowsize, m.vis.quality.offsamples = 8192, 4
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, SS * w), max(m.vis.global_.offheight, SS * h)


def _frame(m, g: int) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[g])
    return rot.reshape(3, 3), m.geom_pos[g].copy()


def _hulls(m, g: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The planes (normals, offsets) of each convex collision part of geom ``g``'s body, in that body's frame."""
    import trimesh

    out = []
    for c in range(m.ngeom):
        if m.geom_bodyid[c] != m.geom_bodyid[g] or m.geom_group[c] != 3 or m.geom_type[c] != 7:
            continue
        mid, (rot, pos) = m.geom_dataid[c], _frame(m, c)
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ rot.T + pos
        h = trimesh.convex.convex_hull(v)
        out.append((h.face_normals, np.einsum("ij,ij->i", h.face_normals, h.triangles[:, 0])))
    return out


def _outside(p: np.ndarray, hulls) -> np.ndarray:
    """How far each point lies outside every convex part (<= 0: inside one), metres."""
    return np.min([np.max(p @ n.T - off[None, :], axis=1) for n, off in hulls], axis=0)


def _shrinkwrap(m, g: int) -> None:
    """Pull the drawn scan of geom ``g`` back inside its body's collision parts.  A decimated scan sticks out
    of the convex parts the simulator collides with by up to 3 mm (the sugar box), so a fingertip resting on
    the simulated surface was drawn inside the scan.  Each vertex outside every part is projected onto the
    nearest part (alternating projections onto its violated faces); the scan collides with nothing, so the
    physics is untouched and what is drawn is what is simulated."""
    hulls = _hulls(m, g)
    if not hulls:
        return
    mid, (rot, pos) = m.geom_dataid[g], _frame(m, g)
    sl = slice(m.mesh_vertadr[mid], m.mesh_vertadr[mid] + m.mesh_vertnum[mid])
    v = m.mesh_vert[sl].astype(float) @ rot.T + pos
    out = _outside(v, hulls) > 0
    if not out.any():
        return
    near = np.argmin([np.max(v[out] @ n.T - off[None, :], axis=1) for n, off in hulls], axis=0)
    p = v[out]
    for k, (n, off) in enumerate(hulls):
        q = p[near == k]
        for _ in range(60):
            viol = q @ n.T - off[None, :]
            worst = np.argmax(viol, axis=1)
            over = viol[np.arange(len(q)), worst]
            if not (over > 0).any():
                break
            q = q - np.maximum(over, 0)[:, None] * n[worst] * 1.02
        p[near == k] = q
    v[out] = p
    m.mesh_vert[sl] = (v - pos) @ rot


def _studio(scn, target: np.ndarray, azimuth: float) -> None:
    """A key light with soft shadows, from the camera's front left and above; a cool rim from behind."""
    import mujoco

    a = math.radians(azimuth)
    fwd = np.array([math.cos(a), math.sin(a), 0.0])
    left = np.array([-math.sin(a), math.cos(a), 0.0])
    for off, diffuse, spec, shadow in ((-0.45 * fwd + 0.35 * left + [0, 0, 0.75], 0.62, 0.25, True),
                                       (0.55 * fwd - 0.2 * left + [0, 0, 0.45], 0.30, 0.10, False)):
        if scn.nlight >= len(scn.lights):
            return
        L = scn.lights[scn.nlight]
        scn.nlight += 1
        pos = target + off
        L.pos, L.dir = pos, (target - pos) / np.linalg.norm(target - pos)
        L.type, L.headlight, L.castshadow = mujoco.mjtLightType.mjLIGHT_SPOT, 0, shadow
        L.diffuse, L.specular, L.ambient = [diffuse] * 3, [spec] * 3, [0.0] * 3
        L.cutoff, L.exponent, L.attenuation = 45.0, 2.0, [1.0, 0.0, 0.0]
        L.bulbradius, L.softness, L.range, L.intensity, L.id, L.texid = 0.08, 0.6, 10.0, 0.0, -1, -1


def _table(m, d) -> list[tuple[np.ndarray, np.ndarray, tuple[float, ...]]]:
    """Boxes (half sizes, centre, colour) for a table under the board and the arm's base.  In the judge's scene
    the board is a 1 cm plank at z = 0 over a floor 0.4 m down, which reads as a plank floating in the air; the
    table goes into the drawn scene only (mjvScene), so nothing collides with it."""
    g = m.geom("panel_board").id
    top = float(d.geom_xpos[g][2] - m.geom_size[g][2]) - 2e-4
    floor = m.geom("floor").id
    ground = float(d.geom_xpos[floor][2] + m.geom_size[floor][2])
    cx, hx, hy, t = 0.2, 0.45, 0.42, 0.02
    boxes = [(np.array([hx, hy, t]), np.array([cx, 0.0, top - t]), (0.80, 0.80, 0.78, 1.0))]
    leg = (top - 2 * t - ground) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            boxes.append((np.array([0.02, 0.02, leg]), np.array([cx + sx * (hx - 0.05), sy * (hy - 0.05), ground + leg]),
                          (0.30, 0.31, 0.33, 1.0)))
    return boxes


class _View:
    """One camera drawn over the studio backdrop: rendered at SS times the size with its depth, the empty
    pixels (nothing hit) painted with the backdrop, then filtered down so silhouettes stay clean.  ``props``
    are boxes drawn into the scene only (``_table``)."""

    def __init__(self, m, w: int, h: int, props=(), cyc=CYC):
        import mujoco

        self.w, self.h, self.props = w, h, list(props)
        self.colour, self.depth = mujoco.Renderer(m, SS * h, SS * w), mujoco.Renderer(m, SS * h, SS * w)
        self.depth.enable_depth_rendering()
        self.far = 0.99 * m.vis.map.zfar * m.stat.extent
        y = np.linspace(0.0, 1.0, SS * h)[:, None, None] ** 0.8
        self.backdrop = (cyc[0] * (1 - y) + cyc[1] * y).astype(np.float32)

    def shot(self, d, cam, opt) -> np.ndarray:
        import mujoco

        for r in (self.colour, self.depth):
            r.update_scene(d, cam, opt)
            scn = r.scene
            for half, pos, rgba in self.props:
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, half, pos, np.eye(3).ravel(),
                                        np.array(rgba, np.float32))
                    scn.ngeom += 1
            _studio(scn, np.array(cam.lookat), cam.azimuth)
        img = self.colour.render().astype(np.float32)
        empty = self.depth.render() >= self.far
        img = np.where(empty[..., None], self.backdrop, img)
        return img.reshape(self.h, SS, self.w, SS, 3).mean((1, 3)).astype(np.uint8)

    def close(self) -> None:
        self.colour.close()
        self.depth.close()


def _tips(m, d, pts: dict[str, dict[int, np.ndarray]]) -> dict[str, np.ndarray]:
    """The lowest drawn point of each finger, world frame: where the hand meets the object."""
    out = {}
    for side, by in pts.items():
        world = np.concatenate([p @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, p in by.items()])
        out[side] = world[np.argmin(world[:, 2])]
    return out


def _inset(img: np.ndarray, small: np.ndarray, label: str) -> np.ndarray:
    """``small`` framed in the top right corner of ``img``, under the label band, with its own label."""
    from PIL import Image, ImageDraw

    h, w = small.shape[:2]
    x, y = img.shape[1] - w - 10, 40
    out = img.copy()
    out[y - 2:y + h + 2, x - 2:x + w + 2] = 250
    out[y:y + h, x:x + w] = small
    im = Image.fromarray(out)
    ImageDraw.Draw(im, "RGBA").rectangle([x, y + h - 20, x + w, y + h], fill=(*PAPER, 170))
    ImageDraw.Draw(im).text((x + 6, y + h - 18), label, fill=INK, font=_font(12))
    return np.asarray(im)


def gripper(body: Body, task: Task, controls: dict[str, float], seed: int = 0, size: tuple[int, int] = (720, 540),
            probe: Callable[[Any], Callable[[Any], None]] | None = None, render: bool = True) -> Clip:
    """Replay the pick in MuJoCo with the recorded grasp parameters under the run's ``seed``, whose
    domain-randomization draw the verdict was judged on.  System 1 records {} for its first attempt and for
    its fixed retry alike (sea.control.local_search.run_system1), so for {} both are run and the one System 1
    kept is filmed.  ``probe(model)`` makes a step hook, fresh for each attempt, called with the MuJoCo data
    every tick; the kept attempt's hook is returned on the clip.  ``render=False`` replays without filming."""
    from dataclasses import replace

    import mujoco

    from sea.compose.library import Library
    from sea.contracts import PanelSpec
    from sea.control import executor as X
    from sea.control.local_search import STRUCTURAL, objective, relocalized_offset
    from sea.design.build import build
    from sea.sim.env import PanelRunner, SuiteConfig, _judge, candidate_programs

    limits = G._limits()
    built = build(G.to_design(body), limits)
    cfg = SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=limits, seed=seed)
    panel = PanelSpec.model_validate(task.data["panel"])
    obj = panel.controls[0]
    params: dict[str, dict[str, float]] = {}
    for k, v in controls.items():
        prim, key = k.split(".", 1)
        params.setdefault(prim, {})[key] = float(v)
    prims = set(params)
    progs = candidate_programs(obj, panel, cfg)
    prog = next((p for _, p in progs if prims <= set(re.findall(r"[a-z_]+", str(p)))), progs[0][1])
    runner = PanelRunner(panel, cfg, 0)
    m = runner.sim.model
    w, h = size
    iw, ih = int(w * 0.36), int(h * 0.36)
    if render:
        _stage(m, w, h)
    main = close = None
    if render:
        d0 = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d0)
        table = _table(m, d0)
        main, close = _View(m, w, h, table), _View(m, iw, ih, table, SLATE)
        # contact force arrows (MuJoCo scales length by the mean body mass, width by the mean size): 25 N to 6 cm
        m.vis.map.force, m.vis.scale.forcewidth = 0.0024 * m.stat.meanmass, 0.003 / m.stat.meansize
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    forces = mujoco.MjvOption()
    forces.geomgroup[:] = opt.geomgroup
    forces.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
    cam, side = mujoco.MjvCamera(), mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 145.0, -20.0
    side.distance, side.elevation = 0.13, 0.0  # level with the fingertip: a top face is seen edge on
    names = {m.body(i).name for i in range(m.nbody)}
    moving = {m.body(b).id for b in ("moving_jaw_so101_v1", "jaw_coupler", "jaw_carriage") if b in names}
    fingers = {"fixed": _robot_points(m, {m.body("gripper").id}, 300), "moving": _robot_points(m, moving, 300)}
    judge = _judge(obj, panel, limits)
    shots: list[list[np.ndarray]] = []
    hooks: list[Any] = []
    depths: list[Any] = []
    orig = X.Episode.tick
    state: dict[str, Any] = {}

    def tick(ep, q):
        orig(ep, q)
        if hooks:
            hooks[-1](ep.sim.data)
        if depths:
            depths[-1](ep.sim.data)
        if main is None:
            return
        d = ep.sim.data
        tips = _tips(m, d, fingers)
        mid, p = (tips["fixed"] + tips["moving"]) / 2, d.xpos[ep.sim.controls[obj.id].body_id].copy()
        # a frame every 2 ticks (real time at 25 frames/s); FAST times as far apart while nothing moves
        prev = state.get("last")
        speed = 0.0 if prev is None else max(np.linalg.norm(mid - prev[1]), np.linalg.norm(p - prev[2])) / max(ep.t - prev[0], 1e-6)
        still = prev is not None and speed < STILL
        due = state.get("due", 0.0)
        if ep.t + 1e-9 < due:
            return
        state["last"], state["due"] = (ep.t, mid, p), ep.t + (2 * FAST if still else 2) * X.TICK_DT
        target = np.array([(mid[0] + p[0]) / 2, (mid[1] + p[1]) / 2, max(0.5 * (mid[2] + p[2]), 0.04)])
        span = float(np.linalg.norm(mid - p))
        look, dist = state.get("look"), float(np.clip(0.34 + 1.1 * span, 0.36, 0.56))
        k = 1.0 if look is None else 0.25
        state["look"] = target if look is None else (1 - k) * look + k * target
        state["dist"] = dist if look is None else (1 - k) * state["dist"] + k * dist
        cam.lookat[:], cam.distance = state["look"], state["dist"]
        img = main.shot(d, cam, opt)
        # the close-up looks at the moving fingertip, level, so a tip on a rim reads as on it, not in it
        line = tips["moving"] - tips["fixed"]
        az = math.degrees(math.atan2(line[1], line[0]))
        across = min((az + 90, az - 90), key=lambda a: abs((a - cam.azimuth + 180) % 360 - 180))
        side.azimuth = across + 35 * np.sign(((cam.azimuth - across + 180) % 360) - 180)  # turned toward the main view
        side.lookat[:] = tips["moving"]
        img = _inset(img, close.shot(d, side, forces), "moving fingertip · arrows: contact forces")
        up = (ep.sim.control_q(obj.id) - ep.q_rest0[obj.id]) * 1000
        pace = f"   ▸▸ {FAST}× (nothing moving)" if still else ""
        shots[-1].append(caption(img, bottom=f"t {ep.t:4.1f} s   lifted {max(up, 0.0):.0f} mm{pace}"))

    def take(ecfg, n: int):
        shots.append([])
        state.clear()
        if probe is not None:
            hooks.append(probe(runner.sim.model))
        if render:
            depths.append(_depth_hook(runner.sim.model, runner.sim.controls[obj.id].body_id))
        X.reset_episode(runner.sim)
        tr = X.run_program(runner.sim, prog, obj.id, ecfg, panel_id=panel.id, attempt=n)
        return tr, judge(tr)

    X.Episode.tick = tick
    try:
        ecfg = runner.config(obj)
        ecfg.param_overrides = {k: dict(v) for k, v in params.items()}
        takes = [take(ecfg, 0)]
        if not params and takes[0][1].verdict != "pass" and takes[0][1].verdict not in STRUCTURAL:
            off = relocalized_offset(runner.sim, obj.id, ecfg, np.random.default_rng(runner.seed))
            takes.append(take(replace(ecfg, perception_offset=off), 1))
    finally:
        X.Episode.tick = orig
        for v in (main, close):
            if v is not None:
                v.close()
    i = 0
    if len(takes) == 2:  # run_system1: the retry if it passed or stopped structurally, else the better of the two
        travel = runner.sim.controls[obj.id].spec.travel
        r1 = takes[1][1].verdict
        i = 1 if r1 == "pass" or r1 in STRUCTURAL else min((0, 1), key=lambda j: objective(takes[j][1], takes[j][0], travel))
    (tr, res), frames = takes[i], shots[i]
    ok = res.verdict == "pass"
    note = (tr.note or "").replace("_", " ")
    return Clip(frames or [np.zeros((h, w, 3), np.uint8)], ok, "held for the full second" if ok else
                f"{res.verdict.replace('_', ' ')}" + (f" ({note})" if note else ""), probe=hooks[i] if hooks else None,
                clipping_mm=depths[i].state["deepest"] if depths else 0.0)


# ---------------------------------------------------------------- the talking robot: posture, formants, sound
def _speak(gain_db: np.ndarray, seconds: float = 1.2, f0: float = 120.0) -> np.ndarray:
    """A pulse-train voice through a tract: harmonic k of a gently falling f0 at the transfer function's gain,
    tilted by the glottal source (-12 dB/octave) and the lips' radiation (+6 dB/octave)."""
    t = np.arange(int(seconds * SR)) / SR
    pitch = f0 * (1.08 - 0.12 * t / seconds) * (1 + 0.004 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    out = np.zeros_like(t)
    for k in range(1, int(4800 / f0)):
        fk = k * pitch
        amp = 10 ** (np.interp(fk, P.FREQ, gain_db) / 20) / k
        out += amp * np.sin(k * phase)
    env = np.minimum(1, np.minimum(t / 0.04, (seconds - t) / 0.08))
    out *= np.clip(env, 0, 1)
    return (0.8 * out / max(np.abs(out).max(), 1e-9)).astype(np.float32)


def _noise(gain_db: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of white noise shaped by a gain over FREQ (above 5 kHz it keeps falling, 12 dB/octave)."""
    f = np.fft.rfftfreq(n, 1 / SR)
    top = gain_db[-1] - 12 * np.log2(np.maximum(f, P.FREQ[-1]) / P.FREQ[-1])
    g = np.where(f <= P.FREQ[-1], np.interp(f, P.FREQ, gain_db), top)
    out = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * 10 ** (g / 20), n)
    return out / max(np.abs(out).max(), 1e-9)


def _utter(r: P.Rig, ctl: dict[str, float], seed: int = 0, f0: float = 120.0) -> np.ndarray:
    """A stop heard the way a listener hears one: inside a vowel, as /a·C·a/.  The tract glides from an open
    posture into the recorded one and back, voiced by a pulse train through the simulator's transfer function
    at each instant.  If the recorded posture seals the tube (< A_CLOSE), the voice stops for the closure,
    then the release bursts: noise at the closure, shaped by the cavity in front of it, which is what tells
    /p/, /t/ and /k/ apart (a front cavity of a few cm rings near 1.5-3 kHz for /k/; none for /p/), then
    aspiration through the opening tract.  A posture that does not seal has no burst: the voice runs on
    through the narrowing, and what is heard is a vowel, not a stop."""
    rng = np.random.default_rng(seed)
    lo = {k: v[0] for k, v in r.controls.items()}
    hi = {k: v[1] for k, v in r.controls.items()}
    open_ = {k: (hi[k] if k in ("jaw", "lip_radius") else lo[k] if k.endswith("height") or k in ("protrusion", "larynx")
                 else ctl[k]) for k in ctl}
    areas_c, lens_c, dist_c = P.tract(r, {k: np.array([v]) for k, v in ctl.items()})
    sealed = float(areas_c.min()) < P.A_CLOSE
    v1, glide, hold, burst, asp, v2 = 0.28, 0.06, 0.10, 0.025, 0.06, 0.32
    # the area function along the utterance, every 5 ms: open -> closure -> (hold) -> open
    hop = 0.005
    t_in, t_hold = v1, v1 + glide
    t_rel = t_hold + (hold if sealed else 0.04)
    total = t_rel + burst + asp + glide + v2
    ts = np.arange(0, total + hop, hop)
    w = np.where(ts < t_rel, np.clip((ts - t_in) / glide, 0, 1), np.clip(1 - (ts - t_rel) / (burst + asp + glide), 0, 1))
    post = {k: np.array([(1 - x) * open_[k] + x * ctl[k] for x in w]) for k in ctl}
    areas, lens, _ = P.tract(r, post)
    gains = P.transfer_db(np.maximum(areas, 0.08), lens)  # (frames, FREQ)
    t = np.arange(int(total * SR)) / SR
    voiced = np.ones_like(t)
    if sealed:  # voiceless: the glottis stops as the seal forms and starts again after the aspiration
        voiced = np.clip(np.maximum((t_hold - t) / 0.015, (t - (t_rel + burst + asp)) / 0.02), 0, 1)
    pitch = f0 * (1.06 - 0.10 * t / total) * (1 + 0.004 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    fi = np.clip(t / hop, 0, len(ts) - 1)
    voice = np.zeros_like(t)
    for k in range(1, int(4800 / f0)):
        g = np.array([np.interp(k * f0, P.FREQ, row) for row in gains])  # gain at harmonic k, per frame
        voice += 10 ** (np.interp(fi, np.arange(len(ts)), g) / 20) / k * np.sin(k * phase)
    voice *= voiced * np.clip(np.minimum(t / 0.04, (total - t) / 0.08), 0, 1)
    voice = 0.75 * voice / max(np.abs(voice).max(), 1e-9)
    if not sealed:
        return voice.astype(np.float32)
    # the burst: noise at the closure through the sections in front of it, mouth radiation included
    i = int(np.argmin(areas_c[0]))
    front_a, front_l = areas_c[:, i + 1:], lens_c[:, i + 1:]
    if front_a.shape[1]:
        g_burst = P.transfer_db(np.maximum(front_a, 0.08), front_l)[0]
    else:  # a lip closure has no front cavity: a diffuse burst, falling with frequency
        g_burst = -6 * np.log2(np.maximum(P.FREQ, 500) / 500)
    nb, na = int(burst * SR), int((asp + glide / 2) * SR)
    b = _noise(g_burst, nb, rng) * np.exp(-np.arange(nb) / (0.008 * SR))  # a release lasts tens of ms (velars longest)
    j = int(np.searchsorted(ts, t_rel + burst + asp / 2))  # aspiration: glottal noise through the opening tract
    a = _noise(gains[min(j, len(ts) - 1)], na, rng) * np.linspace(1, 0, na) ** 1.5
    s0 = int(t_rel * SR)
    voice[s0:s0 + nb] += 0.55 * b[:len(voice) - s0]
    voice[s0 + nb:s0 + nb + na] += 0.25 * a[:max(0, len(voice) - s0 - nb)]
    return (0.9 * voice / max(np.abs(voice).max(), 1e-9)).astype(np.float32)


def phonology(body: Body, task: Task, controls: dict[str, float], seed: int = 0, size: tuple[int, int] = (480, 330)) -> Clip:
    """The body at the recorded posture: its CAD, what the simulator hears and, for a vowel, the sound.  A
    posture's verdict draws on no seed; ``seed`` keeps the signature common to every world."""
    from PIL import Image

    from sea.design.cad import render

    r = P.rig(body)
    ctl = {k: float(np.clip(controls.get(k, (lo + hi) / 2), lo, hi)) for k, (lo, hi) in r.controls.items()}
    arr = {k: np.array([v]) for k, v in ctl.items()}
    f, clo = P.acoustics(r, arr)
    ok = float(P.PhonologyWorld._errors(task, f, clo)[0]) <= 1.0
    said = P.PhonologyWorld._said(task, f[0], clo[0])
    model = C.phonology(body, ctl)
    model.views, model.tile = model.views[:1], size
    img = np.asarray(Image.open(__import__("io").BytesIO(render([model])[0])).convert("RGB"))
    if task.data["kind"] == "vowel":
        areas, lens, _ = P.tract(r, arr)
        audio = _speak(P.transfer_db(np.maximum(areas, 0.08), lens)[0])
        frame = caption(img, bottom=said)
        return Clip([frame] * int(len(audio) * FPS / SR), ok, said, audio)
    audio = _utter(r, ctl, seed)
    return Clip([caption(img, bottom=said)] * int(len(audio) * FPS / SR), ok, said, audio)


# ---------------------------------------------------------------- the mobile explorer: an animation of the verdict
SHOW_M = {"floor": 0.35, "ramp": 0.45, "sand": 0.45, "low": 0.4, "dark": 0.55}  # metres shown of each long feature


def _terrain(route: list[dict]) -> tuple[list[tuple[str, float, float, dict]], np.ndarray, np.ndarray]:
    """Lay the route's features out along x (mm): each as (kind, start, end, feature), and the ground's height
    profile (x, z), with long features shortened for the screen."""
    x, z, out = 300.0, 0.0, []
    xs, zs = [0.0], [0.0]
    for f in route:
        k = f["k"]
        if k == "step":
            xs += [x, x + 0.01]
            zs += [z, z + 10 * f["h"]]
            z += 10 * f["h"]
            out.append((k, x, x + 1, f))
            x += 120
        elif k == "trench":
            w, d = 10 * f["w"], 10 * f["depth"]
            xs += [x, x + 0.01, x + w, x + w + 0.01]
            zs += [z, z - d, z - d, z]
            out.append((k, x, x + w, f))
            x += w + 120
        elif k == "ramp":
            run = 1000 * SHOW_M["ramp"]
            xs += [x, x + run]
            zs += [z, z + run * math.tan(math.radians(f["deg"]))]
            z = zs[-1]
            out.append((k, x, x + run, f))
            x += run
        elif k == "gap":
            out.append((k, x, x + 60, f))
            x += 260
        else:
            run = 1000 * min(f.get("len", 0.3), SHOW_M.get(k, 0.35))
            out.append((k, x, x + run, f))
            x += run
        xs.append(x)
        zs.append(z)
    xs.append(x + 400)
    zs.append(z)
    return out, np.array(xs), np.array(zs)


def _settle(bottom: np.ndarray, com: np.ndarray, gx: np.ndarray, gz: np.ndarray, x0: float, sink: float = 0.0) -> tuple[float, float]:
    """Pitch (rad) and lift (mm) that rest a body's side outline, shifted ``x0`` along the ground, on the
    terrain with its centre of mass lowest (quasi-static).  The outline is every drawn point, not only the
    lowest: a wheel's front or a track's leading run meets a step's riser above the body's bottom, and with
    only the bottom points it was drawn through the riser."""
    best = (math.inf, 0.0, 0.0)
    for th in np.radians(np.arange(-40, 41, 1.0)):
        c, s = math.cos(th), math.sin(th)
        px = x0 + bottom[:, 0] * c - bottom[:, 1] * s
        pz = bottom[:, 0] * s + bottom[:, 1] * c
        lift = float(np.max(np.interp(px, gx, gz) - pz)) - sink
        h = lift + com[0] * s + com[1] * c
        if h < best[0] - 1e-6:
            best = (h, th, lift)
    return best[1], best[2]


@dataclass
class Drive:
    """A mobile film's course: the drawn body, the terrain, and the body's pose (x, pitch, lift) per frame."""

    model: Any
    feats: list[tuple[str, float, float, dict]]
    gx: np.ndarray
    gz: np.ndarray
    poses: list[tuple[float, float, float]]
    detail: str
    ok: bool
    mech: str
    front: float
    sink: float = 0.0  # mm the last pose sinks into sand: the verdict, drawn

    def clipping_mm(self) -> float:
        """Deepest a drawn point goes below the ground over the course, beyond the sand's sinkage."""
        pts = np.vstack([p.mesh.vertices for p in self.model.parts])[:, [0, 2]]
        deepest = 0.0
        for x0, th, lift in self.poses:
            c, s = math.cos(th), math.sin(th)
            px, pz = x0 + pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c + lift
            deepest = max(deepest, float(np.max(np.interp(px, self.gx, self.gz) - pz)))
        return max(0.0, deepest - self.sink)


def drive(body: Body, task: Task, controls: dict[str, float]) -> Drive:
    """Where the robot is in each frame: driven along its route as the simulator's verdict says, stopping
    where the replayed attempt (at the route's nominal dimensions) failed."""
    s = M._spec(body)
    route, secs = task.data["route"], float(task.data["secs"])
    ctl = {**M.PROBE, **controls}
    ok, _, mech, detail = M._attempt(s, route, secs, ctl, M._Calm())
    hit = re.search(r"stopped at feature (\d+)/", detail)
    stop_at = int(hit.group(1)) - 1 if hit else None  # a run that is only too slow drives the whole route
    feats, gx, gz = _terrain(route)
    model = C.mobile(body)
    verts = np.vstack([p.mesh.vertices for p in model.parts])
    outline = verts[:, [0, 2]]
    outline = outline[:: max(1, len(outline) // 4000)]
    com = verts[:, [0, 2]].mean(0)
    front = float(verts[:, 0].max())
    sink = 0.0
    if stop_at is None:
        x_end = gx[-1] - 400 - front
    else:
        kind, a, b, f = feats[stop_at]
        x_end = {"trench": a + 20, "sand": (a + b) / 2, "dark": b - 60}.get(kind, a - 5) - (0 if kind in ("trench", "sand", "dark") else front)
        if kind == "sand":
            sink = 10 * min(4.0, s.m * M.G / (s.area * 1e-4) / 1000 / f["kmod"])
        if detail.find("ran out") >= 0:
            x_end = (a + b) / 2 - front
    xs = np.linspace(0.0, max(x_end, 0.0), max(12, int(abs(x_end) / 12)))
    poses = [(float(x0), *_settle(outline, com, gx, gz, x0, sink if (i == len(xs) - 1 and sink) else 0.0))
             for i, x0 in enumerate(xs)]
    return Drive(model, feats, gx, gz, poses, detail, ok, mech, front, sink)


def mobile(body: Body, task: Task, controls: dict[str, float], seed: int = 0, size: tuple[int, int] = (480, 300)) -> Clip:
    """The robot driven along its route as the simulator's verdict says (``drive``): it stops where the
    replayed attempt (at the route's nominal dimensions, so ``seed`` is unused) failed, and says why."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    from sea.design.cad import MATERIALS, _polydata

    s = M._spec(body)
    d = drive(body, task, controls)
    model, feats, gx, gz, front, ok, detail, mech = d.model, d.feats, d.gx, d.gz, d.front, d.ok, d.detail, d.mech

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.955, 0.95, 0.94)
    kit = vtk.vtkLightKit()
    kit.AddLightsToRenderer(ren)
    actors = []
    for p in model.parts:
        mp = vtk.vtkPolyDataMapper()
        mp.SetInputData(_polydata(p.mesh))
        a = vtk.vtkActor()
        a.SetMapper(mp)
        a.GetProperty().SetColor(*MATERIALS.get(p.material, MATERIALS["printed"]))
        ren.AddActor(a)
        actors.append(a)
    surf = {"grass": (0.55, 0.72, 0.45), "sand": (0.86, 0.78, 0.55), "concrete": (0.72, 0.72, 0.70),
            "gravel": (0.62, 0.60, 0.56), "carpet": (0.66, 0.58, 0.70), "tile": (0.85, 0.84, 0.80)}
    width = max(400.0, 10 * s.width + 200)
    # the ground is cut away 15 mm outside the robot's near side, as a section: a step or a trench wall in front
    # of the robot hid its lower half, which read as it sinking into the ground
    near = float(min(p.mesh.vertices[:, 1].min() for p in model.parts)) - 15.0

    def slab(x0, x1, z0, z1, y0, y1, rgb):
        src = vtk.vtkCubeSource()
        src.SetBounds(x0, x1, y0, y1, z0, z1)
        mp = vtk.vtkPolyDataMapper()
        mp.SetInputConnection(src.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(mp)
        a.GetProperty().SetColor(*rgb)
        ren.AddActor(a)

    for i in range(len(gx) - 1):  # the ground, as columns under its profile
        if gx[i + 1] - gx[i] < 0.5:
            continue
        mid = (gx[i] + gx[i + 1]) / 2
        kind = next((f for k, a, b, f in feats if a <= mid <= b and k != "gap"), None)
        rgb = surf.get(kind["s"], (0.8, 0.76, 0.68)) if kind else (0.8, 0.76, 0.68)
        if kind and kind["k"] == "dark":
            rgb = (0.25, 0.26, 0.30)
        z0, z1 = gz[i], gz[i + 1]
        n = max(1, int((gx[i + 1] - gx[i]) // 40)) if abs(z1 - z0) > 1 else 1
        for j in range(n):
            xa = gx[i] + (gx[i + 1] - gx[i]) * j / n
            xb = gx[i] + (gx[i + 1] - gx[i]) * (j + 1) / n
            za = np.interp((xa + xb) / 2, gx, gz)
            slab(xa, xb, min(gz.min(), 0) - 40, za, near, width / 2, rgb)
    for k, a, b, f in feats:
        z = float(np.interp(a, gx, gz))
        if k == "gap":  # a door left ajar: two leaves, the opening offset from the robot's line
            half, off = 5 * f["w"], 10 * f["off"]
            slab(a, a + 40, z, z + 260, off + half, width / 2, (0.78, 0.66, 0.50))
            slab(a, a + 40, z, z + 260, -width / 2, off - half, (0.78, 0.66, 0.50))
        elif k == "low":  # the sofa's underside
            slab(a, b, z + 10 * f["clear"], z + 10 * f["clear"] + 120, -width / 2, width / 2, (0.45, 0.35, 0.55))
        elif k == "dark":
            for j in range(3):
                slab(a + (j + 1) * (b - a) / 4, a + (j + 1) * (b - a) / 4 + 50, z, z + 80,
                     -width / 2 + 20 + 60 * (j % 2), -width / 2 + 90 + 60 * (j % 2), (0.15, 0.15, 0.17))

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    w, h = size
    win.SetSize(w, h)
    win.AddRenderer(ren)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetParallelScale(170.0)
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.ReadFrontBufferOff()
    frames = []
    for x0, th, lift in d.poses:
        tf = vtk.vtkTransform()
        tf.PostMultiply()
        tf.RotateY(-math.degrees(th))
        tf.Translate(x0, 0, lift)
        for a in actors:
            a.SetUserTransform(tf)
        cx = x0 + front / 2
        cz = float(np.interp(cx, gx, gz)) + 60
        cam.SetFocalPoint(cx, 0, cz)
        # nearly side-on: from ahead of the robot, a step or a ramp between the camera and the robot hid its
        # rear, which read as the body sinking into the ground
        cam.SetPosition(cx + 250, -1900, cz + 650)
        cam.SetViewUp(0, 0, 1)
        ren.ResetCameraClippingRange()
        win.Render()
        grab.Modified()
        grab.Update()
        img = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3].copy()
        here = next((M._desc(f) for k, a, b, f in feats if a - front - 60 <= x0 <= b), "")
        frames.append(caption(img, bottom=here))
    win.Finalize()
    verdict = detail.split(": ", 1)[1] if (not ok and ": " in detail) else detail
    return Clip(frames, ok, verdict if ok else f"{mech.replace('_', ' ')}: {verdict}", clipping_mm=d.clipping_mm())


FILM: dict[str, Callable[..., Clip]] = {"gripper": gripper, "gripper_dense": gripper, "mobile": mobile,
                                        "phonology": phonology}
# The harness components each replay reads (sea/harness.py): a run recorded when any of them differed is not
# filmed, since today's replay would show what the run never saw.
_GRIPPER = ("world_gripper", "physics", "tools", "grasp_world", "verifier", "contracts", "limits", "kinematics",
            "primitives", "sensors", "build")
# A mobile or talking-robot attempt is its world's own calculation: the evolution core gives it only data types
# (Body, Part, Task), and what changes there (costs, ledgers, proposal rules: D-043..D-047) never reaches a
# replay.  A replay that still disagrees with the run's record is not filmed (scripts/film_bodies.py).
READS: dict[str, tuple[str, ...]] = {"gripper": _GRIPPER, "gripper_dense": _GRIPPER, "mobile": ("world_mobile",),
                                     "phonology": ("world_phonology",)}
