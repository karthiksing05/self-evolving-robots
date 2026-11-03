"""Films of a body at a task, for people reading a run (scripts/film_bodies.py, docs/evolve.md, D-050).

Each film replays one attempt with the controls the run recorded, so nothing is searched again:
- **Gripper:** the simulator itself. The pick is replayed in MuJoCo with System 1's recorded grasp
  parameters, and the verdict is the judge's on that replay. The film's copy of the scene is dressed for
  people (`_stage`): printed-part colours, a key and rim light with soft shadows, a table under the board
  and a studio backdrop, all visual only. The camera looks across the closing fingers at the object, a strip
  under it plots fingertip and object height (held: they rise together), and stretches where nothing moves
  play faster, marked.
- **Talking robot:** the tube cut in half at System 1's recorded posture, reed on the left and lips on the
  right, with the sound. The sound is a 120 Hz pulse train through the simulator's own transfer function. A
  stop is heard inside a vowel (/a·C·a/), with a release burst shaped by the cavity in front of the
  closure (`_utter`), and the picture follows the tube closing and opening with it; a vowel's resonances
  are plotted against its targets. The CAD sits small beside it.
- **Mobile explorer:** the simulator itself (D-063). Each step, trench, ramp and opening on the route is the
  judge's MuJoCo episode at the recorded controls, rerun with its trajectory, up to where the robot
  stopped, seen from the side (a doorway from above) with the obstacle's size drawn on it. A failure
  MuJoCo does not drive (sand, an overhang, the dark, the battery) ends on a still labelled as calculated,
  with the world's numbers.

Every film ends each side on its outcome in plain words over the judge's numbers, under a line saying what
the change was (``pair``).

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
    verdict: str  # what happened, in a line, with the judge's numbers
    audio: np.ndarray | None = None  # mono, at SR, as long as the frames
    probe: Any = None  # what a step hook recorded over the filmed attempt (gripper; sea.mechanism.audit)
    clipping_mm: float = 0.0  # deepest a drawn robot part sits inside the object or the ground (CLIP_MM)
    headline: str = ""  # the same in plain words, for someone who has never seen the project


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
@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False):
    """DejaVu Sans, which matplotlib ships: unlike PIL's default it has the IPA a vowel's name is written in."""
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                                                                weight="bold" if bold else "normal")), size)


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


def _wrap(text: str, width: int, size: int, bold: bool = False) -> list[str]:
    font = _font(size, bold)
    out, line = [], ""
    for word in text.split():
        if line and font.getlength(f"{line} {word}") > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*out, line][:4]


LABEL, FOOT = 28, 64  # each side's name band above its picture, and its outcome band below
ACCENT = (232, 128, 30)  # what a change made or altered, wherever a film draws it
MUTED = (150, 154, 160)


def _band(w: int, h: int, fill: tuple[int, int, int], lines: list[tuple[str, int, bool, tuple[int, int, int]]]) -> np.ndarray:
    """A ``w`` x ``h`` band of colour ``fill`` with ``lines`` of (text, size, bold, colour) from its top, wrapped."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), fill)
    d = ImageDraw.Draw(im)
    y = 6
    for text, size, bold, colour in lines:
        for line in _wrap(text, w - 20, size, bold)[:2]:
            if y + size > h:
                break
            d.text((10, y), line, fill=colour, font=_font(size, bold))
            y += size + 4
    return np.asarray(im)


def pair(before: Clip, after: Clip, title: str, labels: tuple[str, str], hold_s: float = 1.6,
         change: str = "") -> tuple[list[np.ndarray], np.ndarray | None]:
    """Before and after under the task and what changed, each named above its picture and ending on its outcome
    below it, in plain words over the judge's numbers.  Silent clips play side by side, together; clips with
    sound play in turn, stacked, the one sounding marked, so each can be heard."""
    hold = int(hold_s * FPS)

    def side(c: Clip) -> tuple[np.ndarray, np.ndarray, int]:
        w = c.frames[0].shape[1]
        mark = "✓ " if c.ok else "✗ "
        end = _band(w, FOOT, GOOD if c.ok else BAD, [(mark + (c.headline or c.verdict), 15, True, INK)]
                    + ([(c.verdict, 12, False, INK)] if c.headline else []))
        return _band(w, FOOT, PAPER, []), end, w

    (idle_b, end_b, w), (idle_a, end_a, _) = side(before), side(after)

    def stack(top: np.ndarray, frame: np.ndarray, foot: np.ndarray) -> np.ndarray:
        return np.concatenate([top, frame, foot])

    def name(label: str, state: str = "") -> np.ndarray:
        if state == "playing":
            return _band(w, LABEL, ACCENT, [("▶ now playing · " + label, 14, True, PAPER)])
        return _band(w, LABEL, PAPER, [(label + (f"   ({state})" if state else ""), 14, True, INK if not state else MUTED)])

    if before.audio is not None or after.audio is not None:  # in turn, stacked: before, then after
        nb, na = len(before.frames), len(after.frames)
        frames = []
        for i in range(nb + hold + na + hold):
            b_on, a_on = i < nb, nb + hold <= i < nb + hold + na
            bf = before.frames[min(i, nb - 1)]
            af = after.frames[min(max(i - nb - hold, 0), na - 1)]
            top = stack(name(labels[0], "playing" if b_on else ""), bf, idle_b if b_on else end_b)
            a_done = i >= nb + hold + na
            bot = stack(name(labels[1], "playing" if a_on else ("" if a_done else "next")), af,
                        end_a if a_done else idle_a)
            frames.append(np.concatenate([top, np.full((6, w, 3), PAPER, np.uint8), bot]))
        audio = np.concatenate([before.audio if before.audio is not None else np.zeros(nb * SR // FPS, np.float32),
                                np.zeros(int(hold * SR / FPS), np.float32),
                                after.audio if after.audio is not None else np.zeros(na * SR // FPS, np.float32),
                                np.zeros(int(hold * SR / FPS), np.float32)])
    else:
        n = max(len(before.frames), len(after.frames))
        frames = []
        for i in range(n + hold):
            sides = []
            for c, label, idle, end in ((before, labels[0], idle_b, end_b), (after, labels[1], idle_a, end_a)):
                done = i >= len(c.frames)
                sides.append(stack(name(label), c.frames[min(i, len(c.frames) - 1)], end if done else idle))
            frames.append(np.concatenate([sides[0], np.full((sides[0].shape[0], 6, 3), PAPER, np.uint8), sides[1]], 1))
        audio = None
    full = frames[0].shape[1]
    tall = 12 + 20 * len(_wrap(title, full - 20, 16, True)[:2]) + 18 * len(_wrap(change, full - 20, 14)[:2] if change else [])
    head = _band(full, tall, PAPER, [(title, 16, True, INK)] + ([(change, 14, False, ACCENT)] if change else []))
    return [np.concatenate([head, f]) for f in frames], audio


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


def gripper(body: Body, task: Task, controls: dict[str, float], seed: int = 0, size: tuple[int, int] = (720, 540),
            probe: Callable[[Any], Callable[[Any], None]] | None = None, render: bool = True,
            changed: frozenset[str] = frozenset()) -> Clip:
    """Replay the pick in MuJoCo with the recorded grasp parameters under the run's ``seed``, whose
    domain-randomization draw the verdict was judged on.  System 1 records {} for its first attempt and for
    its fixed retry alike (sea.control.local_search.run_system1), so for {} both are run and the one System 1
    kept is filmed.  ``probe(model)`` makes a step hook, fresh for each attempt, called with the MuJoCo data
    every tick; the kept attempt's hook is returned on the clip.  ``render=False`` replays without filming.

    The camera looks across the line the fingers close along, so both fingers and the object between them are
    in view; the slots in ``changed`` are drawn in ACCENT (the jaw's moving parts; pads are always drawn so).
    Under the picture a strip plots how high the fingertips and the object are above where the grip closed:
    held, the two rise together; slipping, the fingertips rise and the object stays behind."""
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
    names = {m.body(i).name for i in range(m.nbody)}
    moving = {m.body(b).id for b in ("moving_jaw_so101_v1", "jaw_coupler", "jaw_carriage") if b in names}
    if render:
        _stage(m, w, h)
        if "jaw" in changed:  # the jaw's moving parts, the part the change made
            jaw = moving | {m.body(b).id for b in ("jaw_follower", "jaw_pinion", "jaw_rack") if b in names}
            for g in range(m.ngeom):
                if m.geom_bodyid[g] in jaw and m.geom_group[g] <= 2:
                    m.geom_matid[g], m.geom_rgba[g] = -1, (*np.array(ACCENT) / 255, 1.0)
    main = None
    if render:
        d0 = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d0)
        main = _View(m, w, h, _table(m, d0))
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    cam = mujoco.MjvCamera()
    cam.elevation = -12.0
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
        # across the closing line, the side nearer the arm's front: both fingers flank the object
        line = tips["moving"] - tips["fixed"]
        az = math.degrees(math.atan2(line[1], line[0]))
        across = min((az + 90, az - 90), key=lambda a: abs((a - 150.0 + 180) % 360 - 180))
        look, dist = state.get("look"), float(np.clip(0.22 + 1.0 * span, 0.24, 0.5))
        k = 1.0 if look is None else 0.25
        state["look"] = target if look is None else (1 - k) * look + k * target
        state["dist"] = dist if look is None else (1 - k) * state["dist"] + k * dist
        state["az"] = across if look is None else state["az"] + k * ((across - state["az"] + 180) % 360 - 180)
        cam.lookat[:], cam.distance, cam.azimuth = state["look"], state["dist"], state["az"]
        up = ep.sim.control_q(obj.id) - ep.q_rest0[obj.id]  # how far the object has risen, as the judge reads it
        shots[-1].append((main.shot(d, cam, opt), ep.t, float(mid[2]), float(up), still))

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
        if main is not None:
            main.close()
    i = 0
    if len(takes) == 2:  # run_system1: the retry if it passed or stopped structurally, else the better of the two
        travel = runner.sim.controls[obj.id].spec.travel
        r1 = takes[1][1].verdict
        i = 1 if r1 == "pass" or r1 in STRUCTURAL else min((0, 1), key=lambda j: objective(takes[j][1], takes[j][0], travel))
    (tr, res), raw = takes[i], shots[i]
    ok = res.verdict == "pass"
    note = (tr.note or "").replace("_", " ")
    need = 1000 * runner.sim.controls[obj.id].spec.travel
    frames = _lift_frames(raw, w, need) if raw else [np.zeros((h + STRIP, w, 3), np.uint8)]
    thing = task.group.replace("_", " ")
    return Clip(frames, ok, "held for the full second" if ok else f"{res.verdict.replace('_', ' ')}"
                + (f" ({note})" if note else ""), headline=f"Holds the {thing} for the full second" if ok else "",
                probe=hooks[i] if hooks else None, clipping_mm=depths[i].state["deepest"] if depths else 0.0)


STRIP = 118  # the height plot under a gripper film
# what a gripper failure looks like, in words (the run's mechanism labels)
GRIP_PLAIN = {"slipped_under_load": "slides out of the fingers as the arm lifts",
              "squeezed_down_out": "is squeezed out downward as the fingers close",
              "squeezed_up_out": "is squeezed out upward as the fingers close",
              "arm_stall": "is too heavy: the arm's servos stall",
              "arm_stall_before_lift": "is too heavy: the arm's servos stall before it lifts"}


def grip_headline(task: Task, mechanism: str) -> str:
    """A gripper failure in plain words: the object and what happened to it."""
    thing = task.group.replace("_", " ")
    return f"The {thing} " + GRIP_PLAIN.get(mechanism, f"is not held ({mechanism.replace('_', ' ')})")


def _lift_frames(raw: list[tuple[np.ndarray, float, float, float, bool]], w: int, need: float) -> list[np.ndarray]:
    """Each picture over a strip plotting, up to its moment, how far the fingertips have risen from where they
    closed and how far the object has risen from rest, against the ``need`` mm the task asks it be held at:
    held, the lines rise together past it; slipping, the object's line falls behind."""
    from PIL import Image, ImageDraw

    t = np.array([r[1] for r in raw])
    hand = 1000 * (np.array([r[2] for r in raw]) - min(r[2] for r in raw))
    obj = 1000 * np.array([r[3] for r in raw])
    closed = int(np.argmin(hand))
    top = max(need + 6, float(obj.max()), float(hand[int(np.argmin(hand)):].max())) + 8
    lo = min(-8.0, float(obj.min()) - 4)
    x0, x1, y0, y1 = 64, w - 16, 26, STRIP - 22
    def px(tt, v):
        return x0 + (x1 - x0) * tt / max(t[-1], 1e-6), y1 - (y1 - y0) * (np.clip(v, lo, top) - lo) / (top - lo)
    out = []
    for k, (img, now, _, _, still) in enumerate(raw):
        im = Image.new("RGB", (w, STRIP), (246, 246, 243))
        d = ImageDraw.Draw(im)
        d.line([px(0, 0.0), px(t[-1], 0.0)], fill=(222, 222, 218))
        d.text((8, px(0, 0.0)[1] - 7), "0 mm", fill=(120, 120, 120), font=_font(11))
        yn = px(0, need)[1]
        for xx in range(x0, x1, 10):  # the height the task asks for, dashed
            d.line([(xx, yn), (xx + 5, yn)], fill=(24, 140, 70), width=2)
        d.text((8, yn - 7), f"{need:.0f} mm", fill=(24, 140, 70), font=_font(11, True))
        d.text((x1 - 190, yn + 3), f"needed: {need:.0f} mm up, held 1 s", fill=(24, 140, 70), font=_font(11))
        for series, colour, width in ((hand, (120, 124, 130), 2), (obj, (30, 110, 200), 3)):
            if k:
                d.line([px(tt, v) for tt, v in zip(t[:k + 1], series[:k + 1], strict=True)], fill=colour, width=width)
        d.line([px(now, lo), px(now, top)], fill=(60, 60, 60))
        if k >= closed:  # once the grip has closed, the numbers the lines are too close to show
            slid = hand[k] - obj[k]
            d.text((x0 + 4, 2), f"object {obj[k]:.1f} mm · fingertips {hand[k]:.1f} mm · slid {slid:.1f} mm in the grip",
                   fill=BAD if slid > 1.0 else (60, 60, 60), font=_font(12, True))
        d.text((x0, STRIP - 18), "height above where the grip closed:", fill=(80, 80, 80), font=_font(11))
        d.text((x0 + 214, STRIP - 18), "— fingertips", fill=(120, 124, 130), font=_font(11, True))
        d.text((x0 + 300, STRIP - 18), "— object", fill=(30, 110, 200), font=_font(11, True))
        pic = Image.fromarray(img)
        pd = ImageDraw.Draw(pic, "RGBA")
        tag = f"t {now:4.1f} s" + (f"   ▸▸ {FAST}× fast-forward (nothing moving)" if still else "")
        pd.rectangle([0, 0, 18 + _font(13).getlength(tag), 24], fill=(*PAPER, 170))
        pd.text((8, 4), tag, fill=INK, font=_font(13))
        out.append(np.concatenate([np.asarray(pic), np.asarray(im)]))
    return out


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
    return _course(r, ctl, seed, f0)["audio"]


def _course(r: P.Rig, ctl: dict[str, float], seed: int = 0, f0: float = 120.0) -> dict[str, Any]:
    """``_utter``'s sound with what made it: the area function every 5 ms (``ts``, ``areas``, ``dist``), whether
    the posture seals, and when the closure forms (``t_hold``) and releases (``t_rel``)."""
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
    areas, lens, dist = P.tract(r, post)
    gains = P.transfer_db(np.maximum(areas, 0.08), lens)  # (frames, FREQ)
    course = {"ts": ts, "areas": areas, "dist": dist, "sealed": sealed, "t_hold": t_hold, "t_rel": t_rel,
              "t_burst": t_rel + burst, "t_open": t_rel + burst + asp}
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
        return course | {"audio": voice.astype(np.float32)}
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
    return course | {"audio": (0.9 * voice / max(np.abs(voice).max(), 1e-9)).astype(np.float32)}


AIR, CAST, TONGUE, LIP = (205, 228, 245), (232, 140, 90), (214, 120, 140), (190, 95, 95)
PX_CM = 34  # the tube is drawn at one scale, along and across


def _tube(r: P.Rig, task: Task, areas: np.ndarray, dist: np.ndarray, note: str, w: int = 620, h: int = 236) -> np.ndarray:
    """The tract as a cut through the tube, glottis (the reed) on the left and lips on the right, at one scale:
    the air channel in blue at each section's width (a circle of its area), the tongue pad filling the bore
    below it, the silicone wall around it.  A stop's target region is shaded green; a vowel's resonances are
    plotted under the tube against its target windows.  ``note`` says what the sound is doing now."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (250, 250, 247))
    d = ImageDraw.Draw(im, "RGBA")
    far = float(dist.max()) + 0.4
    x_of = lambda cm: 30 + (far - cm) * PX_CM  # noqa: E731 (distance behind the teeth -> x; lips to the right)
    yc, bore = 92, P.BORE * PX_CM
    zone = P.PLACES.get(task.data.get("place", ""), None) if task.data["kind"] == "stop" else None
    edges = np.concatenate([[far], (dist[:-1] + dist[1:]) / 2, [float(dist.min()) - 0.3]])
    for i, a_cm2 in enumerate(areas):
        x0, x1 = x_of(edges[i]), x_of(edges[i + 1])
        lip = dist[i] < 0
        r_air = math.sqrt(max(a_cm2, 0.0) / math.pi) * PX_CM
        top = yc - (bore if not lip else max(r_air, 2))
        d.rectangle([x0, top - 9, x1, top], fill=LIP if lip else CAST)  # the wall
        d.rectangle([x0, top, x1, top + 2 * r_air], fill=AIR)
        bottom = yc + bore if not lip else top + 2 * r_air
        if not lip and top + 2 * r_air < bottom:
            d.rectangle([x0, top + 2 * r_air, x1, bottom], fill=TONGUE)
        d.rectangle([x0, bottom, x1, bottom + 9], fill=LIP if lip else CAST)
    if zone:  # over the tube, translucent, so the shape shows through it
        lo, hi = zone
        x_a, x_b = x_of(min(hi, far)), x_of(max(lo, float(dist.min()) - 0.3))
        d.rectangle([x_a, yc - bore - 24, x_b, yc + bore + 10], fill=(24, 140, 70, 46), outline=GOOD, width=2)
        d.text((x_a + 4, yc - bore - 22), f"/{task.group}/ must seal here", fill=GOOD, font=_font(12, True))
    i = int(np.argmin(areas))
    xi = (x_of(edges[i]) + x_of(edges[i + 1])) / 2
    shut = areas[i] < P.A_CLOSE
    d.polygon([(xi, yc + bore + 14), (xi - 6, yc + bore + 24), (xi + 6, yc + bore + 24)], fill=BAD if not shut else GOOD)
    where = "at the lips" if dist[i] < 0 else f"{dist[i]:.1f} cm behind the teeth"
    label = f"narrowest {areas[i]:.2f} cm² {where}"
    lx = min(max(xi - _font(12).getlength(label) / 2, 4), w - 4 - _font(12).getlength(label))
    d.text((lx, yc + bore + 26), label, fill=(60, 60, 60), font=_font(12))
    d.text((x_of(far) - 4, yc + bore + 44), "◀ reed (air in)", fill=(90, 90, 90), font=_font(12))
    d.text((x_of(float(dist.min()) - 0.3) - 46, yc + bore + 44), "lips ▶", fill=(90, 90, 90), font=_font(12))
    x_t = x_of(0.0)
    d.line([(x_t, yc - bore - 30), (x_t, yc - bore - 12)], fill=(90, 90, 90), width=2)
    d.text((x_t - 16, yc - bore - 44), "teeth", fill=(90, 90, 90), font=_font(12))
    d.line([(30, h - 30), (30 + 2 * PX_CM, h - 30)], fill=(90, 90, 90), width=2)
    d.text((34, h - 26), "2 cm", fill=(90, 90, 90), font=_font(11))
    d.text((140, h - 28), note, fill=PAPER, font=_font(14, True))
    return np.asarray(im)


def _formants(task: Task, f: np.ndarray, w: int, h: int = 64) -> np.ndarray:
    """A vowel's first two resonances (dots) against its target windows (green, the target +-10%), log scale."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (250, 250, 247))
    d = ImageDraw.Draw(im)
    lo, hi = math.log(150), math.log(3200)
    def x_of(hz):
        return 40 + (w - 60) * (math.log(hz) - lo) / (hi - lo)
    d.line([(x_of(150), 34), (x_of(3200), 34)], fill=(170, 170, 170), width=1)
    for hz in (200, 500, 1000, 2000, 3000):
        d.line([(x_of(hz), 30), (x_of(hz), 38)], fill=(150, 150, 150))
        d.text((x_of(hz) - 14, 42), f"{hz} Hz", fill=(120, 120, 120), font=_font(10))
    for j, t in enumerate(task.data["target"]):
        d.rectangle([x_of(t * (1 - P.TOL)), 26, x_of(t * (1 + P.TOL)), 42], fill=(200, 232, 206))
        d.text((x_of(t) - 22, 6), f"F{j + 1} target", fill=GOOD, font=_font(11, True))
        if not np.isnan(f[j]):
            ok = abs(f[j] / t - 1) <= P.TOL
            d.ellipse([x_of(f[j]) - 6, 28, x_of(f[j]) + 6, 40], fill=GOOD if ok else BAD)
    d.text((w - 230, 6), "dots: the resonances this shape makes", fill=(90, 90, 90), font=_font(10))
    return np.asarray(im)


def phonology(body: Body, task: Task, controls: dict[str, float], seed: int = 0, size: tuple[int, int] = (860, 300)) -> Clip:
    """The body at the recorded posture, heard and seen: the tube's shape as the sound plays (for a stop, the
    glide into the closure and out, frame by frame with the sound), a vowel's resonances against its target,
    and the CAD of the body in a corner.  A posture's verdict draws on no seed; ``seed`` keeps the signature
    common to every world."""
    from PIL import Image

    from sea.design.cad import render

    r = P.rig(body)
    ctl = {k: float(np.clip(controls.get(k, (lo + hi) / 2), lo, hi)) for k, (lo, hi) in r.controls.items()}
    arr = {k: np.array([v]) for k, v in ctl.items()}
    f, clo = P.acoustics(r, arr)
    ok = float(P.PhonologyWorld._errors(task, f, clo)[0]) <= 1.0
    said = P.PhonologyWorld._said(task, f[0], clo[0])
    w, h = size
    model = C.phonology(body, ctl)
    tw = w - 640
    model.views, model.tile = model.views[:1], (tw, int(tw * 0.72))
    thumb = np.asarray(Image.open(__import__("io").BytesIO(render([model])[0])).convert("RGB"))

    def frame(tube: np.ndarray, under: np.ndarray | None) -> np.ndarray:
        canvas = np.full((h, w, 3), 250, np.uint8)
        canvas[:tube.shape[0], :tube.shape[1]] = tube
        if under is not None:
            canvas[h - under.shape[0]:, :under.shape[1]] = under
        th = min(thumb.shape[0], h)
        canvas[(h - th) // 2:(h - th) // 2 + th, w - thumb.shape[1]:] = thumb[:th]
        return canvas

    if task.data["kind"] == "vowel":
        areas, lens, dist = P.tract(r, arr)
        audio = _speak(P.transfer_db(np.maximum(areas, 0.08), lens)[0])
        img = frame(_tube(r, task, areas[0], dist[0], "voice on: a steady vowel", h=h - 64), _formants(task, f[0], 640))
        n = int(len(audio) * FPS / SR)
        headline = (f"Sounds like /{P.VOWELS[task.group][0]}/: both resonances inside their targets" if ok else
                    f"Not /{P.VOWELS[task.group][0]}/ yet: a resonance falls outside its target")
        return Clip([img] * n, ok, said, audio, headline=headline)
    c = _course(r, ctl, seed)
    frames = []
    for k in range(int(len(c["audio"]) * FPS / SR)):
        t = k / FPS
        j = min(int(round(t / (c["ts"][1] - c["ts"][0]))), len(c["ts"]) - 1)
        if not c["sealed"]:
            note = "voice on: the tube narrows, but never seals" if t > c["t_hold"] - 0.03 else "voice on: /a/"
        elif t < c["t_hold"] - 0.015:
            note = "voice on: /a/, closing"
        elif t < c["t_rel"]:
            note = "sealed: silence"
        elif t < c["t_burst"]:
            note = "release: the burst you hear as /" + task.group + "/"
        elif t < c["t_open"]:
            note = "air rushing out"
        else:
            note = "voice on: /a/"
        frames.append(frame(_tube(r, task, c["areas"][j], c["dist"][j], note, h=h), None))
    # before and after the sound, a still of the shape System 1 found (the closest the glide comes to it)
    areas, _, dist = P.tract(r, arr)
    still = [frame(_tube(r, task, areas[0], dist[0], "the shape System 1 found", h=h), None)] * int(0.6 * FPS)
    gap = np.zeros(int(len(still) * SR / FPS), np.float32)
    frames, audio = still + frames + still, np.concatenate([gap, c["audio"], gap])
    where = P.PLACES[task.data["place"]]
    shut = float(clo[0][0]) < P.A_CLOSE
    in_place = where[0] <= float(clo[0][1]) <= where[1]
    headline = (f"Seals the tube in the right place: a clean /{task.group}/" if ok else
                "Never seals the tube, so no /" + task.group + "/: the voice runs straight through" if not shut else
                "Seals the tube in the wrong place: that is not /" + task.group + "/" if not in_place else
                "Seals, but the airway behind the seal is blocked too")
    return Clip(frames, ok, said, audio, headline=headline)


# ---------------------------------------------------------------- the mobile explorer: MuJoCo at each obstacle
GROUND = {"concrete": (0.70, 0.70, 0.68), "tile": (0.86, 0.85, 0.81), "wood": (0.66, 0.52, 0.38),
          "carpet": (0.55, 0.50, 0.60), "gravel": (0.60, 0.58, 0.54), "grass": (0.45, 0.62, 0.36),
          "sand": (0.86, 0.78, 0.55)}
RUBBER = (0.13, 0.13, 0.14, 1.0)
PHYSICAL = ("step", "trench", "ramp", "gap")  # what MuJoCo drives (D-063); the rest is the world's calculation


@dataclass
class Leg:
    feature: dict
    ep: Any  # sea.sim.mobile_model.Episode, traced


def legs(body: Body, task: Task, controls: dict[str, float]) -> tuple[bool, str, str, list[Leg]]:
    """The route at its nominal dimensions, as the world judges it, and the MuJoCo episode of every obstacle the
    robot reached: (success, mechanism, detail, legs).  Each episode is the judge's own, rerun with its trace."""
    s = M._spec(body)
    route, secs = task.data["route"], float(task.data["secs"])
    ctl = {**M.PROBE, **controls}
    ok, _, mech, detail = M._attempt(s, route, secs, ctl, M._Calm())
    hit = re.search(r"stopped at feature (\d+)/", detail)
    upto = int(hit.group(1)) if hit else len(route)
    out = []
    for f in route[:upto]:
        k, mu = f["k"], M._mu(s, f["s"])
        if k in ("step", "trench"):
            out.append(Leg(f, M._drive(s, k, f, mu * s.edge, ctl["approach"], trace=True)))
        elif k == "ramp":
            out.append(Leg(f, M._drive(s, k, f, mu, ctl["speed"], trace=True)))
        elif k == "gap" and f["w"] > s.width:
            out.append(Leg(f, M._drive(s, k, f | {"off": f["off"] - ctl["line"]}, mu, ctl["speed"], trace=True)))
    if not out or (not ok and route[upto - 1]["k"] not in PHYSICAL):  # it stopped where MuJoCo does not drive
        last = route[upto - 1]
        out.append(Leg(last, M._drive(s, "floor", {"k": "floor", "s": last["s"]}, M._mu(s, last["s"]), 0.0,
                                      trace=True)))
    return ok, mech, detail, out


OBSTACLE = (0.80, 0.45, 0.27)  # what the robot must get over or through, against the ground's colour
HALF_Y = 0.3  # the film's terrain is cut 0.3 m either side of the robot's line, so a side camera sees its profile


def _dress_mobile(m, surf: str) -> None:
    """Colours for the film's copy of an episode's model: visual fields only, plus the terrain's width, cut to
    HALF_Y either side of the robot's line so a camera beside it sees the obstacle's profile (a step's height,
    a trench's width and depth, a ramp's slope) rather than its top.  The robot drives within that band, so its
    contacts are the episode's.  A plane's size is drawn only (it collides everywhere), so the floor is drawn as
    a stage inside the studio lights."""
    for g in range(m.ngeom):
        name, body = m.geom(g).name, m.geom_bodyid[g]
        if m.geom_type[g] == 0:  # the floor plane: a stage inside the lights, the backdrop beyond it
            m.geom_size[g][:2] = (3.0, 3.0)
        if body == 0:
            m.geom_rgba[g] = (*(GROUND.get(surf, (0.7, 0.7, 0.7)) if m.geom_type[g] == 0 else OBSTACLE), 1.0)
            if m.geom_type[g] == 6 and m.geom_size[g][1] > HALF_Y and m.geom_size[g][2] < 0.14:  # not a door jamb
                m.geom_size[g][1] = HALF_Y
            elif m.geom_type[g] == 3:  # a step's rounded nosing, along y
                m.geom_size[g][1] = HALF_Y
        elif name == "tub":
            m.geom_rgba[g] = PLA
        elif m.geom_type[g] == 5:  # cylinders: wheels, rollers
            m.geom_rgba[g] = RUBBER
        else:
            m.geom_rgba[g] = (0.72, 0.72, 0.70, 1.0)
    m.light_diffuse[:], m.light_specular[:], m.light_castshadow[:] = 0.12, 0.0, False
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:], m.vis.headlight.specular[:] = 0.36, 0.32, 0.05
    m.vis.headlight.active = 1
    m.vis.quality.shadowsize, m.vis.quality.offsamples = 8192, 4
    m.vis.rgba.fog[:] = (*(CYC[1] / 255), 1.0)  # the floor fades into the backdrop instead of into the dark
    m.vis.map.fogstart, m.vis.map.fogend = 1.2 / m.stat.extent, 3.0 / m.stat.extent


def _project(scn, p: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """Pixel (x, y) of world point ``p`` in the picture the scene's camera drew at ``w`` x ``h``."""
    c = scn.camera[0]
    fwd, up = np.array(c.forward), np.array(c.up)
    rel = np.asarray(p, float) - np.array(c.pos)
    z = rel @ fwd
    x, y = c.frustum_near * (rel @ np.cross(fwd, up)) / z, c.frustum_near * (rel @ up) / z
    half = (c.frustum_top - c.frustum_bottom) / 2 * w / h  # MuJoCo leaves the width to the viewport's aspect
    return w * (x - c.frustum_center + half) / (2 * half), h * (c.frustum_top - y) / (c.frustum_top - c.frustum_bottom)


def _arrow(d, a: tuple[float, float], b: tuple[float, float], text: str, colour=(150, 60, 20)) -> None:
    """A dimension line from ``a`` to ``b`` with end ticks and its label beside the middle."""
    d.line([a, b], fill=colour, width=2)
    (ax, ay), (bx, by) = a, b
    n = np.array([by - ay, ax - bx]) / max(math.hypot(bx - ax, by - ay), 1e-6) * 6
    for x, y in (a, b):
        d.line([(x - n[0], y - n[1]), (x + n[0], y + n[1])], fill=colour, width=2)
    font = _font(15, True)
    tw = font.getlength(text)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    tx, ty = (mx + 10, my - 9) if abs(by - ay) > abs(bx - ax) else (mx - tw / 2, my - 26)
    d.rectangle([tx - 4, ty - 2, tx + tw + 4, ty + 19], fill=(255, 255, 255, 215))
    d.text((tx, ty), text, fill=colour, font=font)


def _dimension(img: np.ndarray, scn, f: dict) -> np.ndarray:
    """The obstacle's size drawn on it, at the near face the side camera sees."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    h, w = img.shape[:2]
    def pt(x, y, z):
        return _project(scn, np.array([x, y, z]), w, h)
    y = -HALF_Y
    if f["k"] == "step":
        _arrow(d, pt(-0.012, y, 0.0), pt(-0.012, y, f["h"] / 100), f"{f['h']:.1f} cm step")
    elif f["k"] == "trench":
        _arrow(d, pt(0.0, y, 0.012), pt(f["w"] / 100, y, 0.012), f"{f['w']:.1f} cm wide, {f['depth']:.1f} cm deep")
    elif f["k"] == "ramp":
        x, yy = pt(0.12, y, 0.0)
        d.rectangle([x - 4, yy - 30, x + 110, yy - 10], fill=(255, 255, 255, 215))
        d.text((x, yy - 30), f"{f['deg']:.1f}° ramp", fill=(150, 60, 20), font=_font(15, True))
    elif f["k"] == "gap":
        off, half = f["off"] / 100, f["w"] / 200
        _arrow(d, pt(0.0, off - half, 0.3), pt(0.0, off + half, 0.3), f"opening {f['w']:.1f} cm")
    return np.asarray(im)


THING = {"step": "the {h:.1f} cm step", "trench": "the {w:.1f} cm trench", "ramp": "the {deg:.0f}° ramp",
         "gap": "the {w:.1f} cm doorway", "sand": "the sand", "dark": "the dark hallway", "low": "the low sofa",
         "floor": "the {s}"}
# what a mobile failure looks like, in words: {thing} is where it stopped (the run's mechanism labels)
MOBILE_PLAIN = {
    "step_higher_than_climbable": "Can't climb {thing}: its {loco} slip at the edge",
    "belly_hits_step_edge": "Its belly grounds on the edge of {thing}",
    "bounced_off_step": "Bounces back off {thing}",
    "tipped_over": "Tips over on {thing}",
    "stalled_at_step": "Its motors stall on {thing} at the speed it chose",
    "motor_torque_short_on_step": "Its motors are too weak for {thing}",
    "gap_longer_than_support": "Drops into {thing}: it is longer than the robot can bridge",
    "chassis_wider_than_gap": "Too wide to fit through {thing}",
    "clipped_gap_edge": "Clips the frame of {thing}",
    "gap_margin_below_drift": "Too little room in {thing} for how far it drifts off line",
    "sensing_range_short": "Can't see far enough in {thing}: it crawls and runs out of time",
    "collided_in_dark": "Hits obstacles in {thing}: its sensor sees them too late",
    "sinks_in_sand": "Sinks in {thing}",
    "too_tall_for_clearance": "Too tall to fit under {thing}",
    "slips_on_ramp": "Slides back down {thing}",
    "motor_torque_short_on_ramp": "Its motors are too weak for {thing}",
    "motor_torque_short": "Its motors are too weak to move on {thing}",
    "battery_exhausted": "Its battery runs flat before the end of the route",
    "drained_at_chosen_speed": "Its battery runs flat at the speed it chose",
    "too_slow_for_time_limit": "Too slow to finish the route in time",
    "too_slow_at_chosen_speed": "Too slow at the speed it chose to finish in time",
    "stalled_at_low_duty": "Stalls at the low speed it chose"}
LOCO = {"wheels4": "wheels", "mecanum4": "mecanum wheels", "rocker_bogie6": "wheels", "tracks": "tracks"}


def mobile_headline(body: Body, ok: bool, mech: str, detail: str, where: dict) -> str:
    """A mobile attempt in plain words: what it got through, or where and how it stopped."""
    if ok:
        got = re.search(r"reached in (\d+) s of (\d+)", detail)
        return f"Makes it: the end of the route in {got[1]} s (limit {got[2]} s)" if got else "Makes it to the end"
    thing = THING.get(where["k"], "the route").format(**where)
    loco = LOCO.get(body.parts["locomotion"].type, body.parts["locomotion"].type.replace("_", " "))
    return MOBILE_PLAIN.get(mech, "Stops at {thing} (" + mech.replace("_", " ") + ")").format(thing=thing, loco=loco)


def mobile(body: Body, task: Task, controls: dict[str, float], seed: int = 0, size: tuple[int, int] = (640, 420)) -> Clip:
    """Each obstacle on the route replayed in MuJoCo at the recorded controls, as the judge drove it (D-063),
    up to where the robot stopped.  A failure MuJoCo does not drive (sand, an overhang, the dark, the battery)
    ends on a still of the robot on that ground with the world's numbers.  ``seed`` is unused: the route
    is replayed at its nominal dimensions.  A step, trench or ramp is seen from the side, level with it, its
    size drawn on its near face; a doorway from above, its opening drawn across it."""
    import mujoco

    ok, mech, detail, route_legs = legs(body, task, controls)
    w, h = size
    frames: list[np.ndarray] = []
    deepest = 0.0
    for leg in route_legs:
        m, ep, f = leg.ep.model, leg.ep, leg.feature
        _dress_mobile(m, f["s"])
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, SS * w), max(m.vis.global_.offheight, SS * h)
        d = mujoco.MjData(m)
        props = []
        if f["k"] == "gap" and m.ngeom and "obstacle" not in {m.geom(g).name for g in range(m.ngeom)}:
            # a doorway the judge refused without driving it (the robot is wider): its jambs, drawn only, where
            # the episode would have put them (sea.sim.mobile_model.terrain_xml)
            for side in (1, -1):
                props.append((np.array([0.02, 0.25, 0.15]), np.array([0.02, f["off"] / 100 + side * (f["w"] / 200 + 0.25), 0.15]),
                              (*OBSTACLE, 1.0)))
        sunk = re.search(r"sinks it ([\d.]+) cm", detail) if f["k"] == "sand" and not ok else None
        if sunk:  # sand is the world's calculation (Winkler), not MuJoCo's: drawn at the depth it computes
            props.append((np.array([1.5, HALF_Y, float(sunk[1]) / 200]), np.array([0.0, 0.0, float(sunk[1]) / 200]),
                          (*GROUND["sand"], 1.0)))
        view = _View(m, w, h, props)
        cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
        top = f["k"] == "gap"
        cam.azimuth, cam.elevation, cam.distance = (90.0, -72.0, 0.85) if top else (90.0, -9.0, 0.75)
        centre = {"trench": f.get("w", 0) / 200, "ramp": 0.2}.get(f["k"], 0.0)
        here = THING.get(f["k"], f["k"]).format(**f)
        idx = range(0, len(ep.t), 2) if len(ep.t) else [None]
        x_cam = None
        for k in idx:
            if k is not None:
                d.qpos[:] = ep.qpos[k]
            mujoco.mj_forward(m, d)
            deepest = max(deepest, -min((c.dist for c in d.contact[:d.ncon]), default=0.0))
            x = 0.5 * (float(d.qpos[0]) + centre) if f["k"] in PHYSICAL else float(d.qpos[0])
            x_cam = x if x_cam is None else 0.85 * x_cam + 0.15 * x
            z = 0.0 if top else 0.6 * max(0.04, float(d.qpos[2]))
            cam.lookat[:] = (x_cam, 0.0, z)
            view.colour.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            img = view.shot(d, cam, opt)
            if f["k"] in PHYSICAL:  # a doorway refused before driving still has its size drawn, over its jambs
                img = _dimension(img, view.colour.scene, f)
            calc = (f"calculated, not simulated: it sinks {float(sunk[1]):.1f} cm into the sand" if sunk else
                    "" if f["k"] in PHYSICAL else "calculated, not simulated: " + here)
            frames.append(caption(img, top=f"t {ep.t[k]:.1f} s · MuJoCo" if k is not None and not calc else calc))
        view.close()
        if f["k"] in PHYSICAL:
            tag = f"gets past {here}" if ep.ok else f"stopped at {here}"
            frames += [caption(frames[-1], bottom=tag, colour=GOOD if ep.ok else BAD)] * int(0.8 * FPS)
    verdict = detail.split(": ", 1)[1] if (not ok and ": " in detail) else detail
    where = route_legs[-1].feature if route_legs else {"k": "floor", "s": "floor"}
    if not ok:  # where it stopped, which may be a feature MuJoCo does not drive
        hit = re.search(r"stopped at feature (\d+)/", detail)
        where = task.data["route"][int(hit.group(1)) - 1] if hit else where
    return Clip(frames, ok, verdict if ok else f"{mech.replace('_', ' ')}: {verdict}",
                headline=mobile_headline(body, ok, mech, detail, where), clipping_mm=1000 * deepest)


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
