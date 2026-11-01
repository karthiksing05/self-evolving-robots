"""Procedural appliance panels (spec 7.4, Appendix D).

``generate(split, n, seed)`` is a pure function of its arguments and ``configs/panels.yaml``.
Panel frame: origin at the board's top-surface centre, +z = outward normal, +x = away from the
base, +y = lateral.  The panel yaws to face the base and tilts its normal toward the base by
``tilt_rad``, so tilt 0 means straight-down presses.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from sea.contracts import CONTROL_CLASSES, ControlSpec, PanelSpec, Pose, Rules, Vec3, recess_bin, tilt_bin
from sea.control.frames import ROOT, mat_to_quat_wxyz
from sea.sim.controls import build_geometry

CONFIG = ROOT / "configs" / "panels.yaml"
SPLIT_CODE = {"train": 1, "val": 2, "test": 3, "compose": 4, "reference": 5}
TEST_WIDEN = 0.25


@lru_cache(maxsize=4)
def load_config(path: str = str(CONFIG)) -> dict:
    return yaml.safe_load(Path(path).read_text())


def widen(rng_lohi: list[float], split: str) -> tuple[float, float]:
    lo, hi = float(rng_lohi[0]), float(rng_lohi[1])
    if split == "test":
        hi = lo + (1.0 + TEST_WIDEN) * (hi - lo)
    return lo, hi


def combo(control: ControlSpec, panel_tilt: float) -> tuple[str, str, str, str]:
    """(class, handle, recess bin, tilt bin): the unit of compositional novelty."""
    return (control.cls, control.handle, recess_bin(control.recess_depth_mm), tilt_bin(panel_tilt))


def matches(c: tuple[str, str, str, str], rule: dict) -> bool:
    keys = ("cls", "handle", "recess", "tilt")
    return all(rule[k] == "*" or rule[k] == v for k, v in zip(keys, c, strict=True))


def held_out(c: tuple[str, str, str, str], cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return any(matches(c, r) for r in cfg["compose_holdouts"])


def success_check(cls: str, handle: str, n_detents: int) -> str:
    if cls in ("door", "drawer"):
        return "~latched($target) & moved_at_least($target, travel)"
    if cls == "toggle" or (cls in ("push_button", "recessed_button") and n_detents > 0):
        return "moved_at_least($target, travel) & detent_crossed($target, 1)"
    return "moved_at_least($target, travel)"


def _u(rng: np.random.Generator, lohi: tuple[float, float]) -> float:
    return float(rng.uniform(lohi[0], lohi[1]))


def sample_control(rng: np.random.Generator, cls: str, cid: str, split: str, cfg: dict,
                   handle: str | None = None, recess_mm: float | None = None) -> ControlSpec:
    c = cfg["classes"][cls]
    size = tuple(round(_u(rng, widen(r, split)), 2) for r in c["size"])
    stiffness = _u(rng, widen(c["stiffness"], split))
    if "travel_mm" in c:
        travel = _u(rng, widen(c["travel_mm"], split)) / 1000.0
    else:
        travel = math.radians(_u(rng, widen(c["travel_deg"], split)))
    friction = _u(rng, widen(c["friction"], split))
    lo, hi = c["detents"]
    n_det = int(rng.integers(lo, hi + 1))
    if cls == "slider":
        travel = min(travel, (size[0] - 10) / 1000.0)
    if cls == "knob" and split == "test":
        travel = min(travel, math.radians(320))
    peak = _u(rng, widen(c["detent_peak"], split)) if n_det and "detent_peak" in c else 0.0
    if cls in ("push_button", "recessed_button"):
        detents = [round(travel * float(rng.uniform(0.45, 0.7)), 6)] if n_det else []
    elif cls == "toggle":
        detents = [round(travel / 2, 6)]
    else:
        detents = [round(travel * (i + 0.5) / n_det, 6) for i in range(n_det)]
    if handle is None:
        handle = str(rng.choice(c["handles"])) if "handles" in c else "none"
    latch = None
    if cls in ("door", "drawer"):
        latch = round(_u(rng, widen(c["latch"], split)), 3)
        if handle == "push_latch":
            travel = math.radians(15.0) if cls == "door" else 0.015
    if recess_mm is None:
        recess_mm = _u(rng, widen(c["recess_mm"], split)) if "recess_mm" in c else 0.0
    return ControlSpec(
        id=cid, cls=cls, pose_panel=Pose(pos=Vec3(x=0.0, y=0.0, z=0.0)), size_mm=size,
        stiffness=round(stiffness, 6), travel=round(travel, 6), detents=detents, detent_peak=round(peak, 6),
        friction=round(friction, 6), latch_threshold=latch, handle=handle, recess_depth_mm=round(recess_mm, 2),
        success=success_check(cls, handle, len(detents)),
    )


def layout(controls: list[ControlSpec], cfg: dict) -> tuple[list[ControlSpec], tuple[float, float]]:
    """Shelf-pack controls into rows along the panel's lateral (y) axis; returns placed controls
    and the board's half extents."""
    gap = cfg["gap_m"]
    row_w = cfg["row_width_m"]
    rows: list[list[tuple[ControlSpec, tuple[float, float]]]] = [[]]
    width = 0.0
    for c in controls:
        fx, fy = build_geometry(c).footprint
        if rows[-1] and width + fy + gap > row_w:
            rows.append([])
            width = 0.0
        rows[-1].append((c, (fx, fy)))
        width += fy + gap
    depths = [max(f[0] for _, f in r) for r in rows]
    total_x = sum(depths) + gap * (len(rows) - 1)
    placed: list[ControlSpec] = []
    x = -total_x / 2
    max_w = 0.0
    for r, depth in zip(rows, depths, strict=True):
        w = sum(f[1] for _, f in r) + gap * (len(r) - 1)
        max_w = max(max_w, w)
        y = -w / 2
        for c, (_fx, fy) in r:
            pos = Vec3(x=round(x + depth / 2, 5), y=round(y + fy / 2, 5), z=0.0)
            placed.append(c.model_copy(update={"pose_panel": Pose(pos=pos)}))
            y += fy + gap
        x += depth + gap
    return placed, (total_x / 2 + 0.02, max_w / 2 + 0.02)


def panel_pose(x: float, y: float, z: float, tilt: float) -> Pose:
    yaw = math.atan2(y, x)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ct, st = math.cos(-tilt), math.sin(-tilt)
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    q = mat_to_quat_wxyz(rz @ ry)
    return Pose(pos=Vec3(x=x, y=y, z=z), quat_wxyz=tuple(float(v) for v in q))


def _rules(cfg: dict) -> Rules:
    return Rules(force_cap_by_class={k: float(v) for k, v in cfg["force_caps"].items()})


def _regular_ok(cls: str, handle: str, recess_mm: float, tilt: float, cfg: dict) -> bool:
    return not held_out((cls, handle, recess_bin(recess_mm), tilt_bin(tilt)), cfg)


def _make_panel(pid: str, seed: int, split: str, classes: list[str], tilt: float, rng: np.random.Generator,
                cfg: dict, holdout: dict | None = None) -> PanelSpec:
    controls: list[ControlSpec] = []
    for i, cls in enumerate(classes):
        cid = f"c{i}"
        if holdout is not None and i == 0:
            recess = 18.0 + float(rng.uniform(0, 7)) if holdout["recess"] == ">10" else None
            handle = holdout["handle"] if holdout["handle"] != "*" else None
            controls.append(sample_control(rng, holdout["cls"], cid, split, cfg, handle=handle, recess_mm=recess))
            continue
        for _ in range(50):
            c = sample_control(rng, cls, cid, split, cfg, handle="none" if cls == "slot" else None)
            if _regular_ok(c.cls, c.handle, c.recess_depth_mm, tilt, cfg):
                break
        controls.append(c)
    placed, _ = layout(controls, cfg)
    pl = cfg["placement"]
    pose = panel_pose(_u(rng, tuple(pl["x"])), _u(rng, tuple(pl["y"])), _u(rng, tuple(pl["z"])), tilt)
    return PanelSpec(id=pid, seed=seed, split=split, pose_world=pose, tilt_rad=round(tilt, 6), controls=placed,
                     rules=_rules(cfg))


def generate(split: str, n: int, seed: int, cfg: dict | None = None) -> list[PanelSpec]:
    """``n`` panels for ``split``; every class appears equally often (>= 20 times once n >= 30)."""
    cfg = cfg or load_config()
    if split == "reference":
        return [reference_panel(cfg)]
    master = np.random.default_rng([seed, SPLIT_CODE[split]])
    lo, hi = cfg["controls_per_panel"]
    counts = [int(master.integers(lo, hi + 1)) for _ in range(n)]
    pool: list[str] = []
    while len(pool) < sum(counts) + len(CONTROL_CLASSES):
        block = list(CONTROL_CLASSES)
        master.shuffle(block)
        pool.extend(block)
    holdouts = cfg["compose_holdouts"]
    panels: list[PanelSpec] = []
    for i, k in enumerate(counts):
        pseed = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(pseed)
        tilt_lo, tilt_hi = widen(cfg["tilt_deg"], split)
        holdout = holdouts[i % len(holdouts)] if split == "compose" else None
        if holdout is not None and holdout["tilt"] == ">45":
            tilt = math.radians(float(rng.uniform(46.0, tilt_hi)))
        else:
            tilt = math.radians(float(rng.uniform(tilt_lo, tilt_hi)))
        classes: list[str] = []
        steep = tilt_bin(tilt) == ">45"
        j = 0
        while len(classes) < k:
            cls = pool[j]
            if steep and cls == "knob":  # knob x steep face is a compose-only combination
                j += 1
                continue
            classes.append(pool.pop(j))
            j = 0
        if holdout is not None:
            classes[0] = holdout["cls"]
        panels.append(_make_panel(f"{split}_p{i:03d}", pseed, split, classes, tilt, rng, cfg, holdout))
    return panels


def reference_panel(cfg: dict | None = None) -> PanelSpec:
    """Fixed panel (seed 0): one control of each class, flat, centred in the stock reach."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(0)
    controls = []
    for i, cls in enumerate(CONTROL_CLASSES):
        handle = {"door": "pull", "drawer": "tab", "slot": "none"}.get(cls)
        recess = 8.0 if cls == "recessed_button" else None
        controls.append(sample_control(rng, cls, f"c{i}", "train", cfg, handle=handle, recess_mm=recess))
    placed, _ = layout(controls, cfg)
    return PanelSpec(id="reference", seed=0, split="reference", pose_world=panel_pose(0.22, 0.0, -0.03, 0.0),
                     tilt_rad=0.0, controls=placed, rules=_rules(cfg))


def board_half_extents(panel: PanelSpec) -> tuple[float, float]:
    cfg = load_config()
    _, half = layout([c.model_copy(update={"pose_panel": Pose(pos=Vec3(x=0, y=0, z=0))}) for c in panel.controls], cfg)
    xs = [abs(c.pose_panel.pos.x) + build_geometry(c).footprint[0] / 2 for c in panel.controls]
    ys = [abs(c.pose_panel.pos.y) + build_geometry(c).footprint[1] / 2 for c in panel.controls]
    return max(max(xs, default=0.0) + 0.015, half[0]), max(max(ys, default=0.0) + 0.015, half[1])
