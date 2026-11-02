"""Films of a body at a task (sea.worlds.film, scripts/film_bodies.py): replays reproduce the run's records."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from sea.evolve.core import Body, Edit, apply
from sea.harness import fingerprint
from sea.worlds import film as F
from sea.worlds.gripper import GripperWorld
from sea.worlds.mobile import MobileWorld
from sea.worlds.phonology import FREQ, PhonologyWorld

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import film_bodies  # noqa: E402

MOBILE, PHON = MobileWorld(), PhonologyWorld()


def _task(world, tid: str):
    return next(t for t in world.tasks() if t.id == tid)


def _body(world, *edits: Edit) -> Body:
    return apply(world.stock(), list(edits), {p.name: p for p in world.catalogue()}, "t")


TRACKS = (Edit(slot="locomotion", type="tracks", params={"track_length": 28.0, "idler_height": 7.0}),
          Edit(slot="chassis", params={"width": 11.0, "length": 28.0, "ride_height": 2.0}),
          Edit(slot="battery", params={"capacity": 14.4}), Edit(slot="sensor", type="ultrasonic"))


def test_mobile_film_stops_where_the_simulator_fails_and_says_why():
    porch = _task(MOBILE, "porch_train0")
    before = F.mobile(MOBILE.stock(), porch, {"approach": 0.5, "speed": 0.6})
    after = F.mobile(_body(MOBILE, *TRACKS), porch, {"approach": 0.5, "speed": 0.6})
    assert not before.ok and before.verdict.startswith("step higher than climbable")
    assert after.ok and after.verdict.startswith("reached")
    assert len(before.frames) >= 12 and before.frames[0].shape == (300, 480, 3)


def test_settling_rests_a_flat_bottom_flat():
    bottom = np.array([[0.0, 5.0], [100.0, 5.0]])
    th, lift = F._settle(bottom, np.array([50.0, 40.0]), np.array([0.0, 1000.0]), np.array([0.0, 0.0]), 200.0)
    assert th == pytest.approx(0.0) and lift == pytest.approx(-5.0)


def test_talking_robot_film_judges_the_recorded_posture_as_the_world_does():
    ao = _task(PHON, "ao_train0")
    stock = F.phonology(PHON.stock(), ao, {"jaw": 0.0, "tongue_height": 0.95, "tongue_place": 9.7296})
    assert not stock.ok and "F1 +24%" in stock.verdict  # the run recorded F 704/1067 Hz, F1 +24%, F2 +27%
    evolved = _body(PHON, Edit(slot="tongue", type="tip_tongue", params={"rail_start": 3.25, "stroke": 1.3}),
                    Edit(slot="lips", type="round_lips", params={"max_protrusion": 1.6}))
    ok = F.phonology(evolved, ao, {"lip_radius": 0.5954, "protrusion": 0.6655, "tip_height": 0.1297, "tip_place": 1.0108,
                                   "tongue_height": 0.9575, "tongue_place": 9.5602})
    assert ok.ok and ok.audio is not None and abs(len(ok.audio) / F.SR - len(ok.frames) / F.FPS) < 0.05


def test_the_voice_puts_its_energy_where_the_tract_resonates():
    gain = -40 + 40 * np.exp(-(((FREQ - 720.0) / 60.0) ** 2))  # one resonance at 720 Hz
    a = F._speak(gain, seconds=0.5, f0=120.0)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(a), 1 / F.SR)
    assert 600 < freqs[np.argmax(spec)] < 840


def test_silent_films_play_together_and_sounding_ones_in_turn():
    still = [np.zeros((40, 60, 3), np.uint8)]
    b, a = F.Clip(still * 10, False, "no"), F.Clip(still * 4, True, "yes")
    frames, audio = F.pair(b, a, "t", ("before", "after"), hold_s=0.4)
    assert audio is None and len(frames) == 10 + 10 and frames[0].shape == (40 + 34, 126, 3)
    b.audio, a.audio = np.zeros(10 * F.SR // F.FPS, np.float32), np.zeros(4 * F.SR // F.FPS, np.float32)
    frames, audio = F.pair(b, a, "t", ("before", "after"), hold_s=0.4)
    assert len(frames) == 10 + 10 + 4 + 10 and abs(len(audio) / F.SR - len(frames) / F.FPS) < 0.05


def test_gripper_film_replays_the_recorded_failure():
    """gripper_s2v3_s0, iteration 3: the stock hinge on the tuna can with System 1's best controls.  It stalled
    on soft contacts; on rigid ones (D-054) the jaw only grazes the 85 mm can, too briefly to count (D-053)."""
    world = GripperWorld()
    tuna = _task(world, "reg_tuna_fish_can")
    ctl = {"grasp.side": 0.004802, "grasp.close": -0.167291, "grasp.close_s": 0.832028, "grasp.lift": 0.059022,
           "grasp.hold_s": 1.429247, "grasp.speed": 0.025509}
    clip = F.gripper(world.stock(), tuna, ctl, size=(160, 120))
    assert not clip.ok and clip.verdict.startswith("no actuation") and len(clip.frames) > 50


class _Stop(Exception):
    pass


def _tick(world, task, seed: int, n: int = 30) -> np.ndarray:
    """The state ``n`` ticks into a replay; the perception draw moves the approach from about tick 15."""
    seen: list[np.ndarray] = []

    def probe(model):
        def hook(data):
            seen.append(data.qpos.copy())
            if len(seen) == n:
                raise _Stop
        return hook

    with pytest.raises(_Stop):
        F.gripper(world.stock(), task, {}, seed, probe=probe, render=False)
    return seen[-1]


def test_gripper_film_replays_under_the_runs_seed():
    """The replay draws its randomization from the run's seed, as System 1's attempt did: the same seed starts
    the same way, another seed does not."""
    world = GripperWorld()
    task = _task(world, "reg_flat_screwdriver")
    a, b, c = (_tick(world, task, s) for s in (0, 2, 2))
    assert np.allclose(b, c) and not np.allclose(a, b)


def test_gripper_film_replays_system1s_fixed_retry_when_the_record_says_defaults():
    """gripper_s2v3_s2, iteration 10: held the mustard bottle on the retry, recorded as {} after 1 attempt."""
    world = GripperWorld()
    slide = _body(world, Edit(slot="jaw", type="prismatic", params={"drive_mm_per_rad": 60.0}),
                  Edit(slot="pads", type="pads", params={"reach_mm": 3.5, "pad_angle_deg": 8.798, "groove_mm": 6.0,
                                                         "lip_mm": 0.0, "pad_tpu_mm": 2.0, "width_mm": 18.0}))
    clip = F.gripper(slide, _task(world, "reg_mustard_bottle"), {}, 2, size=(160, 120))
    assert clip.ok


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="films need ffmpeg")
def test_film_bodies_films_each_kept_change(tmp_path):
    stock = MOBILE.stock().model_dump()
    kept = _body(MOBILE, *TRACKS).model_dump() | {"id": "b001"}
    run = tmp_path / "mobile_x"
    run.mkdir()
    fail = {"task": "porch_train0", "success": False, "mechanism": "step_higher_than_climbable",
            "controls": {"approach": 0.5, "speed": 0.6}}
    trial = {"k": 1, "move": "swap", "gained": ["porch"], "lost": [], "result": "KEPT: gained porch",
             "outcomes": {"porch_train0": {"success": True, "controls": {"approach": 0.5, "speed": 0.6}}},
             "proposal": {"hypothesis": "tracks climb what wheels cannot"}}
    recs = [{"iteration": 0, "body": stock | {"id": "stock"}, "held": [], "failing": {"porch_train0": fail},
             "trials": [trial], "accepted": "b001"},
            {"iteration": 1, "body": kept, "held": ["porch"], "stop": "all groups held"}]
    (run / "iterations.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (run / "summary.json").write_text(json.dumps({"world": "mobile", "train_groups": 10, "final_body": kept,
                                                  "final_held_train": ["porch"], "seed": 0}))
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "summary.json").write_text((run / "summary.json").read_text())
    (run / "harness.json").write_text(json.dumps({"components": fingerprint()}))
    assert film_bodies.main([str(run), str(tmp_path / "old"), "--out", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out" / "old").exists()  # no harness.json: what judged it is unknown
    films = json.loads((tmp_path / "out" / "mobile_x" / "films.json").read_text())["films"]
    assert [(f["group"], f["before"]["ok"], f["after"]["ok"]) for f in films] == [("porch", False, True)]
    assert (tmp_path / "out" / "mobile_x" / films[0]["file"]).stat().st_size > 1000
