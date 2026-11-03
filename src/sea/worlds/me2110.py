"""The ME 2110 world (D-073): one robot for every task of the "Pirates of the Caribbean" brief, in one 40 s match.

The brief (``ME2110_Fall_2026_Final_Project_v1.3.pdf``, an example brief, not a class) asks a robot that starts
inside a 12 x 24 x 18 in box, behind the start line, to score six tasks in 40 s:
- activate and fully leave the dock (11 points);
- sort gold coins and jewels from the turning fortress's top onto their mounds (175);
- run up a flag on the magnetic pole, at least 24 in up (40 if highest);
- deliver cannonballs into our colour's port, never launched upward, at most 12 in unsupported (36);
- topple the mermaid and merman (24);
- bring Davy Jones' heart from the chest back to the dock (61).

It may use only the kit's actuators (2 gearmotors, 2 pistons on one tank, 2 mousetraps, 5 rubber bands) and
gravity, may carry no magnets, and the parts bought must cost under $120 (prints at $0.05 per gram).  At 40 s it
must be static, its pistons retracted.  The match is simulated in MuJoCo (``sea.sim.me2110_model``).

A body is eight parts: the ``frame``, its ``drive``, and a mechanism for each task (``mermaid``, ``cannon``,
``flag``, ``heart``, ``treasure``), with a ``release`` for the stored-energy parts.  A task is achieved when it
scores at least half its maximum; the ``match`` group asks for half of all the points in one 40 s run (D-074).  System 1 places the
robot in the dock, chooses where it parks and how fast it drives, and times each mechanism (``local_search``);
for the match it starts from what it found for each task.
"""

from __future__ import annotations

import math
import re
import time
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from sea.evolve.core import Body, BomLine, Outcome, Param, Part, PartType, Task
from sea.evolve.search import local_search
from sea.sim import me2110_model as MM

BOX = (12.0, 24.0, 18.0)  # in: the go/no-go box, length x width x height
KIT = {"motors": 2, "pistons": 2, "mousetraps": 2, "bands": 5}  # the brief: two mousetraps, five #64 bands
BUDGET = 120.0
PRINT_USD = 0.05  # per gram
THRESH = {k: v / 2 for k, v in MM.MAX_POINTS.items()}
MATCH_POINTS = round(sum(MM.MAX_POINTS.values()) / 2)  # D-074: the match is judged on points
TASKS = ("activate", "treasure", "flag", "cannon", "mermaids", "heart")
# assumed prices (USD) for what the brief makes teams buy; kit parts cost nothing
PRICE = {"plywood_in2": 0.0156, "wheel_pair": 6.0, "ball_caster": 3.0, "hinge": 1.0, "m3_screw": 0.05,
         "alu_tube": 2.5, "dowel": 1.0, "string": 0.5, "steel_hook": 0.5, "drawer_pin": 0.3,
         "ir_range_long": 15.0, "acrylic_sheet": 8.0}  # a bought sensor counts toward the budget (the kit's own IR sensor reaches 10 in)
FLAG_BOOMS = ("piston_boom", "sensed_boom")
PLY_G_IN2, POST_G_IN3 = 2.46, 9.8  # 1/4 in plywood, grams per square inch; posts, grams per cubic inch
ELEC_G, MOTOR_G, TANK_G, VALVE_G, TRAP_G = 300.0, 200.0, 350.0, 80.0, 30.0


def _p(lo: float, hi: float, unit: str, doc: str) -> Param:
    return Param(lo=lo, hi=hi, unit=unit, doc=doc)


BOOM_PARAMS = {"boom_len": _p(6.0, 24.0, "in", "pivot to tip"),
               "fold_deg": _p(100.0, 175.0, "deg", "the folded boom's angle from forward-level (180 lies back along "
                              "the deck)"),
               "pivot_z": _p(4.0, 17.0, "in", "the pivot's height (on a post at the deck's front edge)"),
               "raise_deg": _p(10.0, 85.0, "deg", "the boom's angle above level when raised")}

CATALOGUE = [
    PartType(name="ply_frame", slot="frame", stock=True, doc="a plywood frame: a base plate 1 in above the floor, a "
             "deck plate on four posts; the mechanisms mount on the deck's front edge",
             provides="the footprint and the deck's height",
             params={"length": _p(6, 24, "in", "along the drive direction"), "width": _p(6, 24, "in", "across it"),
                     "deck_z": _p(4, 16, "in", "the deck's top above the floor")}),
    PartType(name="wheels2", slot="drive", stock=True, doc="the kit's two gearmotors (12 V, 1.2 N m stall, 160 rpm "
             "assumed) on two wheels at the rear quarter, a ball caster up front; it drives straight out on the "
             "encoder's count, stops, and can drive back", provides="drives to a park position and back; uses both "
             "gearmotors", params={"wheel_r": _p(1.0, 3.0, "in", "wheel radius: speed per duty, and the push at the "
                                                    "floor (torque / radius)")}),
    PartType(name="mousetrap_car", slot="drive", doc="two wheels on an axle a mousetrap turns through a string wound "
             "on it (the trap's own 1.8 in arm, lengthened by a lever)", provides="one run forward, no return; uses "
             "one mousetrap; needs a cam_timer to release it",
             params={"wheel_r": _p(1.0, 3.0, "in", "wheel radius"),
                     "axle_r": _p(0.15, 0.6, "in", "the string's axle: smaller pulls harder, for less distance"),
                     "lever": _p(2.0, 8.0, "in", "the lever's length: the string pulled is twice it")}),
    PartType(name="static_base", slot="drive", doc="no drive: four rubber feet; the robot stays in the dock",
             provides="nothing moves the robot: it cannot leave the dock"),
    PartType(name="push_bar", slot="mermaid", stock=True, doc="a printed bar across the frame's front",
             provides="pushes whatever the drive drives it into",
             params={"bar_z": _p(2.0, 8.0, "in", "the bar's height above the floor"),
                     "bar_w": _p(4.0, 24.0, "in", "the bar's width")}),
    PartType(name="trap_arms", slot="mermaid", doc="two printed arms, each swung 180 degrees by a mousetrap from "
             "folded back along the frame's side, out and forward; latched until released",
             provides="strikes things beside the robot's front corners; uses two mousetraps and a release",
             params={"arm_len": _p(3.0, 16.0, "in", "arm length from its pivot at a front corner"),
                     "arm_z": _p(2.0, 10.0, "in", "the arms' height above the floor")}),
    PartType(name="no_cannon", slot="cannon", stock=True, doc="no cannon mechanism", provides="nothing"),
    PartType(name="gate_chute", slot="cannon", doc="a printed chute holding the four cannonballs on an 8 degree "
             "slope to an exit ahead of the frame (2.5 in left of centre); a gate at the exit rises on a piston, "
             "timed from the IR sensor seeing our port's tab", provides="drops what it holds from its exit on a sensed cue; uses one piston",
             params={"exit_z": _p(4.0, 14.0, "in", "the exit's floor height"),
                     "reach": _p(0.0, 14.0, "in", "how far ahead of the frame's front the exit is")}),
    PartType(name="no_flag", slot="flag", stock=True, doc="no flag mechanism", provides="nothing"),
    PartType(name="piston_boom", slot="flag", doc="a wooden boom folded back over the deck, swung up and forward "
             "by a piston through a lever; the flag is clipped at its tip, upright when raised", provides="places "
             "what is clipped at its tip at a set time after the robot stops; uses one piston, which must retract "
             "by 40 s", params=BOOM_PARAMS),
    PartType(name="sensed_boom", slot="flag", doc="the piston boom, fired by a bought long-range IR sensor on a 14 in "
             "post at its pivot that looks ahead over the fortress wall and sees when a pole face is square to it",
             provides="places what is clipped at its tip a set delay after a sensed cue; uses one piston and a $15 "
             "sensor", params=BOOM_PARAMS),
    PartType(name="band_mast", slot="flag", doc="nested aluminium tubes on the deck's front edge, leaning forward, "
             "extended by rubber bands once released; the flag is clipped at the top", provides="lifts what is "
             "clipped at its top and stays extended; uses its bands and a release",
             params={"stages": _p(2, 4, "", "tubes, nested (rounded to whole)"),
                     "stage_len": _p(4.0, 14.0, "in", "each tube's length; they overlap 1.5 in"),
                     "lean_deg": _p(0.0, 60.0, "deg", "the mast's lean forward from vertical"),
                     "bands": _p(1, 5, "", "rubber bands pulling the stages out (rounded to whole)")}),
    PartType(name="no_heart", slot="heart", stock=True, doc="no heart mechanism", provides="nothing"),
    PartType(name="hook_boom", slot="heart", doc="a boom standing up at the deck's front edge with a hook at its tip "
             "(a bar down and a toe back toward the robot); a piston pushes it down and pulls it back up, and a "
             "tray on the deck catches what the hook brings", provides="reaches into what is ahead and below; uses "
             "one piston",
             params={"boom_len": _p(3.0, 16.0, "in", "pivot to tip"),
                     "hook_depth": _p(0.5, 6.0, "in", "the hook's drop below the tip"),
                     "drop_deg": _p(0.0, 60.0, "deg", "the boom's angle below level when pushed down")}),
    PartType(name="no_treasure", slot="treasure", stock=True, doc="no treasure mechanism", provides="nothing"),
    PartType(name="peel_chute", slot="treasure", doc="a fold-out acrylic-lined trough: a fixed inner section from the frame's "
             "centre up to its front edge, an outer section folded up and latched that falls forward onto the "
             "inner's slope, with a blade at its tip that rides 1 in above the trough's floor at 40 degrees in plan; "
             "a sieve of two rails near the low end drops jewels to a funnel on the right; coins cross it to a deflector "
             "on the left; two side troughs fold down from the deck's sides onto stops at a set angle",
             provides="peels what the fortress turns into its blade and sorts it by size; uses a release",
             params={"reach": _p(2.0, 16.0, "in", "the outer section's length ahead of the frame's front"),
                     "junction_z": _p(2.0, 14.0, "in", "the trough's low end, at the frame's centre"),
                     "slope_deg": _p(10.0, 50.0, "deg", "the trough's slope"),
                     "side_len": _p(2.0, 16.0, "in", "each side trough's length"),
                     "side_deg": _p(15.0, 60.0, "deg", "the side troughs' slope below level when unfolded"),
                     "sieve_gap": _p(0.5, 3.0, "in", "the gap between the sieve's two rails (a jewel is 1.5 in across, "
                                     "a coin 2.5 in)"),
                     "width": _p(2.8, 6.0, "in", "the troughs' inner width (a coin is 2.5 in across)")}),
    PartType(name="drive_trip", slot="release", stock=True, doc="latches tripped by a lever when the drive "
             "stops", provides="releases every latched part 0.2 s after the robot parks; no actuator"),
    PartType(name="cam_timer", slot="release", doc="a cam shaft on a gearmotor tripping each latch at its own time",
             provides="releases each latched part at a time System 1 chooses; uses one gearmotor"),
]
CAT = {(p.slot, p.name): p for p in CATALOGUE}
LATCHED = {("mermaid", "trap_arms"), ("flag", "band_mast"), ("treasure", "peel_chute"), ("drive", "mousetrap_car")}


# ---------------------------------------------------------------- a body, in numbers
def _robot(body: Body) -> MM.Robot:
    P = body.parts
    fr = P["frame"].params
    L, W, H = fr["length"], fr["width"], fr["deck_z"]
    dr = P["drive"]
    kw: dict[str, Any] = {"drive": {"wheels2": "wheels2", "mousetrap_car": "mousetrap_car"}.get(dr.type, "static")}
    if dr.type in ("wheels2", "mousetrap_car"):
        kw["wheel_r"] = dr.params["wheel_r"]
    if dr.type == "mousetrap_car":
        kw.update(axle_r=dr.params["axle_r"], lever=dr.params["lever"])
    for slot in ("mermaid", "cannon", "flag", "heart", "treasure"):
        kw[slot] = (P[slot].type, dict(P[slot].params))
    kw["release"] = P["release"].type
    return MM.Robot(L=L, W=W, H=H, frame_kg=_frame_g(body) / 1000, **kw)


def _uses(body: Body) -> dict[str, int]:
    P = body.parts
    u = {"motors": 0, "pistons": 0, "mousetraps": 0, "bands": 0}
    u["motors"] += 2 if P["drive"].type == "wheels2" else 0
    u["motors"] += 1 if P["release"].type == "cam_timer" else 0
    u["mousetraps"] += 1 if P["drive"].type == "mousetrap_car" else 0
    u["mousetraps"] += 2 if P["mermaid"].type == "trap_arms" else 0
    u["pistons"] += sum(1 for s, t in (("cannon", ("gate_chute",)), ("flag", FLAG_BOOMS), ("heart", ("hook_boom",)))
                        if P[s].type in t)
    if P["flag"].type == "band_mast":
        u["bands"] += int(round(P["flag"].params["bands"]))
    return u


def _frame_g(body: Body) -> float:
    """The frame and everything carried rigidly on it: plates, posts, electronics, kit actuators, the tank."""
    fr = body.parts["frame"].params
    L, W, H = fr["length"], fr["width"], fr["deck_z"]
    u = _uses(body)
    g = 2 * L * W * PLY_G_IN2 + 4 * 0.5625 * max(H - 1.25, 0.5) * POST_G_IN3 + ELEC_G + u["motors"] * MOTOR_G
    g += (TANK_G + u["pistons"] * VALVE_G) if u["pistons"] else 0.0
    return g + u["mousetraps"] * TRAP_G + 50.0


def _print_g(body: Body) -> dict[str, float]:
    """Printed PLA per slot (grams), from each printed part's volume at 30 % infill."""
    P, out = body.parts, {}
    if P["mermaid"].type == "push_bar":
        out["mermaid"] = 12.4 * P["mermaid"].params["bar_w"] / 8 + 4.0
    else:
        out["mermaid"] = 2 * (4.0 * P["mermaid"].params["arm_len"] + 10.0)
    if P["cannon"].type == "gate_chute":
        out["cannon"] = 45.0 + 3.0 * P["cannon"].params["reach"]
    if P["heart"].type == "hook_boom":
        out["heart"] = 20.0
    if P["treasure"].type == "peel_chute":
        tp = P["treasure"].params
        L = P["frame"].params["length"]
        out["treasure"] = 9.0 * (L / 2 / math.cos(math.radians(tp["slope_deg"])) + tp["reach"] + 2 * tp["side_len"]) + 25.0
    return out


def _boom(rb: MM.Robot, slot: str) -> dict[str, float]:
    """A piston boom's swing and pins (inches, frame coordinates): retracted and extended angles, the pivot, the rod's
    pin along the boom (b) and the cylinder's pin (c from the pivot at phi, at cx, cz)."""
    kind, p = rb.flag if slot == "flag" else rb.heart
    if slot == "flag":
        th0, th1, pz = math.radians(p["fold_deg"]), math.radians(p["raise_deg"]), p["pivot_z"]
    else:
        th0, th1, pz = math.radians(MM.HOOK_FOLD_DEG), -math.radians(p["drop_deg"]), rb.H
    b, c, phi = MM.boom_pins(th0, th1)
    px = rb.L / 2 - 0.5
    return {"th0": th0, "th1": th1, "px": px, "pz": pz, "b": b, "c": c, "phi": phi,
            "cx": px + c * math.cos(phi), "cz": pz + c * math.sin(phi)}


def _part(label: str, P: dict[str, Any]) -> str:
    """'the treasure's treasure_part3' -> 'the treasure (peel_chute)'; a named part keeps its name."""
    mech, part = label[len("the "):].split("'s ", 1)
    slot = "mermaid" if mech == "mermaid" else mech
    kind = f" ({P[slot].type})" if slot in P else ""
    named = "" if re.fullmatch(r"\w+_part\d+", part) else f"'s {part.replace('_', ' ')}"
    return f"the {mech}{kind}{named}"


def envelope(body: Body) -> tuple[float, float, float]:
    """The folded robot's extent before the start (length along x, width along y, height), inches."""
    rb = _robot(body)
    L, W, H = rb.L, rb.W, rb.H
    front = L / 2
    x0, x1, half_y, top = -L / 2, front, W / 2, H
    if rb.drive in ("wheels2", "mousetrap_car"):
        half_y, top = max(half_y, W / 2 + 1.35), max(top, 2 * rb.wheel_r)
    kind, p = rb.mermaid
    if kind == "push_bar":
        x1, half_y, top = max(x1, front + 0.75), max(half_y, p["bar_w"] / 2), max(top, p["bar_z"] + 0.75)
    else:
        x0, half_y, top = min(x0, front - p["arm_len"]), max(half_y, W / 2 + 0.6), max(top, p["arm_z"] + 0.4)
    kind, p = rb.cannon
    if kind == "gate_chute":
        x1, top = max(x1, front + p["reach"] + 0.3), max(top, p["exit_z"] + 2.2)
    kind, p = rb.flag
    if kind in FLAG_BOOMS:
        fold = math.radians(p["fold_deg"])
        bm = _boom(rb, "flag")
        x0 = min(x0, front - 0.5 + p["boom_len"] * math.cos(fold) - 0.3, bm["cx"] - 0.5)
        top = max(top, p["pivot_z"] + p["boom_len"] * math.sin(fold) + MM.FLAG_H, bm["cz"] + 0.5,
                  MM.FLAG_SENSOR_Z + 0.5 if kind == "sensed_boom" else 0.0)
    elif kind == "band_mast":
        lean = math.radians(p["lean_deg"])
        top = max(top, H + p["stage_len"] * math.cos(lean) + 0.6)
        x1 = max(x1, front - 1.0 + p["stage_len"] * math.sin(lean) + 0.6)
    kind, p = rb.heart
    if kind == "hook_boom":
        top = max(top, H + p["boom_len"] + 0.3, _boom(rb, "heart")["cz"] + 0.5)
    kind, p = rb.treasure
    if kind == "peel_chute":
        lay = MM.peel_layout(rb, p)
        fold, run_out = math.radians(MM.OUTER_FOLD_DEG), p["reach"] / math.cos(lay["slope"])
        x1 = max(x1, front + run_out * math.cos(fold) + 1.6)
        side_up = math.radians(MM.SIDE_FOLD_DEG)
        top = max(top, lay["z_front"] + run_out * math.sin(fold) + 1.6,
                  max(lay["z_left"], lay["z_right"]) + p["side_len"] * math.sin(side_up) + 0.6)
        half_y = max(half_y, W / 2 + MM.SIDE_OUT + p["side_len"] * math.cos(side_up) + 0.1)
    return x1 - x0, 2 * half_y, top


def fits_box(env: tuple[float, float, float]) -> bool:
    a, b = sorted(env[:2])
    return a <= BOX[0] + 1e-6 and b <= BOX[1] + 1e-6 and env[2] <= BOX[2] + 1e-6


# ---------------------------------------------------------------- tasks
VARIANTS = {"train": [MM.Variant(barrier=7.0, rpm=3.0, mu=0.5, phase=0.4, others=(6.0, 8.0, 8.75))],
            "test": [MM.Variant(barrier=6.0, rpm=2.0, mu=0.45, phase=1.3, others=(7.0, 8.0, 8.75)),
                     MM.Variant(barrier=8.75, rpm=4.5, mu=0.55, phase=2.2, others=(6.0, 7.0, 8.0))]}
WHAT = {"activate": "activate and fully leave the dock (1 + 10 points)",
        "treasure": "sort the 10 gold coins (2.5 x 0.5 in) and 5 jewels (1.5 in) from the fortress top's rim onto "
                    "their mounds: 10 / 15 points each on the mound's top, 4 / 9 in its 1 in border (175 max)",
        "flag": "affix the flag (5 x 3 in, magnets inside) to the 4 in square pole, its top at least 24 in up; 40 "
                "points as the highest flag",
        "cannon": "deliver the 4 cannonballs (40 mm) into our colour's port: 9 points each in its receptacle; no "
                  "part of the robot may be inside the fortress's cylinder at 40 s",
        "mermaids": "topple the mermaid and merman (12 points each, 45 degrees or more; 5 if only moved off the "
                    "4 in mark)",
        "heart": "the heart out of the chest: 39 on the robot, 61 back behind the start line (the chest itself: "
                 "11 on the robot, 32 back)",
        "match": f"score at least {MATCH_POINTS:g} of the {sum(MM.MAX_POINTS.values())} points the six tasks offer, in "
                 "one 40 s match, breaking no rule"}
HORIZON = {"activate": 8.0, "mermaids": 12.0, "flag": 30.0, "heart": 30.0, "cannon": 40.0, "treasure": 40.0,
           "match": 40.0}


def _variant_facts(v: MM.Variant) -> str:
    return (f"our port's barrier {v.barrier:g} in, fortress turning {v.rpm:g} rpm (a turn in {60 / v.rpm:.0f} s), "
            f"floor friction {v.mu:g}")


# ---------------------------------------------------------------- System 1's controls
def _space(body: Body, task: str) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    P = body.parts
    env = envelope(body)
    half = env[1] / 2
    sp: dict[str, tuple[float, float]] = {"y0": (-MM.DOCK / 2 + half, MM.DOCK / 2 - half) if half < MM.DOCK / 2
                                          else (0.0, 0.0)}
    st: dict[str, float] = {"y0": 0.0}
    if P["drive"].type == "wheels2":
        sp.update(park_x=(0.0, 34.0), duty=(0.25, 1.0))
        st.update(park_x=10.0, duty=0.6)
    cam = P["release"].type == "cam_timer"
    want = set(TASKS) if task == "match" else {task}
    if cam and P["drive"].type == "mousetrap_car":
        sp["t_drive"], st["t_drive"] = (0.1, 5.0), 0.5
    if cam and P["mermaid"].type == "trap_arms" and want & {"mermaids"}:
        sp["t_mermaid"], st["t_mermaid"] = (0.5, 30.0), 4.0
    if P["cannon"].type == "gate_chute" and want & {"cannon"}:
        sp.update(gate_delay=(0.0, 2.0), gate_open=(0.2, 1.5))
        st.update(gate_delay=0.3, gate_open=0.7)
    if P["flag"].type == "piston_boom" and want & {"flag"}:
        sp.update(flag_at=(0.2, 20.0), flag_hold=(0.3, 15.0))
        st.update(flag_at=1.0, flag_hold=4.0)
    if P["flag"].type == "sensed_boom" and want & {"flag"}:
        sp.update(flag_delay=(0.0, 4.0), flag_hold=(0.3, 15.0))
        st.update(flag_delay=0.2, flag_hold=4.0)
    if cam and P["flag"].type == "band_mast" and want & {"flag"}:
        sp["t_flag"], st["t_flag"] = (0.5, 30.0), 5.0
    if P["heart"].type == "hook_boom" and want & {"heart"}:
        sp.update(heart_at=(0.2, 15.0), heart_hold=(0.3, 6.0))
        st.update(heart_at=1.0, heart_hold=1.5)
    if P["drive"].type == "wheels2" and want & {"heart"}:
        sp["return_at"], st["return_at"] = (0.5, 34.0), 5.0
    if cam and P["treasure"].type == "peel_chute" and want & {"treasure"}:
        sp["t_treasure"], st["t_treasure"] = (0.5, 20.0), 4.0
    sp = {k: v for k, v in sp.items() if v[1] > v[0]}
    return sp, {k: st[k] for k in sp}


def _label(task: str, body: Body, r: MM.Result) -> str:
    P = body.parts
    slot = {"flag": "flag", "cannon": "cannon", "heart": "heart", "treasure": "treasure"}.get(task)
    if slot and P[slot].type == f"no_{slot}":
        return f"no_{task}_mechanism"
    for law in r.laws:
        for key, lab in (("still moving", "robot_moving_at_40s"), ("pistons not retracted", "piston_extended_at_40s"),
                         ("quadrant", "left_quadrant"), ("unsupported", "projectile_too_far"),
                         ("rose", "projectile_launched_upward"), ("diverged", "simulation_diverged")):
            if key in law:
                return lab
    n = r.notes.get(task, "")
    if task == "activate":
        return "never_left_dock"
    if task == "mermaids":
        return "merfolk_moved_not_toppled" if "moved" in n and "standing" in n else "merfolk_not_reached"
    if task == "cannon":
        if "void" in n:
            return "robot_inside_fortress_voids_cannon"
        if not any("IR sensor" in e for _, e in r.events):
            return "port_never_seen"
        return "balls_missed_port"
    if task == "flag":
        if any("pulled off" in e for _, e in r.events):
            return "flag_pulled_off_pole"
        if body.parts["flag"].type == "sensed_boom" and not any("pole face square" in e for _, e in r.events):
            return "pole_face_never_sensed"
        if "below the 24 in" in n:
            return "flag_below_24in"
        return "flag_never_met_pole_square"
    if task == "heart":
        if "lid shut" in n:
            return "chest_lid_shut"
        return "heart_not_lifted" if "still in the chest" in n else "heart_not_held"
    if task == "treasure":
        if "10 still on the fortress" in n and "5 still on the fortress" in n:
            return "figures_missed_blade"
        return "figures_not_sorted_onto_mounds"
    return "failed"


def _attempt(body: Body, rb: MM.Robot, v: MM.Variant, task: str, ctl: dict[str, float], seed: int) -> Outcome:
    t0 = time.perf_counter()
    r = MM.Match(rb, v, ctl, seed=seed, horizon=HORIZON[task]).run()
    if task == "match":
        total, top = sum(r.points[k] for k in TASKS), sum(MM.MAX_POINTS.values())
        ok = total >= MATCH_POINTS and not r.laws
        score = (0.8 * total / top + 0.2 * sum(r.progress.get(k, 0.0) for k in TASKS) / len(TASKS)) * \
            (0.5 if r.laws else 1.0) + (0.2 if ok else 0.0)
        detail = f"{total:.0f} of {top} points (needs {MATCH_POINTS}): " + "; ".join(
            f"{k} {r.points[k]:.0f}: {r.notes[k]}" for k in TASKS)
        mech = "" if ok else ("rule_broken" if r.laws else "points_short; scored nothing at:" + ",".join(
            k for k in TASKS if r.points[k] <= 0))
    else:
        pts = r.points[task]
        ok = pts >= THRESH[task] and not r.laws
        score = (0.6 * min(1.0, pts / MM.MAX_POINTS[task]) + 0.4 * r.progress.get(task, 0.0)) * \
            (0.5 if r.laws else 1.0) + (0.2 if ok else 0.0)
        detail = f"{pts:.0f} of {MM.MAX_POINTS[task]} points: {r.notes[task]}"
        mech = "" if ok else _label(task, body, r)
    if r.laws:
        detail += "; broke: " + "; ".join(r.laws[:3])
    return Outcome(task="", success=ok, score=score, mechanism=mech, detail=detail, sim_s=time.perf_counter() - t0)


class ME2110World:
    name = "me2110"
    description = ("The ME 2110 'Pirates of the Caribbean' robot: one machine, started inside a 12 x 24 x 18 in box "
                   "behind the start line, that in a 40 s match leaves the dock, sorts coins and jewels off a turning "
                   "fortress onto their mounds, runs a flag up a magnetic pole, drops cannonballs into its colour's "
                   "port, topples two merfolk and brings a heart back from a chest, using only the kit's two "
                   "gearmotors, two pistons, two mousetraps and five rubber bands, for under $120.  System 1 places "
                   "it in the dock, chooses where it parks and how fast it drives, and times each mechanism.")

    def __init__(self) -> None:
        self._best: dict[tuple[str, str], dict[str, float]] = {}

    def tasks(self) -> list[Task]:
        out = []
        for g in (*TASKS, "match"):
            for split, vs in VARIANTS.items():
                for k, v in enumerate(vs):
                    out.append(Task(id=f"{g}_{split}{k}", split=split, group=g,  # type: ignore[arg-type]
                                    facts=f"{WHAT[g]}; {_variant_facts(v)}",
                                    data={"task": g, "variant": k, "seed": k + (100 if split == "test" else 0)}))
        return out

    def catalogue(self) -> list[PartType]:
        return list(CATALOGUE)

    def stock(self) -> Body:
        return Body(parts={
            "frame": Part(type="ply_frame", params={"length": 11.0, "width": 12.0, "deck_z": 8.0}),
            "drive": Part(type="wheels2", params={"wheel_r": 1.5}),
            "mermaid": Part(type="push_bar", params={"bar_z": 4.0, "bar_w": 12.0}),
            "cannon": Part(type="no_cannon"), "flag": Part(type="no_flag"), "heart": Part(type="no_heart"),
            "treasure": Part(type="no_treasure"), "release": Part(type="drive_trip")})

    def solve(self, body: Body, task: Task, seed: int = 0) -> Outcome:
        g = task.data["task"]
        v = VARIANTS[task.split][task.data["variant"]]
        slot = {"flag": "flag", "cannon": "cannon", "heart": "heart", "treasure": "treasure"}.get(g)
        if slot and body.parts[slot].type == f"no_{slot}":  # nothing on the robot can score it: no simulation needed
            return Outcome(task=task.id, success=False, mechanism=f"no_{g}_mechanism",
                           detail=f"0 of {MM.MAX_POINTS[g]} points: the body has no {slot} mechanism")
        rb = _robot(body)
        key = zlib.crc32(task.id.encode())
        space, start = _space(body, g)

        def run(ctl: dict[str, float], k: int) -> Outcome:
            return _attempt(body, rb, v, g, ctl, int(np.random.default_rng([key, k & 0xFFFFFFFF]).integers(1 << 30)))

        if g == "match":  # start from what System 1 found for each task on this body
            starts = [start] + [{**start, **{k: c[k] for k in c if k in start}}
                                for (bk, _), c in self._best.items() if bk == body.key()]
            best_start = max(starts, key=lambda s: run(s, 0).score) if len(starts) > 1 else start
            out = local_search(run, space, best_start, m=10, confirm=1, seed=seed + int(task.data["seed"]))
        else:
            out = local_search(run, space, start, m=12, confirm=1, seed=seed + int(task.data["seed"]))
            if task.split == "train":
                self._best[(body.key(), g)] = dict(out.controls)
        return out.model_copy(update={"task": task.id})

    def brief(self, body: Body, tasks: Sequence[Task]) -> str:
        rb = _robot(body)
        env = envelope(body)
        u = _uses(body)
        mass = (_frame_g(body) + sum(_print_g(body).values())) / 1000
        rows = [f"folded {env[0]:.1f} x {env[1]:.1f} x {env[2]:.1f} in (length x width x height); the box is "
                f"{BOX[0]:g} x {BOX[1]:g} x {BOX[2]:g} in, either way round, height {BOX[2]:g} in",
                f"kit actuators used: {u['motors']} of 2 gearmotors, {u['pistons']} of 2 pistons, {u['mousetraps']} of "
                f"2 mousetraps, {u['bands']} of 5 rubber bands",
                f"mass about {mass:.2f} kg; parts bought ${self._cost(body):.2f} of ${BUDGET:g}",
                f"frame {rb.L:g} in long, {rb.W:g} in wide, deck at {rb.H:g} in; its front is at park_x + {rb.L / 2:g} in"]
        if rb.drive == "wheels2":
            v0 = MM.MOTOR_W0 * rb.wheel_r
            push = 2 * MM.MOTOR_TS / (rb.wheel_r * MM.IN)
            rows.append(f"drive: {v0:.1f} in/s at full duty with no load, {push:.0f} N of push at stall")
        elif rb.drive == "mousetrap_car":
            run_in = 2 * rb.lever * rb.wheel_r / rb.axle_r
            pull = MM.TRAP_T0 / (rb.lever * MM.IN) * rb.axle_r / rb.wheel_r
            rows.append(f"drive: the string's {2 * rb.lever:.1f} in turns the wheels {run_in:.0f} in; it pushes the "
                        f"robot with at most {pull:.2f} N (rolling it takes about {0.02 * mass * 9.81:.2f} N)")
        kind, p = rb.mermaid
        if kind == "push_bar":
            rows.append(f"push bar: {p['bar_w']:g} in wide at {p['bar_z']:g} in")
        else:
            rows.append(f"trap arms: tips sweep a {p['arm_len']:g} in radius about each front corner (y = "
                        f"±{rb.W / 2:g} in) at {p['arm_z']:g} in, with {MM.TRAP_T0 / (p['arm_len'] * MM.IN):.1f} N at "
                        "the tip at the start of the swing")
        kind, p = rb.cannon
        if kind == "gate_chute":
            ln = MM.chute_len(p["exit_z"], p["reach"], rb)
            rows.append(f"cannon chute: exit floor at {p['exit_z']:g} in, {p['reach']:g} in ahead of the front, "
                        f"{ln:.1f} in long ({'holds 4 balls' if ln >= 6.7 else 'too short for 4 balls'})")
        kind, p = rb.flag
        if kind in FLAG_BOOMS:
            th, bm = math.radians(p["raise_deg"]), _boom(rb, "flag")
            rows.append(f"flag boom raised: tip {p['boom_len'] * math.cos(th):.1f} in ahead of its pivot at "
                        f"{p['pivot_z'] + p['boom_len'] * math.sin(th):.1f} in (the flag's top 1.5 in above); the "
                        f"piston's pin {bm['b']:.2f} in along it, its cylinder pinned {bm['c']:.2f} in from the pivot "
                        f"at {math.degrees(bm['phi']):.0f} deg; the rod meets the boom at "
                        f"{MM.boom_transmission(bm['th0'], bm['th1']):.0f} deg at worst")
        elif kind == "band_mast":
            n, sl, lean = int(round(p["stages"])), p["stage_len"], math.radians(p["lean_deg"])
            full = sl + (n - 1) * (sl - 1.5)
            rows.append(f"mast extended: {full:.1f} in long from the deck's front edge, tip {full * math.sin(lean):.1f} "
                        f"in ahead and {rb.H + full * math.cos(lean):.1f} in up; each band stretched "
                        f"{(sl - 1.5 + 0.5 * MM.BAND_L0) / MM.BAND_L0:.1f} times its length at the start")
        kind, p = rb.heart
        if kind == "hook_boom":
            dr = math.radians(p["drop_deg"])
            rows.append(f"hook boom down: tip {p['boom_len'] * math.cos(dr):.1f} in ahead of the front, at "
                        f"{rb.H - p['boom_len'] * math.sin(dr):.1f} in; the hook reaches {p['hook_depth']:g} in below it")
        kind, p = rb.treasure
        if kind == "peel_chute":
            lay = MM.peel_layout(rb, p)
            rows.append(f"peel chute deployed: trough tip {p['reach']:g} in ahead of the front, its floor at "
                        f"{lay['z_tip']:.1f} in, the blade's lower edge at {lay['blade_bottom']:.1f} in (the fortress "
                        f"top is at {MM.HF:g} in, a coin {MM.COIN_T:g} in thick); side trough exits {lay['side_exit_y']:.1f} "
                        f"in either side of centre, coins (left) at {lay['coin_exit_z']:.1f} in, jewels (right) at "
                        f"{lay['jewel_exit_z']:.1f} in (the mounds' tops are {MM.MOUND_H:g} in "
                        f"up, centred {MM.MOUNDS['gold'][1]:g} in either side, {MM.MOUNDS['gold'][0]:g} in out), at "
                        f"x = park_x; troughs slope {p['slope_deg']:g} deg and sides {p['side_deg']:g} deg (the troughs are "
                        f"acrylic-lined: a piece slides only on a slope steeper than "
                        f"{math.degrees(math.atan(MM.SLICK_MU)):.0f} deg)")
        rows.append("arena: start line x = 0; merfolk at (20, ±10) in, bases 3.5 / 3.7 in; chest 19.9-23.1 in out, "
                    "4.7 in wide, lid propped open (a 0.4 in bump shuts it); fortress edge 27.6 in out, its centre "
                    f"{MM.XF:g} in; pole faces 2 in from its centre; mounds at ({MM.MOUNDS['gold'][0]:g}, ±15) in")
        return "\n".join(rows)

    def principles(self) -> list[str]:
        return [
            "Every part must fit the 12 x 24 x 18 in box folded; anything larger must unfold after the start, by an "
            "actuator, a released spring or gravity.",
            "Two parts cannot be in one place: every mechanism shares the base with the others, stowed, deployed and "
            "on its way between, and fastened to the frame it may not swing into it.",
            "The kit's actuators are the budget that matters: each mechanism that moves needs one, and a gearmotor, "
            "piston, mousetrap or band used by one task is not available to another.",
            "A piston pushes pressure times area through a lever: the lever's arm trades force for swing; at 40 s "
            "it must be retracted, so what it placed must stay where it was put without it.",
            "A mousetrap gives one 180 degree swing of falling torque; a rubber band pulls in proportion to its "
            "stretch, at most about three times its length; both must be latched at the start and released.",
            "A figure on a turning table moves with it; a fixed blade in its path pushes it off along the blade, and "
            "a trough below catches it if its floor is lower than the table and slopes steeply enough to slide.",
            "Pieces of different sizes separate through a slot wider than the small one and narrower than the large.",
            "A standing figure topples when a push above its centre of mass tips it past its base's edge; a push "
            "below it slides it.",
            "A magnet holds only when its face lies flat on the iron; on a turning square pole a face is square to "
            "a fixed direction once every quarter turn.",
            "A gravity-fed release must be timed to a moving target: what falls takes sqrt(2 h / g) to drop h.",
            "Anything left reaching into the fortress at 40 s voids the cannonballs; the robot must also stand still "
            "with its pistons retracted.",
        ]

    def bom(self, body: Body) -> list[BomLine]:
        P = body.parts
        fr = P["frame"].params
        out = [BomLine(item="plywood_in2", qty=round(2 * fr["length"] * fr["width"] + 4 * 0.75 * fr["deck_z"], 1),
                       slot="frame"), BomLine(item="m3_screw", qty=16, slot="frame")]
        out += [BomLine(item="print:PLA", qty=round(g, 1), slot=s) for s, g in _print_g(body).items()]
        if P["drive"].type in ("wheels2", "mousetrap_car"):
            out += [BomLine(item="wheel_pair", qty=1, slot="drive"), BomLine(item="ball_caster", qty=1, slot="drive")]
        if P["drive"].type == "mousetrap_car":
            out.append(BomLine(item="string", qty=1, slot="drive"))
        if P["mermaid"].type == "trap_arms":
            out.append(BomLine(item="drawer_pin", qty=2, slot="mermaid"))
        if P["cannon"].type == "gate_chute":
            out.append(BomLine(item="hinge", qty=1, slot="cannon"))
        if P["flag"].type in FLAG_BOOMS:
            out += [BomLine(item="dowel", qty=1, slot="flag"), BomLine(item="hinge", qty=2, slot="flag")]
        if P["flag"].type == "sensed_boom":
            out.append(BomLine(item="ir_range_long", qty=1, slot="flag"))
        if P["flag"].type == "band_mast":
            out.append(BomLine(item="alu_tube", qty=int(round(P["flag"].params["stages"])), slot="flag"))
        if P["heart"].type == "hook_boom":
            out += [BomLine(item="dowel", qty=1, slot="heart"), BomLine(item="steel_hook", qty=1, slot="heart"),
                    BomLine(item="hinge", qty=2, slot="heart")]
        if P["treasure"].type == "peel_chute":
            out += [BomLine(item="hinge", qty=3, slot="treasure"), BomLine(item="acrylic_sheet", qty=1, slot="treasure")]
        if P["release"].type == "cam_timer":
            out.append(BomLine(item="print:PLA", qty=15.0, slot="release", note="cam shaft"))
        return out

    def _cost(self, body: Body) -> float:
        usd = 0.0
        for b in self.bom(body):
            usd += b.qty * (PRINT_USD if b.item == "print:PLA" else PRICE.get(b.item, 0.0))
        return usd

    def check(self, body: Body) -> list[str]:
        P, errs = body.parts, []
        env = envelope(body)
        if not fits_box(env):
            errs.append(f"folded it is {env[0]:.1f} x {env[1]:.1f} x {env[2]:.1f} in; the go/no-go box is "
                        f"{BOX[0]:g} x {BOX[1]:g} x {BOX[2]:g} in (either way round, {BOX[2]:g} in tall)")
        u = _uses(body)
        for k, n in u.items():
            if n > KIT[k]:
                errs.append(f"it uses {n} {k}; the kit has {KIT[k]}")
        latched = [s for s, t in LATCHED if P[s].type == t]
        if latched and P["release"].type == "drive_trip" and P["drive"].type != "wheels2":
            errs.append(f"{', '.join(sorted(latched))} must be released, and drive_trip trips only when a wheels2 drive "
                        "stops; use a cam_timer")
        cost = self._cost(body)
        if cost > BUDGET:
            errs.append(f"the parts bought cost ${cost:.2f}; the limit is ${BUDGET:g}")
        rb = _robot(body)
        if P["cannon"].type == "gate_chute":
            ln = MM.chute_len(P["cannon"].params["exit_z"], P["cannon"].params["reach"], rb)
            if ln < 6.7:
                errs.append(f"the cannon chute is {ln:.1f} in long ahead of the deck; four 40 mm balls need 6.7 in")
        for slot, kinds in (("flag", FLAG_BOOMS), ("heart", ("hook_boom",))):
            if P[slot].type in kinds:
                kind = P[slot].type
                p, bm = P[slot].params, _boom(rb, slot)
                if bm["th0"] - bm["th1"] < math.radians(10):
                    errs.append(f"the {kind} swings {math.degrees(bm['th0'] - bm['th1']):.0f} degrees; a stroke "
                                "needs at least 10")
                    continue
                mu = MM.boom_transmission(bm["th0"], bm["th1"])
                if mu < MM.MIN_TRANSMISSION_DEG:
                    errs.append(f"the {kind}'s rod meets the boom at {mu:.0f} degrees at worst over its swing; below "
                                f"{MM.MIN_TRANSMISSION_DEG:g} the piston pushes along the boom and cannot steer it")
                if bm["b"] < 0.5 or bm["cx"] < -rb.L / 2 + 0.25 or bm["cz"] < MM.BELLY + 0.5:
                    errs.append(f"the {kind}'s piston would need its rod pin {bm['b']:.2f} in along the boom and its "
                                f"cylinder pinned at ({bm['cx']:.1f}, {bm['cz']:.1f}) in; the frame runs from "
                                f"x = {-rb.L / 2:g} in, its base plate is at {MM.BELLY:g} in, and the rod pin needs "
                                "0.5 in of boom")
                if slot == "flag" and p["pivot_z"] < rb.H:
                    errs.append(f"the flag boom's pivot ({p['pivot_z']:g} in) is below the deck ({rb.H:g} in)")
        if P["flag"].type == "band_mast":
            p = P["flag"].params
            stretch = (p["stage_len"] - 1.5 + 0.5 * MM.BAND_L0) / MM.BAND_L0
            if stretch > 3.0:
                errs.append(f"each band would start stretched {stretch:.1f} times its {MM.BAND_L0:g} in length; a #64 "
                            "band takes 3")
            if int(round(p["bands"])) < int(round(p["stages"])) - 1:
                errs.append(f"{int(round(p['stages'])) - 1} moving stages need a band each; it has {int(round(p['bands']))}")
        if P["treasure"].type == "peel_chute":
            lay = MM.peel_layout(rb, P["treasure"].params)
            inner = rb.L / 2 / math.cos(lay["slope"]) - MM.SLOT0 - MM.SIEVE_LEN
            if inner < 0.5:
                errs.append(f"the inner trough is {rb.L / 2 / math.cos(lay['slope']):.1f} in long; its {MM.SLOT0:g} in "
                            f"ledge and {MM.SIEVE_LEN:g} in sieve leave {inner:.1f} in of floor (0.5 in needed)")
            if P["treasure"].params["sieve_gap"] > P["treasure"].params["width"] - 0.6:
                errs.append("the sieve's gap leaves its rails less than 0.3 in wide in a trough of width "
                            f"{P['treasure'].params['width']:g} in")
            if lay["jewel_exit_z"] < 0.3 or lay["coin_exit_z"] < 0.3:
                errs.append(f"the side troughs would end {min(lay['coin_exit_z'], lay['jewel_exit_z']):.1f} in up; "
                            "the junction is too low for them to fall that far at their slope")
            if lay["z_jewel0"] - lay["run"] * math.tan(lay["side_angle"]) < MM.BELLY + 0.4:
                errs.append("the jewel funnel would pass through the base plate; raise the junction or flatten the "
                            "side slope")
            if lay["z_front"] > rb.H + 6:
                errs.append(f"the inner trough reaches {lay['z_front']:.1f} in at the front, more than 6 in above "
                            "the deck its posts stand on")
        if not errs:  # parts cannot share space: measured on the built model, every mechanism swept through its motion
            for over, a, b, pose in MM.interference(rb):
                errs.append(f"{_part(a, P)} and {_part(b, P)} pass {over:.2f} in into each other with {pose}; two "
                            "parts cannot be in one place")
        return errs

    def manifest(self) -> dict[str, Any]:
        items = {k: {"on_hand": 0, "usd": v, "lead_days": 2, "doc": k.replace("_", " ")} for k, v in PRICE.items()}
        items["plywood_in2"]["on_hand"] = 400
        items["m3_screw"]["on_hand"] = 32
        items["wheel_pair"]["on_hand"] = 1
        items["ball_caster"]["on_hand"] = 1
        return {"budget": {"usd": BUDGET, "lead_days": 7, "fab_minutes": 900},
                "fabrication": {"print:PLA": {"minutes_per_unit": 1.0, "unit": "g", "doc": "FDM printer"}},
                "items": items}
