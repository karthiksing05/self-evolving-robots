# The design language

System 2 proposes bodies as parameter sets on catalogue parts. Something has to turn those parameters into an
object that could be printed, cast and screwed together, and into a picture a person can read. The design language
is the set of rules for that step. It decides how a part is built and drawn. It never decides which part a body
uses or what its parameters are: a redesign still comes only from System 2.

It has two halves:

- **Rules that generators follow by construction.** `design/pad_cad.py` (finger pads) and `worlds/rig_cad.py` (the
  talking robot's bench rig) read their numbers from `design/language.py` and build to them.
- **Laws measured on the drawn assembly.** `language.audit` checks a declared assembly (members, plus the joints
  between them) with the coal geometry in `mechanism/geometry.py`.

All of it is TOOLING (`harness.TOOLING`). It shapes pictures and reports, and no verdict reads it. The laws are written
so they could later join the judge, as the gripper's mechanism laws did (D-059).

## The numbers

| Quantity | PLA (FDM, 0.4 mm nozzle) | TPU | Cast silicone | Why |
|---|---|---|---|---|
| Thinnest wall | 1.2 mm | 0.8 mm | 2.0 mm | Whole perimeters (3 for PLA, 2 for TPU); a cast wall must survive demolding |
| Smallest pin, rib or slot | 0.8 mm | 1.0 mm | 1.5 mm | Below this, FDM drops the feature and silicone tears |
| Least radius on an exposed edge | 1.0 mm | 1.0 mm | 1.5 mm | Sharp edges chip, cut silicone and catch cables |

Other numbers:

- **Slip fit:** 0.2 mm, for a printed socket over its host.
- **Overhang:** at most 45° from vertical in the stated print orientation.
- **Grid:** 5 mm, for where fixed parts stand on a base plate.
- **Contact:** parts within 0.5 mm of each other touch.
- **Sharp edge:** a convex edge whose faces turn by more than 60° (a tessellated 1 mm round turns by about 20°).

## Rules generators follow

| Rule | Why | How it is followed and checked |
|---|---|---|
| **Contact geometry is the simulator's.** A drawn part keeps every surface the simulator collides with exactly where the simulator has it. | The picture must show what was judged. A prettier pad that grips somewhere else would be a lie. | Pad lands are coplanar with the simulated box faces and end where those faces end, within 0.1 mm. The underside lies in the boxes' bottom plane. Checked by `test_pad_lands_are_the_simulated_contact_faces`. |
| **Fillets and chamfers.** Every exposed edge that is not a land is rounded to at least the least radius. Where a land ends, its rounds start tangent to it. | A tangent round keeps the land's extent exact while removing the edge. | By construction for blocks and pads (`_box(fillet=...)`, `_profile`). Measured by the sharp-edge law, which still finds sharp edges on fused profiles: the face plate's rim, the servo mounts and the pharynx clamp. |
| **Conformal and mechanically retained.** A part sits on what carries it by its own shape. The pad shoe's socket is the jaw's own CAD, grown by the slip fit, and two M2 cap screws hold it from behind. The TPU skin is printed onto the shoe. The cast tract is held at the front by the face plate and at the back by the reed housing or a clamp ring. An articulator is a membrane of even thickness set into a pocket in the cast, flush with the bore, and pressed into the bore where its control puts it. No box is laid against a face, and no edge is feathered. | Glued boxes fall off. A socket also locates the part. | Declared joints must touch (the loose law). The shoe's walls come from the host's sections. |
| **Every part is mounted.** Each part reaches the base plate through a named fastener, bracket, boss or contact. Screws are drawn. | Nothing floats in a real build. | The grounded law. |
| **Servos in mounts, horns and linkages visible.** Each MG90S sits in a printed cradle with its tabs screwed down and its horn on the spline. The linkage it drives is pinned to the horn: a crank, a cam, a push rod or a link. | A reader must see what moves what. A loose servo cannot be built. | `_Rig.servo` builds the servo, mount, screws and horn together. Checked by the grounded and loose laws. |
| **Rods only through declared guides.** Push rods and followers pass only through a bushing or a slot that is declared a guide. | An undeclared pass-through is a collision. | The interference law checks guide pairs too: a rod that touches its guide fails. |
| **Symmetry and a grid.** Fixed parts stand on the 5 mm grid and are symmetric about the tract's centre plane. The exception is where the mechanism needs space: the pharynx clamp's post stands beside the tongue sled's travel. | A legible layout and a drillable plate. | By construction. There is no law yet. |
| **Consistent walls.** Printed walls are 1.6 mm (four perimeters) around a host, and never under the thinnest wall. | Even walls print and cool evenly. | The thinnest-wall law. Proposals that cannot meet it are refused (below). |
| **Minimum features.** Pins, rods and ribs are at least the smallest feature. | As for the numbers above. | By construction. The thinnest-wall law catches the rest. |
| **Overhang, or a stated orientation.** Every printed member of the rig states its build direction (`Member.print_up`). The pad shoe prints lying on its back (stated in `pad_cad`). | An overhang is only meaningful against a build direction. | The overhang-fraction law, in the stated orientation. It does not yet run on the pads, whose drawing does not carry its orientation. |
| **Routing.** Air hoses bend no tighter than 1.5 times their outer diameter and leave toward their source (off the bench, or to the manifold). Cords run from a horn to a clip outside the part they pull. | Kinked hoses choke the flow. A cord through a lip would cut it. | By construction. A route through another part fails the interference law. |
| **Fairings and sections where they clarify.** Enclosures are drawn only where they hold something: the reed housing's air chamber and the valve manifold. The tract is drawn in section so the bore and the articulators inside it show. | A body is read through its mechanism, not its covers. | By construction (`Member.section`). |
| **A posture moves only its mechanism.** Positions come from the rig's controls. The base plate is sized over every mechanism's whole travel, so no fixed part changes with a posture. | The same body must look the same at rest and in motion. | `test_a_control_moves_its_mechanism_and_nothing_fixed`. `rig_cad.sweep` applies the laws at both ends of every control. |

## Laws, measured

| Law | Measure | Code |
|---|---|---|
| Grounded | Parts that no chain of parts touching within 0.5 mm joins to the base | `language.floating` → `geometry.grounded` |
| No interference except declared contact | Between every pair not declared a joint or contact: surface crossings (or a part sealed inside another) where the parts also share volume more than 0.5 mm deep. Parts that only touch, such as an articulator sealing against the opposite wall, pass. | `language.interference` → `geometry.overlap`, `language._deep` |
| Joints hold | Declared joints and contacts whose parts are more than 0.5 mm apart | `language.interference` |
| Sharp-edge length | Length of convex edges that turn by more than 60° (mm), excluding the first layer's edges on the bed | `language.sharp_edge_mm` |
| Overhang fraction | Share of the surface facing down more steeply than 45° from vertical, in the stated orientation, excluding the faces on the bed | `language.overhang_fraction` |
| Thinnest wall | 1st percentile of the wall thickness, by inward rays | `language.thinnest_wall` |

`scripts/design_language.py` runs every law on the pad assemblies and on four rig bodies. It also runs them at
both ends of every control's range. It writes `results/design_language/audit.md`.

## Which rules bind proposals

- **Proposal tier.** A parameter set that cannot be built to the language should be refused before anything is
  simulated, just as `design/build.py` refuses an unprintable part. `pad_cad.refusals` names two such cases:
  - a TPU skin thinner than TPU's 0.8 mm wall;
  - a groove that leaves less than 1.2 mm of PLA between its floor and the jaw's socket. The backing is 3 mm and
    the slip fit takes 0.2 mm, so this means any groove deeper than 1.6 mm.

  The laws are the next candidates: a body that interferes at the end of a control's range cannot be built as
  drawn. One example is a tip tongue whose tip range overlaps the body pad's range. None of this is wired into the
  ladder yet. Doing so changes the judge, so it needs a decision entry and a re-baseline.
- **Drawing only.** Rounds, mounts, screws, routing, the grid, sections, colours and callouts change how a body
  looks, never what it scores. The parts the language adds (mounts, screws, clamps, brackets) are not in the bill
  of materials yet.
- **System 2 designs.** The language builds and draws whatever a proposal names. It adds no catalogue part and
  picks no parameter.

## Where it lives

- `src/sea/design/language.py`: the numbers, `Assembly` and `Member`, and the laws.
- `src/sea/design/pad_cad.py`: the printed pads over the simulated boxes. `worlds/cad.gripper` calls `dress`.
- `src/sea/worlds/rig_cad.py`: the talking robot as a bench instrument. `worlds/cad.phonology` calls it.
- `src/sea/design/callouts.py`: labelled pictures, with numbered callouts and a legend of what each part does.
- `tests/test_design_language.py`: the lands, the frames, the audits, the refusals and the posture.
- `scripts/design_language.py`: before-and-after renders and the audit, in `results/design_language/`.

## Simulated geometry the language cannot build

Drawing every body to these rules showed six places where the simulated geometry differs from anything that
could be built. None is fixed yet. Each is a world-tier or judge-tier change, so each needs a decision entry and,
for the gripper, a gold re-record.

1. **V-groove boxes reach into the jaw.** The groove half-boxes extend behind the jaw face by up to the groove
   depth. The jaw collides with nothing in the simulator, so the physics does not notice. But a groove deeper than
   about 1.6 mm leaves no printable backing behind it. So `grooved40` (a 3 mm groove) cannot be built as simulated,
   and `pad_cad.refusals` refuses it.
2. **The tilted moving pad is partly buried.** Past about atan(thickness / 10 mm) of pad angle, the top of the
   moving pad's box is inside the jaw. The simulator still counts that part of the face as contact surface,
   though nothing can reach it.
3. **The lip box sticks out half a pad thickness too far.** It protrudes lip + t/2 beyond the land, not lip.
4. **The tip tongue's ranges overlap the body pad's.** The tip's range (0.3–5 cm) overlaps the body pad's
   (3.25–10.25 cm). At tongue_place 3.25 or tip_place 5, the tongue pad and the tip blade share up to about
   61 × 26 × 9 mm. The acoustics allow a posture the hardware cannot reach. This should become a range
   constraint or a proposal-tier law.
5. **A flat-topped constriction feathers.** The simulated constriction has a flat top across a round bore, so a
   solid pad drawn exactly to it thins to nothing where the flat meets the wall. The membranes fix this wherever
   the top is below the bore's centre line. Above it, a thin edge remains.
6. **The tip sits too close to the face plate.** With tip_place below about 1 cm, the tip mechanism would be inside
   the face plate. The drawing keeps the stem at least 10 mm behind the teeth and lets the blade reach forward to
   the commanded place.

The drawn pads also carry material the simulator does not have: the shoe that wraps the jaw and its screws, a
few grams and a few millimetres around the finger. The mechanism laws (D-059) read the simulated boxes, not the
shoe. Under invariant 11 the shoe belongs in the simulator as well; until then it is listed in
[harness.md](harness.md) under "Where the harness is still thin".
