# Design-language audit

Each assembly is checked by `sea.design.language.audit` (docs/design-language.md): parts that no chain of touching parts carries to the base (`mechanism.geometry.grounded`), pairs with no declared joint whose surfaces cross (`mechanism.geometry.overlap`) and that share volume more than 0.5 mm deep (undeclared), and declared joints whose parts do not touch (loose).  The form columns are over the parts the generators print (PLA and TPU; the stock jaws are their maker's): the worst sharp edge, the worst overhang share in the part's stated print orientation, and the thinnest wall.  The pads state no print orientation yet, so their overhang is not measured.

| assembly | parts | joints | floating | undeclared | loose | sharp edge (mm) | overhang | thinnest wall (mm) |
|---|---|---|---|---|---|---|---|---|
| pad flat, on wrist_roll_follower_so101_v1 | 4 | 4 | 0 | 0 | 0 | 189.8 (fixed pad shoe) | - | 1.63 (fixed pad shoe) |
| pad flat, on moving_jaw_so101_v1 | 4 | 4 | 0 | 0 | 0 | 196.2 (moving pad shoe) | - | 1.21 (moving pad shoe) |
| pad grooved40, on wrist_roll_follower_so101_v1 | 4 | 4 | 0 | 0 | 0 | 253.4 (fixed pad shoe) | - | 0.16 (fixed pad shoe) |
| pad grooved40, on moving_jaw_so101_v1 | 4 | 4 | 0 | 0 | 0 | 242.7 (moving pad shoe) | - | 0.39 (moving pad shoe) |
| pad hooked, on wrist_roll_follower_so101_v1 | 4 | 4 | 0 | 0 | 0 | 244.3 (fixed pad shoe) | - | 1.04 (fixed pad skin) |
| pad hooked, on moving_jaw_so101_v1 | 4 | 4 | 0 | 0 | 0 | 237.2 (moving pad shoe) | - | 0.27 (moving pad skin) |
| rig stock | 33 | 43 | 0 | 0 | 0 | 644.3 (tongue pad) | 0.27 (tongue pad) | 1.17 (tongue pad) |
| rig rich | 61 | 82 | 0 | 0 | 0 | 620.3 (tongue pad) | 0.27 (tongue pad) | 1.17 (tip blade) |
| rig pneumatic_spread | 33 | 47 | 0 | 0 | 0 | 444 (face plate) | 0.27 (servo mount) | 1.6 (horn) |
| rig long_rail_long_cast | 33 | 43 | 0 | 0 | 0 | 764.3 (tongue pad) | 0.27 (tongue pad) | 1.17 (tongue pad) |

At each end of each control's range (`rig_cad.sweep`), the other controls at the default posture:

- stock: nothing floats, interferes or comes loose
- rich, tongue_place = 3.25: tongue pad meets tip stem (3 x 3 x 1 mm); tongue pad meets tip blade (61.3 x 26.4 x 8.9 mm)
- rich, tip_place = 5: tongue pad meets tip stem (3 x 3 x 1 mm); tongue pad meets tip blade (61.3 x 26.4 x 7.6 mm)
- pneumatic_spread: nothing floats, interferes or comes loose
- long_rail_long_cast: nothing floats, interferes or comes loose

Pad parameter sets the proposal-tier rules would refuse:

- flat: buildable
- grooved40: a 3 mm groove leaves 0.0 mm of PLA between its floor and the jaw's socket (at least 1.2)
- hooked: buildable
