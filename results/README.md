# Measured results

Small, tracked outputs that decisions cite. Run directories (`runs/`) stay untracked.

- `real_stock.json`: the stock jaws on the 21 real objects (`scripts/measure_real.py`). It is the
  phase A / phase B split (D-038).
- `gold/<design>.json`: the same measurement on finger designs (the V2 gold check). The file
  name encodes the design: `r` reach in mm, `a` pad angle in degrees, `g` groove in mm, `t` TPU
  skin in mm, `w` pad width in mm. For example, `r10a20t2` is reach 10, angle 20, TPU 2.
- `superseded/`: the gold sweeps run before two faults were fixed (D-038), kept so the
  before-and-after numbers in the decision log can be checked:
  - `opening_bug`: `fingers.opening_for` fell back to fixed jaw angles;
  - `soft_tpu`: the TPU skin had a guessed 0.04 s contact time constant.
