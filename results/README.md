# Measured results

Small, tracked outputs that decisions cite. Run directories (`runs/`) stay untracked.

- `real_stock.json`: the stock jaws on the 21 real objects (`scripts/measure_real.py`). It is the
  phase A / phase B split (D-038).
- `gold/<design>.json`: the same measurement on finger designs (the V2 gold check). The file
  name encodes the design: `r` reach in mm, `a` pad angle in degrees, `g` groove in mm, `t` TPU
  skin in mm, `w` pad width in mm. For example, `r10a20t2` is reach 10, angle 20, TPU 2.
- `design_language/`: the design language (docs/design-language.md) before and after, written by
  `scripts/design_language.py`. It holds the pads and the talking-robot rig as drawn before (`before_*`; the rig's
  before pictures are from commit cbfb3a2) and after (`after_*`, `tips_*`). It also holds labelled rigs with a
  legend of what each part does (`labelled_rig_*`) and the audit of every drawn assembly (`audit.md`, `audit.json`).
- `superseded/`: the gold sweeps run before two faults were fixed (D-038), kept so the
  before-and-after numbers in the decision log can be checked:
  - `opening_bug`: `fingers.opening_for` fell back to fixed jaw angles;
  - `soft_tpu`: the TPU skin had a guessed 0.04 s contact time constant.
- `experiments/`: the first live gripper experiment (D-038, D-039), one directory per arm:
  - `gripper_live_s1`: System 2 on the Claude plan may change the fingers;
  - `gripper_s1_only_s1`: the control, System 1 only.

  Each holds its spec, harness fingerprint, final summary, final design and console log. The live
  arm also keeps every iteration's proposals, decision and LLM call record. `<run>.json` gathers
  all of it per iteration (coverage, escalations, proposals with hypotheses, screening errors,
  trial shares, decisions, token counts) together with the final per-table verdicts. The full run
  directories, with traces and memory, stay in the untracked `runs/`.
