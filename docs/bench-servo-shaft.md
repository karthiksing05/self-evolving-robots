# Bench test: can an STS3215 carry the wide-rod jaw's side load?

`wide_rod` holds all 21 objects in simulation, but the judge doesn't read the load on the servo's shaft. The drive rod pushes on the servo horn with about 106 N while gripping, roughly 6 times the stock jaw. Feetech publishes no radial rating for this servo. The ROBOTIS XM430 and XH430, a comparable size, are rated at 40 N, 10 mm from the horn. This test measures the real number.

**The prediction and the pass rules are fixed here before any data, and they don't change afterwards.**

## The loads to reproduce

| Case | Load on the shaft | From |
|---|---|---|
| Steady grip | 106 N (10.8 kg hanging) | Stall torque over the rod's lever, matches the simulator |
| Worst impact on a test run | 211 N (21.5 kg hanging), for about 20 ms in the simulator | Member audit, D-105 |
| Distance from the printed hub's base to the rod's centreline | About 20 mm | The model (hub base z = 36.9 mm, rod z = 56.5 mm). Assumes the hub base sits on the horn face, so check this on the real horn |

At 106 N and 20 mm, the bending moment on the shaft is about 5 times what the ROBOTIS rating allows that servo (40 N at 10 mm). That comparison is why this is the largest doubt.

## What you need

- One STS3215 you can destroy, the 7.4 V version. Don't use a servo from the arm.
- The SO-101 bus board and a 7.4 V supply, ideally with a current limit.
- A way to hold the servo rigidly by its mounting ears: a vise and a bracket, or a printed cradle.
- A stiff load arm bolted to the horn. Aluminium or steel bar is best; printed parts will flex and spoil the reading.
- A load: weights or water containers in a sling (1 L is about 1 kg, so 10.8 L for the steady case), plus a luggage scale to confirm it.
- A dial indicator with a magnetic base. Failing that, a phone macro video against a steel ruler, which is much coarser.
- Safety glasses, and something soft under the load. Keep feet clear.

## Test 1: side load only, servo unpowered

The load's line of action must pass through the shaft axis, so it bends the shaft without turning it. Hang it from a pin through the arm directly over the axis, 20 mm from the horn face.

Run these steps in order: **0, 20, 40, 60, 80, 106, 130, 160, 211 N**, then keep going up in 25 N steps to failure if you're willing to lose the servo.

Hold each load for 60 s. Record at each step:

1. **Deflection under load:** the indicator on the arm at a fixed point, in mm.
2. **Permanent set:** the same reading with the load removed.
3. **Play:** with the load removed, push the arm gently each way and take the reading's spread.
4. **Feel and sound:** turn the unpowered servo through a few degrees by hand and note any roughness or clicking.

## Test 2: the real load case, servo powered

This combines side load with torque, as the rod does. Bolt a crank of the rod's lever, 15.3 mm, to the horn. Command the servo against a fixed anchor so that it pushes through a link at 106 N, measured with the scale or from the current. Don't let it stall for more than 5 s at a time, and let it cool between holds. Stop at 60 °C or when the servo's protection trips.

Log present load, current and temperature from the servo's registers, and repeat the Test 1 readings after every 10 holds, up to 50 holds.

## Test 3: cycles, only if Tests 1 and 2 pass

Run 1000 grip cycles against the same anchor: close over 3 s, hold for 1 s at the steady load, release and cool as needed. Take the Test 1 readings every 100 cycles, plus no-load current through a fixed sweep and position error at a fixed angle.

## Test 4: the printed jaw, only if Test 3 passes

Print `wide_rod`'s parts on a real arm and pick the objects. This needs a printable export of the built jaw, which doesn't exist yet: the repo renders this jaw but has no STL export for it.

## The rules

| Result | Meaning | What the harness does |
|---|---|---|
| **Pass** | Through Test 3: permanent set under 0.1 mm, play grown by under 0.1 mm, no-load current within 20% of its start, no roughness | The largest doubt closes, and `wide_rod` moves on to Test 4 |
| **Survives but degrades** | Passes at 106 N in Test 1, but fails the limits above in Test 2 or 3 | The servo has a real rating below this jaw's duty |
| **Fails early** | Permanent set at or under 106 N in Test 1 | This jaw family can't be built on this servo as it stands |

The 0.1 mm and 20% limits are my judgement of "no meaningful change", not published figures.

In either failing case the measured load becomes a law the judge reads: a shaft side-load rating, exposed to System 2 as a reading on the body. System 2 then redesigns against it, perhaps with a support bearing on the far side of the horn or a shorter offset. No redesign is made by hand, and the current 21 of 21 stands as a result under the old law.

**Prediction, held weakly.** The shaft survives one static 106 N without visible damage, because small servos are often stronger than their conservative ratings. Under Test 3's repeated loading, play grows past the limit, so the most likely result is "survives but degrades". This rests on one comparable rating and no data on this servo.

## Record sheet

| Test | Load (N) | Under load (mm) | After unload (mm) | Play (mm) | Temp (°C) | Current (A) | Feel, notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
