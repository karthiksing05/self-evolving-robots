# The goal, and what counts as reaching it

A framework that can autonomously iterate on any design challenge and verify the design through feedback from the
designed product, in simulation or in real life. Set by the user on 2026-11-05. The gripper, the mobile robot and the
talking robot are the sample tasks. ME 2110 is the final result and starts only on the user's word.

This was written before the runs it judges. A criterion changes only by a new dated entry here, saying why.

## Done means all of these hold, on all three tasks

1. **One command.** The same `scripts/evolve.py` flags on every task, differing only in `--world` and `--seed`.
2. **From stock.** No `--start`: the run begins at the world's stock body. The notebook and the library start empty, so
   what it knows it researched itself in that run.
3. **Nobody between iterations.** No person edits the design, the code or the world during a run. A change to the
   harness voids the run for this purpose; the run is repeated under the new code.
4. **Only general code.** A world supplies its brief, physics, instruments and kit, and nothing about how to solve it.
   `tests/test_task_agnostic.py` holds this.
5. **Verified by the product's feedback.** Every kept change came from a trial of that design. Every failure comes back
   to System 2 as a labelled measurement, and later proposals read it. The final body is rechecked on held-out tasks and
   on seeds the run never saw.
6. **The feedback channel is not tied to simulation.** The loop reads only `World.solve`'s `Outcome`, so a test drives a
   whole run from feedback produced outside the process, standing in for a real product.
7. **It finishes, or it says why not.** Either every group holds on train and test under every recheck seed, or the
   run ends with what stopped it named and measured. "Ran out of iterations" is not an answer on its own.

## The commands

```
scripts/evolve.py --world W --seed S --iterations 20 --workers 3 --tools --requirements --research \
    --stripped --plans --concepts --library --robust 4 --out runs/evolve/camp_W_sS \
    --notebook runs/evolve/camp_W_sS.notebook.jsonl --library-dir runs/evolve/camp_W_sS.library
scripts/recheck_bodies.py --world W --seeds 7000 ... 7007 --workers 4 --out OUT runs/evolve/camp_W_sS
```

W is `mobile_kit`, `phonology_kit` and `gripper`.

## Not reached by what came before

- The gripper's 21 of 21 (`runs/evolve/gripper_j_s1`) is real and verified, but it came from a chain of runs, each
  started by hand from the last while the laws changed between them, and with a shared notebook. It fails 1, 2 and 3.
- The catalogue worlds' 13 of 13 and 17 of 17 chose among parts written by hand for those tasks. They fail 4.
- `runs/evolve/smoke_mobile_kit_s1` and `runs/evolve/smoke_phonology_kit_s0` ran one iteration each.
