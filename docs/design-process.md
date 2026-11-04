# The harness against the classical design process (audit, 2026-11-04)

The question: does the evolution loop still model how mechanical engineers and industrial designers actually
design, and what does their practice tell us to give System 2, or to change in how the loop decides?

The yardstick is the systematic design literature:
- Pahl and Beitz, with VDI 2221 and VDI 2206 for mechatronics;
- Ulrich and Eppinger's product development process;
- Toyota's set-based concurrent engineering;
- Suh's axiomatic design;
- TRIZ;
- Boothroyd and Dewhurst's design for manufacture and assembly (DFMA);
- the reflective-practice and function-behaviour-structure accounts of designing (Schön, Gero);
- for our first world, the gripper-design literature (Cutkosky's grasp taxonomy; Monkman et al.'s *Robot
  Grippers*).

Sources are at the end.

## 1. What the harness already models well

| Classical practice | In the harness | Where |
|---|---|---|
| Design types: variant, adaptive and original design (Pahl and Beitz) | The ladder: params, swap, idea with a mechanism | `evolve/core.py`, philosophy §5 |
| Analyse the failure mode before redesigning | Every failure carries a mechanism with its numbers | `World.solve`, `diagnose` |
| Embodiment checks: function, layout, strength, assembly | The mechanism laws before any trial: carried, interference, toggle, bushing load, tooth skip | `mechanism/gripper.py` |
| Economic evaluation | A bill of materials priced against the bench, with lead times and fabrication minutes | `resources.py` |
| Prototype testing, with regression against what already works | Paired trials and the ratchet: gain a group, lose none | `loop.py` |
| A fidelity ladder: hand calculation, analysis, focused prototype, comprehensive prototype (Ulrich and Eppinger) | The brief, then `check`/`measure`, then `simulate` (one task), then the trial | D-086 |
| Design catalogues of machine elements, and standard parts | The modules and the planned library (C2); real bought items (LM3UU, MR63, rods) | `mech/`, manifest |
| Design rationale | Every proposal states its mechanism, hypothesis, predicted gains and risks, scored against the trial | `Proposal`, calibration section |

The skeleton is sound: problem → diagnosis → proposal at the cheapest sufficient rung → analytic checks → test → keep only what does not regress. Two of the gaps below are the real constraints on creativity. Several are cheap.

## 2. What classical practice has that the harness lacks

Each gap is listed with what it would look like here. Every item is a fact or a process the harness supplies; none tells System 2 what to design. This matters for philosophy §4: facts, never advice.

### G1. A requirements list with target values (highest value, cheap)

Designers turn needs into a requirements list before any concept. Each metric carries a marginal and an ideal value, and is marked as a demand or a wish (Pahl and Beitz; Ulrich and Eppinger's target specifications). System 2 sees tasks as pass/fail groups and failures one at a time. It never sees the set of metrics the task set implies, or how far the current body is from each.

For the gripper, these are all computable from the task data and the physics:
- the opening each object needs;
- the grip force each needs: the friction-grip rule F ≥ m(g + a)·S / (μ·n), with a safety factor S, used in gripper catalogues;
- the height at which each must be held, and the clearance above the table;
- the mass at the wrist;
- the wrist roll.

**Bring it in.** Add `World.requirements(tasks)`, giving a table of metric, need per task, the current body's value and the margin. It goes in the prompt, and `measure` evaluates drafts against it.

This also exposes contradictions, which TRIZ makes the start of invention. For example, "the face must be parallel at 17 mm and at 95 mm, but a hinged face is parallel at one opening only". The world states the facts; System 2 names the contradiction.

### G2. Develop a concept before judging it (high value: the creativity bottleneck)

Set-based concurrent engineering keeps several concepts alive. It narrows them by trade-off evidence, and it gives a new concept a development budget before comparing it with a mature incumbent (Sobek, Ward and Liker). Our loop already does this for catalogue swaps: the tuning allowance (philosophy §7). It does not do it for mechanisms.

In c1c the long slide was written about ten times. Each time it was a fresh, undeveloped concept judged against a tuned hinged jaw. It needed three developments at once (more bushings, a bracket, a trimmed sector), and it lost to whichever was missing. No seed developed one concept across iterations.

**Bring it in.** Concept lineages:
- A built mechanism that gains a group, or is refused for a named law, becomes a concept on the frontier.
- System 2 may spend proposals developing it with a patch on its last graph (`base` already allows this for bodies).
- The loop gives each concept an allowance of trials before it is dropped, as `tune_trials` does for swaps.
- The record shows each concept's lineage: what each development fixed and what it still fails.

### G3. Work at the level of function before form (medium value, overlaps C2)

Conceptual design abstracts the problem into a function structure, for example:
- "convert servo rotation to jaw travel";
- "keep the contact face parallel";
- "react the squeeze into the frame".

It then lists working principles per subfunction (a morphological chart) and combines them. System 2 jumps from a failure to a part. The modules are embodiments with no stated function.

**Bring it in.**
- Each module declares `provides` (`rotate`, `translate`, `keep_orientation`, a path), as the C-plan already specified.
- The prompt shows a morphological chart: subfunction × the modules that provide it, with each one's measured facts.
- A proposal may state the function it changes.

Promotion in C2 needs `provides` anyway.

### G4. Compare concepts on criteria before building them (medium; partly done by D-086)

Concepts are screened (Pugh) and scored against the requirements before a prototype. With the tools, System 2 can measure several drafts. The loop could ask that a redesign be compared with at least one alternative on the requirement metrics (G1), and record the comparison as rationale. This is a process step, not a judgement: the trial still decides.

### G5. Domain engineering knowledge (medium value, cheap, handle with care)

Practising gripper designers work from the field's handbook knowledge:
- parallel-jaw grippers dominate industrial picking;
- friction grip against form closure (V-grooves, encompassing fingers);
- the grip-force rule with its safety factor;
- the derating of grip force with finger length;
- the grasp taxonomy, as a vocabulary of what a hand must do.

The world's principles hold some of the physics (antipodal contacts, tilt, the fingertip's rise), but none of this sourced knowledge.

**Bring it in.** The world's principles gain sourced engineering rules, each cited. They must be general knowledge, never a recommendation for this body. A rule is not a refusal: evidence before outlawing still holds. The risk is steering: "parallel grippers dominate" could anchor every proposal. So it enters as one lever with its own campaign, so that its effect on diversity is measured.

### G6. Predict failure modes before the test (low to medium)

FMEA and design reviews list how a new design could fail before it is built. Proposals carry named risks to held groups. Missing is the mechanism's own failure modes: tooth skip, bushing overload, interference, toggle, the object jamming on a part. The laws catch some of these; the jam is found only by trial.

**Bring it in.** A redesign states its expected failure modes, and the tools (`check`, `simulate`) confirm or refute them before the trial. Scoring these against the trial extends the calibration System 2 already sees.

### G7. Manufacturing reality: tolerances, part count, assembly (realism, not creativity)

DFMA minimises part count and checks assembly order. Tolerance analysis checks that a printed part at ±0.2 mm, with servo backlash, still works. The laws check interference, mounting and rated loads, but not tolerance stacks or backlash. This matters for sim-to-real, more than for the campaign. Its natural place is the planned fab and sensing work.

### G8. Reformulate the problem; co-design body and behaviour (large, a boundary question)

Designers reframe the brief when a requirement cannot be met (Schön's reflective conversation; Gero's reformulation of function). Mechatronic design (VDI 2206) co-designs the mechanism with its control.

Every gripper seed loses the mustard bottle. A designer would ask whether the grasp itself should change (where on the object, from which side) rather than only the jaw. Our System 2 may change only the body; System 1 searches the controls of a fixed grasp program.

**Bring it in, later.** Let System 2 propose a change to the grasp's structure (for example the approach or the contact height), typed against the skill vocabulary. It is screened and trialled like a body change. This crosses the body/skill boundary and needs its own decision.

## 3. What this means for the order of work

1. **Run the D-086 campaign as built.** One lever per campaign, so its measurement means something (iteration.md).
2. **G1**, the requirements table, in the brief and in `measure`. It is cheap and computable, and it gives every later lever a metric space.
3. **G2**, concept lineages with a development allowance. This addresses the observed bottleneck: undeveloped mechanisms judged against a tuned incumbent.
4. **G3**, function tags and the morphological chart, with C2's library.
5. **G5**, sourced handbook principles, as its own measured lever.
6. **G4 and G6** as prompt-level process, once G1 exists to measure against. **G7** with fabrication, **G8** as its own decision.

## Sources

- Pahl, G., Beitz, W., Feldhusen, J., Grote, K.-H. *Engineering Design: A Systematic Approach*, 3rd ed. Springer, 2007.
- VDI 2221 Part 1 (2019), *Design of technical products and systems*; VDI 2206 (2021), *Development of mechatronic and cyber-physical systems*.
- Ulrich, K. T., Eppinger, S. D. *Product Design and Development*. McGraw-Hill.
- Sobek, D. K., Ward, A. C., Liker, J. K. "Toyota's principles of set-based concurrent engineering." *Sloan Management Review* 40(2), 1999.
- Suh, N. P. *The Principles of Design*. Oxford University Press, 1990.
- Altshuller, G. *The Innovation Algorithm* (TRIZ). Technical Innovation Center, 1999.
- Pugh, S. *Total Design*. Addison-Wesley, 1991.
- Boothroyd, G., Dewhurst, P., Knight, W. *Product Design for Manufacture and Assembly*. CRC Press.
- Schön, D. A. *The Reflective Practitioner*. Basic Books, 1983.
- Gero, J. S. "Design prototypes: a knowledge representation schema for design." *AI Magazine* 11(4), 1990.
- Cutkosky, M. R. "On grasp choice, grasp models, and the design of hands for manufacturing tasks." *IEEE Transactions on Robotics and Automation* 5(3), 1989.
- Monkman, G. J., Hesse, S., Steinmann, R., Schunk, H. *Robot Grippers*. Wiley-VCH, 2007.
