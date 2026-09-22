# Multi-limb robots

Phase two. The single-arm controller ([REACH.md](REACH.md)) drives one two-joint arm without
configuration. The question now is what happens when limbs are attached to each other, so that
one limb's motion changes what the other one feels, and whether limbs should share a feeling.

## First fixture: a stacked chain

Two two-joint limbs in one vertical plane, the second mounted on the tip of the first, giving a
four-joint chain. It is the simplest fixture with real coupling:

- the lower limb carries the upper limb's weight and inertia, which change as the upper limb moves;
- the upper limb's gravity load depends on the lower limb's pose, and its mount accelerates when
  the lower limb moves;
- every joint keeps the single-arm defect model (backlash, stiction, weak and late motors, bad
  encoders) and the mid-episode changes.

Each limb gets its own joint goals in its own encoder units, as now. Success is judged per joint,
so the score can be reported per limb and for the whole chain. The lower limb's motors are rated
twice as strong as the upper's; everything else is sampled from the same populations.

The dynamics kernel becomes a general planar N-joint chain (mass matrix and bias torques by
recursion, one small dense solve per substep), validated against native MuJoCo like the arm was.
The joint model, sensors, populations and defect changes are reused unchanged.

## The experiment

Five controllers on the same held-out defective chains:

| Controller | What it tests |
| --- | --- |
| PID per joint, gains tuned on defective chains | The baseline |
| **A. The single-arm network, one copy per limb, no retraining** | The "one chip per limb" idea as it stands; the other limb is just a disturbance |
| B. One per-limb network, shared weights, retrained on the chain, no communication | How much the coupling can be learned per limb alone |
| C. Same as B, but the two copies exchange a small message each tick (the shared feeling) | Whether sharing a feeling helps |
| D. One network over all four joints | The upper bound on what communication can buy |

B, C and D are distilled from an oracle trained on the chain, as before. If C is close to D and
clearly above B, limbs sharing a feeling is worth building on; if B is already close to D, a chip
per limb is enough and the message can be dropped.

## Out of scope for now

Legs and contact, obstacles, three-dimensional chains, and a real robot. The chain kernel is
written for any N, so a three- or five-joint arm falls out of it for later.
