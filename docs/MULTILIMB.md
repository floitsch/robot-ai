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

## What has been learned so far

- A single four-joint policy trained by PPO from scratch on defective chains does not learn: it either falls into a
  50 Hz command limit cycle (a sign flip every tick through one tick of delay, which the arm's plant filtered and
  the chain's light upper links do not) or, with a stronger smoothness penalty, collapses its exploration and
  barely moves. The same recipe on a one-limb chain trains better than the original arm oracle, so the simulator,
  task and reward are sound; four coupled joints is the optimisation problem.
- A per-limb policy without a message cannot control the chain, structurally: the upper limb's gravity load and
  base motion depend on the lower limb's pose, which its own encoders never show.
- Seeding works. A computed-torque controller with perfect knowledge (`control/computed_torque.py`; inverse
  dynamics plus PD, memoryless, on measured state) scores 99.7% on healthy chains and 14% on defective ones. A
  four-joint student distilled from it with DAgger mixing (`distill --teacher computed-torque
  --teacher-memoryless --teacher-omega 12 --teacher-drive 0.4 --severity 0`) reaches 76% on healthy chains after
  100 iterations. Stiff or ramped variants of the teacher were harder to imitate.
- Two training bugs found on the way: exploration noise must be scaled per joint by rated torque, and the
  smoothness penalty must be charged on the policy's mean command, not the noisy sample, or PPO is paid for
  killing its own exploration. And one instrumentation bug: training-time evaluation used the defective
  population for healthy-stage runs, which made several healthy-stage results look like failures.

- PPO on healthy chains from scratch does not learn either (0% after 50 iterations, exploration collapsing), where
  the computed-torque seed reaches 76%: seeding with a good algorithm is what unlocked the chain.
- The computed-torque teacher cannot be extended to defective chains cheaply: feeding forward the known friction
  changes nothing, because the command and encoder delays make any stiff model-based law chatter. So the second
  stage is PPO, warm-started from the healthy-chain student with gentle exploration (its untrained exploration
  head had to be reset; at 0.5 on an 18 N m joint it wrecked the warm start) and a severity ramp. After 400
  iterations the policy is at 167 mrad median on the full defective population (from 946 at the start), still
  improving but far from the 30 mrad criterion; 1,200-iteration continuations are running.

- The monolithic policy plateaus around 100 mrad median on the full defective population after warm-started PPO
  (about 1,000 iterations); a widened oracle with privileged inputs plateaus at the same place, and a wider
  reward tolerance makes it worse. The failure is stillness at the two coupling joints, not torque or information.
- Per-limb policies imitate the computed-torque teacher far worse than the monolithic one on healthy chains:
  11% with shared sensing (`--peek`, each limb sees the other's raw sensors) against 76% monolithic, and no
  better with a limb-identity input. On a stacked chain, one controller over all joints is much easier to train
  than a shared controller per limb, even when the limbs can see each other; what the per-limb form is missing is
  the other limb's *intent* (its goal and coming motion), which a learned message could carry but which is slow
  to train in the current tick-by-tick implementation.

Where this leaves the design: the shared-feeling experiment (B, C, D) has a clear D and a weak B/C; a fair C needs
a fast message implementation. The monolithic chain controller needs a new idea for stillness at the coupling.

## Out of scope for now

Legs and contact, obstacles, three-dimensional chains, and a real robot. The chain kernel is
written for any N, so a three- or five-joint arm falls out of it for later.
