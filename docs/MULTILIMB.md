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

The curriculum now under test: healthy chains first (seeded by computed torque, or by PPO), then PPO on
defective chains warm-started from that, then the per-limb comparison.

## Out of scope for now

Legs and contact, obstacles, three-dimensional chains, and a real robot. The chain kernel is
written for any N, so a three- or five-joint arm falls out of it for later.
