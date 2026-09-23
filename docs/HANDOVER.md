# Hand-over

Written 2026-09-23, at the end of the first two working sessions. It says where the project
stands, what is settled, what is open, and what I would do next. [README](../README.md) is the
outward-facing description; [REACH.md](REACH.md) and [MULTILIMB.md](MULTILIMB.md) are the method
and result documents. This file is for whoever continues the work.

## The goal, restated

One set of network weights that drives *any* cheap, imperfect robot limb without being configured
for it: no gains to tune, no model of the arm, no calibration. The network reads encoders, motor
current and a joint goal, and works out from the response what kind of arm it is holding (slack,
sticky, weak, late) while it drives it. Defects and mid-motion changes are the point, not a later
milestone. The eventual form is weights on a chip that a robot maker drops into any arm.

## State in one paragraph

Single two-joint arms: **done and good**. One recurrent network reaches its goal on 83% of 4,096
held-out defective, changing arms against 22% for a PID whose gains were grid-searched on the same
population, and it matches that PID's precision on healthy arms. It exports to a single C file.
Stacked limbs (two two-joint limbs, a four-joint chain): **open**. The best chain policy is at
90 mrad median error against a 30 mrad criterion, roughly 1% success. The chain is a genuinely
harder control problem and is not solved.

## Part 1: the single arm (finished)

4,096 held-out robots, none seen in training. Success = every joint within 30 mrad of its goal and
moving slower than 0.1 rad/s through the last 0.3 s of a 3 s episode, with a goal change in the
middle. Errors below are medians at the end of the episode.

| Robots | Tuned PID | Memoryless net | Recurrent net (PPO) | **Deployed net** | Oracle (undeployable) |
| --- | --- | --- | --- | --- | --- |
| Healthy | 97.6% / 1.1 mrad | 77.9% | 87.8% | **98.4% / 1.0 mrad** | 100% |
| Defective | 25.2% / 96.8 mrad | 35.8% | 50.0% | **83.3% / 2.4 mrad** | 96.2% |
| Defective + changing | 22.2% / 142 mrad | 33.9% | 46.2% | **83.3% / 2.4 mrad** | 95.1% |
| ... + pushed (untrained) | 12.6% | 26.1% | 32.1% | **51.8% / 5.3 mrad** | 82.1% |

The recipe, in order of how much each step mattered:

1. **Distil, don't only reinforce.** Train an oracle by PPO that is *told* the true joint state and
   every hidden defect parameter; then have a deployable network drive the arms itself while the
   oracle, watching the hidden truth of those same states, says what it would have done (DAgger).
   PPO alone plateaus at 46%; the first distilled student hit 55% immediately.
2. **Train the teacher against a stricter tolerance than the criterion.** 30 mrad reward → 76%
   teacher; 15 mrad → 86.5%; 10 mrad → 93.4%; 5 mrad → 95.1%. The student tracks its teacher about
   12 points below, so the teacher is the lever. This one change took the student from 66% to 83%.
3. **Feed the goal error magnified**: `tanh(err/0.05)`, `tanh(err/0.5)` alongside the raw error. A
   30 mrad error is otherwise a 0.03 input among inputs of order one, too faint to steer by. 34% → 46%.
4. **Student width**: 64 → 128 → 256 → 384 units gave 55% → 62% → 66% → 83% (with the better teacher).

The artifacts: `artifacts/reach/best-student` (the network), `export/reach_policy.c` (checked in,
compiles with only `<math.h>`), `artifacts/reach/comparison-best.html` and `docs/media/reach.gif`.

Known weakness: external pushes (a neighbour shaking the mount) are the worst condition at 52%, and
training on pushes did not help (+1 point, −5 elsewhere) because the *teacher* was not trained on
them either. A push-trained teacher reaches 72.6% on pushes; distilling from it was never run and is
the cheapest remaining win on the single arm.

## Part 2: stacked limbs (open)

Fixture: two two-joint limbs, the upper mounted on the lower's tip, so one limb's motion changes the
other's gravity load and shakes its mount. Same per-joint defect model; the lower limb's motors are
rated 3× for what they carry. Per-limb goals and per-limb success.

Where it stands: best policy `artifacts/reach/chain-slowramp` at **90 mrad / 0.8% success** on
defective changing chains, from a curriculum of computed-torque seeding on healthy chains (76%) then
PPO with a slow severity ramp. For comparison the computed-torque controller *with perfect
knowledge* gets 99.7% on healthy chains and 14% on defective ones.

### What is settled about the chain

- **PPO from scratch does not work on the chain**, in either of two ways: a 50 Hz command limit cycle
  (sign flip every tick through one tick of delay, which the single arm's plant filtered and the
  chain's light upper links do not), or, with a stronger smoothness penalty, exploration collapsing
  to near zero and a policy that barely moves. The same recipe on a *one-limb* chain trains better
  than the original arm oracle (71% at iteration 50), so the simulator, task and reward are sound.
- **Seeding with a good algorithm is what unlocked it.** A computed-torque controller (inverse
  dynamics + PD, given the true state and parameters) distilled into the network with DAgger mixing
  gets 76% on healthy chains. PPO on healthy chains from scratch gets 0%.
- **The teacher must be imitable, not optimal.** Stiff gains (ω=25) give labels the student cannot
  reproduce (imitation loss 0.2); ω=12, memoryless (no reference ramp, feedback on *measured* state)
  and DAgger teacher-driving bring it to 0.0003.
- **Per-limb policies are much worse than one policy over all joints**, which was a surprise. A
  shared two-joint policy per limb reaches 11% on healthy chains with shared sensing (`--peek`, each
  limb sees the other's raw sensors) and ~5% with a limb-identity input, against 76% monolithic. What
  a limb cannot get from the other's sensors is its *intent* — where it is going next. A learned
  message (`--message`) is implemented but its tick-by-tick Python loop is ~3× slower and was never
  trained to convergence; that is the missing arm of the shared-feeling experiment.
- **No single defect explains the chain's failure.** Ablating any one defect class at evaluation
  leaves success near 1%; removing *all* sensing and timing defects gives 12%; full motor strength
  cuts the error to 67 mrad. The policy is a weak generalist over population variation: on identical
  healthy chains it is at 76%, and any variation collapses it.
- **Dead ends** (all measured, none worth repeating): privileged inputs (a widened oracle plateaus at
  the same ~100 mrad as the student), a wider reward tolerance (worse), stillness weight ×3 (worse),
  friction feedforward in the teacher (nothing — the delays dominate).

### What I would try next, in order

1. **A better chain teacher.** Every single-arm gain came from the teacher, and the chain's teacher
   is a classical controller that is itself only 14% on defective chains. Train a chain *oracle* by
   PPO warm-started from the computed-torque student on healthy chains, ramping severity, with the
   strict-tolerance trick that worked for the arm — then distil. The widened-oracle attempt started
   from a *defective-chain* student, which was already stuck; starting from the healthy student is
   the untried version.
2. **Look at what the 90 mrad policy actually does.** `python -m robot_ai.visualize.animate --run
   artifacts/reach/chain-slowramp --limbs 2` renders it beside computed torque on six characteristic
   chains. The per-joint diagnosis says the two *coupling* joints are worst and that only 22–42% of
   joints are ever "still" at the end, i.e. residual oscillation rather than a steady offset. That
   smells like a control-bandwidth problem (the chain's fast bending mode against a 100 Hz loop with
   delays), which may need a different action interface, not more training.
3. **Fast inter-limb messages.** Fuse the per-limb message loop so all limbs step in one batched call,
   then run the B/C/D comparison the design doc asks for. Only then is the shared-feeling question
   actually answered.
4. **Cheap single-arm win**: distil a student from the push-trained teacher (`oracle-tight-pushes`,
   72.6% on pushes) and see whether the pushed condition moves from 52%.

## How to run things

GPU work goes through the wrapper, which keeps every cache and output inside the project:
`scripts/project-run .venv/bin/python -m robot_ai.train.reach ...`. Everything below assumes that
prefix. Full recipes are in [REACH.md](REACH.md); the essentials:

```sh
# Single arm: teacher (about 3 h on a GTX 1650), then student (about 1 h).
... reach train --fine-error --oracle full --hidden 256 --reward-tolerance 0.005 \
      --iterations 800 --output artifacts/reach/oracle
... reach distill --teacher artifacts/reach/oracle --hidden 384 --iterations 400 \
      --output artifacts/reach/student

# Chain: seed from computed torque on healthy chains, then PPO with a severity ramp.
... reach distill --teacher computed-torque --teacher-omega 12 --teacher-memoryless \
      --teacher-drive 0.4 --limbs 2 --severity 0 --hidden 256 --iterations 150 \
      --output artifacts/reach/chain-seed
... reach train --fine-error --hidden 256 --reward-tolerance 0.01 --limbs 2 \
      --severity 1 --severity-ramp 700 --still-weight 1 --roughness-weight 0.5 \
      --initial artifacts/reach/chain-seed --iterations 1000 --output artifacts/reach/chain

# Report, comparison page, animation, C export.
... reach report --actor net=artifacts/reach/student --output artifacts/reach/report.json
... -m robot_ai.visualize.reach --report ... --run net=... --output artifacts/reach/comparison.html
... -m robot_ai.visualize.animate --run artifacts/reach/student --report artifacts/reach/report.json \
      --gif docs/media/reach.gif --html artifacts/reach/replays.html
... -m robot_ai.control.c_export --run artifacts/reach/student --output export/reach_policy.c
```

## Things that will bite you

- **The GPU is 4 GB.** Two training jobs fit; three do not, and the third dies with an out-of-memory
  error mid-run. Per-limb runs use more. `--minibatch 256` halves the activation memory. Chrome takes
  ~250 MB of it.
- **`pkill -f` and `pgrep -f` match the calling shell's own command line.** Kill by exact PID, or you
  will kill the shell that is doing the killing (I did, twice).
- **Training-time evaluation uses the training severity** (fixed in 022c08e). Before that fix,
  healthy-stage runs were silently scored on fully defective robots, which made several healthy-stage
  results look like failures for hours and cost one run that was actually fine.
- **Warm-started PPO must reset its exploration** (`--initial` does this now). A distilled student's
  exploration head is untrained; left alone it explores at 0.5 of *rated* torque, which on an 18 N m
  base joint wrecks the warm start.
- **Exploration noise is scaled per joint by rated torque** (`ChainEnv.initial_std`), and the chain's
  smoothness penalty is charged on the policy's *mean* command, not the sampled one. Without the
  latter, PPO is rewarded for collapsing its own exploration.
- **Judge chain runs by exploration std and command roughness at iteration ~50**, not by success.
  Success stays at 0 for a long time in every run, good or bad; the std collapsing below ~0.08 or
  roughness above ~0.2 tells you it has failed hours earlier than the success number will.
- **`artifacts/` (1.4 GB) and `datasets/` are git-ignored**, so links from the docs into `artifacts/`
  are dead on GitHub. The exported controller is checked in at `export/reach_policy.c`.
- Old chain attempts are archived under `artifacts/reach/chain-v0` … `chain-v5` rather than deleted.

## Repository map

| Path | What |
| --- | --- |
| `src/robot_ai/sim/joint_model.py` | Per-joint motor, backlash, friction, encoder and current models (Warp) |
| `src/robot_ai/sim/arm_batch.py` | Fused two-joint kernel, ~1e9 robot-steps/s with all defects on |
| `src/robot_ai/sim/chain_batch.py` | General planar N-joint kernel, validated against MuJoCo |
| `src/robot_ai/sim/population.py` | Robot populations, mid-episode changes, neighbour pushes |
| `src/robot_ai/sim/reach_env.py`, `chain_env.py` | The reach task for one arm and for stacked limbs |
| `src/robot_ai/train/reach.py` | PPO, distillation, the tuned PID, the comparison report, the CLI |
| `src/robot_ai/train/limbs.py` | Per-limb shared policy with optional peek/message/identity |
| `src/robot_ai/control/computed_torque.py` | Classical chain teacher with perfect knowledge |
| `src/robot_ai/control/c_export.py` | Export a trained network to one C file |
| `src/robot_ai/visualize/` | Comparison page, learning curves, animated replays |
| `tests/` | 127 tests; the physics ones check the kernels against MuJoCo and the C export against PyTorch |

Everything older (`docs/PROTOTYPE_README.md`, `ROADMAP.md`, `STATUS.md`, `src/robot_ai/{models,data,
evaluate,baselines}`, `sim/native.py`) belongs to the September 11–13 imitation prototype. It is kept
for history and for the MuJoCo reference fixtures the new kernels are validated against; nothing in
the current line of work depends on it.

## Scope decisions on record

Arms only for now; walking deferred until the arm result holds up, then "a bigger network later".
Obstacles deferred until there are 3+ joints (a two-joint planar arm has almost no freedom to route
around anything). Multi-limb is the current phase, with the shared-feeling question as its point.
