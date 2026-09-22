# Adaptive reach controller

One network drives thousands of different imperfect two-joint arms. It is told
where each joint should go and nothing about the robot: no lengths, masses, motor
ratings, or fault flags. What it learns about the robot it learns by feeling it
move, in its recurrent state, with frozen weights.

This is the current line of work. The older documents in this directory describe
the earlier imitation prototype.

## Pieces

| File | Role |
| --- | --- |
| `src/robot_ai/sim/joint_model.py` | Per-joint motor, backlash, friction, encoder, and current-sensor models. Indexed `[world, joint]`; knows nothing about the mechanism. |
| `src/robot_ai/sim/arm_batch.py` | Fused GPU kernel: one launch advances every arm by a control tick. |
| `src/robot_ai/sim/population.py` | Samples robots, mid-episode changes, and external pushes. |
| `src/robot_ai/sim/reach_env.py` | The task: reach commanded joint angles; goals and the robot change mid-episode. |
| `src/robot_ai/train/reach.py` | PPO training, the tuned PID baseline, and the comparison report. |
| `src/robot_ai/visualize/reach.py` | Self-contained HTML comparison page. |
| `src/robot_ai/control/c_export.py` | Exports a trained network as one C file that needs only `<math.h>`. |

Per-step work never runs in Python. On a GTX 1650 the kernel advances about one
billion robot-steps per second with every imperfection enabled.

## What is simulated

Per robot: link lengths, masses, payload. Per joint: motor weakness, response lag,
command latency, viscous and dry friction with breakaway (true stick-slip),
angle-local rubbing, backlash with a hidden rotor, encoder bias, noise,
quantization and delay, and a current sensor with gain error, bias and noise.
During an episode a robot may pick up a payload, lose motor strength, or foul a
joint. Optionally it is pushed by smooth external torques, as a moving neighbouring
limb would push it.

Dry friction is solved as a bounded impulse, not a smoothed sign function, so a
joint really sticks until the applied torque exceeds breakaway.

## How it is trained

1. **Oracle.** PPO trains a recurrent policy that is told the simulator's hidden truth: the true joint state
   and the robot's hidden condition. It cannot be deployed. It shows what is achievable and serves as teacher.
2. **Student.** A recurrent policy that sees only what a real controller sees (encoder angles, a velocity
   derived from them, motor current, the goal, its previous command) drives the arms itself, while the oracle,
   looking at the hidden truth of those same states, says what it would have done. The student learns to
   reproduce that. Alongside, an *insight* head must name the true joint state and the hidden condition from
   the student's recurrent state. It forces that state to filter bad sensors and identify the robot, and it is
   the seed of a "this joint is wearing out" readout. Deployment ignores it.

Two details mattered more than any architecture choice. The goal error is also fed magnified and saturated
(`tanh(error / 0.05)`, `tanh(error / 0.5)`): a 30 mrad error is otherwise a 0.03 input among inputs of order
one, too faint to steer by. And the student has to be wide enough (256 units) to filter noisy, delayed encoders.

## Results

4,096 held-out robots that no controller trained or tuned on. Success is defined below. The PID's gains were
grid-searched (432 settings) on defective robots. Oracles see hidden simulator truth and cannot be deployed.

| Robots | Tuned PID | Memoryless net (PPO) | Recurrent net (PPO) | **Distilled student, 256 units** | Its oracle teacher |
| --- | --- | --- | --- | --- | --- |
| Healthy | 97.6% | 77.9% | 87.8% | **100%** | 80.0% |
| Defective | 25.2% | 35.8% | 50.0% | **75.6%** | 89.4% |
| Defective, changing mid-episode | 22.2% | 33.9% | 46.2% | **74.0%** | 86.5% |
| ... and pushed by a neighbour (never trained on) | 12.6% | 26.1% | 32.1% | **49.0%** | 71.2% |

Median final error on defective, changing robots: PID 142 mrad, student 8 mrad. The student also gets there
about three times sooner (mean error over the episode 0.09 rad against 0.30 rad).

Reproducibility: an earlier version of the student recipe (teacher trained against the plain 30 mrad
criterion) was trained twice from scratch with different seeds and scored 66.1% and 66.4% on defective,
changing robots, within one point of each other on all four rows.

What the numbers say:

* Memory helps. With recipe, seed and budget equal, the recurrent network beats the memoryless one everywhere.
* Mechanical faults are largely handled. With sensing and timing faults switched off at evaluation, even a
  64-unit student succeeds on 87% of robots with friction, backlash, weak motors and payload changes.
* What remains is sensing: noisy, coarse, delayed encoders and command latency, worst on low-friction arms where
  nothing damps a controller's reaction to sensor noise. The tuned PID fails on the same robots.
* Mid-episode changes (payload picked up, motor fading, joint fouling) cost the student under two points.

What mattered, in the order it was found:

1. Feeding the goal error magnified (see above): 34% to 46% for PPO.
2. Distilling from an oracle instead of training the deployable network by PPO alone: 49% to 55%.
3. Student width, 64 to 256 units: 55% to 66%.
4. Training the *teacher* against a stricter tolerance (15 mrad) than the 30 mrad success criterion. The first
   oracle parked robots just outside tolerance; the stricter one reaches 86.5% and lifts its student to 74%
   (600 distillation iterations; the last 300 added under one point).

What did not help: incremental (torque-change) commands (four times smoother, but ~28%), a stillness reward,
a finer error magnification, training on low-friction populations, and a teacher that knows the robot's hidden
condition but not its true state (the bottleneck is estimating the joint state through bad sensors, not
identifying the robot). Training on a population that runs from flawless to badly worn gives 1 mrad on healthy
arms instead of 4 mrad, at the cost of 16 points on the hardest robots (57.6%).

See `artifacts/reach/comparison-best.html` (generated) for learning curves and side-by-side traces.

## Running it

GPU work runs on the host through the project wrapper.

```sh
# Oracle teacher by PPO (about an hour on a GTX 1650; two runs fit side by side in 4 GB).
scripts/project-run .venv/bin/python -m robot_ai.train.reach train --fine-error --oracle full --hidden 128 \
  --reward-tolerance 0.015 --iterations 400 --output artifacts/reach/oracle

# Deployable student by distillation (about an hour).
scripts/project-run .venv/bin/python -m robot_ai.train.reach distill --teacher artifacts/reach/oracle --hidden 256 \
  --iterations 600 --output artifacts/reach/recurrent

# Memoryless PPO reference.
scripts/project-run .venv/bin/python -m robot_ai.train.reach train --fine-error --memoryless --output artifacts/reach/memoryless

# Compare against a PID whose gains are grid-searched on defective robots.
scripts/project-run .venv/bin/python -m robot_ai.train.reach report \
  --actor recurrent=artifacts/reach/recurrent --actor memoryless=artifacts/reach/memoryless \
  --output artifacts/reach/report.json

# Learning curves, results table, and side-by-side traces on characteristic robots.
scripts/project-run .venv/bin/python -m robot_ai.visualize.reach --report artifacts/reach/report.json \
  --run recurrent=artifacts/reach/recurrent --run memoryless=artifacts/reach/memoryless \
  --output artifacts/reach/comparison.html

# Deployable controller.
scripts/project-run .venv/bin/python -m robot_ai.control.c_export --run artifacts/reach/recurrent --output artifacts/reach/reach_policy.c
```

## Success criterion

Over the final 0.3 s of a 3 s episode every joint stays within 30 mrad of its goal,
measured on the robot's own encoder scale, and moves slower than 0.1 rad/s.
Evaluation uses 4,096 held-out robots that no controller trained or tuned on.

## Animations

```sh
# Side-by-side replays (GIF for the README, HTML with a scrub bar) on six characteristic held-out arms.
scripts/project-run .venv/bin/python -m robot_ai.visualize.animate --run artifacts/reach/recurrent \
  --report artifacts/reach/report.json --gif docs/media/reach.gif --html artifacts/reach/replays.html
```
