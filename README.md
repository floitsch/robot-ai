# One controller for many cheap, imperfect robot arms

Cheap arms are hard to drive well. Gears have slack, joints stick at some angles, motors are
weaker than rated, encoders are coarse and late, and everything changes when the arm picks
something up. A PID tuned for one arm is wrong for the next one, and wrong again a year later.

This project trains **one small neural network** that drives *any* such arm without being told
anything about it. It gets the encoder readings, the motor current, and where you want the
joints to go. From how the arm responds it works out, within a fraction of a second, what kind
of arm it is holding: how slack, how sticky, how weak, how late. Then it drives it accordingly.
Its memory is a vector of 384 numbers that we call the arm's **feeling**.

![Same broken arm, two controllers](docs/media/reach.gif)

*Six arms from a held-out test set that the network never trained on, each driven by a PID with
carefully tuned gains (top) and by our network (bottom). The grey ghost is the pose the goal asks
for; green numbers are within tolerance.*

## What it does today

The arm is a two-joint planar arm in simulation, and the task is to reach commanded joint
angles and hold them. The simulator is the interesting part: it models thousands of *different*
arms at once, each with its own combination of

- slack in the gears (backlash), with a hidden rotor that has to cross the gap before the joint moves;
- dry friction that really sticks until the motor pushes hard enough, plus rubbing spots at particular angles;
- motors that are weaker than rated, respond slowly, and receive commands late;
- encoders with offset, noise, coarse steps, and delivery delay; a current sensor with its own errors;
- a payload picked up mid-move, a motor fading, or a joint fouling while the arm is moving;
- optionally, a neighbouring arm shoving it around.

It runs on a consumer GPU at about a billion simulated robot-steps per second, so the network
sees millions of different broken arms during training.

**Results on 4,096 held-out arms** (success = every joint within 30 mrad of its goal and at rest
for the last 0.3 s of a 3 s episode with a goal change in the middle):

| Arms | PID, gains tuned on these kinds of arms | **Our network** |
| --- | --- | --- |
| Healthy | 97.6% | **98.4%** |
| Defective | 25% | **83%** |
| Defective and changing mid-move | 22% | **83%** |
| ... and shoved by a neighbour (never trained on) | 13% | **52%** |

On defective arms the network's typical final error is 2.4 mrad; the PID's is 142 mrad. The
network also gets there about three times sooner. On a *healthy* arm it is exactly as precise as
the tuned PID (1.0 mrad against 1.1 mrad).

See [docs/REACH.md](docs/REACH.md) for the method, the full result tables, what helped and what
did not, and how to run everything; [docs/HANDOVER.md](docs/HANDOVER.md) for the state of the work,
the open multi-limb phase, and what to try next.

## Using the controller

The trained network is exported as **one C file that needs only `<math.h>`**: no framework, no
allocation, about 900k parameters (3.6 MB of float32 weights, 13 MB as source). Call it once
per 10 ms control tick:

```c
float feeling[384] = {0};      /* the network's memory of this arm; zero once at power-up */
float observation[12], command[2];
/* observation: measured angle/pi (2), estimated velocity/5 rad/s (2), motor current as a
   fraction of rated torque (2), goal angle/pi (2), goal - measured angle in rad (2),
   previous command (2) */
reach_policy_step(observation, feeling, command);
/* command: motor torque in [-1, 1] as a fraction of rated torque, per joint */
```

Nothing about the arm is configured. The file is checked in as [`export/reach_policy.c`](export/reach_policy.c)
and is regenerated from a training run with `python -m robot_ai.control.c_export`.

Caveats worth knowing before trying it on hardware: it is trained for a two-joint arm in a
vertical plane with torque-controlled motors and links of roughly 0.25 to 0.35 m and 0.3 to
0.7 kg; position-servo arms would need a different command interface. It has not been on a real
arm yet. The pushed condition (a neighbour moving the mount) is the weakest.

## How it is trained, in one paragraph

A first network is trained by reinforcement learning while being *told* the simulator's hidden
truth (the arm's real joint state and every defect parameter). It cannot be deployed, but it
learns what good control looks like on each kind of arm. A second network, which sees only what a
real controller sees, then drives the arms itself while the first one, watching the hidden truth
of the same moments, says what it would have done. The second network learns to reproduce that
from sensors alone. Along the way it also has to predict the arm's true state and hidden defects
from its own memory, which is what makes the memory a real "feeling" for the arm and, later, a
maintenance signal ("joint 2 is getting stiff around 0.4 rad").

## Several limbs on top of each other (in progress, not solved)

Mount a second arm on the tip of the first and the two stop being independent: the lower arm carries
the upper one's weight and inertia, the upper one's gravity load depends on the lower one's pose, and
whenever either moves the other's mount accelerates under it. That is the next thing this project is
after, because it is what a real multi-limb machine does and because it is the honest test of whether
a "feeling" can be shared between limbs.

![Two limbs stacked, four joints](docs/media/chain.gif)

*The same six defective robots as above, but with a second arm mounted on the first. Computed torque
(top) is a classical controller that is **told** every true parameter of the robot; our network
(bottom) is told nothing. The network is several times closer to the goal, and still outside the
30 mrad tolerance — the numbers stay red where the single-arm ones went green.*

The simulator handles it: one kernel now does planar chains of any number of joints, validated
against MuJoCo, with every per-joint defect as before. The control is not there yet. On a four-joint
stacked chain the best network sits at about 90 mrad typical final error against the same 30 mrad
criterion the single arm meets at 2.4 mrad. A classical controller *told* every true parameter of the
robot manages 99.7% on healthy chains and 14% on defective ones, so the difficulty is real and not an
artefact of the learning setup.

Three things we have learned that are worth knowing before touching this:

* Reinforcement learning from scratch does not get off the ground on the chain, though the identical
  recipe works on a single arm. Seeding it with a classical controller does: that is what took the
  healthy-chain network to 76%.
* One network driving all four joints is much easier to train than one shared network per limb, even
  when each limb is allowed to see the other's sensors (76% against 11% on healthy chains). What a
  limb seems to need from its neighbour is not its measurements but its intent — where it is about to
  go. Testing that properly is the next experiment.
* Nothing fails because of one particular defect. Remove any single defect class and the chain result
  barely moves; it is the *variety* across the population that the chain policy has not yet absorbed.

[docs/MULTILIMB.md](docs/MULTILIMB.md) has the experiment design and the full list of what has been
tried; [docs/HANDOVER.md](docs/HANDOVER.md) has the current state and what to do next.

## Repository

| Path | What |
| --- | --- |
| `src/robot_ai/sim/joint_model.py`, `arm_batch.py` | The fused GPU simulator for two-joint arms (NVIDIA Warp) |
| `src/robot_ai/sim/chain_batch.py`, `chain_env.py` | The same for planar chains of any length: stacked limbs |
| `src/robot_ai/sim/population.py`, `reach_env.py` | Robot populations, mid-move changes, pushes; the reach task |
| `src/robot_ai/train/reach.py` | Training (PPO oracle, distillation), the tuned PID, the comparison report |
| `src/robot_ai/train/limbs.py` | One shared policy per limb, with optional sensing or messages between limbs |
| `src/robot_ai/control/computed_torque.py` | Classical controller with perfect knowledge, used to seed chain learning |
| `src/robot_ai/visualize/` | Comparison page, learning curves, animated replays |
| `src/robot_ai/control/c_export.py` | Export to a single C file |
| `docs/REACH.md` | Method, results, how to run |
| `docs/MULTILIMB.md`, `docs/HANDOVER.md` | The stacked-limb experiment; the state of the work and next steps |
| `docs/PROTOTYPE_README.md` and the other docs | The earlier imitation prototype, kept for history |

Setup and everything generated stay inside this directory; see [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md).
More animations: `artifacts/reach/replays.html` (scrubbable replays) and `artifacts/reach/comparison-best.html`
(learning curves, result tables, joint-angle traces) are generated by the commands in docs/REACH.md.
The GPU used for all numbers above is a GeForce GTX 1650 (4 GB).

## What comes next

Getting stacked limbs to the standard the single arm already meets, which needs a better teacher for
the chain rather than more training. After that: limbs that tell each other where they are going,
more joints per limb, and eventually a real arm.
