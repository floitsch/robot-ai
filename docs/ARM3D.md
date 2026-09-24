# 3D arms: "put the hand there"

Most real arms position a tool in 3D. This phase moves from planar arms to a five-joint arm laid out like
most desktop arms, and from joint-angle goals to tool-pose goals.

## The robot

At zero every link points up. Joints, from the base:

| Joint | Axis | Link length | Link mass | Rated torque |
| --- | --- | --- | --- | --- |
| base yaw | vertical | 0.08 m | 0.35 kg | 3 N m |
| shoulder pitch | horizontal | 0.25 m | 0.45 kg | 8 N m |
| elbow pitch | horizontal | 0.22 m | 0.30 kg | 5 N m |
| wrist pitch | horizontal | 0.07 m | 0.15 kg | 1.5 N m |
| wrist roll | tool axis | 0.09 m (tool) | 0.12 kg | 0.8 N m |

Every robot in a population varies around this: lengths ±15%, masses ±20%, an off-centre gripper, and
a payload of up to 150 g on half of them.

## Simulation

One fused GPU kernel (`src/robot_ai/sim/arm3d_batch.py`) steps each robot through a 100 Hz control
tick as ten 1 ms physics steps. Each physics step:
- forward kinematics with full 3D rotations;
- the mass matrix from the link Jacobians, with full inertia tensors;
- gravity, Coriolis and centrifugal torques by recursive Newton–Euler;
- a small dense solve with viscous damping treated implicitly;
- dry friction as bounded impulses.

**Checked against MuJoCo** (`tests/test_arm3d.py`): the joint angles and the tool point agree to 2 mrad
and 2 mm, both for a 1.7 s swing and for a violent fall from upright with all five joints coupled.

Every joint keeps the planar arms' defect model, with friction and drag scaled to the joint's motor (a
0.8 N m wrist servo has smaller gears and bearings than an 8 N m shoulder):
- weak or lagging motors, command latency;
- backlash;
- viscous, dry and sticky friction, including rough spots at particular angles;
- biased, noisy, quantized and delayed encoders;
- miscalibrated current sensing;
- mid-move changes: payload picked up, motor fading, joint fouling.

### New for 3D

Both of these come from the load each joint carries: the weight of everything beyond it, plus the forces
from moving it (accelerations included).

- **Load-dependent friction.** Cheap gear trains lose a share of the torque they carry, and a bearing
  that has to resist a tipping moment drags in proportion to it. A base turntable under an
  outstretched arm is much harder to turn than under a folded one (tested: 0.3 N m turns the balanced
  arm and does not move the unbalanced one).
- **Bendy links.** Printed and extruded links flex. Each link bends at its root in proportion to the
  bending moment there, and the bent shape is used for all dynamics and for where the tool really
  is. The bend follows the load (gravity and acceleration) without a vibration mode of its own.
  Tested against the textbook cantilever: a compliant shoulder link drops the tool by the predicted
  9 mm, within 10%. Accelerating the base makes an outstretched link trail behind.
- **Motor speed limits.** A motor's torque falls to zero at its no-load speed (4–10 rad/s at the joint,
  as for geared hobby motors); braking is always available.
- **Motor armature.** The motor's rotor, seen through the gearbox, is often heavier than the light
  wrist link it drives. Without it, no 100 Hz controller could hold the wrist roll still.

## The task

- **Goal:** a tool pose: where the tool point should be, which way the tool should point, and the wrist
  roll. Goals are sampled as poses the arm can reach, with the tool above the table.
- **Success:** over the last 0.3 s of a 3 s episode the *real* tool point, bends included, is within
  5 mm, the tool direction within 30 mrad, the roll within 30 mrad, and every joint has stopped moving.
  Half the episodes change the goal midway.
- **What the controller is told:** the arm's link lengths (what a maker reads off the drawing), plus
  the usual noisy joint, speed and current readings. No masses, no defects, no camera. A sagging link
  can only be inferred from how the joints feel.
- **Encoders:** assumed homed. An absolute zero error is indistinguishable from a goal somewhere else
  without an outside reference, so only a ±2 mrad residual remains.

## Results so far

A classical computed-torque controller, given the true masses, inertias and motor strengths of each
robot (knowledge no real controller has), measured on 4,096 robots:

| Robots | Success | Median tool error |
| --- | --- | --- |
| Healthy | 99.4% | 0.0 mm |
| Defective | 1.9% | 14.6 mm, 66 mrad |
| Defective, changing mid-move | 1.9% | 14.5 mm, 68 mrad |

Perfect knowledge of the rigid-body dynamics does not help against dry friction without integral action,
gear slack, latency, load drag and sag.

The controller a cheap-arm builder would actually write (`src/robot_ai/control/pid3d.py`) is told exactly
what the network is told. It does inverse kinematics to joint targets, follows a minimum-jerk reference,
and runs per-joint PID with anti-windup and gravity compensation from the catalogue masses. Its gains are
scheduled on each joint's nominal inertia and tuned on defective arms (bandwidth 20 rad/s, damping ratio
0.7, integral rate 0.1, 0.25 s per radian of travel).

| Robots | Success | Median tool error |
| --- | --- | --- |
| Healthy | 41% | 2.7 mm, 9 mrad |
| Defective | 3.3% | 54 mm, 215 mrad |

Even on healthy arms, half its failures are the three pitch joints still moving at the end. Per-joint
control ignores how strongly the shoulder, elbow and wrist pitch drive each other in 3D, and higher gains
saturate and oscillate. The network has to close that gap.

### Inverse kinematics: computed, not learned

Distilling that controller into a network first stalled at 15 mm and 5–7% success on *healthy* arms,
while the teacher scores 98%. There were two reasons:
- **Several solutions.** A tool pose can usually be reached in up to four ways: elbow up or down, and
  the base turned toward the goal or half way round with every pitch mirrored. The teacher aimed at
  whichever one had been sampled, which the student could not see. It now aims at the solution
  nearest the arm's measured pose when the goal appears.
- **Precise geometry.** To place the tool within 5 mm, the network had to learn the arm's inverse
  kinematics to about 1% of its reach, for arms of different sizes.

Inverse kinematics for this layout has a closed form. The wrist-pitch joint sits behind the tool point
along the tool direction, and the shoulder and elbow form a two-link arm in the vertical plane through
the base. It needs only the configured dimensions, like the forward kinematics the network already
receives.

The network is now also told how far each joint is from the nearest solution. It still sees the
Cartesian goal and the Cartesian error, and it still decides how to get there. That includes how far
to aim past the rigid solution so that a sagging arm ends up on target.

| Network, healthy arms | Success | Median tool error |
| --- | --- | --- |
| Distilled from computed torque, no kinematics hint | 7% | 15 mm |
| Distilled from computed torque, with the hint | 99.6% | 0.7 mm, 4 mrad |

The distilled network, which reads only the robot's imperfect sensors, has 0% success on defective
arms (median 38 mm). It is the starting point for training on them.
