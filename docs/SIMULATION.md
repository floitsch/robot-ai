# Simulation specification

The purpose is credible, fast experience for learning. A good rigid-body solver is
necessary but does not supply realistic motors, sensor failures, bending, or a
wobbly mounting by itself. Use explicit reduced physical models and validate each.

**Current implementation boundary:** the GPU arm is a custom NVIDIA Warp rigid-2R
kernel, with native MuJoCo as reference. The existing combined flex/base fixture
drives separate oscillators and changes endpoint geometry; it does not implement
the coupled mechanics specified below. Retain it as a diagnostic, not P8 acceptance.
See [REVIEW](REVIEW.md) and the recovery sequence in [ROADMAP](ROADMAP.md).

## Frames, robot, and initial task

Use SI units. World x is horizontal, z upward; the arm moves in the x-z plane and
joint axes follow a consistently documented y-axis convention. Define q=0 with
both links hanging down. For fixed-base geometry:

```text
x_tip = L1*sin(q1) + L2*sin(q1+q2)
z_tip = -L1*cos(q1) - L2*cos(q1+q2)
```

Joint-positive direction in the MJCF must match these equations; test, do not infer
it from a rendered camera view. Base pose maps base-relative geometry into world
coordinates. Goal frame defaults to world even when the base is initially fixed.

Nominal fixture:

| Parameter | Initial value |
| --- | --- |
| Link lengths | 0.30 m, 0.25 m |
| Link masses | 0.60 kg, 0.40 kg |
| Geometry/inertia | Slender capsules or rods, COM at midpoint; documented radius and inertia calculation |
| Nominal motor output torque caps | 6 N m, 3 N m |
| Joint limits | q1 in [-2.4, 2.4] rad; q2 in [-2.6, 2.6] rad |
| Gravity | z = -9.81 m/s² |
| Physics/control dt | 0.001 s / 0.010 s initially |
| Initial target deadlines | 1.5 s, then 0.8-2.0 s; integer control ticks |
| Endpoint tolerance | 0.010 m |
| Endpoint settling speed | 0.030 m/s |
| Hold after deadline | 0.20 s |

First healthy target set: sample configurations with q1 in [-0.8, 0.8] and q2 in
[0.3, 1.6], compute endpoints, and use the same elbow branch. Randomize starts from
that region. This avoids asking the first learner to solve singularities and
branch selection simultaneously. Gradually enlarge the region after baseline
success. Give any initial sensor warm-up an explicit duration; it is not secret
free adaptation time. Initially acquire at reset without an action warm-up.

Use analytic inverse kinematics plus a quintic joint trajectory with zero endpoint
velocity/acceleration and a PD + nominal gravity-compensation controller. Tune the
gains for the 100 Hz zero-order-hold command interface, then verify them on the
frozen development suite rather than relying on 1 kHz feedback behavior. It consumes
public observations. A separate truth-informed diagnostic controller may test
feasibility; label its results as privileged.

Feasibility is empirical at first: a successful constrained reference trajectory
demonstrates feasibility for that case; failure does not prove impossibility. Keep
unknown/hard cases separate rather than deleting them after a learned policy fails.

## Mechanical and actuator stages

Implement a parameterized MJCF model generator and one documented physical update
order. Warp is the training backend. Native MuJoCo runs the same fixtures as an
independent backend reference, not the source of perfect real-world truth.

### A. Ideal baseline

Explicit masses, inertia, joint limits, gravity, torque caps. No hidden position
servo when the command contract says torque. Disable irrelevant collisions only in
fixtures that explicitly exclude contact; do not disable limit forces to improve
throughput. Small-angle single-link period, static gravity torque, and energy
behavior should match analytic checks.

### B. Motor response, resistance, and load

Per physics substep, apply command delay, conversion to nominal torque, bounded
motor effectiveness and saturation, first-order torque response, then mechanical
forces. Specify whether a current-like measurement is before or after transmission
loss; use motor torque before gearbox output for the v1 current proxy.

Example first-order update with time constant tau_m > 0:

`motor_torque_next = target + (motor_torque - target) * exp(-physics_dt / tau_m)`.

For tau_m = 0 use the target directly. Maintain per-world actuator state. Avoid
double-applying damping/friction in both MuJoCo and custom kernels.

Friction initially uses viscous resistance and a smooth Coulomb approximation
`-f_c*tanh(dq/v_eps)`. It is not a complete static-friction model; label it as such.
Angle-dependent defects can add a localized nonnegative resistance magnitude or
reduce effectiveness around q0 with a smooth width. Resistance must oppose motion;
an arbitrary force bump can accidentally become an energy source.

Starting randomized ranges after the healthy stage:

| Effect | Starting range and persistence |
| --- | --- |
| Link length | ±15% from nominal; fixed per robot; public descriptor updated |
| True mass/inertia | ±20%; physically consistent; descriptor retains nominal values |
| Payload | 0-0.15 kg, at endpoint; per episode or explicit mid-episode event |
| Motor effectiveness | 0.6-1.0 per motor; persistent, then step changes |
| Motor time constant | 0-0.040 s |
| Command latency | 0-0.030 s |
| Extra viscous friction | 0-0.10 N m s/rad |
| Smooth Coulomb resistance | 0-0.15 N m |
| Angle-local resistance | 0-0.20 N m extra; width 0.05-0.20 rad |

These are synthetic curriculum settings, not specifications of the eventual robot.
Record waveform, timing, and affected joint. Introduce severe failures after the
moderate curriculum works; a locked joint often makes a target infeasible.

Payload changes must update inertia consistently and preserve a physically
declared momentum/attachment rule. Initial payload tests can be episode-constant.
Do not mutate only a displayed mass value without updating the dynamics model.
Backend-dependent model changes require recomputing derived model fields according
to the pinned MuJoCo/Warp API and a fixture proving their effect.

### C. Backlash with memory

Use a separate motor-side coordinate and output-joint coordinate with a transmission
gap. For an output-equivalent motor coordinate, let delta = q_motor - q_joint.
Outside a half-gap b, transmit an elastic torque proportional to
`sign(delta)*max(abs(delta)-b, 0)`, plus damping active in engagement. Apply equal
and opposite coupling generalized forces to motor and output coordinates.

Use positive motor inertia, nonnegative stiffness/damping, and consistent reflected
gear units. A non-unit ratio must transform both angle and reaction torque. Start
with ratio 1 in output-equivalent coordinates to avoid undocumented gearing.
Motor coordinates are hidden unless a sensor explicitly measures them; v1 encoders
are on output joints.

Use a smooth engagement approximation if required for stable integration, record
its width, and verify the effect does not disappear under refinement. A direction
dependent command deadzone may be a cheap diagnostic proxy, but does not satisfy
the physical-backlash milestone. Document rotor-body placement and parent reactions
when adding the extra coordinates to MJCF.

Acceptance fixtures: reversal lost motion, load on engaged teeth, gap crossing,
unpowered passive energy dissipation, and timestep convergence. Explore half-gaps
0.002-0.020 rad only after choosing stable stiffness/inertia. Stiffness is selected
from a desired mechanical mode and measured convergence, not an enormous constant.

### D. Bending and compliance (required)

Start with each link split into two rigid sections joined by a passive torsional
spring-damper. This is a reduced bending model: it stores energy, deflects under
load, and oscillates. It does not claim detailed continuum stress or arbitrary
beam shapes. Split mass/inertia consistently and place the elastic joint explicitly.

Represent elastic coordinates and rates in simulator truth, but do not expose them
directly to the policy. Encoder readings alone no longer determine endpoint pose;
the independent endpoint tracker and action history now matter.

Choose stiffness from a desired deflection: approximately k = applied bending
moment / desired angle for the fixture. Initially target 1-5 mm endpoint static
deflection under a declared test load, then 5-20 mm. Choose damping from a declared
damping ratio and effective inertia; start around 0.5-1.0 and include underdamped
cases later. Verify load-deflection and ring-down curves analytically for a single
elastic mode, then in the full arm.

More compliant modes, torsion, or continuum elements are extensions only if these
fixtures fail to represent an observed behavior. Do not start with finite elements
or a separate soft-body engine. Compile a small family of topology variants; do
not simulate rigidity by an unbounded spring stiffness that destroys the timestep.

### E. Unstable/compliant base (required)

Add planar base x/z translation and pitch, with finite base mass/inertia and
spring-damper attachment to the world. Arm reactions must move the base. Include
known equilibrium support/preload so gravity does not create an accidental
unbounded fall. Initialize consistently by solving/settling the support equilibrium
and recording that state; avoid unexplained teleports each control step.

First fixture permits pitch only. Then permit planar translation and pitch. Choose
support stiffness by desired displacement under a known force or moment, and damped
mode frequency. Include oscillation driven by arm movement, external impulses, and
slow mount displacement. Externally prescribed base motion and passive reaction to
arm motion are separate scenario types; label them accurately.

Measure base deflection relative to its declared support equilibrium, and expose
known operating ratings through descriptor v2 as specified in DESIGN. Keep actual
spring constants, hidden loads, and true pose out of the descriptor.

World-frame target accuracy is the primary metric here. A base-relative target is
a different task. The endpoint tracker must report in a world-anchored frame;
one rigidly attached to the wobbling base cannot by itself establish world position.

Observation schema v2 adds six optional base channels: measured world x/z/pitch,
pitch rate, and two base-frame accelerometer components. Accelerometers measure
specific force; declare gravity subtraction and frame rotation. Sensor availability
is configured, and estimates stay noisy. Test an endpoint-only condition too.
The schema change requires new model/bundle versions and retraining, not silent
padding of an incompatible checkpoint.

## Sensor simulation

Model acquisition, corruption, transport, and availability separately. At each
sensor's acquisition time sample the physical quantity, apply calibration/noise,
and enqueue a packet with acquisition/delivery timestamps. At a control tick expose
only delivered packets. Define packet ordering for variable delay (drop stale
out-of-order packets, retaining the latest acquisition time by default).

Starter sensor settings:

| Sensor | Period | Initial noise standard deviation | Persistent bias range | Delivery delay |
| --- | --- | --- | --- | --- |
| Joint angle | 0.010 s | 0.001 rad | ±0.020 rad | 0-0.030 s |
| Motor torque proxy | 0.010 s | 0.030 N m | ±0.050 N m | 0-0.030 s |
| Endpoint tracker | 0.040 s | 0.002 m per axis | ±0.005 m per axis | 0-0.100 s |

Velocity estimates are computed from noisy angle samples with a documented causal
filter and acquisition interval. Never substitute exact simulator dq into an
allegedly noisy derived sensor. Base sensor values/ranges are configured and
validated at the base milestone; choose them relative to measured base motion.

Add scale errors, quantization, drift, intermittent dropout, outliers, frozen
readings, and correlated errors incrementally. Bias is per sensor/per episode;
drift evolves with time; faults can begin during an episode. Which sensor is
reliable must vary. Do not always make the tracker the ground-truth oracle.

Create three categories: observable redundant sensing; degraded but usable sensing;
and intentionally ambiguous sensing. A common global offset with no independent
world reference is ambiguous. Mark its accuracy limit instead of claiming that
the model should infer unavailable information.

## Randomness, events, and fidelity

Use independent reproducible streams for geometry, dynamics, sensor errors, events,
and policy exploration. Define stream keys by scenario ID/world ID/time/channel,
not launch order. Save realized scenario parameters and acquisition timestamps.
Changing world batch size must not change a named scenario's initial conditions.

For cross-backend checks replay identical precomputed commands and disturbance
signals. RNG implementations need not be bit-identical across libraries. For learned
controllers' paired evaluation, match external scenarios/noise schedules; actions
and subsequent physical trajectories will naturally differ.

Each new physical effect needs: isolated fixture, unit/sign checks, dt versus dt/2
comparison, native/Warp comparison, visualization, and an assertion that its
parameters actually influence behavior. An effect implemented only in visualization
or reward is not a simulated mechanical effect.

Refine physics dt as needed for stiff modes while preserving the public control
interval. Do not speed up by changing the control contract unnoticed. Reduced models
are acceptable with a stated validity envelope; solver throughput alone does not
establish real-world fidelity.
