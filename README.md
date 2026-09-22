# Adaptive robot control

> **Current line of work:** [the adaptive reach controller](docs/REACH.md) — a fused GPU simulator of thousands of
> different imperfect arms, and one recurrent network that drives them without being told anything about the robot.
> Everything below this note describes the earlier imitation prototype and is kept for history.

**Global goal: compute a reusable model that makes cheap, imperfect robots useful
to drive.** The model should work around backlash, bending, rubbing/scratching
friction, wear, weak motors, mounting movement, and imperfect sensing, while
producing accurate, timely, smooth motion within motor and structural load limits.

Design and implementation plan for a robot controller that reaches a target by a
deadline, using a learned internal state (the robot's **feeling**) to adapt to its
body, actuators, and imperfect sensors. Among controllers that meet the positioning
requirements, prefer smoother movement.

The initial experiment is a two-joint arm moving in a vertical plane. The eventual
deliverable is a downloadable inference bundle: observer, predictor, controller,
normalization, and an explicit robot/sensor/command contract.

**Current state: C3d-K direct executed-teacher coverage candidate is fully packaged and audited.**
It distills all 10,880 saved C3d-G deployment-feature/executed-command pairs through the frozen
151-input/128-hidden shared-feeling interface. CUDA fitting completed the fixed 3,000 passes;
post-review engineering equivalence passed all 64 update-1,500 cases. CPU-reference selection
chose update 1,500 (64/64, zero hard limits), and the matched examined-200 result is 200/200
versus the accepted C3c-5 baseline's 195/200, with every primary and physical-motion gate passing.
Mean KE is .052486 versus .100670 J s; median slew is 18.1131 versus 41.1445. A fresh schema-4
bundle retains measurement correction; its clean CPU causal-packet replay has zero feature/action
difference on examined cases 0/39, and its exported-model-only CUDA replay matches the saved
source CUDA actions exactly for development case 0. The prior strict CPU/CUDA dq audit is retained
unchanged as historical evidence. C4a and other fit variants remain paused.

The `.004089/.027964` RMS/max fidelity result and unresolved broad transfer apply only to the
historical four-case C3d-J diagnostic, not to C3d-K's completed broader healthy-motion evaluation.
They do not establish a capacity limit. The earlier anchored and residual methods remain negative
results. See the [J diagnostic](artifacts/p7/c3d-small-teacher-executability-20260913/learning-progress/report.json),
the [C3d-K bundle](artifacts/p7/c3dk-executed-teacher-coverage-20260913/inference-bundle-cpu-reference/),
and saved [examined case-0 before/after replay](artifacts/p7/c3dk-executed-teacher-coverage-20260913/learning-progress-cpu-reference/examined200-case0-before-after.html)
and [case-39 replay](artifacts/p7/c3dk-executed-teacher-coverage-20260913/learning-progress-cpu-reference/examined200-case39-before-after.html).

Per-case tradeoffs remain: KE improves in 197/200 cases (worst increase .0001823 J s, case 50),
return motion in 195/200 (worst increase .0124615 m, case 43), and overshoot in 189/200; three
overshoot cases worsen (largest .006001 m, case 41). Candidate maximum overshoot/return are
.011907/.095411 m versus baseline .112011/.630430 m. Slew is worse in 14 cases, and RMS joint load
increases in 7/6 cases despite mean joint-load deltas of −.143966/−.040046 N m. The aggregate gates
pass, but these figures do not support a universal cleaner-motion or zero-overshoot claim.

The accepted C3c-5 controller remains immutable as the confirmation-backed baseline: 61/64
development, 195/200 examined, and 194/200 confirmation. C3d-K completes bounded healthy-motion
acceptance against its already-examined healthy set; it makes no unseen-set or adaptation claim.
C4a and other fitting variants are paused. Native coupled base and backlash reference work is
recorded separately; it is not learned-controller acceptance.

Start with the [current guidance](docs/ROADMAP.md#current-guidance--learning-approach-reassessment)
and [compact handoff](docs/STATUS.md). The C3d-K candidate bundle is reviewable; no further fit,
confirmation access, or C4a expansion is active.

## Reading order

1. [Stack and decisions](docs/STACK.md): language, libraries, hardware, and tradeoffs.
2. [Design guide](docs/DESIGN.md): architecture, interfaces, learning, and control.
3. [Simulation specification](docs/SIMULATION.md): mechanics, sensors, faults, timing.
4. [Evaluation contract](docs/EVALUATION.md): success, splits, benchmarks, and gates.
5. [Implementation roadmap](docs/ROADMAP.md): ordered work packages and recovery plans.
6. [Agent workflow](docs/AGENT_WORKFLOW.md): how to implement a bounded work package.
7. [Progress ledger](docs/STATUS.md): current work, evidence, and unresolved decisions.
8. [Local setup](docs/LOCAL_SETUP.md): keep every generated file/cache in this directory.
9. [Visualization](docs/VISUALIZATION.md): inspect movement and learning progress.

The installed implementation uses **Python 3.12, PyTorch, TorchRL, NVIDIA Warp,
native MuJoCo, and uv**. The GPU rigid arm is a custom Warp kernel; MuJoCo Warp is
the planned coupled-physics backend, not the current arm stepper. Start models and
simulation on the GPU, subject to measured capacity; native MuJoCo supplies a CPU
reference and explicit fallback. The GPU is accessible **outside the sandbox**.
These libraries are design recommendations, not individually user-selected dependencies. See
[the decision record](docs/STACK.md) before changing it.

The development machine was inspected on 2026-09-11: NVIDIA GeForce GTX 1650,
4 GiB VRAM, driver 610.57.04; Intel i7-7700, four cores/eight threads. The plan
must produce useful results on this machine. Larger training runs can later use
the same configuration and artifact formats on another machine.

Continue from the current recovery assignment in the roadmap and STATUS. Implement
one bounded substep; preserve working components and historical artifacts.

Visualization is a required deliverable: synchronized replays of untrained,
intermediate, and final controllers with deadline/error plots. See
[visualization](docs/VISUALIZATION.md).

All setup and runtime writes stay inside this directory, including interpreters,
environments, caches, temporary files, datasets, and reports. See
[local setup](docs/LOCAL_SETUP.md).
