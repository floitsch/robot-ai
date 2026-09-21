# Evaluation and acceptance contract

Metrics decide progress. A pleasing animation is evidence to inspect, not a
replacement for the fixed evaluation suite. Conversely, a scalar success score is
not enough without replayable examples showing what the controller did.

**Implementation boundary (2026-09-13 guidance follow-up):** C1 renewed
ordinary/direct/packet scoring with physics-rate joint-position contacts. This is
the rigid fixture's declared measured envelope; thermal/flex/base limits remain
unmodeled. The new current-packet comparator has sensor inputs, but its 40/64 versus
the recurrent policy's 60/64 does not isolate history from pretrained representation.
The old zeroed-state comparator remains sensor-blind. Independent packet validation
is still missing. Follow C3a–C3c in [ROADMAP](ROADMAP.md); historical reports retain
their original limitations in the [review](REVIEW_2026-09-13.md).

## Global goal

Train a reusable model that makes **cheap, imperfect robots practical to drive**.
It should work around backlash, flex, rubbing/scratching friction, wear, imperfect
motors, an unstable mounting, and unreliable sensors. It should reach commanded
positions on time and produce smooth movements within the robot's physical limits.
Its ability to compensate must be measured on imperfections and combinations it
did not simply memorize. Physical impossibility is reported, not concealed.

## Primary outcome

For a world-frame goal g and deadline T, define true physical endpoint error
`e(t) = norm(p_world(t) - g)` and speed `v(t) = norm(v_world(t))`.

An episode succeeds only if:

- No hard operating violation occurs at any physics substep.
- At T, e <= 0.010 m and v <= 0.030 m/s.
- Throughout [T, T + 0.20 s], both tolerances continue to hold.

These are initial prototype tolerances. They belong in versioned task configuration,
not scattered constants. Deadline and hold are integer control ticks; verify at
physics resolution so a brief overshoot cannot hide between sensor packets.
Score from simulator truth, never noisy observations or the model's estimate.
Log inferred success separately if displaying what the controller believed.

Do not terminate early when the arm first reaches the target. Record early settling,
but finish the declared deadline/hold test. Deadline failure is a natural task
termination; an external interrupted collection is truncation.

Require a finite, ordered trace covering the complete deadline/hold interval at
the declared physics resolution (or exact substep summaries). A trace ending at T
cannot prove the hold. Use integer ticks or an explicitly tested rounding tolerance
at interval endpoints. Early settling is the earliest in-tolerance interval of the
required duration, including times before T; the separate success test still checks
the whole [T, T + hold]. Test missing tails/gaps, not only complete trajectories.

Baseline and learned controllers act at the same configured control rate and hold
their accepted command through the same substeps. Timestamp physical states after
the step at the actual post-step time. Use the same task/scorer and recorded
scenario manifest for both. Different backends are acceptable for reference checks
with declared parity evidence, not different control opportunities.

## Operating envelope, including the base

Each robot/scenario configuration declares an envelope before control training.
Include actuator torque/speed/thermal limits where modeled, joint limits, link
deflection limits, and **base support forces, support moments, and base displacement
or rotation limits**. The base's spring forces are not the same as its strain limit.

For reduced models, force/moment/deflection are explicitly labeled load/strain
proxies. Material strain/stress requires section geometry and a material model;
do not invent it from a motor-current reading. Measure base reaction loads in the
declared support frame and specify whether values include static gravity support.
The safe envelope must contain the expected static supported load.

A motorized base adds motor limits; it does not eliminate mount/bearing/structural
limits. Check both. A passive base still has load limits even though it has no
command channel. Constraints come before deadline success, including when reaching
the target on time would require an overloaded base.

P8 defines a synthetic rated envelope using isolated load fixtures, nominal static
load, and documented headroom, then freezes it before policy comparisons. Initial
values are synthetic research settings, not real hardware ratings. Do not tune the
rating separately for each failed episode. Log absolute and normalized loads.

Let rho be the maximum utilization across declared limits. An optional soft penalty
begins at rho=0.7 and increases toward rho=1; a hard exceedance follows the declared
instantaneous or duration-based rule. For instantaneous force/deflection limits,
evaluate every physics substep. Motor temperature may have its own integrated
state and duration rule. Do not smooth away physical violations to improve scores.

## Training reward: explicit bounded stages

This is a tractable PPO surrogate, not a mathematical guarantee of optimal
lexicographic control. Final model selection always uses the primary metrics.

**Stage A: achieve deadlines.** For a nonviolating episode, integrate a bounded
distance cost: `-min(e/L_ref, 2) * dt / (T + hold)` with fixed positive L_ref=0.55 m.
At the final hold tick, add +10 for success. Otherwise subtract
`min(max_hold_error/position_tolerance + max_hold_speed/speed_tolerance, 5)`.
Include T in the hold maxima. Hard violation terminates with -10 and records its
category. End-of-episode distance shaping cannot contribute less than -2.

**Stage B: reduce strain and improve smoothness.** Keep the primary reward and add
a nonnegative bounded integrated cost totaling at most 1 per completed episode.
Split that budget between normalized command slew and soft load utilization,
initially 0.5 each. If appropriate, add true joint/endpoint jerk as a separately
normalized term while preserving the total budget. Do not use a single unbounded
jerk term that overwhelms deadline learning.

With undiscounted returns these bounds separate a successful episode's return from
a failed one's, but expected-return optimization still permits probability/error
tradeoffs. Do not claim a hard priority guarantee from reward weights. Reject a
smoother checkpoint that fails the primary noninferiority rule below.

Keep shaping, success, load, and smoothness metrics separate in logs. Derive jerk
and speed from true physical trajectories for evaluation; training can use truth
for reward only. The policy receives no truth through reward-feature wiring.

## Required recorded metrics

- Success fraction, deadline position-error median/p95, deadline speed median/p95.
- Maximum hold error/speed, settling time, failure categories.
- Hard violations per category, peak motor torque/speed and saturation duration.
- Peak base force/moment, peak base translation/pitch, peak flex deflection.
- Integrated command slew `sum(norm((u_t-u_prev)/dt)^2 * dt)`.
- Integrated joint jerk and endpoint jerk, with units and differentiation method.
- One-, five-, and twenty-control-step prediction errors in physical units.
- Observer state-estimation errors; optional uncertainty calibration only if modeled.
- After-event error/recovery curves, compared with no-event and memoryless controls.
- Training wall time, transitions, updates, steps/s, peak total VRAM, inference latency.

For a fixed dt, jerk may be estimated from finite differences of physical velocity
with the appropriate dt factors. Discard only explicitly documented boundary
samples. Compare at the same sampling interval. Do not compare raw difference
numbers across runs with different dt.

After-event settling time is not necessarily the instant the network detected a
fault. Label it as behavioral recovery, not direct measurement of understanding.

## Splits and scenario families

Split by robot/configuration and episode before window extraction. Use train-only
normalization. Scenario manifests contain immutable IDs, parameter seeds, initial
state, target/deadline, event schedule, sensor settings, and family labels.

Required families, introduced in the roadmap:

1. Healthy rigid arm, accurate sensors.
2. Noisy/biased/delayed sensing, observable redundancy.
3. Load, weak motor, friction, and angle-local rubbing.
4. Backlash and command reversal.
5. Link flex, including underdamped cases.
6. Base compliance/motion with load limits.
7. Held-out combinations of the above within trained parameter ranges.
8. Held-out dimensions/dynamics within the supported topology.
9. Out-of-range, severely failed, or deliberately unobservable cases, reported
   separately as stress tests.

Use distinct development/validation/final-test sets. Quick smoke: 8 scenarios;
development: 64; validation: at least 200 per active family; final: 1,000 per
reported family where practical, using three independent training seeds for a
research claim. If compute restricts this, report exact sample count and uncertainty;
do not present a quick run as the final gate. Include every evaluated episode.

Animations use a small fixed **demo set** separate from the final test set; it may
be inspected freely. Keep one representative failure/stress scenario in the final
demo if it remains unsolved. Do not select only successes after viewing results.

The historical CLI names are not sample-size guarantees: its 64-case “validation”
is a development regression set, and its examined 200-case “final” is historical
test evidence. Retain those cases. Freeze explicit new validation/final manifests
for promotion; do not replace difficult cases after seeing results. Changing count
in the current generator changes existing start/goal pairings, so count plus seed
is insufficient to preserve case identity across suite expansion.

## Prediction measurement contract (R1b)

For an anchor t and horizon h in {1, 5, 20}, compare against state at t+h using
exactly h recorded accepted actions. No observations after t enter an open-loop
prediction. Use actual recorded control dt and per-episode descriptor. Include the
last valid anchor; never cross an episode boundary.

Report both physical errors (q rad, dq rad/s, endpoint m and m/s) and this common
normalized state metric for every predictor and history condition:

`E_h = 0.5 * (norm(q_pred - q_true)/pi + norm(dq_pred - dq_true)/5)`.

Compute constant-position and constant-velocity predictions at **each** horizon;
CV holds dq constant and advances q by h*dt*dq. Declare the full q/dq behavior of
the constant-position baseline. Separate posterior estimation error from subsequent
prediction error. Use identical anchor sets and episode aggregation for comparisons;
record sample counts and errors by imperfection/source family.

Record the initialization contract before evaluating candidates. Report a baseline
from causal public measurements and an explicitly labeled oracle baseline from
true starting q/dq. The oracle diagnoses dynamics; it is not a deployable comparator.
A CV rollout from the model's starting decoded posterior is also useful to isolate
learned dynamics from observer error. Do not switch initialization between horizons
or choose the easiest comparator after seeing results. R1b must freeze the gate's
baseline identity and the previously unspecified “material one-step regression”
margin/uncertainty rule in evaluator metadata before candidate comparison. Preserve
the existing 20% improvement target and report all baselines.

For the locked R1b point-estimate gate, “material one-step regression” means more
than **10%** higher normalized-state error than the horizon-matched oracle
constant-velocity diagnostic. The candidate must improve 20-step error by at least
20% and stay at or below that one-step bound. This is an engineering checkpoint,
not a final uncertainty claim: later promotion still reports paired/bootstrap
uncertainty and a causal public baseline. This rule applies to R1b evaluations
created after this paragraph; older reports remain raw historical diagnostics.

History reset/shuffle diagnostics change the past information only. Keep the
current observation, action, descriptor, task time, and normalization fixed. A
shuffle must not inject future observations. For control ablations, preserve the
previous command and clock while changing only feeling history. Diagnose ambiguous
sensing separately; history cannot manufacture an unavailable absolute reference.

Before testing an observer change aimed at a particular sensor condition, report
its count by train/validation/test split. For the fresh-reading correction this is
the count of q/dq channels with `available && fresh && age_s <= 1e-6`, plus the
count of deliberately biased instances and their independent reference. A candidate
trained on zero examples of its declared condition is **inconclusive**, even if its
loss decreases or its weights match an older bundle. Fix collection coverage and
reconstruct a named episode before spending the next training comparison.

## Comparators and prototype gates

The numerical gates below are proposed engineering targets. P0/P1 may reveal a
mistake in a physical specification; revise with evidence before collecting a
locked benchmark. Once locked, do not change the target to make a model pass.

| Gate | Acceptance |
| --- | --- |
| Healthy ordinary controller | >=95% success on the fixed healthy feasible-validation set |
| Simulator convergence, rigid fixture | dt vs dt/2 endpoint discrepancy <=1 mm over the same 2 s command sequence; energy/gravity fixtures also pass |
| Flexible/base convergence | <=10% of position tolerance (1 mm) endpoint discrepancy; dominant frequency within 2%, peak support loads within 5%; document discontinuity-specific comparisons |
| Backend parity | Same fixture inputs; endpoint discrepancy <=1 mm and load differences <=5% on continuous-force fixtures; refine/diagnose impacts separately |
| Learned predictor | At least 20% lower 20-step normalized state error than constant-velocity baseline on the locked varied validation set; no material one-step regression; physical-unit breakdown required |
| Healthy learned controller | >=95% success on the same healthy validation suite, without truth inputs |
| Adaptive benefit | >=10 percentage-point success gain or >=20% p95 deadline-error reduction versus the matched memoryless policy on the predeclared degraded suite; no >2-point healthy success regression |
| Smoothness stage | >=20% median command-slew reduction on jointly successful paired episodes, with no >1-point success regression, no p95 endpoint-error regression >1 mm, and no increased hard violations |
| Runtime | Batch-one observer+policy p99 <10 ms for the 100 Hz prototype on the tested target machine, excluding viewer; full hardware loop later needs its own margin |

Always report the uncertainty of success-rate differences, preferably paired
bootstrap confidence intervals over scenario IDs and variation across training
seeds. If the allowed success regression is inside the uncertainty band, collect
more evidence or mark the comparison inconclusive. A point estimate alone does
not prove noninferiority.

Comparators: ordinary feedback; memoryless learned policy; history-stacked policy
if needed; adaptive policy; adaptive policy with feeling reset/shuffled as a
diagnostic ablation. Ablation degradation is useful evidence but does not replace
the matched baseline. Do not weaken the baseline deliberately to obtain a gain.

The memoryless learned policy receives current public packets including all
availability/freshness/age metadata and the same descriptor/task/command context,
with comparable capacity and training budget. It has no accumulated observer state.
Test that changing the current observation can change its features and actions.
Removing both belief and decoded state without supplying current packets is a
sensor-removal ablation; its failure does not demonstrate the benefit of memory.

Distinguish comparison of complete systems from attribution to history alone.
Matching policy parameter count and cloning passes does not match the recurrent
condition's pretrained observer/decoded representation or its pretraining budget.
Report those differences. A fair current-packet baseline is necessary for the
adaptive-benefit gate; a healthy one-seed point comparison is not that gate.

Missing hard-limit instrumentation must be explicit, not serialized as a verified
zero. A simulator clamp or soft joint stop does not replace recording limit
encounters. Retain each case's physics-rate trace or sufficient substep summaries
to reproduce success and failure, with exact identities for scenario, packet
contract, world model, policy, config, and evaluated source.

## Performance benchmarks

Measure physics-only, full environment (actuators/sensors/reset), observer+policy,
and complete collect/update throughput. Record compile/warm-up separately from
steady state. Synchronize GPU timing at measurement boundaries. Count **control
transitions** and **physics substeps** separately, with world count and dt.

Capacity sweeps include optimizer state, a representative recurrent training batch,
PPO rollout buffers, and all non-PyTorch allocations. Keep a margin against total
available VRAM. Do not extrapolate published high-end GPU benchmarks to the GTX 1650.
The goal is a credible measured local training budget, not a predeclared speedup.

## Artifact acceptance

Every experiment writes resolved config, package/hardware metadata, split/data and
bundle hashes, seeds, metrics, checkpoints, and failure notes under a unique local
run directory. Resume must restore optimizer/RNG state or declare a fresh run.
Stage-to-stage weight reuse must be recorded.

A final report includes aggregate comparisons and links to self-contained HTML/GIF
replays using the exact reported checkpoints. Export/reload the inference bundle
and rerun a subset; outputs must match within recorded numeric tolerance. Reload
cannot depend on an absolute path to a training dataset or an unrecorded environment.
