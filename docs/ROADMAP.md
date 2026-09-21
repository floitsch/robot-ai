# Implementation roadmap

The destination remains a reusable controller for cheap, imperfect robots, using
one learned feeling for prediction and control. Physical constraints come first,
then deadline/hold success, then smoothness. A completed experiment may fail its
performance gate; it does not unlock a dependent learning milestone.

## Current guidance — learning-approach reassessment

**One recommended change:** directly distill the commands actually executed by
successful teachers, and expand case coverage under learner-controlled success gates.
The completed C3d-K test kept the established shared feeling, deployment inputs and
network architecture; anchor, residual, penalty and architecture variants remain paused.
Its selected bundle is reviewable evidence, not a new broad adaptation claim.

**What the bounded diagnostic established:** cases 0/15/39/58 supplied 680 saved
causal feature/action pairs from the successful C3d-G collection controller. A fresh
151-input/128-hidden policy learned those full executed actions with plain MSE,
then passed 4/4 deadline/hold tests while controlling its own physical simulation
and observer history. No case ID, simulator truth, teacher state or schedule was
added to policy inputs. Zero joint-limit violations were recorded. Mean kinetic
energy and command slew were 49% and 58% below the accepted controller on these
four taught cases. The accepted 194/200-confirmation controller is unchanged.

**What this adds:** earlier audits measured existing fits and replayed failing
policies. This intervention attempted direct small-case learning and demonstrated
successful execution. It supports trying direct teacher transfer at broader coverage;
it does not prove that anchors caused the earlier failures, the network is large
enough for all cases, or that the shared feeling is optimal.

**Historical C3d-J limits only:** the four-case, 5,000-step diagnostic transparently amended a
preliminary 200-step run; selected step 4,995 had RMS .004089/max .027964 normalized-action error,
missing .001/.01 targets while loss still decreased. Its exact fidelity and broad transfer remained
unresolved, and it did not promote a policy. Those statements do not describe C3d-K, which has the
separate broader healthy-motion result below.

**C3d-K bounded healthy-motion acceptance is complete:** the frozen protocol is
`artifacts/p7/c3dk-executed-teacher-coverage-20260913/plan.json`. It uses all 64 C3d-G
successful executed trajectories (10,880 rows), plain label MSE from accepted C3c-5, fixed
3,000-step microbatch-equivalent Adam fitting, and own-loop checkpoints 1/100/500/1,500/3,000.
The CUDA observer could not initialize update 3,000 with 56.5–77.4 MiB free. The separately
documented engineering-equivalence audit then passed all 64 update-1,500 cases without a decision
boundary; it did not alter the preserved strict dq-tolerance report. CPU-reference selection chose
update 1,500 at 64/64, and its matched examined-200 result is 200/200 against the accepted
baseline's 195/200 with every primary, KE, slew, load, saturation, settling, return, overshoot and
travel gate passing. No aggregation or confirmation access occurred.

**CPU-reference outcome:** the documented post-review engineering-equivalence audit passed all 64
cases without a decision-boundary case. CPU-reference update 1,500 was the only development-qualified
record (64/64), and its matched CPU examined evaluation is 200/200 versus accepted 195/200 with all
primary, KE, slew, load, saturation, settling, return, overshoot and travel gates passing. Saved
CPU replays are in `artifacts/p7/c3dk-executed-teacher-coverage-20260913/learning-progress-cpu-reference/`.
This establishes the CPU-reference physical result. The new selected bundle retains world schema 4
and `fresh_measurement_correction=true`; clean CPU Runtime replay gives zero feature/action difference
on actual causal packets for examined cases 0/39, and one exported-model-only CUDA development-case-0
replay gives zero action difference against the saved source CUDA trace. The source CUDA features were
not serialized, so this is an action-only CUDA export check; it is not a new broad CUDA evaluation.

**Per-case caveat:** on examined-200, KE improves in 197 cases (worst increase .0001823 J s, case 50),
return motion in 195 (worst increase .0124615 m, case 43), and overshoot in 189; three overshoot cases
worsen (largest .006001 m, case 41). Candidate maximum overshoot/return are .011907/.095411 m versus
baseline .112011/.630430 m. Slew worsens in 14 cases; RMS joint load increases in 7/6 cases despite
mean deltas −.143966/−.040046 N m. Aggregate gates pass, but this is not a universal cleaner-motion
or zero-overshoot claim.

**Coverage acceptance:** freeze the teacher, training
and evaluation case identities, deployment feature contract, fit budget and checkpoint
selection before execution. Train against actual executed successful commands. Report
per-joint/phase command error and convergence, then execute the selected learner in
its own causal loop. Compare teacher labels on both teacher-visited and learner-visited
states. Acceptance requires deadline/hold and physical-safety nonregression against
the incumbent on the fixed broader cases, reduced motion cost with visibly cleaner
trajectories, and retained command smoothness and joint-load limits. Supervised loss
or a fitted-step count alone cannot satisfy these gates.

**Wrong-turn routing:** unresolved fit at a still-improving budget calls for examining
optimization, inputs and representation, not a capacity verdict. Accurate teacher-data
fit with failed execution calls for checking accumulated errors and learner-trajectory
coverage. Success on the taught cases with measured broader failure calls for coverage
analysis before enlarging the network. Do not claim broad failure without measuring it,
or continue an open-ended sequence of fitting variants after another negative result.

Evidence: [C3d-K protocol](../artifacts/p7/c3dk-executed-teacher-coverage-20260913/plan.json),
[CPU selection](../artifacts/p7/c3dk-executed-teacher-coverage-20260913/selection-cpu-reference.json),
[capacity events](../artifacts/p7/c3dk-executed-teacher-coverage-20260913/capacity-events.json),
[CPU/CUDA parity](../artifacts/p7/c3dk-executed-teacher-coverage-20260913/cpu-cuda-parity/report.json),
[amended diagnostic](../artifacts/p7/c3d-small-teacher-executability-20260913/amended-report.json),
[physical comparison](../artifacts/p7/c3d-small-teacher-executability-20260913/learning-progress/physical-summary.html),
[fit curve](../artifacts/p7/c3d-small-teacher-executability-20260913/learning-progress/fit-curve.html),
[CPU export audit](../artifacts/p7/c3dk-executed-teacher-coverage-20260913/export-audit-cpu-reference/report.json),
[CUDA export attempt](../artifacts/p7/c3dk-executed-teacher-coverage-20260913/export-audit-cuda-deployment-attempt/report.json).
C4a and the completed mechanics packages are preserved in [STATUS](STATUS.md).
The completed-package narrative below is historical evidence, not concurrent work.

## Completed bounded package: C3d — motion-quality kinetic-energy comparison

**User-priority package, completed before C4a.** Starting from accepted C3c-5 update 50, compare
two otherwise matched decoded-feedback demonstration selections on the frozen original
development-64: a zero-new-KE control that ranks actual quintic (`hold_goal=false`)
teacher trajectories by primary feasibility then the existing command-slew measure, and a
KE condition that adds only exact rigid-arm `integral 0.5 dq^T M(q) dq dt` ahead of the same
slew tie-break. Both students use the same public packet/shared-observer interface, label-fit
plus existing same-trajectory action anchor, CUDA fitting budget and native CPU physics.
The plan and review amendment are in `artifacts/p7/c3d-motion-quality-ke-20260913/`.

Do not call lower KE accepted by itself. Selection must retain all 61 accepted C3c-5
development successes, have zero hard violations across all episodes, reduce aggregate KE
against the matched control, and keep median command slew within 5% of both control and
accepted start. Report per-joint load/saturation, deadline/hold margins, settling, joint and
endpoint excursion/return diagnostics and synchronized initial/intermediate/selected replay.
This was one bounded comparison, not a duration or coefficient sweep. It is closed:
teacher selection changed 22/64 cases and lowered mean teacher KE (0.02834→0.02602 J s)
while raising slew (29.03→36.49). All saved student updates 1/10/25/50 were evaluated:
zero-KE is 60/60/49/32 and KE is 61/53/38/30. KE update 1 retained primary success but
did not lower KE versus its matched zero-KE update. The record rejects this specific
teacher/distillation method, not kinetic energy as an objective. See
`artifacts/p7/c3d-motion-quality-ke-20260913/selection.json`.

## Completed bounded diagnostic: C3d-F — frozen action mixture

The saved C3d audit is complete. On the exact 10,880 accepted-policy feature/label pairs,
the update-50 students learn only 0.056 (zero-KE) and 0.051 (KE) of the required teacher
action change by projection, although the current label+5x-anchor pointwise objective permits
1/6. Full initial observer features contain more information about the varying zero-KE duration
than task-only fields (four-fold balanced accuracy 0.817 versus 0.464), which supports
availability but proves neither use nor generalization. See
`artifacts/p7/c3d-motion-quality-ke-20260913/{fit-audit.json,fit-route-followup.json}`.

The frozen replacement test ran `u=(5*u_accepted+u_quintic_teacher)/6` for cases
0/15/24/39 under both saved schedules, with the observer consuming that loop's actual
accepted actions and packets. It retained 8/8 accepted successes and had zero hard violations.
It therefore rejects neither stronger fit nor the loss mixture on the evidence of a pointwise
projection alone. It is a diagnostic, not a deployable mixture or all-64 acceptance. The old
`>=0.15` projection requirement is superseded; preserve it as historical provenance in
`fit-route-followup.json`. See `mixture-diagnostic/{report.json,conclusion.json}`.

## Completed bounded package: C3d-G — constrained-slew, trajectory-consistent KE distillation

Use only the saved two-duration C3d teacher candidates. Per case, keep primary-feasible
candidates at or below 105% of the matched zero-KE teacher slew, then choose minimum exact
rigid-arm KE. This changed 9/64 saved selections, reduced mean teacher KE
0.028337→0.027693 J s, and changed mean slew only 29.0290→29.0911. The one shared live
5:1 accepted/KE-teacher collection passed 64/64 with zero hard violations; its anchor was
the original accepted policy queried on mixture-visited features, never the executed mixture.
Both arms then shared the 10,880 collected features/anchors, initialization, architecture and
200-pass Adam budget; only their teacher labels differed. All ten snapshots were evaluated.
Neither arm retained the accepted 61-success set: both update-1 records are 58/64 and later
updates fall to 44–53/64. There were no hard violations. Constrained-KE update 200 narrowly
lowered mean KE (0.058702 versus 0.058760 J s) but failed primary and start-slew gates, so it
cannot be selected. C3d-G rejects this neural fit route; it does not reject KE generally.
See `artifacts/p7/c3dg-constrained-slew-mixture-20260913/selection.json`.

## Completed bounded diagnostic: C3d-H — saved closed-loop student-output diagnosis

The eight frozen student-controlled replays (cases 0/15/39/58 × both update-1 labels)
match saved Runtime actions exactly and have zero hard violations. Cases 15/39/58 fail at
the first post-deadline physics sample in both conditions. The students remain 0.00354–0.00403
mean L2 from the accepted policy on their own causal features, while the intended mixtures
depart 0.01352–0.05793. Observer q/dq error is reported in rad/rad s; the two conditions'
observer histories are nearly identical, so label-specific observer divergence is not supported.
Case 0 succeeds despite a large mixture residual, so no output-distance threshold is claimed.
See `artifacts/p7/c3dh-student-output-diagnosis-20260913/diagnosis/`.

## Completed bounded package: C3d-I — frozen-base residual action controller

The frozen accepted C3c-5 action plus a zero-final residual head passed direct action and
causal Runtime parity before fitting (maximum direct anchor difference `4.99e-7`, Runtime
case-0 action difference zero at `1e-6`). Both matched label arms used the same C3d-G
features, original-policy anchors, initialization (`df32ab...b5c4`), 200-pass budget and
saved 1/25/50/100/200 snapshots. All ten development-64 evaluations had zero hard
violations, but no snapshot retained the 61 accepted-start successes: the best was each
update-1 arm at 60/64, losing case 15. Later constrained-KE updates 100/200 lower exact KE
than their same-update zero-KE controls by `0.000131`/`0.000127 J s`, but score 47/64 and
51/64 and exceed the start slew gate. This closes the frozen-base residual parameterization;
it does not reject kinetic energy generally or justify extending either failed fitting schedule.
See `artifacts/p7/c3di-frozen-base-residual-20260913/{selection.json,learning-progress/}`.

## Completed bounded diagnostic: C3d-J — small executed-teacher transfer

Cases 0/15/39/58 use C3d-G's saved successful **executed** mixture commands and causal
deployment features, with the existing 151-feature/128-hidden policy only. No case ID, truth,
reset/start, duration, or teacher state enters the policy. The preserved 200-step result was
underoptimized. The amended 5,000-step run selected step 4995 at .004089 RMS/.027964 max error,
missing strict .001/.01 targets and still reducing MSE 1.58% over the final 200 steps. Exact
offline fidelity remains unresolved. Yet its own causal Runtime loops succeed 4/4 with zero hard
violations and 6.39e-5 mean teacher MSE. This establishes small-case pure-teacher transfer, not
broad transfer or capacity proof. The one next direction is a predeclared broader pure-teacher
coverage test with the same convergence and own-loop gates. See
`artifacts/p7/c3d-small-teacher-executability-20260913/{amended-report.json,amended-conclusion.json}`.
Its saved-data `learning-progress/` verifies the accepted update-50 file SHA is
`e6f326...11530`, exports synchronized accepted/teacher/learner comparisons plus a fit curve, and
keeps the missing accepted physical samples for cases 15/39/58 explicit. Case 39 learner slew is
58.89 versus teacher 24.01 and accepted 184.29; three learner holds are faster than their teacher.
The recommendation is direct distillation of actually executed successful behavior with explicit
fit and closed-loop coverage gates, not more anchored partial-retargeting or architecture variants.

## Completed independent package: P8b-native r2 — reciprocal pitch-compliant base reference

The original P8b artifact remains immutable historical evidence, but is superseded for a specific
geometry/comparison error: its soft and stiff paths started from different static equilibria and
its shoulder coincided with the support pivot. It cannot support a world-arm reciprocity claim.
The corrected native MuJoCo reference at `src/robot_ai/sim/pitch_base.py` declares a 0.12 m
shoulder offset and base COM at local `(0,-0.08)` m, uses native joint stiffness/springref/damping
at every RK4 substep, and returns state only after `mj_step` then `mj_forward`. `reset()` resets
time; reported passive support is therefore aligned with returned transforms, mass/energy state,
and time.

The r2 dynamic comparison starts both 18 and 72 N m/rad supports at exactly
`qpos=[0.02,0.10,0.40]`, zero velocity, and the same 10 ms commands. It changes base pitch by
`0.11085 rad`, relative q by `[0.12118,0.24582] rad`, and world shoulder/link-1/link-2/endpoint
by `0.01330/0.01419/0.01657/0.02965 m`. Static equilibria are separate and include base, link-1,
and link-2 gravity: independent all-COM force balance and MuJoCo holding force agree within
`4.6e-13 N m`. The unpowered trajectory has no positive energy increment (largest is
`-9.1e-12 J`) and its energy-plus-integrated-damping-work residual is `1.3e-9 J`; complete 1 ms
and 0.5 ms trajectories differ by `1.4e-10 m` at the endpoint. Source, tests, full traces and
an accurate common-world-viewport replay are in
`artifacts/p8b-native-coupled-pitch-base-r2-20260913/`. This completes only CPU mechanics
reference evidence; GPU support/backend parity, P8 learning, and C4 claims remain open.

## Completed independent package: P7-native — reciprocal moving-output backlash reference

The old `artifacts/p7/backlash-fixture-v1.json` remains historical: it clamps output and cannot
show moving-output gap crossing, reciprocal load transfer, or full-trajectory convergence. The
native replacement at `src/robot_ai/sim/backlash_native.py` has one hidden rotor inertia and one
actual gravity-loaded arm-output joint in a single MuJoCo world. A finite +/-0.015 rad gap engages
a 20 N m/rad flank spring plus 0.18 N m s/rad contact damping through MuJoCo's state-dependent
passive-force callback at each RK4 substep; the resulting rotor/output torques are equal and
opposite. Rotor q/dq remains trace-only mechanics state: the public output contract is q/dq.

The frozen same-initial 2 s +/-0.15 N m reversal reaches positive, slack, and negative flanks,
has 1.7547 rad output excursion and 0.3879 N m peak transmitted load. The gap changes 0.04470 rad
between opposite flanks. A zero-command output-motion backdrive produces 0.09397 N m rotor
reaction and 0.93584 rad/s induced rotor speed. The unpowered rotor+arm+gravity+flank trace has
maximum positive increment 0.000329 J, maximum energy excess 1.86e-6 J, and energy-plus-damping
work residual 0.000688 J. A 1 ms/0.5 ms full trace differs by 0.00150 rad rotor q, 9.32e-5 rad
output q, 2.79e-5 m endpoint, and 1.80% peak load. Evidence and synchronized rotor/output/gap/
torque replay are in `artifacts/p7-native-backlash-transmission-20260913/`. This is CPU mechanical
reference only; it does not complete P7 adaptation, Runtime integration, learning, or GPU claims.

## Paused assignment: C4a — observable imperfection with matched controls

The reviewed scratch C4a comparator is historical and unmatched: it began from a fresh
current-packet policy, omitted C3c-4's original-64 stage, and used plain MSE. The replacement
plan at `artifacts/p7/c4a-matched-current-packet-continuation-20260913/plan.json` starts from
the established selected 40/64 checkpoint, uses the already saved C3c-4 original-64 and C3c-5
disjoint-120 public packet traces, and mirrors their decoded-label plus 5x source-action-anchor
losses and 25/50-pass stage budgets. It freezes eight checkpoints, original-64 and
already-examined-200 ranking, and a four-case paired q1-bias event/no-event feasibility manifest.

The historical 16:34Z torch-only probe had 57.62 MiB free and failed allocating 20 MiB; it remains
preserved as minimum-context evidence. A renewed isolated 17:25Z probe passed with 591 MB free
before context and 541 MB after the small 51→164 backward pass. The continuation then passed raw
checkpoint equality and 170-tick CurrentPacketRuntime action parity at zero, and completed all
declared CUDA fitting: C3c4 10,880/25 and C3c5 20,400/50 transitions/passes. It did not need
microbatch accumulation. Frozen selection evaluation is incomplete: initial is 40/64 and 133/200,
while C3c4 update 1 is 39/64. No trained selection, event pilot, or adaptation claim exists.
The long 200-case evaluator exceeds the host-command time slice before writing a report; a
resumable checkpoint/suite entry point now preserves completed reports and does not rerun fitting.

**Historical C3-to-C4 sequence.** C3a, C3b and C3c-1 are complete. C3c-2 made no physical gain and
C3c-3 broadly regressed despite equal mean anchoring; both remain closed. C3c-4
distilled frozen decoded-feedback labels on all 64 actual incumbent trajectories,
with actions from those same trajectories as anchors. Its update-25 checkpoint
reached 61/64, retained every incumbent-success ID, and added 39. Frozen independent
validation was then 185/200, below the required 190/200. The development-selected
candidate therefore cannot replace the preserved 60/64 incumbent or complete C3.
C3c-5 used 120 disjoint training cases (20,400 transitions) and 50 passes. Every
trained checkpoint retained 61/64; post-validation development ranking selected update
50 at 195/200, and its one frozen confirmation was 194/200 with no hard violations.
C3 healthy acceptance is complete. Preserve all prior candidates and failures while
advancing to C4.

| Order | Bounded result | Completion means |
| --- | --- | --- |
| C3a — complete | Trustworthy replay of existing packet failures and learning checkpoints | Viewer checks pass; no learning claim |
| C3b — complete | Frozen-loop diagnosis and one evidence-supported next experiment | Failure layers narrowed; no single correction selected |
| C3c-1 — complete | Packet manifest support and comparator contract audit | Original 64 identities fixed; >=200 validation manifest distinct; comparator behavior/replay audited |
| C3c-2 — closed | Teacher-loop decoded-feedback anchored correction | Update retained 60/64 but made no physical gain; no validation run |
| C3c-3 — closed | Support-matched decoded-feedback correction | Update 1 retained 60/64; update 10 regressed to 6/64, stopping selection |
| C3c-4 — closed after validation | All-development decoded-feedback distillation | Update 25: 61/64 and retained incumbent set; frozen validation: 185/200 < 190/200 |
| C3c-5 — complete | Disjoint coverage correction and one-time confirmation | 61/64 retained; 195/200 examined; 194/200 frozen confirmation; selected SHA `e6f326...11530` |
| C4a — paused | Matched current-packet exposure and one observable-fault pilot | Preserved after fitting/partial frozen evaluation; C3d-K selection/export is complete, but C4a remains paused by the current user-directed scope |

Use [STATUS](STATUS.md) for exact preserved inputs and the next handoff. The
[September 13 review and follow-up](REVIEW_2026-09-13.md) explain the corrections.
Older assignments are historical, including the
[pre-review roadmap](../artifacts/review-2026-09-13/docs-before-review/docs/ROADMAP.md)
and [pre-guidance snapshot](../artifacts/guidance-review-2026-09-13/docs-before-guidance/docs/ROADMAP.md).
The original P0–P10 sections below remain scope specifications, not ten concurrent
assignments. Finish one substep, record its evidence, then advance this table's
current assignment. No permission is needed for that routine handoff.

### C3a — Make the existing failure visible accurately

**Target:** repair `src/robot_ai/visualize/packet_diagnostic.py` and its focused
verification. Reuse the saved case-0/case-39 reference, initial, update-10, and
selected traces listed in STATUS. Read VISUALIZATION's recording and acceptance
sections. This is a renderer/data-contract repair, not a new viewer framework.

**Accept when:** one offline comparison shows those checkpoints, selection curve,
and deadline/hold markers. Each signal uses its actual sample clock: physical q/dq
and speed are at 1 ms; packet/posterior samples and commands are at 10 ms, with
N+1 state samples for N commands. Read timing from trace/report metadata; reject
inconsistent lengths instead of guessing from array length. At 0.230 s, displayed
values must match the original samples (physics index 230, control index 23 for
these fixtures). Show both joints' true, measured, and estimated q/dq, accepted
and same-state teacher commands, and speed against 0.030 m/s. Signed values and
speed violations must fit labeled axes. Missing estimates are visibly unavailable.
Keep historical selection scores distinguished from C1's renewed selected score.

Export a new HTML and a headless GIF comparison under a unique
`artifacts/p7/c3a-packet-replay-*` directory, with input hashes and the reproduction
command. Check sampled values and time alignment independently of the renderer,
and inspect the rendered result. Artifact existence alone is not acceptance.

**Budget / wrong turn:** use saved data and CPU rendering; no training or full-suite
reevaluation. Reuse `export-packet-diagnostic --help` for the existing CLI; add only
missing export support. If packets or metadata are absent, first use the companion
report, then regenerate only a necessary frozen diagnostic trace on host CUDA.
Do not smooth away oscillation or fill absent estimates with truth. A mixed-rate
fixture and negative/over-limit values should expose the current bugs.

**Handoff:** record the verified replay and advance to C3b. A viewer pass does not
complete the control gate.

### C3b — Separate observation error from controller error

**Target:** use the frozen update-15 policy and world bundle. Start with failing
case 39 and passing control 0; summarize whether the explanation also fits IDs
1, 30, and 53. Reuse C2 collector replay and traces rather than repeating that
completed parity experiment. Relevant paths: `evaluate/policy.py`,
`control/runtime.py`, `baselines/controller.py`, and the C2 analysis in STATUS.

First inspect the policy-visited trajectory from reset through the early transient,
not just the largest jump at 0.22–0.23 s. Report q error in rad and dq error in rad/s,
separately per joint; use EVALUATION's normalized metric for a combined statistic.
Compare packet and posterior error against truth at aligned control times. Query
the same teacher with true q/dq and decoded q/dq at the same task time; compare
both actions with the accepted policy action. These queries diagnose sensitivity;
they do not predict a different controller's closed-loop success by themselves.

If needed, run the same feedback teacher in two frozen diagnostic loops: true
state, and decoded observer state. Each observer must receive its own loop's
accepted commands and resulting packets. Use the same initial state, teacher
schedule/gains, public command rate, and scorer. Truth-fed runs are oracle
diagnostics only. Do not splice true q/dq into a stale latent vector and call the
result a compatible packet policy.

**Accept when:** a short report and synchronized traces show onset, per-joint
estimation errors, same-state command differences, and full hold/limit outcomes.
Name the strongest supported failure layer and one intervention that tests it.
Use this routing as a default, not a claim of unique causality:

| Observation | Next bounded work |
| --- | --- |
| Packet/feature timing, reset, or scaling mismatch | Repair that contract and renew frozen evaluation; no training first |
| Truth teacher succeeds, decoded-state teacher fails with early estimation error | Check observer coverage on the actual policy trajectories; test one compatible world-model refresh using DESIGN's procedure |
| Decoded-state teacher succeeds where the policy fails | Isolate controller fit/sensitivity on the visited features; choose one controlled supervision correction |
| Even truth feedback fails | Check dynamics, saturation and teacher feasibility for that case; reference failure alone is not impossibility |
| Evidence remains mixed | State the two live hypotheses and one discriminating measurement; do not announce architecture exhaustion |

**Budget / wrong turn:** no training. Reuse existing traces, then at most three
controller conditions across the five named cases (15 complete rollouts). Model
inference is host CUDA; native simulation is explicitly CPU. Stop this package
once it selects a testable correction or identifies the precise remaining
comparison. Do not repeat the same diagnostic without new information. Smaller
teacher-path average error does not establish observer accuracy in closed loop.

### C3c — Test one correction and earn healthy acceptance

Deliver this stage in separate handoffs: manifest support, comparator contract
audit, then the selected correction and its evaluation. Preserve passed substeps
if the candidate fails. The whole stage is not a single-turn assignment.

**Target:** make one change justified by C3b while preserving the shared feeling,
public interface, task and incumbent. Routine bug fixes, data coverage repair,
and a versioned refresh already specified in DESIGN can proceed autonomously.
Discuss a concrete change to objectives, sensor visibility, action meaning, or
learning architecture only if the evidence actually requires it. Present the
proposed change, alternative and tradeoff; “objective or architecture?” is not a
usable decision request.

For a policy-fitting experiment, predeclare the hypothesis, initial checkpoint,
optimizer reset/resume, small first update, selection intervals and stopping rule.
Use no more than the prior policy budget (20,400 newly collected transitions,
1,500 fitting passes); this is a ceiling, not a target. A world-model refresh
instead uses P5's bounded budget and new bundle compatibility evidence. Recompute
policy features for changed observer weights; old latent features are incompatible.
Do not rerun the rejected DAgger schedule under a new name. A new experiment must
change a factor implicated by C3b, not merely its seed or learning rate.

**Accept when:** selected packet policy reaches >=61/64 on the unchanged fully
scored development suite, retains all incumbent 60 successes, and has no hard
violations in successful episodes. Report all failure IDs, median/p95 deadline
error and speed, maximum hold error/speed, and slew. Keep IDs 1, 30, 39 and 53.
Smoothness is secondary; the historical 39.928 slew ceiling is not this gate.

Before the next candidate selection, implement explicit manifest input for packet
evaluation and freeze a distinct >=200-case validation manifest. The current
packet evaluator hardcodes 64 cases: increasing a count/seed is not an implemented
validation path and may change old case identities. Test that the original 64
are unchanged and the new cases are distinct. Select on development; evaluate the
frozen selection against >=95% independent healthy success. The old direct-policy
197/200 result is not packet validation. An inspected validation set becomes
examined evidence for subsequent tuning, not an endlessly reusable unseen test.

Retain the current-packet baseline and audit it before promotion: the 51 fields
must actually include all public values/masks/freshness/ages/context; histories
ending at the same packet/time/previous command must give the same output without
calling the observer. Verify collector-to-runtime feature/action replay and saved
reload, not only shape. Record parameter counts, data/normalization/checkpoint
identities and optimization budget for both conditions, including the recurrent
condition's pretrained world model. Missing historical provenance is unknown;
newly computed hashes describe the audited files, not proven historical source.
Renew only affected evaluation after a repair; don't retrain just to fill metadata.

The recorded 60/64 versus 40/64 is a one-seed engineering comparison, confounded
with pretrained representation as well as history. It is not isolated memory or
adaptation evidence. Include comparator playback, initial/intermediate/selected
artifacts and exact selection evidence. Research adaptation still needs C4.

**Wrong turn / handoff:** the initial checkpoint fails to reproduce, a controlled
first update regresses, or offline loss improves without physical gains. Stop the
candidate under its declared rule, preserve the incumbent, and close that experiment
as rejected or inconclusive. Update STATUS with the new evidence and next diagnostic
or specific user tradeoff; never mark C3 healthy acceptance complete from a failed
candidate. Do not restart C3a/C3b wholesale when their evidence still applies.

### Completed recovery evidence — reuse, do not rerun by default

- **C1 measurement:** ordinary 61/64, direct 63/64, packet 60/64; all renewed
  with zero physics-rate joint-position contacts. Other physical limits are not
  modeled in this rigid fixture. Exact reports are in STATUS.
- **C2 reconstruction:** real saved collector packets replay through a separate
  Runtime on CUDA with feature/action maximum difference 0 against 1e-5. This
  establishes reconstruction on cases 0/39, not observer accuracy. The subsequent
  trajectory comparison identified an early divergence but not its unique cause;
  the old mixed-unit q/dq norm and faulty viewer do not close that diagnosis.
- **C3 experiments:** one-pass DAgger at fresh Adam 1e-5 regressed 60 to 59/64
  (new failure 42), so selection retained initial. The 30×4×50 current-packet
  comparator selected update 15 at 40/64. Preserve both runs. Neither completes
  healthy acceptance nor rejects an entire loss/architecture family.

### C4 — Demonstrate one adaptive benefit

**Prerequisites:** C1–C3 healthy packet evidence and a competitive current-packet
baseline. Verify that the selected world model is useful on the actual packet
condition, not only its historical prediction dataset. Pick one observable effect
and freeze its event schedule, magnitude, paired cases, and feasibility reference
before comparing adaptive benefit. A q1 bias step is a candidate, not a settled
result. Its derived velocity may change causally; do not demand that only channel
0 changes when velocity is derived from that encoder.

**Accept when:** the unchanged adaptive-benefit gate in [EVALUATION](EVALUATION.md)
passes against the matched baseline, with paired uncertainty, healthy nonregression,
three independent training seeds for a research claim, and an honest recovery
replay. Reset ablations preserve task time and previous command. Frozen weights,
not online retraining, must produce adaptation.

**Wrong turn:** every controller fails, the event is unobservable or has no measured
effect, or the comparison removes current sensor access. Resolve those conditions
before adding faults. If fair healthy comparison eliminates the apparent memory
benefit, design an observable history-dependent task; do not sabotage the baseline.

## Remaining physical and release work

Retain working P0–P6 infrastructure and historical evidence. PPO continuation is a
separate bounded research track, not a requirement to replace a useful cloned
controller. Before resuming it, reconcile physics-substep hold scoring and reward
components with EVALUATION; the repaired episode lifetime alone is insufficient.

After C4, complete the remaining P7 adaptation curriculum and P8a–P8d. P7-native has completed
the reciprocal moving-output backlash reference, but no learning/adaptation gate. The existing
combined arm/flex/base fixture drives separate oscillators; coupled reciprocal
mechanics, support equilibrium, energy/load, and timestep/backend evidence remain
required before combined learning. If packet work awaits a concrete user decision,
the next independent package is **P8b-native: a rigid arm attached to a pitch-only
compliant base in one coupled MJCF**. This implements the already planned topology;
it does not require a new architecture choice. Keep translations and link flex
fixed for this substep; version the model and retain the old diagnostic fixture.

Accept that native substep when identical commands with two support stiffnesses
change both base motion and arm q; nontrivial static equilibrium and support moment
agree with an independent force balance; and unpowered total-system energy plus
full-trajectory timestep comparisons satisfy the relevant EVALUATION bounds.
Save actual body-transform/load traces and a replay. Cancellation of a constant
weight, endpoint-only warping, or another isolated spring test is not reciprocal
mechanics. Native CPU is the declared reference for this package. Its completion
leads to host GPU support/capacity and backend parity for the same model, not P8
learning acceptance. Keep GPU simulation where measured capacity permits, and
record explicit fallbacks. Work on one package at a time.

P9 smoothness/generalization and P10 release retain their original acceptance
criteria below. The old synchronized viewer is reusable, but current learning and
failure traces must accompany each new stage. Do not promote smoke tests or artifact
existence into learned-performance, physical-load, or final-release acceptance.

## Original milestone specifications

The P0–P10 scope and numeric gates remain the destination. Commands below are
**target CLI contracts**; consult `robot-ai <command> --help` for implemented options.
All are launched through the project-local wrapper. Use `--config`, `--seed`,
`--device`, and `--output` consistently; reject unknown config fields and external
output paths. A command must print its resolved device/backend and artifact path.

## P0 — Local environment and actual GPU capacity

**Inputs:** STACK, LOCAL_SETUP, known GTX 1650/4 GiB. No physics/training prerequisite.

**Build:** `scripts/bootstrap`, `scripts/project-run`, pyproject/lockfile, doctor,
typed top-level config, minimal pytest/Ruff/mypy configuration. Run a released
PyTorch CUDA forward/backward workload, a MuJoCo Warp small-model step, and a
Warp/PyTorch shared-array round trip with documented ownership/stream synchronization.
Reserve representative model/optimizer/rollout memory when assessing capacity.

**Commands:**

```sh
scripts/bootstrap
scripts/project-run uv run robot-ai doctor --output artifacts/setup/doctor.json
scripts/project-run uv run pytest tests/test_setup.py
```

**Acceptance:** interpreter/venv/cache paths inside root; dependency compatibility
matrix and CUDA device identity recorded; actual CUDA kernels and gradients pass;
Warp and PyTorch agree on a modified test tensor; a 1-4-world workload fits without
leaking allocations across repeated runs; clean-cache output confinement verified.
Record cold compile time and total VRAM, including non-PyTorch allocations.

**Recovery:** driver inaccessible inside sandbox -> rerun the read/compute check on
the authorized host, not a driver reinstall. Unsupported wheel -> test a compatible
released wheel and record its index. OOM -> reduce worlds/batch and eliminate
rendering, then retest. If minimum Warp cannot run, record precise evidence and
use native physics plus GPU learning; do not falsely claim GPU simulation. Do not
start source-building frameworks or changing global software.

**Do not add:** neural architecture experiments, real hardware, cloud provisioning.

## P1 — Contracts, GPU arm, and physical reference fixtures

**Inputs:** P0; DESIGN/SIMULATION/EVALUATION. Add contracts before consumers.

**Build:** parameterized rigid 2R MJCF, Warp batch backend, native reference adapter,
action/observation/truth types, physics clock, reset semantics, seeded scenarios.
Implement forward kinematics, gravity/load checks, initial torque application, and
the task scorer. Build a TorchRL TensorDict environment adapter with explicit reset
masks. Keep truth in a separate record inaccessible to the policy adapter.

**Commands:**

```sh
scripts/project-run uv run robot-ai simulate --config configs/healthy.toml --steps 200 --output artifacts/p1/trajectory
scripts/project-run uv run robot-ai benchmark --config configs/healthy.toml --output artifacts/p1/benchmark.json
scripts/project-run uv run pytest tests/test_contracts.py tests/test_physics.py
```

**Acceptance:** analytic FK signs/units match; zero-command gravity behavior and
single-link period/energy fixtures pass; dt and backend comparisons pass the
evaluation tolerances. Reset one world without disturbing another. True and public
records are distinct. Scorer distinguishes pass-through from settled arrival,
checks the whole hold interval, and detects substep violations. GPU backend is
reported honestly and runs without per-world CPU copies.

**Recovery:** wrong motion -> inspect axis/units/inertia/torque before solver tuning.
Disagreement -> same precomputed actions/disturbances, same initial state, halve dt,
inspect constraint settings. GPU reset leakage -> isolate two worlds with different
sentinel states. No training until deterministic fixtures pass.

## P2 — Ordinary controller and first animation

**Inputs:** P1.

**Build:** analytic 2R IK, selected-branch quintic trajectory, feedback controller
with nominal gravity compensation; saved replay format and standalone HTML viewer.
Implement truth-informed diagnostic variant as a separately named controller.

**Commands:**

```sh
scripts/project-run uv run robot-ai evaluate --controller baseline --config configs/healthy.toml --suite validation --output artifacts/p2/eval
scripts/project-run uv run robot-ai replay --trajectory artifacts/p2/eval/demo --output artifacts/p2/arm.html
```

**Acceptance:** healthy baseline >=95% success on declared feasible suite; both
reachable IK branches validated mathematically; trajectory has declared endpoint
velocity/acceleration. Viewer opens offline, scrubs correctly, and draws actual
recorded positions. Target/deadline/error visible. Save an HTML and GIF fixture.

**Recovery:** poor baseline -> check torque feasibility, target branch, sign and
latency assumptions, then tune PD within documented gains. Do not begin RL to hide
a broken baseline or impossible task. OpenGL issue -> use Canvas/Pillow exports.

## P3 — Honest sensors and first imperfections

**Inputs:** P1/P2; baseline remains executable.

**Build:** acquisition/transport queues, bias/noise/age/freshness, derived velocity,
current proxy, independent tracker. Add load, weak motors, lag, friction, angle-local
rubbing, and scheduled changes. Each effect gets one isolated physical fixture and
one replay. Do not add backlash/flex yet.

**Commands:**

```sh
scripts/project-run uv run robot-ai simulate --config configs/sensors-and-motors.toml --steps 300 --output artifacts/p3/demo
scripts/project-run uv run pytest tests/test_sensors.py tests/test_actuators.py
```

**Acceptance:** delayed packets never arrive early; constant bias persists; dropout
and frozen-new packets behave differently; noisy velocity is causally derived.
Fault changes affect mechanics and are hidden from public observations. Repeated
scenario IDs reproduce external schedules independent of batch size. Document
which sensor configurations are observable versus intentionally ambiguous.

**Recovery:** all policies fail -> isolate sensing from mechanics; restore one
reliable reference; check whether the task is observable. All policies succeed
identically -> verify the imperfection actually affects force/readings. Never add
the true fault label to inputs as a shortcut.

## P4 — Sequence data and immutable splits

**Inputs:** P3.

**Build:** GPU collection from baseline and bounded exploratory commands; contiguous
per-episode shards; memory-mapped loader; split manifests; normalization; sequence
sampler. Start around 50,000 transitions, expand to 200,000 only after data checks.
Combine roughly 70% baseline trajectories and 30% smooth bounded exploratory
trajectories; include reversals and pauses, not just independent random torques.

**Commands:**

```sh
scripts/project-run uv run robot-ai collect --config configs/world-data.toml --output datasets/world-v1
scripts/project-run uv run robot-ai inspect-data --dataset datasets/world-v1 --output artifacts/p4/data-report
```

**Acceptance:** no episode/window crosses train/validation/test boundaries; actions
align with next-state targets; masks/timestamps preserved; full episode and sampled
window replay agree after reconstructed burn-in. A manifest records robot IDs,
scenario settings, policy source, schema, checksums, realized splits, and RNG seeds.
Dataset inspection shows coverage and finite values. Data are streamed, not placed
wholesale in VRAM. Reconstruct one named episode from its manifest.

**Recovery:** bad predictions later often originate here: inspect one transition
manually, then plot command/sensor/true-state timestamps. Fix data and bump dataset
version; do not overwrite a dataset referenced by existing results.

## P5 — Learned feeling and forward prediction

**Inputs:** P4; small architecture in DESIGN.

**Build:** dynamics GRU, observation correction GRU, physical decoder, normalization,
supervised sequence training, save/reload, multi-step evaluator. Train on CUDA.
Implement constant-position and constant-velocity prediction baselines.

**Commands:**

```sh
scripts/project-run uv run robot-ai train-world --config configs/world-v1.toml --dataset datasets/world-v1 --output artifacts/p5/world
scripts/project-run uv run robot-ai evaluate-world --bundle artifacts/p5/world/best --dataset datasets/world-v1 --split validation --output artifacts/p5/eval
```

**Acceptance:** overfit a small clean sequence batch before full training; multi-step
rollouts use only starting belief and recorded actions; held-out prediction gate
passes or is explicitly inconclusive. Reset/shuffle history diagnostics are
recorded. Reloaded outputs match. Show predicted versus actual endpoint trails in
the viewer. Zero observation correction produces visibly open-loop drift in cases
where correction matters; do not require drift in trivial cases.

**Budget/recovery:** initial 2,000 optimizer updates. If one-step fails: check labels,
normalization, gradient flow, masking, and small-batch overfit. If only multi-step
fails: inspect teacher forcing and hidden-state resets, then increase rollout loss
or data coverage. If bias adaptation fails: test burn-in 32 versus 128. Change one
factor at a time; at most three bounded comparison runs before documenting a new
design hypothesis. Do not automatically replace GRUs with a transformer.

## P6 — GPU policy training using the shared feeling

**Inputs:** P5 bundle, frozen. P2 baseline and healthy task are unchanged.

**Build:** CUDA feature adapter, TorchRL PPO/GAE training loop, memoryless comparator,
evaluation/checkpoint scheduling, initial/intermediate policy snapshots. Use the
architecture/defaults in DESIGN; policy training runs in actual Warp physics.

**Commands:**

```sh
scripts/project-run uv run robot-ai train-policy --config configs/policy-healthy.toml --world-bundle artifacts/p5/world/best --output artifacts/p6/policy
scripts/project-run uv run robot-ai evaluate --bundle artifacts/p6/policy/best --config configs/healthy.toml --suite validation --output artifacts/p6/eval
scripts/project-run uv run robot-ai compare --run artifacts/p6/policy --output artifacts/p6/learning.html
```

**Acceptance:** transformed-action log probabilities finite/correct at bounds;
GAE/termination test passes; observer unchanged during PPO; each real observation
updates feeling once; minibatches use stored rollout features, not re-advanced
observers. No simulator truth in policy/critic features. Healthy learned policy
gate passes, or the training failure is reported without moving the task goalposts.
Animated initial/intermediate/best comparisons exist even when learning fails.

**Budget/recovery:** 100,000 transitions initially, extend to 500,000 only if curves
justify it. No learning -> compare memoryless policy, check action scales/reward
sign, and validate existing PPO components against their minimal reference example.
Training success but evaluation failure -> check deterministic action convention,
normalization, reset, and scenario split. Memory-only issue -> inspect feature
utility and consider a recorded world-model refresh; discard stale PPO rollouts.

## P7 — Adaptation to sensing, motors, and physical backlash

**Inputs:** P6 healthy control, P3 moderate variations.

**Build:** progressive curriculum (one effect, then pairs, then combinations),
physical motor/output backlash model and its fixture, improved data coverage,
new world/policy bundles, and named mid-episode changes. Keep observation/action
semantics explicit when adding internal coordinates.

**Commands:** use the collect/train-world/train-policy/evaluate pipeline with
`configs/adaptation.toml`; write new dataset/run directories. Export comparisons
for load pickup, weak motor, sensor drift, reversal, and localized rubbing.

**Acceptance:** backlash passive-energy and reversal fixtures pass; full physics
dt/backend tests still pass; observer/policy runs with frozen weights at evaluation.
Predeclared adaptation gate versus matched memoryless baseline passes with uncertainty
reported. Held-out combinations and history ablation included. Report unreachable
locked-joint cases separately. Demonstrate reliable-sensor identity is not fixed.

**Recovery:** no adaptive advantage -> prove history matters in the selected physical
cases, compare against a history-stacked baseline, inspect model sensitivity to
actions and history. Do not degrade baseline quality. Instability at reversals ->
investigate contact stiffness/damping and timestep before RL changes. If longer
training simply memorizes combinations, stop adding budget and revise coverage.

## P8 — Bending, unstable base, and strain limits

**Inputs:** P7; remains a required stage, not an optional future idea.

**Build in substeps:** P8a one compliant link mode and static/ring-down fixtures;
P8b pitch-compliant base and support-load instrumentation; P8c planar base motion,
schema-v2 base sensors, world-frame task checks, frozen synthetic structural envelope;
P8d combine and retrain. Complete each substep's physics checks before learning on it.

Extend physical-output prediction to world endpoint position/velocity with a learned
correction to nominal rigid FK. Nominal FK is no longer the true endpoint model.
Use new bundle/schema versions. Draw actual flex/base transforms in replays.

**Acceptance:** static deflection and natural-frequency/ring-down tests; reciprocal
arm/base reaction; no unpowered energy injection; convergence/backend checks;
world-versus-base target fixture. Peak base support forces/moments and elastic
deflections are recorded and bounded. A fast candidate that overloads the base
fails the task even when its endpoint arrives accurately. Moderate imperfect cases
still meet the main adaptation/success gates on a declared feasible suite.

**Recovery:** numerical stiffness -> validate smaller physics dt and physically
appropriate damping, then reduce the retained elastic-mode bandwidth with a stated
validity envelope. Missing observability -> add the declared imperfect base sensor
channels and retrain; never pass true base pose as a hidden shortcut. Excess base
load -> adjust control or deadline feasibility, not the recorded structural rating.
Do not silently fall back to a perfectly fixed base in the combined suite.

## P9 — Generalization and smoother motion

**Inputs:** P8, all primary metrics and operating loads stable.

**Build:** fixed-family geometry variation, withheld combinations, Stage-B bounded
smoothness/load costs, checkpoint selection by primary metrics first, final
validation/ablation matrix. Profile full pipeline and tune batch size within 4 GiB.
No new learning algorithm unless the existing bounded diagnoses establish a need.

Write the candidate training run under `artifacts/p9/candidate` and record its best
validation checkpoint. Use the existing train-policy/evaluate/compare commands with
`configs/policy-smooth.toml`, a recorded P8 world bundle, and an explicit
`--init-policy` argument for the matching P8 policy. The init option restores policy
weights as a new experiment; `--resume` instead restores full optimizer/RNG state.

**Acceptance:** paired smoothness gate passes without unacceptable success/error/load
regression; report jointly-successful subset and overall failure rates. New robot
dimensions use public descriptors and fresh latent reset, not new weights. Record
performance by imperfection family rather than only a pooled average. Statistical
uncertainty and limits of extrapolation remain visible.

**Recovery:** smooth but late -> restore primary checkpoint, lower secondary budget,
inspect whether requested deadline is feasible. Faster but strained -> enforce
envelope and review control path. OOM -> reduce worlds/minibatch/unroll before model
precision, and rerun checks. Speed tuning cannot change dt or eliminate effects
without a separately versioned experiment.

## P10 — Downloadable bundle and learning-progress artifact

**Inputs:** completed prior evidence; failures may be documented, not erased.

**Build:** canonical inference API/bundle loader, final held-out evaluation,
standalone synchronized HTML plus GIF comparison, report, reproducible demo recipe.
Keep weights and large artifacts local; do not publish/upload as part of this task.

**Commands:**

```sh
scripts/project-run uv run robot-ai export --run artifacts/p9/candidate --output artifacts/release/bundle
scripts/project-run uv run robot-ai evaluate --bundle artifacts/release/bundle --suite final --output artifacts/release/eval
scripts/project-run uv run robot-ai compare --run artifacts/p9/candidate --output artifacts/release/learning.html
```

**Acceptance:** clean-process reload reproduces selected actions/replays; metadata
contains sensor/frame/action/timing and support envelope; parameter count/file size
and batch-one latency measured. Final report and animation show initial,
intermediate, and best policies on fixed demo cases, including flex/base and an
honest stress/failure case. Opens offline; GIF export works headless. All created
files remain inside root. No claim of real-robot validation.

## Beyond this roadmap — first real robot

Select hardware, available command mode, sensors, mounting, and rated loads before
specifying an adapter. Measure latency/friction/backlash/flex and sensing behavior;
calibrate simulator ranges; validate recorded hardware trajectories offline. Add
independent command/load limits and a hardware-specific stop mechanism. Start
restricted motion with conservative limits and compare observations/predictions.
Preserve frozen-weight adaptation as the first hardware experiment.

This is a subsequent project stage, not a reason to leave the simulation, trained
bundle, or required learning animation unfinished.
