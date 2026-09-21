# Progress ledger

Updated: 2026-09-13, C3d-K CPU-reference candidate, corrected export, and one exported-only CUDA replay passed.

## Current handoff

**C3d-K candidate completed:** the frozen direct-distillation plan trained 10,880 successful
executed C3d-G actions on the 151-feature, 128-hidden shared-feeling policy. Host CUDA completed
3,000 fitting passes; the capacity record preserves the unrun CUDA update-3,000 observer evaluation.
The post-review engineering-equivalence audit passed all 64 update-1,500 cases. CPU-reference
selection chose update 1,500 SHA `aea45767b5a0286c50753af4a6892668d8cf24bff57ec33ea8e6ca9da1f750cc`:
64/64 development and zero hard limits. Matched examined-200 is 200/200 versus accepted C3c-5's
195/200, retaining every baseline-success ID and passing all primary, KE, slew, per-joint load,
saturation, settling, return, overshoot and travel gates. Aggregate KE .100670→.052486 J s;
median slew 41.1445→18.1131; settle 1.0025→.8005 s; return .167869→.022482 m.
Per-case tradeoffs remain: KE improves 197/200 (worst increase .0001823 J s, case 50), return 195/200
(worst .0124615 m, case 43), overshoot 189/200 with three regressions (worst .006001 m, case 41), and
slew 186/200. RMS joint load increases in 7/6 cases despite mean deltas −.143966/−.040046 N m. These
do not invalidate aggregate gates; they rule out a universal cleaner-motion or zero-overshoot claim.
Candidate maximum overshoot/return are .011907/.095411 m versus baseline .112011/.630430 m.

**Export acceptance:** `inference-bundle-cpu-reference/` carries world schema 4 with
`fresh_measurement_correction=true`, normalization, selected checkpoint/world/manifest hashes and
an explicit rigid physical scope. Clean CPU exported Runtime replay on actual causal examined-200
packets for cases 0/39 has feature/action difference 0. One memory-conscious exported-bundle-only
CUDA replay against saved development-64 CUDA case-0 actions also has difference 0, with
205,520,896 bytes free after bundle load. The initial examined/source-manifest-mismatched CUDA
comparison is preserved as historical and not used. CPU replays plus saved examined before/after
case 0/39 comparison and physical table are in `learning-progress-cpu-reference/`.

**One active next step:** C3d-K requires no further fitting, confirmation access, or reruns. Keep
the accepted C3c-5 bundle immutable as the confirmation-backed baseline. C3d-K healthy-motion
acceptance is complete on its already-examined set, with no unseen-set or adaptation claim; C4a/other
variants remain paused until the next user-directed research package.

## Historical evidence and preserved work

| Handoff field | Current value |
| --- | --- |
| C3c-1 manifest | `configs/healthy-packet-validation-v1.json`, 200 cases, seed 20260914, SHA-256 `c3164a90f49b890b882301794767a930ed6ea7e6004e5950ec8fe68ef0a46303` |
| Preserved incumbent | update-15 checkpoint `r5b4-packet-progressive-bc-30x4x50/best-policy.pt`, SHA-256 `483449...0f1` |
| C3c-1 complete evidence | Comparator audit `artifacts/p7/c3c1-current-packet-comparator-audit-20260913/metrics.json`: all 51 fields covered/action-responsive; histories, collector/runtime, selected replay and reload all 0 at 1e-6 |
| C3c-2 closed | `artifacts/p7/c3c2-decoded-feedback-anchored-update1-20260913/selection.json`: 60→60, failures 1/30/39/53, no hard violations; validation not run |
| C3c-3 closed | `artifacts/p7/c3c3-support-matched-decoded-50-20260913/selection.json`: updates 1/10/25/50 scored 60/6/19/25; no validation; each nonretaining checkpoint rejected |
| Follow-up evidence | Equal anchor gradient starts at 5e-8 versus decoded-label 0.285. The frozen decoded-feedback reference is 62/64, failing only IDs 1/30: it explicitly retains all 60 incumbent-success IDs and adds 39/53; it is a feedback reference, not learned-policy acceptance. |
| C3c-4 plan | `artifacts/p7/c3c4-full-decoded-distillation-25-20260913/plan.json`: 10,880 all-development incumbent-trajectory transitions, fresh Adam 1e-5, 25 passes, labels from frozen decoded feedback and 5x same-trajectory incumbent-action anchor |
| C3c-4 selection | `artifacts/p7/c3c4-full-decoded-distillation-25-20260913/selection.json`: update 25 selected; 61/64, failures 1/30/53, retains all 60 incumbent-success IDs, gains 39, no hard violations |
| C3c-4 validation | `artifacts/p7/c3c4-full-decoded-distillation-25-20260913/validation-200/metrics.json`: 185/200 (92.5%), failures 10/26/51/53/59/63/76/107/127/130/132/144/166/171/182, no hard violations; required >=190/200, so candidate is closed and not promoted |
| C3c-4 replay | `artifacts/p7/c3c4-full-decoded-distillation-25-20260913/learning-progress.{html,gif,json}`; selected-development and validation physical summaries are in `evaluation-summary.json` |
| C3c-5 diagnosis | `artifacts/p7/c3c5-examined-manifest-diagnosis/paired-summary.json`: 11 C3c-4 failures succeed under the same-observer decoded reference; 4 fail under both. All candidate failures are deadline/hold tolerance failures, with no hard violations. |
| C3c-5 coverage correction | `artifacts/p7/c3c5-coverage-decoded-distillation-50-20260913/`: disjoint training manifest SHA `3fe75c...65186`, 20,400 transitions, 50 passes; checkpoints 1/10/25/50 all 61/64 with failures 1/30/53, no hard violations. |
| Frozen confirmation completed once | `configs/healthy-packet-coverage-confirmation-v1.json`, 200 cases, SHA `109a1b...ecfe`; selected C3c-5 update 50 scored 194/200 with no hard violations at `artifacts/p7/c3c5-coverage-decoded-distillation-50-20260913/confirmation-200/metrics.json`. It is examined confirmation evidence, not an available tuning suite. |
| Accepted C3 candidate | `artifacts/p7/c3c5-coverage-decoded-distillation-50-20260913/`: update 50 SHA `e6f326...11530`; corrected clean-process Runtime export parity is `artifacts/p7/c3c5-export-runtime-audit-20260913/report.json` (170 ticks; feature/action differences 0). The older `bundle-reload-audit.json` is historical only. |
| Corrected export acceptance | `artifacts/p7/c3c5-export-runtime-audit-20260913/report.json`: clean CUDA Runtime packet replay, 170 ticks, action/feature differences 0 at 1e-6 after schema-4 correction; prior bundle audit is historical only. |
| Learning-progress acceptance | `artifacts/p7/c3c5-coverage-decoded-distillation-50-20260913/learning-progress-c3c5/`: inspected synchronized HTML/GIF; passing case 0 and failing case 1 selected replays retained. |
| C3d closed | `artifacts/p7/c3d-motion-quality-ke-20260913/selection.json`: all saved updates 1/10/25/50 were evaluated in both conditions. Zero-KE scores 60/60/49/32; KE scores 61/53/38/30. KE update 1 retains primary 61/64 but has 0.107600 J s versus matched zero-KE's 0.107488, so it fails the energy gate. No trained record passes every primary, matched-energy and slew gate; reject this teacher/distillation method, not kinetic energy generally. |
| C3d replay/diagnostics | `artifacts/p7/c3d-motion-quality-ke-20260913/learning-progress/` contains synchronized all-checkpoint replays for fixed passing case 0 and existing failure case 1, fitted-loss curves, and `physical-metrics.html`: aligned saved-trace error/speed/KE/command plots plus case labels. Per-case reports preserve KE, physical-rate safety, per-joint command/torque load, deadline/hold margins, settling, joint travel, endpoint return and target-axis overshoot. |
| C3d-F fit audit | `artifacts/p7/c3d-motion-quality-ke-20260913/fit-audit.json`: saved 10,880 policy-visited features show label MSE remains 0.09738/0.10999 after 50 passes; learned teacher-change projection is only 0.056/0.051. The varying zero-KE duration is linearly probed above its 0.656 majority baseline only with full initial observer features (balanced accuracy 0.817; task-only 0.464); this is availability evidence, not a causal/generalization claim. |
| C3d-F frozen mixture | `artifacts/p7/c3d-motion-quality-ke-20260913/mixture-diagnostic/report.json`: cases 0/15/24/39 under both saved schedules, CUDA inference/native-CPU physics, actual mixed actions/packets into the observer. Exact 5:1 mixture is 8/8 accepted-success retention, zero hard violations. It records teacher, accepted and mixed actions; deadline/hold, KE/slew and per-joint load/saturation; saved C3d student outcomes; and paired incumbent metrics. `conclusion.json` supersedes the old arbitrary >=0.15 projection gate without deleting its historical report. |
| C4a historical unmatched scratch diagnostic | `artifacts/p7/c4a-current-packet-matched-coverage-120-20260913/`: preserved only. Its fresh `PolicyNetwork`, plain label MSE, and omitted C3c-4 10,880/25-pass stage make it an unmatched scratch diagnostic, not a comparator for an adaptation/memory claim. |
| C3d-G frozen plan | `artifacts/p7/c3dg-constrained-slew-mixture-20260913/plan.json`; derived selection is `teacher-selection.json`: 9/64 changed labels, teacher KE 0.028337→0.027693 J s, mean slew 29.0290→29.0911, all per-case 105% slew constraints pass. Anchor is explicitly the original accepted policy queried on mixture-visited features; executed mixture action is separately stored and alone advances Runtime. |
| C3d-G collection gate | `artifacts/p7/c3dg-constrained-slew-mixture-20260913/{collection.json,anchor-contract-audit.json}`: shared KE-mixture development collection is 64/64, retains all 61 accepted-start IDs, and has zero hard violations. Stored original-policy anchor versus executed mixture action mean L2 is 0.039851. Fresh CUDA audit confirms original-policy anchor max difference 4.99e-7 and declared executed-mixture formula max difference 5.96e-8. |
| C3d-G fit/evaluation | `artifacts/p7/c3dg-constrained-slew-mixture-20260913/{training.json,selection.json}`: identical initialization SHA `857be...221c`, feature/anchor hashes, 200 passes and every pair 1/25/50/100/200 evaluated in host-CUDA session `28410`. Both update-1 records are 58/64, losing accepted IDs 15/39/58; later zero-KE/constrained-KE counts are 53/52, 46/44, 46/44, 49/48. All ten have zero hard violations. No record passes the primary gate; update-200 KE is 0.058702 versus zero-KE 0.058760 J s but fails primary and start-slew gates. |
| C3d-G artifacts | `learning-progress/{case0,case1}.{html,gif}` has initial plus both matched 1/25/50/100/200 trajectories; `physical-metrics.html` and `selection.json` retain deadline/hold, KE/slew, per-joint load/saturation, travel/return/overshoot and settling evidence. Shared collection is development evidence, not independent generalization. |
| C3d-H closed-loop evidence | `artifacts/p7/c3dh-student-output-diagnosis-20260913/diagnosis/{report.json,traces.npz,conclusion.json}`: Runtime and saved-evaluation action parity are both 0. Cases 15/39/58 fail at 1.501 s in both labels; students stay 0.00354–0.00403 mean L2 from accepted while intended mixtures depart 0.01352–0.05793. Per-joint observer q/dq errors are in rad/rad s; cross-label observer histories differ at most 0.000296 rad / 0.003563 rad s on the fixed cases. This supports output parameterization sensitivity, while case 0 prevents any universal residual-size claim. |
| C3d-I closed | `artifacts/p7/c3di-frozen-base-residual-20260913/{plan.json,pre-fit-audit.json,setup-repair.json,training.json,selection.json,learning-progress/}`: exact initial base-action/Runtime parity, shared init SHA `df32ab...b5c4`, and all ten frozen 1/25/50/100/200 host-CUDA evaluations (`scripts/project-run .venv/bin/python .../experiment.py evaluate --condition <zero_ke|constrained_ke> --checkpoint <1|25|50|100|200>`; no active process). Update 1 is 60/64 in both arms, losing case 15; later constrained-KE updates 100/200 lower KE by 0.000131/0.000127 J s versus equal-update zero-KE but are 47/64 and 51/64 and fail start-slew. Every record has zero hard violations; none passes every primary/energy/slew/safety gate. This closes only this residual parameterization. |
| C4a matched continuation | `artifacts/p7/c4a-matched-current-packet-continuation-20260913/`: selected 40/64 initialization copied byte-for-byte and 170-tick Runtime action parity is zero. CUDA fitting completed C3c4 10,880/25 and C3c5 20,400/50 labeled+5x-anchor stages with full batches; start/end combined loss is .008078→.007905 and .005723→.005415. Frozen initial is 40/64 and 133/200; C3c4 update 1 is 39/64. All checkpoint/suite evidence remains required before selection. |
| C4a CUDA capacity | Current `cuda-minimal-probe.json` records the renewed 17:25Z torch-only pass: 591,134,720 bytes free before context and 540,803,072 after the small backward pass. `cuda-minimal-probe-20260913-163408Z-historical.json` preserves the earlier 57.62 MiB/20 MiB allocation failure. Current host observation after fitting was 487 MiB free; report actual probe free values, not total-minus-used. |
| P8b-native r2 complete | `artifacts/p8b-native-coupled-pitch-base-20260913/` is preserved but superseded: it compared differing equilibria and had the shoulder at the pivot, so its endpoint result was not reciprocal evidence. `src/robot_ai/sim/pitch_base.py`, `tests/test_pitch_base.py`, and `artifacts/p8b-native-coupled-pitch-base-r2-20260913/report.json` correct this with a 0.12 m shoulder offset, base COM `(0,-0.08)` m, native state-dependent support and post-step state contract. Same `qpos=[.02,.1,.4]`, zero velocity and commands at 18/72 N m/rad change base by .11085 rad, relative q by [.12118,.24582] rad and world endpoint by .02965 m. All-body static residual is <=4.6e-13 N m; maximum unpowered energy increment is -9.1e-12 J and energy/work residual 1.3e-9 J. CPU mechanics only. |
| P7-native complete | `src/robot_ai/sim/backlash_native.py`, `tests/test_backlash_native.py`, and `artifacts/p7-native-backlash-transmission-20260913/report.json`: corrected moving-output reciprocal reference. Same-state reversal crosses +/slack/- flanks with 0.3879 N m peak output load; output backdrive makes 0.0940 N m rotor reaction and 0.9358 rad/s rotor speed. Rotor/output torque residual and native/formula output-torque difference are 0. Unpowered full energy has 0.000329 J maximum increment, 1.86e-6 J excess and 0.000688 J energy/work residual. 1 ms/0.5 ms output difference is 9.32e-5 rad and peak-load difference 1.80%. The older `backlash-fixture-v1.json` remains clamped-output history only. CPU mechanics, no Runtime rotor exposure or P7 learning claim. |
| Superseded C3d-K capacity handoff | CUDA update-3,000 observer evaluation remains unrun and its OOM record is preserved, but CPU-reference full selection/examined gates and selected export audits completed; it is no longer an active command. |

Replace this handoff as work advances. For an unfinished command, add its exact
invocation, output path and process/session identifier; resume it before relaunching.

### C3a complete — repaired acceptance replay

The supervision repair is complete. The exporter uses a shared contained viewport,
appends the exact 1.700 s hold-end frame, samples with `searchsorted` under
accumulated float time, and treats paired empty comparator estimates as unavailable.
The focused suite is 4/4. CPU-only export command:

```text
scripts/project-run .venv/bin/robot-ai export-packet-diagnostic --selection artifacts/p7/r5b4-packet-progressive-bc-30x4x50/metrics.json --trace 0:reference:artifacts/p7/c2-reference-case0/demo.npz --trace 0:initial:artifacts/p7/c2-initial-case0/demo.npz --trace 0:intermediate:artifacts/p7/c2-intermediate-case0/demo.npz --trace 0:selected:artifacts/p7/c2-case0-truth-teacher/demo.npz --trace 39:reference:artifacts/p7/c2-reference-case39/demo.npz --trace 39:initial:artifacts/p7/c2-initial-case39/demo.npz --trace 39:intermediate:artifacts/p7/c2-intermediate-case39/demo.npz --trace 39:selected:artifacts/p7/c2-case39-truth-teacher/demo.npz --output artifacts/p7/c3a-packet-replay-20260913-r2/packet-diagnostic.html --gif artifacts/p7/c3a-packet-replay-20260913-r2/packet-diagnostic.gif --report artifacts/p7/c3a-packet-replay-20260913-r2/report.json
```

The new immutable artifact is `artifacts/p7/c3a-packet-replay-20260913-r2/`:
`packet-diagnostic.html` SHA-256 `ff2813973d1a7be1d7318b4b1acbbd90d4245a7a0f8a4f9267c185fc214c93be`,
`packet-diagnostic.gif` SHA-256 `dfd8dcb17ec120d4be660dd2ce76af80d77af8062c2b02b40ca8fa07d4c49f10`
(1400x1450, 44 frames; final frame visibly labels t=1.700 s), and compact
`report.json` (3,443 bytes) with input/output hashes rather than trajectory copies.
The original `c3a-packet-replay-20260913/` remains preserved.

### C3c-1 complete and C3c-2 closed

`evaluate-packet-policy --case-manifest <path>` now accepts an explicit versioned
case manifest and preserves the frozen 64-case default. The C3c-1 host-CUDA audit
is accepted: all 51 public fields change features and action, equal-ending histories
agree without observer calls, and collector/runtime, selected-trace and reload
differences are 0 at 1e-6. Focused manifest/comparator/viewer tests and the full
suite passed (93 tests).

C3c-2 predeclared a fresh-Adam 1e-5, one-update correction using 340 closed
decoded-feedback labels for cases 39/53 plus 20,400 incumbent-action anchors. It
reduced teacher MSE 0.002186→0.002073 but retained 60/64 with the same failures
1/30/39/53; no hard violations, no validation. The support analysis records a
teacher-loop/incumbent-trajectory feature mismatch; it does not establish why the
candidate failed or diagnose an observer contract fault.
The C3c-3 collector completed its planned 50 passes; all saved checkpoints were
evaluated (1/10/25/50 = 60/6/19/25). Equal mean anchoring did not control physical
drift: its initial gradient was near zero and update-10's fixed case-0 action shift
was 0.0114 L2, changing hold error 3.63→18.33 mm. The frozen decoded-feedback
reference succeeds 62/64 (only 1/30 fail). C3c-4 then used that teacher across all
64 actual incumbent trajectories, with same-trajectory action anchors. Its selected
update 25 retained the 60 incumbent successes and gained 39 for 61/64, but the
frozen independent result was 185/200 rather than the required 190/200. The
candidate is closed; validation is now examined evidence for one distinct bounded
follow-up, not a tuning set.

The CUDA check passed on NVIDIA GeForce GTX 1650 (3,902,275,584 bytes reported).
The three bounded policy rollouts completed under the frozen update-15 checkpoint:

```text
scripts/project-run .venv/bin/robot-ai evaluate-packet-policy --run artifacts/p7/r5b4-packet-progressive-bc-30x4x50 --config configs/healthy-native.toml --checkpoint policy-update-000015 --device cuda --case-index <1|30|53> --diagnostic-truth-teacher --output artifacts/p7/c3b-frozen-policy-case<1|30|53>
```

Their traces feed `artifacts/p7/c3b-policy-visited-analysis.json`. The remaining
ten closed teacher loops completed through `scripts/project-run .venv/bin/python
artifacts/p7/c3b-frozen-loop-diagnosis.py`; the recorded results are
`artifacts/p7/c3b-frozen-loop-diagnosis/{traces.npz,metrics.json}` (120 arrays,
ten complete scores). Truth/decoded success is 2/2 for case 0, 0/0 for cases 1
and 30, 1/1 for case 39, and 0/1 for case 53. All runs use CUDA inference and
native CPU simulation; no training ran. C3c-1 and the first closed decoded-label
candidate follow from this evidence. Do not treat case 39's decoded-teacher success
as a general observer-refresh verdict, or 1/30/53 truth-teacher failures as
packet-policy impossibility.

## Preserved inputs for C3a and C3b

Each directory below contains `demo.npz` and its companion `metrics.json`.

| Condition | Case 0 directory under `artifacts/p7/` | Case 39 directory under `artifacts/p7/` |
| --- | --- | --- |
| Truth-state feedback reference (diagnostic) | `c2-reference-case0/` | `c2-reference-case39/` |
| Initial packet policy | `c2-initial-case0/` | `c2-initial-case39/` |
| Intermediate update 10 | `c2-intermediate-case0/` | `c2-intermediate-case39/` |
| Selected packet policy, same-state teacher labels | `c2-case0-truth-teacher/` | `c2-case39-truth-teacher/` |

Selection history: `artifacts/p7/r5b4-packet-progressive-bc-30x4x50/metrics.json`.
Only the selected checkpoint has renewed C1 limit evidence; label older curve
points as historical. C3a need not regenerate the training curve's full suite.

Incumbent: `artifacts/p7/r5b4-packet-progressive-bc-30x4x50/best-policy.pt`, SHA-256
`48344901bf137f41090a0ed12d6245f8842d49d665e0a4f72234df8bf6efb0f1`.
World bundle: `artifacts/p5/world-v12-r3a-schema4/best`.
C2 raw collector replay: `artifacts/p7/c2b-packet-collector-runtime-replay/collector-traces.npz`.
Existing analysis: `artifacts/p7/c2-state-action-analysis.py` and `.json`.
Its combined q/dq norm mixes units; recompute per-joint physical-unit errors for
C3b. Reuse its inputs and distinguish teacher-driven from policy-driven trajectories.

## Working evidence

These are recorded results. C1 checks the rigid fixture's modeled joint-position
limits; it does not establish unmodeled thermal, flex or base envelopes.

| Component | Evidence | Current interpretation |
| --- | --- | --- |
| Rigid reference | `artifacts/p2/c1c-computed-torque-native-limits/metrics.json` | 61/64, failures 1/30/53; zero physics-rate joint contacts. A reference miss does not prove impossibility. |
| Direct-observation learned control | `artifacts/p7/c1c-direct-healthy-warp-limits/metrics.json` | CUDA Warp 63/64, failure 53; zero joint contacts. The separate historical 197/200 direct result is not packet validation. |
| Best packet candidate | `artifacts/p7/c1c-packet-update15-native-limits/metrics.json` | 60/64, failures 1/30/39/53; median deadline error 2.616 mm, median integrated slew 50.138; zero joint contacts. Native CPU simulation, CUDA inference. |
| Direct policy on clean causal P3 packets | `artifacts/p7/r5b1-direct-policy-packet-healthy-64/metrics.json` | 8/64; healthy direct-state control does not transfer automatically to packet semantics. |
| World model | `artifacts/p5/eval/world-v12-r3a-schema4-validation-with-trail.json` | Historical covered prediction point gate passes. Accuracy on current policy trajectories remains unresolved. |
| C2 collector reconstruction | `artifacts/p7/c2b-packet-collector-runtime-replay/metrics.json` | Cases 0/39, CUDA: actual collector to fresh Runtime feature/action differences 0 at tolerance 1e-5. Parity, not observer accuracy. |
| C2 state/action diagnosis | `artifacts/p7/c2-state-action-analysis.json` | Early trajectory divergence; no unique root cause. Case 39 maximum hold speed 0.24450 m/s against 0.030 m/s, despite only 3.333 mm maximum hold error. |
| C3 one-update correction | `artifacts/p7/c3-on-policy-dagger-update1/metrics.json` | Fresh Adam 1e-5, 10,880 on-policy labels, one fitting pass: 60→59/64, new failure 42. Selected initial; exact candidate rejected, not all on-policy correction. |
| C3 current-packet comparator | `artifacts/p7/c3-current-packet-comparator-30x4x50/metrics.json` | 20,400 transitions, 1,500 fitting passes, selected update 15: 40/64. C3c-1 now verifies its 51-field current packet behavior, memoryless history contract, collector/runtime replay and reload. This remains a one-seed engineering comparison. |
| C3c-2 decoded correction | `artifacts/p7/c3c2-decoded-feedback-anchored-update1-20260913/selection.json` | Closed: one anchored decoded-teacher-loop update retained 60/64 but made no physical gain. The support diagnostic selects a distinct on-incumbent-trajectory decoded-label collector; it does not license a parameter sweep. |
| C3c-4 full decoded distillation | `artifacts/p7/c3c4-full-decoded-distillation-25-20260913/{selection.json,evaluation-summary.json}` | Development-selected update 25 is 61/64, retaining the incumbent 60 and adding 39. Independent frozen validation is 185/200, below 190/200. Candidate is closed and the incumbent remains current. |
| Old sensor-removal ablation | `artifacts/p6/r3b2-memoryless-bc-30-validation/` | Its 0/64 removes all sensor information and cannot establish memory benefit. |
| Transmission fixture | `artifacts/p7/backlash-fixture-v1.json` remains the clamped-output historical diagnostic. `artifacts/p7-native-backlash-transmission-20260913/report.json` is the accepted CPU-native moving-output reference: hidden rotor/actual arm output, reciprocal contact, both flanks, backdrive, energy/work and full-trace timestep evidence. It is not P7 adaptation/learning acceptance. |
| Flex/base fixtures | `tests/test_flex_base.py` | Isolated scalar modes, energy bounded by initial value, and terminal mixed-unit pose comparison. Not total-system passivity or the P8 convergence gate. |
| Current visualization | `artifacts/p7/c3a-packet-replay-20260913-r2/` | C3a repaired replay with contained geometry, exact hold-end frame and explicit unavailable estimates. |
| Adaptation / coupled physics / release | Historical P7–P10 diagnostics | No accepted end-to-end imperfect-robot controller. Required stages remain open. |

## Boundaries that prevent repeating old mistakes

- Case 39's largest command jump is at 0.22→0.23 s, after earlier divergence.
  The teacher on those same states jumps 1.0453 versus policy 1.0574. Comparing
  teacher and policy on different trajectories did not isolate a neural reversal.
- Action-delta candidates were evaluated only at final pass 100. Even the zero
  coefficient's first fresh Adam step disturbed fitting strongly. Preserve rejected
  checkpoints; no loss-family or architecture verdict follows.
- The new comparator has current sensor inputs. C3c-1 verifies all-field use,
  history independence and saved replay; its 40/64 comparison still does not isolate
  memory, pretrained representation or total training budget.
- Combined physics still evolves the arm independently of base/flex. Subtracting
  a constant weight from an identical constant vertical drive is not a solved
  coupled support equilibrium. Follow the independent P8b-native package if needed.
- PPO remains paused until its reward and physics-rate hold/limit contract match
  EVALUATION. That repair does not block frozen packet diagnosis or cloning work.

## Review provenance

The [September 13 review](REVIEW_2026-09-13.md) preserves the original findings and
adds the guidance follow-up. Its 79-test result belongs to that earlier source
state, not the current code. This follow-up read README and all docs, inspected
current source/tests and saved reports, and changed guidance only. No training,
GPU inference or runtime test suite was run for these documentation edits.

[Guidance audit](../artifacts/guidance-review-2026-09-13/audit.json) records saved
metric/trace checks, source hashes and documentation validation. Earlier guidance
is retained in `artifacts/guidance-review-2026-09-13/docs-before-guidance/`.
Historical reviews' future-tense assignments are superseded by ROADMAP's current
assignment; their experiment evidence remains available.

## User constraints

Shared learned feeling; frozen-weight adaptation; no privileged runtime inputs.
Physical motor/link/base limits, then target/deadline/hold, then smoothness.
Imperfect sensing, motors/friction, backlash, bending and unstable base remain in
scope. GPU access is **outside the sandbox**; use the project wrapper on the host
for CUDA work. Native reference physics and CPU rendering are explicit CPU tasks.
Keep environments, caches, outputs and temporary files project-local. Provide
synchronized learning-progress animation, including failures. New notices use
`Copyright (C) 2026 Florian Loitsch. All rights reserved.`
