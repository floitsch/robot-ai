# Review — 2026-09-12

Historical review. The [September 13 follow-up](REVIEW_2026-09-13.md) records
subsequent progress, remaining defects, and corrected experiment interpretations.
Use [ROADMAP](ROADMAP.md) and [STATUS](STATUS.md) for the current assignment.

The project has useful components, but its experiment loop is not yet trustworthy.
Several failures attributed to learning are contract or measurement errors. More
training, a larger network, or a new algorithm is not the next justified step.
Keep the shared learned feeling and existing stack; repair the evidence chain and
prove healthy control before continuing the imperfection curriculum.

Read the recovery assignments at the top of [ROADMAP](ROADMAP.md) for current work.
The findings below explain that order; they are not a request to fix everything
in one assignment.

## Scope and evidence

Reviewed README, all nine existing docs, runtime/training/evaluation/data paths,
physics and visualization implementations, tests, and representative saved reports.
No runtime source or weights were changed during this review. No training was
launched. The previous policy-v15 process has finished and left metrics/checkpoints.

`scripts/project-run .venv/bin/python -m pytest -q`: **40 passed**, 10 warnings
(NVML access, TorchRL extension compatibility, and deprecation). This was the
ordinary sandbox test suite, not host GPU acceptance. The host GPU remains the
place to run CUDA work; no driver or environment reinstall is indicated.

Reproduce the focused CPU arithmetic/native-reference diagnostics with:

```sh
scripts/project-run .venv/bin/python artifacts/review-2026-09-12/probes.py
```

[Results and source hashes](../artifacts/review-2026-09-12/probes.json) record the
reviewed source state. The [previous ledger](../artifacts/review-2026-09-12/status-before-review.md)
is retained as historical evidence. This workspace did not expose a Git repository;
source hashes provide attribution for these probes.

All 12 current guidance documents passed local-link and fenced-block checks.
[Documentation validation](../artifacts/review-2026-09-12/docs-validation.json)
also confirms runtime source hashes remained unchanged during this review.

## Findings, in priority order

### 1. Prediction comparisons do not implement the P5 gate

In [evaluate/world.py](../src/robot_ai/evaluate/world.py), `evaluate_world` scores
the learned model at 20 steps but constant velocity at **one** step. The latter
also starts from privileged true q/dq. Ordinary one-step error omits the q/pi
normalization used by history-reset/shuffle errors. Twenty-step reporting contains
only q, whereas the gate requires normalized q and dq. Constant position is absent.

On the 44 validation episodes in world-gpu-v3, the diagnostic oracle CV q errors
are 0.00812 rad at one step, 0.17146 at five, and **1.83236 at twenty**, using the
current evaluator's anchor selection. World-v10's saved twenty-step q error is
0.56388 rad. Comparing it with 0.00812 did not establish failure. Comparing it
with 1.83236 alone does not establish acceptance either: normalized state metrics,
initial-information fairness, provenance, and one-step behavior still need repair.

**Consequence:** withdraw the old P5 gate verdict; retain raw reports. Reevaluate
existing compatible weights before spending another 2,000-update training budget.

### 2. PPO trains a different task from the documented deadline task

[train/policy.py](../src/robot_ai/train/policy.py), `_collect_rollout`, constructs
and resets a new arm for every rollout. With the recorded defaults it terminates
after 128 × 0.01 = **1.28 s**, before the **1.5 s deadline plus 0.2 s hold**.
All final buffer entries are labeled `terminated`. Reward is distance/progress;
there is no documented terminal success/hold reward or operating-limit termination.
The learner never experiences the very behavior used to judge it.

The ordinary [baseline evaluator](../src/robot_ai/evaluate/baseline.py) recomputes
feedback every physics step (1 kHz); learned evaluation acts every control tick
(100 Hz). It also labels post-step samples with pre-step times. The historical
100% baseline result is promising feasibility evidence, but needs renewal under
the shared timing/scoring contract before serving as the matched P6 reference.

### 3. The scorer can accept an unobserved hold interval

[sim/scorer.py](../src/robot_ai/sim/scorer.py) checks that *some* samples occur in
the hold interval, not that the complete interval was recorded. A two-sample trace
at 0 and 1.5 s, already at the goal, passes despite missing the entire hold.
The settling-time loop only accepts times >= deadline while searching no further
than its first sample; a trace settled from reset reports 1.5 s instead of 0 s.

**Consequence:** add independently specified scorer fixtures before trusting more
success reports. A passing unit suite currently does not cover these cases.

### 4. Freshness is incorrectly treated as sensor accuracy

[models/world.py](../src/robot_ai/models/world.py), `observe`, forces decoded q/dq
to equal every available, fresh, zero-age reading in the physics-prior variant.
That path bypasses the learned correction. A newly timestamped 0.2 rad encoder
bias is copied exactly into q, regardless of the other sensors or history.
Freshness says when a reading arrived; it cannot say whether it is biased or stuck.

Measurement and constant-velocity residual connections are reasonable hypotheses.
An unconditional trust rule is incompatible with the project's changing-sensor-
reliability goal. Healthy ideal measurements may support a diagnostic comparator,
but cannot establish an adaptive observer. Observer semantics also changed while
v9/v10 both advertise schema 3; metadata does not identify that implementation.
Historical weights loaded by today's code are not automatically historical behavior.

### 5. Config names imply experiments that dispatch never implements

In [cli.py](../src/robot_ai/cli.py), `train-world` validates then discards its config;
world training/evaluation hardcode descriptor and 0.01 s dt. `train-policy` only
forwards the two timesteps from config and always builds a healthy rigid arm.
`evaluate_policy` also always runs healthy cases, including for combined configs.
`collect` does not dispatch on scenario; its native branch ignores supplied timing.
`simulate` advances one physics step per reported control step and can report
requested GPU/backend metadata even when an imperfect native branch actually ran.

`policy-smooth.toml` selects no Stage-B reward. Documented `--init-policy` and
`--resume` options do not exist. Fail clearly for unsupported experiments until
they are wired; a successful command with a misleading label is worse evidence
than an explicit unsupported-mode error.

### 6. P8 is a one-way driven fixture, not coupled arm/base mechanics

[sim/combined.py](../src/robot_ai/sim/combined.py) evolves a fixed-base rigid arm,
then separately drives flex oscillators and a base oscillator. Flex/base pose
changes the reported endpoint but does not alter the arm's mass matrix, gravity,
or generalized forces. Base force is constructed from torque/link length and a
constant weight, not derived as a coupled support reaction. Static equilibrium is
not solved before those forces are applied.

Doubling flex/base stiffness in a 500-step identical-command probe changes endpoint
position by up to **50.6 mm**, while arm q remains **identical**. Standalone spring
fixtures are useful; they do not prove reciprocal reaction, total-system passivity,
or the declared strain envelope. P8 needs coupled mechanics before combined learning.

The GPU arm in [warp_backend.py](../src/robot_ai/sim/warp_backend.py) is a custom
NVIDIA Warp rigid-2R kernel, not MuJoCo Warp stepping. Retain its measured rigid
fixtures, label the backend accurately, and test joint-limit encounters separately
from smooth trajectories. Its hard clamp does not establish native contact parity.

### 7. Adaptation comparisons do not isolate learned memory

[evaluate/adaptation.py](../src/robot_ai/evaluate/adaptation.py) calls an ordinary
trajectory-feedback controller “memoryless”; no matched learned comparator is
trained. The history ablation resets `FeatureAdapter.elapsed_s` as well as belief,
changing the deadline feature. Its three seeds are scenario/noise seeds for one
checkpoint, not three independent training seeds. Named variants start from a
combined fixture; the “reversal” label alone does not ensure a reversal occurred.

Zero-success comparisons remain useful failure traces, but neither these comparisons
nor the history ablation establish the P7 gate. Start from one observable imperfection
with a measured feasible reference and match all non-memory inputs.

### 8. Selection and data provenance do not support reproducible promotion

Both trainers name the last weights “best” without validation selection.
[Inference export](../src/robot_ai/models/inference.py) always exports final-policy,
and synthesizes timing/envelope metadata. Reports omit key source/config/data hashes;
world metadata records only `manifest.json`, not its dataset identity.

The current dataset's 245 “baseline” episodes are open-loop sinusoidal commands,
not the P2 feedback controller; the other 105 are bounded exploration. The manifest
has an empty config hash and lacks full realized descriptor/task/sensor/actuator
settings. This can be useful exploratory data, but it does not fulfill P4's claimed
70/30 feedback/exploration coverage or independent reconstruction contract.

### 9. The required learning visualization is still missing

[visualize/learning.py](../src/robot_ai/visualize/learning.py) produces a table of
links, not synchronized playback, intermediate checkpoints, or training curves.
It reads `item.score`, while policy reports contain aggregate fields and `scores`.
The [replay renderer](../src/robot_ai/visualize/replay.py) reconstructs a rigid arm
from q; HTML hardcodes lengths and ignores actual flex/base transforms and recorded
endpoint for error display. GIF frame duration is fixed instead of using sample time.
The current smoothness comparison pools failed episodes and omits the specified
p95-error/load checks. Neither artifact creation nor zero-success smoothness is P9/P10
acceptance.

## Direction

Recover in this order: authoritative measurement; matched healthy task loop;
useful learned prediction/control; a real progress comparison; one measurable
adaptation case; coupled flex/base physics and the remaining curriculum.

Preserve existing infrastructure and historical artifacts. Postpone architecture
search, longer training, Stage B, generalization claims, and final-release work
until their prerequisites hold. Independent deterministic fixtures and viewer work
can proceed without claiming downstream research gates.

Discuss a direction change with the user if corrected measurements and bounded
diagnoses still justify replacing the learning approach or changing the supported
physics/observations. The identified contract repairs do not require such a change.
