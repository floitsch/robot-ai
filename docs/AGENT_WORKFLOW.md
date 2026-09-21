# Implementing a work package

These instructions keep the design stable while leaving implementation choices to
the agent doing the work. They are not a demand to follow pseudocode mechanically.

## Start with one explicit assignment

Start with README, STATUS's current handoff, and the single current assignment at
the top of ROADMAP. Read the relevant contract sections and source files for that
assignment; STACK and DESIGN supply background when needed. After a continuation,
resume from the handoff instead of rereading all historical reviews and run logs.
The user's latest instructions take precedence over this guide. Do not restart
completed experiments or implement the whole roadmap at once. The
[current review](REVIEW_2026-09-13.md) and [September 12 review](REVIEW.md) explain
why downstream runnable commands do not imply completed prerequisites.
No new long training run until the relevant evaluator and task-loop repairs have
acceptance evidence. PPO reward repair is a prerequisite for PPO, not for frozen
clone diagnostics or viewer repair.

A useful assignment contains (a few lines usually suffice):

```text
Package: P<n> / optional substep
Objective: one concrete runnable result
Prerequisites: artifact paths and passed gates
Expected edits: a small set of modules/configs/tests/docs
Out of scope: explicit adjacent work to defer
Acceptance: commands, physical/statistical checks, required visible artifact
Budget: short smoke run, bounded initial experiment, escalation condition
Handoff: exact result paths, metrics, failure evidence, next action
```

Within that assignment, choose clear helpers and tests using normal engineering
judgment. Do not ask the user to select internal variable names or routine library
calls. Update design documents if a clarified contract affects other modules.
Keep the plan short: intended behavior, acceptance evidence, and the main sign
that the approach is wrong. The template is a prompt for judgment, not a requirement
to write a long checklist. Discuss objective/architecture tradeoffs when evidence
leaves the direction unclear; implement ordinary contract repairs autonomously.

## Invariants

- GPU-first execution; CPU reference/fallback is explicit and evidenced.
- GPU access is on the host outside the sandbox. Use the same project-local wrapper.
- Public sensor/task/descriptor/action features are distinct from truth labels.
- Feeling is learned state; it is not an array of manually assigned fault labels.
- Frozen weights during adaptation evaluation. Recurrent state can change.
- Target/deadline/settling first, smoothness second, declared physical limits always.
- Include motor and structural/base loads. A powered base is not exempt.
- Project-local environment, caches, temp files, data, outputs; no host pollution.
- Every training stage records checkpoints usable in the learning-progress animation.
- Preserve units, frames, timing, action meaning, schemas, and reset semantics.
- No unsupported performance claims, real-robot claims, or silent weakened gates.

## Evidence before extra complexity

Run unit/contract fixtures first, then a tiny end-to-end smoke, then the bounded
experiment. Fix the narrowest reproduced failure. For stochastic learning, keep
seed/config fixed while isolating a suspected cause, then use independent seeds
for conclusions. Do not rerun seeds until one looks good.

When things go wrong, record: expected behavior; actual metric/trace; smallest
reproduction; likely failure layer; one proposed change; result. The order to
investigate is usually contracts/timestamps/units -> physical feasibility -> data
visibility/coverage -> optimization -> model capacity. Evidence may justify another
order; write why.

After the package's bounded recovery attempts, document the remaining hypothesis
and the specific decision needed. Keep unaffected work progressing. Do not replace
the stack, rewrite the architecture, change sensor truth exposure, or lower test
thresholds merely to mark the package complete.
Repeated zero success or a falling loss without improving physical outcomes calls
for a smaller diagnosis, not another renamed run. A failed gate may be a completed
experiment, but it does not complete the learning milestone or unlock dependent
curriculum stages. Deterministic fixtures and visualization can still progress.

## Work within the context budget

Choose a substep that can leave one reviewable result before the context is spent.
For example, finish the replay's clock/axis repair and its fixture before starting
controller interventions. Keep large arrays and logs in artifacts; inspect a named
case or a compact summary. A broad research gate can span several such handoffs.

Before a long command, record its exact invocation and output directory in STATUS.
Retain the tool's process/session identifier and poll that process. If a handle is
lost, inspect its process and output first; do not launch a duplicate training run.
When interrupted, record the last completed step and the next executable action.
An exhausted turn budget is an unfinished handoff, not evidence of a research block.

If two attempts reproduce the same failure without narrowing it, stop repeating
them. Change the measurement so it distinguishes the live hypotheses, or take the
roadmap's independent bounded work. Ask the user only when choosing between concrete
directions changes the agreed contract or requires information they alone have.
Include the evidence, recommended option, and consequence. A failed training run
alone does not require permission for another diagnostic.

## Tests and reviews

Prefer tests of independent facts: analytic mechanics, packet causality, conservation,
frame transforms, scorer edge cases, reset isolation, action distributions, no
truth leakage, saved-bundle parity, and actual export behavior. Training quality
belongs in experiment reports; do not make the ordinary unit test suite train a
large policy or require a GPU for every pure unit test.

GPU integration tests have explicit markers and fail clearly when intentionally
run without an accessible GPU. Skipped GPU tests are not a passed GPU gate.
Do not change actual model outputs to expected values just to repair a screenshot.

Use distinct run directories. Checkpoint writes are atomic inside the project.
Keep scenario/test manifests immutable. A changed physics/schema creates a new
dataset and bundle version; the old artifacts remain attributable.
Include inference implementation semantics in versioning: unchanged tensor shapes
do not make a changed observer compatible. Record source/config/data/checkpoint
hashes and selected checkpoint identity; never call the last weights “best” without
selection evidence. A diagnostic reload under changed code is a new result.

## Handoff format

Update STATUS with:

1. Completed package/substep and changed behavior.
2. Exact commands and artifact paths.
3. Gate results: passed/failed/inconclusive, values, samples, seeds, hardware.
4. Known limitations and any contract/decision changes.
5. The next bounded task and what it must not redo.

Keep this short enough for another agent to use. Larger logs and plots belong in
the referenced run directory. Do not claim a roadmap milestone from a smoke test.
Replace the current assignment when it changes; archive completed experiments
instead of appending competing “next” instructions. Keep README, ROADMAP, and
STATUS consistent. A failed experiment does not itself require a user decision:
continue authorized diagnosis and ask only for a concrete unresolved tradeoff.
