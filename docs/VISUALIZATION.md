# Watching the robot learn

Animation is a required product outcome, not an optional final polish task.
Provide something observable before learning exists, then reuse that viewer for
every training stage. Initial visuals can be simple 2D geometry; they must replay
actual simulated trajectories accurately.

**Current assignment: C3a**, in [ROADMAP](ROADMAP.md). The existing
`artifacts/p7/c2-packet-feedback-diagnostic.html` contains current packet traces,
but its signal cursors assume 10 ms even for 1 ms physics samples, and its axes
can hide negative values and speed violations. Repair the existing exporter using
saved inputs in STATUS. R4's historical HTML/GIF implementation remains reusable.
Viewer work needs no new training. Flex/base transforms remain required when
coupled mechanics is validated.

## First visible deliverable (P2)

Provide `view` for a scripted/baseline arm and `replay` for a saved trajectory.
Use a standalone HTML file with embedded JSON and a small Canvas script. It opens
from disk without a server, network access, Node/npm, or CDN. Keep geometry drawing
and playback logic small and specific to the planar robot.

Native MuJoCo's viewer can be an additional diagnostic tool when a display is
available. It is not required to train on the GPU or export the final animation.
Drawing saved link transforms on the CPU is compatible with GPU physics.

Required first-view elements:

- Actual arm, world axes/ground reference, endpoint trail, target tolerance circle.
- Time, deadline/countdown, endpoint error, and settled/success/failure state.
- Play/pause, scrubber, playback speed, reset.
- Fixed camera scale across comparisons; physical time, not optimization updates,
  drives motion.

## Learning-progress comparison

Save the initial untrained policy, intermediate checkpoints at 10/25/50/75/100% of
each bounded training budget, and the best validation checkpoint. At each snapshot
evaluate the same versioned demo scenarios with deterministic inference and the
same external initial states/noise/event schedules. Keep full-world resets and
initial feelings identical. Do not train on these demo rollouts.

Generate a synchronized comparison with panels for untrained, intermediate, best,
and ordinary feedback baseline. Label checkpoint ID and number of training
transitions. Add selector controls for scenario and checkpoint. Also show training
curves, with the selected checkpoint highlighted, so improvements and regressions
can be inspected.

Later-stage comparisons use the same model/observation schema within that stage.
Do not load a rigid-arm checkpoint into a flex/base schema merely for a visual
before/after. Train/save a compatible initial checkpoint for that stage.

The exported replay shows observed noisy endpoint samples as optional markers and
the observer's estimated endpoint as a separate optional marker. True geometry is
for the human viewer/evaluator; it must not enter policy inputs. Show short forward
predictions as a ghost trail only when actually recorded from the predictor; do not
draw a smoothed actual trajectory and label it a prediction.

## Imperfection visibility

- Backlash: optional motor/output angle inset or gap indicator, using simulator truth.
- Bending: draw each actual segment transform, not an ideal rigid line between joints.
- Moving base: draw the mounting and world reference so displacement is visible.
- Strain: display base force/moment/deflection and motor utilization relative to their
  configured limits. Add a timeline when a limit is approached or exceeded.
- Fault events: mark their true activation time in the human diagnostic view, without
  implying the policy was told the event occurred.

If small deflections are magnified, clearly label the scale and provide a true-scale
view. Do not animate fictitious compensation to make a controller look better.
No named psychological interpretation of latent coordinates is required; a small
optional latent heatmap is diagnostic only.

## Recording and export contract

Record control-tick time, task, command, true q/dq, endpoint pose/velocity, body
transforms, base pose, flex coordinates, public observation/masks/ages, estimated
state, selected predictions, load metrics, and event/violation markers. Prediction
trails may be sampled sparsely to keep artifacts small. Simulator truth fields stay
in the replay/evaluation artifact, never the inference feature record.

Use the simulation clock and original samples. Interpolate transforms only for
display, with explicit angle handling. Score metrics from original physics-rate
data; interpolation is not a replacement for real samples. Fixed target, scale,
time origin, speed, and failure display across panels prevent misleading comparison.

Training runs headless. Copy only selected evaluation trajectories to host memory
and render after collection, so visualization does not determine training speed.

Export a standalone HTML comparison as the primary artifact and a GIF as a broadly
viewable fallback using Matplotlib/Pillow. MP4 is optional if an existing encoder
is available or a project-local encoder is explicitly configured; do not install
system ffmpeg. All font/plot/temp caches and exports follow LOCAL_SETUP.

## Acceptance

1. Viewer works from a local file with network disabled; no console errors.
2. A known fixture's drawn endpoint matches recorded coordinates at sampled frames.
3. Scrubbing all panels to t uses the same physical t and keeps ended episodes
   visibly ended rather than silently restarting them.
4. Labels and plotted errors match the authoritative metric files.
5. A clean export works without a display server or GPU rendering context.
6. Final artifact includes initial/intermediate/best comparisons, training curves,
   at least one imperfect case, and honest failure/stress examples.
7. Export metadata links exact checkpoint/scenario/config hashes to the report.
8. A mixed-rate fixture aligns physics, packet/posterior, and accepted-command
   values at a known physical time. Include negative q/dq/actions, speed above the
   limit, and missing estimates. Axes show the data and units; absent values are
   unavailable, not zero or truth. Both joints and measured/posterior velocities
   must be inspectable in the current packet diagnostic.

If a web viewer fails in the available browser, retain the trajectory data and ship
the GIF/report first, then fix the viewer. Lack of OpenGL cannot block recording or
the Pillow export. Do not replace animation with screenshots or prose alone.
