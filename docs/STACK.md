# Stack and decision record

Updated: 2026-09-12. Status: installed prototype under corrective review. Host GPU
checks and bounded training have run; [STATUS](STATUS.md) records their scope.
The [review](REVIEW.md) and recovery assignments distinguish planned capabilities
from implemented paths. Keep this stack during contract repairs.

## What is settled by the discussion

- Global purpose: learned compensation that makes cheap, imperfect robots practical.
- Accurate positioning within a supplied deadline is the primary objective.
- Smoothness is secondary; physical operating limits remain constraints.
- The feeling is learned, recurrent state. Its components do not need named causes.
- Prediction and control should benefit from the same feeling.
- Observations are imperfect, including persistent bias and changing reliability.
- Start in simulation with one small robot; preserve a path to real hardware and
  families of robots.
- Implementation guidance must contain bounded tasks, acceptance criteria, and
  explicit failure recovery.
- Start model training on the GPU; start simulation there too if capacity permits.
- Provide synchronized animations showing learning progress.
- Keep all setup/runtime writes inside the project directory.
- Include structural bending and an unstable base in the required curriculum.
- Constrain base forces/moments/deflection as well as motor loads, including when
  the base is itself actuated.

## Recommended stack

| Component | Choice | Reason and boundary |
| --- | --- | --- |
| Application language | Python 3.12 | Straightforward scientific and ML integration. Compiled libraries do the numerical work. Do not replace the host Python. |
| Environment/dependencies | uv, `pyproject.toml`, committed `uv.lock` | Isolated interpreter and reproducible dependency resolution. |
| Learned models | PyTorch, float32 initially | Small GRU/MLP models, familiar debugging, explicit training loops for supervised prediction. |
| Training physics | NVIDIA Warp rigid kernel currently; MuJoCo Warp planned for coupled models | Current custom 2R kernel has native reference fixtures. MuJoCo Warp arm stepping is not implemented; verify support/capacity for coupled models before claiming it. |
| Reference physics | Native MuJoCo Python bindings | CPU fixture comparisons, diagnostic viewer, fallback. |
| Training/environment API | TorchRL + TensorDict | Device-resident observations, rollouts, resets, and optimization. |
| First policy optimizer | TorchRL PPO losses and advantage estimation | Reuse maintained components; do not implement PPO equations from scratch. |
| Numeric utilities | NumPy; SciPy where needed by baseline control | Analytic checks, trajectory generation, modest offline analysis. |
| Configuration | TOML + `tomllib` + typed dataclasses | One explicit configuration tree, validation, no implicit global state. |
| Data | NumPy `.npy` arrays, memory mapping, JSON manifests | Sequence storage that can be inspected and streamed without loading a dataset into VRAM. |
| Metrics | JSONL and JSON; Matplotlib reports | Local, reproducible artifacts. TensorBoard is optional. No required hosted account. |
| Animation | Standalone HTML/Canvas replay; Matplotlib + Pillow for GIF | No Node toolchain or rendering in the training loop. |
| Tests and static checks | pytest, Ruff, mypy | Physics/contract tests plus typed boundaries; avoid tests that merely restate implementation. |
| Inference packaging | PyTorch `state_dict` bundles + JSON metadata | Exact preprocessing and recurrent reset contract travel with weights. ONNX is a later hardware-driven decision. |

Use Python for coordination, not per-particle/per-environment numerical loops on
the GPU. Keep hot arrays contiguous and batch neural inference. A native MuJoCo
step already executes compiled code; Python does not imply a Python physics engine.

No Rust, C++, Toit, or custom CUDA implementation is needed for the first
experiment. A future hardware adapter may use another language without changing
the trained-model contract. Select that adapter after selecting actual hardware.

## Why PyTorch rather than JAX here

Both are valid. JAX is attractive for a consistently functional, compiled pipeline
and integration with MuJoCo Playground. PyTorch is the working recommendation
because this project initially needs custom recurrent estimation, diagnostics,
and small experiments more than maximum batch throughput. Mixing PyTorch and JAX
would introduce two compilation/memory systems without a demonstrated benefit.

MuJoCo Warp supports PyTorch directly, so choosing PyTorch preserves a GPU physics
path. This is documented in the [MuJoCo Warp learning-framework FAQ](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
and [Warp's PyTorch interoperability guide](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html).
Shared memory does not eliminate stream synchronization and buffer-lifetime work;
those are explicit P0/P1 acceptance items.

If the user chooses JAX before implementation, revise STACK, DESIGN, and the P0/P5/P6
tasks together. Do not maintain two learning implementations. A later migration
requires a specific demonstrated limitation and a replacement validation plan.

## GPU first, with a native reference

P0 tests CUDA and Warp immediately. P1 implements the arm on Warp and compares
physical fixtures with native MuJoCo. The default training loop keeps simulator
state, sensors, feeling, policy, and PPO tensors on the GPU. A 4 GiB GPU is a
constraint to measure, not a reason to promise thousands of environments.

If the minimum Warp workload cannot fit after bounded recovery, record the evidence
and use native physics with GPU model training. Never silently fall back or call
that a GPU simulation result. Slow but fitting configurations are reported before
changing the GPU-first preference.

TorchRL replaces the initially considered SB3 route because GPU simulation must
remain device-resident. Use existing PPO/GAE components with a small adapter and
training loop; pin TorchRL and TensorDict together. Do not force GPU states through
NumPy wrappers each tick.

[MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) provides JAX and Warp paths;
its Warp implementation does not support automatic differentiation. We do not
require gradients through the physics engine: supervised prediction and PPO use
simulator trajectories. Differentiability is not a selection criterion for P0-P8.

## Why a policy before learned-model planning

The predictor teaches the feeling to retain useful physical information. A separate
controller learns in the actual simulator using that feeling. This isolates model
prediction error from control optimization and gives a direct controller baseline.

Planning through the learned predictor is a later option, not a prerequisite.
It brings horizon selection, model exploitation, candidate-search cost, and
uncertainty issues. Do not implement Dreamer, TD-MPC2, or a new MPC optimizer as
part of the initial roadmap. The [TD-MPC2 project](https://www.tdmpc2.com/) is a
useful reference for a later planning branch, not an assertion of compatibility
with our sensors, robot, or latent-state contract.

## Hardware and dependency preflight

Observed on the host, outside the restricted GPU sandbox:

| Item | Observation |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4096 MiB |
| Driver | 610.57.04 |
| CPU | Intel Core i7-7700; 4 physical cores, 8 logical CPUs |
| Host Python visible in this session | 3.14.7; do not overwrite it |

The local environment and lockfile exist; CUDA kernels, gradients, Warp/Torch
sharing, and bounded training have run on the host. The sandbox cannot access the
driver; that does not mean the host lacks a GPU. Run GPU work outside the sandbox
through the local wrapper. No driver reinstall or silent CPU substitution is needed.
Memory available to a process can differ from physical memory and changes over time.

P0 must establish a tested compatibility matrix containing Python, MuJoCo,
PyTorch/CUDA build, TorchRL/TensorDict, driver, and Warp/MuJoCo Warp versions. Select
released versions, not development documentation's version numbers. Pin the
resolved result in the lockfile and record the tested platform. Follow
[uv's project workflow](https://docs.astral.sh/uv/concepts/projects/).

Verify an actual forward/backward pass on this GPU; importing PyTorch is not a
CUDA compatibility test. Verify the wheel includes this GPU's architecture.
If the newest wheel does not, select a supported released wheel through a
documented uv index configuration. Do not upgrade host drivers or build PyTorch
from source as an automatic recovery step.

Working local profiles (starting settings, not performance promises):

| Profile | Initial configuration |
| --- | --- |
| `smoke` | CUDA; 1-4 worlds; tiny data; one short training pass; explicit CPU reference option |
| `local` | Start with 4 worlds; prior full PPO checks fit 8 but OOM at 32. Recheck capacity with the actual changed task/model. |
| `capacity` | World counts 1, 16, 32, 64, 128, 256; include sensors/model/optimizer allocations |

Use float32 for learning, no automatic mixed precision or `torch.compile` initially.
Native physics uses its native precision. A GPU backend's precision difference
must be measured. Keep datasets on disk/host memory. Aim to leave at least 20% of
currently available VRAM free during steady training; allocations by the display
and other programs also count. Measure total device use, not only PyTorch allocations.

## Decisions implementing agents may make

They may choose local function names, internal helpers, and small implementation
details consistent with the interfaces. They may tune documented knobs within the
bounded experiments in the roadmap.

Changing the agreed objective ordering, action meaning, observation visibility,
learning family, supported topology or numerical gates requires an evidence-backed
decision. Implementing an already specified stage (such as P8's pitch-only base),
versioning its declared schema, or refreshing a world bundle under DESIGN is
ordinary roadmap work. It does not require asking for architecture permission
again. Record the evidence and consequences in STATUS. A failing test is not
evidence that its threshold should silently be weakened.
