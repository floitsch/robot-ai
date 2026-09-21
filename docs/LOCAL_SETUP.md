# Project-local setup and execution

This is a user requirement: **all files created by our setup and application stay
inside the project directory**. Do not install into the host Python, modify shell
startup files, install system packages/drivers, or use a user-wide Python/tool cache.
Reading existing system tools/libraries is fine.

`scripts/project-run`, `scripts/bootstrap`, and the local environment exist.
Reuse them; all roadmap commands use `scripts/project-run` for this reason.
The wrapper derives the repository root from its own location, changes to that
directory, exports local paths, and execs its arguments without shell re-evaluation.
It does not change the caller's shell or rewrite HOME/CODEX_HOME.

The NVIDIA GPU is accessible on the host **outside the sandbox**. Run CUDA/Warp
checks and training there through this same wrapper. Do not infer absent hardware
from sandbox NVML/driver warnings, reinstall drivers, or silently switch training
to CPU. Pure unit tests and native reference fixtures may run in the sandbox and
must be labeled accordingly.

## Directory contract

```text
.venv/                 # project Python environment
.local/bin/            # uv if not already available; any explicitly needed tools
.local/python/         # managed Python 3.12
.local/tools/          # uv-managed tools, if used
.cache/uv/
.cache/pip/
.cache/torch/
.cache/torch-extensions/
.cache/torchinductor/
.cache/triton/
.cache/warp/
.cache/cuda/
.cache/matplotlib/
.cache/fontconfig/
.cache/python/
.cache/ruff/
.cache/mypy/
.cache/xdg/
.config/               # process-local XDG config, not host config
.data/                 # process-local XDG data
.state/                # process-local XDG state
.tmp/                  # temporary files, including pytest temp fixtures
artifacts/             # immutable run directories, models, plots, animations
datasets/              # versioned-by-manifest data shards
```

Ignore generated paths in version control; commit source, TOML configs, dependency
lockfile, small manifests/fixtures, and documentation. Do not create this whole tree
unless the tools need it. Source-relative Python bytecode is acceptable inside the
project; redirect it anyway to avoid writes beside outside read-only packages.

## Environment established before importing numeric libraries

The wrapper sets these to absolute paths beneath the resolved project root:

| Variable | Relative path/value |
| --- | --- |
| `UV_CACHE_DIR` | `.cache/uv` |
| `UV_PYTHON_INSTALL_DIR` | `.local/python` |
| `UV_PYTHON_BIN_DIR` | `.local/bin` |
| `UV_TOOL_DIR` | `.local/tools` |
| `UV_TOOL_BIN_DIR` | `.local/bin` |
| `UV_PROJECT_ENVIRONMENT` | `.venv` |
| `UV_NO_MODIFY_PATH` | `1` |
| `PIP_CACHE_DIR` | `.cache/pip` |
| `PYTHONNOUSERSITE` | `1` |
| `PYTHONPYCACHEPREFIX` | `.cache/python` |
| `XDG_CACHE_HOME` | `.cache/xdg` |
| `XDG_CONFIG_HOME` | `.config` |
| `XDG_DATA_HOME` | `.data` |
| `XDG_STATE_HOME` | `.state` |
| `TMPDIR`, `TMP`, `TEMP` | `.tmp` |
| `TORCH_HOME` | `.cache/torch` |
| `TORCH_EXTENSIONS_DIR` | `.cache/torch-extensions` |
| `TORCHINDUCTOR_CACHE_DIR` | `.cache/torchinductor` |
| `TRITON_CACHE_DIR` | `.cache/triton` |
| `CUDA_CACHE_PATH` | `.cache/cuda` |
| `MPLCONFIGDIR` | `.cache/matplotlib` |
| `RUFF_CACHE_DIR` | `.cache/ruff` |
| `MYPY_CACHE_DIR` | `.cache/mypy` |

For Warp explicitly set `warp.config.kernel_cache_dir` to `.cache/warp` before
`warp.init()` or kernel compilation. Verify the supported API in the pinned Warp
release; an unrecognized environment variable alone is not sufficient. Any code
that imports/initializes Warp must pass through the same bootstrap module.

Set pytest's cache directory under `.cache/pytest` and `--basetemp` under `.tmp`.
Configure font cache location if fontconfig tries to write elsewhere; use a bundled
or existing font and avoid system font installation. If any library adds another
cache, configure and test it before use. Do not import optional renderers from the
headless training process.

Do not set HOME to a temporary location to solve these requirements. Use each
tool's explicit path controls. The wrapper may prepend `.local/bin` to PATH for
its child process only.

## Bootstrap behavior

1. Resolve root; reject a requested environment/cache/output location outside root.
   Resolve symlinks when checking confinement.
2. Set local paths before running uv or Python installation commands. Use an already
   installed uv if available. Otherwise unpack an official uv binary into
   `.local/bin`; do not run an installer with default global/user destinations.
3. Use uv to obtain Python 3.12 into `.local/python` and create `.venv`.
4. Resolve/pin released dependencies and the tested CUDA wheel index. Never invoke
   `sudo`, `pip --user`, `uv tool install` without local paths, or `--system`.
5. Print interpreter, environment, package-cache, Warp-cache, and CUDA-cache paths.
6. Run `doctor`, including CUDA kernel/gradient and Warp/PyTorch handoff checks.

Verify supported uv path controls in the [official environment-variable reference](https://docs.astral.sh/uv/reference/environment/).
No full CUDA toolkit installation should be assumed; first test the released
wheels against the installed driver. Source builds and driver changes are outside
P0's recovery scope.

## Acceptance and confinement check

Run a representative clean-cache sequence: bootstrap, imports, one Warp kernel,
one PyTorch training step, one plot/GIF/HTML export, and a pytest temp fixture.
Record all resolved output/cache paths. Where available, use a write-tracing tool
already on the machine to check file-creating/open-for-write/rename operations.
Write the trace under `artifacts/setup/`; do not install a system tracer.

No project-driven regular file write may resolve outside root. Reading system
libraries and opening GPU devices is not file pollution. OS-maintained system logs
are outside the application's file-output contract. If a write trace cannot be
performed, report that narrower verification honestly rather than claiming complete
confinement from environment variables alone.

Subprocesses inherit the wrapper environment. Temporary downloads and atomic-write
staging files live under `.tmp` or beside their destination inside root. CLI output
arguments outside root are rejected. An existing external dataset may be read only
when explicitly configured, but generated derivatives remain local.

If a tool tries to write outside root, stop that tool, correct its path config, and
repeat its smoke check. Do not continue with global pollution as a workaround.
