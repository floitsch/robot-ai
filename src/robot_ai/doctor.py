"""P0 environment, CUDA, Warp, and path-confinement diagnostics."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .config import confined_path, project_root


def _result(status: str, detail: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **values}


def _torch_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment  # noqa: BLE001
        return _result("unavailable", "PyTorch import failed", error=repr(exc))

    result: dict[str, Any] = {"torch_version": torch.__version__}
    cuda = bool(torch.cuda.is_available())
    result["cuda_available"] = cuda
    if not cuda:
        return _result("unavailable", "CUDA device is not visible to this process", **result)

    device = torch.device("cuda:0")
    device_index = 0
    try:
        torch.cuda.init()
        result.update(
            device_name=torch.cuda.get_device_name(device_index),
            capability=list(torch.cuda.get_device_capability(device_index)),
        )
        torch.cuda.reset_peak_memory_stats(device_index)
        free_before, total = torch.cuda.mem_get_info(device_index)
        values = torch.ones((1024,), device=device, dtype=torch.float32, requires_grad=True)
        loss = (values.square() * 0.5).sum()
        loss.backward()
        free_after, _ = torch.cuda.mem_get_info(device_index)
        result.update(
            total_memory_bytes=total,
            free_memory_before_bytes=free_before,
            free_memory_after_bytes=free_after,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device_index),
        )
        return _result("passed", "CUDA forward and backward pass succeeded", **result)
    except Exception as exc:  # pragma: no cover - depends on GPU/runtime  # noqa: BLE001
        return _result("failed", "CUDA kernel or gradient probe failed", error=repr(exc), **result)


def _warp_probe(torch_result: dict[str, Any]) -> dict[str, Any]:
    try:
        import warp as wp
    except Exception as exc:  # pragma: no cover - depends on environment  # noqa: BLE001
        return _result("unavailable", "Warp import failed", error=repr(exc))

    root = project_root()
    cache = root / ".cache" / "warp"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        wp.config.kernel_cache_dir = str(cache)
        wp.init()
        device = "cuda:0" if torch_result.get("cuda_available") else "cpu"

        @wp.kernel
        def add_one(values: wp.array(dtype=wp.float32)):  # type: ignore[misc,valid-type,no-untyped-def]
            index = wp.tid()
            values[index] = values[index] + 1.0

        values = wp.zeros(16, dtype=wp.float32, device=device)
        wp.launch(add_one, dim=16, inputs=[values], device=device)
        wp.synchronize_device(device)
        observed = values.numpy()
        if not bool((observed == 1.0).all()):
            return _result("failed", "Warp kernel returned unexpected values")

        handoff = "not attempted"
        if torch_result.get("cuda_available"):
            import torch

            tensor = torch.zeros(16, device="cuda", dtype=torch.float32)
            shared = wp.from_torch(tensor)
            wp.launch(add_one, dim=16, inputs=[shared], device="cuda:0")
            wp.synchronize_device("cuda:0")
            torch.cuda.synchronize()
            if not bool(torch.all(tensor == 1.0).item()):
                return _result("failed", "Warp/PyTorch shared tensor had unexpected values")
            handoff = "passed; Warp launch synchronized before PyTorch read"
        return _result("passed", "Warp kernel and shared-array probe succeeded", device=device, handoff=handoff)
    except Exception as exc:  # pragma: no cover - depends on Warp version/runtime  # noqa: BLE001
        return _result("failed", "Warp probe failed", error=repr(exc), cache=str(cache))


def _capacity_probe(torch_result: dict[str, Any]) -> dict[str, Any]:
    """Measure learner-shaped tensors and actual reduced Warp allocations."""

    if not torch_result.get("cuda_available"):
        return _result("unavailable", "capacity sweep requires a visible CUDA device")
    try:
        import torch

        device = torch.device("cuda:0")
        counts = (1, 4, 16, 32, 64, 128, 256)
        sweeps: list[dict[str, Any]] = []
        for worlds in counts:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            free_before, total = torch.cuda.mem_get_info(0)
            try:
                # Shapes mirror the first policy rollout and recurrent state. The
                # arrays are deliberately explicit so this remains a capacity
                # probe, not an unsupported simulator-memory extrapolation.
                belief = torch.zeros((worlds, 128), device=device)
                rollout = torch.zeros((worlds, 128, 151), device=device)
                optimizer_state = torch.zeros((40_000, 2), device=device)
                _ = belief, rollout, optimizer_state
                torch.cuda.synchronize()
                free_after, _ = torch.cuda.mem_get_info(0)
                sweeps.append({"worlds": worlds, "status": "passed", "free_before_bytes": free_before,
                               "free_after_bytes": free_after, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                               "margin_against_current_free": free_after >= max(1, int(free_before * 0.2))})
                del belief, rollout, optimizer_state
            except RuntimeError as exc:
                sweeps.append({"worlds": worlds, "status": "oom_or_failed", "error": repr(exc),
                               "free_before_bytes": free_before, "total_bytes": total})
                break
        simulator_sweeps: list[dict[str, Any]] = []
        try:
            import gc

            import numpy as np
            import warp as wp

            from .sim.native import default_descriptor
            from .sim.warp_backend import WarpArmBatch

            descriptor = default_descriptor()
            for worlds in counts:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_before, total = torch.cuda.mem_get_info(0)
                started = time.monotonic()
                try:
                    arm = WarpArmBatch(descriptor, worlds, device="cuda:0")
                    arm.reset()
                    arm.step(np.zeros((worlds, 2), dtype=np.float32))
                    wp.synchronize_device("cuda:0")
                    free_after, _ = torch.cuda.mem_get_info(0)
                    simulator_sweeps.append(
                        {
                            "worlds": worlds,
                            "status": "passed",
                            "free_before_bytes": free_before,
                            "free_after_bytes": free_after,
                            "allocation_delta_bytes": free_before - free_after,
                            "total_bytes": total,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    del arm
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except RuntimeError as exc:
                    simulator_sweeps.append(
                        {
                            "worlds": worlds,
                            "status": "oom_or_failed",
                            "error": repr(exc),
                            "free_before_bytes": free_before,
                            "total_bytes": total,
                        }
                    )
                    break
        except Exception as exc:  # pragma: no cover - depends on GPU/runtime  # noqa: BLE001
            simulator_sweeps.append({"worlds": 0, "status": "unavailable", "error": repr(exc)})

        passed = all(item["status"] == "passed" and item["margin_against_current_free"] for item in sweeps)
        simulator_passed = bool(simulator_sweeps) and all(item["status"] == "passed" for item in simulator_sweeps)
        return _result(
            "passed" if passed and simulator_passed else "failed",
            "learner-shaped tensors and actual reduced Warp allocations; full PPO capacity remains open",
            sweeps=sweeps,
            simulator_sweeps=simulator_sweeps,
        )
    except Exception as exc:  # pragma: no cover - depends on GPU/runtime  # noqa: BLE001
        return _result("failed", "capacity probe failed", error=repr(exc))


def _confinement_probe() -> dict[str, Any]:
    root = project_root()
    inside = confined_path(root / ".tmp" / "doctor-probe.txt", root=root)
    try:
        confined_path(root.parent / "outside.txt", root=root)
    except ValueError:
        pass
    else:
        return _result("failed", "outside-root path was accepted")
    return _result("passed", "project-local path policy accepted an inside path and rejected an outside path", path=str(inside))


def run_doctor(output: str | Path, *, requested_device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    """Run diagnostics and write a JSON report under the project root."""

    root = project_root()
    destination = confined_path(output, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch_result = _torch_probe()
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {
            key: os.environ.get(key)
            for key in ("UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "TMPDIR", "TORCH_HOME", "CUDA_CACHE_PATH")
        },
        "checks": {
            "confinement": _confinement_probe(),
            "pytorch_cuda": torch_result,
            "warp": _warp_probe(torch_result),
            "capacity": _capacity_probe(torch_result),
        },
        "requested": {"device": requested_device, "seed": seed},
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report
