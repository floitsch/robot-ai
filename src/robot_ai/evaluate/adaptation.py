"""Paired frozen-policy adaptation experiments for P7."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import RuntimeConfig, confined_path
from ..contracts import Task
from ..control.policy import POLICY_INPUT_DIM, PolicyNetwork
from ..control.runtime import Runtime
from ..models.bundle import WorldBundle
from ..sim.combined import CombinedImperfectArm, CombinedSettings
from ..sim.native import default_descriptor
from ..sim.scorer import score_episode
from ..sim.sensors import SensorFault


def _task() -> Task:
    goal_q = np.array([0.25, 0.9])
    lengths = np.array([0.30, 0.25])
    goal = np.array([lengths[0] * np.sin(goal_q[0]) + lengths[1] * np.sin(goal_q.sum()),
                     -lengths[0] * np.cos(goal_q[0]) - lengths[1] * np.cos(goal_q.sum())])
    return Task(goal, deadline_s=1.5)


def _settings(variant: str) -> CombinedSettings:
    base = CombinedSettings.research_default()
    if variant == "weak-motor":
        actuator = replace(base.actuator, effectiveness=np.array([0.60, 0.66]))
        return replace(base, actuator=actuator, faults=())
    if variant == "sensor-drift":
        sensors = replace(base.sensors, bias_limit=np.ones(8) * np.array([0.08, 0.08, 0.12, 0.12, 0.08, 0.08, 0.015, 0.015]))
        return replace(base, sensors=sensors, faults=(SensorFault("dropout", (6,), 0.65, 0.95),))
    if variant == "reversal":
        return replace(base, faults=())
    if variant == "localized-rubbing":
        actuator = replace(base.actuator, local_resistance=np.array([0.32, 0.28]))
        return replace(base, actuator=actuator, faults=())
    if variant == "combined":
        return base
    raise ValueError(f"unknown adaptation variant: {variant}")


def _score_trace(times: list[float], endpoints: list[np.ndarray], velocities: list[np.ndarray],
                 violations: list[bool], task: Task, commands: list[np.ndarray], dt: float) -> dict[str, Any]:
    score = score_episode(np.asarray(times), np.asarray(endpoints), np.asarray(velocities), task,
                          hard_violation=np.asarray(violations, dtype=bool), max_sample_interval_s=0.001)
    from .metrics import command_slew

    return {"score": score.__dict__, "command_slew": command_slew(np.asarray(commands), dt)}


def _run_episode(settings: CombinedSettings, *, controller: str, world: WorldBundle | None,
                 policy: PolicyNetwork | None, device: torch.device, task: Task,
                 seed: int, variant: str, history_ablation: bool, memoryless: bool = False) -> dict[str, Any]:
    descriptor = default_descriptor()
    arm = CombinedImperfectArm(descriptor, settings, timestep=0.001, seed=seed)
    initial = np.array([-0.35, 0.75])
    arm.reset(initial)
    initial_observation = arm.observation()
    assert world is not None and policy is not None
    runtime = Runtime(world, descriptor, policy, device, memoryless=memoryless)
    runtime.reset(descriptor, initial_observation, task)
    times = [0.0]
    endpoints = [arm.truth().endpoint_xz.copy()]
    velocities = [arm.truth().endpoint_velocity_xz.copy()]
    violations = [arm.load_utilization() > 1.0]
    commands: list[np.ndarray] = []
    observations = initial_observation
    controls = round(task.total_duration_s / 0.01)
    for control_index in range(controls):
        if history_ablation and control_index > 0:
            runtime.reset_memory(observations)
        command = runtime.act()
        accepted = command.values.copy()
        commands.append(accepted)
        for _ in range(10):
            truth = arm.step(accepted)
            times.append(times[-1] + 0.001)
            endpoints.append(truth.endpoint_xz.copy())
            velocities.append(truth.endpoint_velocity_xz.copy())
            violations.append(arm.load_utilization() > 1.0)
        observations = arm.observation()
        runtime.observe(command, observations)
    result = _score_trace(times, endpoints, velocities, violations, task, commands, 0.01)
    result.update({"controller": controller, "variant": variant, "seed": seed,
                   "history_ablation": history_ablation})
    return result


def _bootstrap_difference(adaptive: np.ndarray, memoryless: np.ndarray, seed: int) -> dict[str, float]:
    difference = adaptive - memoryless
    if len(difference) == 0:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(difference, size=(2000, len(difference)), replace=True).mean(axis=1)
    return {"mean": float(difference.mean()), "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975))}


def run_adaptation_suite(run: str | Path, output: str | Path, *, config: RuntimeConfig,
                         device: str = "cpu", seeds: int = 3,
                         memoryless_run: str | Path | None = None) -> dict[str, Any]:
    """Compare a frozen feeling policy with matched memoryless controls."""

    if seeds < 1:
        raise ValueError("seeds must be positive")
    run_path = confined_path(run, must_exist=True)
    report = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    device_obj = torch.device(device)
    world = WorldBundle.load(report["world_bundle"], device=device_obj)
    policy = PolicyNetwork(input_dim=POLICY_INPUT_DIM).to(device_obj)
    policy.load_state_dict(torch.load(run_path / "best-policy.pt", map_location=device_obj, weights_only=True))
    policy.eval()
    if memoryless_run is None:
        raise ValueError("P7 requires a separately trained learned-memoryless policy run")
    memoryless_path = confined_path(memoryless_run, must_exist=True)
    memoryless_report = json.loads((memoryless_path / "metrics.json").read_text(encoding="utf-8"))
    if not bool(memoryless_report.get("memoryless", False)):
        raise ValueError("P7 memoryless comparator must be a run trained with memoryless=true")
    memoryless_policy = PolicyNetwork(input_dim=int(memoryless_report.get("policy_input_dim", POLICY_INPUT_DIM))).to(device_obj)
    memoryless_policy.load_state_dict(torch.load(memoryless_path / "best-policy.pt", map_location=device_obj, weights_only=True))
    memoryless_policy.eval()
    task = _task()
    variants = ("weak-motor", "sensor-drift", "reversal", "localized-rubbing", "combined")
    episodes: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        settings = _settings(variant)
        for seed in range(seeds):
            episodes.append(_run_episode(settings, controller="memoryless", world=world, policy=memoryless_policy,
                                         device=device_obj, task=task, seed=seed + variant_index * 100,
                                         variant=variant, history_ablation=False, memoryless=True))
            episodes.append(_run_episode(settings, controller="adaptive", world=world, policy=policy,
                                         device=device_obj, task=task, seed=seed + variant_index * 100,
                                         variant=variant, history_ablation=False))
            episodes.append(_run_episode(settings, controller="adaptive", world=world, policy=policy,
                                         device=device_obj, task=task, seed=seed + variant_index * 100,
                                         variant=variant, history_ablation=True))
    paired: dict[str, Any] = {}
    for variant in variants:
        memoryless = [item for item in episodes if item["variant"] == variant and item["controller"] == "memoryless"]
        adaptive = [item for item in episodes if item["variant"] == variant and item["controller"] == "adaptive" and not item["history_ablation"]]
        ablated = [item for item in episodes if item["variant"] == variant and item["history_ablation"]]
        adaptive_success = np.array([float(item["score"]["success"]) for item in adaptive])
        memoryless_success = np.array([float(item["score"]["success"]) for item in memoryless])
        ablated_success = np.array([float(item["score"]["success"]) for item in ablated])
        paired[variant] = {
            "memoryless_success_fraction": float(memoryless_success.mean()),
            "adaptive_success_fraction": float(adaptive_success.mean()),
            "history_ablated_success_fraction": float(ablated_success.mean()),
            "success_difference_bootstrap": _bootstrap_difference(adaptive_success, memoryless_success, 9000 + len(paired)),
            "adaptive_vs_history_ablation_difference": float(adaptive_success.mean() - ablated_success.mean()),
            "memoryless": memoryless, "adaptive": adaptive, "history_ablated": ablated,
        }
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final = {"controller_run": str(run_path), "memoryless_controller_run": str(memoryless_path), "device": str(device_obj),
             "config": {"scenario": config.scenario, "physics_dt": config.physics_dt,
                         "control_dt": config.control_dt, "seed": config.seed},
             "variants": list(variants),
             "seed_count": seeds, "paired": paired, "episodes": episodes,
             "gate": "inconclusive unless adaptive gain and history ablation are established"}
    destination.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final
