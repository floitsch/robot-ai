"""Evaluation and replay export for frozen PPO policy checkpoints."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..baselines.controller import FeedbackController
from ..config import RuntimeConfig, confined_path
from ..contracts import Observation, RobotDescriptor, Task
from ..control.features import CURRENT_PACKET_POLICY_INPUT_DIM
from ..control.policy import POLICY_INPUT_DIM, PolicyNetwork
from ..control.runtime import CurrentPacketRuntime, Runtime
from ..models.bundle import WorldBundle
from ..models.inference import InferenceBundle
from ..sim.actuators import ActuatorSettings
from ..sim.env import RigidArmEnvironment
from ..sim.native import ImperfectNativeArm, default_descriptor
from ..sim.scorer import joint_limit_summary, joint_limit_violations, score_episode
from ..sim.sensors import SensorSettings, SensorSuite
from ..visualize.replay import export_gif, export_html
from .baseline import healthy_case_manifest_identity, healthy_cases, load_healthy_case_manifest
from .metrics import command_slew

_BACKEND_REPLAY_FIELDS = ("q", "dq", "endpoint", "endpoint_velocity", "accepted_action")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _world_bundle_identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "metadata_sha256": _sha256(path / "metadata.json"),
            "weights_sha256": _sha256(path / "weights.pt")}


def _checkpoint_path(run: Path, checkpoint: str) -> Path:
    candidates = [run / checkpoint, run / "checkpoints" / checkpoint,
                  run / "checkpoints" / f"{checkpoint}.pt"]
    if checkpoint == "best":
        candidates.extend((run / "best-policy.pt", run / "final-policy.pt"))
    elif checkpoint == "initial":
        candidates.append(run / "initial-policy.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"policy checkpoint not found in {run}: {checkpoint}")


def _task() -> Task:
    descriptor = default_descriptor()
    q_goal = np.array([0.25, 0.9])
    endpoint = np.array([
        descriptor.link_lengths[0] * np.sin(q_goal[0]) + descriptor.link_lengths[1] * np.sin(q_goal.sum()),
        -descriptor.link_lengths[0] * np.cos(q_goal[0]) - descriptor.link_lengths[1] * np.cos(q_goal.sum()),
    ])
    return Task(endpoint, deadline_s=1.5)


def _goal(descriptor: RobotDescriptor, goal_q: np.ndarray) -> np.ndarray:
    values = descriptor
    return np.array([
        values.link_lengths[0] * np.sin(goal_q[0]) + values.link_lengths[1] * np.sin(goal_q.sum()),
        -values.link_lengths[0] * np.cos(goal_q[0]) - values.link_lengths[1] * np.cos(goal_q.sum()),
    ])


def _truth_teacher_observation(arm: ImperfectNativeArm, time_s: float) -> Observation:
    """Build a training-only teacher query; this never enters Runtime inputs."""

    truth = arm.arm.truth()
    values = np.concatenate((truth.q, truth.dq, np.zeros(2), truth.endpoint_xz))
    return Observation(values, np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), time_s)


def _decoded_state(world: WorldBundle, runtime: Runtime) -> tuple[np.ndarray, np.ndarray]:
    if runtime.features.belief is None:
        raise RuntimeError("runtime must be reset before decoding its posterior")
    with torch.no_grad():
        q, dq = world.model.decode(runtime.features.belief)
    return q.detach().cpu().numpy().copy(), dq.detach().cpu().numpy().copy()


def _replay_difference(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> dict[str, float | int | None]:
    if left.shape != right.shape:
        raise ValueError(f"replay arrays have different shapes: {left.shape} != {right.shape}")
    difference = left.astype(np.float64) - right.astype(np.float64)
    norms = np.linalg.norm(difference, axis=-1) if difference.ndim > 1 else np.abs(difference)
    first = np.flatnonzero(norms > tolerance)
    return {
        "max_abs": float(np.abs(difference).max()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "first_over_tolerance": None if len(first) == 0 else int(first[0]),
    }


def _difference_number(summary: dict[str, float | int | None], field: str) -> float:
    value = summary[field]
    if not isinstance(value, (float, int)):
        raise TypeError(f"replay difference {field} must be numeric")
    return float(value)


def compare_backend_replays(warp_evaluation: str | Path, native_evaluation: str | Path,
                            output: str | Path, *, driver_version: str) -> dict[str, Any]:
    """Verify one selected-policy replay agrees across the declared rigid backends."""

    if not driver_version:
        raise ValueError("driver_version is required for a backend-transfer report")
    warp_root = confined_path(warp_evaluation, must_exist=True)
    native_root = confined_path(native_evaluation, must_exist=True)
    warp_metrics_path = warp_root / "metrics.json"
    native_metrics_path = native_root / "metrics.json"
    warp_metrics = json.loads(warp_metrics_path.read_text(encoding="utf-8"))
    native_metrics = json.loads(native_metrics_path.read_text(encoding="utf-8"))
    if warp_metrics["backend"] != "warp" or native_metrics["backend"] != "native":
        raise ValueError("backend comparison requires one Warp and one native evaluation")
    for field in ("suite", "episodes", "physics_dt", "control_dt", "command_rate_hz", "substeps_per_command"):
        if warp_metrics[field] != native_metrics[field]:
            raise ValueError(f"backend evaluation mismatch for {field}")
    checkpoint = confined_path(warp_metrics["checkpoint"], must_exist=True)
    native_checkpoint = confined_path(native_metrics["checkpoint"], must_exist=True)
    if _sha256(checkpoint) != _sha256(native_checkpoint):
        raise ValueError("backend evaluations use different policy checkpoints")
    with np.load(warp_root / "demo.npz", allow_pickle=False) as warp_trace, \
            np.load(native_root / "demo.npz", allow_pickle=False) as native_trace:
        for field in ("time_s", "target", "link_length_1", "link_length_2"):
            if not np.array_equal(warp_trace[field], native_trace[field]):
                raise ValueError(f"backend replay mismatch for {field}")
        differences = {
            field: _replay_difference(warp_trace[field], native_trace[field], tolerance=1e-5)
            for field in _BACKEND_REPLAY_FIELDS
        }
    import warp as wp

    warp_success = float(warp_metrics["success_fraction"])
    native_success = float(native_metrics["success_fraction"])
    endpoint_difference = _difference_number(differences["endpoint"], "max_norm")
    action_difference = _difference_number(differences["accepted_action"], "max_norm")
    report = {
        "schema_version": 1,
        "purpose": "R5a selected-policy native/Warp transfer check",
        "evaluations": {
            "warp": {"path": str(warp_root), "metrics_sha256": _sha256(warp_metrics_path),
                     "trace_sha256": _sha256(warp_root / "demo.npz"), "metrics": warp_metrics},
            "native": {"path": str(native_root), "metrics_sha256": _sha256(native_metrics_path),
                       "trace_sha256": _sha256(native_root / "demo.npz"), "metrics": native_metrics},
        },
        "provenance": {
            "policy_checkpoint": str(checkpoint), "policy_checkpoint_sha256": _sha256(checkpoint),
            "runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "warp": wp.__version__, "cuda_driver": driver_version},
            "source_sha256": {
                "policy.py": _sha256(Path(__file__)),
                "runtime.py": _sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
                "features.py": _sha256(Path(__file__).parents[1] / "control" / "features.py"),
                "native.py": _sha256(Path(__file__).parents[1] / "sim" / "native.py"),
                "warp_backend.py": _sha256(Path(__file__).parents[1] / "sim" / "warp_backend.py"),
            },
        },
        "trace_difference": differences,
        "acceptance": {
            "success_threshold": 0.95,
            "trace_tolerance": 1e-5,
            "passed": bool(warp_success >= 0.95 and native_success >= 0.95 and
                           endpoint_difference <= 1e-5 and action_difference <= 1e-5),
        },
    }
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_policy(run: str | Path, output: str | Path, *, config: RuntimeConfig,
                    checkpoint: str = "best", device: str | None = None,
                    suite: str = "validation", config_identity: dict[str, str] | None = None) -> dict[str, Any]:
    """Run one frozen policy without exposing simulator truth to its runtime."""

    selected_device = device or config.device
    device_obj = torch.device("cuda:0" if selected_device == "cuda" else "cpu")
    run_path = confined_path(run)
    memoryless = False
    world_path: Path | None = None
    if run_path.name == "best" and not run_path.exists():
        run_path = run_path.parent
    run_path = confined_path(run_path, must_exist=True)
    if (run_path / "metadata.json").exists() and not (run_path / "metrics.json").exists():
        loaded_bundle = InferenceBundle.load(run_path, device=device_obj)
        world = loaded_bundle.world
        policy = loaded_bundle.policy
        selected_checkpoint = run_path
    else:
        if not (run_path / "metrics.json").exists() and run_path.name == "best":
            run_path = run_path.parent
        report = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        memoryless = bool(report.get("memoryless", False))
        world_path = confined_path(report["world_bundle"], must_exist=True)
        world = WorldBundle.load(world_path, device=device_obj)
        input_dim = int(report.get("policy_input_dim", POLICY_INPUT_DIM))
        if input_dim != POLICY_INPUT_DIM:
            raise ValueError(f"unsupported policy feature schema input dimension {input_dim}")
        policy = PolicyNetwork(input_dim=input_dim).to(device_obj)
        selected_checkpoint = _checkpoint_path(run_path, checkpoint)
        policy.load_state_dict(torch.load(selected_checkpoint, map_location=device_obj, weights_only=True))
        policy.eval()
    descriptor = default_descriptor()
    count = 64 if suite == "validation" else 200
    cases = healthy_cases(count, seed=20260911 if suite == "validation" else 20260912)
    scores: list[dict[str, Any]] = []
    demo_data: dict[str, np.ndarray | float | str] | None = None
    substeps = max(1, round(config.control_dt / config.physics_dt))
    environment = RigidArmEnvironment(descriptor, backend=config.backend, device=str(device_obj),
                                      world_count=1, physics_dt=config.physics_dt)
    for case_index, (start, goal_q) in enumerate(cases):
        reset = environment.reset(q=start.reshape(1, 2))
        task = Task(_goal(descriptor, goal_q), deadline_s=1.5)
        controls = round(task.total_duration_s / config.control_dt)
        if not np.isclose(controls * config.control_dt, task.total_duration_s):
            raise ValueError("task duration must be an integer number of control ticks")
        runtime = Runtime(world, descriptor, policy, device_obj, memoryless=memoryless)
        runtime.reset(descriptor, reset.observations[0], task)
        times = [0.0]
        endpoints = [reset.truth[0].endpoint_xz.copy()]
        velocities = [reset.truth[0].endpoint_velocity_xz.copy()]
        q_values = [reset.truth[0].q.copy()]
        dq_values = [reset.truth[0].dq.copy()]
        commands: list[np.ndarray] = []
        result = reset
        for control_index in range(controls):
            command = runtime.act()
            commands.append(command.values.copy())
            for _ in range(substeps):
                result = environment.step(command.values.reshape(1, 2))
                truth = result.truth[0]
                times.append(times[-1] + config.physics_dt)
                endpoints.append(truth.endpoint_xz.copy())
                velocities.append(truth.endpoint_velocity_xz.copy())
                q_values.append(truth.q.copy())
                dq_values.append(truth.dq.copy())
            runtime.observe(command, result.observations[0])
        times_array = np.asarray(times)
        endpoint_array = np.asarray(endpoints)
        velocity_array = np.asarray(velocities)
        q_array = np.asarray(q_values)
        violations = joint_limit_violations(q_array, descriptor.joint_limits)
        score = score_episode(times_array, endpoint_array, velocity_array, task,
                              hard_violation=violations, max_sample_interval_s=config.physics_dt)
        score_dict = dict(score.__dict__)
        score_dict.update(joint_limit_summary(times_array, q_array, descriptor.joint_limits))
        score_dict["case_index"] = case_index
        score_dict["command_slew"] = command_slew(np.asarray(commands), config.control_dt)
        scores.append(score_dict)
        if demo_data is None:
            demo_data = {
                "q": np.asarray(q_values), "dq": np.asarray(dq_values), "endpoint": endpoint_array,
                "endpoint_velocity": velocity_array, "accepted_action": np.asarray(commands),
                "time_s": times_array, "joint_limit_violation": violations, "target": task.goal_xz,
                "link_length_1": descriptor.link_lengths[0], "link_length_2": descriptor.link_lengths[1],
            }
    assert demo_data is not None
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "demo.npz", **demo_data)  # type: ignore[arg-type]
    export_html(demo_data, destination / "demo.html")
    export_gif(demo_data, destination / "demo.gif", stride=max(1, substeps // 2))
    success_fraction = float(np.mean([item["success"] for item in scores]))
    errors = [item["deadline_error_m"] for item in scores if item["deadline_error_m"] is not None]
    result_report = {
        "controller": "learned-policy", "suite": suite, "episodes": count,
        "checkpoint": str(selected_checkpoint), "checkpoint_sha256": _sha256(selected_checkpoint),
        "device": str(device_obj), "backend": config.backend, "physics_dt": config.physics_dt,
        "control_dt": config.control_dt, "scenario_count": count,
        "command_rate_hz": 1.0 / config.control_dt,
        "substeps_per_command": substeps,
        "trace_duration_s": _task().total_duration_s, "modeled_hard_limits": ["joint_position"],
        "case_manifest": healthy_case_manifest_identity() if suite == "validation" else None,
        "config_identity": config_identity,
        "world_bundle_identity": None if world_path is None else _world_bundle_identity(world_path),
        "source_sha256": {
            "evaluate_policy.py": _sha256(Path(__file__)),
            "runtime.py": _sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
            "features.py": _sha256(Path(__file__).parents[1] / "control" / "features.py"),
            "scorer.py": _sha256(Path(__file__).parents[1] / "sim" / "scorer.py"),
            "native.py": _sha256(Path(__file__).parents[1] / "sim" / "native.py"),
            "warp_backend.py": _sha256(Path(__file__).parents[1] / "sim" / "warp_backend.py"),
        },
        "success_fraction": success_fraction,
        "deadline_error_median_m": float(np.median(errors)) if errors else None,
        "command_slew_median": float(np.median([item["command_slew"] for item in scores])),
        "scores": scores,
        "artifacts": {"trajectory": str(destination / "demo.npz"),
                      "html": str(destination / "demo.html"), "gif": str(destination / "demo.gif")},
    }
    (destination / "metrics.json").write_text(json.dumps(result_report, indent=2) + "\n", encoding="utf-8")
    return result_report


def clean_packet_settings(control_dt: float) -> SensorSettings:
    """Return R5b1's explicit no-event P3 packet contract."""

    if not np.isclose(control_dt, 0.01):
        raise ValueError("R5b1 clean packet settings require a 10 ms control period")
    return SensorSettings(np.full(8, control_dt), np.zeros(8), np.zeros(8), np.zeros(8))


def evaluate_packet_policy(run: str | Path, output: str | Path, *, config: RuntimeConfig,
                           checkpoint: str = "best", device: str | None = None,
                           case_index: int | None = None,
                           case_manifest: str | Path | None = None,
                           config_identity: dict[str, str] | None = None,
                           diagnostic_truth_teacher: bool = False,
                           render_demo: bool = True) -> dict[str, Any]:
    """Evaluate a policy through R5b1's clean causal P3 sensor packets."""

    selected_device = device or config.device
    device_obj = torch.device("cuda:0" if selected_device == "cuda" else "cpu")
    run_path = confined_path(run, must_exist=True)
    report = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    memoryless = bool(report.get("memoryless", False))
    current_packet = report.get("controller_kind") == "current-packet-feedforward"
    world_path = confined_path(report["world_bundle"], must_exist=True)
    world = WorldBundle.load(world_path, device=device_obj)
    input_dim = int(report.get("policy_input_dim", POLICY_INPUT_DIM))
    if input_dim != (CURRENT_PACKET_POLICY_INPUT_DIM if current_packet else POLICY_INPUT_DIM):
        raise ValueError(f"unsupported policy feature schema input dimension {input_dim}")
    policy = PolicyNetwork(input_dim=input_dim, hidden_dim=int(report.get("policy_hidden_dim", 128))).to(device_obj)
    selected_checkpoint = _checkpoint_path(run_path, checkpoint)
    policy.load_state_dict(torch.load(selected_checkpoint, map_location=device_obj, weights_only=True))
    policy.eval()
    descriptor = default_descriptor()
    settings = clean_packet_settings(config.control_dt)
    all_cases, manifest_identity = (load_healthy_case_manifest(case_manifest) if case_manifest else
                                    (healthy_cases(64), healthy_case_manifest_identity()))
    if case_index is not None and not 0 <= case_index < len(all_cases):
        raise ValueError(f"packet case index must be in [0, {len(all_cases) - 1}]")
    cases = list(enumerate(all_cases)) if case_index is None else [(case_index, all_cases[case_index])]
    substeps = round(config.control_dt / config.physics_dt)
    if substeps < 1 or not np.isclose(substeps * config.physics_dt, config.control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    scores: list[dict[str, Any]] = []
    demo_data: dict[str, np.ndarray | float | str] | None = None
    for scored_case_index, (start, goal_q) in cases:
        task = Task(_goal(descriptor, goal_q), deadline_s=1.5)
        arm = ImperfectNativeArm(descriptor, ActuatorSettings.healthy(),
                                 SensorSuite(settings, seed=scored_case_index), timestep=config.physics_dt)
        arm.reset(start)
        observation = arm.observation()
        runtime = (CurrentPacketRuntime(world, descriptor, policy, device_obj) if current_packet
                   else Runtime(world, descriptor, policy, device_obj, memoryless=memoryless))
        runtime.reset(descriptor, observation, task)
        times = [0.0]
        truth = arm.arm.truth()
        endpoints, velocities = [truth.endpoint_xz.copy()], [truth.endpoint_velocity_xz.copy()]
        q_values, dq_values, commands = [truth.q.copy()], [truth.dq.copy()], []
        packets = [observation.values.copy()]
        available, fresh, ages = [observation.available.copy()], [observation.fresh.copy()], [observation.age_s.copy()]
        estimated_q, estimated_dq = [], []
        teacher_actions: list[np.ndarray] = []
        teacher = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0.0 else -1,
                                     computed_torque=True, hold_goal=True) if diagnostic_truth_teacher else None
        if not current_packet:
            assert isinstance(runtime, Runtime)
            estimate_q, estimate_dq = _decoded_state(world, runtime)
            estimated_q.append(estimate_q); estimated_dq.append(estimate_dq)
        controls = round(task.total_duration_s / config.control_dt)
        for control_index in range(controls):
            command = runtime.act()
            commands.append(command.values.copy())
            if teacher is not None:
                teacher_actions.append(teacher.act(control_index * config.control_dt,
                                                   _truth_teacher_observation(arm, control_index * config.control_dt)))
            for _ in range(substeps):
                truth = arm.step(command)
                times.append(times[-1] + config.physics_dt)
                endpoints.append(truth.endpoint_xz.copy())
                velocities.append(truth.endpoint_velocity_xz.copy())
                q_values.append(truth.q.copy())
                dq_values.append(truth.dq.copy())
            observation = arm.observation()
            packets.append(observation.values.copy())
            available.append(observation.available.copy())
            fresh.append(observation.fresh.copy())
            ages.append(observation.age_s.copy())
            runtime.observe(command, observation)
            if not current_packet:
                assert isinstance(runtime, Runtime)
                estimate_q, estimate_dq = _decoded_state(world, runtime)
                estimated_q.append(estimate_q); estimated_dq.append(estimate_dq)
        times_array = np.asarray(times)
        q_array = np.asarray(q_values)
        violations = joint_limit_violations(q_array, descriptor.joint_limits)
        score = score_episode(times_array, np.asarray(endpoints), np.asarray(velocities), task,
                              hard_violation=violations, max_sample_interval_s=config.physics_dt)
        score_dict = dict(score.__dict__)
        score_dict.update(joint_limit_summary(times_array, q_array, descriptor.joint_limits))
        score_dict.update({"case_index": scored_case_index,
                           "command_slew": command_slew(np.asarray(commands), config.control_dt)})
        scores.append(score_dict)
        if demo_data is None:
            demo_data = {
                "q": np.asarray(q_values), "dq": np.asarray(dq_values), "endpoint": np.asarray(endpoints),
                "endpoint_velocity": np.asarray(velocities), "accepted_action": np.asarray(commands),
                "time_s": times_array, "joint_limit_violation": violations, "target": task.goal_xz,
                "link_length_1": descriptor.link_lengths[0], "link_length_2": descriptor.link_lengths[1],
                "packet_values": np.asarray(packets), "packet_available": np.asarray(available),
                "packet_fresh": np.asarray(fresh), "packet_age_s": np.asarray(ages),
                "estimated_q": np.asarray(estimated_q), "estimated_dq": np.asarray(estimated_dq),
            }
            if teacher_actions:
                demo_data["truth_teacher_action"] = np.asarray(teacher_actions)
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if render_demo:
        assert demo_data is not None
        np.savez(destination / "demo.npz", **demo_data)
        export_html(demo_data, destination / "demo.html")
        export_gif(demo_data, destination / "demo.gif", stride=max(1, substeps // 2))
        artifacts = {"trajectory": str(destination / "demo.npz"), "html": str(destination / "demo.html"),
                     "gif": str(destination / "demo.gif")}
    errors = [item["deadline_error_m"] for item in scores if item["deadline_error_m"] is not None]
    result = {
        "controller": "learned-packet-policy",
        "suite": "r5b1-packetized-healthy-64" if case_index is None else "r5b5-case-diagnostic",
        "episodes": len(cases), "requested_case_index": case_index,
        "checkpoint": str(selected_checkpoint), "checkpoint_sha256": _sha256(selected_checkpoint),
        "device": str(device_obj), "simulator_backend": "native",
        "simulation_device": "cpu", "feature_and_inference_device": str(device_obj),
        "physics_dt": config.physics_dt, "control_dt": config.control_dt, "command_rate_hz": 1.0 / config.control_dt,
        "substeps_per_command": substeps, "trace_duration_s": task.total_duration_s,
        "modeled_hard_limits": ["joint_position"],
        "case_manifest": manifest_identity, "config_identity": config_identity,
        "world_bundle_identity": _world_bundle_identity(world_path),
        "memoryless": memoryless,
        "controller_kind": "current-packet-feedforward" if current_packet else "recurrent-packet-policy",
        "diagnostic_truth_teacher": diagnostic_truth_teacher,
        "packet_contract": {"periods_s": settings.periods_s.tolist(), "noise_std": settings.noise_std.tolist(),
                            "bias_limit": settings.bias_limit.tolist(), "max_delay_s": settings.max_delay_s.tolist()},
        "source_sha256": {
            "policy.py": _sha256(Path(__file__)),
            "runtime.py": _sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
            "features.py": _sha256(Path(__file__).parents[1] / "control" / "features.py"),
            "native.py": _sha256(Path(__file__).parents[1] / "sim" / "native.py"),
            "sensors.py": _sha256(Path(__file__).parents[1] / "sim" / "sensors.py"),
            "scorer.py": _sha256(Path(__file__).parents[1] / "sim" / "scorer.py"),
        },
        "success_fraction": float(np.mean([item["success"] for item in scores])),
        "deadline_error_median_m": float(np.median(errors)) if errors else None,
        "command_slew_median": float(np.median([item["command_slew"] for item in scores])),
        "scores": scores,
        "artifacts": artifacts,
    }
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
