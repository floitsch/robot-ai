"""Healthy feasible suite for the ordinary controller."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..baselines.controller import FeedbackController
from ..config import RuntimeConfig, confined_path
from ..contracts import Task
from ..sim.env import RigidArmEnvironment
from ..sim.native import ImperfectNativeArm, default_descriptor
from ..sim.scenarios import make_scenario
from ..sim.scorer import joint_limit_summary, joint_limit_violations, score_episode
from ..sim.sensors import SensorSuite
from ..visualize.replay import export_gif, export_html

DEVELOPMENT_CASE_MANIFEST = "configs/healthy-development-v1.json"


def load_healthy_case_manifest(value: str | Path, *, expected_count: int | None = None,
                               expected_seed: int | None = None) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Load one versioned healthy packet-evaluation manifest without changing case IDs."""

    path = confined_path(value, must_exist=True)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1:
        raise ValueError("healthy case manifest has an unsupported schema")
    if expected_seed is not None and manifest.get("seed") != expected_seed:
        raise ValueError("healthy case manifest has an unsupported seed")
    cases = [(np.asarray(item["start_q"], dtype=np.float64), np.asarray(item["goal_q"], dtype=np.float64))
             for item in manifest["cases"]]
    ids = [item.get("id") for item in manifest["cases"]]
    if (expected_count is not None and len(cases) != expected_count) or not cases or len(set(ids)) != len(ids) or \
            any(not isinstance(case_id, str) or start.shape != (2,) or goal.shape != (2,) for case_id, (start, goal) in zip(ids, cases, strict=True)):
        raise ValueError("healthy case manifest must contain uniquely named two-joint cases")
    return cases, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "case_count": len(cases)}


def _manifest_cases() -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    return load_healthy_case_manifest(DEVELOPMENT_CASE_MANIFEST, expected_count=64, expected_seed=20260911)


def healthy_cases(count: int = 64, *, seed: int = 20260911) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return frozen development cases or deterministic non-development cases."""

    if seed == 20260911:
        cases, _ = _manifest_cases()
        if count > len(cases):
            raise ValueError("healthy development suite has exactly 64 frozen cases")
        return cases[:count]

    generator = np.random.default_rng(seed)
    starts = np.column_stack((generator.uniform(-0.8, 0.8, count), generator.uniform(0.3, 1.6, count)))
    goals = np.column_stack((generator.uniform(-0.8, 0.8, count), generator.uniform(0.3, 1.6, count)))
    return [(starts[i], goals[i]) for i in range(count)]


def healthy_case_manifest_identity() -> dict[str, object]:
    """Return the immutable identity of the frozen 64-case development suite."""

    return _manifest_cases()[1]


def run_baseline_suite(output: str | Path, *, config: RuntimeConfig, count: int = 64,
                       kp: np.ndarray | None = None, kd: np.ndarray | None = None,
                       computed_torque: bool = False, hold_goal: bool = False,
                       case_index: int | None = None) -> dict[str, object]:
    """Evaluate the ordinary controller under the policy's control/physics clock."""

    substeps = round(config.control_dt / config.physics_dt)
    if substeps < 1 or not np.isclose(substeps * config.physics_dt, config.control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    descriptor = default_descriptor()
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, object]] = []
    demo_data: dict[str, np.ndarray | float | str] | None = None
    environment = RigidArmEnvironment(descriptor, backend=config.backend,
                                     device="cuda:0" if config.device == "cuda" else "cpu",
                                     world_count=1, physics_dt=config.physics_dt)
    all_cases = healthy_cases(count)
    if case_index is not None and not 0 <= case_index < len(all_cases):
        raise ValueError(f"baseline case index must be in [0, {len(all_cases) - 1}]")
    cases = list(enumerate(all_cases)) if case_index is None else [(case_index, all_cases[case_index])]
    for scored_case_index, (start, goal_q) in cases:
        target = np.array([
            descriptor.link_lengths[0] * np.sin(goal_q[0]) + descriptor.link_lengths[1] * np.sin(goal_q.sum()),
            -descriptor.link_lengths[0] * np.cos(goal_q[0]) - descriptor.link_lengths[1] * np.cos(goal_q.sum()),
        ])
        task = Task(target, deadline_s=1.5)
        reset = environment.reset(q=start.reshape(1, 2))
        controller = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0 else -1,
                                        kp=kp, kd=kd, computed_torque=computed_torque, hold_goal=hold_goal)
        controls = round(task.total_duration_s / config.control_dt)
        if not np.isclose(controls * config.control_dt, task.total_duration_s):
            raise ValueError("task duration must be an integer number of control ticks")
        times = [0.0]
        endpoints = [reset.truth[0].endpoint_xz.copy()]
        velocities = [reset.truth[0].endpoint_velocity_xz.copy()]
        q_values = [reset.truth[0].q.copy()]
        dq_values = [reset.truth[0].dq.copy()]
        commands: list[np.ndarray] = []
        result = reset
        for control_index in range(controls):
            command = controller.act(control_index * config.control_dt, result.observations[0])
            commands.append(command.copy())
            for _ in range(substeps):
                result = environment.step(command.reshape(1, 2))
                truth = result.truth[0]
                times.append(times[-1] + config.physics_dt)
                endpoints.append(truth.endpoint_xz.copy())
                velocities.append(truth.endpoint_velocity_xz.copy())
                q_values.append(truth.q.copy())
                dq_values.append(truth.dq.copy())
        times_array = np.asarray(times)
        q_array = np.asarray(q_values)
        violations = joint_limit_violations(q_array, descriptor.joint_limits)
        score = score_episode(times_array, np.asarray(endpoints), np.asarray(velocities), task,
                              hard_violation=violations, max_sample_interval_s=config.physics_dt)
        score_dict = dict(score.__dict__)
        score_dict.update(joint_limit_summary(times_array, q_array, descriptor.joint_limits))
        score_dict["case_index"] = scored_case_index
        scores.append(score_dict)
        if demo_data is None:
            demo_data = {
                "q": np.asarray(q_values), "dq": np.asarray(dq_values), "endpoint": np.asarray(endpoints),
                "endpoint_velocity": np.asarray(velocities), "accepted_action": np.asarray(commands),
                "time_s": times_array, "joint_limit_violation": violations, "target": target,
                "link_length_1": descriptor.link_lengths[0], "link_length_2": descriptor.link_lengths[1],
            }
    assert demo_data is not None
    np.savez(destination / "demo.npz", **demo_data)  # type: ignore[arg-type]
    export_html(demo_data, destination / "demo.html")
    export_gif(demo_data, destination / "demo.gif", stride=5)
    success_fraction = float(np.mean([bool(score["success"]) for score in scores]))
    report = {
        "controller": "baseline",
        "suite": "healthy-development-v2",
        "scenario_count": len(cases), "requested_case_index": case_index,
        "backend": config.backend,
        "device": config.device,
        "physics_dt": config.physics_dt,
        "control_dt": config.control_dt,
        "command_rate_hz": 1.0 / config.control_dt,
        "substeps_per_command": substeps,
        "controller_gains": {"kp": controller.kp.tolist(), "kd": controller.kd.tolist(),
                               "computed_torque": controller.computed_torque, "hold_goal": controller.hold_goal},
        "case_manifest": healthy_case_manifest_identity(),
        "evaluated_case_ids": [f"healthy-development-{index:03d}" for index, _ in cases],
        "source_hashes": {
            "baseline.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "controller.py": hashlib.sha256((Path(__file__).parents[1] / "baselines" / "controller.py").read_bytes()).hexdigest(),
            "mechanics.py": hashlib.sha256((Path(__file__).parents[1] / "sim" / "mechanics.py").read_bytes()).hexdigest(),
            "scorer.py": hashlib.sha256((Path(__file__).parents[1] / "sim" / "scorer.py").read_bytes()).hexdigest(),
        },
        "trace_duration_s": task.total_duration_s,
        "success_fraction": success_fraction,
        "modeled_hard_limits": ["joint_position"],
        "deadline_error_median_m": float(np.median([score["deadline_error_m"] for score in scores if score["deadline_error_m"] is not None])),
        "deadline_speed_median_m_s": float(np.median([score["deadline_speed_m_s"] for score in scores if score["deadline_speed_m_s"] is not None])),
        "scores": scores,
        "artifacts": {"trajectory": str(destination / "demo.npz"), "html": str(destination / "demo.html"), "gif": str(destination / "demo.gif")},
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_imperfect_reference_case(output: str | Path, *, config: RuntimeConfig,
                                 scenario_id: str = "r4-imperfect-reference") -> dict[str, object]:
    """Record one scored native sensor/motor-imperfect task for offline inspection.

    The native imperfect arm is the available implementation of these P3 effects.
    Its CPU backend is recorded explicitly; this is a visualization fixture, not a
    substitute for a CUDA learning or evaluation gate.
    """

    if config.scenario != "degraded":
        raise ValueError("imperfect reference requires a degraded-scenario config")
    substeps = round(config.control_dt / config.physics_dt)
    if substeps < 1 or not np.isclose(substeps * config.physics_dt, config.control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    descriptor = default_descriptor()
    start, goal_q = healthy_cases(1)[0]
    target = np.array([
        descriptor.link_lengths[0] * np.sin(goal_q[0]) + descriptor.link_lengths[1] * np.sin(goal_q.sum()),
        -descriptor.link_lengths[0] * np.cos(goal_q[0]) - descriptor.link_lengths[1] * np.cos(goal_q.sum()),
    ])
    task = Task(target, deadline_s=1.5)
    scenario = make_scenario(config.seed, scenario_id, degraded=True)
    arm = ImperfectNativeArm(
        descriptor, scenario.actuator,
        SensorSuite(scenario.sensors, seed=scenario.seed, faults=scenario.faults),
        timestep=config.physics_dt,
    )
    arm.reset(start)
    controller = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0 else -1)
    times = [0.0]
    truth = arm.arm.truth()
    endpoints, velocities = [truth.endpoint_xz.copy()], [truth.endpoint_velocity_xz.copy()]
    q_values, dq_values, commands = [truth.q.copy()], [truth.dq.copy()], []
    controls = round(task.total_duration_s / config.control_dt)
    for control_index in range(controls):
        command = controller.act(control_index * config.control_dt, arm.observation())
        commands.append(command.copy())
        for _ in range(substeps):
            truth = arm.step(command)
            times.append(times[-1] + config.physics_dt)
            endpoints.append(truth.endpoint_xz.copy())
            velocities.append(truth.endpoint_velocity_xz.copy())
            q_values.append(truth.q.copy())
            dq_values.append(truth.dq.copy())
    times_array = np.asarray(times)
    q_array = np.asarray(q_values)
    violations = joint_limit_violations(q_array, descriptor.joint_limits)
    score = score_episode(times_array, np.asarray(endpoints), np.asarray(velocities), task,
                          hard_violation=violations, max_sample_interval_s=config.physics_dt)
    score_dict = dict(score.__dict__)
    score_dict.update(joint_limit_summary(times_array, q_array, descriptor.joint_limits))
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    demo_data: dict[str, np.ndarray | float | str] = {
        "q": np.asarray(q_values), "dq": np.asarray(dq_values), "endpoint": np.asarray(endpoints),
        "endpoint_velocity": np.asarray(velocities), "accepted_action": np.asarray(commands),
        "time_s": times_array, "joint_limit_violation": violations, "target": target, "link_length_1": descriptor.link_lengths[0],
        "link_length_2": descriptor.link_lengths[1],
    }
    np.savez(destination / "demo.npz", **demo_data)  # type: ignore[arg-type]
    export_html(demo_data, destination / "demo.html")
    export_gif(demo_data, destination / "demo.gif", stride=max(1, substeps // 2))
    report: dict[str, object] = {
        "controller": "ordinary-feedback-reference", "suite": "P3-degraded-single-case",
        "scenario_id": scenario_id, "scenario_seed": scenario.seed, "backend": "native",
        "device": "cpu", "physics_dt": config.physics_dt, "control_dt": config.control_dt,
        "command_rate_hz": 1.0 / config.control_dt, "substeps_per_command": substeps,
        "trace_duration_s": task.total_duration_s, "modeled_hard_limits": ["joint_position"],
        "success_fraction": float(score.success), "deadline_error_median_m": score.deadline_error_m,
        "scores": [score_dict],
        "artifacts": {"trajectory": str(destination / "demo.npz"),
                      "html": str(destination / "demo.html"), "gif": str(destination / "demo.gif")},
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
