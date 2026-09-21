"""Causal, versioned world-model data collection and coverage inspection."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np

from ..baselines.controller import FeedbackController
from ..config import confined_path
from ..contracts import Observation, PrivilegedRecord, RobotDescriptor, Task
from ..sim.actuators import ActuatorModel
from ..sim.mechanics import endpoint_velocity, forward_kinematics
from ..sim.native import ImperfectNativeArm, default_descriptor
from ..sim.scenarios import Scenario, make_scenario
from ..sim.sensors import SensorFault, SensorSettings, SensorSuite
from ..sim.warp_backend import WarpArmBatch
from .storage import ARRAY_FIELDS, DatasetWriter, load_array, load_manifest

COLLECTION_PROFILES = ("historical", "r3a")


def _arrays(values: list[Observation], truth: list[PrivilegedRecord], actions: list[np.ndarray],
            summaries: list[np.ndarray], *, control_dt: float) -> dict[str, np.ndarray]:
    return {
        "observation_values": np.stack([item.values for item in values]),
        "available": np.stack([item.available for item in values]),
        "fresh": np.stack([item.fresh for item in values]),
        "age_s": np.stack([item.age_s for item in values]),
        "time_s": np.arange(len(actions) + 1, dtype=np.float64) * control_dt,
        "accepted_action": np.stack(actions),
        "true_q": np.stack([item.q for item in truth]),
        "true_dq": np.stack([item.dq for item in truth]),
        "true_endpoint": np.stack([item.endpoint_xz for item in truth]),
        "true_endpoint_velocity": np.stack([item.endpoint_velocity_xz for item in truth]),
        "terminated": np.zeros(len(actions), dtype=bool),
        "truncated": np.zeros(len(actions), dtype=bool),
        "substep_summary": np.stack(summaries),
    }


def _gpu_truth(descriptor: RobotDescriptor, q: np.ndarray, dq: np.ndarray, time_s: float) -> PrivilegedRecord:
    endpoint = forward_kinematics(q, descriptor.link_lengths)
    velocity = endpoint_velocity(q, dq, descriptor.link_lengths)
    return PrivilegedRecord(q.copy(), dq.copy(), endpoint, velocity, np.zeros(3), np.zeros(4))


def _plan(episode_index: int, profile: str) -> tuple[bool, bool, bool, str]:
    if profile not in COLLECTION_PROFILES:
        raise ValueError(f"unsupported collection profile: {profile}")
    if profile == "historical":
        exploratory = episode_index % 10 >= 7
        return False, exploratory, False, (
            "smooth_bounded_exploration_with_reversals" if exploratory else "bounded_sinusoidal_baseline"
        )
    feedback = episode_index % 10 < 7
    fresh_biased_qdq = episode_index % 4 == 0
    return feedback, not feedback, fresh_biased_qdq, (
        "public_observation_feedback" if feedback else "smooth_bounded_exploration_with_reversals"
    )


def _r3a_sensor_contract(scenario: Scenario, *, fresh_biased_qdq: bool) -> tuple[SensorSettings, tuple[SensorFault, ...]]:
    """Give q/dq samples zero transport delay and optionally inject a declared bias."""

    delays = scenario.sensors.max_delay_s.copy()
    delays[:4] = 0.0
    settings = SensorSettings(scenario.sensors.periods_s, scenario.sensors.noise_std,
                              scenario.sensors.bias_limit, delays)
    faults = list(scenario.faults)
    if fresh_biased_qdq:
        faults.extend((
            SensorFault("bias", (0,), 0.0, offset=0.18),
            SensorFault("bias", (1,), 0.0, offset=-0.14),
            SensorFault("bias", (2,), 0.0, offset=0.45),
            SensorFault("bias", (3,), 0.0, offset=-0.35),
        ))
    return settings, tuple(faults)


def _json_settings(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _episode_metadata(scenario: Scenario, *, degraded: bool, exploratory: bool, feedback: bool,
                      fresh_biased_qdq: bool, profile: str, backend: str,
                      sensors: SensorSettings, faults: tuple[SensorFault, ...], target_q: np.ndarray | None,
                      target_xz: np.ndarray | None, collection_index: int) -> dict[str, object]:
    return {
        "seed": scenario.seed,
        "degraded": degraded,
        "exploratory": exploratory,
        "collection_profile": profile,
        "collection_index": collection_index,
        "collection_backend": backend,
        "collection_source": "public_feedback" if feedback else "bounded_exploration",
        "feedback_uses": "public observation q/dq only" if feedback else None,
        "fresh_zero_age_qdq": profile == "r3a",
        "fresh_biased_qdq": fresh_biased_qdq,
        "independent_reference": "endpoint tracker channels 6/7; privileged labels are evaluation-only",
        "target_q": target_q.tolist() if target_q is not None else None,
        "target_xz": target_xz.tolist() if target_xz is not None else None,
        "actuator_settings": {name: _json_settings(value) for name, value in vars(scenario.actuator).items()},
        "sensor_settings": {name: _json_settings(value) for name, value in vars(sensors).items()},
        "sensor_faults": [
            {"kind": fault.kind, "channels": list(fault.channels), "start_s": fault.start_s,
             "end_s": fault.end_s, "offset": fault.offset}
            for fault in faults
        ],
    }


def _feedback_task(descriptor: RobotDescriptor, rng: np.random.Generator, *, duration_s: float,
                   initial_observation: Observation) -> tuple[FeedbackController, np.ndarray, np.ndarray]:
    target_q = np.array([rng.uniform(-1.3, 1.3), rng.uniform(0.15, 2.2)])
    target_xz = forward_kinematics(target_q, descriptor.link_lengths)
    task = Task(target_xz, deadline_s=max(0.05, duration_s * 0.8), hold_duration_s=0.0)
    return FeedbackController(descriptor, initial_observation.values[:2], task), target_q, target_xz


def _command(*, feedback_controller: FeedbackController | None, observation: Observation, exploratory: bool,
             action_index: int, phase: np.ndarray, previous_command: np.ndarray,
             rng: np.random.Generator, time_s: float) -> np.ndarray:
    if feedback_controller is not None:
        return feedback_controller.act(time_s, observation)
    if exploratory:
        if action_index % 20 in (8, 9, 10):
            return np.zeros(2, dtype=np.float64)
        if action_index % 20 == 11:
            return -previous_command
        return np.clip(0.82 * previous_command + rng.normal(0.0, 0.20, 2), -0.75, 0.75)
    return np.array([0.35 * np.sin(action_index / 7.0 + phase[0]),
                     0.28 * np.cos(action_index / 9.0 + phase[1])])


def _collection_writer(output: str | Path, *, seed: int, backend: str, device: str, profile: str,
                       config_hash: str, physics_dt: float, control_dt: float) -> DatasetWriter:
    return DatasetWriter(output, seed=seed, config_hash=config_hash, collection={
        "backend": backend,
        "device": device,
        "profile": profile,
        "physics_dt": physics_dt,
        "control_dt": control_dt,
        "physics_device_resident": backend == "warp",
        "actuator_sensor_boundary": "cpu snapshots per physics step" if backend == "warp" else "native process",
        "source_schedule": "70% public feedback / 30% bounded exploration" if profile == "r3a" else "historical",
    })


def _collect_dataset_gpu(output: str | Path, *, seed: int, episodes: int, actions_per_episode: int,
                         physics_dt: float, control_dt: float, profile: str, config_hash: str,
                         episode_indices: tuple[int, ...]) -> Path:
    """Collect with Warp physics and an explicit CPU actuator/sensor boundary."""

    descriptor = default_descriptor()
    substeps = round(control_dt / physics_dt)
    writer = _collection_writer(output, seed=seed, backend="warp", device="cuda:0", profile=profile,
                                config_hash=config_hash, physics_dt=physics_dt, control_dt=control_dt)
    for episode_index in episode_indices:
        degraded = episode_index % 3 == 0
        scenario = make_scenario(seed, f"episode-{episode_index:06d}", degraded=degraded)
        feedback, exploratory, fresh_biased_qdq, policy_source = _plan(episode_index, profile)
        sensors_settings, faults = (_r3a_sensor_contract(scenario, fresh_biased_qdq=fresh_biased_qdq)
                                   if profile == "r3a" else (scenario.sensors, scenario.faults))
        arm = WarpArmBatch(descriptor, 1, device="cuda:0", timestep=physics_dt)
        rng = np.random.default_rng(scenario.seed)
        start = np.array([rng.uniform(-0.8, 0.8), rng.uniform(0.3, 1.6)])
        arm.reset(q=start.reshape(1, 2).astype(np.float32))
        q, dq = arm.snapshot()
        actuator = ActuatorModel(descriptor, scenario.actuator, physics_dt)
        sensors = SensorSuite(sensors_settings, seed=scenario.seed, faults=faults)
        truth = _gpu_truth(descriptor, q[0], dq[0], 0.0)
        actuator.reset()
        sensors.reset()
        sensors.advance(0.0, truth, np.zeros(2))
        values: list[Observation] = [sensors.observe(0.0)]
        if feedback:
            feedback_controller, target_q, target_xz = _feedback_task(
                descriptor, rng, duration_s=actions_per_episode * control_dt, initial_observation=values[0]
            )
        else:
            feedback_controller, target_q, target_xz = None, None, None
        truth_records = [truth]
        actions: list[np.ndarray] = []
        summaries: list[np.ndarray] = []
        phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
        previous_command = np.zeros(2, dtype=np.float64)
        for action_index in range(actions_per_episode):
            command = _command(feedback_controller=feedback_controller, observation=values[-1], exploratory=exploratory,
                               action_index=action_index, phase=phase, previous_command=previous_command, rng=rng,
                               time_s=action_index * control_dt)
            previous_command = command
            peak_speed = 0.0
            peak_torque = 0.0
            for substep in range(substeps):
                motor_torque, resistance = actuator.update(command, q[0], dq[0])
                effective = (motor_torque + resistance) / descriptor.nominal_torque_scales
                arm.step(np.clip(effective, -1.0, 1.0).reshape(1, 2).astype(np.float32))
                q, dq = arm.snapshot()
                time_s = (action_index * substeps + substep + 1) * physics_dt
                truth = _gpu_truth(descriptor, q[0], dq[0], time_s)
                sensors.advance(time_s, truth, motor_torque)
                peak_speed = max(peak_speed, float(np.linalg.norm(truth.endpoint_velocity_xz)))
                peak_torque = max(peak_torque, float(np.abs(motor_torque).max()))
            actions.append(command)
            summaries.append(np.array([peak_speed, peak_torque, 0.0]))
            truth_records.append(truth)
            values.append(sensors.observe((action_index + 1) * control_dt))
        writer.add_episode(_arrays(values, truth_records, actions, summaries, control_dt=control_dt),
                           scenario_id=scenario.scenario_id, robot_id="rigid-2r-v1",
                           scenario=_episode_metadata(
                               scenario, degraded=degraded, exploratory=exploratory, feedback=feedback,
                               fresh_biased_qdq=fresh_biased_qdq, profile=profile, backend="warp",
                               sensors=sensors_settings, faults=faults, target_q=target_q, target_xz=target_xz,
                               collection_index=episode_index,
                           ), policy_source=policy_source)
    return writer.write()


def collect_dataset(output: str | Path, *, seed: int = 0, episodes: int = 20,
                    actions_per_episode: int = 50, backend: str = "native", device: str = "cpu",
                    physics_dt: float = 0.001, control_dt: float = 0.010,
                    profile: str = "historical", config_hash: str = "",
                    episode_indices: tuple[int, ...] | None = None) -> Path:
    """Collect an immutable dataset; R3a adds explicit sensor-condition coverage."""

    if episodes < 1 or actions_per_episode < 1:
        raise ValueError("episodes and actions_per_episode must be positive")
    if profile not in COLLECTION_PROFILES:
        raise ValueError(f"unsupported collection profile: {profile}")
    if abs(control_dt / physics_dt - round(control_dt / physics_dt)) > 1e-9:
        raise ValueError("control_dt must be an integer number of physics steps")
    indices = tuple(range(episodes)) if episode_indices is None else episode_indices
    if not indices or any(index < 0 for index in indices):
        raise ValueError("episode_indices must be nonempty nonnegative indices")
    if backend == "warp" and device.startswith("cuda"):
        return _collect_dataset_gpu(output, seed=seed, episodes=episodes, actions_per_episode=actions_per_episode,
                                    physics_dt=physics_dt, control_dt=control_dt, profile=profile,
                                    config_hash=config_hash, episode_indices=indices)
    writer = _collection_writer(output, seed=seed, backend="native", device="cpu", profile=profile,
                                config_hash=config_hash, physics_dt=physics_dt, control_dt=control_dt)
    descriptor = default_descriptor()
    substeps = round(control_dt / physics_dt)
    for episode_index in indices:
        degraded = episode_index % 3 == 0
        scenario = make_scenario(seed, f"episode-{episode_index:06d}", degraded=degraded)
        feedback, exploratory, fresh_biased_qdq, policy_source = _plan(episode_index, profile)
        sensors_settings, faults = (_r3a_sensor_contract(scenario, fresh_biased_qdq=fresh_biased_qdq)
                                   if profile == "r3a" else (scenario.sensors, scenario.faults))
        arm = ImperfectNativeArm(descriptor, scenario.actuator,
                                 SensorSuite(sensors_settings, seed=scenario.seed, faults=faults), timestep=physics_dt)
        rng = np.random.default_rng(scenario.seed)
        start = np.array([rng.uniform(-0.8, 0.8), rng.uniform(0.3, 1.6)])
        arm.reset(start)
        values = [arm.observation()]
        if feedback:
            feedback_controller, target_q, target_xz = _feedback_task(
                descriptor, rng, duration_s=actions_per_episode * control_dt, initial_observation=values[0]
            )
        else:
            feedback_controller, target_q, target_xz = None, None, None
        truth = [arm.arm.truth()]
        actions: list[np.ndarray] = []
        summaries: list[np.ndarray] = []
        phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
        previous_command = np.zeros(2, dtype=np.float64)
        for action_index in range(actions_per_episode):
            command = _command(feedback_controller=feedback_controller, observation=values[-1], exploratory=exploratory,
                               action_index=action_index, phase=phase, previous_command=previous_command, rng=rng,
                               time_s=action_index * control_dt)
            previous_command = command
            peak_speed = 0.0
            peak_torque = 0.0
            for _ in range(substeps):
                current = arm.step(command)
                peak_speed = max(peak_speed, float(np.linalg.norm(current.endpoint_velocity_xz)))
                peak_torque = max(peak_torque, float(np.abs(arm.actuator.measured_motor_torque).max()))
            actions.append(command)
            summaries.append(np.array([peak_speed, peak_torque, 0.0]))
            values.append(arm.observation())
            truth.append(current)
        writer.add_episode(_arrays(values, truth, actions, summaries, control_dt=control_dt),
                           scenario_id=scenario.scenario_id, robot_id="rigid-2r-v1",
                           scenario=_episode_metadata(
                               scenario, degraded=degraded, exploratory=exploratory, feedback=feedback,
                               fresh_biased_qdq=fresh_biased_qdq, profile=profile, backend="native",
                               sensors=sensors_settings, faults=faults, target_q=target_q, target_xz=target_xz,
                               collection_index=episode_index,
                           ), policy_source=policy_source)
    return writer.write()


def reconstruct_episode(dataset: str | Path, scenario_id: str, output: str | Path) -> dict[str, object]:
    """Regenerate one episode from its manifest provenance and compare all arrays."""

    root = confined_path(dataset)
    manifest = load_manifest(root)
    matches = [episode for episode in manifest["episodes"] if episode["scenario_id"] == scenario_id]
    if len(matches) != 1:
        raise ValueError("scenario_id must identify exactly one dataset episode")
    original = matches[0]
    collection = manifest["collection"]
    regenerated_path = collect_dataset(
        output, seed=int(manifest["seed"]), episodes=1,
        actions_per_episode=int(original["lengths"]["accepted_action"]), backend=str(collection["backend"]),
        device=str(collection["device"]), physics_dt=float(collection["physics_dt"]),
        control_dt=float(collection["control_dt"]), profile=str(collection["profile"]),
        config_hash=str(manifest["config_hash"]),
        episode_indices=(int(original["scenario"]["collection_index"]),),
    )
    regenerated_root = regenerated_path.parent
    regenerated_manifest = load_manifest(regenerated_root)
    regenerated = regenerated_manifest["episodes"][0]
    equal_arrays: dict[str, bool] = {}
    for field in ARRAY_FIELDS:
        start = int(original["offsets"][field])
        count = int(original["lengths"][field])
        expected = np.asarray(load_array(root, manifest, field)[start:start + count])
        actual = np.asarray(load_array(regenerated_root, regenerated_manifest, field))
        equal_arrays[field] = bool(np.array_equal(expected, actual))
    return {
        "scenario_id": scenario_id,
        "collection_index": original["scenario"]["collection_index"],
        "regenerated_manifest": str(regenerated_path),
        "scenario_metadata_equal": original["scenario"] == regenerated["scenario"],
        "array_equality": equal_arrays,
        "reproduced": all(equal_arrays.values()) and original["scenario"] == regenerated["scenario"],
    }


def inspect_dataset(dataset: str | Path) -> dict[str, object]:
    """Report generic validity and the source coverage needed by the R3a observer claim."""

    root = confined_path(dataset)
    manifest = load_manifest(root)
    finite: dict[str, bool] = {}
    lengths = 0
    for field in ("observation_values", "accepted_action", "true_q", "true_endpoint"):
        values = load_array(root, manifest, field)
        finite[field] = bool(np.isfinite(values).all())
        lengths += int(values.shape[0])
    available = load_array(root, manifest, "available")
    fresh = load_array(root, manifest, "fresh")
    age_s = load_array(root, manifest, "age_s")
    coverage: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        episodes = [(index, episode) for index, episode in enumerate(manifest["episodes"])
                    if episode["split"] == split]
        source_counts = Counter(str(episode["scenario"].get("collection_source", "unknown"))
                                for _, episode in episodes)
        fresh_zero_age = 0
        fresh_biased_zero_age = 0
        for _, episode in episodes:
            start = int(episode["offsets"]["available"])
            count = int(episode["lengths"]["available"])
            mask = (available[start:start + count, :4] & fresh[start:start + count, :4] &
                    (age_s[start:start + count, :4] <= 1e-6))
            fresh_zero_age += int(mask.sum())
            if episode["scenario"].get("fresh_biased_qdq", False):
                fresh_biased_zero_age += int(mask.sum())
        coverage[split] = {
            "episode_count": len(episodes),
            "source_counts": dict(sorted(source_counts.items())),
            "fresh_zero_age_qdq_channels": fresh_zero_age,
            "fresh_biased_zero_age_qdq_channels": fresh_biased_zero_age,
        }
    r3a_profile = manifest.get("collection", {}).get("profile") == "r3a"

    def covered(item: dict[str, object]) -> bool:
        source_counts = cast(dict[str, int], item["source_counts"])
        fresh_biased = cast(int, item["fresh_biased_zero_age_qdq_channels"])
        return (source_counts.get("public_feedback", 0) > 0 and
                source_counts.get("bounded_exploration", 0) > 0 and fresh_biased > 0)

    r3a_complete = all(covered(item) for item in coverage.values()) if r3a_profile else None
    return {
        "dataset": str(root),
        "episode_count": len(manifest["episodes"]),
        "transition_count": sum(item["lengths"]["accepted_action"] for item in manifest["episodes"]),
        "array_samples_checked": lengths,
        "finite": finite,
        "splits": {split: sum(item["split"] == split for item in manifest["episodes"])
                   for split in ("train", "validation", "test")},
        "coverage": coverage,
        "r3a_coverage_complete": r3a_complete,
    }
