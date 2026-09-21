"""Held-out world-model evaluation with horizon-matched causal baselines."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from ..config import confined_path
from ..data.storage import load_array, load_manifest
from ..models.bundle import WorldBundle
from ..models.world import WorldModel
from ..sim.native import default_descriptor
from ..train.world import _features, computed_torque_hold_action
from ..visualize.world import export_world_trail_html

HORIZONS = (1, 5, 20)
Q_SCALE_RAD = float(np.pi)
DQ_SCALE_RAD_S = 5.0
HISTORY_DIAGNOSTIC_STRIDE = 8
MAX_ONE_STEP_REGRESSION_FRACTION = 0.10


def _constant_position_prediction(q: Tensor) -> tuple[Tensor, Tensor]:
    """Hold public/oracle position and declare zero velocity at every horizon."""

    return q, torch.zeros_like(q)


def _constant_velocity_prediction(q: Tensor, dq: Tensor, *, horizon: int, dt: float) -> tuple[Tensor, Tensor]:
    """Advance q with constant dq for exactly ``horizon`` control intervals."""

    return q + dq * (horizon * dt), dq


def _state_errors(predicted_q: Tensor, predicted_dq: Tensor, target_q: Tensor,
                  target_dq: Tensor) -> dict[str, Tensor]:
    """Return physical and common normalized q/dq state errors."""

    q_error = torch.linalg.vector_norm(predicted_q - target_q)
    dq_error = torch.linalg.vector_norm(predicted_dq - target_dq)
    return {
        "q_error_mean_rad": q_error,
        "dq_error_mean_rad_s": dq_error,
        "normalized_state_error": 0.5 * (q_error / Q_SCALE_RAD + dq_error / DQ_SCALE_RAD_S),
    }


def _empty_state_metrics(*, physical: bool = False) -> dict[str, list[Tensor]]:
    metrics: dict[str, list[Tensor]] = {
        name: [] for name in ("q_error_mean_rad", "dq_error_mean_rad_s", "normalized_state_error")
    }
    if physical:
        metrics.update({"endpoint_position_error_mean_m": [], "endpoint_velocity_error_mean_m_s": []})
    return metrics


def _append_state(metrics: dict[str, list[Tensor]], predicted_q: Tensor, predicted_dq: Tensor,
                  target_q: Tensor, target_dq: Tensor) -> None:
    for name, value in _state_errors(predicted_q, predicted_dq, target_q, target_dq).items():
        metrics[name].append(value)


def _append_endpoint(metrics: dict[str, list[Tensor]], predicted_endpoint: Tensor,
                     predicted_velocity: Tensor, target_endpoint: Tensor,
                     target_velocity: Tensor) -> None:
    metrics["endpoint_position_error_mean_m"].append(torch.linalg.vector_norm(predicted_endpoint - target_endpoint))
    metrics["endpoint_velocity_error_mean_m_s"].append(torch.linalg.vector_norm(predicted_velocity - target_velocity))


def _summarize(metrics: dict[str, list[Tensor]]) -> dict[str, float | int | None]:
    count = len(next(iter(metrics.values()))) if metrics else 0
    result: dict[str, float | int | None] = {"anchor_count": count}
    for name, values in metrics.items():
        result[name] = float(torch.stack(values).mean().detach().cpu()) if values else None
    return result


def _prediction_gate(horizons: dict[str, dict[str, dict[str, float | int | None]]]) -> dict[str, float | bool | str | None]:
    """Apply the locked P5 point-estimate gate to common normalized metrics."""

    one_step = horizons["1"]
    twenty_step = horizons["20"]
    one_learned = one_step["learned_open_loop"]["normalized_state_error"]
    one_baseline = one_step["oracle_constant_velocity"]["normalized_state_error"]
    twenty_learned = twenty_step["learned_open_loop"]["normalized_state_error"]
    twenty_baseline = twenty_step["oracle_constant_velocity"]["normalized_state_error"]
    values = (one_learned, one_baseline, twenty_learned, twenty_baseline)
    if not all(isinstance(value, float) and value > 0.0 for value in values):
        return {"status": "inconclusive", "accepted": False, "twenty_step_improvement_fraction": None,
                "one_step_regression_fraction": None,
                "maximum_one_step_regression_fraction": MAX_ONE_STEP_REGRESSION_FRACTION}
    one_learned_value, one_baseline_value, twenty_learned_value, twenty_baseline_value = cast(
        tuple[float, float, float, float], values
    )
    improvement = 1.0 - twenty_learned_value / twenty_baseline_value
    regression = one_learned_value / one_baseline_value - 1.0
    accepted = improvement >= 0.20 and regression <= MAX_ONE_STEP_REGRESSION_FRACTION
    return {"status": "pass" if accepted else "fail", "accepted": accepted,
            "twenty_step_improvement_fraction": improvement, "one_step_regression_fraction": regression,
            "maximum_one_step_regression_fraction": MAX_ONE_STEP_REGRESSION_FRACTION}


def _control_dt(time_s: np.ndarray) -> float:
    deltas = np.diff(np.asarray(time_s, dtype=np.float64))
    if len(deltas) == 0 or not np.isfinite(deltas).all() or (deltas <= 0.0).any():
        raise ValueError("episode time_s must contain strictly increasing control timestamps")
    dt = float(deltas[0])
    if not np.allclose(deltas, dt, rtol=0.0, atol=1e-9):
        raise ValueError("world evaluation requires a fixed control interval per episode")
    return dt


def _shuffled_history_belief(model: WorldModel, features: Tensor, actions: Tensor, descriptor: Tensor,
                              *, index: int, seed: int, dt: float) -> Tensor:
    """Rebuild a belief with past packets shuffled but current packet unchanged.

    Actions remain chronological. The diagnostic deliberately makes packet history
    incoherent without allowing a future packet into the posterior at ``index``.
    It is an ablation, not a causal controller state.
    """

    if index == 0:
        return model.initial(features[0], descriptor)
    order = np.random.default_rng(seed).permutation(index)
    belief = model.initial(features[int(order[0])], descriptor)
    for previous in range(1, index):
        prior = model.predict_prior(belief, actions[previous - 1], descriptor, dt)
        belief = model.observe(prior, features[int(order[previous])], descriptor)
    prior = model.predict_prior(belief, actions[index - 1], descriptor, dt)
    return model.observe(prior, features[index], descriptor)


def _selected_episodes(manifest: dict[str, Any], split: str) -> Iterable[tuple[int, dict[str, Any]]]:
    for episode_index, episode in enumerate(manifest["episodes"]):
        if episode["split"] == split:
            yield episode_index, episode


def export_endpoint_trail(bundle: str | Path, dataset: str | Path, output: str | Path, *, split: str,
                          device: str, horizon: int = 20) -> dict[str, object]:
    """Export one held-out actual/predicted endpoint trail without future observations."""

    if horizon not in HORIZONS:
        raise ValueError(f"unsupported trail horizon: {horizon}")
    device_obj = torch.device(device)
    loaded = WorldBundle.load(bundle, device=device_obj)
    manifest = load_manifest(dataset)
    root = confined_path(dataset)
    selected = next(iter(_selected_episodes(manifest, split)), None)
    if selected is None:
        raise ValueError(f"dataset has no {split} episode")
    _, episode = selected
    arrays = {field: load_array(root, manifest, field) for field in
              ("observation_values", "available", "fresh", "age_s", "time_s", "accepted_action", "true_endpoint")}
    start = int(episode["offsets"]["observation_values"])
    action_start = int(episode["offsets"]["accepted_action"])
    count = int(episode["lengths"]["accepted_action"])
    if count < horizon:
        raise ValueError("selected trail episode is shorter than the requested horizon")
    normalization = loaded.metadata["normalization"]
    values = np.array(arrays["observation_values"][start:start + count + 1], dtype=np.float32, copy=True)
    available = np.array(arrays["available"][start:start + count + 1], dtype=bool, copy=True)
    fresh = np.array(arrays["fresh"][start:start + count + 1], dtype=bool, copy=True)
    age = np.array(arrays["age_s"][start:start + count + 1], dtype=np.float32, copy=True)
    time_s = np.array(arrays["time_s"][start:start + count + 1], dtype=np.float64, copy=True)
    dt = _control_dt(time_s)
    features = _features(values, available, fresh, age, np.asarray(normalization["mean"], dtype=np.float32),
                         np.asarray(normalization["std"], dtype=np.float32), device_obj)
    actions = torch.as_tensor(np.array(arrays["accepted_action"][action_start:action_start + count], copy=True),
                              dtype=torch.float32, device=device_obj)
    target_endpoint = np.array(arrays["true_endpoint"][start:start + count + 1], dtype=np.float32, copy=True)
    descriptor = torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32, device=device_obj)
    anchors: list[float] = []
    targets: list[float] = []
    actual: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    belief = loaded.model.initial(features[0], descriptor)
    with torch.no_grad():
        for index in range(count):
            if index + horizon <= count:
                rollout = belief
                for offset in range(horizon):
                    rollout = loaded.model.predict_prior(rollout, actions[index + offset], descriptor, dt)
                _, _, endpoint, _ = loaded.model.decode_physical(rollout, descriptor)
                anchors.append(float(time_s[index]))
                targets.append(float(time_s[index + horizon]))
                actual.append(target_endpoint[index + horizon])
                predicted.append(endpoint.detach().cpu().numpy())
            belief = loaded.model.predict_prior(belief, actions[index], descriptor, dt)
            belief = loaded.model.observe(belief, features[index + 1], descriptor)
    trail: dict[str, np.ndarray | int | str] = {
        "scenario_id": str(episode["scenario_id"]), "horizon_control_steps": horizon,
        "anchor_time_s": np.asarray(anchors), "target_time_s": np.asarray(targets),
        "actual_endpoint_xz": np.asarray(actual), "predicted_endpoint_xz": np.asarray(predicted),
    }
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, anchor_time_s=trail["anchor_time_s"], target_time_s=trail["target_time_s"],
             actual_endpoint_xz=trail["actual_endpoint_xz"], predicted_endpoint_xz=trail["predicted_endpoint_xz"])
    html = export_world_trail_html(trail, destination.with_suffix(".html"))
    return {"npz": str(destination), "html": str(html), "scenario_id": trail["scenario_id"],
            "horizon_control_steps": horizon, "anchor_count": len(anchors)}


def evaluate_world(bundle: str | Path, dataset: str | Path, output: str | Path, *, split: str = "validation",
                   device: str = "cpu", trail_output: str | Path | None = None) -> dict[str, Any]:
    """Evaluate open-loop prediction from a posterior using fixed held-out traces."""

    device_obj = torch.device(device)
    loaded = WorldBundle.load(bundle, device=device_obj)
    manifest = load_manifest(dataset)
    root = confined_path(dataset)
    arrays = {field: load_array(root, manifest, field) for field in
              ("observation_values", "available", "fresh", "age_s", "time_s", "accepted_action", "true_q",
               "true_dq", "true_endpoint", "true_endpoint_velocity")}
    normalization = loaded.metadata["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    descriptor = torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32, device=device_obj)
    model = loaded.model
    learned = {horizon: _empty_state_metrics(physical=True) for horizon in HORIZONS}
    oracle_position = {horizon: _empty_state_metrics() for horizon in HORIZONS}
    oracle_velocity = {horizon: _empty_state_metrics() for horizon in HORIZONS}
    public_velocity = {horizon: _empty_state_metrics() for horizon in HORIZONS}
    belief_velocity = {horizon: _empty_state_metrics() for horizon in HORIZONS}
    posterior = _empty_state_metrics(physical=True)
    reset_history = _empty_state_metrics()
    shuffled_history = _empty_state_metrics()
    episode_dts: list[float] = []
    control_action_errors: list[Tensor] = []

    with torch.no_grad():
        for episode_index, episode in _selected_episodes(manifest, split):
            obs_start = int(episode["offsets"]["observation_values"])
            action_start = int(episode["offsets"]["accepted_action"])
            count = int(episode["lengths"]["accepted_action"])
            if count < max(HORIZONS):
                continue
            values = np.array(arrays["observation_values"][obs_start:obs_start + count + 1], dtype=np.float32, copy=True)
            available = np.array(arrays["available"][obs_start:obs_start + count + 1], dtype=bool, copy=True)
            fresh = np.array(arrays["fresh"][obs_start:obs_start + count + 1], dtype=bool, copy=True)
            age = np.array(arrays["age_s"][obs_start:obs_start + count + 1], dtype=np.float32, copy=True)
            time_s = np.array(arrays["time_s"][obs_start:obs_start + count + 1], dtype=np.float64, copy=True)
            dt = _control_dt(time_s)
            episode_dts.append(dt)
            features = _features(values, available, fresh, age, mean, std, device_obj)
            public_values = torch.as_tensor(values, dtype=torch.float32, device=device_obj)
            public_available = torch.as_tensor(available, dtype=torch.bool, device=device_obj)
            actions = torch.as_tensor(np.array(arrays["accepted_action"][action_start:action_start + count], copy=True),
                                      dtype=torch.float32, device=device_obj)
            target_q = torch.as_tensor(np.array(arrays["true_q"][obs_start:obs_start + count + 1], copy=True),
                                       dtype=torch.float32, device=device_obj)
            target_dq = torch.as_tensor(np.array(arrays["true_dq"][obs_start:obs_start + count + 1], copy=True),
                                        dtype=torch.float32, device=device_obj)
            target_endpoint = torch.as_tensor(
                np.array(arrays["true_endpoint"][obs_start:obs_start + count + 1], copy=True),
                dtype=torch.float32, device=device_obj)
            target_endpoint_velocity = torch.as_tensor(
                np.array(arrays["true_endpoint_velocity"][obs_start:obs_start + count + 1], copy=True),
                dtype=torch.float32, device=device_obj)
            scenario = episode["scenario"]
            goal_q_values = scenario.get("target_q") if not scenario.get("degraded", True) else None
            control_goal = (None if goal_q_values is None else torch.as_tensor(goal_q_values, dtype=torch.float32,
                                                                                device=device_obj))
            belief = model.initial(features[0], descriptor)
            for index in range(count):
                decoded_q, decoded_dq = model.decode(belief)
                _append_state(posterior, decoded_q, decoded_dq, target_q[index], target_dq[index])
                _, _, decoded_endpoint, decoded_endpoint_velocity = model.decode_physical(belief, descriptor)
                _append_endpoint(posterior, decoded_endpoint, decoded_endpoint_velocity,
                                 target_endpoint[index], target_endpoint_velocity[index])
                if control_goal is not None:
                    decoded_action = computed_torque_hold_action(decoded_q, decoded_dq, control_goal, descriptor)
                    truth_action = computed_torque_hold_action(target_q[index], target_dq[index], control_goal, descriptor)
                    control_action_errors.append(torch.mean((decoded_action - truth_action) ** 2))
                if index % HISTORY_DIAGNOSTIC_STRIDE == 0 or index == count - 1:
                    reset_belief = model.initial(features[index], descriptor)
                    reset_prior = model.predict_prior(reset_belief, actions[index], descriptor, dt)
                    reset_q, reset_dq = model.decode(reset_prior)
                    _append_state(reset_history, reset_q, reset_dq, target_q[index + 1], target_dq[index + 1])
                    shuffled_belief = _shuffled_history_belief(
                        model, features, actions, descriptor, index=index,
                        seed=20260912 + episode_index * 1000 + index, dt=dt,
                    )
                    shuffled_prior = model.predict_prior(shuffled_belief, actions[index], descriptor, dt)
                    shuffled_q, shuffled_dq = model.decode(shuffled_prior)
                    _append_state(shuffled_history, shuffled_q, shuffled_dq,
                                  target_q[index + 1], target_dq[index + 1])
                for horizon in HORIZONS:
                    if index + horizon > count:
                        continue
                    rollout = belief
                    for offset in range(horizon):
                        rollout = model.predict_prior(rollout, actions[index + offset], descriptor, dt)
                    predicted_q, predicted_dq = model.decode(rollout)
                    _append_state(learned[horizon], predicted_q, predicted_dq,
                                  target_q[index + horizon], target_dq[index + horizon])
                    _, _, predicted_endpoint, predicted_endpoint_velocity = model.decode_physical(rollout, descriptor)
                    _append_endpoint(learned[horizon], predicted_endpoint, predicted_endpoint_velocity,
                                     target_endpoint[index + horizon], target_endpoint_velocity[index + horizon])
                    position_q, position_dq = _constant_position_prediction(target_q[index])
                    _append_state(oracle_position[horizon], position_q, position_dq,
                                  target_q[index + horizon], target_dq[index + horizon])
                    velocity_q, velocity_dq = _constant_velocity_prediction(
                        target_q[index], target_dq[index], horizon=horizon, dt=dt,
                    )
                    _append_state(oracle_velocity[horizon], velocity_q, velocity_dq,
                                  target_q[index + horizon], target_dq[index + horizon])
                    public_q = torch.where(public_available[index, :2], public_values[index, :2],
                                           torch.zeros_like(public_values[index, :2]))
                    public_dq = torch.where(public_available[index, 2:4], public_values[index, 2:4],
                                            torch.zeros_like(public_values[index, 2:4]))
                    public_q, public_dq = _constant_velocity_prediction(public_q, public_dq, horizon=horizon, dt=dt)
                    _append_state(public_velocity[horizon], public_q, public_dq,
                                  target_q[index + horizon], target_dq[index + horizon])
                    belief_q, belief_dq = _constant_velocity_prediction(
                        decoded_q, decoded_dq, horizon=horizon, dt=dt,
                    )
                    _append_state(belief_velocity[horizon], belief_q, belief_dq,
                                  target_q[index + horizon], target_dq[index + horizon])
                prior = model.predict_prior(belief, actions[index], descriptor, dt)
                belief = model.observe(prior, features[index + 1], descriptor)

    if not episode_dts:
        raise ValueError(f"dataset has no {split} episodes with {max(HORIZONS)} actions")
    if not np.allclose(episode_dts, episode_dts[0], rtol=0.0, atol=1e-9):
        raise ValueError("world evaluation requires one fixed control_dt across selected episodes")
    horizon_report = {
        str(horizon): {
            "learned_open_loop": _summarize(learned[horizon]),
            "oracle_constant_position": _summarize(oracle_position[horizon]),
            "oracle_constant_velocity": _summarize(oracle_velocity[horizon]),
            "public_constant_velocity": _summarize(public_velocity[horizon]),
            "belief_constant_velocity": _summarize(belief_velocity[horizon]),
        }
        for horizon in HORIZONS
    }
    report = {
        "schema_version": 2,
        "bundle": str(confined_path(bundle)),
        "dataset": str(root),
        "split": split,
        "episodes": len(episode_dts),
        "evaluation_contract": {
            "control_dt_s": episode_dts[0],
            "horizons_control_steps": list(HORIZONS),
            "normalized_state_error": "0.5 * (norm(q_error)/pi + norm(dq_error)/5)",
            "open_loop": "posterior at t and recorded accepted actions t through t+horizon-1; no later observations",
            "descriptor_source": "default_descriptor rigid-2r-v1; dataset schema v1 has no per-episode descriptor",
            "model_inference": {
                "bundle_schema_version": loaded.metadata["schema_version"],
                "physics_prior": model.physics_prior,
                "fresh_measurement_correction": model.fresh_measurement_correction,
            },
            "oracle_baselines": "true q/dq at the anchor, diagnostics only; not deployable",
            "public_baseline": "observed q/dq at the anchor with unavailable channels replaced by zero",
            "constant_position": "q held at anchor and dq set to zero",
            "constant_velocity": "q advanced from anchor dq, with dq held constant",
            "p5_point_gate": "20-step improvement >=20% versus oracle constant velocity and one-step regression <=10%",
        },
        "posterior_estimation": _summarize(posterior),
        "horizons": horizon_report,
        "p5_point_gate": _prediction_gate(horizon_report),
        "history_diagnostics": {
            "horizon_control_steps": 1,
            "sample_stride": HISTORY_DIAGNOSTIC_STRIDE,
            "reset_to_current_observation": _summarize(reset_history),
            "shuffled_past_packets_current_packet_fixed": _summarize(shuffled_history),
        },
        "control_ready_bridge": {
            "contract": "MSE of goal-hold computed-torque actions from frozen decoded versus true q/dq; healthy feedback episodes only",
            "anchor_count": len(control_action_errors),
            "action_mse": (float(torch.stack(control_action_errors).mean().detach().cpu())
                           if control_action_errors else None),
            "maximum_action_mse": 0.02,
            "accepted": (bool(torch.stack(control_action_errors).mean() <= 0.02)
                         if control_action_errors else False),
        },
    }
    if trail_output is not None:
        report["endpoint_trail"] = export_endpoint_trail(bundle, dataset, trail_output, split=split, device=device)
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
