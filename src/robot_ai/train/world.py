"""Supervised recurrent world-model training for the bounded P5 experiment."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..config import confined_path
from ..data.sequences import SequenceWindow, load_window, windows
from ..data.storage import load_manifest
from ..models.bundle import WorldBundle
from ..models.world import WorldModel
from ..sim.native import default_descriptor


def _features(values: np.ndarray, available: np.ndarray, fresh: np.ndarray, age: np.ndarray,
              mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    normalized = (values - mean) / std
    result = np.concatenate((normalized, available.astype(np.float32), fresh.astype(np.float32), np.clip(age / 0.1, 0.0, 10.0)), axis=-1)
    return torch.as_tensor(result, dtype=torch.float32, device=device)


def computed_torque_hold_action(q: torch.Tensor, dq: torch.Tensor, goal_q: torch.Tensor,
                                descriptor: torch.Tensor) -> torch.Tensor:
    """Public-state computed-torque goal-hold action used by the R3b2 bridge."""

    l1, l2 = descriptor[..., 0], descriptor[..., 1]
    m1, m2 = descriptor[..., 2], descriptor[..., 3]
    torque_scales = descriptor[..., 6:8]
    qsum = q[..., 0] + q[..., 1]
    gravity_1 = m1 * l1 * 0.5 + m2 * l1
    gravity_2 = m2 * l2 * 0.5
    gravity = -9.81 * torch.stack((gravity_1 * torch.sin(q[..., 0]) + gravity_2 * torch.sin(qsum),
                                   gravity_2 * torch.sin(qsum)), dim=-1)
    coupling = -m2 * l1 * l2 * 0.5 * torch.sin(q[..., 1])
    coriolis = torch.stack((coupling * (2 * dq[..., 0] * dq[..., 1] + dq[..., 1] ** 2),
                            -coupling * dq[..., 0] ** 2), dim=-1)
    torque = (torch.as_tensor([8.0, 6.0], dtype=q.dtype, device=q.device) * (goal_q - q)
              - torch.as_tensor([0.7, 0.56], dtype=q.dtype, device=q.device) * dq
              - gravity + coriolis + 0.02 * dq)
    return torch.clamp(torque / torque_scales, -1.0, 1.0)


def train_world(dataset: str | Path, output: str | Path, *, device: str = "cuda", updates: int = 2000,
                seed: int = 0, endpoint_residual: bool = False, physics_prior: bool = False,
                overfit_scenario_id: str | None = None, control_action_weight: float = 0.0) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA world training but CUDA is unavailable")
    if control_action_weight < 0.0:
        raise ValueError("control_action_weight must be nonnegative")
    manifest = load_manifest(dataset)
    root = confined_path(dataset)
    normalization = json.loads((root / manifest["normalization"]).read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    descriptor = np.asarray(default_descriptor().as_array(), dtype=np.float32)
    fresh_measurement_correction = physics_prior
    model = WorldModel(endpoint_residual=endpoint_residual, physics_prior=physics_prior,
                       fresh_measurement_correction=fresh_measurement_correction).to(device_obj)
    model.set_observation_normalization(torch.as_tensor(mean, device=device_obj),
                                        torch.as_tensor(std, device=device_obj))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    training_windows = list(windows(manifest, split="train", burn_in=32, loss_length=64))
    selected_episode_indices: set[int] | None = None
    if overfit_scenario_id is not None:
        selected_episode_indices = {
            index for index, episode in enumerate(manifest["episodes"])
            if episode["scenario_id"] == overfit_scenario_id and episode["split"] == "train"
        }
        if len(selected_episode_indices) != 1:
            raise ValueError("overfit scenario must identify exactly one training episode")
        training_windows = [window for window in training_windows if window.episode_index in selected_episode_indices]
    window_mode = "burn-in-32-loss-64"
    if not training_windows:
        window_mode = "full-episode-fallback"
        training_windows = [SequenceWindow(index, episode["split"], 0, 0,
                                           int(episode["lengths"]["accepted_action"]))
                            for index, episode in enumerate(manifest["episodes"])
                            if episode["split"] == "train" and
                            (selected_episode_indices is None or index in selected_episode_indices)]
    if not training_windows:
        raise ValueError("dataset has no training episodes")
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for update in range(updates):
        # Uniform sampling prevents a bounded run from fitting only the first
        # few episodes when long episodes produce many overlapping windows.
        window = training_windows[int(np.random.randint(len(training_windows)))]
        data = load_window(root, manifest, window)
        action_count = len(data["accepted_action"])
        obs_values = data["observation_values"]
        obs_available = data["available"]
        obs_fresh = data["fresh"]
        obs_age = data["age_s"]
        actions = data["accepted_action"]
        true_q = data["true_q"]
        true_dq = data["true_dq"]
        true_endpoint = data["true_endpoint"]
        true_endpoint_velocity = data["true_endpoint_velocity"]
        features = _features(obs_values, obs_available, obs_fresh, obs_age, mean, std, device_obj)
        action_tensor = torch.as_tensor(np.array(actions, copy=True), dtype=torch.float32, device=device_obj)
        target_q = torch.as_tensor(np.array(true_q, copy=True), dtype=torch.float32, device=device_obj)
        target_dq = torch.as_tensor(np.array(true_dq, copy=True), dtype=torch.float32, device=device_obj)
        target_endpoint = torch.as_tensor(np.array(true_endpoint, copy=True), dtype=torch.float32, device=device_obj)
        target_endpoint_velocity = torch.as_tensor(np.array(true_endpoint_velocity, copy=True), dtype=torch.float32, device=device_obj)
        descriptor_tensor = torch.as_tensor(descriptor, dtype=torch.float32, device=device_obj).expand(action_count + 1, -1)
        scenario = manifest["episodes"][window.episode_index]["scenario"]
        goal_q_values = scenario.get("target_q") if not scenario.get("degraded", True) else None
        control_goal = (None if goal_q_values is None else torch.as_tensor(goal_q_values, dtype=torch.float32,
                                                                            device=device_obj))
        belief = model.initial(features[0], descriptor_tensor[0])
        losses: list[torch.Tensor] = []
        one_step_losses: list[torch.Tensor] = []
        five_step_losses: list[torch.Tensor] = []
        twenty_step_losses: list[torch.Tensor] = []
        endpoint_losses: list[torch.Tensor] = []
        control_action_losses: list[torch.Tensor] = []
        for index in range(action_count):
            if index == window.burn_in:
                belief = belief.detach()
            if index < window.burn_in:
                with torch.no_grad():
                    prior = model.predict_prior(belief, action_tensor[index], descriptor_tensor[index], 0.01)
                    belief = model.observe(prior, features[index + 1], descriptor_tensor[index + 1])
                continue
            prior = model.predict_prior(belief, action_tensor[index], descriptor_tensor[index], 0.01)
            predicted_q, predicted_dq = model.decode(prior)
            one_step_losses.append(F.huber_loss(predicted_q, target_q[index + 1]) / np.pi +
                                  F.huber_loss(predicted_dq, target_dq[index + 1]) / 5.0)
            if endpoint_residual:
                _, _, predicted_endpoint, predicted_endpoint_velocity = model.decode_physical(
                    prior, descriptor_tensor[index])
                endpoint_losses.append(
                    F.huber_loss(predicted_endpoint, target_endpoint[index + 1]) / 0.55 +
                    F.huber_loss(predicted_endpoint_velocity, target_endpoint_velocity[index + 1])
                )
            if (index - window.burn_in) % 8 == 0 and index + 20 < action_count:
                # Train the dynamics on the same open-loop use case as the
                # evaluation gate. Detaching the anchor avoids turning this
                # auxiliary loss into an observer BPTT path.
                rollout = prior.detach()
                for offset in range(1, 5):
                    rollout = model.predict_prior(rollout, action_tensor[index + offset],
                                                  descriptor_tensor[index + offset], 0.01)
                rollout_q, rollout_dq = model.decode(rollout)
                five_step_losses.append(
                    F.huber_loss(rollout_q, target_q[index + 5]) / np.pi +
                    F.huber_loss(rollout_dq, target_dq[index + 5]) / 5.0
                )
                for offset in range(5, 20):
                    rollout = model.predict_prior(rollout, action_tensor[index + offset],
                                                  descriptor_tensor[index + offset], 0.01)
                rollout_q, rollout_dq = model.decode(rollout)
                twenty_step_losses.append(
                    F.huber_loss(rollout_q, target_q[index + 20]) / np.pi +
                    F.huber_loss(rollout_dq, target_dq[index + 20]) / 5.0
                )
            belief = model.observe(prior, features[index + 1], descriptor_tensor[index + 1])
            posterior_q, posterior_dq = model.decode(belief)
            losses.append(F.huber_loss(posterior_q, target_q[index + 1]) / np.pi +
                          F.huber_loss(posterior_dq, target_dq[index + 1]) / 5.0)
            if control_goal is not None and control_action_weight > 0.0:
                predicted_action = computed_torque_hold_action(posterior_q, posterior_dq, control_goal,
                                                               descriptor_tensor[index + 1])
                target_action = computed_torque_hold_action(target_q[index + 1], target_dq[index + 1], control_goal,
                                                             descriptor_tensor[index + 1])
                control_action_losses.append(F.mse_loss(predicted_action, target_action))
        loss = torch.stack(losses).mean() + torch.stack(one_step_losses).mean()
        if five_step_losses:
            loss = loss + 0.5 * torch.stack(five_step_losses).mean()
        if twenty_step_losses:
            loss = loss + 0.25 * torch.stack(twenty_step_losses).mean()
        if endpoint_losses:
            loss = loss + torch.stack(endpoint_losses).mean()
        if control_action_losses:
            loss = loss + control_action_weight * torch.stack(control_action_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % max(1, updates // 20) == 0 or update == updates - 1:
            history.append({"update": float(update), "loss": float(loss.detach().cpu()),
                            "posterior_loss": float(torch.stack(losses).mean().detach().cpu()),
                            "one_step_loss": float(torch.stack(one_step_losses).mean().detach().cpu()),
                            "five_step_loss": float(torch.stack(five_step_losses).mean().detach().cpu())
                            if five_step_losses else 0.0,
                            "twenty_step_loss": float(torch.stack(twenty_step_losses).mean().detach().cpu())
                            if twenty_step_losses else 0.0,
                            "endpoint_loss": float(torch.stack(endpoint_losses).mean().detach().cpu())
                            if endpoint_losses else 0.0,
                            "control_action_loss": float(torch.stack(control_action_losses).mean().detach().cpu())
                            if control_action_losses else 0.0})
    metadata = {
        "schema_version": 4 if physics_prior else 2 if endpoint_residual else WorldBundle.SCHEMA_VERSION,
        "model": {"belief_dim": model.belief_dim, "observation_dim": model.observation_dim,
                  "descriptor_dim": model.descriptor_dim, "endpoint_residual": model.endpoint_residual,
                  "physics_prior": model.physics_prior,
                  "fresh_measurement_correction": model.fresh_measurement_correction},
        "observation_schema": 1, "descriptor_schema": 1, "action_schema": 1,
        "normalization": normalization, "dataset_manifest": str((root / "manifest.json").relative_to(root)),
        "seed": seed, "updates": updates, "device": str(device_obj),
        "window_mode": window_mode, "window_count": len(training_windows),
        "window_sampling": "uniform", "overfit_scenario_id": overfit_scenario_id,
        "control_action_weight": control_action_weight,
    }
    destination = confined_path(output)
    bundle_path = WorldBundle(model.eval(), metadata).save(destination / "best")
    report = {"bundle": str(bundle_path), "updates": updates, "history": history,
              "elapsed_seconds": time.perf_counter() - started, "device": str(device_obj),
              "endpoint_residual": endpoint_residual, "window_mode": window_mode,
              "physics_prior": physics_prior, "window_count": len(training_windows), "window_sampling": "uniform",
              "overfit_scenario_id": overfit_scenario_id, "control_action_weight": control_action_weight}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
