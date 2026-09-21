"""TorchRL PPO/GAE components for the shared-feeling policy."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torch.nn import functional as F
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from ..baselines.controller import FeedbackController
from ..config import confined_path
from ..contracts import Command, Observation, RobotDescriptor, Task
from ..control.features import (
    CURRENT_PACKET_POLICY_INPUT_DIM,
    CurrentPacketFeatureAdapter,
    FeatureAdapter,
)
from ..control.policy import LEARNED_STATE_DIM, POLICY_INPUT_DIM, CriticNetwork, PolicyNetwork
from ..control.runtime import CurrentPacketRuntime, Runtime
from ..evaluate.baseline import healthy_cases, load_healthy_case_manifest
from ..models.bundle import WorldBundle
from ..models.world import WorldModel
from ..sim.actuators import ActuatorSettings
from ..sim.mechanics import forward_kinematics
from ..sim.native import ImperfectNativeArm, default_descriptor
from ..sim.sensors import SensorSettings, SensorSuite
from ..sim.warp_backend import WarpArmBatch

DEVELOPMENT_TEACHER_FAILURES = (1, 30, 53)


def _initialize_policy(policy: PolicyNetwork, path: str | Path | None, *, device: torch.device,
                       action_scale: float | None = None) -> dict[str, object] | None:
    """Load an explicit warm start and retain its immutable identity in the report."""

    if path is None and action_scale is None:
        return None
    identity: dict[str, object] = {}
    if path is not None:
        source = confined_path(path, must_exist=True)
        policy.load_state_dict(torch.load(source, map_location=device, weights_only=True))
        identity.update({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    if action_scale is not None:
        if not 1e-4 < action_scale <= 1.0:
            raise ValueError("initial action scale must be in (0.0001, 1]")
        with torch.no_grad():
            policy.scale.weight.zero_()
            policy.scale.bias.fill_(math.log(math.expm1(action_scale - 1e-4)))
        identity["action_scale"] = action_scale
    return identity


class _PolicyParameters(nn.Module):
    def __init__(self, policy: PolicyNetwork) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.policy.parameters_for_distribution(features)


@dataclass
class PPOComponents:
    policy: PolicyNetwork
    critic: CriticNetwork
    actor: ProbabilisticActor
    critic_module: TensorDictModule
    loss_module: ClipPPOLoss
    gae: GAE


@dataclass
class RolloutState:
    """Live per-world state carried across PPO buffer boundaries."""

    arm: WarpArmBatch
    descriptor: RobotDescriptor
    descriptor_tensor: Tensor
    q: Tensor
    dq: Tensor
    observation: Tensor
    belief: Tensor
    goal_q: Tensor
    goal: Tensor
    start_q: Tensor
    previous: Tensor
    task_tick: Tensor | None = None
    hold_valid: Tensor | None = None


def make_ppo_components(*, input_dim: int = POLICY_INPUT_DIM, device: str | torch.device = "cpu") -> PPOComponents:
    policy = PolicyNetwork(input_dim=input_dim).to(device)
    critic = CriticNetwork(input_dim=input_dim).to(device)
    actor_parameters = TensorDictModule(_PolicyParameters(policy), in_keys=["features"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(actor_parameters, in_keys=["loc", "scale"], spec=None,
                              distribution_class=TanhNormal,
                               distribution_kwargs={"low": -1.0, "high": 1.0, "event_dims": 1},
                               return_log_prob=True)
    critic_module = TensorDictModule(critic, in_keys=["features"], out_keys=["state_value"])
    loss_module = ClipPPOLoss(actor_network=actor, critic_network=critic_module,
                              clip_epsilon=0.2, entropy_bonus=False, entropy_coef=0.0,
                              normalize_advantage=True)
    gae = GAE(gamma=1.0, lmbda=0.95, value_network=critic_module, time_dim=1)
    return PPOComponents(policy, critic, actor, critic_module, loss_module, gae)


def hand_checked_gae(components: PPOComponents) -> TensorDict:
    """Create a tiny rollout and run TorchRL GAE for termination-contract tests."""

    features = torch.zeros(1, 3, POLICY_INPUT_DIM)
    next_features = torch.zeros(1, 3, POLICY_INPUT_DIM)
    td = TensorDict({
        "features": features,
        "action": torch.zeros(1, 3, 2),
        "sample_log_prob": torch.zeros(1, 3),
        ("next", "reward"): torch.ones(1, 3),
        "done": torch.tensor([[False, False, True]]),
        "terminated": torch.tensor([[False, False, True]]),
        "truncated": torch.zeros(1, 3, dtype=torch.bool),
        ("next", "features"): next_features,
        ("next", "done"): torch.tensor([[False, False, True]]),
        ("next", "terminated"): torch.tensor([[False, False, True]]),
        ("next", "truncated"): torch.zeros(1, 3, dtype=torch.bool),
    }, batch_size=[1, 3])
    return components.gae(td)


def numeric_gae(rewards: Tensor, values: Tensor, final_value: Tensor,
                terminated: Tensor, truncated: Tensor, *, gamma: float = 1.0,
                lmbda: float = 0.95) -> Tensor:
    """Reference GAE with distinct terminal, truncation, and buffer semantics."""

    if rewards.ndim != 1 or values.shape != rewards.shape:
        raise ValueError("rewards and values must be matching one-dimensional tensors")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("termination flags must match rewards")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        next_value = final_value if index == len(rewards) - 1 else values[index + 1]
        bootstrap = 0.0 if bool(terminated[index]) else 1.0
        continuation = 0.0 if bool(terminated[index] or truncated[index]) else 1.0
        delta = rewards[index] + gamma * bootstrap * next_value - values[index]
        carry = delta + gamma * lmbda * continuation * carry
        advantages[index] = carry
    return advantages


def task_terminal_reward(success: Tensor, hard_violation: Tensor) -> Tensor:
    """Stage-A terminal reward, preserving scorer failures as explicit penalties."""

    return torch.where(hard_violation, -10.0, torch.where(success, 10.0, -1.0))


def ppo_update(components: PPOComponents, rollout: TensorDict, optimizer: torch.optim.Optimizer, *,
               epochs: int = 5, max_approximate_kl: float | None = None,
               teacher_regularization_weight: float = 0.0, anchor_policy: PolicyNetwork | None = None,
               anchor_regularization_weight: float = 0.0) -> dict[str, float]:
    """Run one TorchRL PPO update on stored rollout features.

    The observer is absent from this function by design: rollout features are
    already detached snapshots from the frozen world-model bundle.
    """

    flat = rollout.reshape(-1)
    sample_count = flat.batch_size.numel()
    if teacher_regularization_weight < 0.0:
        raise ValueError("teacher regularization weight must be non-negative")
    if anchor_regularization_weight < 0.0 or (anchor_regularization_weight > 0.0 and anchor_policy is None):
        raise ValueError("anchor regularization requires a frozen policy and non-negative weight")
    with torch.no_grad():
        loc, scale = components.policy.parameters_for_distribution(flat["features"])
        advantage = flat["advantage"]
        value_target = flat.get("value_target", torch.zeros_like(advantage))
        value_prediction = components.critic(flat["features"])
        action = flat["action"]
        terminal_reward = flat.get(("next", "reward_terminal"), torch.zeros_like(flat["next", "reward"]))
        terminal_mask = terminal_reward.ne(0.0)
        terminal_count = terminal_mask.sum()
        diagnostics = {
            "action_loc_abs_mean": float(torch.tanh(loc).abs().mean()),
            "action_scale_mean": float(scale.mean()),
            "action_scale_p95": float(torch.quantile(scale, 0.95)),
            "sampled_action_abs_mean": float(action.abs().mean()),
            "sampled_action_saturation_fraction": float((action.abs() >= 0.95).float().mean()),
            "advantage_mean": float(advantage.mean()),
            "advantage_std": float(advantage.std(unbiased=False)),
            "value_prediction_mean": float(value_prediction.mean()),
            "value_prediction_std": float(value_prediction.std(unbiased=False)),
            "value_target_mean": float(value_target.mean()),
            "value_target_std": float(value_target.std(unbiased=False)),
            "value_abs_error_mean": float((value_prediction - value_target).abs().mean()),
            "rollout_reward_mean": float(flat["next", "reward"].mean()),
            "rollout_terminal_count": float(terminal_count),
            "rollout_terminal_success_fraction": float(
                (terminal_reward.eq(10.0) & terminal_mask).sum() / terminal_count.clamp_min(1)
            ),
            "rollout_terminal_hard_violation_fraction": float(
                (terminal_reward.eq(-10.0) & terminal_mask).sum() / terminal_count.clamp_min(1)
            ),
        }
    policy_losses: list[float] = []
    critic_losses: list[float] = []
    teacher_losses: list[float] = []
    anchor_losses: list[float] = []
    total_losses: list[float] = []
    ratio_means: list[float] = []
    ratio_clip_fractions: list[float] = []
    approximate_kls: list[float] = []
    updates_applied = 0
    stopped_for_kl = False
    for _ in range(epochs):
        permutation = torch.randperm(sample_count, device=flat.device)
        for start in range(0, sample_count, 256):
            batch = flat[permutation[start:start + 256]]
            distribution = components.policy.distribution(batch["features"])
            current_log_prob = distribution.log_prob(batch["action"])
            log_ratio = current_log_prob - batch["sample_log_prob"]
            ratio = log_ratio.exp()
            losses = components.loss_module(batch)
            teacher_action = batch.get("teacher_action", batch["action"])
            teacher_loss = F.mse_loss(components.policy.deterministic(batch["features"]), teacher_action)
            if anchor_policy is None:
                anchor_loss = torch.zeros((), device=batch.device)
            else:
                with torch.no_grad():
                    anchor_action = anchor_policy.deterministic(batch["features"])
                anchor_loss = F.mse_loss(components.policy.deterministic(batch["features"]), anchor_action)
            total = (losses["loss_objective"] + losses["loss_critic"] + teacher_regularization_weight * teacher_loss
                     + anchor_regularization_weight * anchor_loss)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(components.policy.parameters()) + list(components.critic.parameters()), 0.5)
            optimizer.step()
            with torch.no_grad():
                updated_distribution = components.policy.distribution(batch["features"])
                updated_log_prob = updated_distribution.log_prob(batch["action"])
                updated_log_ratio = updated_log_prob - batch["sample_log_prob"]
                updated_ratio = updated_log_ratio.exp()
                updated_kl = ((updated_ratio - 1.0) - updated_log_ratio).mean()
            total_losses.append(float(total.detach()))
            policy_losses.append(float(losses["loss_objective"].detach()))
            critic_losses.append(float(losses["loss_critic"].detach()))
            teacher_losses.append(float(teacher_loss.detach()))
            anchor_losses.append(float(anchor_loss.detach()))
            ratio_means.append(float(ratio.detach().mean()))
            ratio_clip_fractions.append(float((ratio.detach().sub(1.0).abs() > 0.2).float().mean()))
            approximate_kls.append(float(updated_kl))
            updates_applied += 1
            if max_approximate_kl is not None and updated_kl > max_approximate_kl:
                stopped_for_kl = True
                break
        if stopped_for_kl:
            break
    return {"loss": float(np.mean(total_losses)), "policy_loss": float(np.mean(policy_losses)),
            "critic_loss": float(np.mean(critic_losses)), "teacher_action_loss": float(np.mean(teacher_losses)),
            "teacher_regularization_weight": teacher_regularization_weight, **diagnostics,
            "anchor_action_loss": float(np.mean(anchor_losses)),
            "anchor_regularization_weight": anchor_regularization_weight,
            "ratio_mean": float(np.mean(ratio_means)),
            "ratio_clip_fraction": float(np.mean(ratio_clip_fractions)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "ppo_updates_applied": float(updates_applied),
            "ppo_early_kl_stop": float(stopped_for_kl)}


def critic_warmup(components: PPOComponents, rollout: TensorDict, optimizer: torch.optim.Optimizer, *, epochs: int) -> float:
    """Fit the value baseline before PPO recomputes advantages for a fresh rollout."""

    if epochs < 0:
        raise ValueError("critic warmup epochs must be non-negative")
    if epochs == 0:
        return 0.0
    features = rollout["features"].reshape(-1, POLICY_INPUT_DIM)
    target = rollout["value_target"].reshape(-1).detach()
    losses: list[float] = []
    for _ in range(epochs):
        prediction = components.critic(features)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(components.critic.parameters(), 0.5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def behavior_clone(components: PPOComponents, rollout: TensorDict,
                   optimizer: torch.optim.Optimizer, epochs: int = 3) -> float:
    """Warm-start the actor from public-observation baseline actions.

    The teacher is used only during training. Its targets are stored beside
    detached policy features; runtime still receives only the shared belief,
    descriptor, task, and previous command.
    """

    features = rollout["features"].reshape(-1, POLICY_INPUT_DIM)
    targets = rollout["teacher_action"].reshape(-1, 2)
    losses: list[float] = []
    for _ in range(epochs):
        prediction = components.policy.deterministic(features)
        loss = F.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(components.policy.parameters()), 0.5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _warp_observation_features(q: np.ndarray, dq: np.ndarray, descriptor: np.ndarray,
                               mean: Tensor, std: Tensor, device: torch.device) -> Tensor:
    endpoints = np.stack([forward_kinematics(values, descriptor[:2]) for values in q])
    values = np.concatenate((q, dq, np.zeros_like(q), endpoints), axis=-1)
    raw = torch.as_tensor(values, dtype=torch.float32, device=device)
    available = torch.ones((len(q), 8), dtype=torch.float32, device=device)
    fresh = torch.ones_like(available)
    age = torch.zeros_like(available)
    return torch.cat(((raw - mean) / std, available, fresh, age), dim=-1)


def _warp_observation_features_torch(q: Tensor, dq: Tensor, descriptor: Tensor,
                                     mean: Tensor, std: Tensor) -> Tensor:
    q12 = q[:, 0] + q[:, 1]
    lengths = descriptor[:2]
    endpoint = torch.stack((lengths[0] * torch.sin(q[:, 0]) + lengths[1] * torch.sin(q12),
                            -lengths[0] * torch.cos(q[:, 0]) - lengths[1] * torch.cos(q12)), dim=-1)
    values = torch.cat((q, dq, torch.zeros_like(q), endpoint), dim=-1)
    available = torch.ones((q.shape[0], 8), dtype=q.dtype, device=q.device)
    fresh = torch.ones_like(available)
    age = torch.zeros_like(available)
    return torch.cat(((values - mean) / std, available, fresh, age), dim=-1)


def _development_tasks(*, worlds: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Sample paired starts/goals from the frozen P2 development manifest."""

    manifest = json.loads(confined_path("configs/healthy-development-v1.json", must_exist=True).read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if manifest.get("schema_version") != 1 or not isinstance(cases, list) or len(cases) != 64:
        raise ValueError("healthy development manifest has an unsupported identity")
    starts = torch.as_tensor([case["start_q"] for case in cases], dtype=torch.float32, device=device)
    goals = torch.as_tensor([case["goal_q"] for case in cases], dtype=torch.float32, device=device)
    # The P2 reference itself fails these three cases. They remain in every
    # evaluation, but cannot serve as demonstration data for behavior cloning.
    eligible = torch.as_tensor([index for index in range(len(cases)) if index not in DEVELOPMENT_TEACHER_FAILURES],
                               dtype=torch.long, device=device)
    indices = eligible[torch.randint(len(eligible), (worlds,), device=device)]
    return starts[indices], goals[indices]


def _new_rollout_state(model: WorldModel, *, worlds: int, device: torch.device,
                       mean: Tensor, std: Tensor, physics_dt: float) -> RolloutState:
    descriptor = default_descriptor()
    descriptor_np = descriptor.as_array().astype(np.float32)
    descriptor_tensor = torch.as_tensor(descriptor_np, dtype=torch.float32, device=device).expand(worlds, -1)
    arm = WarpArmBatch(descriptor, worlds, device="cuda:0" if device.type == "cuda" else "cpu", timestep=physics_dt)
    start, goal_q = _development_tasks(worlds=worlds, device=device)
    arm.reset(q=start.detach().cpu().numpy())
    q_raw, dq_raw = arm.torch_state()
    q, dq = cast(Tensor, q_raw), cast(Tensor, dq_raw)
    observation = _warp_observation_features_torch(q, dq, descriptor_tensor[0], mean, std)
    with torch.no_grad():
        belief = model.initial(observation, descriptor_tensor)
    goal_sum = goal_q[:, 0] + goal_q[:, 1]
    lengths = torch.as_tensor(descriptor.link_lengths, dtype=torch.float32, device=device)
    goal = torch.stack((lengths[0] * torch.sin(goal_q[:, 0]) + lengths[1] * torch.sin(goal_sum),
                        -lengths[0] * torch.cos(goal_q[:, 0]) - lengths[1] * torch.cos(goal_sum)), dim=-1)
    previous = torch.zeros((worlds, 2), dtype=torch.float32, device=device)
    return RolloutState(arm, descriptor, descriptor_tensor, q, dq, observation, belief,
                        goal_q, goal, start, previous,
                        task_tick=torch.zeros(worlds, dtype=torch.int64, device=device),
                        hold_valid=torch.ones(worlds, dtype=torch.bool, device=device))


def _collect_rollout(model: WorldModel, components: PPOComponents, *, worlds: int, steps: int,
                     device: torch.device, mean: Tensor, std: Tensor,
                     physics_dt: float, control_dt: float, state: RolloutState | None = None,
                     memoryless: bool = False, teacher_forcing: bool = False) -> tuple[TensorDict, float, RolloutState]:
    substeps = round(control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    state = state or _new_rollout_state(model, worlds=worlds, device=device, mean=mean, std=std,
                                        physics_dt=physics_dt)
    assert state.task_tick is not None
    descriptor = state.descriptor
    descriptor_tensor = state.descriptor_tensor
    arm = state.arm
    q, dq = state.q, state.dq
    observation, belief, goal_q, goal, start_q, previous = (state.observation, state.belief, state.goal_q,
                                                              state.goal, state.start_q, state.previous)
    features: list[Tensor] = []
    next_features: list[Tensor] = []
    actions: list[Tensor] = []
    log_probs: list[Tensor] = []
    teacher_actions: list[Tensor] = []
    rewards: list[Tensor] = []
    progress_rewards: list[Tensor] = []
    distance_rewards: list[Tensor] = []
    terminal_rewards: list[Tensor] = []
    hard_violations: list[Tensor] = []
    dones: list[Tensor] = []
    terminated: list[Tensor] = []
    started = time.perf_counter()
    with torch.no_grad(), set_exploration_type(
        ExplorationType.RANDOM if not teacher_forcing else ExplorationType.DETERMINISTIC
    ):
        for step in range(steps):
            time_remaining = (1.5 - state.task_tick * control_dt).clamp_min(0.0).unsqueeze(-1)
            decoded_q, decoded_dq = model.decode(belief)
            decoded_state = torch.cat((decoded_q, decoded_dq), dim=-1)
            if memoryless:
                policy_belief, policy_decoded = torch.zeros_like(belief), torch.zeros_like(decoded_state)
            else:
                policy_belief, policy_decoded = belief, decoded_state
            policy_features = torch.cat((policy_belief, policy_decoded, descriptor_tensor, goal, goal_q,
                                         time_remaining, previous), dim=-1)
            actor_input = TensorDict({"features": policy_features}, batch_size=[worlds], device=device)
            actor_output = components.actor(actor_input)
            action = actor_output["action"].clamp(-1.0, 1.0)
            # This is the accepted public-state-only computed-torque reference:
            # it commands the task goal directly, without hidden start-state or
            # trajectory inputs that frozen runtime cannot reproduce.
            target_q = goal_q
            target_dq = torch.zeros_like(dq)
            gravity_terms = torch.as_tensor(
                [0.6 * 0.3 * 0.5 + 0.4 * 0.3, 0.4 * 0.25 * 0.5],
                dtype=torch.float32, device=device,
            )
            qsum = q[:, 0] + q[:, 1]
            gravity = -9.81 * torch.stack((gravity_terms[0] * torch.sin(q[:, 0]) +
                                            gravity_terms[1] * torch.sin(qsum),
                                            gravity_terms[1] * torch.sin(qsum)), dim=-1)
            l1, _ = descriptor.link_lengths
            _, m2 = descriptor.nominal_masses
            _, lc2 = descriptor.link_lengths * 0.5
            coupling = -m2 * l1 * lc2 * torch.sin(q[:, 1])
            coriolis = torch.stack((coupling * (2 * dq[:, 0] * dq[:, 1] + dq[:, 1] ** 2),
                                    -coupling * dq[:, 0] ** 2), dim=-1)
            teacher_torque = (torch.as_tensor([8.0, 6.0], device=device) * (target_q - q) +
                              torch.as_tensor([0.7, 0.56], device=device) * (target_dq - dq) - gravity +
                              coriolis + 0.02 * dq)
            teacher = torch.clamp(teacher_torque / torch.as_tensor(descriptor.nominal_torque_scales,
                                  dtype=torch.float32, device=device), -1.0, 1.0)
            if teacher_forcing:
                # Behavior-cloning data must remain on the feasible teacher
                # distribution. Random untrained actions reach states absent
                # from the frozen observer's healthy feedback data.
                action = teacher
            features.append(policy_features.detach().clone())
            actions.append(action.detach().clone())
            log_probs.append(actor_output["sample_log_prob"].detach().clone())
            teacher_actions.append(teacher.detach().clone())
            hard_violation = torch.zeros(worlds, dtype=torch.bool, device=device)
            for _ in range(substeps):
                arm.step_torch(action)
                q_sub_raw, _ = arm.torch_state()
                q_sub = cast(Tensor, q_sub_raw)
                limits = torch.as_tensor(descriptor.joint_limits, dtype=q_sub.dtype, device=device)
                hard_violation |= ((q_sub <= limits[:, 0]) | (q_sub >= limits[:, 1])).any(dim=-1)
            q_next_raw, dq_next_raw = arm.torch_state()
            q_next, dq_next = cast(Tensor, q_next_raw), cast(Tensor, dq_next_raw)
            next_observation = _warp_observation_features_torch(q_next, dq_next, descriptor_tensor[0], mean, std)
            prior = model.predict_prior(belief, action, descriptor_tensor, control_dt)
            belief = model.observe(prior, next_observation, descriptor_tensor)
            endpoint = next_observation[:, 6:8] * std[6:8] + mean[6:8]
            current_endpoint = observation[:, 6:8] * std[6:8] + mean[6:8]
            previous_distance = torch.linalg.vector_norm(current_endpoint - goal, dim=-1)
            distance = torch.linalg.vector_norm(endpoint - goal, dim=-1).clamp(max=1.1)
            progress = (previous_distance - distance) / 0.55
            distance_cost = -distance / 0.55 * (control_dt / 1.7)
            next_tick = state.task_tick + 1
            speed = torch.linalg.vector_norm(torch.stack((
                descriptor.link_lengths[0] * torch.cos(q_next[:, 0]) * dq_next[:, 0] +
                descriptor.link_lengths[1] * torch.cos(q_next[:, 0] + q_next[:, 1]) * (dq_next[:, 0] + dq_next[:, 1]),
                descriptor.link_lengths[0] * torch.sin(q_next[:, 0]) * dq_next[:, 0] +
                descriptor.link_lengths[1] * torch.sin(q_next[:, 0] + q_next[:, 1]) * (dq_next[:, 0] + dq_next[:, 1]),
            ), dim=-1), dim=-1)
            in_hold = next_tick >= 150
            hold_ok = (distance <= 0.010) & (speed <= 0.030)
            if in_hold.any():
                assert state.hold_valid is not None
                state.hold_valid = torch.where(in_hold, state.hold_valid & hold_ok, state.hold_valid)
            natural_terminal = next_tick == 170
            if natural_terminal.any():
                assert state.hold_valid is not None
                success = state.hold_valid & natural_terminal
            else:
                success = torch.zeros(worlds, dtype=torch.bool, device=device)
            terminal_reward = torch.where(natural_terminal, task_terminal_reward(success, hard_violation),
                                          torch.where(hard_violation, -10.0, 0.0))
            rewards.append((progress + distance_cost + terminal_reward).detach().clone())
            progress_rewards.append(progress.detach().clone())
            distance_rewards.append(distance_cost.detach().clone())
            terminal_rewards.append(terminal_reward.detach().clone())
            hard_violations.append(hard_violation.detach().clone())
            observation = next_observation
            next_time_remaining = (1.5 - next_tick * control_dt).clamp_min(0.0).unsqueeze(-1)
            next_decoded_q, next_decoded_dq = model.decode(belief)
            next_decoded = torch.cat((next_decoded_q, next_decoded_dq), dim=-1)
            if memoryless:
                next_belief, next_decoded = torch.zeros_like(belief), torch.zeros_like(next_decoded)
            else:
                next_belief = belief
            next_features.append(torch.cat((next_belief, next_decoded, descriptor_tensor, goal, goal_q,
                                            next_time_remaining, action), dim=-1).detach().clone())
            previous = action
            # Joint-limit failures are terminal transitions for their own worlds.
            # Resetting the whole batch here would restart unrelated task clocks.
            terminal = natural_terminal | hard_violation
            dones.append(terminal)
            terminated.append(terminal)
            if terminal.any():
                fresh_start, fresh_goal_q = _development_tasks(worlds=worlds, device=device)
                arm.reset(mask=terminal.detach().cpu().numpy().astype(np.int32),
                          q=fresh_start.detach().cpu().numpy())
                reset_q_raw, reset_dq_raw = arm.torch_state()
                reset_q, reset_dq = cast(Tensor, reset_q_raw), cast(Tensor, reset_dq_raw)
                reset_observation = _warp_observation_features_torch(reset_q, reset_dq, descriptor_tensor[0], mean, std)
                reset_belief = model.initial(reset_observation, descriptor_tensor)
                fresh_goal_sum = fresh_goal_q[:, 0] + fresh_goal_q[:, 1]
                lengths = torch.as_tensor(descriptor.link_lengths, dtype=torch.float32, device=device)
                fresh_goal = torch.stack((lengths[0] * torch.sin(fresh_goal_q[:, 0]) + lengths[1] * torch.sin(fresh_goal_sum),
                                          -lengths[0] * torch.cos(fresh_goal_q[:, 0]) - lengths[1] * torch.cos(fresh_goal_sum)), dim=-1)
                mask = terminal.unsqueeze(-1)
                q, dq = torch.where(mask, reset_q, q_next), torch.where(mask, reset_dq, dq_next)
                observation, belief = torch.where(mask, reset_observation, observation), torch.where(mask, reset_belief, belief)
                start_q, goal_q, goal = (torch.where(mask, fresh_start, start_q), torch.where(mask, fresh_goal_q, goal_q),
                                          torch.where(mask, fresh_goal, goal))
                previous = torch.where(mask, torch.zeros_like(previous), previous)
                assert state.hold_valid is not None
                state.hold_valid = torch.where(terminal, torch.ones_like(state.hold_valid), state.hold_valid)
            state.q, state.dq, state.observation, state.belief = q, dq, observation, belief
            state.goal_q, state.goal, state.start_q, state.previous = goal_q, goal, start_q, previous
            state.task_tick = torch.where(terminal, torch.zeros_like(next_tick), next_tick)
    elapsed = time.perf_counter() - started
    rollout = TensorDict({
        "features": torch.stack(features, dim=1), "action": torch.stack(actions, dim=1),
        "teacher_action": torch.stack(teacher_actions, dim=1),
        "sample_log_prob": torch.stack(log_probs, dim=1),
        ("next", "features"): torch.stack(next_features, dim=1),
        ("next", "reward"): torch.stack(rewards, dim=1),
        ("next", "reward_progress"): torch.stack(progress_rewards, dim=1),
        ("next", "reward_distance"): torch.stack(distance_rewards, dim=1),
        ("next", "reward_terminal"): torch.stack(terminal_rewards, dim=1),
        ("next", "hard_violation"): torch.stack(hard_violations, dim=1),
        ("next", "done"): torch.stack(dones, dim=1),
        ("next", "terminated"): torch.stack(terminated, dim=1),
        ("next", "truncated"): torch.zeros((worlds, steps), dtype=torch.bool, device=device),
    }, batch_size=[worlds, steps], device=device)
    return rollout, elapsed, state


def train_policy(world_bundle: str | Path, output: str | Path, *, device: str = "cuda",
                 worlds: int = 4, rollout_steps: int = 128, transitions: int = 100_000,
                 seed: int = 0, physics_dt: float = 0.001, control_dt: float = 0.01,
                 warmup_updates: int = 10, memoryless: bool = False,
                 config_identity: dict[str, str] | None = None,
                 init_policy: str | Path | None = None,
                 init_action_scale: float | None = None, ppo_learning_rate: float = 3e-4,
                 ppo_epochs: int = 5, max_approximate_kl: float | None = None,
                 teacher_regularization_weight: float = 0.0, anchor_regularization_weight: float = 0.0,
                 critic_warmup_epochs: int = 0, critic_warmup_learning_rate: float = 1e-3) -> dict[str, object]:
    """Run bounded PPO updates on actual Warp trajectories with a frozen world model."""

    torch.manual_seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA policy training but CUDA is unavailable")
    if ppo_learning_rate <= 0.0 or ppo_epochs < 1:
        raise ValueError("PPO learning rate must be positive and PPO epochs must be at least one")
    if max_approximate_kl is not None and max_approximate_kl <= 0.0:
        raise ValueError("maximum approximate KL must be positive")
    if teacher_regularization_weight < 0.0 or anchor_regularization_weight < 0.0:
        raise ValueError("regularization weights must be non-negative")
    if critic_warmup_epochs < 0 or critic_warmup_learning_rate <= 0.0:
        raise ValueError("critic warmup settings must be non-negative epochs and positive learning rate")
    bundle = WorldBundle.load(world_bundle, device=device_obj)
    bundle.model.eval()
    metadata = bundle.metadata["normalization"]
    mean = torch.as_tensor(metadata["mean"], dtype=torch.float32, device=device_obj)
    std = torch.as_tensor(metadata["std"], dtype=torch.float32, device=device_obj)
    components = make_ppo_components(device=device_obj)
    initialization = _initialize_policy(components.policy, init_policy, device=device_obj,
                                        action_scale=init_action_scale)
    anchor_policy = deepcopy(components.policy).eval() if anchor_regularization_weight > 0.0 else None
    optimizer = torch.optim.Adam(list(components.policy.parameters()) + list(components.critic.parameters()),
                                 lr=ppo_learning_rate)
    critic_optimizer = torch.optim.Adam(components.critic.parameters(), lr=critic_warmup_learning_rate)
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(components.policy.state_dict(), destination / "initial-policy.pt")
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(components.policy.state_dict(), checkpoints / "initial-policy.pt")
    updates = max(1, int(np.ceil(transitions / (worlds * rollout_steps))))
    history: list[dict[str, float]] = []
    rollout_state: RolloutState | None = None
    clone_rollouts: list[TensorDict] = []
    for update in range(updates):
        # BC demonstrations are complete 170-tick teacher episodes. Do not
        # carry a terminal reset or a failed reference case into the next
        # demonstration buffer; PPO collection below retains normal continuity.
        collect_state = None if update < warmup_updates else rollout_state
        with torch.no_grad():
            rollout, collection_seconds, rollout_state = _collect_rollout(
                bundle.model, components, worlds=worlds, steps=rollout_steps, device=device_obj, mean=mean,
                std=std, physics_dt=physics_dt, control_dt=control_dt, state=collect_state,
                memoryless=memoryless, teacher_forcing=update < warmup_updates,
        )
        if update < warmup_updates:
            clone_rollouts.append(rollout.select("features", "teacher_action"))
            clone_data = torch.cat(clone_rollouts, dim=1)
            clone_loss = behavior_clone(components, clone_data, optimizer, epochs=50)
            # The stored log probabilities belong to the pre-clone actor. Do
            # not feed this stale rollout to PPO; the next collection is on
            # policy again.
            metrics = {"loss": 0.0, "policy_loss": 0.0, "critic_loss": 0.0,
                       "behavior_clone_loss": clone_loss,
                       "behavior_clone_samples": float(clone_data.numel())}
        else:
            components.gae(rollout)
            warmup_loss = critic_warmup(components, rollout, critic_optimizer, epochs=critic_warmup_epochs)
            if critic_warmup_epochs:
                components.gae(rollout)
            metrics = ppo_update(components, rollout, optimizer, epochs=ppo_epochs,
                                 max_approximate_kl=max_approximate_kl,
                                 teacher_regularization_weight=teacher_regularization_weight,
                                 anchor_policy=anchor_policy,
                                 anchor_regularization_weight=anchor_regularization_weight)
            metrics["critic_warmup_loss"] = warmup_loss
        metrics.update({"update": float(update), "transitions": float((update + 1) * worlds * rollout_steps),
                        "collection_seconds": collection_seconds,
                        "terminal_transitions": float(rollout["next", "terminated"].sum().item()),
                        "buffer_boundary_is_terminal": float(rollout["next", "terminated"][:, -1].all().item())})
        history.append(metrics)
        torch.save(components.policy.state_dict(), checkpoints / f"policy-update-{update + 1:06d}.pt")
    torch.save(components.policy.state_dict(), destination / "final-policy.pt")
    memory = (None if device_obj.type != "cuda" else {
        "total_bytes": torch.cuda.get_device_properties(device_obj).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_obj),
    })
    report = {"world_bundle": str(confined_path(world_bundle)), "device": str(device_obj), "worlds": worlds,
              "rollout_steps": rollout_steps, "transitions": updates * worlds * rollout_steps,
              "physics_dt": physics_dt, "control_dt": control_dt,
              "warmup_updates": warmup_updates, "memoryless": memoryless,
              "ppo_learning_rate": ppo_learning_rate, "ppo_epochs": ppo_epochs,
              "max_approximate_kl": max_approximate_kl,
              "teacher_regularization_weight": teacher_regularization_weight,
              "anchor_regularization_weight": anchor_regularization_weight,
              "critic_warmup_epochs": critic_warmup_epochs, "critic_warmup_learning_rate": critic_warmup_learning_rate,
              "policy_input_dim": POLICY_INPUT_DIM, "policy_feature_schema": "belief-decoded-state-goal-q-v3",
              "selected_checkpoint": None,
              "initialization": initialization,
              "config_identity": config_identity,
              "history": history, "device_resident_collection": True,
              "cuda_memory": memory,
              "limitation": "Warp state and commands use shared Torch views; simulator diagnostics still require explicit snapshots."}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _clean_packet_settings(control_dt: float) -> SensorSettings:
    if not np.isclose(control_dt, 0.01):
        raise ValueError("R5b1 clean packet settings require a 10 ms control period")
    return SensorSettings(np.full(8, control_dt), np.zeros(8), np.zeros(8), np.zeros(8))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_from_goal_q(descriptor: RobotDescriptor, goal_q: np.ndarray) -> Task:
    goal = forward_kinematics(np.asarray(goal_q, dtype=np.float64), descriptor.link_lengths)
    return Task(goal, deadline_s=1.5)


def _truth_teacher_observation(arm: ImperfectNativeArm, time_s: float) -> Observation:
    """Form a training-only label input; never pass it to the learned runtime."""

    truth = arm.arm.truth()
    values = np.concatenate((truth.q, truth.dq, np.zeros(2), truth.endpoint_xz))
    return Observation(values, np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), time_s)


@dataclass(frozen=True)
class _PacketCloneEpisode:
    """One teacher-driven clean-packet sequence used by cloning and C2 replay."""

    case_index: int
    features: Tensor
    teacher_actions: Tensor
    packet_values: np.ndarray
    packet_available: np.ndarray
    packet_fresh: np.ndarray
    packet_age_s: np.ndarray
    true_q: np.ndarray
    true_dq: np.ndarray


def _collect_packet_clone_episode(bundle: WorldBundle, *, device: torch.device, case_index: int,
                                  physics_dt: float, control_dt: float) -> _PacketCloneEpisode:
    """Collect one real clean-P3 teacher sequence through the clone feature path."""

    substeps = round(control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    cases = healthy_cases(64)
    if not 0 <= case_index < len(cases):
        raise ValueError("packet clone case index must be in [0, 63]")
    descriptor = default_descriptor()
    start, goal_q = cases[case_index]
    task = _task_from_goal_q(descriptor, goal_q)
    arm = ImperfectNativeArm(descriptor, ActuatorSettings.healthy(),
                             SensorSuite(_clean_packet_settings(control_dt), seed=case_index), timestep=physics_dt)
    arm.reset(start)
    adapter = FeatureAdapter(bundle, descriptor, device)
    observation = arm.observation()
    adapter.reset(observation)
    teacher = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0.0 else -1,
                                 computed_torque=True, hold_goal=True)
    features: list[Tensor] = []
    teacher_actions: list[Tensor] = []
    packets = [observation.values.copy()]
    available = [observation.available.copy()]
    fresh = [observation.fresh.copy()]
    ages = [observation.age_s.copy()]
    truth = arm.arm.truth()
    true_q = [truth.q.copy()]
    true_dq = [truth.dq.copy()]
    previous = np.zeros(2, dtype=np.float64)
    controls = round(task.total_duration_s / control_dt)
    with torch.no_grad():
        for control_index in range(controls):
            features.append(adapter.policy_features(task, previous).detach().clone())
            action = teacher.act(control_index * control_dt,
                                 _truth_teacher_observation(arm, control_index * control_dt))
            teacher_actions.append(torch.as_tensor(action, dtype=torch.float32, device=device))
            for _ in range(substeps):
                arm.step(action)
            previous = action
            observation = arm.observation()
            truth = arm.arm.truth()
            packets.append(observation.values.copy())
            available.append(observation.available.copy())
            fresh.append(observation.fresh.copy())
            ages.append(observation.age_s.copy())
            true_q.append(truth.q.copy())
            true_dq.append(truth.dq.copy())
            adapter.observe(action, observation, dt=control_dt)
    return _PacketCloneEpisode(case_index, torch.stack(features), torch.stack(teacher_actions),
                               np.asarray(packets), np.asarray(available), np.asarray(fresh), np.asarray(ages),
                               np.asarray(true_q), np.asarray(true_dq))


def _collect_packet_clone_data(bundle: WorldBundle, *, device: torch.device, episodes: int,
                               physics_dt: float, control_dt: float,
                               case_offset: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Collect packet-derived features beside simulation-truth teacher labels."""

    features: list[Tensor] = []
    teacher_actions: list[Tensor] = []
    case_indices: list[int] = []
    cases = healthy_cases(64)
    for episode_index in range(episodes):
        case_index = (case_offset + episode_index) % len(cases)
        episode = _collect_packet_clone_episode(bundle, device=device, case_index=case_index,
                                                 physics_dt=physics_dt, control_dt=control_dt)
        features.extend(episode.features)
        teacher_actions.extend(episode.teacher_actions)
        case_indices.extend([case_index] * len(episode.teacher_actions))
    return torch.stack(features), torch.stack(teacher_actions), torch.as_tensor(case_indices, dtype=torch.int64)


def _collect_current_packet_data(bundle: WorldBundle, *, device: torch.device, episodes: int,
                                 physics_dt: float, control_dt: float,
                                 case_offset: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Collect teacher labels paired with the entire current public packet schema."""

    descriptor = default_descriptor()
    adapter = CurrentPacketFeatureAdapter(bundle, descriptor, device)
    features: list[Tensor] = []
    labels: list[Tensor] = []
    case_ids: list[int] = []
    for offset in range(episodes):
        episode = _collect_packet_clone_episode(bundle, device=device, case_index=(case_offset + offset) % 64,
                                                 physics_dt=physics_dt, control_dt=control_dt)
        _, goal_q = healthy_cases(64)[episode.case_index]
        task = _task_from_goal_q(descriptor, goal_q)
        previous = Command.accepted([0.0, 0.0])
        for tick, label in enumerate(episode.teacher_actions):
            observation = Observation(episode.packet_values[tick], episode.packet_available[tick],
                                      episode.packet_fresh[tick], episode.packet_age_s[tick], tick * control_dt)
            features.append(adapter.policy_features(observation, task, tick * control_dt, previous))
            labels.append(label)
            previous = Command.accepted(label.detach().cpu().numpy())
        case_ids.extend([episode.case_index] * len(episode.teacher_actions))
    return torch.stack(features), torch.stack(labels), torch.as_tensor(case_ids, dtype=torch.int64)


def train_current_packet_comparator(world_bundle: str | Path, output: str | Path, *, device: str = "cuda",
                                    updates: int = 30, trajectories_per_update: int = 4,
                                    fit_passes_per_update: int = 50, checkpoint_interval: int = 5,
                                    seed: int = 20260913, physics_dt: float = 0.001,
                                    control_dt: float = 0.01,
                                    config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Train C3's parameter-matched, feedforward current-packet comparator."""

    if min(updates, trajectories_per_update, fit_passes_per_update, checkpoint_interval) < 1:
        raise ValueError("comparator schedule counts must be positive")
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA current-packet comparator training but CUDA is unavailable")
    world_path = confined_path(world_bundle, must_exist=True)
    bundle = WorldBundle.load(world_path, device=device_obj)
    bundle.model.eval()
    destination = confined_path(output); destination.mkdir(parents=True, exist_ok=True)
    hidden_dim = 164
    policy = PolicyNetwork(input_dim=CURRENT_PACKET_POLICY_INPUT_DIM, hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    checkpoints = destination / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination / "initial-policy.pt")
    torch.save(policy.state_dict(), checkpoints / "initial-policy.pt")
    batches: list[Tensor] = []; label_batches: list[Tensor] = []; case_batches: list[Tensor] = []; history = []
    for update in range(1, updates + 1):
        features, labels, case_ids = _collect_current_packet_data(
            bundle, device=device_obj, episodes=trajectories_per_update, physics_dt=physics_dt,
            control_dt=control_dt, case_offset=(update - 1) * trajectories_per_update)
        batches.append(features); label_batches.append(labels); case_batches.append(case_ids)
        aggregate_features, aggregate_labels = torch.cat(batches), torch.cat(label_batches)
        for _ in range(fit_passes_per_update):
            loss = F.mse_loss(policy.deterministic(aggregate_features), aggregate_labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
        history.append({"update": float(update), "behavior_clone_loss": float(loss.detach()), "demonstrations": float(len(aggregate_features))})
        if update % checkpoint_interval == 0 or update == updates:
            torch.save(policy.state_dict(), checkpoints / f"policy-update-{update:06d}.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    all_features, all_labels, all_cases = torch.cat(batches), torch.cat(label_batches), torch.cat(case_batches)
    torch.save({"features": all_features.detach().cpu(), "teacher_actions": all_labels.detach().cpu(), "case_indices": all_cases}, destination / "packet-demonstrations.pt")
    report = {"algorithm": "C3 feedforward current-packet behavior cloning", "controller_kind": "current-packet-feedforward",
              "world_bundle": str(world_path), "device": str(device_obj), "simulation_device": "cpu", "feature_and_optimizer_device": str(device_obj),
              "updates": updates, "trajectories_per_update": trajectories_per_update, "fit_passes_per_update": fit_passes_per_update, "checkpoint_interval": checkpoint_interval,
              "seed": seed, "physics_dt": physics_dt, "control_dt": control_dt, "memoryless": False,
              "policy_input_dim": CURRENT_PACKET_POLICY_INPUT_DIM, "policy_hidden_dim": hidden_dim,
              "policy_feature_schema": "current-packet-values-masks-freshness-ages-task-descriptor-previous-command-v1",
              "selected_checkpoint": None, "config_identity": config_identity, "demonstration_case_ids": sorted(set(all_cases.tolist())),
              "teacher": "simulation-truth training labels only; comparator receives current SensorSuite packet only", "history": history}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _current_packet_checkpoint(run: Path, report: dict[str, object]) -> Path:
    """Resolve the explicitly selected comparator checkpoint without a fallback."""

    selected = report.get("selected_checkpoint")
    if not isinstance(selected, str) or not selected:
        raise ValueError("current-packet comparator report has no selected_checkpoint")
    candidates = (run / selected, run / f"{selected}.pt", run / "checkpoints" / selected,
                  run / "checkpoints" / f"{selected}.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"selected comparator checkpoint is missing: {selected}")


def _current_packet_observation(trace: np.lib.npyio.NpzFile, index: int, control_dt: float,
                                episode_index: int | None = None) -> Observation:
    """Read one saved public packet with its control-clock timestamp."""

    if episode_index is None:
        return Observation(trace["packet_values"][index], trace["packet_available"][index],
                           trace["packet_fresh"][index], trace["packet_age_s"][index], index * control_dt)
    return Observation(trace["packet_values"][episode_index, index],
                       trace["packet_available"][episode_index, index],
                       trace["packet_fresh"][episode_index, index],
                       trace["packet_age_s"][episode_index, index], index * control_dt)


def _current_packet_feature_coverage(adapter: CurrentPacketFeatureAdapter, descriptor: RobotDescriptor,
                                     observation: Observation, task: Task) -> dict[str, object]:
    """Exercise each public-input field and record the resulting feature positions."""

    baseline = adapter.policy_features(observation, task, 0.0, Command.accepted([0.0, 0.0]))
    changes: dict[str, list[int]] = {}

    def changed_indices(candidate: Tensor) -> list[int]:
        return torch.nonzero((candidate - baseline).abs() > 1e-7, as_tuple=False).flatten().tolist()

    for index in range(8):
        values = observation.values.copy(); values[index] += 0.125
        changes[f"value_{index}"] = changed_indices(adapter.policy_features(
            Observation(values, observation.available, observation.fresh, observation.age_s, observation.time_s),
            task, 0.0, Command.accepted([0.0, 0.0])))
        available = observation.available.copy(); available[index] = ~available[index]
        changes[f"available_{index}"] = changed_indices(adapter.policy_features(
            Observation(observation.values, available, observation.fresh, observation.age_s, observation.time_s),
            task, 0.0, Command.accepted([0.0, 0.0])))
        fresh = observation.fresh.copy(); fresh[index] = ~fresh[index]
        changes[f"fresh_{index}"] = changed_indices(adapter.policy_features(
            Observation(observation.values, observation.available, fresh, observation.age_s, observation.time_s),
            task, 0.0, Command.accepted([0.0, 0.0])))
        ages = observation.age_s.copy(); ages[index] += 0.01
        changes[f"age_{index}"] = changed_indices(adapter.policy_features(
            Observation(observation.values, observation.available, observation.fresh, ages, observation.time_s),
            task, 0.0, Command.accepted([0.0, 0.0])))
    for index in range(2):
        goal = task.goal_xz.copy(); goal[index] += 0.01
        changes[f"goal_{index}"] = changed_indices(adapter.policy_features(
            observation, Task(goal, deadline_s=task.deadline_s), 0.0, Command.accepted([0.0, 0.0])))
        command = np.zeros(2); command[index] = 0.125
        changes[f"previous_command_{index}"] = changed_indices(adapter.policy_features(
            observation, task, 0.0, Command.accepted(command)))
    changes["time_remaining"] = changed_indices(adapter.policy_features(
        observation, task, 0.01, Command.accepted([0.0, 0.0])))
    descriptor_values = descriptor.as_array()
    for index in range(len(descriptor_values)):
        varied = descriptor_values.copy()
        varied[index] += 0.01
        varied_descriptor = RobotDescriptor(varied[:2], varied[2:4], varied[4:6], varied[6:8],
                                             varied[8:].reshape(2, 2))
        varied_adapter = CurrentPacketFeatureAdapter(adapter.bundle, varied_descriptor, adapter.device)
        changes[f"descriptor_{index}"] = changed_indices(varied_adapter.policy_features(
            observation, task, 0.0, Command.accepted([0.0, 0.0])))
    expected = set(range(CURRENT_PACKET_POLICY_INPUT_DIM))
    covered = {index for indices in changes.values() for index in indices}
    return {"feature_width": CURRENT_PACKET_POLICY_INPUT_DIM,
            "changed_feature_indices": changes,
            "uncovered_feature_indices": sorted(expected - covered),
            "passed": not (expected - covered)}


def _current_packet_action_sensitivity(policy: PolicyNetwork, features: Tensor) -> dict[str, object]:
    """Check that every saved input coordinate can affect the selected action."""

    sample_count = min(len(features), 128)
    sample_indices = torch.linspace(0, len(features) - 1, sample_count, device=features.device).round().long()
    samples = features[sample_indices]
    with torch.no_grad():
        reference = policy.deterministic(samples)
        responses: list[float] = []
        for index in range(CURRENT_PACKET_POLICY_INPUT_DIM):
            candidate = samples.clone()
            span = float((features[:, index].max() - features[:, index].min()).abs().cpu())
            candidate[:, index] += max(0.05, span * 0.1)
            responses.append(float((policy.deterministic(candidate) - reference).abs().max().cpu()))
    threshold = 1e-7
    inactive = [index for index, response in enumerate(responses) if response <= threshold]
    return {"sample_count": sample_count, "threshold": threshold,
            "max_abs_action_change_by_feature": responses,
            "inactive_feature_indices": inactive, "passed": not inactive}


def audit_current_packet_comparator(run: str | Path, trace: str | Path, output: str | Path, *,
                                    device: str = "cuda", physics_dt: float = 0.001,
                                    control_dt: float = 0.01,
                                    collector_case_indices: tuple[int, ...] = (0, 39),
                                    config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Audit the saved C3 current-packet comparator without changing its weights.

    The historical training tensor has no raw packet fields.  This audit therefore
    saves new, teacher-driven public packets from the exact collector, reloads them
    through a fresh ``CurrentPacketRuntime``, and separately replays the immutable
    selected evaluation trace.  Neither path passes simulation truth to the
    runtime.
    """

    if not collector_case_indices or len(set(collector_case_indices)) != len(collector_case_indices):
        raise ValueError("collector_case_indices must be a nonempty sequence of distinct case IDs")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA current-packet audit but CUDA is unavailable")
    if not np.isclose(round(control_dt / physics_dt) * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    run_path = confined_path(run, must_exist=True)
    report_path = run_path / "metrics.json"
    training = json.loads(report_path.read_text(encoding="utf-8"))
    if training.get("controller_kind") != "current-packet-feedforward":
        raise ValueError("audit requires a current-packet comparator run")
    if int(training.get("policy_input_dim", -1)) != CURRENT_PACKET_POLICY_INPUT_DIM:
        raise ValueError("comparator does not declare the current 51-field schema")
    world_path = confined_path(str(training["world_bundle"]), must_exist=True)
    checkpoint = _current_packet_checkpoint(run_path, training)
    trace_path = confined_path(trace, must_exist=True)
    destination = confined_path(output); destination.mkdir(parents=True, exist_ok=True)
    bundle = WorldBundle.load(world_path, device=device_obj); bundle.model.eval()
    policy = PolicyNetwork(input_dim=CURRENT_PACKET_POLICY_INPUT_DIM,
                           hidden_dim=int(training["policy_hidden_dim"])).to(device_obj)
    state = torch.load(checkpoint, map_location=device_obj, weights_only=True)
    policy.load_state_dict(state); policy.eval()
    demonstrations_path = run_path / "packet-demonstrations.pt"
    demonstrations = torch.load(demonstrations_path, map_location=device_obj, weights_only=True)
    required = {"features", "teacher_actions", "case_indices"}
    if set(demonstrations) != required:
        raise ValueError("saved current-packet demonstrations have an unexpected schema")
    features = demonstrations["features"]
    labels = demonstrations["teacher_actions"]
    case_ids = demonstrations["case_indices"]
    if (features.ndim != 2 or features.shape[1] != CURRENT_PACKET_POLICY_INPUT_DIM or
            labels.shape != (len(features), 2) or case_ids.shape != (len(features),)):
        raise ValueError("saved current-packet demonstrations have invalid tensor shapes")
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError("saved current-packet demonstrations must be finite")
    reloaded_policy = PolicyNetwork(input_dim=CURRENT_PACKET_POLICY_INPUT_DIM,
                                    hidden_dim=int(training["policy_hidden_dim"])).to(device_obj)
    reloaded_policy.load_state_dict(torch.load(checkpoint, map_location=device_obj, weights_only=True))
    reloaded_policy.eval()
    with torch.no_grad():
        reload_difference = float((policy.deterministic(features) - reloaded_policy.deterministic(features)).abs().max().cpu())
    descriptor = default_descriptor()
    cases = healthy_cases(64)
    collector_episodes = [_collect_packet_clone_episode(bundle, device=device_obj, case_index=case_index,
                                                         physics_dt=physics_dt, control_dt=control_dt)
                          for case_index in collector_case_indices]
    collector_path = destination / "collector-current-packet-traces.npz"
    collector_features = []
    for episode in collector_episodes:
        _, goal_q = cases[episode.case_index]
        task = _task_from_goal_q(descriptor, goal_q)
        previous = Command.accepted([0.0, 0.0])
        current = []
        for tick, label in enumerate(episode.teacher_actions):
            observation = Observation(episode.packet_values[tick], episode.packet_available[tick],
                                      episode.packet_fresh[tick], episode.packet_age_s[tick], tick * control_dt)
            current.append(CurrentPacketFeatureAdapter(bundle, descriptor, device_obj).policy_features(
                observation, task, tick * control_dt, previous))
            previous = Command.accepted(label.detach().cpu().numpy())
        collector_features.append(torch.stack(current).detach().cpu().numpy())
    np.savez(collector_path,
             case_indices=np.asarray(collector_case_indices, dtype=np.int64),
             collector_features=np.stack(collector_features),
             teacher_actions=torch.stack([episode.teacher_actions for episode in collector_episodes]).detach().cpu().numpy(),
             packet_values=np.stack([episode.packet_values for episode in collector_episodes]),
             packet_available=np.stack([episode.packet_available for episode in collector_episodes]),
             packet_fresh=np.stack([episode.packet_fresh for episode in collector_episodes]),
             packet_age_s=np.stack([episode.packet_age_s for episode in collector_episodes]))
    max_collector_feature_difference = 0.0
    max_collector_action_difference = 0.0
    with np.load(collector_path, allow_pickle=False) as collector, torch.no_grad():
        for episode_index, case_index in enumerate(collector["case_indices"]):
            _, goal_q = cases[int(case_index)]
            task = _task_from_goal_q(descriptor, goal_q)
            runtime = CurrentPacketRuntime(bundle, descriptor, policy, device_obj)
            runtime.reset(descriptor, _current_packet_observation(collector, 0, control_dt, episode_index), task)
            for tick, teacher_action in enumerate(collector["teacher_actions"][episode_index]):
                expected_features = torch.as_tensor(collector["collector_features"][episode_index, tick],
                                                    dtype=torch.float32, device=device_obj)
                actual_features = runtime.features.policy_features(runtime.observation, task,
                                                                   runtime.elapsed_s, runtime.previous_command)
                max_collector_feature_difference = max(max_collector_feature_difference,
                                                        float((actual_features - expected_features).abs().max().cpu()))
                expected_action = policy.deterministic(expected_features).detach().cpu().numpy()
                max_collector_action_difference = max(max_collector_action_difference,
                                                       float(np.abs(runtime.act().values - expected_action).max()))
                runtime.observe(Command.accepted(teacher_action),
                                _current_packet_observation(collector, tick + 1, control_dt, episode_index))
    max_trace_action_difference = 0.0
    with np.load(trace_path, allow_pickle=False) as saved_trace:
        required_trace = {"packet_values", "packet_available", "packet_fresh", "packet_age_s", "accepted_action", "target"}
        if not required_trace.issubset(saved_trace.files):
            raise ValueError("selected comparator trace lacks required public packet fields")
        if len(saved_trace["packet_values"]) != len(saved_trace["accepted_action"]) + 1:
            raise ValueError("selected comparator trace must have N+1 packets for N commands")
        saved_task = Task(saved_trace["target"], deadline_s=1.5)
        runtime = CurrentPacketRuntime(bundle, descriptor, policy, device_obj)
        runtime.reset(descriptor, _current_packet_observation(saved_trace, 0, control_dt), saved_task)
        for tick, expected_action in enumerate(saved_trace["accepted_action"]):
            max_trace_action_difference = max(max_trace_action_difference,
                                              float(np.abs(runtime.act().values - expected_action).max()))
            runtime.observe(Command.accepted(expected_action),
                            _current_packet_observation(saved_trace, tick + 1, control_dt))
        reference = _current_packet_observation(saved_trace, 0, control_dt)
    first = CurrentPacketRuntime(bundle, descriptor, policy, device_obj)
    second = CurrentPacketRuntime(bundle, descriptor, policy, device_obj)
    first.reset(descriptor, reference, saved_task); second.reset(descriptor, reference, saved_task)
    first.observe(Command.accepted([0.2, -0.1]), reference)
    second.observe(Command.accepted([-0.4, 0.5]),
                   Observation(reference.values + 0.2, ~reference.available, ~reference.fresh,
                               reference.age_s + 0.01, reference.time_s))
    shared_previous = Command.accepted([0.3, -0.2])
    shared_packet = Observation(reference.values - 0.1, reference.available, reference.fresh,
                                reference.age_s + 0.02, reference.time_s + control_dt)
    first.observe(shared_previous, shared_packet); second.observe(shared_previous, shared_packet)
    history_difference = float(np.abs(first.act().values - second.act().values).max())
    coverage = _current_packet_feature_coverage(
        CurrentPacketFeatureAdapter(bundle, descriptor, device_obj), descriptor, reference, saved_task)
    sensitivity = _current_packet_action_sensitivity(policy, features)
    tolerance = 1e-6
    replay_passed = (max_collector_feature_difference <= tolerance and
                     max_collector_action_difference <= tolerance and
                     max_trace_action_difference <= tolerance and reload_difference <= tolerance and
                     history_difference <= tolerance)
    report = {
        "purpose": "C3c-1 current-packet comparator contract and reload audit",
        "run": str(run_path), "run_metrics_sha256": _file_sha256(report_path),
        "checkpoint": {"path": str(checkpoint), "sha256": _file_sha256(checkpoint)},
        "world_bundle": {"path": str(world_path), "metadata_sha256": _file_sha256(world_path / "metadata.json"),
                         "weights_sha256": _file_sha256(world_path / "weights.pt")},
        "demonstrations": {"path": str(demonstrations_path), "sha256": _file_sha256(demonstrations_path),
                             "transitions": len(features), "feature_width": int(features.shape[1]),
                             "case_ids": sorted(set(case_ids.detach().cpu().tolist()))},
        "optimization_budget": {key: training[key] for key in ("updates", "trajectories_per_update", "fit_passes_per_update", "checkpoint_interval", "seed")},
        "policy_parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "device": str(device_obj), "simulation_device": "cpu", "physics_dt": physics_dt,
        "control_dt": control_dt, "config_identity": config_identity,
        "feature_coverage": coverage, "action_sensitivity": sensitivity,
        "history_independence": {"same_packet_time_previous_command": True,
                                 "max_abs_action_difference": history_difference, "tolerance": tolerance,
                                 "passed": history_difference <= tolerance},
        "collector_runtime_replay": {"path": str(collector_path), "sha256": _file_sha256(collector_path),
                                       "case_indices": list(collector_case_indices), "feature_tolerance": tolerance,
                                       "action_tolerance": tolerance,
                                       "max_feature_abs_difference": max_collector_feature_difference,
                                       "max_action_abs_difference": max_collector_action_difference,
                                       "passed": max_collector_feature_difference <= tolerance and max_collector_action_difference <= tolerance},
        "selected_trace_replay": {"path": str(trace_path), "sha256": _file_sha256(trace_path),
                                  "max_action_abs_difference": max_trace_action_difference,
                                  "tolerance": tolerance, "passed": max_trace_action_difference <= tolerance},
        "saved_reload": {"max_action_abs_difference": reload_difference, "tolerance": tolerance,
                         "passed": reload_difference <= tolerance},
        "acceptance": {"passed": bool(coverage["passed"] and sensitivity["passed"] and replay_passed)},
        "source_sha256": {"train_policy.py": _file_sha256(Path(__file__)),
                          "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
                          "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py")},
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def audit_packet_clone_runtime_replay(world_bundle: str | Path, output: str | Path, *, device: str = "cuda",
                                      case_indices: tuple[int, ...] = (0, 39), physics_dt: float = 0.001,
                                      control_dt: float = 0.01,
                                      config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Replay saved real clone packets through a fresh Runtime without training.

    The clone collector uses ``FeatureAdapter`` directly. This diagnostic saves its
    raw packets/features/actions, reloads them, and drives a separate ``Runtime``
    instance using the collector's accepted teacher actions. It therefore detects a
    divergence between the two public-inference paths without exposing truth to the
    replayed runtime.
    """

    if not case_indices or len(set(case_indices)) != len(case_indices):
        raise ValueError("case_indices must be a nonempty sequence of distinct case IDs")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA packet replay but CUDA is unavailable")
    torch.manual_seed(20260913)
    world_path = confined_path(world_bundle, must_exist=True)
    bundle = WorldBundle.load(world_path, device=device_obj)
    bundle.model.eval()
    episodes = [_collect_packet_clone_episode(bundle, device=device_obj, case_index=index,
                                               physics_dt=physics_dt, control_dt=control_dt)
                for index in case_indices]
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / "collector-traces.npz"
    np.savez(trace_path,
             case_indices=np.asarray(case_indices, dtype=np.int64),
             collector_features=torch.stack([item.features for item in episodes]).detach().cpu().numpy(),
             teacher_actions=torch.stack([item.teacher_actions for item in episodes]).detach().cpu().numpy(),
             packet_values=np.stack([item.packet_values for item in episodes]),
             packet_available=np.stack([item.packet_available for item in episodes]),
             packet_fresh=np.stack([item.packet_fresh for item in episodes]),
             packet_age_s=np.stack([item.packet_age_s for item in episodes]),
             true_q=np.stack([item.true_q for item in episodes]),
             true_dq=np.stack([item.true_dq for item in episodes]))
    policy = PolicyNetwork().to(device_obj).eval()
    descriptor = default_descriptor()
    cases = healthy_cases(64)
    max_feature_difference = 0.0
    max_action_difference = 0.0
    with np.load(trace_path, allow_pickle=False) as traces, torch.no_grad():
        for episode_index, case_index in enumerate(traces["case_indices"]):
            case_id = int(case_index)
            _, goal_q = cases[case_id]
            task = _task_from_goal_q(descriptor, goal_q)
            initial = Observation(traces["packet_values"][episode_index, 0],
                                  traces["packet_available"][episode_index, 0],
                                  traces["packet_fresh"][episode_index, 0],
                                  traces["packet_age_s"][episode_index, 0], 0.0)
            runtime = Runtime(bundle, descriptor, policy, device_obj)
            runtime.reset(descriptor, initial, task)
            for tick, action in enumerate(traces["teacher_actions"][episode_index]):
                expected_features = torch.as_tensor(traces["collector_features"][episode_index, tick],
                                                    dtype=torch.float32, device=device_obj)
                runtime_features = runtime.features.policy_features(task, runtime.previous_command)
                max_feature_difference = max(max_feature_difference,
                                             float((runtime_features - expected_features).abs().max().cpu()))
                expected_action = policy.deterministic(expected_features).detach().cpu().numpy()
                runtime_action = runtime.act().values
                max_action_difference = max(max_action_difference,
                                            float(np.abs(runtime_action - expected_action).max()))
                next_index = tick + 1
                packet = Observation(traces["packet_values"][episode_index, next_index],
                                     traces["packet_available"][episode_index, next_index],
                                     traces["packet_fresh"][episode_index, next_index],
                                     traces["packet_age_s"][episode_index, next_index],
                                     next_index * control_dt)
                runtime.observe(Command.accepted(action), packet)
    report = {
        "purpose": "C2 real clone-collector packet replay through fresh Runtime",
        "world_bundle": str(world_path),
        "world_bundle_identity": {"metadata_sha256": _file_sha256(world_path / "metadata.json"),
                                  "weights_sha256": _file_sha256(world_path / "weights.pt")},
        "device": str(device_obj), "simulation_device": "cpu", "case_indices": list(case_indices),
        "physics_dt": physics_dt, "control_dt": control_dt, "config_identity": config_identity,
        "trace": {"path": str(trace_path), "sha256": _file_sha256(trace_path)},
        "replay": {"feature_tolerance": 1e-5, "action_tolerance": 1e-5,
                   "max_feature_abs_difference": max_feature_difference,
                   "max_action_abs_difference": max_action_difference,
                   "passed": bool(max_feature_difference <= 1e-5 and max_action_difference <= 1e-5)},
        "source_sha256": {"train_policy.py": _file_sha256(Path(__file__)),
                          "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
                          "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py"),
                          "sensors.py": _file_sha256(Path(__file__).parents[1] / "sim" / "sensors.py")},
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train_packet_behavior_clone(world_bundle: str | Path, output: str | Path, *, device: str = "cuda",
                                episodes: int = 30, clone_epochs: int = 30, checkpoint_interval: int = 5,
                                seed: int = 0, physics_dt: float = 0.001, control_dt: float = 0.01,
                                memoryless: bool = False,
                                config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Train a P6-sized policy on R5b1's clean causal P3 sensor packets."""

    if episodes < 1 or clone_epochs < 1 or checkpoint_interval < 1:
        raise ValueError("episodes, clone_epochs, and checkpoint_interval must be positive")
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA packet behavior cloning but CUDA is unavailable")
    world_path = confined_path(world_bundle, must_exist=True)
    bundle = WorldBundle.load(world_path, device=device_obj)
    bundle.model.eval()
    features, teacher_actions, case_indices = _collect_packet_clone_data(
        bundle, device=device_obj, episodes=episodes, physics_dt=physics_dt, control_dt=control_dt
    )
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.detach().cpu(), "teacher_actions": teacher_actions.detach().cpu(),
                "case_indices": case_indices},
               destination / "packet-demonstrations.pt")
    policy = PolicyNetwork().to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination / "initial-policy.pt")
    torch.save(policy.state_dict(), checkpoints / "initial-policy.pt")
    train_features = features.clone()
    if memoryless:
        train_features[:, :LEARNED_STATE_DIM] = 0.0
    history: list[dict[str, float]] = []
    for epoch in range(1, clone_epochs + 1):
        prediction = policy.deterministic(train_features)
        loss = F.mse_loss(prediction, teacher_actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        history.append({"epoch": float(epoch), "behavior_clone_loss": float(loss.detach()),
                        "demonstrations": float(len(train_features))})
        if epoch % checkpoint_interval == 0 or epoch == clone_epochs:
            torch.save(policy.state_dict(), checkpoints / f"policy-update-{epoch:06d}.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    report = {
        "world_bundle": str(world_path),
        "world_bundle_identity": {
            "metadata_sha256": _file_sha256(world_path / "metadata.json"),
            "weights_sha256": _file_sha256(world_path / "weights.pt"),
        },
        "device": str(device_obj), "simulation_device": "cpu",
        "feature_and_optimizer_device": str(device_obj), "episodes": episodes, "clone_epochs": clone_epochs,
        "checkpoint_interval": checkpoint_interval, "seed": seed, "physics_dt": physics_dt,
        "control_dt": control_dt, "memoryless": memoryless, "policy_input_dim": POLICY_INPUT_DIM,
        "policy_feature_schema": "belief-decoded-state-goal-q-v3", "selected_checkpoint": None,
        "config_identity": config_identity, "packet_contract": {
            "periods_s": _clean_packet_settings(control_dt).periods_s.tolist(),
            "noise_std": [0.0] * 8, "bias_limit": [0.0] * 8, "max_delay_s": [0.0] * 8,
        },
        "teacher": "simulation-truth training labels only; learned features use SensorSuite packets",
        "demonstration_case_ids": sorted(set(case_indices.tolist())),
        "packet_demonstrations_sha256": _file_sha256(destination / "packet-demonstrations.pt"),
        "source_sha256": {
            "train_policy.py": _file_sha256(Path(__file__)),
            "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
            "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py"),
            "native.py": _file_sha256(Path(__file__).parents[1] / "sim" / "native.py"),
            "sensors.py": _file_sha256(Path(__file__).parents[1] / "sim" / "sensors.py"),
        },
        "history": history,
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train_packet_progressive_clone(world_bundle: str | Path, output: str | Path, *, device: str = "cuda",
                                   updates: int = 30, trajectories_per_update: int = 4,
                                   fit_passes_per_update: int = 50, checkpoint_interval: int = 5,
                                   seed: int = 20260913, physics_dt: float = 0.001,
                                   control_dt: float = 0.01,
                                   config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Match P6's progressive clone schedule while using only clean P3 packets."""

    if min(updates, trajectories_per_update, fit_passes_per_update, checkpoint_interval) < 1:
        raise ValueError("R5b4 schedule counts must be positive")
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA progressive packet cloning but CUDA is unavailable")
    world_path = confined_path(world_bundle, must_exist=True)
    bundle = WorldBundle.load(world_path, device=device_obj)
    bundle.model.eval()
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    policy = PolicyNetwork().to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination / "initial-policy.pt")
    torch.save(policy.state_dict(), checkpoints / "initial-policy.pt")
    feature_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    case_batches: list[Tensor] = []
    history: list[dict[str, float]] = []
    for update in range(1, updates + 1):
        features, labels, case_ids = _collect_packet_clone_data(
            bundle, device=device_obj, episodes=trajectories_per_update, physics_dt=physics_dt,
            control_dt=control_dt, case_offset=(update - 1) * trajectories_per_update
        )
        feature_batches.append(features)
        label_batches.append(labels)
        case_batches.append(case_ids)
        aggregate_features = torch.cat(feature_batches)
        aggregate_labels = torch.cat(label_batches)
        loss = torch.zeros((), device=device_obj)
        for _ in range(fit_passes_per_update):
            prediction = policy.deterministic(aggregate_features)
            loss = F.mse_loss(prediction, aggregate_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        history.append({"update": float(update), "behavior_clone_loss": float(loss.detach()),
                        "demonstrations": float(len(aggregate_features))})
        if update % checkpoint_interval == 0 or update == updates:
            torch.save(policy.state_dict(), checkpoints / f"policy-update-{update:06d}.pt")
    all_features, all_labels, all_case_ids = torch.cat(feature_batches), torch.cat(label_batches), torch.cat(case_batches)
    torch.save({"features": all_features.detach().cpu(), "teacher_actions": all_labels.detach().cpu(),
                "case_indices": all_case_ids}, destination / "packet-demonstrations.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    report = {
        "algorithm": "R5b4 progressive clean-packet behavior cloning", "world_bundle": str(world_path),
        "world_bundle_identity": {"metadata_sha256": _file_sha256(world_path / "metadata.json"),
                                  "weights_sha256": _file_sha256(world_path / "weights.pt")},
        "device": str(device_obj), "simulation_device": "cpu", "feature_and_optimizer_device": str(device_obj),
        "updates": updates, "trajectories_per_update": trajectories_per_update,
        "fit_passes_per_update": fit_passes_per_update, "checkpoint_interval": checkpoint_interval,
        "seed": seed, "physics_dt": physics_dt, "control_dt": control_dt, "memoryless": False,
        "policy_input_dim": POLICY_INPUT_DIM, "policy_feature_schema": "belief-decoded-state-goal-q-v3",
        "selected_checkpoint": None, "config_identity": config_identity,
        "demonstration_case_ids": sorted(set(all_case_ids.tolist())),
        "packet_demonstrations_sha256": _file_sha256(destination / "packet-demonstrations.pt"),
        "teacher": "simulation-truth training labels only; learned features use SensorSuite packets",
        "packet_contract": {"periods_s": _clean_packet_settings(control_dt).periods_s.tolist(),
                            "noise_std": [0.0] * 8, "bias_limit": [0.0] * 8, "max_delay_s": [0.0] * 8},
        "source_sha256": {"train_policy.py": _file_sha256(Path(__file__)),
                          "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
                          "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py")},
        "history": history,
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def train_packet_delta_clone(source: str | Path, output: str | Path, *, device: str = "cuda",
                             delta_weight: float = 0.1, epochs: int = 100,
                             checkpoint_interval: int = 25, seed: int = 20260913) -> dict[str, object]:
    """Fit ordered packet demonstrations with an adjacent-action teacher loss."""
    if delta_weight < 0 or epochs < 1 or checkpoint_interval < 1:
        raise ValueError("delta weight and schedule must be non-negative/positive")
    torch.manual_seed(seed)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA delta clone but CUDA is unavailable")
    run = confined_path(source, must_exist=True)
    metrics = json.loads((run / "metrics.json").read_text())
    checkpoint = run / "best-policy.pt"
    if not checkpoint.exists():
        raise ValueError("delta clone requires a selected source checkpoint")
    data = torch.load(run / "packet-demonstrations.pt", map_location=dev, weights_only=True)
    features, labels = data["features"], data["teacher_actions"]
    case_ids = data["case_indices"]
    if len(features) % 170 or not torch.equal(case_ids.reshape(-1, 170), case_ids.reshape(-1, 170)[:, :1].expand(-1, 170)):
        raise ValueError("delta clone requires ordered 170-tick packet demonstrations")
    sequence_features = features.reshape(-1, 170, POLICY_INPUT_DIM)
    sequence_labels = labels.reshape(-1, 170, 2)
    policy = PolicyNetwork().to(dev)
    policy.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    destination = confined_path(output); destination.mkdir(parents=True, exist_ok=True)
    checks = destination / "checkpoints"; checks.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination / "initial-policy.pt"); torch.save(policy.state_dict(), checks / "initial-policy.pt")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        prediction = policy.deterministic(sequence_features.reshape(-1, POLICY_INPUT_DIM)).reshape_as(sequence_labels)
        action_loss = F.mse_loss(prediction, sequence_labels)
        delta_loss = F.mse_loss(prediction[:, 1:] - prediction[:, :-1], sequence_labels[:, 1:] - sequence_labels[:, :-1])
        loss = action_loss + delta_weight * delta_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), .5); optimizer.step()
        history.append({"epoch": float(epoch), "loss": float(loss.detach()), "action_loss": float(action_loss.detach()), "delta_loss": float(delta_loss.detach())})
        if epoch % checkpoint_interval == 0 or epoch == epochs: torch.save(policy.state_dict(), checks / f"policy-update-{epoch:06d}.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    report = {"algorithm":"R5b5 action-delta packet clone","world_bundle":metrics["world_bundle"],"device":str(dev),"simulation_device":"cpu","feature_and_optimizer_device":str(dev),"memoryless":False,"policy_input_dim":POLICY_INPUT_DIM,"selected_checkpoint":None,"delta_weight":delta_weight,"epochs":epochs,"checkpoint_interval":checkpoint_interval,"seed":seed,"initialization":{"path":str(checkpoint),"sha256":_file_sha256(checkpoint)},"history":history}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def _collect_packet_dagger_data(bundle: WorldBundle, policy: PolicyNetwork, *, device: torch.device,
                                physics_dt: float, control_dt: float) -> tuple[Tensor, Tensor, dict[str, np.ndarray], float]:
    """Collect clean P3 Runtime features with truth used only for stored teacher labels."""

    substeps = round(control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    descriptor = default_descriptor()
    settings = _clean_packet_settings(control_dt)
    controls = round(1.7 / control_dt)
    features: list[Tensor] = []
    teacher_actions: list[Tensor] = []
    packets: list[np.ndarray] = []
    available: list[np.ndarray] = []
    fresh: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    truth_q: list[np.ndarray] = []
    truth_dq: list[np.ndarray] = []
    max_action_replay_difference = 0.0
    policy.eval()
    with torch.no_grad():
        for case_index, (start, goal_q) in enumerate(healthy_cases(64)):
            task = _task_from_goal_q(descriptor, goal_q)
            if round(task.total_duration_s / control_dt) != controls:
                raise ValueError("R5b2 task duration must be 170 packet-control ticks")
            arm = ImperfectNativeArm(descriptor, ActuatorSettings.healthy(),
                                     SensorSuite(settings, seed=case_index), timestep=physics_dt)
            arm.reset(start)
            observation = arm.observation()
            runtime = Runtime(bundle, descriptor, policy, device)
            runtime.reset(descriptor, observation, task)
            teacher = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0.0 else -1,
                                         computed_torque=True, hold_goal=True)
            case_packets = [observation.values.copy()]
            case_available = [observation.available.copy()]
            case_fresh = [observation.fresh.copy()]
            case_ages = [observation.age_s.copy()]
            truth = arm.arm.truth()
            case_q = [truth.q.copy()]
            case_dq = [truth.dq.copy()]
            case_actions: list[np.ndarray] = []
            case_labels: list[np.ndarray] = []
            for control_index in range(controls):
                runtime_features = runtime.features.policy_features(task, runtime.previous_command).detach().clone()
                command = runtime.act()
                expected_action = policy.deterministic(runtime_features).detach().cpu().numpy()
                max_action_replay_difference = max(
                    max_action_replay_difference, float(np.max(np.abs(command.values - expected_action)))
                )
                features.append(runtime_features)
                label = teacher.act(control_index * control_dt,
                                    _truth_teacher_observation(arm, control_index * control_dt))
                teacher_actions.append(torch.as_tensor(label, dtype=torch.float32, device=device))
                case_actions.append(command.values.copy())
                case_labels.append(label.copy())
                for _ in range(substeps):
                    truth = arm.step(command)
                observation = arm.observation()
                runtime.observe(command, observation)
                case_packets.append(observation.values.copy())
                case_available.append(observation.available.copy())
                case_fresh.append(observation.fresh.copy())
                case_ages.append(observation.age_s.copy())
                case_q.append(truth.q.copy())
                case_dq.append(truth.dq.copy())
            packets.append(np.asarray(case_packets))
            available.append(np.asarray(case_available))
            fresh.append(np.asarray(case_fresh))
            ages.append(np.asarray(case_ages))
            actions.append(np.asarray(case_actions))
            labels.append(np.asarray(case_labels))
            truth_q.append(np.asarray(case_q))
            truth_dq.append(np.asarray(case_dq))
    if max_action_replay_difference > 1e-6:
        raise RuntimeError(f"R5b2 Runtime feature/action replay differs by {max_action_replay_difference}")
    trace = {
        "packet_values": np.asarray(packets), "packet_available": np.asarray(available),
        "packet_fresh": np.asarray(fresh), "packet_age_s": np.asarray(ages),
        "accepted_action": np.asarray(actions), "teacher_action_training_only": np.asarray(labels),
        "truth_q_training_only": np.asarray(truth_q), "truth_dq_training_only": np.asarray(truth_dq),
    }
    return torch.stack(features), torch.stack(teacher_actions), trace, max_action_replay_difference


def train_packet_dagger(run: str | Path, output: str | Path, *, device: str = "cuda", clone_epochs: int = 500,
                        checkpoint_interval: int = 125, seed: int = 20260913, physics_dt: float = 0.001,
                        control_dt: float = 0.01, learning_rate: float = 3e-4,
                        config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Run R5b2's one-pass clean-packet DAgger correction from a selected policy."""

    if clone_epochs < 1 or checkpoint_interval < 1 or learning_rate <= 0.0:
        raise ValueError("clone_epochs, checkpoint_interval, and learning_rate must be positive")
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA packet DAgger training but CUDA is unavailable")
    source_run = confined_path(run, must_exist=True)
    source_metrics = json.loads((source_run / "metrics.json").read_text(encoding="utf-8"))
    if bool(source_metrics.get("memoryless", False)):
        raise ValueError("R5b2 must initialize from the selected adaptive packet policy")
    source_checkpoint = source_run / "best-policy.pt"
    if not source_checkpoint.exists() or not source_metrics.get("selected_checkpoint"):
        raise ValueError("R5b2 requires a selected source packet checkpoint")
    world_path = confined_path(source_metrics["world_bundle"], must_exist=True)
    bundle = WorldBundle.load(world_path, device=device_obj)
    bundle.model.eval()
    source_policy = PolicyNetwork().to(device_obj)
    source_policy.load_state_dict(torch.load(source_checkpoint, map_location=device_obj, weights_only=True))
    original = torch.load(source_run / "packet-demonstrations.pt", map_location="cpu", weights_only=True)
    original_features = original["features"].to(device_obj)
    original_labels = original["teacher_actions"].to(device_obj)
    if original_features.ndim != 2 or original_features.shape[1] != POLICY_INPUT_DIM or original_labels.shape != (len(original_features), 2):
        raise ValueError("source packet demonstrations have an unsupported shape")
    dagger_features, dagger_labels, trace, replay_difference = _collect_packet_dagger_data(
        bundle, source_policy, device=device_obj, physics_dt=physics_dt, control_dt=control_dt
    )
    expected_on_policy_examples = 64 * round(1.7 / control_dt)
    if dagger_features.shape != (expected_on_policy_examples, POLICY_INPUT_DIM) or dagger_labels.shape != (
            expected_on_policy_examples, 2):
        raise RuntimeError("on-policy DAgger feature/teacher-label collection is misaligned")
    features = torch.cat((original_features, dagger_features))
    labels = torch.cat((original_labels, dagger_labels))
    destination = confined_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({
        "original_features": original_features.detach().cpu(), "original_teacher_actions": original_labels.detach().cpu(),
        "on_policy_features": dagger_features.detach().cpu(), "on_policy_teacher_actions": dagger_labels.detach().cpu(),
    }, destination / "packet-demonstrations.pt")
    np.savez(destination / "dagger-rollouts.npz", **trace)  # type: ignore[arg-type]
    policy = PolicyNetwork().to(device_obj)
    policy.load_state_dict(torch.load(source_checkpoint, map_location=device_obj, weights_only=True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination / "initial-policy.pt")
    torch.save(policy.state_dict(), checkpoints / "initial-policy.pt")
    history: list[dict[str, float]] = []
    for epoch in range(1, clone_epochs + 1):
        prediction = policy.deterministic(features)
        loss = F.mse_loss(prediction, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        history.append({"epoch": float(epoch), "behavior_clone_loss": float(loss.detach()),
                        "demonstrations": float(len(features))})
        if epoch % checkpoint_interval == 0 or epoch == clone_epochs:
            torch.save(policy.state_dict(), checkpoints / f"policy-update-{epoch:06d}.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    report = {
        "algorithm": "R5b2 clean-packet DAgger", "world_bundle": str(world_path),
        "world_bundle_identity": {"metadata_sha256": _file_sha256(world_path / "metadata.json"),
                                  "weights_sha256": _file_sha256(world_path / "weights.pt")},
        "initialization": {"run": str(source_run), "checkpoint": str(source_checkpoint),
                           "checkpoint_sha256": _file_sha256(source_checkpoint),
                           "selected_checkpoint": source_metrics["selected_checkpoint"]},
        "device": str(device_obj), "simulation_device": "cpu", "feature_and_optimizer_device": str(device_obj),
        "clone_epochs": clone_epochs, "checkpoint_interval": checkpoint_interval, "seed": seed,
        "optimizer": {"name": "Adam", "state": "fresh", "learning_rate": learning_rate,
                      "gradient_norm_cap": 0.5},
        "physics_dt": physics_dt, "control_dt": control_dt, "memoryless": False,
        "policy_input_dim": POLICY_INPUT_DIM, "policy_feature_schema": "belief-decoded-state-goal-q-v3",
        "selected_checkpoint": None, "config_identity": config_identity,
        "packet_contract": {"periods_s": _clean_packet_settings(control_dt).periods_s.tolist(),
                            "noise_std": [0.0] * 8, "bias_limit": [0.0] * 8, "max_delay_s": [0.0] * 8},
        "demonstrations": {"original_teacher_forced": len(original_features), "on_policy": len(dagger_features),
                           "aggregate": len(features)},
        "teacher": "truth-derived stored training labels only; policy Runtime receives SensorSuite packets",
        "runtime_action_replay_max_abs": replay_difference,
        "artifacts": {"demonstrations": str(destination / "packet-demonstrations.pt"),
                      "on_policy_rollouts": str(destination / "dagger-rollouts.npz")},
        "source_sha256": {"train_policy.py": _file_sha256(Path(__file__)),
                          "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
                          "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py"),
                          "native.py": _file_sha256(Path(__file__).parents[1] / "sim" / "native.py"),
                          "sensors.py": _file_sha256(Path(__file__).parents[1] / "sim" / "sensors.py")},
        "history": history,
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _collect_decoded_feedback_correction_data(bundle: WorldBundle, policy: PolicyNetwork, *,
                                               device: torch.device, case_indices: tuple[int, ...],
                                               physics_dt: float, control_dt: float) -> tuple[Tensor, Tensor, dict[str, np.ndarray]]:
    """Collect successful feedback labels from decoded state and public packets only."""

    substeps = round(control_dt / physics_dt)
    if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    descriptor = default_descriptor()
    settings = _clean_packet_settings(control_dt)
    features: list[Tensor] = []
    labels: list[Tensor] = []
    traces: dict[str, list[np.ndarray]] = {
        "case_indices": [], "packet_values": [], "packet_available": [], "packet_fresh": [],
        "packet_age_s": [], "accepted_action": [], "estimated_q": [], "estimated_dq": [], "target": [],
    }
    cases = healthy_cases(64)
    for case_index in case_indices:
        start, goal_q = cases[case_index]
        task = _task_from_goal_q(descriptor, goal_q)
        arm = ImperfectNativeArm(descriptor, ActuatorSettings.healthy(),
                                 SensorSuite(settings, seed=case_index), timestep=physics_dt)
        arm.reset(start)
        observation = arm.observation()
        runtime = Runtime(bundle, descriptor, policy, device)
        runtime.reset(descriptor, observation, task)
        teacher = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0.0 else -1,
                                     computed_torque=True, hold_goal=True)
        packets, available, fresh, ages = [observation.values.copy()], [observation.available.copy()], [observation.fresh.copy()], [observation.age_s.copy()]
        actions: list[np.ndarray] = []
        estimated_q: list[np.ndarray] = []
        estimated_dq: list[np.ndarray] = []
        for tick in range(round(task.total_duration_s / control_dt)):
            if runtime.features.belief is None:
                raise RuntimeError("decoded-feedback collector runtime was not reset")
            with torch.no_grad():
                decoded_q, decoded_dq = bundle.model.decode(runtime.features.belief)
            q = decoded_q.detach().cpu().numpy(); dq = decoded_dq.detach().cpu().numpy()
            features.append(runtime.features.policy_features(task, runtime.previous_command).detach().clone())
            teacher_packet = Observation(np.concatenate((q, dq, observation.values[4:])), observation.available,
                                         observation.fresh, observation.age_s, tick * control_dt)
            action = teacher.act(tick * control_dt, teacher_packet)
            labels.append(torch.as_tensor(action, dtype=torch.float32, device=device))
            actions.append(action.copy()); estimated_q.append(q.copy()); estimated_dq.append(dq.copy())
            for _ in range(substeps):
                arm.step(action)
            observation = arm.observation()
            runtime.observe(Command.accepted(action), observation)
            packets.append(observation.values.copy()); available.append(observation.available.copy())
            fresh.append(observation.fresh.copy()); ages.append(observation.age_s.copy())
        traces["case_indices"].append(np.asarray(case_index, dtype=np.int64))
        traces["packet_values"].append(np.asarray(packets)); traces["packet_available"].append(np.asarray(available))
        traces["packet_fresh"].append(np.asarray(fresh)); traces["packet_age_s"].append(np.asarray(ages))
        traces["accepted_action"].append(np.asarray(actions)); traces["estimated_q"].append(np.asarray(estimated_q))
        traces["estimated_dq"].append(np.asarray(estimated_dq)); traces["target"].append(task.goal_xz.copy())
    return torch.stack(features), torch.stack(labels), {key: np.asarray(value) for key, value in traces.items()}


def train_decoded_feedback_correction(run: str | Path, output: str | Path, *, device: str = "cuda",
                                      physics_dt: float = 0.001, control_dt: float = 0.01,
                                      config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Run C3c-2's one-step decoded-feedback correction with incumbent anchoring."""

    correction_cases = (39, 53)
    learning_rate = 1e-5
    torch.manual_seed(20260913)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA decoded-feedback correction but CUDA is unavailable")
    source_run = confined_path(run, must_exist=True)
    source_metrics_path = source_run / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    source_checkpoint = source_run / "best-policy.pt"
    if not source_checkpoint.exists() or source_metrics.get("selected_checkpoint") is None:
        raise ValueError("decoded-feedback correction requires a selected source packet checkpoint")
    if int(source_metrics.get("policy_input_dim", -1)) != POLICY_INPUT_DIM:
        raise ValueError("decoded-feedback correction requires the recurrent packet policy schema")
    world_path = confined_path(str(source_metrics["world_bundle"]), must_exist=True)
    destination = confined_path(output); destination.mkdir(parents=True, exist_ok=True)
    plan = {
        "purpose": "C3c-2 predeclared decoded-feedback supervision correction",
        "hypothesis": "Cases 39 and 53 can improve when the packet policy is taught the successful closed decoded-state feedback behavior.",
        "changed_factor": "teacher labels come from independent decoded-state feedback loops for cases 39 and 53; no truth-state DAgger labels are collected.",
        "incumbent_anchor": "all 20,400 stored feature inputs target the frozen incumbent actions",
        "initial_checkpoint": {"path": str(source_checkpoint), "sha256": _file_sha256(source_checkpoint)},
        "optimizer": {"name": "Adam", "state": "fresh", "learning_rate": learning_rate, "gradient_norm_cap": 0.5},
        "budget": {"new_teacher_transitions": 340, "fitting_passes": 1, "selection_checkpoints": ["initial", "policy-update-000001"],
                   "prior_ceiling": {"transitions": 20400, "fitting_passes": 1500}},
        "falsifier": "Reject this candidate if update 1 drops below 60/64, loses any incumbent success, or has no physical development gain; only a >=61/64 retained-success result proceeds to frozen 200-case validation.",
    }
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    bundle = WorldBundle.load(world_path, device=device_obj); bundle.model.eval()
    source_policy = PolicyNetwork().to(device_obj)
    source_policy.load_state_dict(torch.load(source_checkpoint, map_location=device_obj, weights_only=True)); source_policy.eval()
    original = torch.load(source_run / "packet-demonstrations.pt", map_location=device_obj, weights_only=True)
    original_features = original["features"]
    if original_features.ndim != 2 or original_features.shape[1] != POLICY_INPUT_DIM:
        raise ValueError("source demonstration features have an unsupported shape")
    with torch.no_grad():
        incumbent_actions = source_policy.deterministic(original_features)
    decoded_features, decoded_actions, trace = _collect_decoded_feedback_correction_data(
        bundle, source_policy, device=device_obj, case_indices=correction_cases,
        physics_dt=physics_dt, control_dt=control_dt)
    if decoded_features.shape != (len(correction_cases) * 170, POLICY_INPUT_DIM) or decoded_actions.shape != (len(correction_cases) * 170, 2):
        raise RuntimeError("decoded-feedback collection must contain two complete 170-tick traces")
    torch.save({"anchor_features": original_features.detach().cpu(), "incumbent_actions": incumbent_actions.detach().cpu(),
                "decoded_feedback_features": decoded_features.detach().cpu(), "decoded_feedback_actions": decoded_actions.detach().cpu()},
               destination / "packet-demonstrations.pt")
    np.savez(destination / "decoded-feedback-rollouts.npz", **trace)
    policy = PolicyNetwork().to(device_obj)
    policy.load_state_dict(torch.load(source_checkpoint, map_location=device_obj, weights_only=True))
    checkpoints = destination / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), destination / "initial-policy.pt")
    torch.save(policy.state_dict(), checkpoints / "initial-policy.pt")
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    prediction = policy.deterministic(decoded_features)
    decoded_loss = F.mse_loss(prediction, decoded_actions)
    anchor_loss = F.mse_loss(policy.deterministic(original_features), incumbent_actions)
    loss = decoded_loss + anchor_loss
    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
    with torch.no_grad():
        post_decoded_loss = float(F.mse_loss(policy.deterministic(decoded_features), decoded_actions).cpu())
        post_anchor_loss = float(F.mse_loss(policy.deterministic(original_features), incumbent_actions).cpu())
    torch.save(policy.state_dict(), checkpoints / "policy-update-000001.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    report = {
        "algorithm": "C3c-2 decoded-feedback anchored correction", "controller_kind": "recurrent-packet-policy",
        "world_bundle": str(world_path), "world_bundle_identity": {"metadata_sha256": _file_sha256(world_path / "metadata.json"),
                                                                       "weights_sha256": _file_sha256(world_path / "weights.pt")},
        "initialization": {"run": str(source_run), "checkpoint": str(source_checkpoint),
                           "checkpoint_sha256": _file_sha256(source_checkpoint), "selected_checkpoint": source_metrics["selected_checkpoint"]},
        "teacher": "closed decoded-state feedback on cases 39 and 53; packets and task context remain public; no truth-label DAgger collection",
        "correction_cases": list(correction_cases), "device": str(device_obj), "simulation_device": "cpu",
        "feature_and_optimizer_device": str(device_obj), "physics_dt": physics_dt, "control_dt": control_dt,
        "policy_input_dim": POLICY_INPUT_DIM, "policy_hidden_dim": 128, "policy_feature_schema": "belief-decoded-state-goal-q-v3",
        "optimizer": {"name": "Adam", "state": "fresh", "learning_rate": learning_rate, "gradient_norm_cap": 0.5},
        "budget": plan["budget"], "selected_checkpoint": None, "config_identity": config_identity,
        "demonstrations": {"anchor": len(original_features), "decoded_feedback": len(decoded_features),
                             "packet_demonstrations_sha256": _file_sha256(destination / "packet-demonstrations.pt"),
                             "decoded_rollouts_sha256": _file_sha256(destination / "decoded-feedback-rollouts.npz")},
        "history": [{"update": 1.0, "loss_before": float(loss.detach()), "decoded_loss_before": float(decoded_loss.detach()),
                     "anchor_loss_before": float(anchor_loss.detach()), "decoded_loss_after": post_decoded_loss,
                     "anchor_loss_after": post_anchor_loss}],
        "predeclared_falsifier": plan["falsifier"],
        "artifacts": {"plan": str(destination / "plan.json"), "demonstrations": str(destination / "packet-demonstrations.pt"),
                      "decoded_rollouts": str(destination / "decoded-feedback-rollouts.npz")},
        "source_sha256": {"train_policy.py": _file_sha256(Path(__file__)),
                          "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"),
                          "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py")},
    }
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _collect_support_matched_decoded_labels(bundle: WorldBundle, policy: PolicyNetwork, *, device: torch.device,
                                            case_indices: tuple[int, ...], physics_dt: float,
                                            control_dt: float,
                                            explicit_cases: list[tuple[np.ndarray, np.ndarray]] | None = None) -> tuple[Tensor, Tensor, dict[str, np.ndarray]]:
    """Label the incumbent's own public-packet trajectories with decoded feedback."""

    substeps = round(control_dt / physics_dt)
    descriptor, settings = default_descriptor(), _clean_packet_settings(control_dt)
    features: list[Tensor] = []; labels: list[Tensor] = []
    traces: dict[str, list[np.ndarray]] = {key: [] for key in ("case_indices", "packet_values", "packet_available",
        "packet_fresh", "packet_age_s", "incumbent_action", "decoded_teacher_action", "estimated_q", "estimated_dq", "target")}
    source_cases = healthy_cases(64) if explicit_cases is None else explicit_cases
    for case_index in case_indices:
        if not 0 <= case_index < len(source_cases):
            raise IndexError(f"case index {case_index} is outside the explicit {len(source_cases)}-case collector suite")
        start, goal_q = source_cases[case_index]; task = _task_from_goal_q(descriptor, goal_q)
        arm = ImperfectNativeArm(descriptor, ActuatorSettings.healthy(), SensorSuite(settings, seed=case_index), timestep=physics_dt)
        arm.reset(start); observation = arm.observation(); runtime = Runtime(bundle, descriptor, policy, device)
        runtime.reset(descriptor, observation, task)
        teacher = FeedbackController(descriptor, start, task, elbow=1 if goal_q[1] >= 0.0 else -1,
                                     computed_torque=True, hold_goal=True)
        packets, available, fresh, ages = [observation.values.copy()], [observation.available.copy()], [observation.fresh.copy()], [observation.age_s.copy()]
        incumbent_actions: list[np.ndarray] = []; labels_case: list[np.ndarray] = []; qs: list[np.ndarray] = []; dqs: list[np.ndarray] = []
        for tick in range(round(task.total_duration_s / control_dt)):
            if runtime.features.belief is None: raise RuntimeError("support-matched runtime was not reset")
            features.append(runtime.features.policy_features(task, runtime.previous_command).detach().clone())
            with torch.no_grad(): decoded_q, decoded_dq = bundle.model.decode(runtime.features.belief)
            q, dq = decoded_q.detach().cpu().numpy(), decoded_dq.detach().cpu().numpy()
            teacher_packet = Observation(np.concatenate((q, dq, observation.values[4:])), observation.available,
                                         observation.fresh, observation.age_s, tick * control_dt)
            label = teacher.act(tick * control_dt, teacher_packet); command = runtime.act()
            labels.append(torch.as_tensor(label, dtype=torch.float32, device=device)); incumbent_actions.append(command.values.copy())
            labels_case.append(label.copy()); qs.append(q.copy()); dqs.append(dq.copy())
            for _ in range(substeps): arm.step(command)
            observation = arm.observation(); runtime.observe(command, observation)
            packets.append(observation.values.copy()); available.append(observation.available.copy()); fresh.append(observation.fresh.copy()); ages.append(observation.age_s.copy())
        values = (np.asarray(case_index, dtype=np.int64), np.asarray(packets), np.asarray(available), np.asarray(fresh),
                  np.asarray(ages), np.asarray(incumbent_actions), np.asarray(labels_case), np.asarray(qs), np.asarray(dqs), task.goal_xz.copy())
        for key, value in zip(traces, values, strict=True): traces[key].append(value)
    return torch.stack(features), torch.stack(labels), {key: np.asarray(value) for key, value in traces.items()}


def train_support_matched_decoded_correction(run: str | Path, output: str | Path, *, device: str = "cuda",
                                             physics_dt: float = 0.001, control_dt: float = 0.01,
                                             config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """Run C3c-3's predeclared short decoded-label coverage correction."""

    cases, updates, checkpoints_to_save, learning_rate = (39, 53), 50, {1, 10, 25, 50}, 1e-5
    torch.manual_seed(20260913); device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested CUDA C3c-3 correction but CUDA is unavailable")
    source_run = confined_path(run, must_exist=True); source_metrics = json.loads((source_run / "metrics.json").read_text())
    source_checkpoint = source_run / "best-policy.pt"; world_path = confined_path(str(source_metrics["world_bundle"]), must_exist=True)
    if not source_checkpoint.exists() or int(source_metrics.get("policy_input_dim", -1)) != POLICY_INPUT_DIM: raise ValueError("C3c-3 requires selected recurrent packet incumbent")
    destination = confined_path(output); destination.mkdir(parents=True, exist_ok=True)
    plan = {"purpose": "C3c-3 predeclared support-matched decoded-feedback correction",
            "hypothesis": "Decoded labels on incumbent case-39/53 trajectories address the measured trajectory/feature mismatch enough to improve a failure without losing incumbent successes.",
            "uncertainty": "The mismatch supports this coverage test but does not establish the cause of C3c-2's no-gain result.",
            "changed_factor": "labels are queried on incumbent trajectories, not closed teacher loops and not truth-label DAgger.",
            "initial_checkpoint": {"path": str(source_checkpoint), "sha256": _file_sha256(source_checkpoint)},
            "optimizer": {"name": "Adam", "state": "fresh", "learning_rate": learning_rate, "gradient_norm_cap": 0.5},
            "loss": {"decoded_label_mean_weight": 1.0, "incumbent_anchor_mean_weight": 1.0},
            "budget": {"new_teacher_transitions": 340, "fitting_passes": updates, "selection_checkpoints": ["initial", "policy-update-000001", "policy-update-000010", "policy-update-000025", "policy-update-000050"], "prior_ceiling": {"transitions": 20400, "fitting_passes": 1500}},
            "stopping_rule": "Update 1 must retain 60/64 and show no hard violation. Select only checkpoints with all incumbent successes; stop after 50 passes. Reject if no selected checkpoint improves case-39/53 hold/error trend or reaches >=61/64. Run frozen validation only after that development gate."}
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    bundle = WorldBundle.load(world_path, device=device_obj); bundle.model.eval(); source_policy = PolicyNetwork().to(device_obj)
    source_policy.load_state_dict(torch.load(source_checkpoint, map_location=device_obj, weights_only=True)); source_policy.eval()
    original_features = torch.load(source_run / "packet-demonstrations.pt", map_location=device_obj, weights_only=True)["features"]
    with torch.no_grad(): anchors = source_policy.deterministic(original_features)
    features, labels, trace = _collect_support_matched_decoded_labels(bundle, source_policy, device=device_obj, case_indices=cases, physics_dt=physics_dt, control_dt=control_dt)
    if features.shape != (340, POLICY_INPUT_DIM) or labels.shape != (340, 2): raise RuntimeError("C3c-3 must collect two complete 170-tick trajectories")
    torch.save({"anchor_features": original_features.detach().cpu(), "incumbent_actions": anchors.detach().cpu(), "support_features": features.detach().cpu(), "decoded_feedback_actions": labels.detach().cpu()}, destination / "packet-demonstrations.pt")
    np.savez(destination / "support-matched-rollouts.npz", **trace)
    policy = PolicyNetwork().to(device_obj); policy.load_state_dict(torch.load(source_checkpoint, map_location=device_obj, weights_only=True)); optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    checkpoint_dir = destination / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True); torch.save(policy.state_dict(), destination / "initial-policy.pt"); torch.save(policy.state_dict(), checkpoint_dir / "initial-policy.pt")
    history: list[dict[str, float]] = []
    for update in range(1, updates + 1):
        decoded_loss, anchor_loss = F.mse_loss(policy.deterministic(features), labels), F.mse_loss(policy.deterministic(original_features), anchors)
        loss = decoded_loss + anchor_loss; optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()
        with torch.no_grad(): post_decoded = float(F.mse_loss(policy.deterministic(features), labels).cpu()); post_anchor = float(F.mse_loss(policy.deterministic(original_features), anchors).cpu())
        history.append({"update": float(update), "decoded_loss_before": float(decoded_loss.detach()), "anchor_loss_before": float(anchor_loss.detach()), "decoded_loss_after": post_decoded, "anchor_loss_after": post_anchor})
        if update in checkpoints_to_save: torch.save(policy.state_dict(), checkpoint_dir / f"policy-update-{update:06d}.pt")
    torch.save(policy.state_dict(), destination / "final-policy.pt")
    report = {"algorithm": "C3c-3 support-matched decoded-feedback anchored correction", "controller_kind": "recurrent-packet-policy", "world_bundle": str(world_path), "initialization": plan["initial_checkpoint"], "teacher": plan["changed_factor"], "correction_cases": list(cases), "device": str(device_obj), "simulation_device": "cpu", "feature_and_optimizer_device": str(device_obj), "physics_dt": physics_dt, "control_dt": control_dt, "policy_input_dim": POLICY_INPUT_DIM, "policy_hidden_dim": 128, "optimizer": plan["optimizer"], "budget": plan["budget"], "selected_checkpoint": None, "config_identity": config_identity, "demonstrations": {"anchor": len(original_features), "decoded_feedback": len(features)}, "history": history, "predeclared_stopping_rule": plan["stopping_rule"], "artifacts": {"plan": str(destination / "plan.json"), "rollouts": str(destination / "support-matched-rollouts.npz")}, "source_sha256": {"train_policy.py": _file_sha256(Path(__file__)), "runtime.py": _file_sha256(Path(__file__).parents[1] / "control" / "runtime.py"), "features.py": _file_sha256(Path(__file__).parents[1] / "control" / "features.py")}}
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n"); return report


def train_full_decoded_feedback_distillation(run: str | Path, output: str | Path, *, device: str = "cuda",
                                             physics_dt: float = 0.001, control_dt: float = 0.01,
                                             config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """C3c-4: distill decoded feedback on all incumbent packet trajectories."""
    updates, saved, anchor_weight, learning_rate = 25, {1, 5, 10, 25}, 5.0, 1e-5
    torch.manual_seed(20260913); dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested CUDA decoded distillation but CUDA is unavailable")
    source = confined_path(run, must_exist=True); metrics = json.loads((source / "metrics.json").read_text())
    checkpoint = source / "best-policy.pt"; world_path = confined_path(str(metrics["world_bundle"]), must_exist=True)
    if not checkpoint.exists(): raise ValueError("C3c-4 requires incumbent checkpoint")
    dest = confined_path(output); dest.mkdir(parents=True, exist_ok=True)
    plan = {"purpose":"C3c-4 predeclared all-development decoded-feedback distillation","hypothesis":"A learned policy can move toward the 62/64 decoded-feedback reference when labels and anchors share all 64 incumbent trajectories.","teacher":"frozen decoded-state feedback; public packets plus shared frozen observer only","collection":{"cases":"all original development 64","new_transitions":10880},"loss":{"decoded_label_mean_weight":1.0,"incumbent_trajectory_action_mean_weight":anchor_weight},"optimizer":{"name":"Adam","state":"fresh","learning_rate":learning_rate,"gradient_norm_cap":0.5},"budget":{"fitting_passes":updates,"selection_checkpoints":["initial","policy-update-000001","policy-update-000005","policy-update-000010","policy-update-000025"],"ceiling":{"new_transitions":20400,"fitting_passes":1500}},"stopping_rule":"Evaluate every saved checkpoint on development. Select only >=61/64 retaining all incumbent 60 and no successful-episode hard violations; update 1 checks stability, later checkpoints may improve. Otherwise close this candidate; validation only after selection."}
    (dest / "plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    bundle=WorldBundle.load(world_path,device=dev); bundle.model.eval(); incumbent=PolicyNetwork().to(dev); incumbent.load_state_dict(torch.load(checkpoint,map_location=dev,weights_only=True)); incumbent.eval()
    features, labels, trace = _collect_support_matched_decoded_labels(bundle,incumbent,device=dev,case_indices=tuple(range(64)),physics_dt=physics_dt,control_dt=control_dt)
    with torch.no_grad(): anchors=incumbent.deterministic(features)
    if features.shape != (10880,POLICY_INPUT_DIM): raise RuntimeError("C3c-4 collector did not cover 64 complete trajectories")
    torch.save({"incumbent_trajectory_features":features.detach().cpu(),"incumbent_actions":anchors.detach().cpu(),"decoded_feedback_actions":labels.detach().cpu()},dest/"packet-demonstrations.pt");np.savez(dest/"decoded-feedback-rollouts.npz",**trace)
    policy=PolicyNetwork().to(dev);policy.load_state_dict(torch.load(checkpoint,map_location=dev,weights_only=True));opt=torch.optim.Adam(policy.parameters(),lr=learning_rate);checks=dest/"checkpoints";checks.mkdir(parents=True,exist_ok=True);torch.save(policy.state_dict(),dest/"initial-policy.pt");torch.save(policy.state_dict(),checks/"initial-policy.pt")
    history=[]
    for step in range(1,updates+1):
        target=F.mse_loss(policy.deterministic(features),labels);anchor=F.mse_loss(policy.deterministic(features),anchors);loss=target+anchor_weight*anchor;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);opt.step()
        with torch.no_grad(): history.append({"update":float(step),"decoded_loss":float(F.mse_loss(policy.deterministic(features),labels)),"anchor_loss":float(F.mse_loss(policy.deterministic(features),anchors))})
        if step in saved:torch.save(policy.state_dict(),checks/f"policy-update-{step:06d}.pt")
    torch.save(policy.state_dict(),dest/"final-policy.pt")
    report={"algorithm":"C3c-4 all-development decoded-feedback distillation","controller_kind":"recurrent-packet-policy","world_bundle":str(world_path),"initialization":{"path":str(checkpoint),"sha256":_file_sha256(checkpoint)},"device":str(dev),"simulation_device":"cpu","feature_and_optimizer_device":str(dev),"physics_dt":physics_dt,"control_dt":control_dt,"policy_input_dim":POLICY_INPUT_DIM,"policy_hidden_dim":128,"optimizer":plan["optimizer"],"budget":plan["budget"],"loss":plan["loss"],"selected_checkpoint":None,"config_identity":config_identity,"history":history,"artifacts":{"plan":str(dest/"plan.json"),"rollouts":str(dest/"decoded-feedback-rollouts.npz")}}
    (dest/"metrics.json").write_text(json.dumps(report,indent=2)+"\n");return report


def train_coverage_decoded_feedback_distillation(run: str | Path, training_manifest: str | Path, output: str | Path, *, device: str = "cuda", physics_dt: float = 0.001, control_dt: float = 0.01, config_identity: dict[str, str] | None = None) -> dict[str, object]:
    """C3c-5: fit decoded feedback on 120 disjoint healthy trajectories."""
    updates, saved, weight, rate = 50, {1, 10, 25, 50}, 5.0, 1e-5
    torch.manual_seed(20260915); dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested CUDA C3c-5 coverage distillation but CUDA is unavailable")
    source = confined_path(run, must_exist=True); source_metrics = json.loads((source / "metrics.json").read_text()); checkpoint = source / "best-policy.pt"
    world_path = confined_path(str(source_metrics["world_bundle"]), must_exist=True)
    cases, manifest = load_healthy_case_manifest(training_manifest, expected_count=120)
    dest = confined_path(output); dest.mkdir(parents=True, exist_ok=True)
    plan = {"purpose":"C3c-5 predeclared decoded-feedback coverage correction","hypothesis":"Disjoint 120-case decoded-feedback supervision closes candidate/reference disagreement without losing original development successes.","initialization":{"path":str(checkpoint),"sha256":_file_sha256(checkpoint)},"training_manifest":manifest,"collection":{"new_transitions":20400,"cases":120,"ticks_per_case":170},"loss":{"decoded_label_weight":1.0,"same_trajectory_action_anchor_weight":weight},"optimizer":{"name":"Adam","state":"fresh","learning_rate":rate,"gradient_norm_cap":0.5},"budget":{"fitting_passes":updates,"selection_checkpoints":["initial","policy-update-000001","policy-update-000010","policy-update-000025","policy-update-000050"],"ceiling":{"new_transitions":20400,"fitting_passes":1500}},"selection_rule":"Evaluate every saved checkpoint on original development-64. Select only >=61/64 retaining original incumbent 60 and no hard violations. Only then renew examined-200 and evaluate frozen confirmation-200 once."}
    (dest / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    bundle = WorldBundle.load(world_path, device=dev); bundle.model.eval(); base = PolicyNetwork().to(dev); base.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True)); base.eval()
    features, labels, trace = _collect_support_matched_decoded_labels(bundle, base, device=dev, case_indices=tuple(range(120)), explicit_cases=cases, physics_dt=physics_dt, control_dt=control_dt)
    if features.shape != (20400, POLICY_INPUT_DIM): raise RuntimeError("C3c-5 must collect 120 complete trajectories")
    with torch.no_grad(): anchors = base.deterministic(features)
    torch.save({"features":features.detach().cpu(),"decoded_feedback_actions":labels.detach().cpu(),"base_actions":anchors.detach().cpu()}, dest / "packet-demonstrations.pt"); np.savez(dest / "coverage-rollouts.npz", **trace)
    policy = PolicyNetwork().to(dev); policy.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True)); opt = torch.optim.Adam(policy.parameters(), lr=rate); checks = dest / "checkpoints"; checks.mkdir(parents=True, exist_ok=True); torch.save(policy.state_dict(), dest / "initial-policy.pt"); torch.save(policy.state_dict(), checks / "initial-policy.pt")
    history=[]
    for step in range(1, updates + 1):
        decoded, anchor = F.mse_loss(policy.deterministic(features), labels), F.mse_loss(policy.deterministic(features), anchors); loss = decoded + weight * anchor; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
        with torch.no_grad(): history.append({"update":float(step),"decoded_loss":float(F.mse_loss(policy.deterministic(features),labels)),"anchor_loss":float(F.mse_loss(policy.deterministic(features),anchors))})
        if step in saved: torch.save(policy.state_dict(), checks / f"policy-update-{step:06d}.pt")
    torch.save(policy.state_dict(), dest / "final-policy.pt")
    report={"algorithm":"C3c-5 disjoint-manifest decoded-feedback coverage distillation","controller_kind":"recurrent-packet-policy","world_bundle":str(world_path),"initialization":plan["initialization"],"training_manifest":manifest,"device":str(dev),"simulation_device":"cpu","feature_and_optimizer_device":str(dev),"physics_dt":physics_dt,"control_dt":control_dt,"policy_input_dim":POLICY_INPUT_DIM,"policy_hidden_dim":128,"optimizer":plan["optimizer"],"budget":plan["budget"],"loss":plan["loss"],"selected_checkpoint":None,"config_identity":config_identity,"history":history,"artifacts":{"plan":str(dest / "plan.json"),"rollouts":str(dest / "coverage-rollouts.npz")}}
    (dest / "metrics.json").write_text(json.dumps(report, indent=2) + "\n"); return report
