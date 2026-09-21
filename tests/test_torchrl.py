"""P6 TorchRL loss/GAE integration fixtures."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import json
from copy import deepcopy
from pathlib import Path

import torch

from robot_ai.models.world import WorldModel
from robot_ai.train.policy import (
    DEVELOPMENT_TEACHER_FAILURES,
    _collect_rollout,
    _development_tasks,
    critic_warmup,
    hand_checked_gae,
    make_ppo_components,
    numeric_gae,
    ppo_update,
    task_terminal_reward,
)


def test_torchrl_ppo_components_and_gae_are_constructible() -> None:
    components = make_ppo_components()
    rollout = hand_checked_gae(components)
    assert "advantage" in rollout
    assert torch.isfinite(rollout["advantage"]).all()
    optimizer = torch.optim.Adam(list(components.policy.parameters()) + list(components.critic.parameters()), lr=3e-4)
    metrics = ppo_update(components, rollout, optimizer)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for key in ("action_scale_mean", "action_scale_p95", "sampled_action_saturation_fraction",
                "advantage_std", "rollout_reward_mean", "rollout_terminal_success_fraction",
                "rollout_terminal_hard_violation_fraction", "ratio_mean", "ratio_clip_fraction",
                "approximate_kl", "teacher_action_loss", "value_prediction_mean",
                "value_target_mean", "value_abs_error_mean"):
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_ppo_teacher_regularization_is_reported() -> None:
    components = make_ppo_components()
    rollout = hand_checked_gae(components)
    rollout["teacher_action"] = torch.full_like(rollout["action"], 0.25)
    optimizer = torch.optim.Adam(list(components.policy.parameters()) + list(components.critic.parameters()), lr=3e-4)
    metrics = ppo_update(components, rollout, optimizer, epochs=1, teacher_regularization_weight=0.5)
    assert metrics["teacher_regularization_weight"] == 0.5
    assert metrics["teacher_action_loss"] > 0.0


def test_ppo_policy_anchor_regularization_is_reported() -> None:
    components = make_ppo_components()
    rollout = hand_checked_gae(components)
    optimizer = torch.optim.Adam(list(components.policy.parameters()) + list(components.critic.parameters()), lr=3e-4)
    anchor = deepcopy(components.policy).eval()
    metrics = ppo_update(components, rollout, optimizer, epochs=1, anchor_policy=anchor,
                         anchor_regularization_weight=1.0)
    assert metrics["anchor_regularization_weight"] == 1.0
    assert metrics["anchor_action_loss"] >= 0.0


def test_critic_warmup_reduces_value_target_error() -> None:
    components = make_ppo_components()
    rollout = hand_checked_gae(components)
    optimizer = torch.optim.Adam(list(components.policy.parameters()) + list(components.critic.parameters()), lr=1e-2)
    features = rollout["features"].reshape(-1, rollout["features"].shape[-1])
    target = rollout["value_target"].reshape(-1)
    before = (components.critic(features) - target).abs().mean()
    critic_warmup(components, rollout, optimizer, epochs=20)
    after = (components.critic(features) - target).abs().mean()
    assert after < before


def test_ppo_kl_stop_records_the_actual_minibatch_count() -> None:
    components = make_ppo_components()
    rollout = hand_checked_gae(components)
    optimizer = torch.optim.Adam(list(components.policy.parameters()) + list(components.critic.parameters()), lr=3e-4)
    metrics = ppo_update(components, rollout, optimizer, epochs=3, max_approximate_kl=1e-12)
    assert metrics["ppo_updates_applied"] == 1.0
    assert metrics["ppo_early_kl_stop"] == 1.0


def test_stored_rollout_log_prob_matches_the_policy_distribution() -> None:
    components = make_ppo_components()
    with torch.no_grad():
        rollout, _, _ = _collect_rollout(WorldModel().eval(), components, worlds=2, steps=3,
                                          device=torch.device("cpu"), mean=torch.zeros(8), std=torch.ones(8),
                                          physics_dt=0.001, control_dt=0.01)
    features = rollout["features"].reshape(-1, rollout["features"].shape[-1])
    actions = rollout["action"].reshape(-1, 2)
    actual = components.policy.distribution(features).log_prob(actions)
    expected = rollout["sample_log_prob"].reshape(-1)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actions, components.policy.deterministic(features))


def test_rollout_boundary_preserves_task_until_tick_170() -> None:
    model = WorldModel().eval()
    components = make_ppo_components()
    mean, std = torch.zeros(8), torch.ones(8)
    with torch.no_grad():
        first, _, state = _collect_rollout(model, components, worlds=1, steps=128, device=torch.device("cpu"),
                                           mean=mean, std=std, physics_dt=0.001, control_dt=0.01,
                                           teacher_forcing=True)
        second, _, state = _collect_rollout(model, components, worlds=1, steps=42, device=torch.device("cpu"),
                                            mean=mean, std=std, physics_dt=0.001, control_dt=0.01, state=state,
                                            teacher_forcing=True)
    assert not first["next", "terminated"].any()
    assert second["next", "terminated"][0, -1]
    assert second["next", "done"][0, -1]
    assert state.task_tick is not None
    assert torch.equal(state.task_tick, torch.zeros(1, dtype=torch.int64))


def test_numeric_gae_distinguishes_terminal_truncation_and_buffer_boundary() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.0, 5.0, 5.0])
    final = torch.tensor(2.0)
    none = torch.zeros(3, dtype=torch.bool)
    boundary = numeric_gae(rewards, values, final, none, none, lmbda=1.0)
    terminal = numeric_gae(rewards, values, final, torch.tensor([False, True, False]), none, lmbda=1.0)
    truncated = numeric_gae(rewards, values, final, none, torch.tensor([False, True, False]), lmbda=1.0)
    torch.testing.assert_close(boundary, torch.tensor([5.0, -1.0, -2.0]))
    torch.testing.assert_close(terminal, torch.tensor([2.0, -4.0, -2.0]))
    torch.testing.assert_close(truncated, torch.tensor([7.0, 1.0, -2.0]))


def test_terminal_reward_keeps_success_and_hard_failure_distinct() -> None:
    reward = task_terminal_reward(torch.tensor([True, False, True]), torch.tensor([False, False, True]))
    torch.testing.assert_close(reward, torch.tensor([10.0, -1.0, -10.0]))


def test_teacher_forcing_records_the_feasible_teacher_actions() -> None:
    model = WorldModel().eval()
    components = make_ppo_components()
    with torch.no_grad():
        rollout, _, _ = _collect_rollout(model, components, worlds=1, steps=4, device=torch.device("cpu"),
                                          mean=torch.zeros(8), std=torch.ones(8), physics_dt=0.001,
                                          control_dt=0.01, teacher_forcing=True)
    torch.testing.assert_close(rollout["action"], rollout["teacher_action"])


def test_teacher_curriculum_excludes_only_documented_reference_failures() -> None:
    torch.manual_seed(3)
    starts, goals = _development_tasks(worlds=256, device=torch.device("cpu"))
    manifest = json.loads(Path("configs/healthy-development-v1.json").read_text(encoding="utf-8"))
    for index in DEVELOPMENT_TEACHER_FAILURES:
        failed_start = starts.new_tensor(manifest["cases"][index]["start_q"])
        failed_goal = goals.new_tensor(manifest["cases"][index]["goal_q"])
        assert not ((starts == failed_start).all(dim=-1) & (goals == failed_goal).all(dim=-1)).any()
    assert starts.shape == goals.shape == (256, 2)
    assert torch.isfinite(starts).all() and torch.isfinite(goals).all()
