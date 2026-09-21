"""P1 analytic and backend fixture tests."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np
import torch

from robot_ai.contracts import Task
from robot_ai.sim.mechanics import (
    coriolis_torque,
    forward_kinematics,
    gravity_torque,
    jacobian,
    potential_energy,
)
from robot_ai.sim.native import NativeArm, default_descriptor
from robot_ai.sim.scorer import joint_limit_summary, joint_limit_violations, score_episode
from robot_ai.sim.warp_backend import WarpArmBatch


def test_forward_kinematics_reference_signs() -> None:
    descriptor = default_descriptor()
    np.testing.assert_allclose(forward_kinematics(np.zeros(2), descriptor.link_lengths), [0.0, -0.55])
    np.testing.assert_allclose(forward_kinematics([np.pi / 2, 0.0], descriptor.link_lengths), [0.55, 0.0], atol=1e-12)


def test_jacobian_matches_finite_difference() -> None:
    descriptor = default_descriptor()
    q = np.array([0.3, -0.7])
    eps = 1e-7
    numerical = np.column_stack(
        ((forward_kinematics(q + [eps, 0], descriptor.link_lengths) - forward_kinematics(q - [eps, 0], descriptor.link_lengths)) / (2 * eps),
         (forward_kinematics(q + [0, eps], descriptor.link_lengths) - forward_kinematics(q - [0, eps], descriptor.link_lengths)) / (2 * eps))
    )
    np.testing.assert_allclose(jacobian(q, descriptor.link_lengths), numerical, rtol=1e-6, atol=1e-8)


def test_gravity_is_zero_when_links_hang_down() -> None:
    np.testing.assert_allclose(gravity_torque(np.zeros(2), default_descriptor()), [0.0, 0.0], atol=1e-12)


def test_gravity_matches_negative_potential_gradient() -> None:
    descriptor = default_descriptor()
    q = np.array([0.3, -0.4])
    eps = 1e-7
    numerical = np.array([
        (potential_energy(q + np.eye(2)[index] * eps, descriptor) -
         potential_energy(q - np.eye(2)[index] * eps, descriptor)) / (2 * eps)
        for index in range(2)
    ])
    np.testing.assert_allclose(gravity_torque(q, descriptor), -numerical, rtol=1e-7, atol=1e-9)


def test_coriolis_torque_is_zero_at_rest_and_has_expected_shape() -> None:
    descriptor = default_descriptor()
    q = np.array([[0.3, -0.4], [-0.2, 0.7]])
    np.testing.assert_allclose(coriolis_torque(q, np.zeros_like(q), descriptor), 0.0)
    assert coriolis_torque(q, np.ones_like(q), descriptor).shape == (2, 2)


def test_native_endpoint_matches_analytic_fixture() -> None:
    descriptor = default_descriptor()
    arm = NativeArm(descriptor)
    q = np.array([0.4, -0.2])
    arm.reset(q=q)
    np.testing.assert_allclose(arm.truth().endpoint_xz, forward_kinematics(q, descriptor.link_lengths), atol=1e-9)


def test_nominal_native_and_warp_public_packets_have_the_same_torque_contract() -> None:
    descriptor = default_descriptor()
    initial = np.array([0.4, -0.2])
    command = np.array([0.3, -0.15])
    native = NativeArm(descriptor)
    native.reset(initial)
    native.step(command)
    warp = WarpArmBatch(descriptor, 1, device="cpu")
    warp.reset(q=initial.reshape(1, 2).astype(np.float32))
    warp.step(command.reshape(1, 2).astype(np.float32))
    q, dq = warp.snapshot()
    expected = np.concatenate((q[0], dq[0], np.zeros(2), forward_kinematics(q[0], descriptor.link_lengths)))
    np.testing.assert_allclose(native.observation().values, expected, atol=1e-6)


def test_warp_reset_mask_isolated_on_cpu() -> None:
    descriptor = default_descriptor()
    arm = WarpArmBatch(descriptor, 2, device="cpu")
    arm.reset(q=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    arm.reset(mask=np.array([1, 0], dtype=np.int32), q=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32))
    q, _ = arm.snapshot()
    np.testing.assert_allclose(q, [[0.5, 0.6], [0.3, 0.4]], atol=1e-6)


def test_warp_torch_handoff_steps_without_numpy_state_copy() -> None:
    arm = WarpArmBatch(default_descriptor(), 2, device="cpu")
    arm.reset()
    arm.step_torch(torch.tensor([[0.2, -0.3], [0.1, 0.4]], dtype=torch.float32))
    q, dq = arm.torch_state()
    assert isinstance(q, torch.Tensor)
    assert isinstance(dq, torch.Tensor)
    assert q.device.type == "cpu"
    assert torch.isfinite(q).all()
    assert torch.isfinite(dq).all()


def test_scorer_checks_the_entire_hold_interval() -> None:
    task = Task(np.zeros(2), deadline_s=1.0, hold_duration_s=0.2)
    times = np.arange(0.0, 1.21, 0.05)
    endpoint = np.zeros((len(times), 2))
    endpoint[(times > 1.0) & (times <= 1.1), 0] = 0.02
    score = score_episode(times, endpoint, np.zeros_like(endpoint), task)
    assert not score.success
    assert score.failure_reason == "deadline_or_hold_tolerance"


def test_scorer_rejects_trace_that_ends_at_deadline() -> None:
    task = Task(np.zeros(2), deadline_s=1.0, hold_duration_s=0.2)
    times = np.arange(0.0, 1.01, 0.05)
    endpoint = np.zeros((len(times), 2))
    score = score_episode(times, endpoint, np.zeros_like(endpoint), task, max_sample_interval_s=0.05)
    assert not score.success
    assert score.failure_reason == "insufficient_physical_trace"


def test_scorer_rejects_missing_physics_sample_in_hold() -> None:
    task = Task(np.zeros(2), deadline_s=1.0, hold_duration_s=0.2)
    times = np.arange(0.0, 1.21, 0.05)
    times = times[~np.isclose(times, 1.1)]
    endpoint = np.zeros((len(times), 2))
    score = score_episode(times, endpoint, np.zeros_like(endpoint), task, max_sample_interval_s=0.05)
    assert not score.success
    assert score.failure_reason == "insufficient_physical_trace"


def test_scorer_reports_early_settling_from_reset() -> None:
    task = Task(np.zeros(2), deadline_s=1.0, hold_duration_s=0.2)
    times = np.arange(0.0, 1.21, 0.05)
    endpoint = np.zeros((len(times), 2))
    score = score_episode(times, endpoint, np.zeros_like(endpoint), task, max_sample_interval_s=0.05)
    assert score.success
    assert score.settling_time_s == 0.0


def test_scorer_rejects_physics_violation_before_deadline() -> None:
    task = Task(np.zeros(2), deadline_s=1.0)
    times = np.array([0.0, 0.5, 1.0, 1.2])
    endpoint = np.zeros((4, 2))
    score = score_episode(times, endpoint, np.zeros_like(endpoint), task, hard_violation=np.array([False, True, False, False]))
    assert not score.success
    assert score.hard_violation
    assert score.failure_reason == "hard_operating_violation"


def test_joint_limit_contact_between_control_ticks_is_a_hard_violation() -> None:
    times = np.array([0.0, 0.001, 0.002])
    q = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.2]])
    violations = joint_limit_violations(q, np.array([[-2.0, 2.0], [-1.0, 1.0]]))

    assert violations.tolist() == [False, True, False]
    summary = joint_limit_summary(times, q, np.array([[-2.0, 2.0], [-1.0, 1.0]]))
    assert summary["joint_limit_violation_sample_count"] == 1
    assert summary["first_joint_limit_violation_s"] == 0.001
    assert summary["joint_limit_violation_joints"] == [1]
    task = Task(np.zeros(2), deadline_s=0.001, hold_duration_s=0.001)
    score = score_episode(times, np.zeros((3, 2)), np.zeros((3, 2)), task,
                          hard_violation=violations, max_sample_interval_s=0.001)
    assert not score.success
    assert score.failure_reason == "hard_operating_violation"
