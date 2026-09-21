"""P2 baseline and offline replay contract tests."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import json
from pathlib import Path

import numpy as np
import pytest

from robot_ai.baselines.controller import FeedbackController
from robot_ai.baselines.kinematics import QuinticTrajectory, inverse_kinematics
from robot_ai.config import RuntimeConfig
from robot_ai.contracts import Observation, Task
from robot_ai.evaluate.baseline import healthy_cases, load_healthy_case_manifest, run_baseline_suite
from robot_ai.sim.mechanics import forward_kinematics
from robot_ai.sim.native import default_descriptor


def test_both_ik_branches_reconstruct_the_goal() -> None:
    descriptor = default_descriptor()
    q = np.array([0.35, 0.9])
    goal = forward_kinematics(q, descriptor.link_lengths)
    for elbow in (-1, 1):
        solved = inverse_kinematics(goal, descriptor.link_lengths, elbow=elbow)
        np.testing.assert_allclose(forward_kinematics(solved, descriptor.link_lengths), goal, atol=1e-10)


def test_healthy_development_cases_are_frozen_at_sixty_four() -> None:
    cases = healthy_cases()
    assert len(cases) == 64
    np.testing.assert_allclose(cases[0][0], [0.2611725, 1.04030175])
    with pytest.raises(ValueError, match="exactly 64"):
        healthy_cases(65)


def test_explicit_healthy_case_manifest_retains_distinct_named_cases(tmp_path) -> None:
    manifest = tmp_path / "validation.json"
    manifest.write_text(json.dumps({"schema_version": 1, "seed": 7, "cases": [
        {"id": "validation-000", "start_q": [0.1, 0.5], "goal_q": [0.2, 0.6]},
        {"id": "validation-001", "start_q": [-0.1, 0.7], "goal_q": [0.3, 0.8]},
    ]}))
    cases, identity = load_healthy_case_manifest(manifest, expected_count=2)
    assert len(cases) == identity["case_count"] == 2
    assert not np.array_equal(cases[0][0], cases[1][0])


def test_packet_validation_manifest_is_distinct_from_frozen_development_cases() -> None:
    validation, identity = load_healthy_case_manifest(
        "configs/healthy-packet-validation-v1.json", expected_count=200, expected_seed=20260914
    )
    development = healthy_cases(64)
    pairs = lambda cases: {(tuple(start), tuple(goal)) for start, goal in cases}
    assert identity["case_count"] == 200
    assert not pairs(development).intersection(pairs(validation))


def test_quintic_has_zero_endpoint_velocity_and_acceleration() -> None:
    trajectory = QuinticTrajectory(np.array([0.0, 0.2]), np.array([0.5, -0.1]), 1.0)
    for time_s in (0.0, 1.0):
        _, velocity, acceleration = trajectory.sample(time_s)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-12)


def test_computed_torque_adds_the_nominal_trajectory_acceleration() -> None:
    descriptor = default_descriptor()
    task = Task(forward_kinematics(np.array([0.5, 0.8]), descriptor.link_lengths), deadline_s=1.5)
    observation = Observation(np.zeros(8), np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), 0.0)
    ordinary = FeedbackController(descriptor, np.zeros(2), task)
    coupled = FeedbackController(descriptor, np.zeros(2), task, computed_torque=True)
    assert not np.allclose(ordinary.act(0.2, observation), coupled.act(0.2, observation))


def test_baseline_holds_100_hz_commands_through_physics_substeps(tmp_path: Path) -> None:
    destination = str(tmp_path)
    config = RuntimeConfig(device="cpu", backend="native", physics_dt=0.001, control_dt=0.01)
    report = run_baseline_suite(destination, config=config, count=1)
    trajectory = np.load(tmp_path / "demo.npz", allow_pickle=False)
    assert report["command_rate_hz"] == 100.0
    assert report["substeps_per_command"] == 10
    assert trajectory["accepted_action"].shape == (170, 2)
    assert trajectory["time_s"].shape == (1701,)
    assert trajectory["joint_limit_violation"].shape == (1701,)
    assert np.isclose(trajectory["time_s"][-1], 1.7)
    assert report["modeled_hard_limits"] == ["joint_position"]
    assert report["scores"][0]["joint_limit_violation_sample_count"] == 0
    assert report["success_fraction"] == 1.0
