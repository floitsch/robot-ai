"""R1b horizon and baseline contracts for world-model evaluation."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from pathlib import Path

import numpy as np
import pytest
import torch

from robot_ai.data.storage import DatasetWriter, load_manifest
from robot_ai.evaluate.world import (
    _constant_position_prediction,
    _constant_velocity_prediction,
    _control_dt,
    _prediction_gate,
    _state_errors,
    evaluate_world,
)
from robot_ai.models.bundle import WorldBundle
from robot_ai.models.world import WorldModel
from robot_ai.sim.native import default_descriptor
from robot_ai.train.world import computed_torque_hold_action


def _episode(length: int) -> dict[str, np.ndarray]:
    return {
        "observation_values": np.zeros((length + 1, 8)), "available": np.ones((length + 1, 8), dtype=bool),
        "fresh": np.ones((length + 1, 8), dtype=bool), "age_s": np.zeros((length + 1, 8)),
        "time_s": np.arange(length + 1) * 0.01, "accepted_action": np.zeros((length, 2)),
        "true_q": np.zeros((length + 1, 2)), "true_dq": np.zeros((length + 1, 2)),
        "true_endpoint": np.zeros((length + 1, 2)), "true_endpoint_velocity": np.zeros((length + 1, 2)),
        "terminated": np.zeros(length, dtype=bool), "truncated": np.zeros(length, dtype=bool),
        "substep_summary": np.zeros((length, 3)),
    }


def test_constant_baselines_use_the_declared_horizon() -> None:
    q = torch.tensor([1.0, -2.0])
    dq = torch.tensor([3.0, -4.0])
    position_q, position_dq = _constant_position_prediction(q)
    velocity_q, velocity_dq = _constant_velocity_prediction(q, dq, horizon=5, dt=0.1)
    torch.testing.assert_close(position_q, q)
    torch.testing.assert_close(position_dq, torch.zeros_like(q))
    torch.testing.assert_close(velocity_q, torch.tensor([2.5, -4.0]))
    torch.testing.assert_close(velocity_dq, dq)


def test_normalized_state_error_has_common_q_and_dq_scales() -> None:
    errors = _state_errors(torch.tensor([np.pi, 0.0]), torch.tensor([5.0, 0.0]),
                           torch.zeros(2), torch.zeros(2))
    assert float(errors["q_error_mean_rad"]) == pytest.approx(np.pi)
    assert float(errors["dq_error_mean_rad_s"]) == pytest.approx(5.0)
    assert float(errors["normalized_state_error"]) == pytest.approx(1.0)


def test_control_bridge_action_is_identical_for_identical_state() -> None:
    descriptor = torch.as_tensor(default_descriptor().as_array(), dtype=torch.float32)
    q = torch.tensor([0.2, 0.8])
    dq = torch.tensor([0.1, -0.2])
    goal = torch.tensor([0.4, 1.0])
    torch.testing.assert_close(computed_torque_hold_action(q, dq, goal, descriptor),
                               computed_torque_hold_action(q, dq, goal, descriptor))


def test_control_dt_rejects_nonuniform_timestamps() -> None:
    with pytest.raises(ValueError, match="fixed control interval"):
        _control_dt(np.array([0.0, 0.01, 0.03]))


def test_p5_gate_rejects_a_material_one_step_regression() -> None:
    def state(error: float) -> dict[str, float | int | None]:
        return {"anchor_count": 1, "normalized_state_error": error}

    horizons = {
        "1": {"learned_open_loop": state(0.12), "oracle_constant_velocity": state(0.10)},
        "20": {"learned_open_loop": state(0.70), "oracle_constant_velocity": state(1.0)},
    }
    gate = _prediction_gate(horizons)
    assert gate["twenty_step_improvement_fraction"] == pytest.approx(0.30)
    assert gate["one_step_regression_fraction"] == pytest.approx(0.20)
    assert not gate["accepted"]


def test_evaluator_includes_the_last_valid_horizon_anchor(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    writer = DatasetWriter(dataset, seed=3)
    writer.add_episode(_episode(20), scenario_id="episode-a", robot_id="r", scenario={}, policy_source="test")
    writer.write()
    split = load_manifest(dataset)["episodes"][0]["split"]
    normalization = {"source_split": "test", "mean": [0.0] * 8, "std": [1.0] * 8}
    model = WorldModel()
    bundle = WorldBundle(model, {
        "schema_version": 1,
        "model": {"belief_dim": 128, "observation_dim": 32, "descriptor_dim": 12,
                  "endpoint_residual": False, "physics_prior": False},
        "normalization": normalization,
    }).save(tmp_path / "bundle")
    trail = tmp_path / "trail.npz"
    report = evaluate_world(bundle, dataset, tmp_path / "report.json", split=split, device="cpu",
                            trail_output=trail)
    assert report["horizons"]["1"]["learned_open_loop"]["anchor_count"] == 20
    assert report["horizons"]["5"]["learned_open_loop"]["anchor_count"] == 16
    assert report["horizons"]["20"]["learned_open_loop"]["anchor_count"] == 1
    assert report["endpoint_trail"]["anchor_count"] == 1
    assert not report["control_ready_bridge"]["accepted"]
    assert trail.exists()
    assert trail.with_suffix(".html").exists()
    exported = np.load(trail, allow_pickle=False)
    assert exported["actual_endpoint_xz"].shape == (1, 2)
