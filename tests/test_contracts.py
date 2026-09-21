"""P1 public/truth contract tests."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np
import pytest

from robot_ai.contracts import Command, ContractError, Observation, Task
from robot_ai.sim.native import default_descriptor


def test_descriptor_schema_order_and_size() -> None:
    descriptor = default_descriptor()
    values = descriptor.as_array()
    assert values.shape == (12,)
    np.testing.assert_allclose(values[:2], [0.30, 0.25])


def test_command_boundary_records_clipped_request() -> None:
    accepted = Command.accepted([2.0, -3.0])
    np.testing.assert_allclose(accepted.values, [1.0, -1.0])
    with pytest.raises(ContractError):
        Command(np.array([1.1, 0.0]))


def test_missing_observation_values_are_finite_features() -> None:
    observation = Observation(np.arange(8), [True, False, True, True, True, True, True, True],
                              np.ones(8, dtype=bool), np.zeros(8), 0.0)
    features = observation.feature_values()
    assert features.shape == (32,)
    assert np.isfinite(features).all()
    assert features[1] == 0.0


def test_task_rejects_non_world_frames() -> None:
    with pytest.raises(ContractError):
        Task(np.zeros(2), 1.0, frame="base")
