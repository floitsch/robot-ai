"""P9 primary-before-secondary metric fixtures."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from robot_ai.evaluate.metrics import command_slew, compare_smoothness, trajectory_jerk


def test_slew_and_jerk_use_declared_dt() -> None:
    actions = np.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0]])
    assert command_slew(actions, 0.1) > 0.0
    assert trajectory_jerk(np.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0]]), 0.1) > 0.0


def test_smoothness_cannot_pass_with_primary_regression() -> None:
    comparison = compare_smoothness(0.95, 0.93, np.ones(10), np.ones(10) * 0.5)
    assert comparison.slew_reduction_fraction == 0.5
    assert not comparison.accepted


def test_smoothness_cannot_pass_when_primary_task_is_unmet() -> None:
    comparison = compare_smoothness(0.0, 0.0, np.ones(10), np.ones(10) * 0.5)
    assert comparison.slew_reduction_fraction == 0.5
    assert not comparison.accepted
