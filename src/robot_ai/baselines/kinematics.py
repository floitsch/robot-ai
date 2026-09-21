"""Analytic two-link inverse kinematics and smooth joint trajectories."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def inverse_kinematics(goal_xz: np.ndarray, lengths: np.ndarray, *, elbow: int) -> np.ndarray:
    """Solve both planar branches using the project's downward-zero convention."""

    goal = np.asarray(goal_xz, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if goal.shape != (2,) or lengths.shape != (2,):
        raise ValueError("goal and lengths must have shape (2,)")
    if elbow not in (-1, 1):
        raise ValueError("elbow must be -1 or 1")
    x, downward = goal[0], -goal[1]
    radius = float(np.hypot(x, downward))
    cosine = (radius * radius - lengths[0] ** 2 - lengths[1] ** 2) / (2 * lengths[0] * lengths[1])
    if cosine < -1.0 or cosine > 1.0:
        raise ValueError("goal is outside the two-link workspace")
    q2 = float(elbow * np.arccos(np.clip(cosine, -1.0, 1.0)))
    absolute_angle = float(np.arctan2(x, downward))
    offset = float(np.arctan2(lengths[1] * np.sin(q2), lengths[0] + lengths[1] * np.cos(q2)))
    q1 = absolute_angle - offset
    return np.array([q1, q2], dtype=np.float64)


@dataclass(frozen=True)
class QuinticTrajectory:
    """Zero-velocity/zero-acceleration trajectory between two joint states."""

    start: np.ndarray
    goal: np.ndarray
    duration_s: float

    def __post_init__(self) -> None:
        if self.start.shape != (2,) or self.goal.shape != (2,):
            raise ValueError("trajectory endpoints must have shape (2,)")
        if self.duration_s <= 0:
            raise ValueError("trajectory duration must be positive")

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ratio = float(np.clip(time_s / self.duration_s, 0.0, 1.0))
        delta = self.goal - self.start
        position_blend = 10 * ratio**3 - 15 * ratio**4 + 6 * ratio**5
        velocity_blend = (30 * ratio**2 - 60 * ratio**3 + 30 * ratio**4) / self.duration_s
        acceleration_blend = (60 * ratio - 180 * ratio**2 + 120 * ratio**3) / self.duration_s**2
        return (self.start + position_blend * delta, velocity_blend * delta, acceleration_blend * delta)

