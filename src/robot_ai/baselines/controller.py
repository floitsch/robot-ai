"""Ordinary public-observation feedback controller for the first arm."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import numpy as np

from ..contracts import Observation, RobotDescriptor, Task
from ..sim.mechanics import coriolis_torque, gravity_torque, mass_matrix
from .kinematics import QuinticTrajectory, inverse_kinematics


class FeedbackController:
    """PD plus nominal gravity compensation using only public observation fields."""

    def __init__(self, descriptor: RobotDescriptor, start_q: np.ndarray, task: Task, *, elbow: int = 1,
                 kp: np.ndarray | None = None, kd: np.ndarray | None = None,
                 computed_torque: bool = False, hold_goal: bool = False,
                 arrival_duration_s: float | None = None) -> None:
        self.descriptor = descriptor
        self.task = task
        # These development gains are selected for the public 100 Hz
        # zero-order-hold command interface. The older 1 kHz gains became
        # unstable when each command was held through ten physics steps.
        self.kp = np.asarray(kp if kp is not None else [8.0, 6.0], dtype=np.float64)
        self.kd = np.asarray(kd if kd is not None else [0.7, 0.56], dtype=np.float64)
        self.computed_torque = computed_torque
        self.hold_goal = hold_goal
        self.start_q = np.asarray(start_q, dtype=np.float64).copy()
        self.goal_q = inverse_kinematics(task.goal_xz, descriptor.link_lengths, elbow=elbow)
        # Complete the nominal motion early enough to verify the declared hold.
        duration = task.deadline_s * 0.55 if arrival_duration_s is None else float(arrival_duration_s)
        if not 0.0 < duration <= task.total_duration_s:
            raise ValueError("arrival_duration_s must be within the task duration")
        self.trajectory = QuinticTrajectory(self.start_q, self.goal_q, duration)

    def act(self, time_s: float, observation: Observation) -> np.ndarray:
        q = observation.values[:2]
        dq = observation.values[2:4]
        if self.hold_goal:
            target_q, target_dq, target_ddq = self.goal_q, np.zeros(2), np.zeros(2)
        else:
            target_q, target_dq, target_ddq = self.trajectory.sample(time_s)
        torque = self.kp * (target_q - q) + self.kd * (target_dq - dq) - gravity_torque(q, self.descriptor)
        if self.computed_torque:
            torque += mass_matrix(q, self.descriptor) @ target_ddq
            torque += coriolis_torque(q, dq, self.descriptor) + 0.02 * dq
        return np.clip(torque / self.descriptor.nominal_torque_scales, -1.0, 1.0)
