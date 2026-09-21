"""Frozen world-model feature adapter for policy training and inference."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..baselines.kinematics import inverse_kinematics
from ..contracts import Command, Observation, RobotDescriptor, Task
from ..models.bundle import WorldBundle

CURRENT_PACKET_POLICY_INPUT_DIM = 51


@dataclass
class FeatureAdapter:
    """Advance the observer exactly once per real transition."""

    bundle: WorldBundle
    descriptor: RobotDescriptor
    device: torch.device

    def __post_init__(self) -> None:
        metadata = self.bundle.metadata["normalization"]
        self.mean = torch.as_tensor(metadata["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(metadata["std"], dtype=torch.float32, device=self.device)
        self.descriptor_tensor = torch.as_tensor(self.descriptor.as_array(), dtype=torch.float32, device=self.device)
        self.belief: Tensor | None = None
        self.elapsed_s = 0.0

    def _observation_features(self, observation: Observation) -> Tensor:
        values = torch.as_tensor(observation.values, dtype=torch.float32, device=self.device)
        available = torch.as_tensor(observation.available, dtype=torch.float32, device=self.device)
        fresh = torch.as_tensor(observation.fresh, dtype=torch.float32, device=self.device)
        age = torch.as_tensor(np.clip(observation.age_s / 0.1, 0.0, 10.0), dtype=torch.float32, device=self.device)
        return torch.cat(((values - self.mean) / self.std, available, fresh, age))

    def reset(self, initial_observation: Observation) -> None:
        self.elapsed_s = 0.0
        self.reset_belief(initial_observation)

    def reset_belief(self, observation: Observation) -> None:
        """Forget observer history while preserving public task time."""
        with torch.no_grad():
            self.belief = self.bundle.model.initial(self._observation_features(observation), self.descriptor_tensor)

    def observe(self, accepted_previous_command: Command | np.ndarray, observation: Observation, dt: float = 0.01) -> None:
        if self.belief is None:
            raise RuntimeError("feature adapter must be reset before observe")
        command = accepted_previous_command.values if isinstance(accepted_previous_command, Command) else Command.accepted(accepted_previous_command).values
        action = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prior = self.bundle.model.predict_prior(self.belief, action, self.descriptor_tensor, dt)
            self.belief = self.bundle.model.observe(prior, self._observation_features(observation), self.descriptor_tensor)
        self.elapsed_s += dt

    def policy_features(self, task: Task, previous_command: Command | np.ndarray) -> Tensor:
        if self.belief is None:
            raise RuntimeError("feature adapter must be reset before policy_features")
        command = previous_command.values if isinstance(previous_command, Command) else Command.accepted(previous_command).values
        time_remaining = max(0.0, task.deadline_s - self.elapsed_s)
        task_values = np.concatenate((task.goal_xz, [time_remaining]))
        goal_q = inverse_kinematics(task.goal_xz, self.descriptor.link_lengths, elbow=1)
        decoded_q, decoded_dq = self.bundle.model.decode(self.belief)
        decoded_state = torch.cat((decoded_q, decoded_dq), dim=-1)
        return torch.cat((self.belief, decoded_state, self.descriptor_tensor,
                          torch.as_tensor(task_values[:2], dtype=torch.float32, device=self.device),
                          torch.as_tensor(goal_q, dtype=torch.float32, device=self.device),
                          torch.as_tensor(task_values[2:], dtype=torch.float32, device=self.device),
                          torch.as_tensor(command, dtype=torch.float32, device=self.device)))


@dataclass
class CurrentPacketFeatureAdapter:
    """Build the fair feedforward policy input from the current public packet only."""

    bundle: WorldBundle
    descriptor: RobotDescriptor
    device: torch.device

    def __post_init__(self) -> None:
        normalization = self.bundle.metadata["normalization"]
        self.mean = torch.as_tensor(normalization["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(normalization["std"], dtype=torch.float32, device=self.device)
        self.descriptor_tensor = torch.as_tensor(self.descriptor.as_array(), dtype=torch.float32, device=self.device)

    def policy_features(self, observation: Observation, task: Task, elapsed_s: float,
                        previous_command: Command | np.ndarray) -> Tensor:
        """Return values, masks, freshness, ages, and the common public context."""

        command = previous_command.values if isinstance(previous_command, Command) else Command.accepted(previous_command).values
        values = torch.as_tensor(observation.values, dtype=torch.float32, device=self.device)
        available = torch.as_tensor(observation.available, dtype=torch.float32, device=self.device)
        fresh = torch.as_tensor(observation.fresh, dtype=torch.float32, device=self.device)
        age = torch.as_tensor(np.clip(observation.age_s / 0.1, 0.0, 10.0), dtype=torch.float32, device=self.device)
        goal_q = inverse_kinematics(task.goal_xz, self.descriptor.link_lengths, elbow=1)
        time_remaining = max(0.0, task.deadline_s - elapsed_s)
        result = torch.cat((
            (values - self.mean) / self.std, available, fresh, age, self.descriptor_tensor,
            torch.as_tensor(task.goal_xz, dtype=torch.float32, device=self.device),
            torch.as_tensor(goal_q, dtype=torch.float32, device=self.device),
            torch.as_tensor([time_remaining], dtype=torch.float32, device=self.device),
            torch.as_tensor(command, dtype=torch.float32, device=self.device),
        ))
        if result.shape != (CURRENT_PACKET_POLICY_INPUT_DIM,):
            raise RuntimeError("current-packet policy feature schema has the wrong width")
        return result
