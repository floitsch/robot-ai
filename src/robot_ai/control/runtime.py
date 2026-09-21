"""Canonical frozen-bundle inference runtime."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..contracts import Command, Observation, RobotDescriptor, Task
from ..models.bundle import WorldBundle
from .features import CurrentPacketFeatureAdapter, FeatureAdapter
from .policy import LEARNED_STATE_DIM, PolicyNetwork


@dataclass
class Runtime:
    bundle: WorldBundle
    descriptor: RobotDescriptor
    policy: PolicyNetwork
    device: torch.device
    memoryless: bool = False

    def __post_init__(self) -> None:
        self.features = FeatureAdapter(self.bundle, self.descriptor, self.device)
        self.previous_command = Command.accepted([0.0, 0.0])

    def reset(self, descriptor: RobotDescriptor, initial_observation: Observation, task: Task) -> None:
        if descriptor.as_array().shape != self.descriptor.as_array().shape or not np.allclose(descriptor.as_array(), self.descriptor.as_array()):
            raise ValueError("runtime descriptor does not match the frozen world bundle")
        self.features.reset(initial_observation)
        self.previous_command = Command.accepted([0.0, 0.0])
        self.task = task

    def observe(self, accepted_previous_command: Command, observation: Observation) -> None:
        self.features.observe(accepted_previous_command, observation)
        self.previous_command = accepted_previous_command

    def reset_memory(self, observation: Observation) -> None:
        """Ablate only recurrent observer state, retaining task time and command context."""
        self.features.reset_belief(observation)

    def act(self, task: Task | None = None) -> Command:
        selected_task = task or self.task
        features = self.features.policy_features(selected_task, self.previous_command)
        if self.memoryless:
            features = features.clone()
            features[..., :LEARNED_STATE_DIM] = 0.0
        with torch.no_grad():
            action = self.policy.deterministic(features)
        return Command.accepted(action.detach().cpu().numpy())

    def predict(self, command_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.features.belief is None:
            raise RuntimeError("runtime must be reset before predict")
        commands = np.asarray(command_sequence, dtype=np.float32)
        if commands.ndim != 2 or commands.shape[1] != 2:
            raise ValueError("command_sequence must have shape [time, 2]")
        belief = self.features.belief.detach().clone()
        descriptor = self.features.descriptor_tensor
        predicted_q: list[np.ndarray] = []
        predicted_dq: list[np.ndarray] = []
        with torch.no_grad():
            for command in commands:
                action = torch.as_tensor(command, dtype=torch.float32, device=self.device)
                belief = self.bundle.model.predict_prior(belief, action, descriptor, 0.01)
                q, dq = self.bundle.model.decode(belief)
                predicted_q.append(q.cpu().numpy())
                predicted_dq.append(dq.cpu().numpy())
        return np.stack(predicted_q), np.stack(predicted_dq)


@dataclass
class CurrentPacketRuntime:
    """Feedforward policy runtime: no belief, history, or simulator truth input."""

    bundle: WorldBundle
    descriptor: RobotDescriptor
    policy: PolicyNetwork
    device: torch.device

    def __post_init__(self) -> None:
        self.features = CurrentPacketFeatureAdapter(self.bundle, self.descriptor, self.device)
        self.previous_command = Command.accepted([0.0, 0.0])
        self.elapsed_s = 0.0

    def reset(self, descriptor: RobotDescriptor, initial_observation: Observation, task: Task) -> None:
        if not np.allclose(descriptor.as_array(), self.descriptor.as_array()):
            raise ValueError("runtime descriptor does not match the frozen world bundle")
        self.observation, self.task = initial_observation, task
        self.previous_command = Command.accepted([0.0, 0.0])
        self.elapsed_s = 0.0

    def observe(self, accepted_previous_command: Command, observation: Observation) -> None:
        self.previous_command, self.observation = accepted_previous_command, observation
        self.elapsed_s += 0.01

    def act(self, task: Task | None = None) -> Command:
        with torch.no_grad():
            action = self.policy.deterministic(self.features.policy_features(
                self.observation, task or self.task, self.elapsed_s, self.previous_command
            ))
        return Command.accepted(action.detach().cpu().numpy())
