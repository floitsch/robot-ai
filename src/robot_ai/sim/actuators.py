"""Hidden motor response, latency, effectiveness, and resistance models."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..contracts import Command, RobotDescriptor


@dataclass(frozen=True)
class ActuatorSettings:
    """Per-episode actuator parameters; never part of public policy features."""

    effectiveness: np.ndarray
    time_constant_s: np.ndarray
    command_latency_s: np.ndarray
    viscous_friction: np.ndarray
    coulomb_friction: np.ndarray
    local_resistance: np.ndarray
    local_angle: np.ndarray
    local_width: np.ndarray

    def __post_init__(self) -> None:
        for name in ("effectiveness", "time_constant_s", "command_latency_s", "viscous_friction",
                     "coulomb_friction", "local_resistance", "local_angle", "local_width"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector of length two")
            if name == "effectiveness" and (value <= 0).any():
                raise ValueError("effectiveness must be positive")
            if (name.endswith("friction") or name in {"local_resistance", "local_width"}) and (value < 0).any():
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)

    @classmethod
    def healthy(cls) -> ActuatorSettings:
        zeros = np.zeros(2)
        return cls(np.ones(2), zeros, zeros, zeros, zeros, zeros, zeros, np.ones(2) * 0.1)


class ActuatorModel:
    """Causal actuator state update and generalized friction force."""

    def __init__(self, descriptor: RobotDescriptor, settings: ActuatorSettings, timestep: float) -> None:
        self.descriptor = descriptor
        self.settings = settings
        self.timestep = timestep
        self._commands: deque[np.ndarray] = deque()
        self._motor_torque = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self._commands.clear()
        self._motor_torque[:] = 0.0

    def update(self, command: Command | np.ndarray, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        accepted = command.values if isinstance(command, Command) else Command.accepted(command).values
        self._commands.append(np.asarray(accepted, dtype=np.float64).copy())
        delays = np.rint(self.settings.command_latency_s / self.timestep).astype(int)
        delayed = self._commands[0].copy()
        for joint in range(2):
            index = max(0, len(self._commands) - 1 - delays[joint])
            delayed[joint] = list(self._commands)[index][joint]
        if len(self._commands) > int(delays.max()) + 2:
            self._commands.popleft()
        target = delayed * self.descriptor.nominal_torque_scales * self.settings.effectiveness
        for joint, tau in enumerate(self.settings.time_constant_s):
            if tau > 0:
                alpha = 1.0 - np.exp(-self.timestep / tau)
                self._motor_torque[joint] += alpha * (target[joint] - self._motor_torque[joint])
            else:
                self._motor_torque[joint] = target[joint]
        velocity = np.asarray(dq, dtype=np.float64)
        resistance = -self.settings.viscous_friction * velocity
        resistance -= self.settings.coulomb_friction * np.tanh(velocity / 1e-3)
        local = self.settings.local_resistance * np.exp(-0.5 * ((np.asarray(q) - self.settings.local_angle) / self.settings.local_width) ** 2)
        resistance -= local * np.tanh(velocity / 1e-3)
        return self._motor_torque.copy(), resistance

    @property
    def measured_motor_torque(self) -> np.ndarray:
        return self._motor_torque.copy()
