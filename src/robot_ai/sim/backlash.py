"""Motor/output backlash transmission with memory and passive engagement."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacklashSettings:
    half_gap_rad: float = 0.01
    stiffness_nm_per_rad: float = 8.0
    damping_nms_per_rad: float = 0.08
    motor_inertia_kg_m2: float = 0.002
    smooth_width_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.half_gap_rad < 0 or self.stiffness_nm_per_rad < 0 or self.damping_nms_per_rad < 0:
            raise ValueError("backlash gap, stiffness, and damping must be nonnegative")
        if self.motor_inertia_kg_m2 <= 0:
            raise ValueError("motor inertia must be positive")


class BacklashTransmission:
    """One motor-side coordinate coupled to a measured output coordinate."""

    def __init__(self, settings: BacklashSettings | None = None) -> None:
        self.settings = settings or BacklashSettings()
        self.motor_q = 0.0
        self.motor_dq = 0.0

    def reset(self, output_q: float) -> None:
        self.motor_q = float(output_q)
        self.motor_dq = 0.0

    def _engagement(self, delta: float) -> float:
        excess = abs(delta) - self.settings.half_gap_rad
        if self.settings.smooth_width_rad > 0:
            width = self.settings.smooth_width_rad
            excess = width * np.log1p(np.exp(excess / width))
        return float(np.sign(delta) * max(0.0, excess))

    def coupling_torque(self, output_q: float, output_dq: float) -> float:
        delta = self.motor_q - float(output_q)
        engagement = self._engagement(delta)
        if engagement == 0.0:
            return 0.0
        relative_velocity = self.motor_dq - float(output_dq)
        return self.settings.stiffness_nm_per_rad * engagement + self.settings.damping_nms_per_rad * relative_velocity

    def step(self, motor_torque: float, output_q: float, output_dq: float, dt: float) -> float:
        """Advance the hidden motor and return equal/opposite output torque."""

        coupling = self.coupling_torque(output_q, output_dq)
        acceleration = (float(motor_torque) - coupling) / self.settings.motor_inertia_kg_m2
        self.motor_dq += dt * acceleration
        self.motor_q += dt * self.motor_dq
        return coupling

    def stored_energy(self, output_q: float) -> float:
        engagement = self._engagement(self.motor_q - float(output_q))
        return 0.5 * self.settings.stiffness_nm_per_rad * engagement**2 + 0.5 * self.settings.motor_inertia_kg_m2 * self.motor_dq**2

