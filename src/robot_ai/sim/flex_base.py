"""Reduced torsional link-flex and compliant-base dynamics for P8."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlexSettings:
    stiffness_nm_per_rad: float
    damping_nms_per_rad: float
    effective_inertia_kg_m2: float

    def __post_init__(self) -> None:
        if self.stiffness_nm_per_rad <= 0 or self.damping_nms_per_rad < 0 or self.effective_inertia_kg_m2 <= 0:
            raise ValueError("flex stiffness/inertia must be positive and damping nonnegative")


class TorsionalFlexMode:
    def __init__(self, settings: FlexSettings) -> None:
        self.settings = settings
        self.angle = 0.0
        self.rate = 0.0

    def reset(self) -> None:
        self.angle = 0.0
        self.rate = 0.0

    def step(self, applied_moment: float, dt: float) -> float:
        acceleration = (float(applied_moment) - self.settings.stiffness_nm_per_rad * self.angle -
                         self.settings.damping_nms_per_rad * self.rate) / self.settings.effective_inertia_kg_m2
        self.rate += dt * acceleration
        self.angle += dt * self.rate
        return self.angle

    @property
    def stored_energy(self) -> float:
        return 0.5 * self.settings.stiffness_nm_per_rad * self.angle**2 + 0.5 * self.settings.effective_inertia_kg_m2 * self.rate**2


@dataclass(frozen=True)
class BaseSettings:
    mass_kg: float
    inertia_kg_m2: float
    stiffness_xz_n_per_m: np.ndarray
    damping_xz_ns_per_m: np.ndarray
    pitch_stiffness_nm_per_rad: float
    pitch_damping_nms_per_rad: float

    def __post_init__(self) -> None:
        stiffness = np.asarray(self.stiffness_xz_n_per_m, dtype=np.float64)
        damping = np.asarray(self.damping_xz_ns_per_m, dtype=np.float64)
        if stiffness.shape != (2,) or damping.shape != (2,) or (stiffness <= 0).any() or (damping < 0).any():
            raise ValueError("base translation stiffness/damping must have two valid values")
        if self.mass_kg <= 0 or self.inertia_kg_m2 <= 0 or self.pitch_stiffness_nm_per_rad <= 0 or self.pitch_damping_nms_per_rad < 0:
            raise ValueError("base mass, inertia, and pitch stiffness must be positive")
        object.__setattr__(self, "stiffness_xz_n_per_m", stiffness)
        object.__setattr__(self, "damping_xz_ns_per_m", damping)


class CompliantBase:
    """Passive base whose reactions move the arm reference in world coordinates."""

    def __init__(self, settings: BaseSettings) -> None:
        self.settings = settings
        self.pose = np.zeros(3, dtype=np.float64)  # x, z, pitch relative to support equilibrium
        self.velocity = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.pose[:] = 0.0
        self.velocity[:] = 0.0

    def step(self, reaction_force_xz: np.ndarray, reaction_moment: float, dt: float) -> np.ndarray:
        force = np.asarray(reaction_force_xz, dtype=np.float64)
        acceleration = (force - self.settings.stiffness_xz_n_per_m * self.pose[:2] -
                        self.settings.damping_xz_ns_per_m * self.velocity[:2]) / self.settings.mass_kg
        pitch_acceleration = (float(reaction_moment) - self.settings.pitch_stiffness_nm_per_rad * self.pose[2] -
                              self.settings.pitch_damping_nms_per_rad * self.velocity[2]) / self.settings.inertia_kg_m2
        self.velocity[:2] += dt * acceleration
        self.velocity[2] += dt * pitch_acceleration
        self.pose += dt * self.velocity
        return self.pose.copy()

    @property
    def stored_energy(self) -> float:
        """Spring plus kinetic energy relative to the declared support equilibrium."""

        translation = 0.5 * float(np.sum(self.settings.stiffness_xz_n_per_m * self.pose[:2]**2))
        pitch = 0.5 * self.settings.pitch_stiffness_nm_per_rad * self.pose[2]**2
        kinetic = 0.5 * self.settings.mass_kg * float(np.dot(self.velocity[:2], self.velocity[:2]))
        kinetic += 0.5 * self.settings.inertia_kg_m2 * self.velocity[2]**2
        return translation + pitch + kinetic
