"""Public and privileged records shared by simulation, training, and evaluation."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


class ContractError(ValueError):
    """Raised when a public contract record has an invalid shape or value."""


def _vector(value: object, size: int, name: str, *, finite: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ContractError(f"{name} must have shape ({size},), got {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ContractError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class RobotDescriptor:
    """Known nominal robot information supplied to an inference bundle."""

    SCHEMA_VERSION: ClassVar[int] = 1
    link_lengths: np.ndarray
    nominal_masses: np.ndarray
    nominal_inertias: np.ndarray
    nominal_torque_scales: np.ndarray
    joint_limits: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "link_lengths",
            "nominal_masses",
            "nominal_inertias",
            "nominal_torque_scales",
        ):
            values = _vector(getattr(self, field_name), 2, field_name)
            if (values <= 0).any():
                raise ContractError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, values)
        limits = np.asarray(self.joint_limits, dtype=np.float64)
        if limits.shape != (2, 2) or not np.isfinite(limits).all():
            raise ContractError("joint_limits must have shape (2, 2) with finite values")
        if not (limits[:, 0] < limits[:, 1]).all():
            raise ContractError("each joint lower limit must be below its upper limit")
        object.__setattr__(self, "joint_limits", limits)

    def as_array(self) -> np.ndarray:
        """Return descriptor schema v1 in the documented 12-value order."""

        return np.concatenate(
            (
                self.link_lengths,
                self.nominal_masses,
                self.nominal_inertias,
                self.nominal_torque_scales,
                self.joint_limits.reshape(-1),
            )
        )


@dataclass(frozen=True)
class OperatingEnvelope:
    """Known v2 support and flex ratings supplied to later policy bundles."""

    max_base_force_xz: np.ndarray
    max_support_moment: float
    max_base_deflection_xz: np.ndarray
    max_base_pitch: float
    max_link_flex_angles: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_base_force_xz", _vector(self.max_base_force_xz, 2, "max_base_force_xz"))
        object.__setattr__(self, "max_base_deflection_xz", _vector(self.max_base_deflection_xz, 2, "max_base_deflection_xz"))
        object.__setattr__(self, "max_link_flex_angles", _vector(self.max_link_flex_angles, 2, "max_link_flex_angles"))
        if self.max_support_moment <= 0 or self.max_base_pitch <= 0:
            raise ContractError("operating envelope limits must be positive")
        for name in ("max_base_force_xz", "max_base_deflection_xz", "max_link_flex_angles"):
            if (getattr(self, name) <= 0).any():
                raise ContractError(f"{name} limits must be positive")

    def as_array(self) -> np.ndarray:
        return np.concatenate((self.max_base_force_xz, [self.max_support_moment], self.max_base_deflection_xz,
                               [self.max_base_pitch], self.max_link_flex_angles))


@dataclass(frozen=True)
class EnvelopeDescriptor:
    """Schema-v2 descriptor without changing the v1 bundle contract."""

    nominal: RobotDescriptor
    envelope: OperatingEnvelope

    SCHEMA_VERSION: ClassVar[int] = 2

    def as_array(self) -> np.ndarray:
        return np.concatenate((self.nominal.as_array(), self.envelope.as_array()))


@dataclass(frozen=True)
class Task:
    """A single world-frame endpoint task."""

    goal_xz: np.ndarray
    deadline_s: float
    position_tolerance_m: float = 0.010
    speed_tolerance_m_s: float = 0.030
    hold_duration_s: float = 0.200
    frame: str = "world"

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_xz", _vector(self.goal_xz, 2, "goal_xz"))
        if self.deadline_s <= 0 or self.hold_duration_s < 0:
            raise ContractError("deadline must be positive and hold duration nonnegative")
        if self.position_tolerance_m <= 0 or self.speed_tolerance_m_s <= 0:
            raise ContractError("task tolerances must be positive")
        if self.frame != "world":
            raise ContractError(f"unsupported task frame: {self.frame}")

    @property
    def total_duration_s(self) -> float:
        return self.deadline_s + self.hold_duration_s


@dataclass(frozen=True)
class Command:
    """Accepted dimensionless motor request in the public [-1, 1] interface."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = _vector(self.values, 2, "command")
        if (values < -1.0).any() or (values > 1.0).any():
            raise ContractError("command must be within [-1, 1]")
        object.__setattr__(self, "values", values)

    @classmethod
    def accepted(cls, values: object) -> Command:
        """Apply the public boundary rule and return the exact recorded request."""

        raw = _vector(values, 2, "command")
        return cls(np.clip(raw, -1.0, 1.0))


@dataclass(frozen=True)
class Observation:
    """Eight-channel public sensor packet with explicit availability metadata."""

    SCHEMA_VERSION: ClassVar[int] = 1
    values: np.ndarray
    available: np.ndarray
    fresh: np.ndarray
    age_s: np.ndarray
    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _vector(self.values, 8, "observation_values"))
        for name in ("available", "fresh"):
            flags = np.asarray(getattr(self, name), dtype=bool)
            if flags.shape != (8,):
                raise ContractError(f"{name} must have shape (8,)")
            object.__setattr__(self, name, flags)
        ages = _vector(self.age_s, 8, "age_s")
        if (ages < 0).any():
            raise ContractError("age_s must be nonnegative")
        object.__setattr__(self, "age_s", ages)
        if self.time_s < 0 or not np.isfinite(self.time_s):
            raise ContractError("time_s must be finite and nonnegative")

    def feature_values(self) -> np.ndarray:
        """Return finite network inputs; masks carry missing-value information."""

        values = np.where(self.available, self.values, 0.0)
        return np.concatenate((values, self.available.astype(np.float64), self.fresh.astype(np.float64), self.age_s))


@dataclass(frozen=True)
class PrivilegedRecord:
    """Simulator truth kept out of runtime policy features."""

    q: np.ndarray
    dq: np.ndarray
    endpoint_xz: np.ndarray
    endpoint_velocity_xz: np.ndarray
    base_pose: np.ndarray
    load_summary: np.ndarray

    def __post_init__(self) -> None:
        for name in ("q", "dq", "endpoint_xz", "endpoint_velocity_xz"):
            object.__setattr__(self, name, _vector(getattr(self, name), 2, name))
        object.__setattr__(self, "base_pose", _vector(self.base_pose, 3, "base_pose"))
        loads = np.asarray(self.load_summary, dtype=np.float64)
        if loads.ndim != 1 or not np.isfinite(loads).all():
            raise ContractError("load_summary must be a finite vector")
        object.__setattr__(self, "load_summary", loads)
