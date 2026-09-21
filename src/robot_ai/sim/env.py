"""Small batched rigid-arm environment used by P1 commands and later adapters."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Observation, PrivilegedRecord, RobotDescriptor
from .native import NativeArm
from .warp_backend import WarpArmBatch


@dataclass(frozen=True)
class StepBatch:
    """A diagnostic step result; truth is intentionally a separate field."""

    observations: tuple[Observation, ...]
    truth: tuple[PrivilegedRecord, ...]


class RigidArmEnvironment:
    """Backend-selecting environment with explicit per-world reset masks."""

    def __init__(self, descriptor: RobotDescriptor, *, backend: str = "native", device: str = "cpu",
                 world_count: int = 1, physics_dt: float = 0.001) -> None:
        self.backend = backend
        self.device = device
        self.world_count = world_count
        self.descriptor = descriptor
        self.physics_dt = physics_dt
        if backend == "native":
            if world_count != 1:
                raise ValueError("native P1 adapter currently supports one world")
            self._native: NativeArm | None = NativeArm(descriptor, timestep=physics_dt)
            self._warp: WarpArmBatch | None = None
        elif backend == "warp":
            self._native = None
            self._warp = WarpArmBatch(descriptor, world_count, device=device, timestep=physics_dt)
        else:
            raise ValueError(f"unsupported backend: {backend}")

    def reset(self, mask: np.ndarray | None = None, q: np.ndarray | None = None,
              dq: np.ndarray | None = None) -> StepBatch:
        if self._native is not None:
            self._native.reset(None if q is None else np.asarray(q)[0], None if dq is None else np.asarray(dq)[0])
            truth = self._native.truth()
            return StepBatch((self._native.observation(),), (truth,))
        assert self._warp is not None
        self._warp.reset(mask=mask, q=q, dq=dq)
        return self.snapshot()

    def step(self, commands: np.ndarray) -> StepBatch:
        values = np.asarray(commands, dtype=np.float64)
        if self._native is not None:
            truth = self._native.step(values.reshape(2))
            return StepBatch((self._native.observation(),), (truth,))
        assert self._warp is not None
        self._warp.step(values)
        return self.snapshot()

    def snapshot(self) -> StepBatch:
        if self._native is not None:
            return StepBatch((self._native.observation(),), (self._native.truth(),))
        assert self._warp is not None
        q, dq = self._warp.snapshot()
        records: list[PrivilegedRecord] = []
        observations: list[Observation] = []
        for index in range(self.world_count):
            record = self._warp.truth(index)
            records.append(record)
            observations.append(Observation(np.concatenate((q[index], dq[index], np.zeros(2), record.endpoint_xz)),
                                            np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8), 0.0))
        return StepBatch(tuple(observations), tuple(records))
