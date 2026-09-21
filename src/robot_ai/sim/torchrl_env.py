"""Minimal TensorDict adapter with explicit per-world reset semantics."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ..contracts import RobotDescriptor
from .env import RigidArmEnvironment


class TensorDictRigidArm:
    """Adapter boundary used by later TorchRL collectors.

    The simulator keeps truth in its own diagnostic result. This adapter emits
    only public observations and explicit done/terminated/truncated keys.
    """

    def __init__(self, descriptor: RobotDescriptor, *, backend: str = "warp", device: str = "cpu",
                 world_count: int = 1, physics_dt: float = 0.001) -> None:
        self.device = torch.device(device)
        self.environment = RigidArmEnvironment(descriptor, backend=backend, device=device,
                                                world_count=world_count, physics_dt=physics_dt)
        self.world_count = world_count

    def _public(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def reset(self, mask: torch.Tensor | None = None) -> TensorDict:
        reset_mask = None if mask is None else mask.detach().to("cpu").numpy().astype(np.int32)
        result = self.environment.reset(mask=reset_mask)
        observations = np.stack([item.feature_values() for item in result.observations])
        return TensorDict({
            "observation": self._public(observations),
            "done": torch.zeros(self.world_count, dtype=torch.bool, device=self.device),
            "terminated": torch.zeros(self.world_count, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.world_count, dtype=torch.bool, device=self.device),
        }, batch_size=[self.world_count], device=self.device)

    def step(self, action: torch.Tensor) -> TensorDict:
        values = action.detach().to("cpu").numpy()
        result = self.environment.step(values)
        observations = np.stack([item.feature_values() for item in result.observations])
        return TensorDict({
            "observation": self._public(observations),
            "reward": torch.zeros(self.world_count, dtype=torch.float32, device=self.device),
            "done": torch.zeros(self.world_count, dtype=torch.bool, device=self.device),
            "terminated": torch.zeros(self.world_count, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.world_count, dtype=torch.bool, device=self.device),
        }, batch_size=[self.world_count], device=self.device)

