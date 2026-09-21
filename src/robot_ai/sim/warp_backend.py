"""Batched Warp rigid-arm backend with device-resident state."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import numpy as np
import warp as wp

from ..contracts import PrivilegedRecord, RobotDescriptor
from .mechanics import endpoint_velocity, forward_kinematics


@wp.kernel
def _integrate(
    q: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dq: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    commands: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    lengths: wp.vec2,
    masses: wp.vec2,
    inertias: wp.vec2,
    torques: wp.vec2,
    joint_limits: wp.vec4,
    dt: float,
    gravity: float,
    damping: float,
) -> None:  # type: ignore[no-untyped-def]
    world = wp.tid()
    q1 = q[2 * world]
    q2 = q[2 * world + 1]
    dq1 = dq[2 * world]
    dq2 = dq[2 * world + 1]
    q12 = q1 + q2
    lc1 = lengths[0] * 0.5
    lc2 = lengths[1] * 0.5
    c2 = wp.cos(q2)
    m11 = inertias[0] + inertias[1] + masses[0] * lc1 * lc1 + masses[1] * (lengths[0] * lengths[0] + lc2 * lc2 + 2.0 * lengths[0] * lc2 * c2)
    m12 = inertias[1] + masses[1] * (lc2 * lc2 + lengths[0] * lc2 * c2)
    m22 = inertias[1] + masses[1] * lc2 * lc2
    g1 = -gravity * (masses[0] * lengths[0] * 0.5 + masses[1] * lengths[0]) * wp.sin(q1)
    g1 = g1 - gravity * masses[1] * lengths[1] * 0.5 * wp.sin(q12)
    g2 = -gravity * masses[1] * lengths[1] * 0.5 * wp.sin(q12)
    coupling = -masses[1] * lengths[0] * lc2 * wp.sin(q2)
    coriolis1 = coupling * (2.0 * dq1 * dq2 + dq2 * dq2)
    coriolis2 = -coupling * dq1 * dq1
    rhs1 = commands[2 * world] * torques[0] + g1 - coriolis1 - damping * dq1
    rhs2 = commands[2 * world + 1] * torques[1] + g2 - coriolis2 - damping * dq2
    # MuJoCo's Euler integrator treats joint damping implicitly.  Applying the
    # same diagonal update keeps the reduced GPU reference aligned with the
    # native fixture at the public physics timestep.
    m11 = m11 + dt * damping
    m22 = m22 + dt * damping
    determinant = m11 * m22 - m12 * m12
    ddq1 = (m22 * rhs1 - m12 * rhs2) / determinant
    ddq2 = (m11 * rhs2 - m12 * rhs1) / determinant
    dq[2 * world] = dq1 + dt * ddq1
    dq[2 * world + 1] = dq2 + dt * ddq2
    q[2 * world] = q1 + dt * dq[2 * world]
    q[2 * world + 1] = q2 + dt * dq[2 * world + 1]
    if q[2 * world] < joint_limits[0]:
        q[2 * world] = joint_limits[0]
        dq[2 * world] = max(dq[2 * world], 0.0)
    if q[2 * world] > joint_limits[1]:
        q[2 * world] = joint_limits[1]
        dq[2 * world] = min(dq[2 * world], 0.0)
    if q[2 * world + 1] < joint_limits[2]:
        q[2 * world + 1] = joint_limits[2]
        dq[2 * world + 1] = max(dq[2 * world + 1], 0.0)
    if q[2 * world + 1] > joint_limits[3]:
        q[2 * world + 1] = joint_limits[3]
        dq[2 * world + 1] = min(dq[2 * world + 1], 0.0)


@wp.kernel
def _reset(
    q: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dq: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    mask: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    q_values: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dq_values: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
) -> None:  # type: ignore[no-untyped-def]
    world = wp.tid()
    if mask[world] != 0:
        q[2 * world] = q_values[2 * world]
        q[2 * world + 1] = q_values[2 * world + 1]
        dq[2 * world] = dq_values[2 * world]
        dq[2 * world + 1] = dq_values[2 * world + 1]


class WarpArmBatch:
    """Vectorized arm state whose simulation arrays remain on the selected device."""

    def __init__(self, descriptor: RobotDescriptor, world_count: int, *, device: str = "cuda:0",
                 timestep: float = 0.001) -> None:
        if world_count < 1:
            raise ValueError("world_count must be positive")
        self.descriptor = descriptor
        self.world_count = world_count
        self.device = device
        self.timestep = timestep
        wp.config.kernel_cache_dir = str(__import__("robot_ai.config", fromlist=["project_root"]).project_root() / ".cache" / "warp")
        wp.init()
        self.q = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
        self.dq = wp.zeros(world_count * 2, dtype=wp.float32, device=device)
        self.commands = wp.zeros(world_count * 2, dtype=wp.float32, device=device)

    def reset(self, mask: np.ndarray | None = None, q: np.ndarray | None = None,
              dq: np.ndarray | None = None) -> None:
        mask_values = np.ones(self.world_count, dtype=np.int32) if mask is None else np.asarray(mask, dtype=np.int32)
        if mask_values.shape != (self.world_count,):
            raise ValueError("reset mask must have one entry per world")
        q_values = np.zeros((self.world_count, 2), dtype=np.float32) if q is None else np.asarray(q, dtype=np.float32)
        dq_values = np.zeros((self.world_count, 2), dtype=np.float32) if dq is None else np.asarray(dq, dtype=np.float32)
        if q_values.shape != (self.world_count, 2) or dq_values.shape != (self.world_count, 2):
            raise ValueError("reset states must have shape (world_count, 2)")
        wp.launch(_reset, dim=self.world_count, inputs=[self.q, self.dq, wp.array(mask_values, device=self.device),
                  wp.array(q_values.reshape(-1), device=self.device), wp.array(dq_values.reshape(-1), device=self.device)],
                  device=self.device)

    def step(self, commands: np.ndarray) -> None:
        values = np.asarray(commands, dtype=np.float32)
        if values.shape != (self.world_count, 2):
            raise ValueError("commands must have shape (world_count, 2)")
        values = np.clip(values, -1.0, 1.0)
        wp.copy(self.commands, wp.array(values.reshape(-1), dtype=wp.float32, device=self.device))
        inputs = [self.q, self.dq, self.commands,
                  wp.vec2(*self.descriptor.link_lengths.astype(np.float32)),
                  wp.vec2(*self.descriptor.nominal_masses.astype(np.float32)),
                  wp.vec2(*self.descriptor.nominal_inertias.astype(np.float32)),
                  wp.vec2(*self.descriptor.nominal_torque_scales.astype(np.float32)),
                  wp.vec4(*self.descriptor.joint_limits.astype(np.float32).reshape(-1)),
                  self.timestep, 9.81, 0.02]
        wp.launch(_integrate, dim=self.world_count, inputs=inputs, device=self.device)

    def step_torch(self, commands: object) -> None:
        """Step from a Torch tensor without staging commands through the CPU."""

        import torch

        if not isinstance(commands, torch.Tensor):
            raise TypeError("commands must be a torch.Tensor")
        if commands.shape != (self.world_count, 2):
            raise ValueError("commands must have shape (world_count, 2)")
        if commands.device.type != self.device.split(":", 1)[0]:
            raise ValueError("Torch command tensor must use the Warp device")
        values = commands.detach().clamp(-1.0, 1.0).contiguous()
        wp.copy(self.commands, wp.from_torch(values.reshape(-1)))
        inputs = [self.q, self.dq, self.commands,
                  wp.vec2(*self.descriptor.link_lengths.astype(np.float32)),
                  wp.vec2(*self.descriptor.nominal_masses.astype(np.float32)),
                  wp.vec2(*self.descriptor.nominal_inertias.astype(np.float32)),
                  wp.vec2(*self.descriptor.nominal_torque_scales.astype(np.float32)),
                  wp.vec4(*self.descriptor.joint_limits.astype(np.float32).reshape(-1)),
                  self.timestep, 9.81, 0.02]
        wp.launch(_integrate, dim=self.world_count, inputs=inputs, device=self.device)

    def torch_state(self) -> tuple[object, object]:
        """Return zero-copy Torch views of q and dq after queued work completes."""

        import torch

        wp.synchronize_device(self.device)
        q = wp.to_torch(self.q).reshape(self.world_count, 2)
        dq = wp.to_torch(self.dq).reshape(self.world_count, 2)
        if not isinstance(q, torch.Tensor) or not isinstance(dq, torch.Tensor):
            raise TypeError("Warp did not return Torch state views")
        return q, dq

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        wp.synchronize_device(self.device)
        return self.q.numpy().reshape(self.world_count, 2).copy(), self.dq.numpy().reshape(self.world_count, 2).copy()

    def truth(self, index: int = 0) -> PrivilegedRecord:
        q, dq = self.snapshot()
        endpoint = forward_kinematics(q[index], self.descriptor.link_lengths)
        velocity = endpoint_velocity(q[index], dq[index], self.descriptor.link_lengths)
        return PrivilegedRecord(q[index], dq[index], endpoint, velocity, np.zeros(3), np.zeros(4))
