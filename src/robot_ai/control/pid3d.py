"""The classical way to drive a cheap 3D arm, as a baseline: inverse kinematics to joint targets, a
minimum-jerk reference, and per-joint PID with anti-windup and gravity compensation from the nominal
masses and the configured link lengths. Gains are scheduled on each joint's nominal inertia, so every
joint gets the same bandwidth and damping. It knows exactly what the network is told, nothing more.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from ..sim.arm3d_env import (
    MASSES,
    NOMINAL_TORQUES,
    TOOL_SIDE_OFFSET,
    Arm3DEnv,
    _links,
    arm3d_dynamics,
)

ARMATURE = 0.0025  # a typical geared motor's rotor inertia at the joint, kg m^2


class Pid3D:
    def __init__(self, gains: Sequence[float], device: torch.device) -> None:
        """gains = (bandwidth rad/s, damping ratio, integral rate as a fraction of bandwidth, reference s/rad)."""

        self.omega, self.zeta, self.integral_rate, self.seconds_per_rad = (float(g) for g in gains)
        self.device = device
        self.torque = torch.as_tensor(NOMINAL_TORQUES, dtype=torch.float32, device=device)
        self.env_id: int | None = None
        self.goal: Tensor | None = None

    def _nominal(self, env: Arm3DEnv) -> None:
        """The arm as configured: its link lengths, catalogue masses, no payload."""

        lengths = env._geometry.cpu().numpy() * 0.25
        links, _ = _links(lengths.astype(np.float64), np.tile(MASSES, (env.worlds, 1)), np.full(env.worlds, TOOL_SIDE_OFFSET))
        self.links = {name: torch.as_tensor(np.ascontiguousarray(links[name]), dtype=torch.float32, device=self.device)
                      for name in ("joint_pos", "axis", "com", "mass", "inertia")}
        self.env_id, self.goal = id(env), None

    def act(self, env: Arm3DEnv) -> Tensor:
        if self.env_id != id(env) or env.tick == 0:
            self._nominal(env)
        goal, q = env.joint_goal(), env.measured_q
        if self.goal is None:
            self.origin = q.clone()
            previous = torch.full_like(goal, torch.nan)
            self.elapsed, self.duration = torch.zeros_like(q[:, :1]), torch.ones_like(q[:, :1])
            self.integral = torch.zeros_like(q)
        else:
            previous = self.goal
        moved = (goal != previous).any(dim=1, keepdim=True)
        self.origin = torch.where(moved, q, self.origin)
        self.elapsed = torch.where(moved, torch.zeros_like(self.elapsed), self.elapsed + 0.01)
        span = (goal - self.origin).abs().amax(dim=1, keepdim=True)
        self.duration = torch.where(moved, (self.seconds_per_rad * span).clamp(min=0.25), self.duration)
        self.goal = goal.clone()
        phase = (self.elapsed / self.duration).clamp(0.0, 1.0)
        blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
        rate = 30.0 * phase**2 * (1.0 - phase) ** 2 / self.duration
        reference = self.origin + (goal - self.origin) * blend
        error = reference - q
        links = self.links
        m, gravity = arm3d_dynamics(q, torch.zeros_like(q), links["joint_pos"], links["axis"], links["com"], links["mass"],
                                    links["inertia"])
        inertia = torch.diagonal(m, dim1=1, dim2=2) + ARMATURE
        kp, kd = inertia * self.omega**2, 2.0 * self.zeta * inertia * self.omega
        ki = kp * self.omega * self.integral_rate
        self.integral = torch.where(error.abs() < 0.15, self.integral + 0.01 * error, self.integral)
        self.integral = torch.maximum(torch.minimum(self.integral, 0.3 * self.torque / ki.clamp(min=1e-6)),
                                      -0.3 * self.torque / ki.clamp(min=1e-6))  # anti-windup: 30% of rated torque
        torque = kp * error + ki * self.integral + kd * ((goal - self.origin) * rate - env.velocity) + gravity
        return (torque / self.torque).clamp(-1.0, 1.0)
