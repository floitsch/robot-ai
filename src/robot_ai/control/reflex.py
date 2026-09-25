"""The classical way to handle contact on a servo arm, as a baseline: a collision reflex and a guarded move.

Both use the stall detection of industrial drives (a following-error limit). A collision is declared once
the arm had been moving towards its target and then, still far from it, has all but stopped, for a
little while. With position servos the host knows only positions and currents. The current alone cannot
tell the start of a move from a blow: both are a jump to full duty, and the servos' gains are not known.
Haddadin, De Luca and Albu-Schaeffer (2017) survey the model-based alternatives.
- Reach: the servos get the inverse-kinematics targets; a collision engages the brake, which holds the arm
  where it is.
- Touch: a guarded move. The brake lets the arm advance towards the solution at a limited joint speed, and
  holds it once it stalls against the surface.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import torch
from torch import Tensor

from ..sim.arm3d_env import TOUCH, Arm3DEnv


class ContactReflex:
    def __init__(self, following_error: float, lead: float, device: torch.device, *, window: float = 0.1,
                 progress: float = 0.002) -> None:
        """following_error: rad from the target that still counts as on the way; lead: how far a guarded move's
        target runs ahead of the arm, rad. Stalled: the gap shrank by less than `progress` rad over `window` s,
        after it had been shrinking faster (differenced velocities are too noisy at a count of flicker)."""

        self.following_error, self.lead, self.device = following_error, lead, device
        self.window, self.progress = round(window / 0.01), progress
        self.env_id: int | None = None

    def _begin(self, env: Arm3DEnv) -> None:
        self.env_id = id(env)
        self.frozen = torch.zeros(env.worlds, dtype=torch.bool, device=self.device)
        self.started = torch.zeros(env.worlds, dtype=torch.bool, device=self.device)
        self.goal = env.joint_goal().clone()
        gap = (self.goal - env.measured_q).abs().amax(dim=1)
        self.history = gap[:, None].repeat(1, self.window)

    def act(self, env: Arm3DEnv) -> Tensor:
        if self.env_id != id(env) or env.tick == 0:
            self._begin(env)
        q = env.measured_q
        target = env.joint_goal()
        gap = (target - q).abs().amax(dim=1)
        new_goal = (target != self.goal).any(dim=1)
        self.goal = target.clone()
        self.history = torch.where(new_goal[:, None], gap[:, None].expand_as(self.history), self.history)
        self.started = self.started & ~new_goal
        shrunk = self.history[:, 0] - gap  # over the window
        self.history = torch.cat((self.history[:, 1:], gap[:, None]), dim=1)
        self.started = self.started | (shrunk > 5.0 * self.progress)
        stalled = self.started & (gap > self.following_error) & (shrunk < self.progress)
        self.frozen = self.frozen | stalled
        # Servo targets at the solution, and the brake: after a collision it backs off as far as it can; during a
        # guarded move it keeps the target a fixed lead ahead of the arm. (The action is the brake / 1.5.)
        touch = env._mode == TOUCH
        creep = torch.where(touch, (1.0 - self.lead / gap.clamp(min=1e-6)).clamp(0.0, 1.0), torch.zeros_like(gap))
        brake = torch.where(self.frozen, torch.full_like(gap, 1.5), creep) / 1.5
        return torch.cat((torch.zeros_like(q), brake[:, None]), dim=1)
