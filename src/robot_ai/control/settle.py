"""Settle-and-hold around a servo policy: once the arm has arrived, stop moving the servo targets.

A position servo follows even a one-count change of its target, and a network reacting to a flickering
encoder keeps changing them. This wrapper does what an "in position" hold does in industrial motion
control. `settle` seconds after each new goal it freezes the targets, and it releases them whenever the
measured pose error grows past `rearm` (a payload picked up, a push). Everything it uses is known to
the host: when it sent the goal, and what the encoders say.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from ..train.reach import Actor

CONTROL_DT = 0.01


class SettleHold(nn.Module):
    """Wraps a servo policy with the Actor interface; the error block comes from the policy's saved layout."""

    def __init__(self, policy: "Actor", *, settle: float = 1.0, rearm: float = 0.03) -> None:
        super().__init__()
        self.policy: Actor = policy
        self.settle, self.rearm = settle, rearm
        self.n: int = policy.n
        self.oracle, self.incremental, self.task = policy.oracle, policy.incremental, policy.task
        self.error_start = int(policy.layout[2])
        goal_start = self.error_start + int(policy.layout[3])
        self.goal_slice = slice(goal_start, goal_start + 7)

    def initial(self, worlds: int, device: torch.device) -> Tensor:
        inner = self.policy.initial(worlds, device)
        # Appended: time since the goal changed (1), the goal last seen (7), the held command (n), held yet (1).
        extra = torch.zeros((worlds, 9 + self.n), device=device)
        return torch.cat((inner, extra), dim=1)

    def forward(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor]:
        inner_size = feeling.shape[1] - 9 - self.n
        inner, since, seen = feeling[:, :inner_size], feeling[:, inner_size], feeling[:, inner_size + 1:inner_size + 8]
        held, holding = feeling[:, inner_size + 8:inner_size + 8 + self.n], feeling[:, -1] > 0.5
        mean, inner = self.policy(observations, inner)
        command = mean[-1].clamp(-1.0, 1.0)
        observation = observations[-1]
        goal = observation[:, self.goal_slice]
        since = torch.where((goal != seen).any(dim=1), torch.zeros_like(since), since + CONTROL_DT)
        error = observation[:, self.error_start:self.error_start + 7]
        worst = torch.stack((error[:, :3].norm(dim=1), error[:, 3:6].norm(dim=1), error[:, 6].abs()), dim=1).amax(dim=1)
        freeze = (since >= self.settle) & (worst <= self.rearm)
        held = torch.where((freeze & holding)[:, None], held, command)
        feeling = torch.cat((inner, since[:, None], goal, held, freeze.float()[:, None]), dim=1)
        return held[None], feeling
