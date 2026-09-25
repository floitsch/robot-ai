"""A scripted teacher for the contact tasks, to seed what reinforcement learning does not find by itself.

Its targets come from a trained reaching policy; its brake follows simple rules, knowing the true contact
force:
- after any unexpected blow, back off fully and stay there;
- when told to touch (the goal slides along the tool's direction), follow it and hold at the first touch,
  pressing lightly; optionally creep within `creep_distance` of the goal it was given.
A student imitating it has to recognize contact from how the joints feel.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..sim.arm3d_env import POSITION_UNIT, RELEASED, TOUCH, Arm3DEnv

if TYPE_CHECKING:
    from ..train.reach import Actor


class ContactExpert(nn.Module):
    def __init__(self, env: Arm3DEnv, reach: "Actor", *, lead: float = 0.03, creep_distance: float = 0.0,
                 press: float = 0.95) -> None:
        super().__init__()
        self.reach: Actor = reach
        self.env, self.lead, self.creep_distance, self.press = env, lead, creep_distance, press
        self.n = env.action_dim
        self.observation_dim = env.observation_dim
        self.privileged_dim = torch.tensor(env.privileged_dim)
        self.register_buffer("oracle", torch.tensor(1))
        self.register_buffer("incremental", torch.tensor(False))
        self.insight = None
        self.hidden = 0

    def initial(self, worlds: int, device: torch.device) -> Tensor:
        # The reaching policy's feeling, then two latches: backing off after a blow, and holding after a touch.
        return torch.cat((self.reach.initial(worlds, device), torch.zeros((worlds, 2), device=device)), dim=1)

    def forward(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor]:
        env = self.env
        inner, backing, holding = feeling[:, :-2], feeling[:, -2] > 0.5, feeling[:, -1] > 0.5
        seen = observations[..., :int(self.reach.observation_dim)]
        mean, inner = self.reach(seen, inner)
        targets = mean[-1].clamp(-1.0, 1.0)
        observation = observations[-1]
        start = int(self.reach.error_start)
        gap = observation[:, start + 7:start + 7 + env.n].abs().amax(dim=1)  # joints' way to the inverse-kinematics solution
        distance = observation[:, start:start + 3].norm(dim=1) * POSITION_UNIT  # tool to the goal it was told, m
        contact = env._contact[:, 1] > RELEASED
        touch = env._mode == TOUCH
        backing = backing | (contact & ~touch)
        holding = holding | (contact & touch)
        creep = touch & (distance < self.creep_distance)
        creeping = (1.0 - self.lead / gap.clamp(min=1e-6)).clamp(0.0, 0.94)  # below the hold: the anchor follows the arm
        brake = torch.where(creep, creeping, torch.zeros_like(gap))
        # Holding the touch point (the anchor freezes above the hold), with the target a little towards the goal, so the
        # servo keeps a light force on the surface.
        brake = torch.where(holding, torch.full_like(brake, self.press), brake)
        brake = torch.where(backing, torch.full_like(brake, 1.5), brake)
        action = torch.cat((targets, (brake / 1.5)[:, None]), dim=1)
        return action[None], torch.cat((inner, backing.float()[:, None], holding.float()[:, None]), dim=1)
