"""One two-joint policy, shared by every limb of a chain, with an optional message between limbs.

Each limb runs the same `Actor` on its own slice of the chain observation, exactly as a
single-arm controller would, and carries its own feeling. With `message > 0` every limb also
emits a short vector each tick that the other limbs read on the next tick: the shared feeling.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .reach import OBSERVATION_BLOCKS, SENSOR_BLOCKS, Actor

JOINTS_PER_LIMB = 2
# Per joint, the chain's privileged vector holds the 6 observation blocks, true q, true dq, then 6 hidden fields.
PRIVILEGED_BLOCKS = OBSERVATION_BLOCKS + 2 + 6


def limb_slices(total: Tensor, blocks: int, limbs: int, limb: int) -> Tensor:
    """Take one limb's joints out of every block of a [..., blocks * n] tensor."""

    n = limbs * JOINTS_PER_LIMB
    k = JOINTS_PER_LIMB
    pieces = [total[..., block * n + limb * k: block * n + limb * k + k] for block in range(blocks)]
    return torch.cat(pieces, dim=-1)


class LimbPolicy(nn.Module):
    """Presents the `Actor` interface for the whole chain while sharing one two-joint actor across limbs."""

    limbs: Tensor
    message_dim: Tensor
    peek: Tensor

    def __init__(self, *, limbs: int, message: int = 0, hidden: int = 64, fine_scales: Sequence[float] = (),
                 oracle: int = 0, initial_std: Sequence[float] | None = None, peek: bool = False) -> None:
        super().__init__()
        self.register_buffer("limbs", torch.tensor(limbs))
        self.register_buffer("message_dim", torch.tensor(message))
        # With `peek`, every limb also sees the other limbs' raw sensor readings (angles, velocities, currents) at the
        # same tick: shared sensing rather than a learned message, fully batched and one tick fresher.
        self.register_buffer("peek", torch.tensor(peek))
        self.count, self.msg, self.peeking = limbs, message, peek
        self.n = limbs * JOINTS_PER_LIMB
        self.observation_dim = OBSERVATION_BLOCKS * self.n
        self.privileged_dim = torch.tensor(PRIVILEGED_BLOCKS * self.n)
        per_limb_std = None if initial_std is None else list(initial_std)[:JOINTS_PER_LIMB]
        self.actor = Actor(recurrent=True, fine_scales=fine_scales, hidden=hidden, oracle=oracle, joints=JOINTS_PER_LIMB,
                           privileged_dim=PRIVILEGED_BLOCKS * JOINTS_PER_LIMB, initial_std=per_limb_std,
                           extra_inputs=message * (limbs - 1) + (SENSOR_BLOCKS * JOINTS_PER_LIMB * (limbs - 1) if peek else 0),
                           message=message)
        self.oracle, self.incremental, self.insight, self.hidden = self.actor.oracle, self.actor.incremental, None, hidden

    @property
    def state_dim(self) -> int:
        return self.count * (self.actor.state_dim + self.msg)

    def initial(self, worlds: int, device: torch.device) -> Tensor:
        return torch.zeros((worlds, self.state_dim), device=device)

    def _split(self, feeling: Tensor) -> tuple[Tensor, Tensor]:
        """[worlds, limbs * (state + msg)] -> states [limbs, worlds, state], messages [limbs, worlds, msg]."""

        worlds = feeling.shape[0]
        per = feeling.reshape(worlds, self.count, self.actor.state_dim + self.msg).transpose(0, 1)
        return per[..., :self.actor.state_dim], per[..., self.actor.state_dim:]

    def _join(self, states: Tensor, messages: Tensor) -> Tensor:
        return torch.cat((states, messages), dim=-1).transpose(0, 1).reshape(states.shape[1], self.state_dim)

    def distribution_and_features(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        steps, worlds = observations.shape[:2]
        blocks = PRIVILEGED_BLOCKS if observations.shape[-1] == int(self.privileged_dim) else OBSERVATION_BLOCKS
        per_limb = [limb_slices(observations, blocks, self.count, limb) for limb in range(self.count)]
        if self.peeking:
            sensors = [limb_slices(observations, SENSOR_BLOCKS, self.count, limb) for limb in range(self.count)]
            per_limb = [torch.cat((per_limb[limb], *(sensors[o] for o in range(self.count) if o != limb)), dim=-1)
                        for limb in range(self.count)]
        states, messages = self._split(feeling)
        if self.msg == 0:
            # No coupling: every limb is an independent sequence, so run them all through the GRU at once.
            stacked = torch.cat(per_limb, dim=1)  # [steps, limbs * worlds, dim]
            mean, std, state, features = self.actor.distribution_and_features(stacked, states.reshape(-1, states.shape[-1]))
            mean = torch.cat(mean.split(worlds, dim=1), dim=-1)
            std = torch.cat(std.split(worlds, dim=1), dim=-1)
            return mean, std, self._join(state.reshape(self.count, worlds, -1), messages), features
        # With messages the limbs are coupled tick by tick: step through time, feeding each limb the others' last messages.
        means, stds, feats = [], [], []
        limb_states: list[Tensor] = list(states)
        limb_messages: list[Tensor] = list(messages)
        assert self.actor.message is not None
        for t in range(steps):
            new_states, new_messages, mean_t, std_t, feat_t = [], [], [], [], []
            for limb in range(self.count):
                others = torch.cat([limb_messages[o] for o in range(self.count) if o != limb], dim=-1)
                inputs = torch.cat((per_limb[limb][t], others), dim=-1)[None]
                mean, std, state, features = self.actor.distribution_and_features(inputs, limb_states[limb])
                new_states.append(state)
                new_messages.append(self.actor.message(features[0]))
                mean_t.append(mean[0])
                std_t.append(std[0])
                feat_t.append(features[0])
            limb_states, limb_messages = new_states, new_messages
            means.append(torch.cat(mean_t, dim=-1))
            stds.append(torch.cat(std_t, dim=-1))
            feats.append(torch.cat(feat_t, dim=-1))
        feeling = self._join(torch.stack(limb_states), torch.stack(limb_messages))
        return torch.stack(means), torch.stack(stds), feeling, torch.stack(feats)

    def distribution(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, std, feeling, _ = self.distribution_and_features(observations, feeling)
        return mean, std, feeling

    def forward(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor]:
        mean, _, feeling = self.distribution(observations, feeling)
        return mean, feeling
