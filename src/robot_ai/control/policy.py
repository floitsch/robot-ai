"""Bounded policy and critic with correct transformed-action probabilities."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchrl.modules import TanhNormal

POLICY_INPUT_DIM = 151
LEARNED_STATE_DIM = 132


class PolicyNetwork(nn.Module):
    """Two-layer policy producing a TorchRL TanhNormal over [-1, 1]."""

    def __init__(self, input_dim: int = POLICY_INPUT_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.mean = nn.Linear(hidden_dim, 2)
        self.scale = nn.Linear(hidden_dim, 2)

    def distribution(self, features: Tensor) -> TanhNormal:
        loc, scale = self.parameters_for_distribution(features)
        return TanhNormal(loc, scale, low=-1.0, high=1.0, event_dims=1)

    def parameters_for_distribution(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(features)
        return self.mean(hidden), torch.nn.functional.softplus(self.scale(hidden)) + 1e-4

    def sample(self, features: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(features)
        action = distribution.rsample()
        return action, distribution.log_prob(action)

    def deterministic(self, features: Tensor) -> Tensor:
        loc, _ = self.parameters_for_distribution(features)
        return torch.tanh(loc)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int = POLICY_INPUT_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)
