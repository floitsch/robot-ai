"""Small deterministic recurrent observer/dynamics model for P5."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import torch
from torch import Tensor, nn


class WorldModel(nn.Module):
    """One posterior feeling shared by physical decoding and future prediction."""

    observation_mean: Tensor
    observation_std: Tensor

    def __init__(self, *, belief_dim: int = 128, observation_dim: int = 32, descriptor_dim: int = 12,
                 endpoint_residual: bool = False, physics_prior: bool = False,
                 fresh_measurement_correction: bool = False) -> None:
        super().__init__()
        self.belief_dim = belief_dim
        self.observation_dim = observation_dim
        self.observation_channels = observation_dim // 4
        self.descriptor_dim = descriptor_dim
        self.endpoint_residual = endpoint_residual
        self.physics_prior = physics_prior
        self.fresh_measurement_correction = fresh_measurement_correction
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim + descriptor_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.observer = nn.GRUCell(128, belief_dim)
        self.dynamics = nn.GRUCell(2 + descriptor_dim + 1, belief_dim)
        self.decoder = nn.Sequential(nn.Linear(belief_dim, 128), nn.SiLU(), nn.Linear(128, 4))
        self.state_observer = None
        self.residual_dynamics = None
        if physics_prior:
            self.state_observer = nn.Sequential(
                nn.Linear(128 + belief_dim, 128), nn.SiLU(), nn.Linear(128, 4),
            )
            state_output = self.state_observer[-1]
            assert isinstance(state_output, nn.Linear)
            nn.init.zeros_(state_output.weight)
            nn.init.zeros_(state_output.bias)
            self.residual_dynamics = nn.Sequential(
                nn.Linear(belief_dim + 4 + 2 + descriptor_dim + 1, 128), nn.SiLU(),
                nn.Linear(128, 4),
            )
            final = self.residual_dynamics[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.register_buffer("observation_mean", torch.zeros(self.observation_channels), persistent=False)
            self.register_buffer("observation_std", torch.ones(self.observation_channels), persistent=False)
        self.endpoint_residual_head = nn.Linear(belief_dim, 4) if endpoint_residual else None

    def set_observation_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Set train-split normalization used to read public state channels."""

        if not self.physics_prior:
            return
        self.observation_mean.copy_(mean[:self.observation_channels])
        self.observation_std.copy_(std[:self.observation_channels])

    def encode_observation(self, observation: Tensor, descriptor: Tensor) -> Tensor:
        return self.observation_encoder(torch.cat((observation, descriptor), dim=-1))

    def observe(self, prior: Tensor, observation: Tensor, descriptor: Tensor) -> Tensor:
        encoded = self.encode_observation(observation, descriptor)
        hidden = self.observer(encoded, prior)
        if self.state_observer is not None:
            # Channels 0:4 are public joint position/velocity measurements.
            # Schema-3 bundles bypassed correction for a fresh zero-age packet.
            # Schema-4 trains the residual on every available packet: transport
            # metadata says when it arrived, not whether it is accurate.
            correction = self.state_observer(torch.cat((encoded, hidden), dim=-1))
            measurement = observation[..., :4] * self.observation_std[:4] + self.observation_mean[:4]
            available = observation[..., self.observation_channels:self.observation_channels + 4] > 0.5
            fresh = observation[..., 2 * self.observation_channels:2 * self.observation_channels + 4] > 0.5
            age = observation[..., 3 * self.observation_channels:3 * self.observation_channels + 4]
            reliable = available & fresh & (age <= 1e-6)
            if self.fresh_measurement_correction:
                state = torch.where(available, measurement + correction, correction)
            else:
                state = torch.where(reliable, measurement,
                                    torch.where(available, measurement + correction, correction))
            hidden = torch.cat((state, hidden[..., 4:]), dim=-1)
        return hidden

    def predict_prior(self, posterior: Tensor, action: Tensor, descriptor: Tensor, dt: float) -> Tensor:
        delta_t = torch.full((*action.shape[:-1], 1), dt, dtype=action.dtype, device=action.device)
        hidden = self.dynamics(torch.cat((action, descriptor, delta_t), dim=-1), posterior)
        if self.residual_dynamics is None:
            return hidden
        q, dq = self.decode(posterior)
        constant_velocity = torch.cat((q + dq * delta_t, dq), dim=-1)
        residual = self.residual_dynamics(torch.cat((hidden, constant_velocity, action, descriptor, delta_t), dim=-1))
        return torch.cat((constant_velocity + residual, hidden[..., 4:]), dim=-1)

    def decode(self, belief: Tensor) -> tuple[Tensor, Tensor]:
        if self.physics_prior:
            return belief[..., :2], belief[..., 2:4]
        decoded = self.decoder(belief)
        return decoded[..., :2], decoded[..., 2:]

    def decode_physical(self, belief: Tensor, descriptor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode joints and world endpoint, including optional P8 residuals."""

        q, dq = self.decode(belief)
        lengths = descriptor[..., :2]
        qsum = q[..., 0] + q[..., 1]
        endpoint = torch.stack((lengths[..., 0] * torch.sin(q[..., 0]) + lengths[..., 1] * torch.sin(qsum),
                                -lengths[..., 0] * torch.cos(q[..., 0]) - lengths[..., 1] * torch.cos(qsum)), dim=-1)
        jacobian = torch.stack((
            torch.stack((lengths[..., 0] * torch.cos(q[..., 0]) + lengths[..., 1] * torch.cos(qsum),
                         lengths[..., 1] * torch.cos(qsum)), dim=-1),
            torch.stack((lengths[..., 0] * torch.sin(q[..., 0]) + lengths[..., 1] * torch.sin(qsum),
                         lengths[..., 1] * torch.sin(qsum)), dim=-1),
        ), dim=-2)
        velocity = torch.einsum("...ij,...j->...i", jacobian, dq)
        if self.endpoint_residual_head is not None:
            residual = self.endpoint_residual_head(belief)
            endpoint = endpoint + residual[..., :2]
            velocity = velocity + residual[..., 2:]
        return q, dq, endpoint, velocity

    def initial(self, observation: Tensor, descriptor: Tensor) -> Tensor:
        zero = torch.zeros((*observation.shape[:-1], self.belief_dim), dtype=observation.dtype, device=observation.device)
        return self.observe(zero, observation, descriptor)
