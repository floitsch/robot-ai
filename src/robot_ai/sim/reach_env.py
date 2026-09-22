"""Batched reach task over a population of imperfect arms.

The controller is told only where each joint should be, in the units its own
encoders report. It gets no robot description: no lengths, masses, motor ratings,
or fault flags. Goals and the robot itself can change mid-episode.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from torch import Tensor

from .arm_batch import JOINTS, ArmBatch
from .population import NOMINAL_TORQUES, sample_change, sample_population, sample_push

OBSERVATION_DIM = 6 * JOINTS
ERROR_SLICE = slice(4 * JOINTS, 5 * JOINTS)  # goal minus measured angle, rad
PRIVILEGED_DIM = OBSERVATION_DIM + 2 * JOINTS + 5 * JOINTS + 1
GOAL_RANGE = np.array([1.4, 1.6])
TOLERANCE = 0.03  # rad, per joint, in encoder units
SETTLED_SPEED = 0.1  # rad/s
FINAL_WINDOW = 30  # ticks
COMMAND_RATE = 0.2  # largest command change per tick for an incremental controller


@dataclass
class EpisodeSummary:
    success: Tensor  # [worlds] bool: inside tolerance and at rest over the final window
    final_error: Tensor  # [worlds] worst |error| over the final window, rad
    mean_error: Tensor  # [worlds] mean |error| over the whole episode, rad: penalizes slowness
    roughness: Tensor  # [worlds] mean squared command change per tick
    limit_time: Tensor  # [worlds] fraction of the episode spent on a hard stop
    episode_return: Tensor  # [worlds]


class ReachEnv:
    def __init__(self, worlds: int, *, device: str = "cuda:0", seed: int = 0, severity: float = 1.0,
                 episode_ticks: int = 300, changes: bool = True, pushes: bool = False, still_weight: float = 0.0,
                 friction_probability: float = 0.7, mixed: bool = False, reward_tolerance: float = TOLERANCE) -> None:
        self.worlds, self.device, self.severity = worlds, device, severity
        self.episode_ticks, self.changes = episode_ticks, changes
        self.rng = np.random.default_rng(seed)
        self.pushes, self.still_weight, self.friction_probability = pushes, still_weight, friction_probability
        self.mixed = mixed
        # Rewards may demand more than the success criterion, so that a trained controller clears it with margin.
        self.reward_tolerance = reward_tolerance
        self._mix_rng = np.random.default_rng([seed, 2])
        self._push_rng = np.random.default_rng([seed, 1])  # its own stream, so pushes never shift the robots drawn
        self.torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
        if self.torch_device.type == "cuda":
            wp.init()
            # One stream for both libraries, so zero-copy views never race a kernel.
            wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream(self.torch_device)), device)
        self._torque = torch.as_tensor(NOMINAL_TORQUES, dtype=torch.float32, device=self.torch_device)
        self._scale = torch.tensor([torch.pi, torch.pi], dtype=torch.float32, device=self.torch_device)

    def _tensor(self, values: np.ndarray) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=self.torch_device)

    def _hidden(self, arms: np.ndarray, joints: np.ndarray) -> Tensor:
        """What a training-only critic may know about the robot. Never a policy input."""

        return self._tensor(np.concatenate((
            joints["torque_scale"] / NOMINAL_TORQUES, joints["coulomb"] * 4.0, joints["half_gap"] * 50.0,
            joints["damping"] * 10.0, joints["delay_steps"] / 30.0, arms["m2"][:, None]), axis=1))

    def reset(self) -> Tensor:
        severity: float | np.ndarray = self.severity
        if self.mixed:
            # A training population from flawless to badly worn, some shoved around by a neighbour: a controller
            # shipped to unknown robots must not be tuned to one level of wear.
            severity = np.where(self._mix_rng.random(self.worlds) < 0.15, 0.0, self._mix_rng.uniform(0.0, 1.0, self.worlds))
        arms, joints = sample_population(self.rng, self.worlds, severity=severity,
                                         friction_probability=self.friction_probability)
        changed = sample_change(self.rng, arms, joints, severity=severity)
        ticks = self.episode_ticks
        change_tick = np.where(self.rng.random(self.worlds) < (0.5 if self.changes else 0.0),
                               self.rng.integers(20, ticks - 60, self.worlds), np.iinfo(np.int32).max)
        self.batch = ArmBatch(arms, joints, device=self.device, seed=int(self.rng.integers(2**31)),
                              changed=changed, change_tick=change_tick,
                              push=sample_push(self._push_rng, self.worlds, severity=severity) if self.pushes or self.mixed else None)
        self.batch.reset(self.rng.uniform(-GOAL_RANGE, GOAL_RANGE, (self.worlds, JOINTS)))
        goals = self.rng.uniform(-GOAL_RANGE, GOAL_RANGE, (2, self.worlds, JOINTS))
        regoal = np.where(self.rng.random(self.worlds) < 0.7, self.rng.integers(80, ticks - 120, self.worlds), ticks + 1)
        self.arms, self.joints, self.changed, self.change_tick, self.regoal_tick = arms, joints, changed, change_tick, regoal
        self._goals, self._regoal = self._tensor(goals), self._tensor(regoal)
        self._change_tick = self._tensor(np.minimum(change_tick, ticks + 1))
        self._bias = self._tensor(joints["enc_bias"])
        self._hidden_before, self._hidden_after = self._hidden(arms, joints), self._hidden(*changed)
        self._observation = wp.to_torch(self.batch.observation)
        self._truth = wp.to_torch(self.batch.truth)
        self._metrics = wp.to_torch(self.batch.metrics)
        self.tick = 0
        self._previous = torch.zeros((self.worlds, JOINTS), device=self.torch_device)
        zeros = torch.zeros(self.worlds, device=self.torch_device)
        self._sum_error, self._sum_rough, self._sum_limit, self._return = zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()
        self._window_error, self._window_speed = zeros.clone(), zeros.clone()
        return self._observe()

    def _goal(self) -> Tensor:
        return torch.where((self.tick >= self._regoal)[:, None], self._goals[1], self._goals[0])

    def _observe(self) -> Tensor:
        raw = self._observation
        goal = self._goal()
        self.measured_q, self.velocity = raw[:, 0:2].clone(), raw[:, 2:4].clone()
        return torch.cat((self.measured_q / self._scale, self.velocity / 5.0, raw[:, 4:6] / self._torque,
                          goal / self._scale, goal - self.measured_q, self._previous), dim=1)

    def integrate(self, change: Tensor) -> Tensor:
        """Command for a controller that outputs changes: holding a load steady is then a zero output."""

        return (self._previous + COMMAND_RATE * change.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)

    def goal(self) -> Tensor:
        return self._goal()

    def privileged(self, observation: Tensor) -> Tensor:
        hidden = torch.where((self.tick >= self._change_tick)[:, None], self._hidden_after, self._hidden_before)
        return torch.cat((observation, self._truth[:, 0:2] / self._scale, self._truth[:, 2:4] / 5.0, hidden), dim=1)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        action = action.clamp(-1.0, 1.0)
        self.batch.step(action)
        self.tick += 1
        # Error is judged where the user can see it: on the robot's own encoder scale.
        error = (self._goal() - (self._truth[:, 0:2] + self._bias)).abs()
        speed = self._truth[:, 2:4].abs().amax(dim=1)
        rough = (action - self._previous).square().sum(dim=1)
        limit = self._metrics[:, 4:6].sum(dim=1)
        arrived = ((error < self.reward_tolerance).all(dim=1) & (speed < SETTLED_SPEED)).float()
        # Smooth closeness keeps a gradient toward the goal even while exploration shakes the arm.
        close = torch.exp(-error / self.reward_tolerance).mean(dim=1)
        reward = 0.1 * (-error.sum(dim=1) + close + arrived - 0.5 * rough - 0.002 * self._metrics[:, 0] / 0.01 - limit)
        if self.still_weight:
            # Being on target only counts if the arm has also stopped: graded, unlike the all-or-nothing `arrived`.
            reward = reward + 0.1 * self.still_weight * close * torch.exp(-speed / (2.0 * SETTLED_SPEED))
        self._previous = action
        self._sum_error += error.mean(dim=1)
        self._sum_rough += rough
        self._sum_limit += limit / JOINTS
        self._return += reward
        if self.tick > self.episode_ticks - FINAL_WINDOW:
            self._window_error = torch.maximum(self._window_error, error.amax(dim=1))
            self._window_speed = torch.maximum(self._window_speed, speed)
        return self._observe(), reward

    def summary(self) -> EpisodeSummary:
        ticks = float(self.tick)
        return EpisodeSummary((self._window_error < TOLERANCE) & (self._window_speed < SETTLED_SPEED),
                              self._window_error, self._sum_error / ticks, self._sum_rough / ticks,
                              self._sum_limit / ticks, self._return)
