"""Batched reach task over populations of imperfect N-joint chains.

A chain is `limbs` two-joint limbs stacked end to end. Each joint keeps the single-arm defect
model; the lower limbs' motors are rated stronger because they carry the limbs above. Goals are
per joint in encoder units, as for the single arm, so a controller for one limb can be applied to
each limb of a chain unchanged.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from torch import Tensor

from .chain_batch import ChainBatch, LinkParams
from .population import (
    NOMINAL_LENGTHS,
    NOMINAL_MASSES,
    NOMINAL_TORQUES,
    _per_world,
    sample_joint_defects,
)
from .reach_env import FINAL_WINDOW, GOAL_RANGE, SETTLED_SPEED, TOLERANCE

JOINTS_PER_LIMB = 2
LIMIT = 2.4


def nominal_torques(limbs: int) -> np.ndarray:
    """Rated torques per joint: each limb below the top one is rated stronger for what it carries."""

    return np.concatenate([NOMINAL_TORQUES * (1.0 + 2.0 * (limbs - 1 - limb)) for limb in range(limbs)])


def sample_chain_population(rng: np.random.Generator, worlds: int, limbs: int, *, severity: float | np.ndarray = 1.0,
                            friction_probability: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    n = limbs * JOINTS_PER_LIMB
    shape = (worlds, n)
    per_robot, severity_column = _per_world(severity, worlds)
    lengths = np.tile(NOMINAL_LENGTHS, (worlds, limbs)) * (1.0 + severity_column * rng.uniform(-0.15, 0.15, shape))
    masses = np.tile(NOMINAL_MASSES, (worlds, limbs)) * (1.0 + severity_column * rng.uniform(-0.20, 0.20, shape))
    payload = np.where(rng.random(worlds) < 0.5, rng.uniform(0.0, 0.15, worlds), 0.0) * per_robot
    links = np.zeros(shape, dtype=LinkParams.numpy_dtype())
    links["length"], links["mass"], links["com"] = lengths, masses, lengths / 2.0
    links["inertia"] = masses * lengths**2 / 12.0
    _add_tip_mass(links, payload)
    limits = np.array([[-LIMIT, LIMIT]] * n)
    joints = sample_joint_defects(rng, worlds, n, nominal_torques(limbs), limits, severity=severity,
                                  friction_probability=friction_probability)
    return links, joints


def _add_tip_mass(links: np.ndarray, added: np.ndarray) -> None:
    """Attach a point mass at the tip of the last link, in place."""

    last = links[:, -1]
    total = last["mass"] + added
    com = (last["mass"] * last["com"] + added * last["length"]) / total
    last["inertia"] = last["inertia"] + last["mass"] * (com - last["com"]) ** 2 + added * (last["length"] - com) ** 2
    last["mass"], last["com"] = total, com


def sample_chain_change(rng: np.random.Generator, links: np.ndarray, joints: np.ndarray, *,
                        severity: float | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray]:
    worlds, n = links.shape
    per_robot, severity_column = _per_world(severity, worlds)
    changed_links, changed_joints = links.copy(), joints.copy()
    _add_tip_mass(changed_links, np.where(rng.random(worlds) < 0.6, rng.uniform(0.05, 0.20, worlds), 0.0) * per_robot)
    fading = np.where(rng.random((worlds, n)) < 0.3, rng.uniform(0.0, 0.4, (worlds, n)), 0.0) * severity_column
    changed_joints["torque_scale"] = joints["torque_scale"] * (1.0 - fading)
    changed_joints["coulomb"] = joints["coulomb"] + np.where(rng.random((worlds, n)) < 0.3, rng.uniform(0.0, 0.2, (worlds, n)), 0.0) * severity_column
    return changed_links, changed_joints


@dataclass
class ChainSummary:
    success: Tensor  # [worlds] every joint within tolerance and at rest over the final window
    limb_success: Tensor  # [worlds, limbs]
    final_error: Tensor  # [worlds] worst joint error over the final window, rad
    mean_error: Tensor  # [worlds] mean joint error over the episode, rad
    roughness: Tensor  # [worlds]
    episode_return: Tensor  # [worlds]


class ChainEnv:
    """Same contract as `ReachEnv`, with per-joint layouts of size n = 2 * limbs.

    Observation: q/pi (n), dq/5 (n), current/rated (n), goal/pi (n), goal - q (n), previous command (n).
    Privileged: observation, true q/pi (n), true dq/5 (n), hidden condition (5n), link masses (n).
    """

    def __init__(self, worlds: int, limbs: int = 2, *, device: str = "cuda:0", seed: int = 0, severity: float = 1.0,
                 episode_ticks: int = 300, changes: bool = True, reward_tolerance: float = TOLERANCE,
                 still_weight: float = 1.0, roughness_weight: float = 4.0) -> None:
        self.worlds, self.limbs, self.n = worlds, limbs, limbs * JOINTS_PER_LIMB
        self.limb_joints = JOINTS_PER_LIMB
        self.device, self.severity, self.episode_ticks, self.changes = device, severity, episode_ticks, changes
        # A chain has a fast bending mode that a 100 Hz controller can excite; position error alone barely sees the
        # resulting 0.01 rad chatter, so the reward also asks for rest directly.
        self.reward_tolerance, self.still_weight = reward_tolerance, still_weight
        # Through one tick of delay a chain admits a 50 Hz command limit cycle (sign flip every tick) that the arm's
        # plant filtered out; a strong roughness penalty removes that local optimum.
        self.roughness_weight = roughness_weight
        self.rng = np.random.default_rng([seed, limbs])
        self.torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
        if self.torch_device.type == "cuda":
            wp.init()
            wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream(self.torch_device)), device)
        self._torque = torch.as_tensor(nominal_torques(limbs), dtype=torch.float32, device=self.torch_device)
        # Exploration noise as a fraction of rated torque, scaled so every joint gets the arm's absolute noise level.
        self.initial_std = (0.5 * np.tile(NOMINAL_TORQUES, limbs) / nominal_torques(limbs)).tolist()
        self.observation_dim = 6 * self.n
        self.privileged_dim = self.observation_dim + 2 * self.n + 6 * self.n

    def _tensor(self, values: np.ndarray) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), device=self.torch_device)

    def _hidden(self, links: np.ndarray, joints: np.ndarray) -> Tensor:
        return self._tensor(np.concatenate((joints["torque_scale"] / nominal_torques(self.limbs), joints["coulomb"] * 4.0,
                                            joints["half_gap"] * 50.0, joints["damping"] * 10.0, joints["delay_steps"] / 30.0,
                                            links["mass"]), axis=1))

    def reset(self) -> Tensor:
        links, joints = sample_chain_population(self.rng, self.worlds, self.limbs, severity=self.severity)
        changed = sample_chain_change(self.rng, links, joints, severity=self.severity)
        ticks = self.episode_ticks
        change_tick = np.where(self.rng.random(self.worlds) < (0.5 if self.changes else 0.0),
                               self.rng.integers(20, ticks - 60, self.worlds), np.iinfo(np.int32).max)
        self.batch = ChainBatch(links, joints, device=self.device, seed=int(self.rng.integers(2**31)),
                                changed=changed, change_tick=change_tick)
        # Start and goals in a range where a stacked chain does not fold onto itself.
        span = np.tile(GOAL_RANGE * 0.6, self.limbs)
        self.batch.reset(self.rng.uniform(-span, span, (self.worlds, self.n)))
        goals = self.rng.uniform(-span, span, (2, self.worlds, self.n))
        regoal = np.where(self.rng.random(self.worlds) < 0.7, self.rng.integers(80, ticks - 120, self.worlds), ticks + 1)
        self.links, self.joints, self.changed, self.change_tick, self.regoal_tick = links, joints, changed, change_tick, regoal
        self._tensors: dict[int, tuple[np.ndarray, np.ndarray, dict[str, tuple[Tensor, Tensor]]]] = {}
        self._goals, self._regoal = self._tensor(goals), self._tensor(regoal)
        self._change_tick = self._tensor(np.minimum(change_tick, ticks + 1))
        self._bias = self._tensor(joints["enc_bias"])
        self._hidden_before, self._hidden_after = self._hidden(links, joints), self._hidden(*changed)
        self._observation = wp.to_torch(self.batch.observation)
        self._truth = wp.to_torch(self.batch.truth)
        self._metrics = wp.to_torch(self.batch.metrics)
        self.tick = 0
        self._previous = torch.zeros((self.worlds, self.n), device=self.torch_device)
        self._previous_smooth = torch.zeros((self.worlds, self.n), device=self.torch_device)
        zeros = torch.zeros(self.worlds, device=self.torch_device)
        self._sum_error, self._sum_rough, self._return = zeros.clone(), zeros.clone(), zeros.clone()
        self._window_error = torch.zeros((self.worlds, self.n), device=self.torch_device)
        self._window_speed = torch.zeros((self.worlds, self.n), device=self.torch_device)
        return self._observe()

    def goal(self) -> Tensor:
        return torch.where((self.tick >= self._regoal)[:, None], self._goals[1], self._goals[0])

    def _observe(self) -> Tensor:
        raw, n = self._observation, self.n
        goal = self.goal()
        self.measured_q, self.velocity = raw[:, :n].clone(), raw[:, n:2 * n].clone()
        return torch.cat((self.measured_q / torch.pi, self.velocity / 5.0, raw[:, 2 * n:3 * n] / self._torque,
                          goal / torch.pi, goal - self.measured_q, self._previous), dim=1)

    def limb_observation(self, observation: Tensor, limb: int) -> Tensor:
        """The single-arm observation layout for one limb, so a single-arm controller can drive it."""

        n, k = self.n, JOINTS_PER_LIMB
        pieces = [observation[..., block * n + limb * k: block * n + limb * k + k] for block in range(6)]
        return torch.cat(pieces, dim=-1)

    def merge_limb_actions(self, actions: list[Tensor]) -> Tensor:
        return torch.cat(actions, dim=-1)

    def privileged(self, observation: Tensor) -> Tensor:
        n = self.n
        hidden = torch.where((self.tick >= self._change_tick)[:, None], self._hidden_after, self._hidden_before)
        return torch.cat((observation, self._truth[:, :n] / torch.pi, self._truth[:, n:] / 5.0, hidden), dim=1)

    def step(self, action: Tensor, reference: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """`reference` is the policy's mean command when `action` is an exploration sample: roughness is charged
        on the mean, so exploration noise is not punished and the policy is not pushed to stop exploring."""

        n = self.n
        action = action.clamp(-1.0, 1.0)
        smooth = action if reference is None else reference.clamp(-1.0, 1.0)
        self.batch.step(action)
        self.tick += 1
        error = (self.goal() - (self._truth[:, :n] + self._bias)).abs()
        speed = self._truth[:, n:].abs()
        rough = (smooth - self._previous_smooth).square().sum(dim=1)
        self._previous_smooth = smooth
        limit = self._metrics[:, 2 + n:2 + 2 * n].sum(dim=1)
        arrived = ((error < self.reward_tolerance).all(dim=1) & (speed.amax(dim=1) < SETTLED_SPEED)).float()
        close = torch.exp(-error / self.reward_tolerance).mean(dim=1)
        # Same shape of reward as the arm, per joint pair so a longer chain is not penalized more per joint.
        scale = JOINTS_PER_LIMB / n
        reward = 0.1 * (-error.sum(dim=1) * scale + close + arrived - self.roughness_weight * rough * scale
                        - 0.002 * self._metrics[:, 0] / 0.01 * scale - limit * scale)
        if self.still_weight:
            reward = reward + 0.1 * self.still_weight * (close * torch.exp(-speed.amax(dim=1) / (2.0 * SETTLED_SPEED))
                                                         - 0.2 * speed.mean(dim=1))
        self._previous = action
        self._sum_error += error.mean(dim=1)
        self._sum_rough += rough * scale
        self._return += reward
        if self.tick > self.episode_ticks - FINAL_WINDOW:
            self._window_error = torch.maximum(self._window_error, error)
            self._window_speed = torch.maximum(self._window_speed, speed)
        return self._observe(), reward

    def _current(self, before: np.ndarray, after: np.ndarray, device: torch.device) -> dict[str, Tensor]:
        """Every field of a structured parameter array as a tensor, after the mid-episode change where it happened."""

        # Teachers ask every tick: convert each population once, not once per tick.
        cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, tuple[Tensor, Tensor]]]] = self.__dict__.setdefault("_tensors", {})
        key = id(before)
        if key not in cache or cache[key][0] is not before or cache[key][1] is not after:
            cache[key] = (before, after, {name: tuple(  # type: ignore[misc]
                torch.as_tensor(np.ascontiguousarray(values[name]), dtype=torch.float32, device=device) for values in (before, after))
                for name in before.dtype.names or ()})
        changed = torch.as_tensor(self.tick >= self.change_tick, device=device)
        return {name: torch.where(changed.reshape(-1, *([1] * (a.dim() - 1))), b, a) for name, (a, b) in cache[key][2].items()}

    def true_dynamics(self, q: Tensor, dq: Tensor) -> tuple[Tensor, Tensor]:
        """Mass matrix and bias torques of the robots as they really are now; for teachers, never for policies."""

        from ..control.computed_torque import chain_dynamics

        links = self._current(self.links, self.changed[0], q.device)
        return chain_dynamics(q, dq, links["length"], links["mass"], links["com"], links["inertia"])

    def true_actuation(self, device: torch.device) -> tuple[Tensor, Tensor]:
        """True torque per unit command and viscous damping per joint, now."""

        joints = self._current(self.joints, self.changed[1], device)
        return joints["torque_scale"], joints["damping"]

    def summary(self) -> ChainSummary:
        ok = (self._window_error < TOLERANCE) & (self._window_speed < SETTLED_SPEED)  # [worlds, n]
        limb_ok = ok.reshape(self.worlds, self.limbs, self.limb_joints).all(dim=2)
        ticks = float(self.tick)
        return ChainSummary(ok.all(dim=1), limb_ok, self._window_error.amax(dim=1), self._sum_error / ticks,
                            self._sum_rough / ticks, self._return)
