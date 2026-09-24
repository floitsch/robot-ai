"""Computed-torque control of a planar chain, with perfect knowledge, as a classical teacher.

Given the true state and the true link parameters, inverse dynamics cancels gravity, Coriolis
and inertia coupling exactly, and a PD law on a minimum-jerk reference does the rest. On a
healthy chain this is close to the best a 100 Hz torque controller can do; it knows nothing
about slack, stiction or weak motors, so on defective chains it is only a baseline. It reads
simulator truth and cannot be deployed. Its role is to seed learning: a network distilled from
it starts from smooth, competent motion instead of random flailing.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import torch
from torch import Tensor, nn

from ..sim.chain_env import ChainEnv


def chain_dynamics(q: Tensor, dq: Tensor, length: Tensor, mass: Tensor, com: Tensor, inertia: Tensor,
                   gravity: float = 9.81) -> tuple[Tensor, Tensor]:
    """Mass matrix [W, n, n] and bias torque [W, n] (Coriolis plus gravity) of planar chains.

    M ddq = tau - bias. Same formulation as the simulation kernel: absolute link angles from cumulative joint
    angles, link Jacobians J_ki = (-(c_k - o_i)_z, (c_k - o_i)_x), centripetal accelerations at zero joint
    acceleration.
    """

    n = q.shape[1]
    theta = torch.cumsum(q, dim=1)
    omega = torch.cumsum(dq, dim=1)
    sin, cos = torch.sin(theta), torch.cos(theta)
    step = torch.stack((length * sin, -length * cos), dim=-1)  # joint-to-joint vectors [W, n, 2]
    origin = torch.cumsum(step, dim=1) - step  # origin of joint k [W, n, 2]
    centre = origin + torch.stack((com * sin, -com * cos), dim=-1)
    # Centripetal acceleration of each centre of mass with ddq = 0.
    turn = omega.square()
    link_acc = -torch.stack((length * turn * sin, -length * turn * cos), dim=-1)  # from link k's own rotation, for later links
    own_acc = -torch.stack((com * turn * sin, -com * turn * cos), dim=-1)
    acc = torch.cumsum(link_acc, dim=1) - link_acc + own_acc  # [W, n, 2]
    # r[w, k, i] = centre_k - origin_i for k >= i (masked otherwise).
    r = centre[:, :, None, :] - origin[:, None, :, :]  # [W, k, i, 2]
    jac = torch.stack((-r[..., 1], r[..., 0]), dim=-1)  # [W, k, i, 2]
    after = (torch.arange(n)[:, None] >= torch.arange(n)[None, :]).to(q.device)  # k >= i
    jac = jac * after[None, :, :, None]
    # M_ij = sum_k m_k J_ki . J_kj + I_k (k >= max(i, j)).
    weighted = jac * mass[:, :, None, None]
    inertia_terms = torch.cumsum(inertia.flip(1), dim=1).flip(1)  # sum_{k >= i} I_k
    idx = torch.maximum(torch.arange(n)[:, None], torch.arange(n)[None, :]).to(q.device)
    m = torch.einsum("wkid,wkjd->wij", weighted, jac) + inertia_terms[:, idx]
    # bias_i = sum_k m_k J_ki . a_k + sum_k m_k g (c_k - o_i)_x.
    bias = torch.einsum("wkid,wkd->wi", weighted, acc) + torch.einsum("wki,wk->wi", r[..., 0] * after[None], mass) * gravity
    return m, bias


def minimum_jerk(phase: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Position, velocity and acceleration profiles of a unit move over phase in [0, 1]."""

    p = phase.clamp(0.0, 1.0)
    pos = p**3 * (10.0 - 15.0 * p + 6.0 * p**2)
    vel = 30.0 * p**2 * (1.0 - p) ** 2
    acc = 60.0 * p * (1.0 - 3.0 * p + 2.0 * p**2)
    return pos, vel, acc


class ComputedTorqueTeacher(nn.Module):
    """Presents the oracle interface used by distillation, but reads the environment's truth directly."""

    def __init__(self, env: ChainEnv, *, omega: float = 15.0, seconds_per_rad: float = 0.5, measured: bool = False,
                 ramp: bool = True, friction: bool = False) -> None:
        super().__init__()
        self.env = env
        self.kp, self.kd = omega * omega, 2.0 * omega
        self.seconds_per_rad = seconds_per_rad
        # With `measured`, feedback uses the encoder angle and its lagged velocity estimate, as a student would, so the
        # taught law is one that is stable on the student's own inputs; dynamics still use the true state and parameters.
        self.measured = measured
        # Without the ramp the law is memoryless (PD straight toward the goal): a pure function of the current state,
        # which a student can imitate without reconstructing when the goal changed.
        self.ramp = ramp
        # Feed forward the known dry friction (Coulomb, breakaway bump and angle-local rubbing) against the intended
        # direction of motion; the simulator's bound is the same expression as its `friction_bound`.
        self.friction = friction
        self.n = env.n
        self.observation_dim = env.observation_dim
        self.privileged_dim = torch.tensor(env.privileged_dim)
        self.register_buffer("oracle", torch.tensor(1))
        self.register_buffer("incremental", torch.tensor(False))
        self.insight = None
        self.hidden = 0

    def initial(self, worlds: int, device: torch.device) -> Tensor:
        # Reference state per world: goal seen last (n), move origin (n), elapsed (1), duration (1); NaN goal = unset.
        state = torch.zeros((worlds, 2 * self.n + 2), device=device)
        state[:, :self.n] = torch.nan
        return state

    def _friction_bound(self, q: Tensor, dq: Tensor, device: torch.device) -> Tensor:
        env = self.env
        changed = torch.as_tensor(env.tick >= env.change_tick, device=device)[:, None]
        j, cj = env.joints, env.changed[1]
        field = lambda name: torch.where(changed, torch.as_tensor(cj[name], dtype=torch.float32, device=device),
                                         torch.as_tensor(j[name], dtype=torch.float32, device=device))
        rub = field("bump0_mag") * torch.exp(-0.5 * ((q - field("bump0_center")) / field("bump0_width")) ** 2)
        rub = rub + field("bump1_mag") * torch.exp(-0.5 * ((q - field("bump1_center")) / field("bump1_width")) ** 2)
        return field("coulomb") * (1.0 + field("stribeck") * torch.exp(-(dq / 0.05) ** 2)) + rub

    def forward(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor]:
        env, n = self.env, self.n
        device = feeling.device
        goal = env.joint_goal() if hasattr(env, "joint_goal") else env.goal()  # tool-pose tasks: the rigid arm's IK
        q_true, dq_true = env._truth[:, :n], env._truth[:, n:]
        bias = env._bias
        q = q_true + bias  # the controller reasons in encoder units, where the goal is given
        fb_q, fb_dq = (env.measured_q, env.velocity) if self.measured else (q, dq_true)
        seen, origin, elapsed, duration = feeling[:, :n], feeling[:, n:2 * n], feeling[:, 2 * n], feeling[:, 2 * n + 1]
        moved = (goal != seen).any(dim=1) | torch.isnan(seen).any(dim=1)
        origin = torch.where(moved[:, None], q, origin)
        elapsed = torch.where(moved, torch.zeros_like(elapsed), elapsed + 0.01)
        span = (goal - origin).abs().amax(dim=1)
        duration = torch.where(moved, (self.seconds_per_rad * span).clamp(min=0.3), duration)
        pos, vel, acc = minimum_jerk(elapsed / duration)
        delta = goal - origin
        ref = origin + delta * pos[:, None]
        ref_v = delta * vel[:, None] / duration[:, None]
        ref_a = delta * acc[:, None] / duration[:, None] ** 2
        if not self.ramp:
            ref, ref_v, ref_a = goal, torch.zeros_like(goal), torch.zeros_like(goal)
        desired = ref_a + self.kp * (ref - fb_q) + self.kd * (ref_v - fb_dq)
        torque_scale, damping = env.true_actuation(device)
        m, dyn_bias = env.true_dynamics(q_true, dq_true)
        torque = torch.einsum("wij,wj->wi", m, desired) + dyn_bias + damping * dq_true
        if self.friction:
            intent = torch.where(fb_dq.abs() > 0.05, fb_dq, (ref - fb_q) * 2.0)
            torque = torque + self._friction_bound(q_true, dq_true, device) * torch.tanh(intent / 0.05)
        command = (torque / torque_scale).clamp(-1.0, 1.0)
        next_feeling = torch.cat((goal, origin, elapsed[:, None], duration[:, None]), dim=1)
        return command[None], next_feeling

    def distribution(self, observations: Tensor, feeling: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, feeling = self(observations, feeling)
        return mean, torch.zeros_like(mean), feeling


