"""Fused GPU simulation of many different, imperfect planar N-joint chains.

The two-joint arm in `arm_batch.py` is a closed-form special case. This kernel handles any number
of revolute joints in one vertical plane, so two arms stacked one on the other (a four-joint
chain) or a five-joint arm share the same code. Every joint keeps the full per-joint defect
model; every link has its own length, mass and centre of mass.

Dynamics per physics substep: the mass matrix from link Jacobians, gravity and centripetal bias
torques by the same Jacobians, one small dense solve (viscous damping implicit), then dry
friction as bounded impulses through the explicit inverse mass matrix.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np
import warp as wp

from .joint_model import (
    RING_DEPTH,
    JointParams,
    JointState,
    at_limit,
    clamp_to_limits,
    current_reading,
    drive_joint,
    encoder_reading,
    friction_bound,
)

MAX_JOINTS = 8
METRIC_DIM = 2  # effort, friction heat; per-joint peak load and limit time follow
VELOCITY_FILTER = 0.5
FRICTION_ITERATIONS = 4


@wp.struct
class LinkParams:
    """Hidden true rigid-body parameters of one link."""

    length: float  # joint to next joint, m
    mass: float  # including any payload carried at the tip
    com: float  # joint to centre of mass, m
    inertia: float  # about the centre of mass, kg m^2


@wp.func
def _delayed_command(ring: wp.array3d(dtype=wp.float32), w: int, j: int, tick: int, substep: int, substeps: int,
                     delay_steps: int) -> float:
    step = tick * substeps + substep - delay_steps
    if step < 0:
        return 0.0
    return ring[w, j, (step // substeps) % RING_DEPTH]


@wp.kernel
def _step_tick(
    n: int,
    links: wp.array2d(dtype=LinkParams),
    changed_links: wp.array2d(dtype=LinkParams),
    change_tick: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.float32),
    params: wp.array2d(dtype=JointParams),
    changed_params: wp.array2d(dtype=JointParams),
    push: wp.array3d(dtype=wp.float32),
    states: wp.array2d(dtype=JointState),
    commands: wp.array2d(dtype=wp.float32),
    command_ring: wp.array3d(dtype=wp.float32),
    encoder_ring: wp.array3d(dtype=wp.float32),
    ticks: wp.array(dtype=wp.int32),
    rng: wp.array(dtype=wp.uint32),
    mass: wp.array3d(dtype=wp.float32),
    inverse: wp.array3d(dtype=wp.float32),
    scratch: wp.array3d(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
    truth: wp.array2d(dtype=wp.float32),
    metrics: wp.array2d(dtype=wp.float32),
    dt: float,
    substeps: int,
):
    w = wp.tid()
    tick = ticks[w]
    changed = tick >= change_tick[w]
    g = gravity[w]
    for j in range(n):
        command_ring[w, j, tick % RING_DEPTH] = wp.clamp(commands[w, j], -1.0, 1.0)
        metrics[w, 2 + j] = 0.0
        metrics[w, 2 + n + j] = 0.0
        scratch[w, 7, j] = 0.0  # mean motor torque accumulator for the current sensor
    effort = float(0.0)
    heat = float(0.0)

    for substep in range(substeps):
        # Motors and transmissions, per joint.
        for j in range(n):
            p = params[w, j]
            if changed:
                p = changed_params[w, j]
            s = states[w, j]
            s = drive_joint(p, s, _delayed_command(command_ring, w, j, tick, substep, substeps, p.delay_steps), dt)
            states[w, j] = s
            scratch[w, 7, j] = scratch[w, 7, j] + s.motor_torque

        # Kinematics: absolute angles, angular rates, joint origins, centres of mass.
        # scratch rows: 0 theta, 1 omega, 2 origin x, 3 origin z, 4 com x, 5 com z, 6 centripetal-acceleration
        theta = float(0.0)
        omega = float(0.0)
        ox = float(0.0)
        oz = float(0.0)
        for j in range(n):
            link = links[w, j]
            if changed:
                link = changed_links[w, j]
            s = states[w, j]
            theta = theta + s.q
            omega = omega + s.dq
            scratch[w, 0, j] = theta
            scratch[w, 1, j] = omega
            scratch[w, 2, j] = ox
            scratch[w, 3, j] = oz
            scratch[w, 4, j] = ox + link.com * wp.sin(theta)
            scratch[w, 5, j] = oz - link.com * wp.cos(theta)
            ox = ox + link.length * wp.sin(theta)
            oz = oz - link.length * wp.cos(theta)

        # Mass matrix M_ij = sum over links k >= max(i, j) of m_k J_ki . J_kj + I_k, with
        # J_ki = d(com_k)/d(q_i) = (-(com_k - o_i)_z, (com_k - o_i)_x). Bias torque per joint i:
        # gravity  -sum_k m_k g (com_k - o_i)_x  and centripetal  sum_k m_k J_ki . a_k, where a_k is the
        # centre-of-mass acceleration at zero joint acceleration.
        for i in range(n):
            pi = params[w, i]
            if changed:
                pi = changed_params[w, i]
            bias = float(0.0)
            for j in range(n):
                mass[w, i, j] = 0.0
            for k in range(i, n):
                link = links[w, k]
                if changed:
                    link = changed_links[w, k]
                rx = scratch[w, 4, k] - scratch[w, 2, i]
                rz = scratch[w, 5, k] - scratch[w, 3, i]
                jx = -rz
                jz = rx
                # Centripetal acceleration of com_k: each earlier link j < k contributes -L_j w_j^2 u(theta_j),
                # the link itself -com_k w_k^2 u(theta_k).
                ax = float(0.0)
                az = float(0.0)
                for j in range(k):
                    lj = links[w, j]
                    if changed:
                        lj = changed_links[w, j]
                    wj = scratch[w, 1, j]
                    tj = scratch[w, 0, j]
                    ax = ax - lj.length * wj * wj * wp.sin(tj)
                    az = az + lj.length * wj * wj * wp.cos(tj)
                wk = scratch[w, 1, k]
                tk = scratch[w, 0, k]
                ax = ax - link.com * wk * wk * wp.sin(tk)
                az = az + link.com * wk * wk * wp.cos(tk)
                bias = bias + link.mass * (jx * ax + jz * az) + link.mass * g * rx
                for j in range(i + 1):
                    qx = scratch[w, 4, k] - scratch[w, 2, j]
                    qz = scratch[w, 5, k] - scratch[w, 3, j]
                    value = link.mass * (jx * (-qz) + jz * qx) + link.inertia
                    mass[w, i, j] = mass[w, i, j] + value
                    if j != i:
                        mass[w, j, i] = mass[w, j, i] + value
            si = states[w, i]
            # Right-hand side: transmission torque, external push, minus bias, viscous damping implicit.
            now = float(tick * substeps + substep) * dt
            pushed = push[w, i, 0] * wp.sin(6.2831853 * push[w, i, 1] * now + push[w, i, 2])
            pushed = pushed + push[w, i, 3] * wp.sin(6.2831853 * push[w, i, 4] * now + push[w, i, 5])
            scratch[w, 6, i] = si.tau_out + pushed - bias - pi.damping * si.dq
            mass[w, i, i] = mass[w, i, i] + dt * pi.damping

        # Invert M by Gauss-Jordan (small, symmetric positive definite; no pivoting needed).
        for i in range(n):
            for j in range(n):
                if i == j:
                    inverse[w, i, j] = 1.0
                else:
                    inverse[w, i, j] = 0.0
        for c in range(n):
            pivot = 1.0 / mass[w, c, c]
            for j in range(n):
                mass[w, c, j] = mass[w, c, j] * pivot
                inverse[w, c, j] = inverse[w, c, j] * pivot
            for r in range(n):
                if r != c:
                    factor = mass[w, r, c]
                    for j in range(n):
                        mass[w, r, j] = mass[w, r, j] - factor * mass[w, c, j]
                        inverse[w, r, j] = inverse[w, r, j] - factor * inverse[w, c, j]

        # Unconstrained velocity update, stored in scratch row 0 (theta no longer needed).
        for i in range(n):
            acc = float(0.0)
            for j in range(n):
                acc = acc + inverse[w, i, j] * scratch[w, 6, j]
            scratch[w, 0, i] = states[w, i].dq + dt * acc

        # Dry friction as bounded impulses, projected Gauss-Seidel through the inverse mass matrix.
        for j in range(n):
            scratch[w, 1, j] = 0.0  # impulse so far
        for _ in range(FRICTION_ITERATIONS):
            for i in range(n):
                p = params[w, i]
                if changed:
                    p = changed_params[w, i]
                s = states[w, i]
                bound = friction_bound(p, s.q, s.dq) * dt
                old = scratch[w, 1, i]
                updated = wp.clamp(old - scratch[w, 0, i] / inverse[w, i, i], -bound, bound)
                delta = updated - old
                for j in range(n):
                    scratch[w, 0, j] = scratch[w, 0, j] + inverse[w, j, i] * delta
                scratch[w, 1, i] = updated
        for i in range(n):
            heat = heat - scratch[w, 1, i] * scratch[w, 0, i]

        # Integrate, hard stops, per-joint metrics.
        for i in range(n):
            p = params[w, i]
            if changed:
                p = changed_params[w, i]
            s = states[w, i]
            s.dq = scratch[w, 0, i]
            s.q = s.q + dt * s.dq
            s = clamp_to_limits(p, s)
            states[w, i] = s
            effort = effort + dt * s.tau_out * s.tau_out
            metrics[w, 2 + i] = wp.max(metrics[w, 2 + i], wp.abs(s.tau_out))
            metrics[w, 2 + n + i] = metrics[w, 2 + n + i] + at_limit(p, s) / float(substeps)

    # Sensors sample once per control tick; encoder packets arrive `enc_delay_ticks` late.
    tick = tick + 1
    state = rng[w]
    control_dt = dt * float(substeps)
    for j in range(n):
        p = params[w, j]
        if changed:
            p = changed_params[w, j]
        s = states[w, j]
        encoder_ring[w, j, tick % RING_DEPTH] = encoder_reading(p, s.q, wp.randn(state))
        delivered = encoder_ring[w, j, wp.max(tick - p.enc_delay_ticks, 0) % RING_DEPTH]
        s.vel_est = (1.0 - VELOCITY_FILTER) * s.vel_est + VELOCITY_FILTER * (delivered - s.enc_prev) / control_dt
        s.enc_prev = delivered
        states[w, j] = s
        observation[w, j] = delivered
        observation[w, n + j] = s.vel_est
        observation[w, 2 * n + j] = current_reading(p, scratch[w, 7, j] / float(substeps), wp.randn(state))
        truth[w, j] = s.q
        truth[w, n + j] = s.dq
    rng[w] = state
    metrics[w, 0] = effort
    metrics[w, 1] = heat
    ticks[w] = tick


@wp.kernel
def _reset(
    n: int,
    mask: wp.array(dtype=wp.int32),
    start_q: wp.array2d(dtype=wp.float32),
    params: wp.array2d(dtype=JointParams),
    states: wp.array2d(dtype=JointState),
    command_ring: wp.array3d(dtype=wp.float32),
    encoder_ring: wp.array3d(dtype=wp.float32),
    ticks: wp.array(dtype=wp.int32),
    rng: wp.array(dtype=wp.uint32),
    observation: wp.array2d(dtype=wp.float32),
    truth: wp.array2d(dtype=wp.float32),
    metrics: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    if mask[w] == 0:
        return
    state = rng[w]
    for j in range(n):
        p = params[w, j]
        s = JointState()
        s.q = wp.clamp(start_q[w, j], p.q_min, p.q_max)
        s.qm = s.q
        reading = encoder_reading(p, s.q, wp.randn(state))
        s.enc_prev = reading
        for k in range(RING_DEPTH):
            command_ring[w, j, k] = 0.0
            encoder_ring[w, j, k] = reading
        states[w, j] = s
        observation[w, j] = reading
        observation[w, n + j] = 0.0
        observation[w, 2 * n + j] = current_reading(p, 0.0, wp.randn(state))
        truth[w, j] = s.q
        truth[w, n + j] = 0.0
        metrics[w, 2 + j] = 0.0
        metrics[w, 2 + n + j] = 0.0
    metrics[w, 0] = 0.0
    metrics[w, 1] = 0.0
    rng[w] = state
    ticks[w] = 0


class ChainBatch:
    """A population of distinct imperfect N-joint chains, stepped one control tick per launch.

    `links` is a structured array [worlds, joints] of `LinkParams`, `joints` one of `JointParams`.
    Observation layout: measured q (n), derived dq (n), motor current (n). Truth: q (n), dq (n).
    Metrics: effort, friction heat, then per-joint peak transmission torque (n) and hard-stop time (n).
    """

    def __init__(self, links: np.ndarray, joints: np.ndarray, *, device: str = "cuda:0", physics_dt: float = 0.001,
                 substeps: int = 10, seed: int = 0, gravity: float | np.ndarray = 9.81,
                 changed: tuple[np.ndarray, np.ndarray] | None = None, change_tick: np.ndarray | None = None,
                 push: np.ndarray | None = None) -> None:
        if links.ndim != 2 or joints.shape != links.shape:
            raise ValueError("expected links and joints as [worlds, joints]")
        self.worlds, self.joints = links.shape
        if not 1 <= self.joints <= MAX_JOINTS:
            raise ValueError(f"joint count must be in 1..{MAX_JOINTS}")
        if int(joints["delay_steps"].max()) > (RING_DEPTH - 1) * substeps:
            raise ValueError("command latency exceeds the command ring")
        if int(joints["enc_delay_ticks"].max()) > RING_DEPTH - 1:
            raise ValueError("encoder delay exceeds the encoder ring")
        self.device, self.physics_dt, self.substeps = device, physics_dt, substeps
        from ..config import project_root

        wp.config.kernel_cache_dir = str(project_root() / ".cache" / "warp")
        wp.init()
        n, w = self.joints, self.worlds
        self.links = wp.array(links, dtype=LinkParams, device=device)
        self.params = wp.array(joints, dtype=JointParams, device=device)
        self.changed_links = self.links if changed is None else wp.array(changed[0], dtype=LinkParams, device=device)
        self.changed_params = self.params if changed is None else wp.array(changed[1], dtype=JointParams, device=device)
        never = np.full(w, np.iinfo(np.int32).max, dtype=np.int32)
        self.change_tick = wp.array(never if changed is None or change_tick is None else np.asarray(change_tick, dtype=np.int32),
                                    dtype=wp.int32, device=device)
        self.gravity = wp.array(np.broadcast_to(np.asarray(gravity, dtype=np.float32), (w,)).copy(), dtype=wp.float32, device=device)
        waves = np.zeros((w, n, 6), dtype=np.float32) if push is None else np.asarray(push, dtype=np.float32)
        if waves.shape != (w, n, 6):
            raise ValueError("push must have shape [worlds, joints, 6]")
        self.push = wp.array(waves, dtype=wp.float32, device=device)
        self.states = wp.zeros((w, n), dtype=JointState, device=device)
        self.command_ring = wp.zeros((w, n, RING_DEPTH), dtype=wp.float32, device=device)
        self.encoder_ring = wp.zeros((w, n, RING_DEPTH), dtype=wp.float32, device=device)
        self.ticks = wp.zeros(w, dtype=wp.int32, device=device)
        streams = np.random.default_rng(seed).integers(1, 2**32, w, dtype=np.uint32)
        self.rng = wp.array(streams, dtype=wp.uint32, device=device)
        self.mass = wp.zeros((w, n, n), dtype=wp.float32, device=device)
        self.inverse = wp.zeros((w, n, n), dtype=wp.float32, device=device)
        self.scratch = wp.zeros((w, 8, n), dtype=wp.float32, device=device)
        self.observation = wp.zeros((w, 3 * n), dtype=wp.float32, device=device)
        self.truth = wp.zeros((w, 2 * n), dtype=wp.float32, device=device)
        self.metrics = wp.zeros((w, METRIC_DIM + 2 * n), dtype=wp.float32, device=device)
        self._shared = [self.params, self.states, self.command_ring, self.encoder_ring, self.ticks, self.rng,
                        self.observation, self.truth, self.metrics]

    def reset(self, start_q: np.ndarray, mask: np.ndarray | None = None) -> None:
        mask_values = np.ones(self.worlds, dtype=np.int32) if mask is None else np.asarray(mask, dtype=np.int32)
        start = np.ascontiguousarray(np.broadcast_to(np.asarray(start_q, dtype=np.float32), (self.worlds, self.joints)))
        wp.launch(_reset, dim=self.worlds, device=self.device,
                  inputs=[self.joints, wp.array(mask_values, device=self.device), wp.array(start, device=self.device), *self._shared])

    def step(self, commands: object) -> None:
        if isinstance(commands, np.ndarray):
            source = wp.array(np.ascontiguousarray(commands, dtype=np.float32), device=self.device)
        else:
            source = wp.from_torch(commands.detach().contiguous())  # type: ignore[attr-defined]
        if source.shape != (self.worlds, self.joints):
            raise ValueError("commands must have shape [worlds, joints]")
        wp.launch(_step_tick, dim=self.worlds, device=self.device,
                  inputs=[self.joints, self.links, self.changed_links, self.change_tick, self.gravity, self.params,
                          self.changed_params, self.push, self.states, source, self.command_ring, self.encoder_ring,
                          self.ticks, self.rng, self.mass, self.inverse, self.scratch, self.observation, self.truth,
                          self.metrics, self.physics_dt, self.substeps])

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)
