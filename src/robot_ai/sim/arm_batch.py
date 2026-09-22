"""Fused GPU simulation of many different, imperfect two-joint arms.

One kernel launch advances every world by a whole control tick: actuator lag and
latency, backlash, stick-slip friction, rigid-body dynamics, hard stops, and the
sensor suite. Every world has its own physical parameters, so a batch is a
population of distinct robots rather than copies of one.
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

JOINTS = 2
OBSERVATION_DIM = 3 * JOINTS  # measured q, derived dq, motor current
TRUTH_DIM = 2 * JOINTS  # true q, dq
# Per-tick physical cost terms, see `_METRIC_*`.
METRIC_DIM = 6
_METRIC_EFFORT = 0  # integral of squared output torque, (N m)^2 s
_METRIC_FRICTION_HEAT = 1  # energy dissipated by dry friction, J
_METRIC_PEAK_LOAD = 2  # [2:4] peak |transmission torque| per joint, N m
_METRIC_LIMIT = 4  # [4:6] fraction of the tick spent on a hard stop, per joint
VELOCITY_FILTER = 0.5
FRICTION_ITERATIONS = 3


@wp.struct
class ArmParams:
    """Hidden true rigid-body parameters of one planar two-link arm."""

    l1: float  # link length, m
    l2: float
    m1: float  # link mass including any payload, kg
    m2: float
    lc1: float  # joint-to-centre-of-mass distance, m
    lc2: float
    i1: float  # inertia about the centre of mass, kg m^2
    i2: float
    gravity: float  # m/s^2 along -z; q = 0 hangs straight down


@wp.func
def _delayed_command(ring: wp.vec4, tick: int, substep: int, substeps: int, delay_steps: int) -> float:
    step = tick * substeps + substep - delay_steps
    if step < 0:
        return 0.0
    return ring[(step // substeps) % RING_DEPTH]


@wp.func
def _push(wave: wp.vec3, now: float) -> float:
    """wave = (amplitude N m, frequency Hz, phase rad)."""

    return wave[0] * wp.sin(6.2831853 * wave[1] * now + wave[2])


@wp.kernel
def _step_tick(
    arms: wp.array(dtype=ArmParams),
    changed_arms: wp.array(dtype=ArmParams),
    change_tick: wp.array(dtype=wp.int32),
    params: wp.array2d(dtype=JointParams),
    changed_params: wp.array2d(dtype=JointParams),
    push: wp.array3d(dtype=wp.float32),
    states: wp.array2d(dtype=JointState),
    commands: wp.array2d(dtype=wp.float32),
    command_ring: wp.array3d(dtype=wp.float32),
    encoder_ring: wp.array3d(dtype=wp.float32),
    ticks: wp.array(dtype=wp.int32),
    rng: wp.array(dtype=wp.uint32),
    observation: wp.array2d(dtype=wp.float32),
    truth: wp.array2d(dtype=wp.float32),
    metrics: wp.array2d(dtype=wp.float32),
    dt: float,
    substeps: int,
):
    w = wp.tid()
    arm = arms[w]
    p0 = params[w, 0]
    p1 = params[w, 1]
    s0 = states[w, 0]
    s1 = states[w, 1]
    tick = ticks[w]
    # The robot or its surroundings change mid-episode: payload picked up, motor fading, joint fouling.
    if tick >= change_tick[w]:
        arm = changed_arms[w]
        p0 = changed_params[w, 0]
        p1 = changed_params[w, 1]
    slot = tick % RING_DEPTH
    command_ring[w, 0, slot] = wp.clamp(commands[w, 0], -1.0, 1.0)
    command_ring[w, 1, slot] = wp.clamp(commands[w, 1], -1.0, 1.0)
    ring0 = wp.vec4(command_ring[w, 0, 0], command_ring[w, 0, 1], command_ring[w, 0, 2], command_ring[w, 0, 3])
    ring1 = wp.vec4(command_ring[w, 1, 0], command_ring[w, 1, 1], command_ring[w, 1, 2], command_ring[w, 1, 3])

    # External load nobody commanded, e.g. a neighbouring limb moving: two sinusoids per joint.
    push0a = wp.vec3(push[w, 0, 0], push[w, 0, 1], push[w, 0, 2])
    push0b = wp.vec3(push[w, 0, 3], push[w, 0, 4], push[w, 0, 5])
    push1a = wp.vec3(push[w, 1, 0], push[w, 1, 1], push[w, 1, 2])
    push1b = wp.vec3(push[w, 1, 3], push[w, 1, 4], push[w, 1, 5])

    effort = float(0.0)
    heat = float(0.0)
    peak0 = float(0.0)
    peak1 = float(0.0)
    limit0 = float(0.0)
    limit1 = float(0.0)
    current0 = float(0.0)
    current1 = float(0.0)

    for substep in range(substeps):
        s0 = drive_joint(p0, s0, _delayed_command(ring0, tick, substep, substeps, p0.delay_steps), dt)
        s1 = drive_joint(p1, s1, _delayed_command(ring1, tick, substep, substeps, p1.delay_steps), dt)

        c2 = wp.cos(s1.q)
        q12 = s0.q + s1.q
        m11 = arm.i1 + arm.i2 + arm.m1 * arm.lc1 * arm.lc1 + arm.m2 * (arm.l1 * arm.l1 + arm.lc2 * arm.lc2 + 2.0 * arm.l1 * arm.lc2 * c2)
        m12 = arm.i2 + arm.m2 * (arm.lc2 * arm.lc2 + arm.l1 * arm.lc2 * c2)
        m22 = arm.i2 + arm.m2 * arm.lc2 * arm.lc2
        g2 = -arm.gravity * arm.m2 * arm.lc2 * wp.sin(q12)
        g1 = -arm.gravity * (arm.m1 * arm.lc1 + arm.m2 * arm.l1) * wp.sin(s0.q) + g2
        h = -arm.m2 * arm.l1 * arm.lc2 * wp.sin(s1.q)
        now = float(tick * substeps + substep) * dt
        rhs0 = s0.tau_out + g1 - h * (2.0 * s0.dq * s1.dq + s1.dq * s1.dq) - p0.damping * s0.dq
        rhs1 = s1.tau_out + g2 + h * s0.dq * s0.dq - p1.damping * s1.dq
        rhs0 = rhs0 + _push(push0a, now) + _push(push0b, now)
        rhs1 = rhs1 + _push(push1a, now) + _push(push1b, now)
        # Viscous damping is integrated implicitly, as MuJoCo's Euler integrator does.
        m11 = m11 + dt * p0.damping
        m22 = m22 + dt * p1.damping
        inverse_det = 1.0 / (m11 * m22 - m12 * m12)
        a00 = m22 * inverse_det
        a01 = -m12 * inverse_det
        a11 = m11 * inverse_det
        dq0 = s0.dq + dt * (a00 * rhs0 + a01 * rhs1)
        dq1 = s1.dq + dt * (a01 * rhs0 + a11 * rhs1)

        # Dry friction as a bounded impulse (projected Gauss-Seidel). Unlike a
        # smoothed sign function this is stable at any stiffness and lets a joint
        # truly stick until the applied torque exceeds the breakaway bound.
        bound0 = friction_bound(p0, s0.q, s0.dq) * dt
        bound1 = friction_bound(p1, s1.q, s1.dq) * dt
        impulse0 = float(0.0)
        impulse1 = float(0.0)
        for _ in range(FRICTION_ITERATIONS):
            updated0 = wp.clamp(impulse0 - dq0 / a00, -bound0, bound0)
            dq0 = dq0 + a00 * (updated0 - impulse0)
            dq1 = dq1 + a01 * (updated0 - impulse0)
            impulse0 = updated0
            updated1 = wp.clamp(impulse1 - dq1 / a11, -bound1, bound1)
            dq0 = dq0 + a01 * (updated1 - impulse1)
            dq1 = dq1 + a11 * (updated1 - impulse1)
            impulse1 = updated1
        heat = heat - impulse0 * dq0 - impulse1 * dq1

        s0.dq = dq0
        s1.dq = dq1
        s0.q = s0.q + dt * dq0
        s1.q = s1.q + dt * dq1
        s0 = clamp_to_limits(p0, s0)
        s1 = clamp_to_limits(p1, s1)

        effort = effort + dt * (s0.tau_out * s0.tau_out + s1.tau_out * s1.tau_out)
        peak0 = wp.max(peak0, wp.abs(s0.tau_out))
        peak1 = wp.max(peak1, wp.abs(s1.tau_out))
        limit0 = limit0 + at_limit(p0, s0)
        limit1 = limit1 + at_limit(p1, s1)
        current0 = current0 + s0.motor_torque
        current1 = current1 + s1.motor_torque

    # Sensors sample once per control tick; encoder packets arrive `enc_delay_ticks` late.
    tick = tick + 1
    slot = tick % RING_DEPTH
    state = rng[w]
    encoder_ring[w, 0, slot] = encoder_reading(p0, s0.q, wp.randn(state))
    encoder_ring[w, 1, slot] = encoder_reading(p1, s1.q, wp.randn(state))
    delivered0 = encoder_ring[w, 0, wp.max(tick - p0.enc_delay_ticks, 0) % RING_DEPTH]
    delivered1 = encoder_ring[w, 1, wp.max(tick - p1.enc_delay_ticks, 0) % RING_DEPTH]
    control_dt = dt * float(substeps)
    s0.vel_est = (1.0 - VELOCITY_FILTER) * s0.vel_est + VELOCITY_FILTER * (delivered0 - s0.enc_prev) / control_dt
    s1.vel_est = (1.0 - VELOCITY_FILTER) * s1.vel_est + VELOCITY_FILTER * (delivered1 - s1.enc_prev) / control_dt
    s0.enc_prev = delivered0
    s1.enc_prev = delivered1
    scale = 1.0 / float(substeps)
    observation[w, 0] = delivered0
    observation[w, 1] = delivered1
    observation[w, 2] = s0.vel_est
    observation[w, 3] = s1.vel_est
    observation[w, 4] = current_reading(p0, current0 * scale, wp.randn(state))
    observation[w, 5] = current_reading(p1, current1 * scale, wp.randn(state))
    rng[w] = state

    truth[w, 0] = s0.q
    truth[w, 1] = s1.q
    truth[w, 2] = s0.dq
    truth[w, 3] = s1.dq
    metrics[w, 0] = effort
    metrics[w, 1] = heat
    metrics[w, 2] = peak0
    metrics[w, 3] = peak1
    metrics[w, 4] = limit0 * scale
    metrics[w, 5] = limit1 * scale
    states[w, 0] = s0
    states[w, 1] = s1
    ticks[w] = tick


@wp.kernel
def _reset(
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
    for j in range(JOINTS):
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
        observation[w, JOINTS + j] = 0.0
        observation[w, 2 * JOINTS + j] = current_reading(p, 0.0, wp.randn(state))
        truth[w, j] = s.q
        truth[w, JOINTS + j] = 0.0
    for m in range(METRIC_DIM):
        metrics[w, m] = 0.0
    rng[w] = state
    ticks[w] = 0


class ArmBatch:
    """A population of distinct imperfect arms, stepped one control tick per launch."""

    def __init__(self, arms: np.ndarray, joints: np.ndarray, *, device: str = "cuda:0",
                 physics_dt: float = 0.001, substeps: int = 10, seed: int = 0,
                 changed: tuple[np.ndarray, np.ndarray] | None = None, change_tick: np.ndarray | None = None,
                 push: np.ndarray | None = None) -> None:
        if arms.ndim != 1 or joints.shape != (len(arms), JOINTS):
            raise ValueError("expected arms [worlds] and joints [worlds, 2]")
        if int(joints["delay_steps"].max()) > (RING_DEPTH - 1) * substeps:
            raise ValueError("command latency exceeds the command ring")
        if int(joints["enc_delay_ticks"].max()) > RING_DEPTH - 1:
            raise ValueError("encoder delay exceeds the encoder ring")
        self.worlds = len(arms)
        self.device = device
        self.physics_dt = physics_dt
        self.substeps = substeps
        from ..config import project_root

        wp.config.kernel_cache_dir = str(project_root() / ".cache" / "warp")
        wp.init()
        self.arms = wp.array(arms, dtype=ArmParams, device=device)
        self.params = wp.array(joints, dtype=JointParams, device=device)
        self.changed_arms = self.arms if changed is None else wp.array(changed[0], dtype=ArmParams, device=device)
        self.changed_params = self.params if changed is None else wp.array(changed[1], dtype=JointParams, device=device)
        waves = np.zeros((self.worlds, JOINTS, 6), dtype=np.float32) if push is None else np.asarray(push, dtype=np.float32)
        if waves.shape != (self.worlds, JOINTS, 6):
            raise ValueError("push must have shape [worlds, 2, 6]: two (amplitude, frequency, phase) waves per joint")
        self.push = wp.array(waves, dtype=wp.float32, device=device)
        never = np.full(self.worlds, np.iinfo(np.int32).max, dtype=np.int32)
        self.change_tick = wp.array(never if changed is None or change_tick is None else np.asarray(change_tick, dtype=np.int32),
                                    dtype=wp.int32, device=device)
        self.states = wp.zeros((self.worlds, JOINTS), dtype=JointState, device=device)
        self.command_ring = wp.zeros((self.worlds, JOINTS, RING_DEPTH), dtype=wp.float32, device=device)
        self.encoder_ring = wp.zeros((self.worlds, JOINTS, RING_DEPTH), dtype=wp.float32, device=device)
        self.ticks = wp.zeros(self.worlds, dtype=wp.int32, device=device)
        streams = np.random.default_rng(seed).integers(1, 2**32, self.worlds, dtype=np.uint32)
        self.rng = wp.array(streams, dtype=wp.uint32, device=device)
        self.observation = wp.zeros((self.worlds, OBSERVATION_DIM), dtype=wp.float32, device=device)
        self.truth = wp.zeros((self.worlds, TRUTH_DIM), dtype=wp.float32, device=device)
        self.metrics = wp.zeros((self.worlds, METRIC_DIM), dtype=wp.float32, device=device)
        self._shared = [self.params, self.states, self.command_ring, self.encoder_ring, self.ticks, self.rng,
                        self.observation, self.truth, self.metrics]

    def reset(self, start_q: np.ndarray, mask: np.ndarray | None = None) -> None:
        mask_values = np.ones(self.worlds, dtype=np.int32) if mask is None else np.asarray(mask, dtype=np.int32)
        start = np.ascontiguousarray(np.broadcast_to(np.asarray(start_q, dtype=np.float32), (self.worlds, JOINTS)))
        wp.launch(_reset, dim=self.worlds, device=self.device,
                  inputs=[wp.array(mask_values, device=self.device), wp.array(start, device=self.device), *self._shared])

    def step(self, commands: object) -> None:
        """Advance one control tick. Accepts a Torch tensor on this device or a NumPy array."""

        if isinstance(commands, np.ndarray):
            source = wp.array(np.ascontiguousarray(commands, dtype=np.float32), device=self.device)
        else:
            source = wp.from_torch(commands.detach().contiguous())  # type: ignore[attr-defined]
        if source.shape != (self.worlds, JOINTS):
            raise ValueError("commands must have shape [worlds, 2]")
        wp.launch(_step_tick, dim=self.worlds, device=self.device,
                  inputs=[self.arms, self.changed_arms, self.change_tick, self.params, self.changed_params,
                          self.push, self.states, source, self.command_ring, self.encoder_ring,
                          self.ticks, self.rng, self.observation, self.truth, self.metrics,
                          self.physics_dt, self.substeps])

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)
