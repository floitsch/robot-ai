"""Per-joint actuator, transmission, friction, and sensor models for GPU kernels.

Everything here is indexed [world, joint] and knows nothing about the mechanism
the joints belong to. A dynamics kernel (the planar arm today; a walker or an
external engine later) calls these functions to turn commands into joint torques
and true joint state into imperfect measurements.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import warp as wp

# Velocity scale of the Stribeck (breakaway) friction bump, rad/s.
STRIBECK_VELOCITY = 0.05
# Depth of the command and encoder delay rings, in control ticks.
RING_DEPTH = 4


@wp.struct
class JointParams:
    """Hidden physical truth of one joint. Never a policy input."""

    # Actuator.
    torque_scale: float  # N m delivered at |command| = 1, including lost effectiveness
    motor_alpha: float  # first-order torque response per physics step, 1 = instant
    delay_steps: int  # command latency in physics steps
    # Friction at the output joint.
    damping: float  # viscous, N m s/rad
    coulomb: float  # kinetic dry friction, N m
    stribeck: float  # extra breakaway friction as a fraction of coulomb
    bump0_mag: float  # angle-local rubbing, N m
    bump0_center: float
    bump0_width: float
    bump1_mag: float
    bump1_center: float
    bump1_width: float
    # Transmission between rotor and output, in output-equivalent coordinates.
    half_gap: float  # backlash half gap, rad; 0 selects a rigid transmission
    mesh_stiffness: float  # N m/rad once the teeth engage
    mesh_damping: float  # N m s/rad while engaged
    rotor_inertia: float  # kg m^2
    # Hard stops.
    q_min: float
    q_max: float
    # Encoder on the output joint.
    enc_bias: float  # rad
    enc_noise: float  # rad, standard deviation
    enc_quantum: float  # rad per count; 0 = continuous
    enc_delay_ticks: int  # delivery delay in control ticks
    # Motor current sensor, reported in torque-equivalent units.
    cur_gain: float
    cur_bias: float
    cur_noise: float


@wp.struct
class JointState:
    q: float  # output angle
    dq: float
    qm: float  # rotor angle; only meaningful with backlash
    dqm: float
    motor_torque: float  # after the first-order response
    tau_out: float  # torque the transmission put on the output this physics step
    enc_prev: float  # previously delivered encoder value
    vel_est: float  # causal velocity estimate derived from delivered encoder values


@wp.func
def friction_bound(p: JointParams, q: float, dq: float) -> float:
    """Largest dry-friction torque the joint can exert at this angle and speed."""

    z0 = (q - p.bump0_center) / p.bump0_width
    z1 = (q - p.bump1_center) / p.bump1_width
    rubbing = p.bump0_mag * wp.exp(-0.5 * z0 * z0) + p.bump1_mag * wp.exp(-0.5 * z1 * z1)
    v = dq / STRIBECK_VELOCITY
    return p.coulomb * (1.0 + p.stribeck * wp.exp(-v * v)) + rubbing


@wp.func
def drive_joint(p: JointParams, s: JointState, command: float, dt: float) -> JointState:
    """Advance motor and rotor by one physics step and set `tau_out`."""

    target = wp.clamp(command, -1.0, 1.0) * p.torque_scale
    s.motor_torque = s.motor_torque + p.motor_alpha * (target - s.motor_torque)
    if p.half_gap <= 0.0:
        s.tau_out = s.motor_torque
        s.qm = s.q
        s.dqm = s.dq
        return s
    delta = s.qm - s.q
    excess = wp.abs(delta) - p.half_gap
    coupling = float(0.0)
    if excess > 0.0:
        # Teeth push but never pull: damping may not reverse the contact force.
        if delta > 0.0:
            coupling = wp.max(0.0, p.mesh_stiffness * excess + p.mesh_damping * (s.dqm - s.dq))
        else:
            coupling = wp.min(0.0, -p.mesh_stiffness * excess + p.mesh_damping * (s.dqm - s.dq))
    s.dqm = s.dqm + dt * (s.motor_torque - coupling) / p.rotor_inertia
    s.qm = s.qm + dt * s.dqm
    s.tau_out = coupling
    return s


@wp.func
def clamp_to_limits(p: JointParams, s: JointState) -> JointState:
    """Inelastic hard stops."""

    if s.q < p.q_min:
        s.q = p.q_min
        s.dq = wp.max(s.dq, 0.0)
    if s.q > p.q_max:
        s.q = p.q_max
        s.dq = wp.min(s.dq, 0.0)
    return s


@wp.func
def at_limit(p: JointParams, s: JointState) -> float:
    if s.q <= p.q_min or s.q >= p.q_max:
        return 1.0
    return 0.0


@wp.func
def encoder_reading(p: JointParams, q: float, noise: float) -> float:
    value = q + p.enc_bias + p.enc_noise * noise
    if p.enc_quantum > 0.0:
        value = wp.round(value / p.enc_quantum) * p.enc_quantum
    return value


@wp.func
def current_reading(p: JointParams, mean_motor_torque: float, noise: float) -> float:
    return p.cur_gain * mean_motor_torque + p.cur_bias + p.cur_noise * noise
