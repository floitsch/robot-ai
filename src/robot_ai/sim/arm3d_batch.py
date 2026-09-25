"""Fused GPU simulation of many different, imperfect serial arms moving in 3D.

Every joint is revolute about its own axis, at its own position in the parent link; links have a
full inertia tensor. This covers the common desktop and industrial layouts (base yaw, shoulder and
elbow pitch, wrist pitch and roll, ...) and anything else that is a serial chain. The per-joint
defect model (`joint_model.py`) is the same as for the planar arms.

Dynamics per physics substep: forward kinematics; the mass matrix from link Jacobians plus each
motor's reflected rotor inertia (armature, which dominates light wrist links as it does on real geared
joints); gravity,
Coriolis and centripetal torques by recursive Newton-Euler with zero joint acceleration; one small
dense solve with viscous damping implicit; dry friction as bounded impulses through the inverse mass
matrix, exactly as in the planar kernels.

Two defects are specific to arms in 3D, and both follow from the load each joint carries (the moment
of everything beyond it, from the same Newton-Euler pass):
- Load-dependent friction. A cheap gear train loses a share of the torque it transmits, and a bearing
  that has to resist a tipping moment (a base turntable under an outstretched arm) drags in
  proportion to it. Both raise the joint's dry-friction limit: a balanced joint turns more easily
  than one with its weight hanging off to one side.
- Bendy links. Printed and extruded links are not stiff. Each link bends at its root by its
  compliance times the bending moment there (torsion about the link's own length is ignored), and the
  bent shape is used for all kinematics and dynamics. The bend follows the load quasi-statically; the
  links' own vibration is not modelled.
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
    servo_reading,
)

MAX_JOINTS = 8
METRIC_DIM = 2  # effort, friction heat; per-joint peak load and limit time follow
VELOCITY_FILTER = 0.5
FRICTION_ITERATIONS = 4
BEND_LIMIT = 0.1  # rad; caps the one-substep spike of hitting a hard stop, which the quasi-static bend can't resolve


@wp.struct
class Link3D:
    """Hidden true rigid-body parameters of one link and of the joint that drives it."""

    joint_pos: wp.vec3  # joint origin in the parent link's frame (the base frame for the first joint), m
    axis: wp.vec3  # unit joint axis in the parent link's frame; the joint's own rotation leaves it unchanged
    com: wp.vec3  # centre of mass in this link's frame, m
    mass: float  # kg, including anything carried
    inertia: wp.mat33  # about the centre of mass, in this link's frame, kg m^2
    gear_loss: float  # extra dry friction per unit of load torque the joint's gears carry
    bearing_loss: float  # extra dry friction per unit of tipping moment on the joint's bearing
    compliance: float  # bend of the link at its root per unit of bending moment, rad / (N m)
    radius: float  # the link's collision capsule around its axis (joint to next joint, or to the tool point), m


MAX_OBSTACLES = 2
NO_OBSTACLE, HALF_SPACE, CAPSULE = 0, 1, 2
CONTACT_SLIP = 0.05  # m/s over which contact friction builds up (regularized Coulomb; stiffer is unstable explicitly)


@wp.struct
class Obstacle:
    """Something the arm can run into. It is not part of the controller's input: it is only felt."""

    kind: int  # NO_OBSTACLE, HALF_SPACE (a wall, a table top) or CAPSULE (a post, a box edge, a forearm)
    a: wp.vec3  # half-space: a point on its surface; capsule: one end of its axis
    b: wp.vec3  # half-space: the outward unit normal; capsule: the other end of its axis
    radius: float  # capsule radius, m
    stiffness: float  # N/m of penetration
    damping: float  # N s/m, on approach and separation, never pulling
    friction: float  # Coulomb coefficient


@wp.func
def _link(links: wp.array2d(dtype=Link3D), changed_links: wp.array2d(dtype=Link3D), w: int, k: int,
          changed: bool) -> Link3D:
    if changed:
        return changed_links[w, k]
    return links[w, k]


@wp.func
def _joint(params: wp.array2d(dtype=JointParams), changed_params: wp.array2d(dtype=JointParams), w: int, k: int,
           changed: bool) -> JointParams:
    if changed:
        return changed_params[w, k]
    return params[w, k]


@wp.func
def _closest_on_segments(p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3) -> wp.vec2:
    """Parameters (s, t) in [0, 1] of the closest points p0 + s (p1 - p0) and q0 + t (q1 - q0) (Ericson 5.1.9)."""

    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    s = float(0.0)
    t = float(0.0)
    if a <= 1.0e-12 and e <= 1.0e-12:
        return wp.vec2(0.0, 0.0)
    if a <= 1.0e-12:
        t = wp.clamp(f / e, 0.0, 1.0)
        return wp.vec2(0.0, t)
    c = wp.dot(d1, r)
    if e <= 1.0e-12:
        s = wp.clamp(-c / a, 0.0, 1.0)
        return wp.vec2(s, 0.0)
    b = wp.dot(d1, d2)
    denominator = a * e - b * b
    if denominator > 1.0e-12:
        s = wp.clamp((b * f - c * e) / denominator, 0.0, 1.0)
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = wp.clamp(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = wp.clamp((b - c) / a, 0.0, 1.0)
    return wp.vec2(s, t)


@wp.func
def _bent(bend: wp.vec3) -> wp.mat33:
    angle = wp.length(bend)
    if angle < 1.0e-9:
        return wp.identity(n=3, dtype=wp.float32)
    return wp.quat_to_matrix(wp.quat_from_axis_angle(bend / angle, angle))


@wp.func
def _delayed_command(ring: wp.array3d(dtype=wp.float32), w: int, j: int, tick: int, substep: int, substeps: int,
                     delay_steps: int) -> float:
    step = tick * substeps + substep - delay_steps
    if step < 0:
        return ring[w, j, RING_DEPTH - 1]  # what reset left there, not yet overwritten: nothing / hold the start
    return ring[w, j, (step // substeps) % RING_DEPTH]


@wp.kernel
def _step_tick(
    n: int,
    links: wp.array2d(dtype=Link3D),
    changed_links: wp.array2d(dtype=Link3D),
    change_tick: wp.array(dtype=wp.int32),
    tip_offset: wp.array(dtype=wp.vec3),
    gravity: wp.array(dtype=wp.float32),
    params: wp.array2d(dtype=JointParams),
    changed_params: wp.array2d(dtype=JointParams),
    push: wp.array3d(dtype=wp.float32),
    states: wp.array2d(dtype=JointState),
    commands: wp.array2d(dtype=wp.float32),
    command_ring: wp.array3d(dtype=wp.float32),
    stiffness: wp.array2d(dtype=wp.float32),
    stiffness_ring: wp.array3d(dtype=wp.float32),
    encoder_ring: wp.array3d(dtype=wp.float32),
    ticks: wp.array(dtype=wp.int32),
    rng: wp.array(dtype=wp.uint32),
    rot: wp.array2d(dtype=wp.mat33),
    origin: wp.array2d(dtype=wp.vec3),
    axis_world: wp.array2d(dtype=wp.vec3),
    centre: wp.array2d(dtype=wp.vec3),
    omega: wp.array2d(dtype=wp.vec3),
    force: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
    load_force: wp.array2d(dtype=wp.vec3),
    load_moment: wp.array2d(dtype=wp.vec3),
    bend: wp.array2d(dtype=wp.vec3),
    mass: wp.array3d(dtype=wp.float32),
    inverse: wp.array3d(dtype=wp.float32),
    scratch: wp.array3d(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
    truth: wp.array2d(dtype=wp.float32),
    tip: wp.array(dtype=wp.vec3),
    tool_axis: wp.array(dtype=wp.vec3),
    metrics: wp.array2d(dtype=wp.float32),
    obstacles: wp.array2d(dtype=Obstacle),
    contact: wp.array2d(dtype=wp.float32),
    dt: float,
    substeps: int,
):
    w = wp.tid()
    tick = ticks[w]
    changed = tick >= change_tick[w]
    g = gravity[w]
    contact[w, 1] = 0.0  # peak of this tick
    contact[w, 3] = 0.0  # impulse of this tick
    for j in range(n):
        stiffness_ring[w, j, tick % RING_DEPTH] = stiffness[w, j]
        if params[w, j].servo_gain > 0.0:
            command_ring[w, j, tick % RING_DEPTH] = commands[w, j]  # a target angle
        else:
            command_ring[w, j, tick % RING_DEPTH] = wp.clamp(commands[w, j], -1.0, 1.0)
        metrics[w, 2 + j] = 0.0
        metrics[w, 2 + n + j] = 0.0
        scratch[w, 7, j] = 0.0  # mean motor torque accumulator for the current sensor
    effort = float(0.0)
    heat = float(0.0)

    for substep in range(substeps):
        # Motors and transmissions, per joint.
        for j in range(n):
            p = _joint(params, changed_params, w, j, changed)
            s = states[w, j]
            # A servo's stiffness (its gain register) is set by the host too, and arrives as late as its target.
            p.servo_gain = p.servo_gain * _delayed_command(stiffness_ring, w, j, tick, substep, substeps, p.delay_steps)
            s = drive_joint(p, s, _delayed_command(command_ring, w, j, tick, substep, substeps, p.delay_steps), dt)
            states[w, j] = s
            scratch[w, 7, j] = scratch[w, 7, j] + s.motor_torque

        # Forward kinematics: link orientations, joint origins, joint axes, centres of mass, angular velocities.
        parent_rot = wp.identity(n=3, dtype=wp.float32)
        parent_pos = wp.vec3(0.0, 0.0, 0.0)
        parent_omega = wp.vec3(0.0, 0.0, 0.0)
        for k in range(n):
            link = _link(links, changed_links, w, k, changed)
            s = states[w, k]
            joint_origin = parent_pos + parent_rot * link.joint_pos
            joint_axis = parent_rot * link.axis
            link_rot = parent_rot * wp.quat_to_matrix(wp.quat_from_axis_angle(link.axis, s.q)) * _bent(bend[w, k])
            link_omega = parent_omega + joint_axis * s.dq
            rot[w, k] = link_rot
            origin[w, k] = joint_origin
            axis_world[w, k] = joint_axis
            centre[w, k] = joint_origin + link_rot * link.com
            omega[w, k] = link_omega
            parent_rot = link_rot
            parent_pos = joint_origin
            parent_omega = link_omega

        # Mass matrix: M_ij = sum over links k >= max(i, j) of m_k Jv_ki . Jv_kj + a_i . I_k a_j, where
        # Jv_ki = a_i x (c_k - o_i) is the velocity of link k's centre of mass per unit rate of joint i.
        for i in range(n):
            for j in range(n):
                mass[w, i, j] = 0.0
        for k in range(n):
            link = _link(links, changed_links, w, k, changed)
            link_rot = rot[w, k]
            inertia_world = link_rot * link.inertia * wp.transpose(link_rot)
            c = centre[w, k]
            for i in range(k + 1):
                ai = axis_world[w, i]
                vi = wp.cross(ai, c - origin[w, i])
                ia = inertia_world * ai
                for j in range(i + 1):
                    aj = axis_world[w, j]
                    vj = wp.cross(aj, c - origin[w, j])
                    value = link.mass * wp.dot(vi, vj) + wp.dot(aj, ia)
                    mass[w, i, j] = mass[w, i, j] + value
                    if j != i:
                        mass[w, j, i] = mass[w, j, i] + value

        # Bias torques (Coriolis, centripetal and gravity) by recursive Newton-Euler with zero joint
        # acceleration; gravity enters as an upward acceleration of the base. Alongside, the loads the links
        # actually carry, with the joints' accelerations from the previous substep (scratch row 3).
        parent_alpha = wp.vec3(0.0, 0.0, 0.0)
        parent_acc = wp.vec3(0.0, 0.0, g)
        parent_alpha_load = wp.vec3(0.0, 0.0, 0.0)
        parent_acc_load = wp.vec3(0.0, 0.0, g)
        parent_pos = wp.vec3(0.0, 0.0, 0.0)
        parent_omega = wp.vec3(0.0, 0.0, 0.0)
        for k in range(n):
            link = _link(links, changed_links, w, k, changed)
            s = states[w, k]
            joint_origin = origin[w, k]
            link_omega = omega[w, k]
            alpha = parent_alpha + wp.cross(parent_omega, axis_world[w, k] * s.dq)
            lever = joint_origin - parent_pos
            acc = parent_acc + wp.cross(parent_alpha, lever) + wp.cross(parent_omega, wp.cross(parent_omega, lever))
            arm = centre[w, k] - joint_origin
            acc_centre = acc + wp.cross(alpha, arm) + wp.cross(link_omega, wp.cross(link_omega, arm))
            link_rot = rot[w, k]
            inertia_world = link_rot * link.inertia * wp.transpose(link_rot)
            force[w, k] = link.mass * acc_centre
            moment[w, k] = inertia_world * alpha + wp.cross(link_omega, inertia_world * link_omega)
            alpha_load = parent_alpha_load + wp.cross(parent_omega, axis_world[w, k] * s.dq) + axis_world[w, k] * scratch[w, 3, k]
            acc_load = (parent_acc_load + wp.cross(parent_alpha_load, lever)
                        + wp.cross(parent_omega, wp.cross(parent_omega, lever)))
            acc_centre_load = acc_load + wp.cross(alpha_load, arm) + wp.cross(link_omega, wp.cross(link_omega, arm))
            load_force[w, k] = link.mass * acc_centre_load
            load_moment[w, k] = inertia_world * alpha_load + wp.cross(link_omega, inertia_world * link_omega)
            parent_alpha_load = alpha_load
            parent_acc_load = acc_load
            parent_alpha = alpha
            parent_acc = acc
            parent_pos = joint_origin
            parent_omega = link_omega
        child_force = wp.vec3(0.0, 0.0, 0.0)
        child_moment = wp.vec3(0.0, 0.0, 0.0)
        child_origin = wp.vec3(0.0, 0.0, 0.0)
        child_load_force = wp.vec3(0.0, 0.0, 0.0)
        child_load_moment = wp.vec3(0.0, 0.0, 0.0)
        now = float(tick * substeps + substep) * dt
        for back in range(n):
            k = n - 1 - back
            joint_origin = origin[w, k]
            own_force = force[w, k]
            # Moment about this joint's origin, of this link and of everything it carries.
            total_moment = (moment[w, k] + wp.cross(centre[w, k] - joint_origin, own_force) + child_moment
                            + wp.cross(child_origin - joint_origin, child_force))
            load = (load_moment[w, k] + wp.cross(centre[w, k] - joint_origin, load_force[w, k]) + child_load_moment
                    + wp.cross(child_origin - joint_origin, child_load_force))
            child_load_force = load_force[w, k] + child_load_force
            child_load_moment = load
            child_force = own_force + child_force
            child_moment = total_moment
            child_origin = joint_origin
            bias = wp.dot(axis_world[w, k], total_moment)
            # The load strains the gears (along the axis) and the bearing (across it) ...
            link = _link(links, changed_links, w, k, changed)
            carried = wp.dot(axis_world[w, k], load)
            tipping = wp.length(load - axis_world[w, k] * carried)
            scratch[w, 2, k] = link.gear_loss * wp.abs(carried) + link.bearing_loss * tipping
            # ... and bends the link, about the axes across its length; used from the next substep on.
            along = tip_offset[w]
            if k + 1 < n:
                along = _link(links, changed_links, w, k + 1, changed).joint_pos
            along = wp.normalize(along)
            local = wp.transpose(rot[w, k]) * load
            bent = -link.compliance * (local - along * wp.dot(along, local))
            size = wp.length(bent)
            if size > BEND_LIMIT:
                bent = bent * (BEND_LIMIT / size)
            bend[w, k] = bent
            p = _joint(params, changed_params, w, k, changed)
            s = states[w, k]
            pushed = push[w, k, 0] * wp.sin(6.2831853 * push[w, k, 1] * now + push[w, k, 2])
            pushed = pushed + push[w, k, 3] * wp.sin(6.2831853 * push[w, k, 4] * now + push[w, k, 5])
            scratch[w, 6, k] = s.tau_out + pushed - bias - p.damping * s.dq
            mass[w, k, k] = mass[w, k, k] + dt * p.damping
            if p.half_gap <= 0.0:
                # A stiff gear train moves the motor's rotor with the joint: its inertia, reflected through the
                # gearbox, adds to the joint's (armature). With backlash the rotor is a separate coordinate instead.
                mass[w, k, k] = mass[w, k, k] + p.rotor_inertia
            elif wp.abs(s.qm - s.q) > p.half_gap:
                # Engaged teeth are a stiff spring onto a possibly very light link (a wrist roll): linearly implicit, like
                # the damping, or explicit integration rings up at a thousand rad/s and explodes.
                mass[w, k, k] = mass[w, k, k] + dt * (p.mesh_damping + dt * p.mesh_stiffness)

        # Contacts with obstacles: every link is a capsule from its joint to the next (the last one to the tool point).
        # A penetrating point gets a spring-damper force along the obstacle's normal and regularized Coulomb friction
        # across it; its Jacobian carries that force to every joint below.
        pressing = float(0.0)
        for k in range(1, n):  # the first link is the base, mounted where it stands
            link = _link(links, changed_links, w, k, changed)
            start = origin[w, k]
            end = start + rot[w, k] * tip_offset[w]
            if k + 1 < n:
                end = origin[w, k + 1]
            for o in range(MAX_OBSTACLES):
                obstacle = obstacles[w, o]
                if obstacle.kind == NO_OBSTACLE:
                    continue
                for side in range(2):
                    point = start
                    normal = wp.vec3(0.0, 0.0, 1.0)
                    depth = float(-1.0)
                    if obstacle.kind == HALF_SPACE:
                        if side == 1:
                            point = end
                        normal = obstacle.b
                        depth = link.radius - wp.dot(point - obstacle.a, normal)
                        point = point - normal * link.radius  # the capsule's surface point deepest in the obstacle
                    elif side == 0:  # a capsule touches a segment in one place
                        st = _closest_on_segments(start, end, obstacle.a, obstacle.b)
                        point = start + (end - start) * st[0]
                        nearest = obstacle.a + (obstacle.b - obstacle.a) * st[1]
                        gap = point - nearest
                        distance = wp.length(gap)
                        if distance > 1.0e-9:
                            normal = gap / distance
                        depth = link.radius + obstacle.radius - distance
                        point = point - normal * link.radius
                    if depth <= 0.0:
                        continue
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                    for i in range(k + 1):
                        velocity = velocity + wp.cross(axis_world[w, i], point - origin[w, i]) * states[w, i].dq
                    approach = wp.dot(velocity, normal)
                    push_force = wp.max(0.0, obstacle.stiffness * depth - obstacle.damping * approach)
                    slide = velocity - normal * approach
                    speed = wp.length(slide)
                    reaction = normal * push_force
                    if speed > 1.0e-9:
                        reaction = reaction - slide * (obstacle.friction * push_force * wp.tanh(speed / CONTACT_SLIP) / speed)
                    for i in range(k + 1):
                        scratch[w, 6, i] = scratch[w, 6, i] + wp.dot(wp.cross(axis_world[w, i], point - origin[w, i]), reaction)
                    pressing = pressing + push_force
        contact[w, 0] = pressing
        contact[w, 1] = wp.max(contact[w, 1], pressing)
        contact[w, 2] = wp.max(contact[w, 2], pressing)
        contact[w, 3] = contact[w, 3] + pressing * dt

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

        # Unconstrained velocity update.
        for i in range(n):
            acc_i = float(0.0)
            for j in range(n):
                acc_i = acc_i + inverse[w, i, j] * scratch[w, 6, j]
            scratch[w, 0, i] = states[w, i].dq + dt * acc_i

        # Dry friction as bounded impulses, projected Gauss-Seidel through the inverse mass matrix.
        for j in range(n):
            scratch[w, 1, j] = 0.0
        for _ in range(FRICTION_ITERATIONS):
            for i in range(n):
                p = _joint(params, changed_params, w, i, changed)
                s = states[w, i]
                bound = (friction_bound(p, s.q, s.dq) + scratch[w, 2, i]) * dt
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
            p = _joint(params, changed_params, w, i, changed)
            s = states[w, i]
            before = s.dq
            s.dq = scratch[w, 0, i]
            s.q = s.q + dt * s.dq
            s = clamp_to_limits(p, s)
            scratch[w, 3, i] = (s.dq - before) / dt  # a hard stop's force is a load too
            states[w, i] = s
            effort = effort + dt * s.tau_out * s.tau_out
            metrics[w, 2 + i] = wp.max(metrics[w, 2 + i], wp.abs(s.tau_out))
            metrics[w, 2 + n + i] = metrics[w, 2 + n + i] + at_limit(p, s) / float(substeps)

    # Tool tip in the world, from the final state.
    parent_rot = wp.identity(n=3, dtype=wp.float32)
    parent_pos = wp.vec3(0.0, 0.0, 0.0)
    for k in range(n):
        link = _link(links, changed_links, w, k, changed)
        parent_pos = parent_pos + parent_rot * link.joint_pos
        parent_rot = parent_rot * wp.quat_to_matrix(wp.quat_from_axis_angle(link.axis, states[w, k].q)) * _bent(bend[w, k])
    tip[w] = parent_pos + parent_rot * tip_offset[w]
    tool_axis[w] = parent_rot * wp.normalize(tip_offset[w])

    # Sensors sample once per control tick; encoder packets arrive `enc_delay_ticks` late.
    tick = tick + 1
    state = rng[w]
    control_dt = dt * float(substeps)
    for j in range(n):
        p = _joint(params, changed_params, w, j, changed)
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
    bend: wp.array2d(dtype=wp.vec3),
    scratch: wp.array3d(dtype=wp.float32),
    stiffness_ring: wp.array3d(dtype=wp.float32),
    contact: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    if mask[w] == 0:
        return
    for c in range(4):
        contact[w, c] = 0.0
    state = rng[w]
    for j in range(n):
        p = params[w, j]
        bend[w, j] = wp.vec3(0.0, 0.0, 0.0)  # settles within the first substep
        scratch[w, 3, j] = 0.0
        s = JointState()
        s.q = wp.clamp(start_q[w, j], p.q_min, p.q_max)
        s.qm = s.q
        reading = encoder_reading(p, s.q, wp.randn(state))
        s.enc_prev = reading
        hold = float(0.0)
        if p.servo_gain > 0.0:
            hold = servo_reading(p, s.q)  # a servo holds where it is until the first target arrives
            s.servo_prev = hold
        for k in range(RING_DEPTH):
            command_ring[w, j, k] = hold
            stiffness_ring[w, j, k] = 1.0
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


class Arm3DBatch:
    """A population of distinct imperfect 3D serial arms, stepped one control tick per launch.

    `links` is a structured array [worlds, joints] of `Link3D`, `joints` one of `JointParams`, and `tip_offset`
    [worlds, 3] the tool point in the last link's frame. Observation layout: measured q (n), derived dq (n),
    motor current (n). Truth: q (n), dq (n); `tip` holds the true tool point in the world and `tool_axis` the
    direction the tool points (from the last joint towards the tool point), bends included. Metrics: effort,
    friction heat, then per-joint peak transmission torque (n) and hard-stop time (n).
    """

    def __init__(self, links: np.ndarray, joints: np.ndarray, tip_offset: np.ndarray, *, device: str = "cuda:0",
                 physics_dt: float = 0.001, substeps: int = 10, seed: int = 0, gravity: float | np.ndarray = 9.81,
                 changed: tuple[np.ndarray, np.ndarray] | None = None, change_tick: np.ndarray | None = None,
                 push: np.ndarray | None = None, obstacles: np.ndarray | None = None) -> None:
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
        self.links = wp.array(links, dtype=Link3D, device=device)
        self.params = wp.array(joints, dtype=JointParams, device=device)
        self.changed_links = self.links if changed is None else wp.array(changed[0], dtype=Link3D, device=device)
        self.changed_params = self.params if changed is None else wp.array(changed[1], dtype=JointParams, device=device)
        never = np.full(w, np.iinfo(np.int32).max, dtype=np.int32)
        self.change_tick = wp.array(never if changed is None or change_tick is None else np.asarray(change_tick, dtype=np.int32),
                                    dtype=wp.int32, device=device)
        offsets = np.ascontiguousarray(np.broadcast_to(np.asarray(tip_offset, dtype=np.float32), (w, 3)))
        self.tip_offset = wp.array(offsets, dtype=wp.vec3, device=device)
        self.gravity = wp.array(np.broadcast_to(np.asarray(gravity, dtype=np.float32), (w,)).copy(), dtype=wp.float32, device=device)
        waves = np.zeros((w, n, 6), dtype=np.float32) if push is None else np.asarray(push, dtype=np.float32)
        if waves.shape != (w, n, 6):
            raise ValueError("push must have shape [worlds, joints, 6]")
        self.push = wp.array(waves, dtype=wp.float32, device=device)
        self.states = wp.zeros((w, n), dtype=JointState, device=device)
        self.command_ring = wp.zeros((w, n, RING_DEPTH), dtype=wp.float32, device=device)
        self.stiffness_ring = wp.ones((w, n, RING_DEPTH), dtype=wp.float32, device=device)
        self.nominal_stiffness = wp.ones((w, n), dtype=wp.float32, device=device)
        self.encoder_ring = wp.zeros((w, n, RING_DEPTH), dtype=wp.float32, device=device)
        self.ticks = wp.zeros(w, dtype=wp.int32, device=device)
        streams = np.random.default_rng(seed).integers(1, 2**32, w, dtype=np.uint32)
        self.rng = wp.array(streams, dtype=wp.uint32, device=device)
        self.rot = wp.zeros((w, n), dtype=wp.mat33, device=device)
        self.origin = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.axis_world = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.centre = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.omega = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.force = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.moment = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.load_force = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.load_moment = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.bend = wp.zeros((w, n), dtype=wp.vec3, device=device)
        self.mass = wp.zeros((w, n, n), dtype=wp.float32, device=device)
        self.inverse = wp.zeros((w, n, n), dtype=wp.float32, device=device)
        self.scratch = wp.zeros((w, 8, n), dtype=wp.float32, device=device)
        self.observation = wp.zeros((w, 3 * n), dtype=wp.float32, device=device)
        self.truth = wp.zeros((w, 2 * n), dtype=wp.float32, device=device)
        self.tip = wp.zeros(w, dtype=wp.vec3, device=device)
        self.tool_axis = wp.zeros(w, dtype=wp.vec3, device=device)
        self.metrics = wp.zeros((w, METRIC_DIM + 2 * n), dtype=wp.float32, device=device)
        # Obstacles [worlds, MAX_OBSTACLES] of `Obstacle`; contact: pressing force now, peak this tick, peak so far (N),
        # impulse this tick (N s).
        none = np.zeros((w, MAX_OBSTACLES), dtype=Obstacle.numpy_dtype())
        self.obstacles = wp.array(none if obstacles is None else obstacles, dtype=Obstacle, device=device)
        self.contact = wp.zeros((w, 4), dtype=wp.float32, device=device)
        self._shared = [self.params, self.states, self.command_ring, self.encoder_ring, self.ticks, self.rng,
                        self.observation, self.truth, self.metrics]

    def reset(self, start_q: np.ndarray, mask: np.ndarray | None = None) -> None:
        mask_values = np.ones(self.worlds, dtype=np.int32) if mask is None else np.asarray(mask, dtype=np.int32)
        start = np.ascontiguousarray(np.broadcast_to(np.asarray(start_q, dtype=np.float32), (self.worlds, self.joints)))
        wp.launch(_reset, dim=self.worlds, device=self.device,
                  inputs=[self.joints, wp.array(mask_values, device=self.device), wp.array(start, device=self.device), *self._shared,
                          self.bend, self.scratch, self.stiffness_ring, self.contact])

    def step(self, commands: object, stiffness: object = None) -> None:
        """`stiffness` [worlds, joints] scales each position servo's gain (1 = as tuned); ignored by torque drives."""

        def device_array(values: object) -> wp.array:
            if isinstance(values, np.ndarray):
                return wp.array(np.ascontiguousarray(values, dtype=np.float32), device=self.device)
            return wp.from_torch(values.detach().contiguous())  # type: ignore[attr-defined]

        source = device_array(commands)
        scale = self.nominal_stiffness if stiffness is None else device_array(stiffness)
        if source.shape != (self.worlds, self.joints) or scale.shape != (self.worlds, self.joints):
            raise ValueError("commands and stiffness must have shape [worlds, joints]")
        wp.launch(_step_tick, dim=self.worlds, device=self.device,
                  inputs=[self.joints, self.links, self.changed_links, self.change_tick, self.tip_offset, self.gravity,
                          self.params, self.changed_params, self.push, self.states, source, self.command_ring,
                          scale, self.stiffness_ring, self.encoder_ring, self.ticks, self.rng, self.rot, self.origin, self.axis_world, self.centre,
                          self.omega, self.force, self.moment, self.load_force, self.load_moment, self.bend, self.mass, self.inverse, self.scratch, self.observation,
                          self.truth, self.tip, self.tool_axis, self.metrics, self.obstacles, self.contact, self.physics_dt,
                          self.substeps])

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)
