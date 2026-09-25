""""Put the hand there": a reach task for populations of imperfect five-joint arms that move in 3D.

The nominal arm is laid out like most desktop and light industrial arms: a base that turns about the
vertical (yaw), then shoulder, elbow and wrist joints that pitch about horizontal axes, then a wrist
roll about the tool axis. At zero every link points straight up. Each robot in a population gets its
own link lengths, masses, tool and payload, every joint its own defects exactly as for the planar arms,
and on top the 3D arm's own: joints that drag more the more load they carry, and links that bend.

A goal is a tool pose: where the tool point should be, which way the tool should point, and the wrist
roll. It is judged on where the tool really is, bends included. The controller is told the arm's
dimensions (what a maker reads off the drawing) and nothing else about the robot; it has no camera,
so a sagging link can only be read from the joints' feel. Encoders are assumed homed: an absolute zero
error cannot be told apart from a goal elsewhere without an outside reference, so only a small
residual remains.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from torch import Tensor

from .arm3d_batch import CAPSULE, HALF_SPACE, MAX_OBSTACLES, Arm3DBatch, Link3D, Obstacle
from .chain_env import ChainEnv, ChainSummary
from .population import _per_world, sample_joint_defects
from .reach_env import FINAL_WINDOW, SETTLED_SPEED, TOLERANCE

JOINTS = 5
AXES = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# Link k runs from joint k to joint k + 1 along its own +z; the last link's length is the tool.
LENGTHS = np.array([0.08, 0.25, 0.22, 0.07, 0.09])
MASSES = np.array([0.35, 0.45, 0.30, 0.15, 0.12])
RADII = np.array([0.04, 0.02, 0.018, 0.02, 0.02])
COM_FRACTION = np.array([0.4, 0.5, 0.5, 0.5, 0.45])
TOOL_SIDE_OFFSET = 0.012  # a gripper is rarely symmetric about its roll axis, which gives the roll joint a load
NOMINAL_TORQUES = np.array([3.0, 8.0, 5.0, 1.5, 0.8])
LIMITS = np.array([[-2.6, 2.6], [-1.7, 1.7], [-2.4, 2.4], [-2.0, 2.0], [-2.9, 2.9]])
GOAL_SPAN = np.array([1.6, 1.0, 1.4, 1.4, 2.0])
TASK_CODE = 3
# Position servos: the policy moves each target by up to this much from the inverse-kinematics solution, rad.
SERVO_REACH = 0.3
# The shared defect model's friction is sized for the planar arm's 3 N m elbow; gears and bearings scale with the joint.
FRICTION_SCALE = NOMINAL_TORQUES / 3.0
POSITION_UNIT = 0.005 / TOLERANCE  # m of tool position error that count like 1 rad of joint error: 5 mm ~ 30 mrad
TABLE_CLEARANCE = 0.03  # goals keep the tool point above the surface the base stands on
# Largest root bend per link at its motor's rated torque, rad: the stubby base column is much stiffer.
BEND_AT_RATED = np.array([0.005, 0.02, 0.02, 0.02, 0.02])


def _cylinder_inertia(mass: np.ndarray, length: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """[..., 3, 3] inertia of solid cylinders along z about their centre."""

    transverse = mass * (3.0 * radius**2 + length**2) / 12.0
    axial = mass * radius**2 / 2.0
    inertia = np.zeros((*mass.shape, 3, 3))
    inertia[..., 0, 0], inertia[..., 1, 1], inertia[..., 2, 2] = transverse, transverse, axial
    return inertia


def _links(lengths: np.ndarray, masses: np.ndarray, side_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Link3D records [worlds, joints] and tool offsets [worlds, 3] for arms of the nominal layout."""

    worlds = lengths.shape[0]
    links = np.zeros((worlds, JOINTS), dtype=Link3D.numpy_dtype())
    links["axis"] = AXES
    links["joint_pos"][:, 1:, 2] = lengths[:, :-1]
    links["com"][..., 2] = lengths * COM_FRACTION
    links["com"][:, -1, 0] = side_offset
    links["mass"] = masses
    links["inertia"] = _cylinder_inertia(masses, lengths, np.broadcast_to(RADII, lengths.shape))
    links["radius"] = RADII
    tip = np.zeros((worlds, 3))
    tip[:, 2] = lengths[:, -1]
    return links, tip


def add_tool_mass(links: np.ndarray, tip: np.ndarray, added: np.ndarray) -> None:
    """Attach a point mass at the tool point of the last link, in place, keeping the full inertia tensor exact."""

    last = links[:, -1]
    mass, com, inertia = last["mass"].astype(np.float64), last["com"].astype(np.float64), last["inertia"].astype(np.float64)
    total = mass + added
    new_com = (mass[:, None] * com + added[:, None] * tip) / total[:, None]

    def shift(m: np.ndarray, d: np.ndarray) -> np.ndarray:
        return m[:, None, None] * (np.einsum("wi,wi->w", d, d)[:, None, None] * np.eye(3) - np.einsum("wi,wj->wij", d, d))

    last["inertia"] = inertia + shift(mass, com - new_com) + shift(added, tip - new_com)
    last["mass"], last["com"] = total, new_com


def sample_servos(rng: np.random.Generator, joints: np.ndarray) -> None:
    """Make every joint a position servo with its own tuning, in place: a property of the servo, not a defect.

    Full duty at 0.04-0.16 rad of error (a 4x spread of stiffness, as between LeRobot's and Feetech's default gains),
    a 12-bit magnetic encoder that the host reads too, a deadband of up to two counts, and a 0.5-3 ms motor.
    """

    shape = joints.shape
    count = 2.0 * np.pi / 4096
    joints["servo_gain"] = 1.0 / rng.uniform(0.04, 0.16, shape)
    joints["servo_damping"] = rng.uniform(0.005, 0.03, shape)
    joints["servo_quantum"] = count
    joints["servo_deadband"] = rng.integers(0, 3, shape) * count
    # The host reads the servo's own encoder over the bus: the same counts, flickering by about one, late by the
    # bus delay. A coarser or noisier reading than the servo's own belongs to torque drives with separate encoders.
    joints["enc_quantum"] = count
    joints["enc_noise"] = np.minimum(joints["enc_noise"], 0.5 * count)
    # A voltage-driven motor's current follows within its electrical time constant, and the servo's firmware is tuned
    # for it; a slow torque response is a torque-drive defect and would make any servo's own loop oscillate.
    joints["motor_alpha"] = 1.0 - np.exp(-0.001 / rng.uniform(0.0005, 0.003, shape))


def sample_arm3d_population(rng: np.random.Generator, worlds: int, *, severity: float | np.ndarray = 1.0,
                            friction_probability: float = 0.7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_robot, column = _per_world(severity, worlds)
    shape = (worlds, JOINTS)
    lengths = LENGTHS * (1.0 + column * rng.uniform(-0.15, 0.15, shape))
    masses = MASSES * (1.0 + column * rng.uniform(-0.20, 0.20, shape))
    side = TOOL_SIDE_OFFSET * (1.0 + per_robot * rng.uniform(-0.5, 0.5, worlds))
    links, tip = _links(lengths, masses, side)
    add_tool_mass(links, tip, np.where(rng.random(worlds) < 0.5, rng.uniform(0.0, 0.15, worlds), 0.0) * per_robot)
    links["gear_loss"] = np.where(rng.random(shape) < 0.6, rng.uniform(0.0, 0.25, shape), 0.0) * column
    links["bearing_loss"] = np.where(rng.random(shape) < 0.6, rng.uniform(0.0, 0.15, shape), 0.0) * column
    links["compliance"] = np.where(rng.random(shape) < 0.7, rng.uniform(0.0, 1.0, shape), 0.0) * column * BEND_AT_RATED / NOMINAL_TORQUES
    joints = sample_joint_defects(rng, worlds, JOINTS, NOMINAL_TORQUES, LIMITS, severity=severity,
                                  friction_probability=friction_probability, friction_scale=FRICTION_SCALE)
    joints["enc_bias"] *= 0.1  # homed: +-2 mrad left
    joints["no_load_speed"] = rng.uniform(4.0, 10.0, shape)  # geared hobby motors; a property, not a defect
    return links, joints, tip


def sample_arm3d_change(rng: np.random.Generator, links: np.ndarray, joints: np.ndarray, tip: np.ndarray, *,
                        severity: float | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The same robots after something changed: a payload picked up, a motor fading, a joint fouling."""

    worlds = links.shape[0]
    per_robot, column = _per_world(severity, worlds)
    changed_links, changed_joints = links.copy(), joints.copy()
    add_tool_mass(changed_links, tip, np.where(rng.random(worlds) < 0.6, rng.uniform(0.05, 0.20, worlds), 0.0) * per_robot)
    shape = (worlds, JOINTS)
    fading = np.where(rng.random(shape) < 0.3, rng.uniform(0.0, 0.4, shape), 0.0) * column
    changed_joints["torque_scale"] = joints["torque_scale"] * (1.0 - fading)
    changed_joints["coulomb"] = joints["coulomb"] + np.where(rng.random(shape) < 0.3, rng.uniform(0.0, 0.2, shape), 0.0) * column * FRICTION_SCALE
    return changed_links, changed_joints


def _rotation(axis: Tensor, angle: Tensor) -> Tensor:
    """Rodrigues: [W, 3] unit axes and [W] angles to [W, 3, 3] rotations."""

    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((torch.stack((zero, -z, y), -1), torch.stack((z, zero, -x), -1), torch.stack((-y, x, zero), -1)), -2)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, dtype=axis.dtype, device=axis.device) + s * k + (1.0 - c) * (k @ k)


def arm3d_points(q: Tensor, joint_pos: Tensor, axis: Tensor, tip: Tensor) -> Tensor:
    """[W, n + 1, 3]: every joint's position and then the tool point, of rigid arms."""

    worlds, n = q.shape
    rot = torch.eye(3, dtype=q.dtype, device=q.device).expand(worlds, 3, 3)
    pos = torch.zeros((worlds, 3), dtype=q.dtype, device=q.device)
    points = []
    for k in range(n):
        pos = pos + (rot @ joint_pos[:, k, :, None])[..., 0]
        points.append(pos)
        rot = rot @ _rotation(axis[:, k], q[:, k])
    points.append(pos + (rot @ tip[..., None])[..., 0])
    return torch.stack(points, dim=1)


def _segment_distance(p0: Tensor, p1: Tensor, q0: Tensor, q1: Tensor) -> Tensor:
    """Distances [...] between segments p0-p1 and q0-q1 (all [..., 3]), by dense sampling: for placement checks only."""

    s = torch.linspace(0.0, 1.0, 17, dtype=p0.dtype, device=p0.device)
    a = p0[..., None, :] + (p1 - p0)[..., None, :] * s[:, None]
    b = q0[..., None, :] + (q1 - q0)[..., None, :] * s[:, None]
    return (a[..., :, None, :] - b[..., None, :, :]).norm(dim=-1).amin(dim=(-1, -2))


def arm3d_tool(q: Tensor, joint_pos: Tensor, axis: Tensor, tip: Tensor) -> tuple[Tensor, Tensor]:
    """Tool point [W, 3] and tool direction [W, 3] of rigid arms: what a controller can work out from its encoders."""

    worlds, n = q.shape
    rot = torch.eye(3, dtype=q.dtype, device=q.device).expand(worlds, 3, 3)
    pos = torch.zeros((worlds, 3), dtype=q.dtype, device=q.device)
    for k in range(n):
        pos = pos + (rot @ joint_pos[:, k, :, None])[..., 0]
        rot = rot @ _rotation(axis[:, k], q[:, k])
    return pos + (rot @ tip[..., None])[..., 0], (rot @ torch.nn.functional.normalize(tip, dim=-1)[..., None])[..., 0]


def arm3d_inverse(pose: Tensor, joint_pos: Tensor, tip: Tensor) -> Tensor:
    """[W, 4, 5]: the joint solutions that put the nominal-layout arm's tool at `pose` [W, 7] (point, direction, roll).

    Base turned towards the tool or half round from it, each with the elbow either way; angles in (-pi, pi], some
    possibly outside the joint limits. Closed form: the wrist-pitch joint sits behind the tool point along the tool
    direction, and the shoulder and elbow form a two-link arm in the vertical plane through the base.
    """

    base, upper, lower = joint_pos[:, 1, 2], joint_pos[:, 2, 2], joint_pos[:, 3, 2]
    point, direction, roll = pose[:, :3], pose[:, 3:6], pose[:, 6]
    wrist = point - (joint_pos[:, 4, 2] + tip[:, 2])[:, None] * direction
    towards = torch.where(direction[:, :2].norm(dim=1) > 1e-3, torch.atan2(direction[:, 1], direction[:, 0]),
                          torch.atan2(wrist[:, 1], wrist[:, 0]))
    solutions = []
    for yaw in (towards, towards + torch.pi):
        c, s = torch.cos(yaw), torch.sin(yaw)
        reach, height = wrist[:, 0] * c + wrist[:, 1] * s, wrist[:, 2] - base
        pitch = torch.atan2(direction[:, 0] * c + direction[:, 1] * s, direction[:, 2])
        bend = ((reach**2 + height**2 - upper**2 - lower**2) / (2.0 * upper * lower)).clamp(-1.0, 1.0)
        for sign in (1.0, -1.0):
            elbow = sign * torch.acos(bend)
            shoulder = torch.atan2(reach, height) - torch.atan2(lower * torch.sin(elbow), upper + lower * torch.cos(elbow))
            solutions.append(torch.stack((yaw, shoulder, elbow, pitch - shoulder - elbow, roll), dim=1))
    return torch.remainder(torch.stack(solutions, dim=1) + torch.pi, 2.0 * torch.pi) - torch.pi


def arm3d_dynamics(q: Tensor, dq: Tensor, joint_pos: Tensor, axis: Tensor, com: Tensor, mass: Tensor, inertia: Tensor,
                   gravity: float = 9.81) -> tuple[Tensor, Tensor]:
    """Mass matrix [W, n, n] and bias torques [W, n] of 3D serial arms, the kernel's formulation in Torch.

    q, dq [W, n]; joint_pos, axis, com [W, n, 3]; mass [W, n]; inertia [W, n, 3, 3]. M ddq = tau - bias.
    """

    worlds, n = q.shape
    rot = torch.eye(3, dtype=q.dtype, device=q.device).expand(worlds, 3, 3)
    pos = torch.zeros((worlds, 3), dtype=q.dtype, device=q.device)
    spin = torch.zeros_like(pos)
    rots, origins, axes, centres, omegas = [], [], [], [], []
    for k in range(n):
        origin = pos + (rot @ joint_pos[:, k, :, None])[..., 0]
        joint_axis = (rot @ axis[:, k, :, None])[..., 0]
        rot = rot @ _rotation(axis[:, k], q[:, k])
        spin = spin + joint_axis * dq[:, k:k + 1]
        rots.append(rot)
        origins.append(origin)
        axes.append(joint_axis)
        centres.append(origin + (rot @ com[:, k, :, None])[..., 0])
        omegas.append(spin)
        pos = origin
    world_inertia = [rots[k] @ inertia[:, k] @ rots[k].transpose(-1, -2) for k in range(n)]
    m = torch.zeros((worlds, n, n), dtype=q.dtype, device=q.device)
    for k in range(n):
        jv = torch.stack([torch.cross(axes[i], centres[k] - origins[i], dim=-1) for i in range(k + 1)], 1)
        a = torch.stack(axes[: k + 1], 1)
        block = mass[:, k, None, None] * (jv @ jv.transpose(-1, -2)) + a @ world_inertia[k] @ a.transpose(-1, -2)
        m = m + torch.nn.functional.pad(block, (0, n - k - 1, 0, n - k - 1))
    alpha_parent = torch.zeros_like(pos)
    acc_parent = torch.zeros_like(pos)
    acc_parent[:, 2] = gravity
    pos_parent = torch.zeros_like(pos)
    omega_parent = torch.zeros_like(pos)
    forces, moments = [], []
    for k in range(n):
        alpha = alpha_parent + torch.cross(omega_parent, axes[k] * dq[:, k:k + 1], dim=-1)
        lever = origins[k] - pos_parent
        acc = acc_parent + torch.cross(alpha_parent, lever, dim=-1) + torch.cross(omega_parent, torch.cross(omega_parent, lever, dim=-1), dim=-1)
        arm = centres[k] - origins[k]
        acc_centre = acc + torch.cross(alpha, arm, dim=-1) + torch.cross(omegas[k], torch.cross(omegas[k], arm, dim=-1), dim=-1)
        forces.append(mass[:, k, None] * acc_centre)
        moments.append((world_inertia[k] @ alpha[..., None])[..., 0]
                       + torch.cross(omegas[k], (world_inertia[k] @ omegas[k][..., None])[..., 0], dim=-1))
        alpha_parent, acc_parent, pos_parent, omega_parent = alpha, acc, origins[k], omegas[k]
    bias = []
    child_force = torch.zeros_like(pos)
    child_moment = torch.zeros_like(pos)
    child_origin = torch.zeros_like(pos)
    for k in reversed(range(n)):
        total = (moments[k] + torch.cross(centres[k] - origins[k], forces[k], dim=-1) + child_moment
                 + torch.cross(child_origin - origins[k], child_force, dim=-1))
        child_force, child_moment, child_origin = forces[k] + child_force, total, origins[k]
        bias.append((axes[k] * total).sum(-1))
    return m, torch.stack(bias[::-1], dim=1)


REACH, ACCIDENT, TOUCH = 0, 1, 2
RETREAT = 0.05  # rad: how far the brake can pull the targets back past where the arm is


def braked(targets: Tensor, measured: Tensor, action: Tensor) -> Tensor:
    """The brake action in [-1, 1] as b = 1.5 * action, from 0 (off, and for negative actions) to 1.5.

    Up to 1 it pulls every target from the goal side towards where the arm is now: 1 stops the arm where it stands.
    Beyond 1 it backs off: the targets move past the arm, away from the goal, by up to RETREAT on the joint furthest
    from its target.
    """

    b = (1.5 * action).clamp(0.0, 1.5)[:, None]
    blended = targets + b.clamp(max=1.0) * (measured - targets)
    away = measured - targets
    back = away / away.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    return blended + (b - 1.0).clamp(min=0.0) / 0.5 * RETREAT * back
TOUCH_BAND = (0.5, 10.0)  # N: touching the surface, not pressing into it
TOUCH_PEAK = 20.0  # N: the touch must not come in hard
RELEASE_TIME = 0.3  # s: after an accident, the contact must be released this soon
RELEASED = 0.5  # N: below this, the arm has let go
CONTACT_UNIT = 10.0  # N per unit in observations and rewards


@dataclass
class Arm3DSummary(ChainSummary):
    position_error: Tensor  # [worlds] worst tool position error over the final window, m
    direction_error: Tensor  # [worlds] worst tool direction error over the final window, rad
    mode: Tensor | None = None  # [worlds] REACH, ACCIDENT or TOUCH
    peak_force: Tensor | None = None  # [worlds] highest contact force of the episode, N
    impulse: Tensor | None = None  # [worlds] contact force integrated over the episode, N s


class Arm3DEnv(ChainEnv):
    """Tool-pose goals on a population of five-joint 3D arms, with `ChainEnv`'s reward shape and success rule.

    Observation (55): q/pi, dq/5, current/rated, previous command (n each); the pose error the encoders see:
    tool position (3, in POSITION_UNIT), tool direction (3), wrist roll (1), and per joint towards the nearest
    inverse-kinematics solution (n); the goal: tool point/0.5 m (3), direction (3), roll/pi (1), and that
    solution/pi (n); where the encoders say the tool is: point/0.5 m (3), direction (3); the arm's link
    lengths/0.25 m (n). Privileged: observation, true q/pi, dq/5 (n each), true pose error (7), hidden
    condition (12n). Errors are in joint-like units, so the usual 0.03 tolerance means 5 mm and 30 mrad.
    """

    task_code = TASK_CODE
    components = 3  # tool position, tool direction, wrist roll

    def __init__(self, worlds: int, *, device: str = "cuda:0", seed: int = 0, severity: float = 1.0,
                 episode_ticks: int = 300, changes: bool = True, reward_tolerance: float = TOLERANCE,
                 still_weight: float = 1.0, roughness_weight: float = 4.0, mixed: bool = False, servo: bool = False,
                 hold_weight: float = 0.0, stiffness: bool = False, accidents: float = 0.0, touches: float = 0.0) -> None:
        self.worlds, self.limbs, self.n, self.limb_joints = worlds, 1, JOINTS, JOINTS
        # Mixed: every robot gets its own severity, from flawless to `severity`, so training on worn arms does not
        # cost precision on good ones.
        self.mixed = mixed
        # Servo: the joints are position servos, as on nearly every cheap arm, and the policy's action moves their
        # targets around the inverse-kinematics solution; otherwise it commands torque directly.
        self.servo = servo
        # Stiffness: a second action per joint scales that servo's gain by 2^action (0.5-2x), the gain register LeRobot
        # lowers "to avoid shakiness"; the network may soften a hunting servo while it holds.
        self.stiffness = stiffness and servo
        self.task_code = TASK_CODE + (2 if self.stiffness else 1 if servo else 0)
        self.action_dim = 2 * self.n if self.stiffness else self.n
        # Hold: an L1 charge on every change of the commanded targets. Unlike the squared roughness charge it makes
        # exactly-still targets worth having: a servo follows even a 1 mrad jitter, and a worn one never settles.
        self.hold_weight = hold_weight
        # Contact tasks (servo arms): in a share of episodes an unseen post stands in the tool's way (the controller must
        # recognize the blow from its feel, stop and let go), and in another share it is told to find a surface along
        # the tool's direction somewhere before the goal and to touch it gently.
        self.accidents, self.touches = accidents, touches
        self.contact_tasks = servo and (accidents > 0 or touches > 0)
        # With contact tasks the policy also has a brake (see `braked`): stop where it stands, or back off.
        if self.contact_tasks:
            self.task_code = TASK_CODE + 3
            self.action_dim = self.action_dim + 1
        self.device, self.severity, self.episode_ticks, self.changes = device, severity, episode_ticks, changes
        self.reward_tolerance, self.still_weight, self.roughness_weight = reward_tolerance, still_weight, roughness_weight
        self.rng = np.random.default_rng([seed, 3000 + JOINTS])
        self.torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
        if self.torch_device.type == "cuda":
            wp.init()
            wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream(self.torch_device)), device)
        self.nominal_torques = NOMINAL_TORQUES
        self._torque = torch.as_tensor(NOMINAL_TORQUES, dtype=torch.float32, device=self.torch_device)
        self.initial_std = [0.3 if servo else 0.5] * self.action_dim
        if self.contact_tasks:
            self.initial_std[-1] = 1.0  # the brake has to be tried hard enough to discover backing off
        self.observation_dim = 4 * self.n + 7 + self.n + 7 + self.n + 6 + self.n + (1 if self.contact_tasks else 0)
        self.error_slice = (4 * self.n, 7 + self.n)
        self.privileged_dim = self.observation_dim + 2 * self.n + 7 + 12 * self.n + (5 if self.contact_tasks else 0)

    def _hidden(self, links: np.ndarray, joints: np.ndarray) -> Tensor:
        return self._tensor(np.concatenate((joints["torque_scale"] / NOMINAL_TORQUES, joints["coulomb"] * 4.0,
                                            joints["half_gap"] * 50.0, joints["damping"] * 10.0, joints["delay_steps"] / 30.0,
                                            links["mass"], links["gear_loss"] * 4.0, links["bearing_loss"] * 7.0,
                                            links["compliance"] * NOMINAL_TORQUES / BEND_AT_RATED,
                                            joints["no_load_speed"] / 10.0, joints["servo_gain"] / 25.0,
                                            joints["servo_damping"] * 50.0), axis=1))

    def _poses(self, links: np.ndarray, tip: np.ndarray) -> np.ndarray:
        """Joint configurations [worlds, n] in the goal span whose tool point clears the table."""

        q = self.rng.uniform(-GOAL_SPAN, GOAL_SPAN, (self.worlds, self.n))
        t = torch.as_tensor
        for _ in range(20):
            point, _ = arm3d_tool(t(q), t(links["joint_pos"]).double(), t(links["axis"]).double(), t(tip).double())
            low = (point[:, 2] < TABLE_CLEARANCE).numpy()
            if not low.any():
                break
            q[low] = self.rng.uniform(-GOAL_SPAN, GOAL_SPAN, (int(low.sum()), self.n))
        return q

    def reset(self) -> Tensor:
        severity = self.rng.uniform(0.0, self.severity, self.worlds) if self.mixed else self.severity
        links, joints, tip = sample_arm3d_population(self.rng, self.worlds, severity=severity)
        if self.servo:
            sample_servos(self.rng, joints)
        changed = sample_arm3d_change(self.rng, links, joints, tip, severity=severity)
        ticks = self.episode_ticks
        change_tick = np.where(self.rng.random(self.worlds) < (0.5 if self.changes else 0.0),
                               self.rng.integers(20, ticks - 60, self.worlds), np.iinfo(np.int32).max)
        start = self._poses(links, tip)
        joint_goals = np.stack((self._poses(links, tip), self._poses(links, tip)))
        regoal = np.where(self.rng.random(self.worlds) < 0.7, self.rng.integers(80, ticks - 120, self.worlds), ticks + 1)
        obstacles, mode, surface = self._contact_tasks(links, tip, start, joint_goals[0])
        regoal = np.where(mode == REACH, regoal, ticks + 1)  # contact tasks have one goal
        self.batch = Arm3DBatch(links, joints, tip, device=self.device, seed=int(self.rng.integers(2**31)),
                                changed=changed, change_tick=change_tick, obstacles=obstacles)
        self.batch.reset(start)
        self.links, self.joints, self.changed, self.change_tick, self.regoal_tick = links, joints, changed, change_tick, regoal
        self._tensors: dict[int, tuple[np.ndarray, np.ndarray, dict[str, tuple[Tensor, Tensor]]]] = {}
        self.tip_offset = tip
        # The arm as its maker describes it: kinematics for the controller's own reckoning, and lengths as inputs.
        self._joint_pos, self._axis = self._tensor(links["joint_pos"]), self._tensor(links["axis"])
        self._tip_offset = self._tensor(tip)
        self._geometry = self._tensor(np.concatenate((links["joint_pos"][:, 1:, 2], tip[:, 2:]), axis=1) / 0.25)
        # Goals are poses the rigid arm reaches at sampled joint angles; a bending arm must aim past them.
        self._goals = self._tensor(joint_goals)
        self._pose_goals = torch.stack([torch.cat((*arm3d_tool(g, self._joint_pos, self._axis, self._tip_offset),
                                                   g[:, -1:]), dim=1) for g in self._goals])
        # Touch: the goal the controller is told lies beyond the surface; it is judged on the surface point.
        self.mode, self._mode = mode, self._tensor(mode).long()
        self._surface = self._tensor(surface)
        self._judged = self._pose_goals.clone()
        touching = self._mode == TOUCH
        self._judged[0, touching, :3] = self._surface[touching]
        self._contact = wp.to_torch(self.batch.contact)
        self._first_contact = torch.full((self.worlds,), float(ticks + 1), device=self.torch_device)
        self._released = torch.zeros(self.worlds, dtype=torch.bool, device=self.torch_device)
        self._window_force = torch.zeros((self.worlds, 2), device=self.torch_device)  # min and max over the window
        self._window_force[:, 0] = torch.inf
        self._impulse = torch.zeros(self.worlds, device=self.torch_device)
        self._regoal = self._tensor(regoal)
        self._solution = self._goals[0].clone()
        self._solution_phase = torch.ones(self.worlds, dtype=torch.bool, device=self.torch_device)  # none chosen yet
        self._change_tick = self._tensor(np.minimum(change_tick, ticks + 1))
        self._bias = self._tensor(joints["enc_bias"])
        self._hidden_before, self._hidden_after = self._hidden(links, joints), self._hidden(*changed)
        self._observation = wp.to_torch(self.batch.observation)
        self._truth = wp.to_torch(self.batch.truth)
        self._metrics = wp.to_torch(self.batch.metrics)
        self._tip = wp.to_torch(self.batch.tip)
        self._tool_axis = wp.to_torch(self.batch.tool_axis)
        self.tick = 0
        self._previous = torch.zeros((self.worlds, self.n), device=self.torch_device)
        self._previous_smooth = torch.zeros((self.worlds, self.action_dim), device=self.torch_device)
        self._previous_hold = torch.zeros((self.worlds, self.action_dim), device=self.torch_device)
        zeros = torch.zeros(self.worlds, device=self.torch_device)
        self._sum_error, self._sum_rough, self._return = zeros.clone(), zeros.clone(), zeros.clone()
        self._window_error = torch.zeros((self.worlds, self.components), device=self.torch_device)
        self._window_speed = torch.zeros((self.worlds, self.n), device=self.torch_device)
        return self._observe()

    def _contact_tasks(self, links: np.ndarray, tip: np.ndarray, start: np.ndarray,
                       goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obstacles [worlds, MAX_OBSTACLES], mode per world, and the touch surface point (m) per world."""

        obstacles = np.zeros((self.worlds, MAX_OBSTACLES), dtype=Obstacle.numpy_dtype())
        mode = np.full(self.worlds, REACH)
        surface = np.zeros((self.worlds, 3))
        if not self.contact_tasks:
            return obstacles, mode, surface
        draw = self.rng.random(self.worlds)
        mode[draw < self.accidents + self.touches] = TOUCH
        mode[draw < self.accidents] = ACCIDENT
        obstacles["stiffness"] = self.rng.uniform(3e3, 3e4, (self.worlds, MAX_OBSTACLES))  # foam to wood
        # Real contacts dissipate: without this a servo pushing into a stiff surface rings against it.
        obstacles["damping"] = 0.6 * np.sqrt(obstacles["stiffness"])
        obstacles["friction"] = self.rng.uniform(0.1, 0.6, (self.worlds, MAX_OBSTACLES))
        t = torch.as_tensor
        geometry = (t(links["joint_pos"]).double(), t(links["axis"]).double(), t(tip).double())
        before, after = arm3d_points(t(start), *geometry), arm3d_points(t(goal), *geometry)
        radius = t(links["radius"]).double()
        point_radius = torch.cat((radius[:, 1:], radius[:, -1:]), dim=1)  # joints 1.. and the tool point, per link
        # Touch: a wall across the tool's direction, 2-6 cm short of the goal; the arm starts clear of it.
        direction = arm3d_tool(t(goal), *geometry)[1]
        short = t(self.rng.uniform(0.02, 0.06, self.worlds))
        point = after[:, -1] - direction * short[:, None]  # where the tool point rests when it touches
        wall = point + direction * radius[:, -1:]  # the tool's capsule reaches one radius beyond its point
        clear = ((before - wall[:, None]) * -direction[:, None]).sum(-1)[:, 1:] > point_radius + 0.01
        touch = (mode == TOUCH) & clear.all(dim=1).numpy()
        mode[(mode == TOUCH) & ~touch] = REACH
        obstacles["kind"][touch, 0] = HALF_SPACE
        obstacles["a"][touch, 0] = wall[touch].numpy()
        obstacles["b"][touch, 0] = -direction[touch].numpy()
        surface[touch] = point[touch].numpy()
        # Accident: something in the tool's way, clear of the arm at the start and at the goal. Half the time a wall
        # across the straight path (a box, a person), otherwise a thick post or limb.
        along = t(self.rng.uniform(0.3, 0.7, self.worlds))
        centre = before[:, -1] + (after[:, -1] - before[:, -1]) * along[:, None]
        heading = torch.nn.functional.normalize(after[:, -1] - before[:, -1], dim=1)
        tilt = t(self.rng.normal(0.0, 1.0, (self.worlds, 3)))
        tilt[:, 2] = tilt[:, 2].abs() + 1.0  # mostly upright
        tilt = tilt / tilt.norm(dim=1, keepdim=True)
        half = t(self.rng.uniform(0.05, 0.15, self.worlds))[:, None]
        a, b = centre - tilt * half, centre + tilt * half
        size = t(self.rng.uniform(0.03, 0.06, self.worlds))
        margin = (size + 0.01)[:, None] + radius[:, 1:]
        apart = torch.stack([_segment_distance(pose[:, 1:-1], pose[:, 2:], a[:, None].expand(-1, self.n - 1, -1),
                                               b[:, None].expand(-1, self.n - 1, -1)) for pose in (before, after)])
        post_clear = (apart > margin).all(dim=(0, 2))
        # A wall: its surface through the crossing point, facing back towards the start; the start pose must be in front.
        facing = ((before[:, 1:] - centre[:, None]) * -heading[:, None]).sum(-1) > point_radius + 0.01
        wall = t(self.rng.random(self.worlds) < 0.5)
        travel = (after[:, -1] - before[:, -1]).norm(dim=1)
        clear = torch.where(wall, facing.all(dim=1), post_clear)
        accident = (mode == ACCIDENT) & clear.numpy() & (travel > 0.08).numpy()
        mode[(mode == ACCIDENT) & ~accident] = REACH
        walls, posts = accident & wall.numpy(), accident & ~wall.numpy()
        obstacles["kind"][walls, 0] = HALF_SPACE
        obstacles["a"][walls, 0], obstacles["b"][walls, 0] = centre[walls].numpy(), -heading[walls].numpy()
        obstacles["kind"][posts, 0] = CAPSULE
        obstacles["a"][posts, 0], obstacles["b"][posts, 0] = a[posts].numpy(), b[posts].numpy()
        obstacles["radius"][posts, 0] = size[posts].numpy()
        return obstacles, mode, surface

    def _later(self) -> Tensor:
        return (self.tick >= self._regoal)[:, None]

    def goal(self) -> Tensor:
        """Tool point (3, m), tool direction (3), wrist roll (1, encoder units)."""

        return torch.where(self._later(), self._pose_goals[1], self._pose_goals[0])

    def joint_goal(self) -> Tensor:
        """Joint angles at which the rigid arm reaches the goal pose; for classical teachers, never for policies.

        Of the solutions within the joint limits, the one nearest the measured pose when the goal appeared. Policies
        see how far each joint is from it: the arm's inverse kinematics, worked out from its configured dimensions
        like the forward kinematics, is a hint; the network still decides how to move and how far to aim past it.
        """

        later = self._later()[:, 0]
        stale = self._solution_phase != later
        if stale.any():
            candidates = arm3d_inverse(self.goal(), self._joint_pos, self._tip_offset)
            limits = torch.as_tensor(LIMITS, dtype=torch.float32, device=self.torch_device) * 0.97
            valid = ((candidates >= limits[:, 0]) & (candidates <= limits[:, 1])).all(dim=2)
            here = self.measured_q
            distance = torch.where(valid, (candidates - here[:, None]).abs().sum(dim=2), torch.inf)
            nearest = candidates[torch.arange(self.worlds, device=self.torch_device), distance.argmin(dim=1)]
            self._solution = torch.where(stale[:, None], nearest, self._solution)
            self._solution_phase = later.clone()
        return self._solution

    def _pose_error(self, point: Tensor, direction: Tensor, roll: Tensor) -> Tensor:
        goal = self.goal()
        return torch.cat(((goal[:, :3] - point) / POSITION_UNIT, goal[:, 3:6] - direction, goal[:, 6:] - roll[:, None]), dim=1)

    def _true_error(self) -> Tensor:
        """Pose error as judged: for a touch, towards the surface point rather than the goal the controller was told."""

        judged = torch.where(self._later(), self._judged[1], self._judged[0])
        roll = self._truth[:, self.n - 1] + self._bias[:, -1]
        return torch.cat(((judged[:, :3] - self._tip) / POSITION_UNIT, judged[:, 3:6] - self._tool_axis,
                          judged[:, 6:] - roll[:, None]), dim=1)

    def _observe(self) -> Tensor:
        raw, n = self._observation, self.n
        self.measured_q, self.velocity = raw[:, :n].clone(), raw[:, n:2 * n].clone()
        point, direction = arm3d_tool(self.measured_q, self._joint_pos, self._axis, self._tip_offset)
        goal, solution = self.goal(), self.joint_goal()
        return torch.cat((self.measured_q / torch.pi, self.velocity / 5.0, raw[:, 2 * n:3 * n] / self._torque, self._previous,
                          self._pose_error(point, direction, self.measured_q[:, -1]), solution - self.measured_q,
                          goal[:, :3] / 0.5, goal[:, 3:6], goal[:, 6:] / torch.pi, solution / torch.pi,
                          point / 0.5, direction, self._geometry,
                          *(((self._mode == TOUCH).float()[:, None],) if self.contact_tasks else ())), dim=1)

    def privileged(self, observation: Tensor) -> Tensor:
        n = self.n
        hidden = torch.where((self.tick >= self._change_tick)[:, None], self._hidden_after, self._hidden_before)
        extra = ()
        if self.contact_tasks:
            extra = (self._contact[:, :1] / CONTACT_UNIT, self._contact[:, 2:3] / (2.0 * CONTACT_UNIT),
                     torch.nn.functional.one_hot(self._mode, 3).float())
        return torch.cat((observation, self._truth[:, :n] / torch.pi, self._truth[:, n:] / 5.0, self._true_error(), hidden,
                          *extra), dim=1)

    def step(self, action: Tensor, reference: Tensor | None = None) -> tuple[Tensor, Tensor]:
        n = self.n
        action = action.clamp(-1.0, 1.0)
        smooth = action if reference is None else reference.clamp(-1.0, 1.0)
        if self.servo:
            limits = torch.as_tensor(LIMITS, dtype=torch.float32, device=self.torch_device)
            targets = self.joint_goal() + SERVO_REACH * action[:, :n]
            if self.contact_tasks:
                targets = braked(targets, self.measured_q, action[:, -1])
            targets = torch.maximum(torch.minimum(targets, limits[:, 1]), limits[:, 0])
            self.batch.step(targets, torch.exp2(action[:, n:2 * n]) if self.stiffness else None)
        else:
            self.batch.step(action)
        self.tick += 1
        pose_error = self._true_error()
        error = torch.stack((pose_error[:, :3].norm(dim=1), pose_error[:, 3:6].norm(dim=1), pose_error[:, 6].abs()), dim=1)
        speed = self._truth[:, n:].abs()
        rough = (smooth - self._previous_smooth).square().sum(dim=1)
        self._previous_smooth = smooth
        limit = self._metrics[:, 2 + n:2 + 2 * n].sum(dim=1)
        arrived = ((error < self.reward_tolerance).all(dim=1) & (speed.amax(dim=1) < SETTLED_SPEED)).float()
        close = torch.exp(-error / self.reward_tolerance).mean(dim=1)
        # The chain's reward per pair of joints; the pose error counts like two joints' worth.
        scale = 2.0 / n
        reward = 0.1 * (-error.sum(dim=1) * (2.0 / self.components) + close + arrived - self.roughness_weight * rough * scale
                        - 0.002 * self._metrics[:, 0] / 0.01 * scale - limit * scale)
        if self.hold_weight:
            reward = reward - 0.1 * self.hold_weight * (smooth - self._previous_hold).abs().sum(dim=1) * scale
        self._previous_hold = smooth
        if self.still_weight:
            reward = reward + 0.1 * self.still_weight * (close * torch.exp(-speed.amax(dim=1) / (2.0 * SETTLED_SPEED))
                                                         - 0.2 * speed.mean(dim=1))
        if self.contact_tasks:
            reward = self._contact_reward(reward, speed)
        self._previous = action[:, :n]
        self._sum_error += error.mean(dim=1)
        self._sum_rough += rough * scale
        self._return += reward
        if self.tick > self.episode_ticks - FINAL_WINDOW:
            self._window_error = torch.maximum(self._window_error, error)
            self._window_speed = torch.maximum(self._window_speed, speed)
        return self._observe(), reward

    def _contact_reward(self, reward: Tensor, speed: Tensor) -> Tensor:
        """Accident: after the first blow, only letting go and keeping still count. Touch: come in softly, then rest
        against the surface in the touch band. Also tracks what the summary judges."""

        force, peak = self._contact[:, 0], self._contact[:, 1]
        self._impulse += self._contact[:, 3]
        touched = peak > RELEASED
        self._first_contact = torch.where(touched & (self._first_contact > self.tick), float(self.tick), self._first_contact)
        hit = (self._mode == ACCIDENT) & (self._first_contact <= self.tick)
        in_time = self.tick <= self._first_contact + RELEASE_TIME / 0.01
        self._released = self._released | (hit & ~touched & in_time)
        after_blow = 0.1 * (-peak / CONTACT_UNIT - 0.5 * speed.mean(dim=1) + (~touched).float() * torch.exp(
            -speed.amax(dim=1) / (2.0 * SETTLED_SPEED)))
        reward = torch.where(hit, after_blow, reward)
        touch = self._mode == TOUCH
        low, high = TOUCH_BAND
        band = ((force >= low) & (force <= high)).float()
        soft = 0.1 * (band - torch.relu(force - high) / CONTACT_UNIT - torch.relu(peak - TOUCH_PEAK) / CONTACT_UNIT)
        reward = torch.where(touch, reward + soft, reward)
        if self.tick > self.episode_ticks - FINAL_WINDOW:
            self._window_force[:, 0] = torch.minimum(self._window_force[:, 0], force)
            self._window_force[:, 1] = torch.maximum(self._window_force[:, 1], force)
        return reward

    def summary(self) -> Arm3DSummary:
        still = (self._window_speed < SETTLED_SPEED).all(dim=1)
        ok = (self._window_error < TOLERANCE).all(dim=1) & still
        if self.contact_tasks:
            low, high = TOUCH_BAND
            touched = ((self._window_error < TOLERANCE).all(dim=1) & still & (self._window_force[:, 0] >= low)
                       & (self._window_force[:, 1] <= high) & (self._contact[:, 2] <= TOUCH_PEAK))
            hit = self._first_contact <= self.tick
            let_go = self._released & (self._window_force[:, 1] <= 2.0 * RELEASED) & still
            ok = torch.where(self._mode == TOUCH, touched, torch.where((self._mode == ACCIDENT) & hit, let_go, ok))
        ticks = float(self.tick)
        return Arm3DSummary(ok, ok[:, None], self._window_error.amax(dim=1), self._sum_error / ticks, self._sum_rough / ticks,
                            self._return, self._window_error[:, 0] * POSITION_UNIT, self._window_error[:, 1],
                            self._mode.clone(), self._contact[:, 2].clone(), self._impulse.clone())

    def true_dynamics(self, q: Tensor, dq: Tensor) -> tuple[Tensor, Tensor]:
        links = self._current(self.links, self.changed[0], q.device)
        m, bias = arm3d_dynamics(q, dq, links["joint_pos"], links["axis"], links["com"], links["mass"], links["inertia"])
        # Armature: exact for stiff gears; with backlash a fair approximation while the teeth are engaged.
        joints = self._current(self.joints, self.changed[1], q.device)
        return m + torch.diag_embed(joints["rotor_inertia"]), bias
