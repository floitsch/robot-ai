"""Sample populations of distinct, imperfect arms for `ArmBatch`."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from .arm_batch import JOINTS, ArmParams
from .joint_model import JointParams

NOMINAL_LENGTHS = np.array([0.30, 0.25])
NOMINAL_MASSES = np.array([0.60, 0.40])
NOMINAL_TORQUES = np.array([6.0, 3.0])
JOINT_LIMITS = np.array([[-2.4, 2.4], [-2.6, 2.6]])
BASE_DAMPING = 0.02


def _arm_array(lengths: np.ndarray, masses: np.ndarray, payload: np.ndarray, gravity: float) -> np.ndarray:
    """Rod links with a point payload at the tip of link two."""

    arms = np.zeros(len(lengths), dtype=ArmParams.numpy_dtype())
    rod_inertia = masses * lengths**2 / 12.0
    arms["l1"], arms["l2"] = lengths[:, 0], lengths[:, 1]
    arms["m1"], arms["lc1"], arms["i1"] = masses[:, 0], lengths[:, 0] / 2.0, rod_inertia[:, 0]
    total = masses[:, 1] + payload
    com = (masses[:, 1] * lengths[:, 1] / 2.0 + payload * lengths[:, 1]) / total
    arms["m2"], arms["lc2"] = total, com
    arms["i2"] = rod_inertia[:, 1] + masses[:, 1] * (com - lengths[:, 1] / 2.0) ** 2 + payload * (lengths[:, 1] - com) ** 2
    arms["gravity"] = gravity
    return arms


def _healthy_joints(worlds: int) -> np.ndarray:
    joints = np.zeros((worlds, JOINTS), dtype=JointParams.numpy_dtype())
    joints["torque_scale"] = NOMINAL_TORQUES
    joints["motor_alpha"] = 1.0
    joints["damping"] = BASE_DAMPING
    joints["bump0_width"] = joints["bump1_width"] = 0.1
    joints["mesh_stiffness"], joints["rotor_inertia"] = 100.0, 0.002
    joints["q_min"], joints["q_max"] = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    joints["cur_gain"] = 1.0
    return joints


def healthy_population(worlds: int, *, gravity: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """Identical nominal arms with ideal actuators and sensors."""

    lengths = np.tile(NOMINAL_LENGTHS, (worlds, 1))
    masses = np.tile(NOMINAL_MASSES, (worlds, 1))
    return _arm_array(lengths, masses, np.zeros(worlds), gravity), _healthy_joints(worlds)


def _per_world(severity: float | np.ndarray, worlds: int) -> tuple[np.ndarray, np.ndarray]:
    """Severity as [worlds] and [worlds, 1], from one number or one number per robot."""

    values = np.broadcast_to(np.asarray(severity, dtype=np.float64), (worlds,))
    return values, values[:, None]


def sample_population(rng: np.random.Generator, worlds: int, *, severity: float | np.ndarray = 1.0,
                      friction_probability: float = 0.7, physics_dt: float = 0.001,
                      gravity: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """Draw `worlds` different robots. `severity` in [0, 1], one value or one per robot, scales every imperfection.

    Each defect is present in only some robots and joints, so the population spans
    healthy arms, single faults, and compounded faults. Friction hides poor control:
    it holds a parked arm still. Training with a lower `friction_probability` stops a
    controller from relying on it. The random stream does not depend on this value.
    """

    shape = (worlds, JOINTS)
    per_robot, severity = _per_world(severity, worlds)

    def some(probability: float, low: float, high: float) -> np.ndarray:
        present = rng.random(shape) < probability
        return np.where(present, rng.uniform(low, high, shape), 0.0) * severity

    lengths = NOMINAL_LENGTHS * (1.0 + severity * rng.uniform(-0.15, 0.15, shape))
    masses = NOMINAL_MASSES * (1.0 + severity * rng.uniform(-0.20, 0.20, shape))
    payload = np.where(rng.random(worlds) < 0.5, rng.uniform(0.0, 0.15, worlds), 0.0) * per_robot
    arms = _arm_array(lengths, masses, payload, gravity)

    return arms, sample_joint_defects(rng, worlds, JOINTS, NOMINAL_TORQUES, JOINT_LIMITS, severity=severity,
                                       friction_probability=friction_probability, physics_dt=physics_dt)


def sample_joint_defects(rng: np.random.Generator, worlds: int, joints_per_robot: int, nominal_torques: np.ndarray,
                         limits: np.ndarray, *, severity: float | np.ndarray = 1.0, friction_probability: float = 0.7,
                         physics_dt: float = 0.001, friction_scale: float | np.ndarray = 1.0) -> np.ndarray:
    """Per-joint defects for `worlds` robots with `joints_per_robot` joints each; see `sample_population`.

    Friction and drag defects are sized for joints rated around 3 N m; `friction_scale` (per joint) resizes
    them for smaller or larger joints, whose gears and bearings are smaller or larger too.
    """

    shape = (worlds, joints_per_robot)
    severity = _per_world(np.asarray(severity, dtype=np.float64).reshape(-1) if np.ndim(severity) else severity, worlds)[1]

    def some(probability: float, low: float, high: float) -> np.ndarray:
        present = rng.random(shape) < probability
        return np.where(present, rng.uniform(low, high, shape), 0.0) * severity

    limits = np.broadcast_to(np.asarray(limits, dtype=np.float64), (joints_per_robot, 2))
    joints = np.zeros(shape, dtype=JointParams.numpy_dtype())
    joints["motor_alpha"], joints["bump0_width"], joints["bump1_width"] = 1.0, 0.1, 0.1
    joints["mesh_stiffness"], joints["rotor_inertia"], joints["cur_gain"] = 100.0, 0.002, 1.0
    joints["q_min"], joints["q_max"] = limits[:, 0], limits[:, 1]
    joints["torque_scale"] = nominal_torques * (1.0 - some(0.6, 0.0, 0.4))
    time_constant = some(0.6, 0.002, 0.040)
    joints["motor_alpha"] = np.where(time_constant > 0, 1.0 - np.exp(-physics_dt / np.maximum(time_constant, 1e-9)), 1.0)
    joints["delay_steps"] = np.rint(some(0.5, 0.0, 0.030) / physics_dt).astype(np.int32)
    joints["damping"] = BASE_DAMPING + some(0.6, 0.0, 0.10) * friction_scale
    joints["coulomb"] = some(friction_probability, 0.0, 0.25) * friction_scale
    joints["stribeck"] = some(0.5, 0.0, 0.6)
    for bump in ("bump0", "bump1"):
        joints[f"{bump}_mag"] = some(0.5 * friction_probability, 0.05, 0.40) * friction_scale
        joints[f"{bump}_center"] = rng.uniform(limits[:, 0], limits[:, 1], shape)
        joints[f"{bump}_width"] = rng.uniform(0.05, 0.25, shape)
    joints["half_gap"] = some(0.5, 0.002, 0.020)
    joints["mesh_stiffness"] = rng.uniform(60.0, 400.0, shape)
    joints["mesh_damping"] = rng.uniform(0.05, 0.5, shape)
    joints["rotor_inertia"] = rng.uniform(0.001, 0.004, shape)
    joints["enc_bias"] = severity * rng.uniform(-0.02, 0.02, shape)
    joints["enc_noise"] = some(0.8, 0.0, 0.002)
    joints["enc_quantum"] = np.where(rng.random(shape) < 0.6 * severity, 2.0 * np.pi / rng.choice([1024, 2048, 4096], shape), 0.0)
    joints["enc_delay_ticks"] = np.rint(some(0.4, 0.0, 3.0)).astype(np.int32)
    joints["cur_gain"] = 1.0 + severity * rng.uniform(-0.15, 0.15, shape)
    joints["cur_bias"] = severity * rng.uniform(-0.05, 0.05, shape)
    joints["cur_noise"] = some(0.8, 0.0, 0.05)
    return joints


def sample_change(rng: np.random.Generator, arms: np.ndarray, joints: np.ndarray, *,
                  severity: float | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The same robots after something changed: a payload picked up, a motor fading, a joint fouling."""

    worlds = len(arms)
    shape = (worlds, JOINTS)
    changed_arms, changed_joints = arms.copy(), joints.copy()
    per_robot, severity = _per_world(severity, worlds)
    added = np.where(rng.random(worlds) < 0.6, rng.uniform(0.05, 0.20, worlds), 0.0) * per_robot
    total = arms["m2"] + added
    com = (arms["m2"] * arms["lc2"] + added * arms["l2"]) / total
    changed_arms["i2"] = arms["i2"] + arms["m2"] * (com - arms["lc2"]) ** 2 + added * (arms["l2"] - com) ** 2
    changed_arms["m2"], changed_arms["lc2"] = total, com
    fading = np.where(rng.random(shape) < 0.3, rng.uniform(0.0, 0.4, shape), 0.0) * severity
    changed_joints["torque_scale"] = joints["torque_scale"] * (1.0 - fading)
    changed_joints["coulomb"] = joints["coulomb"] + np.where(rng.random(shape) < 0.3, rng.uniform(0.0, 0.2, shape), 0.0) * severity
    return changed_arms, changed_joints


def sample_push(rng: np.random.Generator, worlds: int, *, severity: float | np.ndarray = 1.0) -> np.ndarray:
    """Uncommanded, smoothly varying joint loads, as a moving neighbouring limb would cause.

    A neighbour accelerating its mount at a few m/s^2 loads these links with a few percent of rated motor
    torque, below about 1.5 Hz. Much stronger or faster loads move the light second link faster than any
    100 Hz controller with realistic delays can cancel, which tests the criterion rather than the controller.
    """

    waves = np.zeros((worlds, JOINTS, 6), dtype=np.float32)
    severity = _per_world(severity, worlds)[1]
    present = rng.random((worlds, 1)) < 0.7
    for offset, (low, high) in ((0, (0.1, 0.5)), (3, (0.5, 1.5))):
        waves[:, :, offset] = present * rng.uniform(0.0, 0.04, (worlds, JOINTS)) * NOMINAL_TORQUES * severity
        waves[:, :, offset + 1] = rng.uniform(low, high, (worlds, JOINTS))
        waves[:, :, offset + 2] = rng.uniform(0.0, 2.0 * np.pi, (worlds, JOINTS))
    return waves
