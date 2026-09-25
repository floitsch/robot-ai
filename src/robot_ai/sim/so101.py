"""The SO-101 (LeRobot's open desktop arm) in the 3D simulator.

The kinematics and inertias come from TheRobotStudio's model,
https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf (onshape-to-robot, new
calibration: each joint's zero is the middle of its range). Its five arm joints, from the base: shoulder_pan (yaw),
shoulder_lift, elbow_flex, wrist_flex (pitches), wrist_roll. The gripper jaw is not a controlled joint here: its mass
is lumped into the gripper, closed. The tool point is the URDF's gripper_frame.

URDF joints carry a fixed rotation (origin rpy) between links; the simulator's links do not. Any serial chain can be
rewritten without them: with P_k the product of the fixed rotations up to joint k, joint k's axis becomes P_k a_k, its
position in the previous link P_{k-1} p_k, and link k's centre of mass and inertia P_k c_k and P_k I_k P_k^T. The
rewritten chain moves exactly like the URDF one (tested against it).

Every joint is a Feetech STS3215 position servo (1/345 gearing, 12-bit magnetic encoder); the follower kit's
rating is taken as 2.5 N m stall and 5 rad/s no load, and robots vary around it.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from .arm3d_batch import Link3D

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Per joint: origin xyz and rpy in the parent link, axis in the joint frame, limits (rad).
JOINT_XYZ = np.array([[0.0388353, -8.97657e-09, 0.0624], [-0.0303992, -0.0182778, -0.0542], [-0.11257, -0.028, 1.73763e-16],
                      [-0.1349, 0.0052, 3.62355e-17], [5.55112e-17, -0.0611, 0.0181]])
JOINT_RPY = np.array([[3.14159, 4.18253e-17, -3.14159], [-1.5708, -1.5708, 0.0], [-3.63608e-16, 8.74301e-16, 1.5708],
                      [4.02456e-15, 8.67362e-16, -1.5708], [1.5708, 0.0486795, 3.14159]])
JOINT_AXIS = np.array([[0.0, 0.0, 1.0]] * 5)
LIMITS = np.array([[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69], [-1.65806, 1.65806], [-2.74385, 2.84121]])
# Per moving link (shoulder, upper arm, lower arm, wrist, gripper): mass (kg), centre of mass (m) and inertia about it
# (kg m^2: xx, xy, xz, yy, yz, zz), in the URDF link frame.
LINK_MASS = np.array([0.100006, 0.103, 0.104, 0.079, 0.087])
LINK_COM = np.array([[-0.0307604, -1.66727e-05, -0.0252713], [-0.0898471, -0.00838224, 0.0184089],
                     [-0.0980701, 0.00324376, 0.0182831], [-0.000103312, -0.0386143, 0.0281156],
                     [0.000213627, 0.000245138, -0.025187]])
LINK_INERTIA = np.array([[8.3759e-05, 7.55525e-08, -1.16342e-06, 8.10403e-05, 1.54663e-07, 2.39783e-05],
                         [4.08002e-05, -1.97819e-05, -4.03016e-08, 0.000147318, 8.97326e-09, 0.000142487],
                         [2.87438e-05, 7.41152e-06, 1.26409e-06, 0.000159844, -4.90188e-08, 0.00014529],
                         [3.68263e-05, 1.7893e-08, -5.28128e-08, 2.5391e-05, 3.6412e-06, 2.1e-05],
                         [2.75087e-05, -3.35241e-07, -5.7352e-06, 4.33657e-05, -5.17847e-08, 3.45059e-05]])
# The moving jaw (closed): its joint origin in the gripper link, and its own mass, centre of mass and inertia.
JAW_XYZ, JAW_RPY = np.array([0.0202, 0.0188, -0.0234]), np.array([1.5708, -5.24284e-08, -1.41553e-15])
JAW_MASS, JAW_COM = 0.012, np.array([-0.00157495, -0.0300244, 0.0192755])
JAW_INERTIA = np.array([6.61427e-06, -3.19807e-07, -5.90717e-09, 1.89032e-06, -1.09945e-07, 5.28738e-06])
TOOL_XYZ = np.array([-0.0079, -0.000218121, -0.0981274])  # gripper_frame in the gripper link
RADII = np.array([0.025, 0.022, 0.022, 0.02, 0.02])  # collision capsules around the links (servo bodies, printed shells)
NOMINAL_TORQUES = np.full(5, 2.5)  # N m stall at the joint
NO_LOAD_SPEED = 5.0  # rad/s


def rotation(rpy: np.ndarray) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed axes x, y, z) as a rotation matrix."""

    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _tensor(values: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = values
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _gripper_with_jaw() -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia of the gripper link with its closed jaw, in the gripper link frame."""

    turn = rotation(JAW_RPY)
    jaw_com = JAW_XYZ + turn @ JAW_COM
    jaw_inertia = turn @ _tensor(JAW_INERTIA) @ turn.T
    mass = LINK_MASS[-1] + JAW_MASS
    com = (LINK_MASS[-1] * LINK_COM[-1] + JAW_MASS * jaw_com) / mass

    def shifted(m: float, inertia: np.ndarray, d: np.ndarray) -> np.ndarray:
        return inertia + m * (d @ d * np.eye(3) - np.outer(d, d))

    inertia = shifted(LINK_MASS[-1], _tensor(LINK_INERTIA[-1]), LINK_COM[-1] - com) + shifted(JAW_MASS, jaw_inertia, jaw_com - com)
    return mass, com, inertia


def so101_links(worlds: int) -> tuple[np.ndarray, np.ndarray]:
    """Link3D records [worlds, 5] and the tool offset [worlds, 3] of the nominal SO-101."""

    links = np.zeros((worlds, 5), dtype=Link3D.numpy_dtype())
    grip_mass, grip_com, grip_inertia = _gripper_with_jaw()
    masses = np.append(LINK_MASS[:-1], grip_mass)
    coms = np.vstack((LINK_COM[:-1], grip_com))
    inertias = [_tensor(v) for v in LINK_INERTIA[:-1]] + [grip_inertia]
    previous = np.eye(3)
    for k in range(5):
        current = previous @ rotation(JOINT_RPY[k])
        links["joint_pos"][:, k] = previous @ JOINT_XYZ[k]
        links["axis"][:, k] = current @ JOINT_AXIS[k]
        links["com"][:, k] = current @ coms[k]
        links["inertia"][:, k] = current @ inertias[k] @ current.T
        links["mass"][:, k] = masses[k]
        links["radius"][:, k] = RADII[k]
        previous = current
    tip = np.tile(previous @ TOOL_XYZ, (worlds, 1))
    return links, tip


def urdf_tool_point(q: np.ndarray) -> np.ndarray:
    """The tool point for joint angles [5], straight from the URDF's own frames (for tests)."""

    frame, position = np.eye(3), np.zeros(3)
    for k in range(5):
        position = position + frame @ JOINT_XYZ[k]
        frame = frame @ rotation(JOINT_RPY[k])
        c, s = np.cos(q[k]), np.sin(q[k])
        frame = frame @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # every axis is the joint frame's z
    return position + frame @ TOOL_XYZ
