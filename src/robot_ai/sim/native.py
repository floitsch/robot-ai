"""Native MuJoCo reference backend for the rigid two-link arm."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..contracts import Command, Observation, PrivilegedRecord, RobotDescriptor
from .actuators import ActuatorModel, ActuatorSettings
from .mechanics import endpoint_velocity, forward_kinematics
from .sensors import SensorSettings, SensorSuite


def default_descriptor() -> RobotDescriptor:
    lengths = np.array([0.30, 0.25])
    masses = np.array([0.60, 0.40])
    inertias = np.array([masses[0] * lengths[0] ** 2 / 12, masses[1] * lengths[1] ** 2 / 12])
    return RobotDescriptor(lengths, masses, inertias, np.array([6.0, 3.0]), np.array([[-2.4, 2.4], [-2.6, 2.6]]))


def make_mjcf(descriptor: RobotDescriptor | None = None, *, timestep: float = 0.001) -> str:
    """Build the deterministic fixed-base MJCF fixture."""

    descriptor = descriptor or default_descriptor()
    l1, l2 = descriptor.link_lengths
    m1, m2 = descriptor.nominal_masses
    i1, i2 = descriptor.nominal_inertias
    (q1_min, q1_max), (q2_min, q2_max) = descriptor.joint_limits
    t1, t2 = descriptor.nominal_torque_scales
    return f"""<mujoco model=\"robot-ai-2r\">
  <compiler angle=\"radian\" coordinate=\"local\"/>
  <option timestep=\"{timestep:.9g}\" gravity=\"0 0 -9.81\" integrator=\"Euler\"/>
  <worldbody>
    <body name=\"link1\" pos=\"0 0 0\">
      <joint name=\"joint1\" type=\"hinge\" axis=\"0 -1 0\" range=\"{q1_min} {q1_max}\" limited=\"true\" damping=\"0.02\"/>
      <inertial pos=\"0 0 {-l1 / 2}\" mass=\"{m1}\" diaginertia=\"{i1} {i1} 1e-6\"/>
      <geom name=\"link1_geom\" type=\"capsule\" fromto=\"0 0 0 0 0 {-l1}\" size=\"0.018\" density=\"0\" contype=\"0\" conaffinity=\"0\"/>
      <body name=\"link2\" pos=\"0 0 {-l1}\">
        <joint name=\"joint2\" type=\"hinge\" axis=\"0 -1 0\" range=\"{q2_min} {q2_max}\" limited=\"true\" damping=\"0.02\"/>
        <inertial pos=\"0 0 {-l2 / 2}\" mass=\"{m2}\" diaginertia=\"{i2} {i2} 1e-6\"/>
        <geom name=\"link2_geom\" type=\"capsule\" fromto=\"0 0 0 0 0 {-l2}\" size=\"0.016\" density=\"0\" contype=\"0\" conaffinity=\"0\"/>
        <site name=\"endpoint\" pos=\"0 0 {-l2}\" size=\"0.005\"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name=\"motor1\" joint=\"joint1\" gear=\"1\" ctrllimited=\"true\" ctrlrange=\"{-t1} {t1}\"/>
    <motor name=\"motor2\" joint=\"joint2\" gear=\"1\" ctrllimited=\"true\" ctrlrange=\"{-t2} {t2}\"/>
  </actuator>
</mujoco>"""


@dataclass
class NativeArm:
    """Single-world native MuJoCo simulation with explicit public/truth reads."""

    descriptor: RobotDescriptor
    timestep: float = 0.001

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(make_mjcf(self.descriptor, timestep=self.timestep))
        self.data = mujoco.MjData(self.model)
        self.endpoint_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
        self.reset()

    def reset(self, q: np.ndarray | None = None, dq: np.ndarray | None = None) -> None:
        self.data.qpos[:] = 0.0 if q is None else np.asarray(q, dtype=np.float64)
        self.data.qvel[:] = 0.0 if dq is None else np.asarray(dq, dtype=np.float64)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, command: Command | np.ndarray) -> PrivilegedRecord:
        accepted = command if isinstance(command, Command) else Command.accepted(command)
        self.data.ctrl[:] = accepted.values * self.descriptor.nominal_torque_scales
        mujoco.mj_step(self.model, self.data)
        return self.truth()

    def truth(self) -> PrivilegedRecord:
        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()
        # MuJoCo's post-integrator site cache can still describe the state
        # before the final Euler position update. Derive the public truth from
        # the committed qpos so endpoint and joint records share one timestamp.
        endpoint = forward_kinematics(q, self.descriptor.link_lengths)
        velocity = endpoint_velocity(q, dq, self.descriptor.link_lengths)
        return PrivilegedRecord(q, dq, endpoint, velocity, np.zeros(3), np.zeros(4))

    def observation(self, time_s: float | None = None) -> Observation:
        truth = self.truth()
        # The nominal rigid public contract has no current/torque measurement.
        # Keep these channels aligned with Warp rollout/evaluation packets. The
        # imperfect P3 path publishes a current-like measurement through
        # SensorSuite instead.
        values = np.concatenate((truth.q, truth.dq, np.zeros(2), truth.endpoint_xz))
        return Observation(values, np.ones(8, dtype=bool), np.ones(8, dtype=bool), np.zeros(8),
                           self.data.time if time_s is None else time_s)


class ImperfectNativeArm:
    """Native reference arm with hidden actuator defects and packetized sensors."""

    def __init__(self, descriptor: RobotDescriptor, actuator: ActuatorSettings | None = None,
                 sensors: SensorSuite | None = None, *, timestep: float = 0.001) -> None:
        self.arm = NativeArm(descriptor, timestep=timestep)
        self.actuator = ActuatorModel(descriptor, actuator or ActuatorSettings.healthy(), timestep)
        self.sensors = sensors or SensorSuite(SensorSettings.healthy())

    def reset(self, q: np.ndarray | None = None, dq: np.ndarray | None = None) -> None:
        self.arm.reset(q, dq)
        self.actuator.reset()
        self.sensors.reset()
        self.sensors.advance(self.arm.data.time, self.arm.truth(), np.zeros(2))

    def step(self, command: Command | np.ndarray) -> PrivilegedRecord:
        torque, resistance = self.actuator.update(command, self.arm.data.qpos, self.arm.data.qvel)
        self.arm.data.ctrl[:] = torque
        self.arm.data.qfrc_applied[:] = resistance
        mujoco.mj_step(self.arm.model, self.arm.data)
        truth = self.arm.truth()
        self.sensors.advance(self.arm.data.time, truth, torque)
        return truth

    def observation(self) -> Observation:
        return self.sensors.observe(self.arm.data.time)
