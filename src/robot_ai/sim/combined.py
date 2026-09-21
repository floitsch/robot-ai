"""Combined imperfect arm used by the P7/P8 integration fixtures."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..contracts import Command, Observation, OperatingEnvelope, PrivilegedRecord, RobotDescriptor
from .actuators import ActuatorModel, ActuatorSettings
from .backlash import BacklashSettings, BacklashTransmission
from .envelope import LoadMetrics
from .flex_base import BaseSettings, CompliantBase, FlexSettings, TorsionalFlexMode
from .mechanics import forward_kinematics
from .native import NativeArm
from .sensors import SensorFault, SensorSettings, SensorSuite


@dataclass(frozen=True)
class CombinedSettings:
    """Hidden dynamics and rated limits for the reduced combined fixture."""

    actuator: ActuatorSettings
    sensors: SensorSettings
    faults: tuple[SensorFault, ...]
    backlash: BacklashSettings
    flex: FlexSettings
    base: BaseSettings
    envelope: OperatingEnvelope

    @classmethod
    def research_default(cls) -> CombinedSettings:
        return cls(
            actuator=ActuatorSettings(
                effectiveness=np.array([0.82, 0.88]), time_constant_s=np.array([0.015, 0.02]),
                command_latency_s=np.array([0.01, 0.0]), viscous_friction=np.array([0.04, 0.06]),
                coulomb_friction=np.array([0.05, 0.07]), local_resistance=np.array([0.10, 0.08]),
                local_angle=np.array([0.45, -0.35]), local_width=np.array([0.12, 0.15]),
            ),
            sensors=SensorSettings.healthy(),
            faults=(SensorFault("dropout", (6,), 0.65, 0.95), SensorFault("freeze", (0,), 0.85, 1.15)),
            backlash=BacklashSettings(half_gap_rad=0.012, stiffness_nm_per_rad=18.0,
                                      damping_nms_per_rad=0.20, motor_inertia_kg_m2=0.002),
            flex=FlexSettings(stiffness_nm_per_rad=120.0, damping_nms_per_rad=0.6,
                              effective_inertia_kg_m2=0.012),
            base=BaseSettings(mass_kg=1.2, inertia_kg_m2=0.08,
                              stiffness_xz_n_per_m=np.array([180.0, 260.0]),
                              damping_xz_ns_per_m=np.array([2.0, 2.5]),
                              pitch_stiffness_nm_per_rad=18.0, pitch_damping_nms_per_rad=0.4),
            envelope=OperatingEnvelope(np.array([18.0, 24.0]), 4.0, np.array([0.08, 0.08]), 0.12,
                                       np.array([0.06, 0.06])),
        )


class CombinedImperfectArm:
    """One-world native arm with hidden transmission, flex, and base state."""

    def __init__(self, descriptor: RobotDescriptor, settings: CombinedSettings | None = None,
                 *, timestep: float = 0.001, seed: int = 0) -> None:
        self.descriptor = descriptor
        self.settings = settings or CombinedSettings.research_default()
        self.timestep = timestep
        self.arm = NativeArm(descriptor, timestep=timestep)
        self.actuator = ActuatorModel(descriptor, self.settings.actuator, timestep)
        self.sensors = SensorSuite(self.settings.sensors, seed=seed, faults=self.settings.faults)
        self.backlash = [BacklashTransmission(self.settings.backlash) for _ in range(2)]
        self.flex = [TorsionalFlexMode(self.settings.flex) for _ in range(2)]
        self.base = CompliantBase(self.settings.base)
        self._support_preload_force = np.array([0.0, np.sum(self.descriptor.nominal_masses) * 9.81])
        self._support_preload_moment = 0.0
        self._truth: PrivilegedRecord | None = None
        self._previous_endpoint = np.zeros(2, dtype=np.float64)
        self.reset()

    def reset(self, q: np.ndarray | None = None, dq: np.ndarray | None = None) -> None:
        self.arm.reset(q, dq)
        self.arm.data.qfrc_applied[:] = 0.0
        self.actuator.reset()
        self.sensors.reset()
        for index, transmission in enumerate(self.backlash):
            transmission.reset(float(self.arm.data.qpos[index]))
        for mode in self.flex:
            mode.reset()
        self.base.reset()
        self._previous_endpoint = self._world_endpoint()
        self._truth = self._make_truth(np.zeros(2))
        self._previous_endpoint = self._truth.endpoint_xz.copy()
        self.sensors.advance(self.arm.data.time, self._truth, np.zeros(2))

    def _world_endpoint(self) -> np.ndarray:
        effective_q = self.arm.data.qpos.copy() + np.array([mode.angle for mode in self.flex])
        local = forward_kinematics(effective_q, self.descriptor.link_lengths)
        pitch = self.base.pose[2]
        rotation = np.array([[np.cos(pitch), np.sin(pitch)], [-np.sin(pitch), np.cos(pitch)]])
        return self.base.pose[:2] + rotation @ local

    def _make_truth(self, reaction_force: np.ndarray) -> PrivilegedRecord:
        endpoint = self._world_endpoint()
        velocity = (endpoint - self._previous_endpoint) / self.timestep
        metrics = LoadMetrics(reaction_force, float(np.sum(self.arm.data.qfrc_applied)),
                              self.base.pose[:2].copy(), float(self.base.pose[2]),
                              np.array([mode.angle for mode in self.flex]))
        return PrivilegedRecord(self.arm.data.qpos.copy(), self.arm.data.qvel.copy(), endpoint, velocity,
                                self.base.pose.copy(),
                                np.concatenate((metrics.base_force_xz, [metrics.support_moment],
                                                metrics.base_deflection_xz, [metrics.base_pitch],
                                                metrics.link_flex_angles)))

    def step(self, command: Command | np.ndarray) -> PrivilegedRecord:
        accepted = command if isinstance(command, Command) else Command.accepted(command)
        q = self.arm.data.qpos.copy()
        dq = self.arm.data.qvel.copy()
        motor_torque, resistance = self.actuator.update(accepted, q, dq)
        coupling = np.array([
            transmission.step(motor_torque[index], q[index], dq[index], self.timestep)
            for index, transmission in enumerate(self.backlash)
        ])
        self.arm.data.ctrl[:] = 0.0
        self.arm.data.qfrc_applied[:] = coupling + resistance
        mujoco.mj_step(self.arm.model, self.arm.data)
        for index, mode in enumerate(self.flex):
            mode.step(coupling[index], self.timestep)
        reaction_force = self._support_preload_force.copy()
        reaction_force[0] = float(np.sum(coupling) / max(self.descriptor.link_lengths))
        # Base coordinates are relative to the declared support equilibrium. Static
        # gravity/preload therefore belongs to the support, not a perpetual drive.
        self.base.step(reaction_force - self._support_preload_force,
                       float(np.sum(coupling)) - self._support_preload_moment, self.timestep)
        truth = self._make_truth(reaction_force)
        self._previous_endpoint = truth.endpoint_xz.copy()
        self._truth = truth
        self.sensors.advance(self.arm.data.time, truth, motor_torque)
        return truth

    def truth(self) -> PrivilegedRecord:
        if self._truth is None:
            raise RuntimeError("combined arm has not been reset")
        return self._truth

    def observation(self) -> Observation:
        return self.sensors.observe(self.arm.data.time)

    def load_utilization(self) -> float:
        truth = self.truth()
        metrics = LoadMetrics(truth.load_summary[:2], float(truth.load_summary[2]), truth.load_summary[3:5],
                              float(truth.load_summary[5]), truth.load_summary[6:8])
        return metrics.utilization(self.settings.envelope)
