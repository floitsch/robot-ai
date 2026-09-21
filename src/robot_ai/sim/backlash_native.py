"""Native reciprocal one-joint backlash transmission reference.

Copyright (C) 2026 Florian Loitsch. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class NativeBacklashSettings:
    """Declared rotor, arm-output, and finite-gap contact parameters."""

    half_gap_rad: float = 0.015
    engagement_stiffness_nm_per_rad: float = 20.0
    contact_damping_nms_per_rad: float = 0.18
    rotor_inertia_kg_m2: float = 0.002
    rotor_mass_kg: float = 0.05
    arm_mass_kg: float = 0.60
    arm_inertia_kg_m2: float = 0.018
    arm_length_m: float = 0.30

    def __post_init__(self) -> None:
        values = np.asarray([
            self.half_gap_rad, self.engagement_stiffness_nm_per_rad,
            self.contact_damping_nms_per_rad, self.rotor_inertia_kg_m2,
            self.rotor_mass_kg, self.arm_mass_kg, self.arm_inertia_kg_m2, self.arm_length_m,
        ])
        if not np.isfinite(values).all():
            raise ValueError("native backlash settings must be finite")
        if self.half_gap_rad <= 0 or self.engagement_stiffness_nm_per_rad <= 0:
            raise ValueError("gap and engagement stiffness must be positive")
        if self.contact_damping_nms_per_rad < 0:
            raise ValueError("contact damping must be nonnegative")
        if min(self.rotor_inertia_kg_m2, self.rotor_mass_kg, self.arm_mass_kg,
               self.arm_inertia_kg_m2, self.arm_length_m) <= 0:
            raise ValueError("declared masses, inertia, and length must be positive")


_SETTINGS_BY_MODEL_ADDRESS: dict[int, NativeBacklashSettings] = {}


def _engagement(relative_angle_rad: float, half_gap_rad: float) -> float:
    """Signed tooth compression after the finite free-play gap."""

    return float(np.sign(relative_angle_rad) * max(0.0, abs(relative_angle_rad) - half_gap_rad))


def _passive_backlash(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply reciprocal passive flank force at every MuJoCo integrator substep."""

    # The callback can be entered while another model is being constructed or
    # destroyed. Such a model is not this fixture, and must receive no force.
    settings = _SETTINGS_BY_MODEL_ADDRESS.get(getattr(model, "_address", -1))
    if settings is None:
        return
    relative = float(data.qpos[0] - data.qpos[1])
    compression = _engagement(relative, settings.half_gap_rad)
    if compression == 0.0:
        return
    relative_velocity = float(data.qvel[0] - data.qvel[1])
    transmitted = (
        settings.engagement_stiffness_nm_per_rad * compression
        + settings.contact_damping_nms_per_rad * relative_velocity
    )
    # A positive motor-ahead compression pulls output positive and reacts on rotor negative.
    data.qfrc_passive[0] -= transmitted
    data.qfrc_passive[1] += transmitted


def make_native_backlash_mjcf(
    settings: NativeBacklashSettings | None = None, *, timestep: float = 0.001
) -> str:
    """Return a rotor and actual gravity-loaded output arm in one MuJoCo world."""

    settings = settings or NativeBacklashSettings()
    length = settings.arm_length_m
    return f"""<mujoco model="robot-ai-native-backlash">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{timestep:.9g}" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="rotor" pos="0 0 0">
      <joint name="motor_rotor" type="hinge" axis="0 -1 0" damping="0"/>
      <inertial pos="0 0 0" mass="{settings.rotor_mass_kg}" diaginertia="{settings.rotor_inertia_kg_m2} {settings.rotor_inertia_kg_m2} {settings.rotor_inertia_kg_m2}"/>
      <geom name="rotor_marker" type="cylinder" size="0.03 0.02" mass="0" rgba="0.25 0.25 0.25 1"/>
    </body>
    <body name="output_arm" pos="0 0 0">
      <joint name="output_joint" type="hinge" axis="0 -1 0" range="-1.4 1.4" limited="true" damping="0"/>
      <inertial pos="0 0 {-length / 2}" mass="{settings.arm_mass_kg}" diaginertia="{settings.arm_inertia_kg_m2} {settings.arm_inertia_kg_m2} 1e-6"/>
      <geom name="output_link" type="capsule" fromto="0 0 0 0 0 {-length}" size="0.018" mass="0" rgba="0.15 0.45 0.85 1"/>
      <site name="endpoint" pos="0 0 {-length}" size="0.005"/>
    </body>
  </worldbody>
  <actuator><motor name="motor_torque" joint="motor_rotor" gear="1" ctrllimited="true" ctrlrange="-0.25 0.25"/></actuator>
</mujoco>"""


class NativeBacklashArm:
    """CPU-native reciprocal rotor/output backlash mechanics.

    The motor rotor is hidden internal mechanics. ``output_observation`` exposes
    only the physical arm-output q/dq, matching the public output-joint contract.
    The custom passive callback is evaluated by MuJoCo for each RK4 substep; it is
    a contact spring/damper, not a command deadzone or torque scale.
    """

    def __init__(self, settings: NativeBacklashSettings | None = None, *, timestep: float = 0.001) -> None:
        self.settings = settings or NativeBacklashSettings()
        self.timestep = timestep
        # MuJoCo callbacks are process-global. Construct with no callback and only
        # install ours around this fixture's own forward/step calls below.
        mujoco.set_mjcb_passive(None)
        self.model = mujoco.MjModel.from_xml_string(make_native_backlash_mjcf(self.settings, timestep=timestep))
        _SETTINGS_BY_MODEL_ADDRESS[self.model._address] = self.settings
        self.data = mujoco.MjData(self.model)
        self.output_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "output_arm")
        self.endpoint_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
        self.reset()

    @staticmethod
    def _activate_passive_callback() -> None:
        mujoco.set_mjcb_passive(_passive_backlash)

    @staticmethod
    def _deactivate_passive_callback() -> None:
        mujoco.set_mjcb_passive(None)

    def reset(
        self, output_q_rad: float = 0.0, *, rotor_q_rad: float | None = None,
        output_dq_rad_s: float = 0.0, rotor_dq_rad_s: float = 0.0,
    ) -> None:
        rotor_q = output_q_rad if rotor_q_rad is None else rotor_q_rad
        self.data.time = 0.0
        self.data.qpos[:] = (rotor_q, output_q_rad)
        self.data.qvel[:] = (rotor_dq_rad_s, output_dq_rad_s)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self._activate_passive_callback()
        try:
            mujoco.mj_forward(self.model, self.data)
        finally:
            self._deactivate_passive_callback()

    @property
    def rotor_q_rad(self) -> float:
        return float(self.data.qpos[0])

    @property
    def rotor_dq_rad_s(self) -> float:
        return float(self.data.qvel[0])

    @property
    def output_q_rad(self) -> float:
        return float(self.data.qpos[1])

    @property
    def output_dq_rad_s(self) -> float:
        return float(self.data.qvel[1])

    def output_observation(self) -> dict[str, float]:
        """Public physical output state; deliberately excludes hidden rotor state."""

        return {"output_q_rad": self.output_q_rad, "output_dq_rad_s": self.output_dq_rad_s}

    def relative_angle_rad(self) -> float:
        return self.rotor_q_rad - self.output_q_rad

    def engagement_compression_rad(self) -> float:
        return _engagement(self.relative_angle_rad(), self.settings.half_gap_rad)

    def expected_transmitted_torque_nm(self) -> float:
        compression = self.engagement_compression_rad()
        if compression == 0.0:
            return 0.0
        return float(
            self.settings.engagement_stiffness_nm_per_rad * compression
            + self.settings.contact_damping_nms_per_rad * (self.rotor_dq_rad_s - self.output_dq_rad_s)
        )

    def transmitted_torques_nm(self) -> tuple[float, float]:
        """Returned-state passive torque on rotor and output, respectively."""

        return float(self.data.qfrc_passive[0]), float(self.data.qfrc_passive[1])

    def energy_components_j(self) -> dict[str, float]:
        matrix = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, matrix, self.data.qM)
        kinetic = 0.5 * float(self.data.qvel @ matrix @ self.data.qvel)
        gravity = float(sum(
            self.model.body_mass[body_id] * 9.81 * self.data.xipos[body_id, 2]
            for body_id in range(1, self.model.nbody)
        ))
        spring = 0.5 * self.settings.engagement_stiffness_nm_per_rad * self.engagement_compression_rad() ** 2
        return {
            "rotor_arm_kinetic_energy_j": kinetic,
            "gravity_potential_j": gravity,
            "flank_spring_energy_j": spring,
            "total_energy_j": kinetic + gravity + spring,
        }

    def step(self, motor_torque_nm: float) -> dict[str, float | np.ndarray]:
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = float(motor_torque_nm)
        self._activate_passive_callback()
        try:
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        finally:
            self._deactivate_passive_callback()
        rotor_torque, output_torque = self.transmitted_torques_nm()
        energy = self.energy_components_j()
        compression = self.engagement_compression_rad()
        return {
            "time_s": float(self.data.time),
            "motor_torque_nm": float(motor_torque_nm),
            "rotor_q_rad": self.rotor_q_rad,
            "rotor_dq_rad_s": self.rotor_dq_rad_s,
            "output_q_rad": self.output_q_rad,
            "output_dq_rad_s": self.output_dq_rad_s,
            "relative_angle_rad": self.relative_angle_rad(),
            "engagement_compression_rad": compression,
            "flank": float(np.sign(compression)),
            "rotor_transmission_torque_nm": rotor_torque,
            "output_transmission_torque_nm": output_torque,
            "expected_output_transmission_torque_nm": self.expected_transmitted_torque_nm(),
            "reciprocal_torque_residual_nm": rotor_torque + output_torque,
            "contact_damping_power_w": self.settings.contact_damping_nms_per_rad
            * (self.rotor_dq_rad_s - self.output_dq_rad_s) ** 2 if compression != 0.0 else 0.0,
            "endpoint_xz_m": self.data.site_xpos[self.endpoint_site, (0, 2)].copy(),
            **energy,
        }
