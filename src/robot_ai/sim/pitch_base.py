"""Reciprocal native MuJoCo reference for a rigid arm on a pitch-compliant support.

Copyright (C) 2026 Florian Loitsch. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..contracts import Command, RobotDescriptor
from .native import default_descriptor


@dataclass(frozen=True)
class PitchBaseSettings:
    """Pitch support and declared rigid-body geometry for the P8b reference.

    The support pivot is the world origin. The base COM is below it and the
    shoulder is forward of it, so support pitch changes the arm's world geometry.
    Both offsets are local to the pitched base and expressed as ``(x, z)`` metres.
    """

    base_mass_kg: float = 1.2
    base_inertia_kg_m2: float = 0.08
    base_com_local_xz_m: tuple[float, float] = (0.0, -0.08)
    shoulder_offset_local_xz_m: tuple[float, float] = (0.12, 0.0)
    support_stiffness_nm_per_rad: float = 36.0
    support_damping_nms_per_rad: float = 0.45
    support_equilibrium_pitch_rad: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.base_mass_kg,
                self.base_inertia_kg_m2,
                *self.base_com_local_xz_m,
                *self.shoulder_offset_local_xz_m,
                self.support_stiffness_nm_per_rad,
                self.support_damping_nms_per_rad,
                self.support_equilibrium_pitch_rad,
            ]
        )
        if not np.isfinite(values).all() or self.base_mass_kg <= 0 or self.base_inertia_kg_m2 <= 0:
            raise ValueError("base mass and inertia must be finite and positive")
        if self.support_stiffness_nm_per_rad <= 0 or self.support_damping_nms_per_rad < 0:
            raise ValueError("support stiffness must be positive and damping nonnegative")
        if np.linalg.norm(self.shoulder_offset_local_xz_m) <= 0:
            raise ValueError("shoulder offset must be nonzero from the support pivot")


def make_pitch_base_mjcf(
    descriptor: RobotDescriptor | None = None,
    settings: PitchBaseSettings | None = None,
    *,
    timestep: float = 0.001,
) -> str:
    """Return one coupled base/arm topology, not a post-hoc endpoint transform."""

    descriptor = descriptor or default_descriptor()
    settings = settings or PitchBaseSettings()
    l1, l2 = descriptor.link_lengths
    m1, m2 = descriptor.nominal_masses
    i1, i2 = descriptor.nominal_inertias
    t1, t2 = descriptor.nominal_torque_scales
    (q1_min, q1_max), (q2_min, q2_max) = descriptor.joint_limits
    base_x, base_z = settings.base_com_local_xz_m
    shoulder_x, shoulder_z = settings.shoulder_offset_local_xz_m
    return f"""<mujoco model="robot-ai-coupled-pitch-base">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{timestep:.9g}" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="base_pitch" type="hinge" axis="0 -1 0"
             stiffness="{settings.support_stiffness_nm_per_rad}" springref="{settings.support_equilibrium_pitch_rad}"
             damping="{settings.support_damping_nms_per_rad}"/>
      <inertial pos="{base_x} 0 {base_z}" mass="{settings.base_mass_kg}" diaginertia="{settings.base_inertia_kg_m2} {settings.base_inertia_kg_m2} {settings.base_inertia_kg_m2}"/>
      <geom name="base_marker" type="sphere" size="0.025" mass="0" rgba="0.2 0.2 0.2 1"/>
      <body name="link1" pos="{shoulder_x} 0 {shoulder_z}">
        <joint name="joint1" type="hinge" axis="0 -1 0" range="{q1_min} {q1_max}" limited="true" damping="0"/>
        <inertial pos="0 0 {-l1 / 2}" mass="{m1}" diaginertia="{i1} {i1} 1e-6"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 {-l1}" size="0.018" density="0" mass="0"/>
        <body name="link2" pos="0 0 {-l1}">
          <joint name="joint2" type="hinge" axis="0 -1 0" range="{q2_min} {q2_max}" limited="true" damping="0"/>
          <inertial pos="0 0 {-l2 / 2}" mass="{m2}" diaginertia="{i2} {i2} 1e-6"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 {-l2}" size="0.016" density="0" mass="0"/>
          <site name="endpoint" pos="0 0 {-l2}" size="0.005"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="{t1}" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor2" joint="joint2" gear="{t2}" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""


class CoupledPitchBaseArm:
    """One reciprocal system with native state-dependent passive support.

    ``step`` returns the state after one ``mj_step`` then ``mj_forward``. All
    returned transforms, mass matrices, energies, and support moment describe that
    same returned state. MuJoCo evaluates the support spring/damper at every RK4
    substep; ``qfrc_applied`` remains zero.
    """

    def __init__(
        self,
        descriptor: RobotDescriptor | None = None,
        settings: PitchBaseSettings | None = None,
        *,
        timestep: float = 0.001,
    ) -> None:
        self.descriptor = descriptor or default_descriptor()
        self.settings = settings or PitchBaseSettings()
        self.timestep = timestep
        self.model = mujoco.MjModel.from_xml_string(
            make_pitch_base_mjcf(self.descriptor, self.settings, timestep=timestep)
        )
        self.data = mujoco.MjData(self.model)
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.link1_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link1")
        self.link2_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link2")
        self.endpoint_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "endpoint")
        self._mass_body_ids = (self.base_body, self.link1_body, self.link2_body)
        self.reset()

    def reset(
        self, base_pitch: float = 0.0, q: np.ndarray | None = None, dq: np.ndarray | None = None
    ) -> None:
        arm_q = np.zeros(2) if q is None else np.asarray(q, dtype=np.float64)
        arm_dq = np.zeros(2) if dq is None else np.asarray(dq, dtype=np.float64)
        if arm_q.shape != (2,) or arm_dq.shape != (2,):
            raise ValueError("arm q and dq must each have shape (2,)")
        self.data.time = 0.0
        self.data.qpos[:] = np.concatenate(([base_pitch], arm_q))
        self.data.qvel[:] = np.concatenate(([0.0], arm_dq))
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def base_pitch(self) -> float:
        return float(self.data.qpos[0])

    @property
    def base_pitch_rate(self) -> float:
        return float(self.data.qvel[0])

    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[1:].copy()

    @property
    def arm_dq(self) -> np.ndarray:
        return self.data.qvel[1:].copy()

    def expected_support_moment_nm(self) -> float:
        return float(
            -self.settings.support_stiffness_nm_per_rad
            * (self.base_pitch - self.settings.support_equilibrium_pitch_rad)
            - self.settings.support_damping_nms_per_rad * self.base_pitch_rate
        )

    def support_moment_nm(self) -> float:
        """Native passive support torque at the current returned state."""

        return float(self.data.qfrc_passive[0])

    def generalized_mass_matrix(self) -> np.ndarray:
        matrix = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, matrix, self.data.qM)
        return matrix

    def gravity_moment_components_nm(self) -> dict[str, float]:
        """Positive-y holding moment from every declared massive body."""

        components: dict[str, float] = {}
        for body_id in self._mass_body_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            components[str(name)] = float(
                self.model.body_mass[body_id] * 9.81 * self.data.xipos[body_id, 0]
            )
        return components

    def independent_holding_support_moment_nm(self) -> float:
        """Holding moment from all COMs' gravity force, independent of qfrc_bias."""

        return float(sum(self.gravity_moment_components_nm().values()))

    def static_balance(self, q: np.ndarray, *, bracket_rad: float = 0.7) -> dict[str, float]:
        """Solve base pitch where native spring support and all-body gravity balance."""

        arm_q = np.asarray(q, dtype=np.float64)
        if arm_q.shape != (2,):
            raise ValueError("static balance q must have shape (2,)")

        def residual(theta: float) -> float:
            self.reset(theta, arm_q)
            return self.expected_support_moment_nm() - self.independent_holding_support_moment_nm()

        lower, upper = -bracket_rad, bracket_rad
        left, right = residual(lower), residual(upper)
        if left * right > 0:
            raise RuntimeError("static support bracket does not contain an equilibrium")
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            value = residual(middle)
            if abs(value) < 1e-12:
                break
            if left * value <= 0:
                upper, right = middle, value
            else:
                lower, left = middle, value
        theta = 0.5 * (lower + upper)
        self.reset(theta, arm_q)
        support = self.support_moment_nm()
        analytic = self.independent_holding_support_moment_nm()
        required_mujoco = float(self.data.qfrc_bias[0])
        components = self.gravity_moment_components_nm()
        return {
            "base_pitch_rad": theta,
            "support_moment_nm": support,
            "expected_support_moment_nm": self.expected_support_moment_nm(),
            "independent_force_balance_moment_nm": analytic,
            "mujoco_required_holding_moment_nm": required_mujoco,
            "independent_residual_nm": support - analytic,
            "mujoco_residual_nm": support - required_mujoco,
            **{f"gravity_moment_{name}_nm": value for name, value in components.items()},
        }

    def system_energy_components_j(self) -> dict[str, float]:
        mass = self.generalized_mass_matrix()
        kinetic = 0.5 * float(self.data.qvel @ mass @ self.data.qvel)
        gravity = float(
            sum(
                self.model.body_mass[body_id] * 9.81 * self.data.xipos[body_id, 2]
                for body_id in self._mass_body_ids
            )
        )
        support = (
            0.5
            * self.settings.support_stiffness_nm_per_rad
            * (self.base_pitch - self.settings.support_equilibrium_pitch_rad) ** 2
        )
        return {
            "kinetic_energy_j": kinetic,
            "gravity_potential_j": gravity,
            "support_potential_j": support,
            "total_energy_j": kinetic + gravity + support,
        }

    def system_energy_j(self) -> float:
        return self.system_energy_components_j()["total_energy_j"]

    def body_transforms(self) -> dict[str, np.ndarray]:
        return {
            "base_pivot_xz_m": self.data.xpos[self.base_body, (0, 2)].copy(),
            "base_com_xz_m": self.data.xipos[self.base_body, (0, 2)].copy(),
            "base_rotation": self.data.xmat[self.base_body].reshape(3, 3).copy(),
            "shoulder_xz_m": self.data.xpos[self.link1_body, (0, 2)].copy(),
            "link1_com_xz_m": self.data.xipos[self.link1_body, (0, 2)].copy(),
            "link1_rotation": self.data.xmat[self.link1_body].reshape(3, 3).copy(),
            "elbow_xz_m": self.data.xpos[self.link2_body, (0, 2)].copy(),
            "link2_com_xz_m": self.data.xipos[self.link2_body, (0, 2)].copy(),
            "link2_rotation": self.data.xmat[self.link2_body].reshape(3, 3).copy(),
            "endpoint_xz_m": self.data.site_xpos[self.endpoint_site, (0, 2)].copy(),
        }

    def step(self, command: Command | np.ndarray) -> dict[str, np.ndarray | float]:
        accepted = command if isinstance(command, Command) else Command.accepted(command)
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = accepted.values
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        transforms = self.body_transforms()
        energy = self.system_energy_components_j()
        return {
            "time_s": float(self.data.time),
            "base_pitch_rad": self.base_pitch,
            "base_pitch_rate_rad_s": self.base_pitch_rate,
            "arm_q_rad": self.arm_q,
            "arm_dq_rad_s": self.arm_dq,
            "endpoint_xz_m": transforms["endpoint_xz_m"],
            "support_moment_nm": self.support_moment_nm(),
            "expected_support_moment_nm": self.expected_support_moment_nm(),
            "independent_holding_support_moment_nm": self.independent_holding_support_moment_nm(),
            "passive_damping_power_w": self.settings.support_damping_nms_per_rad
            * self.base_pitch_rate**2,
            **energy,
            **transforms,
        }
