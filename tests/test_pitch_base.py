"""P8b reciprocal pitch-base native mechanics fixtures.

Copyright (C) 2026 Florian Loitsch. All rights reserved.
"""

import numpy as np

from robot_ai.sim.pitch_base import CoupledPitchBaseArm, PitchBaseSettings


INITIAL_BASE_PITCH = 0.02
INITIAL_Q = np.array([0.10, 0.40])


def _command(time_s: float) -> np.ndarray:
    return np.array(
        [
            0.04 * np.sin(2 * np.pi * 0.75 * time_s),
            -0.03 * np.sin(2 * np.pi * 0.45 * time_s + 0.3),
        ]
    )


def _run(stiffness: float, dt: float = 0.001) -> list[dict[str, np.ndarray | float]]:
    arm = CoupledPitchBaseArm(
        settings=PitchBaseSettings(
            support_stiffness_nm_per_rad=stiffness, support_damping_nms_per_rad=0.65
        ),
        timestep=dt,
    )
    arm.reset(INITIAL_BASE_PITCH, INITIAL_Q)
    samples: list[dict[str, np.ndarray | float]] = []
    substeps = round(0.01 / dt)
    for index in range(round(2.0 / dt)):
        control_time = (index // substeps) * 0.01
        samples.append(arm.step(_command(control_time)))
    return samples


def test_pitch_base_static_support_matches_all_body_force_balance() -> None:
    arm = CoupledPitchBaseArm()
    balance = arm.static_balance(np.array([0.45, -0.55]))
    assert abs(balance["independent_residual_nm"]) < 1e-9
    assert abs(balance["mujoco_residual_nm"]) < 1e-9
    assert abs(balance["gravity_moment_base_nm"]) > 1e-3
    assert abs(balance["gravity_moment_link1_nm"]) > 0.1
    assert abs(balance["gravity_moment_link2_nm"]) > 0.1


def test_reset_restores_time_and_returned_support_matches_returned_state() -> None:
    arm = CoupledPitchBaseArm()
    arm.reset(0.08, INITIAL_Q)
    output = arm.step(np.zeros(2))
    assert output["time_s"] == 0.001
    assert abs(output["support_moment_nm"] - output["expected_support_moment_nm"]) < 1e-12
    arm.reset(0.08, INITIAL_Q)
    assert arm.data.time == 0.0
    assert arm.support_moment_nm() == arm.expected_support_moment_nm()


def test_support_stiffness_changes_world_geometry_and_relative_arm_at_same_initial_state() -> None:
    soft, stiff = _run(18.0), _run(72.0)
    keys = ("shoulder_xz_m", "link1_com_xz_m", "link2_com_xz_m", "endpoint_xz_m")
    for key in keys:
        soft_positions = np.asarray([item[key] for item in soft])
        stiff_positions = np.asarray([item[key] for item in stiff])
        assert np.max(np.linalg.norm(soft_positions - stiff_positions, axis=1)) > 1e-4
    soft_base = np.asarray([item["base_pitch_rad"] for item in soft])
    stiff_base = np.asarray([item["base_pitch_rad"] for item in stiff])
    soft_q = np.asarray([item["arm_q_rad"] for item in soft])
    stiff_q = np.asarray([item["arm_q_rad"] for item in stiff])
    assert np.max(np.abs(soft_base - stiff_base)) > 0.01
    assert np.max(np.abs(soft_q - stiff_q)) > 1e-4


def test_unpowered_coupled_system_has_damping_work_energy_balance() -> None:
    arm = CoupledPitchBaseArm(
        settings=PitchBaseSettings(
            support_stiffness_nm_per_rad=36.0, support_damping_nms_per_rad=0.65
        )
    )
    arm.reset(0.08, INITIAL_Q)
    energy = [arm.system_energy_j()]
    damping_work = 0.0
    for _ in range(2_000):
        output = arm.step(np.zeros(2))
        energy.append(float(output["total_energy_j"]))
        damping_work += float(output["passive_damping_power_w"]) * arm.timestep
    increments = np.diff(energy)
    assert np.max(increments) <= 1e-7
    assert energy[-1] < energy[0] - 0.01
    assert abs(energy[-1] - energy[0] + damping_work) < 2e-3


def test_full_coupled_trajectory_converges_when_timestep_is_halved() -> None:
    coarse, fine = _run(36.0, 0.001), _run(36.0, 0.0005)
    aligned = fine[1::2]
    for key in ("shoulder_xz_m", "link1_com_xz_m", "link2_com_xz_m", "endpoint_xz_m"):
        coarse_position = np.asarray([item[key] for item in coarse])
        fine_position = np.asarray([item[key] for item in aligned])
        assert np.max(np.linalg.norm(coarse_position - fine_position, axis=1)) <= 0.001
    coarse_support = np.asarray([item["support_moment_nm"] for item in coarse])
    fine_support = np.asarray([item["support_moment_nm"] for item in aligned])
    assert (
        abs(np.max(np.abs(coarse_support)) - np.max(np.abs(fine_support)))
        / np.max(np.abs(fine_support))
        <= 0.05
    )
