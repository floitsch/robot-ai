"""P7 native moving-output backlash mechanics fixtures.

Copyright (C) 2026 Florian Loitsch. All rights reserved.
"""

import numpy as np

from robot_ai.sim.backlash_native import NativeBacklashArm


def _command(time_s: float) -> float:
    if time_s < 0.50:
        return 0.15
    if time_s < 1.20:
        return -0.15
    return 0.15


def _run(dt: float) -> list[dict[str, float | np.ndarray]]:
    arm = NativeBacklashArm(timestep=dt)
    arm.reset(0.0, rotor_q_rad=0.0)
    samples: list[dict[str, float | np.ndarray]] = []
    control_substeps = round(0.01 / dt)
    for index in range(round(2.0 / dt)):
        samples.append(arm.step(_command((index // control_substeps) * 0.01)))
    return samples


def test_public_output_contract_excludes_hidden_rotor() -> None:
    arm = NativeBacklashArm()
    arm.reset(0.2, rotor_q_rad=0.23, output_dq_rad_s=0.1)
    assert arm.output_observation() == {"output_q_rad": 0.2, "output_dq_rad_s": 0.1}


def test_moving_output_reversal_crosses_both_flanks_with_reciprocal_load() -> None:
    samples = _run(0.001)
    flanks = np.asarray([item["flank"] for item in samples])
    output_q = np.asarray([item["output_q_rad"] for item in samples])
    residual = np.asarray([item["reciprocal_torque_residual_nm"] for item in samples])
    assert {-1.0, 0.0, 1.0}.issubset(set(flanks))
    assert np.ptp(output_q) > 0.1
    assert np.max(np.abs(residual)) < 1e-12


def test_output_motion_backdrives_hidden_motor_without_motor_command() -> None:
    arm = NativeBacklashArm()
    arm.reset(0.0, rotor_q_rad=arm.settings.half_gap_rad + 0.006, output_dq_rad_s=0.30)
    rotor_speeds = []
    reactions = []
    for _ in range(200):
        output = arm.step(0.0)
        rotor_speeds.append(output["rotor_dq_rad_s"])
        reactions.append(output["rotor_transmission_torque_nm"])
    assert max(abs(value) for value in reactions) > 0.01
    assert max(abs(value) for value in rotor_speeds) > 0.1


def test_unpowered_total_energy_and_damping_work_are_bounded() -> None:
    arm = NativeBacklashArm()
    arm.reset(0.0, rotor_q_rad=0.030)
    energies = [arm.energy_components_j()["total_energy_j"]]
    damping_work = 0.0
    for _ in range(2_000):
        output = arm.step(0.0)
        energies.append(float(output["total_energy_j"]))
        damping_work += float(output["contact_damping_power_w"]) * arm.timestep
    assert np.max(np.diff(energies)) <= 4e-4
    assert np.max(np.asarray(energies) - energies[0]) <= 2e-6
    assert abs(energies[-1] - energies[0] + damping_work) <= 1e-3


def test_whole_driven_trace_converges_when_timestep_is_halved() -> None:
    coarse, fine = _run(0.001), _run(0.0005)
    aligned = fine[1::2]
    for key, tolerance in (("rotor_q_rad", 0.003), ("output_q_rad", 2e-4), ("relative_angle_rad", 0.003)):
        coarse_values = np.asarray([item[key] for item in coarse])
        fine_values = np.asarray([item[key] for item in aligned])
        assert np.max(np.abs(coarse_values - fine_values)) <= tolerance
    coarse_endpoint = np.asarray([item["endpoint_xz_m"] for item in coarse])
    fine_endpoint = np.asarray([item["endpoint_xz_m"] for item in aligned])
    assert np.max(np.linalg.norm(coarse_endpoint - fine_endpoint, axis=1)) <= 1e-4
    coarse_peak = max(abs(float(item["output_transmission_torque_nm"])) for item in coarse)
    fine_peak = max(abs(float(item["output_transmission_torque_nm"])) for item in aligned)
    assert abs(coarse_peak - fine_peak) / fine_peak <= 0.05
