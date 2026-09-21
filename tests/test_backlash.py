"""P7 backlash lost-motion and passive-energy fixtures."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from robot_ai.evaluate.physics import run_backlash_fixtures
from robot_ai.sim.backlash import BacklashSettings, BacklashTransmission


def test_reversal_has_lost_motion_inside_the_gap() -> None:
    transmission = BacklashTransmission(BacklashSettings(half_gap_rad=0.01, stiffness_nm_per_rad=20.0))
    transmission.reset(0.0)
    for _ in range(20):
        transmission.step(0.01, 0.0, 0.0, 0.001)
    motor_before_reversal = transmission.motor_q
    assert transmission.coupling_torque(0.0, 0.0) == 0.0
    for _ in range(20):
        transmission.step(-0.01, 0.0, 0.0, 0.001)
    assert transmission.motor_q != motor_before_reversal
    assert transmission.coupling_torque(0.0, 0.0) == 0.0


def test_engaged_teeth_transmit_load_and_unpowered_damping_dissipates() -> None:
    transmission = BacklashTransmission(BacklashSettings(half_gap_rad=0.001, stiffness_nm_per_rad=30.0,
                                                          damping_nms_per_rad=0.3))
    transmission.reset(0.0)
    transmission.motor_q = 0.02
    torque = transmission.coupling_torque(0.0, 0.0)
    assert torque > 0.0
    before = transmission.stored_energy(0.0)
    for _ in range(200):
        transmission.step(0.0, 0.0, 0.0, 0.0005)
    assert transmission.stored_energy(0.0) < before


def test_backlash_fixture_records_reversal_passivity_and_timestep_evidence(tmp_path) -> None:
    report = run_backlash_fixtures(tmp_path / "backlash.json")
    assert report["passed"]
    assert report["coarse"]["reversal_lost_motion"]
    assert report["fine"]["unpowered_energy_dissipates"]
