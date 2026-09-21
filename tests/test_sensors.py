"""P3 sensor packet and actuator defect fixtures."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from robot_ai.contracts import PrivilegedRecord
from robot_ai.sim.actuators import ActuatorModel, ActuatorSettings
from robot_ai.sim.native import default_descriptor
from robot_ai.sim.scenarios import make_scenario
from robot_ai.sim.sensors import SensorFault, SensorSettings, SensorSuite


def _truth(q: list[float], dq: list[float] | None = None) -> PrivilegedRecord:
    return PrivilegedRecord(np.array(q), np.array(dq or [0.0, 0.0]), np.zeros(2), np.zeros(2), np.zeros(3), np.zeros(4))


def _zero_noise_settings(*, delay: float = 0.0) -> SensorSettings:
    return SensorSettings(np.ones(8) * 0.01, np.zeros(8), np.ones(8) * 0.02, np.ones(8) * delay)


def test_delivered_packets_never_have_negative_age() -> None:
    suite = SensorSuite(_zero_noise_settings(delay=0.02), seed=2)
    suite.advance(0.0, _truth([0.1, 0.2]), np.zeros(2))
    assert all(packet.delivery_s >= packet.acquisition_s for packet in suite._queue)
    suite.advance(0.05, _truth([0.1, 0.2]), np.zeros(2))
    observation = suite.observe(0.05)
    assert (observation.age_s >= 0).all()


def test_dropout_and_frozen_new_packets_are_distinct() -> None:
    faults = (SensorFault("dropout", (0,), 0.0), SensorFault("freeze", (1,), 0.0))
    suite = SensorSuite(_zero_noise_settings(), seed=3, faults=faults)
    suite.advance(0.0, _truth([0.1, 0.2]), np.zeros(2))
    first = suite.observe(0.0)
    assert not first.available[0]
    assert first.available[1]
    old_value = first.values[1]
    suite.advance(0.01, _truth([0.9, 0.8]), np.zeros(2))
    second = suite.observe(0.01)
    assert not second.available[0]
    assert second.fresh[1]
    assert second.values[1] == old_value
    assert second.age_s[1] == 0.0


def test_bias_fault_changes_a_fresh_packet_without_changing_its_transport_metadata() -> None:
    settings = SensorSettings(np.ones(8) * 0.01, np.zeros(8), np.zeros(8), np.zeros(8))
    suite = SensorSuite(settings, seed=5, faults=(SensorFault("bias", (0,), 0.0, offset=0.25),))
    suite.advance(0.0, _truth([0.1, 0.2]), np.zeros(2))
    observation = suite.observe(0.0)
    assert observation.available[0]
    assert observation.fresh[0]
    assert observation.age_s[0] == 0.0
    assert observation.values[0] == 0.35


def test_velocity_is_derived_from_angle_samples() -> None:
    suite = SensorSuite(_zero_noise_settings(), seed=4)
    suite.advance(0.0, _truth([0.0, 0.0]), np.zeros(2))
    suite.advance(0.01, _truth([0.1, 0.0]), np.zeros(2))
    observation = suite.observe(0.01)
    assert observation.values[2] > 0.0
    assert observation.values[2] != 0.0


def test_actuator_resistance_opposes_motion_and_effectiveness_is_hidden() -> None:
    descriptor = default_descriptor()
    settings = ActuatorSettings(np.ones(2) * 0.5, np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2),
                                np.zeros(2), np.zeros(2), np.ones(2) * 0.1)
    model = ActuatorModel(descriptor, settings, 0.001)
    torque, resistance = model.update([1.0, 1.0], np.zeros(2), np.ones(2))
    np.testing.assert_allclose(torque, descriptor.nominal_torque_scales * 0.5)
    assert (resistance < 0).all()


def test_scenario_parameters_are_independent_of_batch_order() -> None:
    first = make_scenario(10, "scenario-17", degraded=True)
    second = make_scenario(10, "other", degraded=True)
    repeated = make_scenario(10, "scenario-17", degraded=True)
    np.testing.assert_allclose(first.actuator.effectiveness, repeated.actuator.effectiveness)
    assert first.faults == repeated.faults
    assert not np.allclose(first.actuator.effectiveness, second.actuator.effectiveness)
