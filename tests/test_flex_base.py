"""P8 compliance, reaction, and load-envelope fixtures."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from robot_ai.contracts import OperatingEnvelope
from robot_ai.sim.combined import CombinedImperfectArm
from robot_ai.sim.envelope import LoadMetrics
from robot_ai.sim.flex_base import BaseSettings, CompliantBase, FlexSettings, TorsionalFlexMode
from robot_ai.sim.native import default_descriptor


def test_flex_static_deflection_matches_moment_over_stiffness() -> None:
    flex = TorsionalFlexMode(FlexSettings(100.0, 1.0, 0.01))
    for _ in range(20_000):
        flex.step(1.0, 0.0005)
    assert abs(flex.angle - 0.01) < 0.001


def test_flex_ringdown_dissipates_energy() -> None:
    flex = TorsionalFlexMode(FlexSettings(100.0, 0.5, 0.01))
    flex.angle = 0.05
    before = flex.stored_energy
    for _ in range(500):
        flex.step(0.0, 0.001)
    assert flex.stored_energy < before


def test_flex_undamped_period_matches_declared_stiffness_and_inertia() -> None:
    settings = FlexSettings(100.0, 0.0, 0.01)
    flex = TorsionalFlexMode(settings)
    flex.angle = 0.05
    dt = 0.0001
    crossings: list[float] = []
    previous = flex.angle
    for index in range(20_000):
        flex.step(0.0, dt)
        if previous > 0.0 >= flex.angle:
            crossings.append((index + 1) * dt)
            if len(crossings) == 2:
                break
        previous = flex.angle
    measured_period = crossings[1] - crossings[0]
    expected_period = 2.0 * np.pi * np.sqrt(settings.effective_inertia_kg_m2 / settings.stiffness_nm_per_rad)
    assert abs(measured_period - expected_period) / expected_period < 0.02


def test_base_reaction_moves_relative_to_support_equilibrium() -> None:
    base = CompliantBase(BaseSettings(1.0, 0.1, np.array([100.0, 200.0]), np.array([1.0, 1.0]), 10.0, 0.2))
    for _ in range(1000):
        base.step(np.array([1.0, 0.0]), 0.2, 0.001)
    assert base.pose[0] > 0.0
    assert base.pose[2] > 0.0


def test_unpowered_base_ringdown_is_passive_relative_to_support_equilibrium() -> None:
    base = CompliantBase(BaseSettings(1.0, 0.1, np.array([100.0, 200.0]), np.array([1.5, 1.0]), 10.0, 0.3))
    base.pose[:] = [0.03, -0.02, 0.04]
    base.velocity[:] = [0.1, -0.2, 0.3]
    before = base.stored_energy
    energies = [before]
    for _ in range(1000):
        base.step(np.zeros(2), 0.0, 0.0005)
        energies.append(base.stored_energy)
    assert max(energies) <= before + 1e-10
    assert energies[-1] < before


def test_passive_base_ringdown_converges_when_timestep_is_halved() -> None:
    settings = BaseSettings(1.0, 0.1, np.array([100.0, 200.0]), np.array([1.5, 1.0]), 10.0, 0.3)

    def simulate(dt: float) -> np.ndarray:
        base = CompliantBase(settings)
        base.pose[:] = [0.03, -0.02, 0.04]
        base.velocity[:] = [0.1, -0.2, 0.3]
        for _ in range(round(0.5 / dt)):
            base.step(np.zeros(2), 0.0, dt)
        return base.pose.copy()

    np.testing.assert_allclose(simulate(0.001), simulate(0.0005), atol=5e-4)


def test_envelope_reports_force_flex_and_base_utilization() -> None:
    envelope = OperatingEnvelope(np.array([2.0, 2.0]), 1.0, np.array([0.1, 0.1]), 0.1, np.array([0.05, 0.05]))
    metrics = LoadMetrics(np.array([1.0, 0.5]), 0.2, np.array([0.02, 0.01]), 0.02, np.array([0.01, 0.02]))
    assert 0.4 < metrics.utilization(envelope) < 0.6


def test_combined_arm_keeps_hidden_body_state_out_of_observation() -> None:
    arm = CombinedImperfectArm(default_descriptor(), seed=4)
    for _ in range(100):
        arm.step(np.array([0.7, -0.5]))
    truth = arm.truth()
    observation = arm.observation()
    assert truth.base_pose.shape == (3,)
    assert truth.load_summary.shape == (8,)
    assert observation.values.shape == (8,)
    assert np.isfinite(truth.load_summary).all()
    assert np.isfinite(observation.values).all()
    assert arm.load_utilization() >= 0.0


def test_combined_support_preload_does_not_drive_base_from_equilibrium() -> None:
    arm = CombinedImperfectArm(default_descriptor(), seed=9)
    for _ in range(200):
        arm.step(np.zeros(2))
    np.testing.assert_allclose(arm.truth().base_pose, np.zeros(3), atol=1e-12)
