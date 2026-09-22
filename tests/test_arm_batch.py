# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from robot_ai.sim.actuators import ActuatorSettings
from robot_ai.sim.arm_batch import ArmBatch
from robot_ai.sim.native import ImperfectNativeArm, NativeArm, default_descriptor
from robot_ai.sim.population import (
    healthy_population,
    sample_change,
    sample_population,
    sample_push,
)
from robot_ai.sim.sensors import SensorSettings, SensorSuite

START = np.array([0.3, 0.9])


def _commands(ticks: int) -> np.ndarray:
    time = np.arange(ticks) * 0.01
    # Small enough to stay off the hard stops, where MuJoCo's soft limits legitimately differ.
    values = np.stack((0.12 * np.sin(3.0 * time) + 0.05, 0.10 * np.cos(2.0 * time) - 0.03), axis=1)
    values[:4] = 0.0  # longer than any latency, so both models start from a zero command
    return values.astype(np.float32)


def _run(batch: ArmBatch, commands: np.ndarray, start: np.ndarray = START) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch.reset(start)
    truth, observation, metrics = [], [], []
    for command in commands:
        batch.step(np.tile(command, (batch.worlds, 1)))
        truth.append(batch.truth.numpy().copy())
        observation.append(batch.observation.numpy().copy())
        metrics.append(batch.metrics.numpy().copy())
    return np.stack(truth), np.stack(observation), np.stack(metrics)


def test_healthy_arm_matches_native_mujoco() -> None:
    commands = _commands(170)
    truth, _, _ = _run(ArmBatch(*healthy_population(2), device="cpu"), commands)
    native = NativeArm(default_descriptor())
    native.reset(START)
    for tick, command in enumerate(commands):
        for _ in range(10):
            native.step(command)
        assert np.abs(truth[tick, 0, :2] - native.data.qpos).max() < 2e-3
    assert np.abs(truth[:, 0] - truth[:, 1]).max() == 0.0


def test_weak_slow_late_motor_matches_cpu_reference() -> None:
    commands = _commands(170)
    arms, joints = healthy_population(1)
    joints["torque_scale"] = np.array([6.0, 3.0]) * np.array([0.7, 0.85])
    joints["motor_alpha"] = 1.0 - np.exp(-0.001 / np.array([0.02, 0.035]))
    joints["delay_steps"] = np.array([12, 25], dtype=np.int32)
    joints["damping"] = 0.02 + np.array([0.06, 0.03])
    truth, _, _ = _run(ArmBatch(arms, joints, device="cpu"), commands)
    settings = ActuatorSettings(np.array([0.7, 0.85]), np.array([0.02, 0.035]), np.array([0.012, 0.025]),
                                np.array([0.06, 0.03]), np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2) * 0.1)
    clean = SensorSettings(np.full(8, 0.01), np.zeros(8), np.zeros(8), np.zeros(8))
    reference = ImperfectNativeArm(default_descriptor(), settings, SensorSuite(clean))
    reference.reset(START)
    for tick, command in enumerate(commands):
        for _ in range(10):
            reference.step(command)
        assert np.abs(truth[tick, 0, :2] - reference.arm.data.qpos).max() < 5e-3


def test_dry_friction_holds_a_joint_still_until_breakaway() -> None:
    arms, joints = healthy_population(2, gravity=0.0)
    joints["coulomb"] = 0.5
    batch = ArmBatch(arms, joints, device="cpu")
    commands = np.zeros((60, 2), dtype=np.float32)
    commands[:, 0] = 0.05  # 0.30 N m, below the bound
    stuck, _, stuck_metrics = _run(batch, commands)
    assert np.abs(stuck[:, 0, :2] - START.astype(np.float32)).max() == 0.0 and np.abs(stuck[:, 0, 2:]).max() < 1e-9
    assert stuck_metrics[:, 0, 1].max() < 1e-12
    commands[:, 0] = 0.5  # 3 N m breaks away
    moving, _, moving_metrics = _run(batch, commands)
    assert moving[-1, 0, 0] - START[0] > 0.2
    assert moving_metrics[:, 0, 1].sum() > 0.0  # and dissipates heat while sliding


def test_backlash_loses_motion_on_reversal() -> None:
    arms, joints = healthy_population(2, gravity=0.0)
    joints["damping"] = 0.3
    joints["half_gap"][1] = 0.02
    joints["mesh_stiffness"], joints["mesh_damping"] = 300.0, 0.4
    batch = ArmBatch(arms, joints, device="cpu")
    commands = np.zeros((80, 2), dtype=np.float32)
    commands[:40, 1], commands[40:, 1] = 0.2, -0.2
    truth, _, _ = _run(batch, commands, np.zeros(2))
    states = batch.states.numpy()
    assert states["qm"][1, 1] - states["q"][1, 1] < -0.02  # rotor crossed the whole gap and drives the other flank
    rigid_reversal = int(np.argmax(truth[40:, 0, 3] < 0.0))
    slack_reversal = int(np.argmax(truth[40:, 1, 3] < 0.0))
    assert slack_reversal > rigid_reversal  # output keeps coasting while the rotor crosses the gap


def test_sensors_are_biased_quantized_delayed_and_reproducible() -> None:
    arms, joints = healthy_population(1)
    quantum = 2.0 * np.pi / 1024
    joints["enc_bias"], joints["enc_quantum"] = 0.015, quantum
    joints["enc_delay_ticks"] = np.array([0, 2], dtype=np.int32)
    joints["cur_gain"], joints["cur_bias"] = 1.2, 0.04
    commands = _commands(60)
    truth, observation, _ = _run(ArmBatch(arms, joints, device="cpu"), commands)
    counts = observation[:, 0, :2] / quantum
    assert np.abs(counts - np.rint(counts)).max() < 1e-3
    assert np.abs(observation[5:, 0, 0] - truth[5:, 0, 0] - 0.015).max() <= quantum / 2 + 1e-6
    assert np.abs(observation[5:, 0, 1] - truth[3:-2, 0, 1] - 0.015).max() <= quantum / 2 + 1e-6
    assert np.allclose(observation[10:, 0, 4], 1.2 * commands[10:, 0] * 6.0 + 0.04, atol=1e-5)

    noisy_arms, noisy_joints = sample_population(np.random.default_rng(3), 8)
    first = _run(ArmBatch(noisy_arms, noisy_joints, device="cpu", seed=5), commands)[1]
    second = _run(ArmBatch(noisy_arms, noisy_joints, device="cpu", seed=5), commands)[1]
    other = _run(ArmBatch(noisy_arms, noisy_joints, device="cpu", seed=6), commands)[1]
    assert np.array_equal(first, second) and not np.array_equal(first, other)


def test_sampled_population_is_diverse_and_stays_finite() -> None:
    arms, joints = sample_population(np.random.default_rng(11), 512)
    truth, observation, metrics = _run(ArmBatch(arms, joints, device="cpu"), _commands(170))
    assert np.isfinite(truth).all() and np.isfinite(observation).all() and np.isfinite(metrics).all()
    assert np.abs(truth[:, :, 2:]).max() < 60.0
    assert np.unique(np.round(truth[-1, :, 0], 4)).size > 400


def test_mid_episode_change_takes_effect_at_its_tick() -> None:
    arms, joints = healthy_population(2)
    heavier, weaker = sample_change(np.random.default_rng(1), arms, joints)
    heavier["m2"] += 0.2
    weaker["torque_scale"] *= 0.5
    batch = ArmBatch(arms, joints, device="cpu", changed=(heavier, weaker), change_tick=np.array([2**31 - 1, 50]))
    truth, _, _ = _run(batch, _commands(100))
    assert np.abs(truth[:50, 0] - truth[:50, 1]).max() == 0.0
    assert np.abs(truth[60:, 0, :2] - truth[60:, 1, :2]).max() > 1e-2


def test_external_push_moves_an_uncommanded_arm_and_is_off_by_default() -> None:
    arms, joints = healthy_population(2, gravity=0.0)
    push = sample_push(np.random.default_rng(2), 2)
    push[0] = 0.0
    push[1, :, 0] = 0.3
    truth, _, _ = _run(ArmBatch(arms, joints, device="cpu", push=push), np.zeros((100, 2), dtype=np.float32))
    assert np.abs(truth[:, 0, :2] - START.astype(np.float32)).max() == 0.0
    assert np.abs(truth[:, 1, :2] - START.astype(np.float32)).max() > 0.05
