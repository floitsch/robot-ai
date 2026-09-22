# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import mujoco
import numpy as np

from robot_ai.sim.arm_batch import ArmBatch
from robot_ai.sim.chain_batch import ChainBatch, LinkParams
from robot_ai.sim.joint_model import JointParams
from robot_ai.sim.population import healthy_population


def _chain_links(lengths: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Rod links, one row of worlds; lengths and masses are [worlds, joints]."""

    links = np.zeros(lengths.shape, dtype=LinkParams.numpy_dtype())
    links["length"], links["mass"], links["com"] = lengths, masses, lengths / 2.0
    links["inertia"] = masses * lengths**2 / 12.0
    return links


def _healthy_joints(worlds: int, n: int, torques: np.ndarray) -> np.ndarray:
    joints = np.zeros((worlds, n), dtype=JointParams.numpy_dtype())
    joints["torque_scale"], joints["motor_alpha"], joints["damping"] = torques, 1.0, 0.02
    joints["bump0_width"] = joints["bump1_width"] = 0.1
    joints["mesh_stiffness"], joints["rotor_inertia"], joints["cur_gain"] = 100.0, 0.002, 1.0
    joints["q_min"], joints["q_max"] = -2.6, 2.6
    return joints


def _commands(ticks: int, n: int, amplitude: float = 0.12) -> np.ndarray:
    time = np.arange(ticks) * 0.01
    values = np.stack([amplitude * np.sin((1.0 + 0.7 * j) * time + j) for j in range(n)], axis=1)
    values[:4] = 0.0
    return values.astype(np.float32)


def _run(batch: ChainBatch | ArmBatch, commands: np.ndarray, start: np.ndarray) -> np.ndarray:
    batch.reset(start)
    truth = []
    for command in commands:
        batch.step(np.tile(command, (batch.worlds, 1)))
        truth.append(batch.truth.numpy().copy())
    return np.stack(truth)


def test_two_joint_chain_matches_the_arm_kernel() -> None:
    arms, joints = healthy_population(2)
    joints["coulomb"], joints["half_gap"][:, 1], joints["delay_steps"][:, 0] = 0.15, 0.01, 12
    joints["mesh_stiffness"], joints["mesh_damping"] = 200.0, 0.3
    links = _chain_links(np.stack((arms["l1"], arms["l2"]), axis=1), np.stack((arms["m1"], arms["m2"]), axis=1))
    commands = _commands(170, 2)
    start = np.array([0.3, 0.9])
    arm = _run(ArmBatch(arms, joints, device="cpu"), commands, start)
    chain = _run(ChainBatch(links, joints, device="cpu"), commands, start)
    assert np.abs(arm - chain).max() < 2e-4


def test_four_joint_chain_matches_native_mujoco() -> None:
    n = 4
    lengths = np.array([[0.30, 0.25, 0.22, 0.18]])
    masses = np.array([[0.60, 0.40, 0.35, 0.25]])
    torques = np.array([12.0, 6.0, 6.0, 3.0])
    links, joints = _chain_links(lengths, masses), _healthy_joints(1, n, torques)
    bodies = ""
    for j in range(n):
        length, mass, inertia = lengths[0, j], masses[0, j], masses[0, j] * lengths[0, j] ** 2 / 12
        bodies += (f'<body name="link{j}" pos="0 0 {0 if j == 0 else -lengths[0, j - 1]}">'
                   f'<joint name="j{j}" type="hinge" axis="0 -1 0" range="-2.6 2.6" limited="true" damping="0.02"/>'
                   f'<inertial pos="0 0 {-length / 2}" mass="{mass}" diaginertia="{inertia} {inertia} 1e-6"/>')
    bodies += "</body>" * n
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>'
        f"<worldbody>{bodies}</worldbody><actuator>"
        + "".join(f'<motor joint="j{j}" gear="1" ctrllimited="true" ctrlrange="{-torques[j]} {torques[j]}"/>' for j in range(n))
        + "</actuator></mujoco>")
    data = mujoco.MjData(model)
    start = np.array([0.3, 0.6, -0.4, 0.5])
    commands = _commands(170, n, 0.04)  # keeps every joint off its hard stop, where MuJoCo's soft limits differ
    truth = _run(ChainBatch(links, joints, device="cpu"), commands, start)
    data.qpos[:] = start
    mujoco.mj_forward(model, data)
    worst = 0.0
    for tick, command in enumerate(commands):
        for _ in range(10):
            data.ctrl[:] = command * torques
            mujoco.mj_step(model, data)
        worst = max(worst, np.abs(truth[tick, 0, :n] - data.qpos).max())
    assert np.ptp(truth[:, 0, 0]) > 0.3  # it actually swung
    assert worst < 2e-3


def test_chain_energy_is_bounded_without_drive() -> None:
    n = 5
    lengths, masses = np.full((3, n), 0.2), np.full((3, n), 0.3)
    joints = _healthy_joints(3, n, np.full(n, 4.0))
    joints["damping"] = 0.0
    batch = ChainBatch(_chain_links(lengths, masses), joints, device="cpu")
    batch.reset(np.array([1.2, -0.8, 0.6, 0.4, -0.9]))
    for _ in range(300):
        batch.step(np.zeros((3, n), dtype=np.float32))
    truth = batch.truth.numpy()
    assert np.isfinite(truth).all() and np.abs(truth[:, n:]).max() < 40.0
