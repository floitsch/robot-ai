# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from robot_ai.control.computed_torque import ComputedTorqueTeacher, chain_dynamics
from robot_ai.sim.chain_batch import ChainBatch
from robot_ai.sim.chain_env import ChainEnv
from test_chain_batch import _chain_links, _healthy_joints


def test_torch_dynamics_match_the_kernel_for_one_substep() -> None:
    rng = np.random.default_rng(4)
    worlds, n = 6, 4
    lengths, masses = rng.uniform(0.15, 0.35, (worlds, n)), rng.uniform(0.2, 0.7, (worlds, n))
    links, joints = _chain_links(lengths, masses), _healthy_joints(worlds, n, np.full(n, 6.0))
    joints["damping"] = rng.uniform(0.0, 0.1, (worlds, n))
    batch = ChainBatch(links, joints, device="cpu", substeps=1)
    q0, dq0 = rng.uniform(-1.0, 1.0, (worlds, n)), rng.uniform(-3.0, 3.0, (worlds, n))
    batch.reset(q0.astype(np.float32))
    # Give the kernel an initial velocity by writing joint states directly.
    states = batch.states.numpy()
    states["dq"] = dq0
    batch.states.assign(states)
    command = rng.uniform(-0.5, 0.5, (worlds, n)).astype(np.float32)
    batch.step(command)
    kernel_dq = batch.truth.numpy()[:, n:]
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    m, bias = chain_dynamics(t(q0), t(dq0), t(lengths), t(masses), t(lengths / 2), t(masses * lengths**2 / 12))
    damping = t(joints["damping"])
    torque = t(command) * 6.0
    m_implicit = m + torch.diag_embed(0.001 * damping)
    rhs = torque - bias - damping * t(dq0)
    predicted = t(dq0) + 0.001 * torch.linalg.solve(m_implicit, rhs)
    assert np.abs(predicted.numpy() - kernel_dq).max() < 1e-4


def test_computed_torque_drives_healthy_chains_to_their_goals() -> None:
    env = ChainEnv(64, 2, device="cpu", seed=9, severity=0.0, changes=False)
    observation = env.reset()
    teacher = ComputedTorqueTeacher(env)
    feeling = teacher.initial(64, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(env.episode_ticks):
            command, feeling = teacher(observation[None], feeling)
            observation, _ = env.step(command[0])
    summary = env.summary()
    assert summary.success.float().mean() > 0.9
    assert summary.final_error.median() < 0.005
