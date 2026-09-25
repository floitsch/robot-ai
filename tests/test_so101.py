# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from robot_ai.sim import so101
from robot_ai.sim.arm3d_env import arm3d_tool

URDF = Path(__file__).resolve().parents[1] / "datasets" / "so101" / "so101_new_calib.urdf"


def test_the_rewritten_chain_moves_like_the_urdf() -> None:
    links, tip = so101.so101_links(1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(so101.LIMITS[:, 0], so101.LIMITS[:, 1])
        point, _ = arm3d_tool(torch.as_tensor(q[None]), torch.as_tensor(links["joint_pos"]).double(),
                              torch.as_tensor(links["axis"]).double(), torch.as_tensor(tip))
        assert np.abs(point[0].numpy() - so101.urdf_tool_point(q)).max() < 1e-6  # float32 storage


def test_the_vendored_numbers_match_the_urdf_when_it_is_present() -> None:
    if not URDF.exists():  # datasets/ is not in the repository; fetch it with the URL in so101.py
        return
    root = ET.parse(URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    for k, name in enumerate(so101.JOINT_NAMES):
        origin = joints[name].find("origin")
        assert np.allclose([float(v) for v in origin.get("xyz").split()], so101.JOINT_XYZ[k])
        assert np.allclose([float(v) for v in origin.get("rpy").split()], so101.JOINT_RPY[k])
        limit = joints[name].find("limit")
        assert np.allclose([float(limit.get("lower")), float(limit.get("upper"))], so101.LIMITS[k])


def test_the_so101_is_about_the_size_of_an_so101() -> None:
    links, tip = so101.so101_links(1)
    stretched = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    reach = np.linalg.norm(so101.urdf_tool_point(stretched)[:2])
    total_mass = links["mass"].sum()
    assert 0.2 < np.linalg.norm(so101.urdf_tool_point(stretched)) < 0.6 and 0.3 < total_mass < 0.8, (reach, total_mass)


def test_the_so101_in_the_kernel_moves_like_the_so101_in_mujoco() -> None:
    import mujoco

    from robot_ai.sim.arm3d_batch import Arm3DBatch
    from test_arm3d import _healthy_joints

    links, tip = so101.so101_links(1)
    xml = ""
    for k in range(5):
        link = links[0, k]
        i = link["inertia"]
        xml += (f'<body pos="{" ".join(map(str, link["joint_pos"]))}">'
                f'<joint name="j{k}" type="hinge" axis="{" ".join(map(str, link["axis"]))}" limited="false" damping="0.02" armature="0.002"/>'
                f'<inertial pos="{" ".join(map(str, link["com"]))}" mass="{link["mass"]}" '
                f'fullinertia="{i[0, 0]} {i[1, 1]} {i[2, 2]} {i[0, 1]} {i[0, 2]} {i[1, 2]}"/>')
    xml += f'<site name="tip" pos="{" ".join(map(str, tip[0]))}"/>' + "</body>" * 5
    motors = "".join(f'<motor joint="j{k}" gear="1"/>' for k in range(5))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>'
        f"<worldbody>{xml}</worldbody><actuator>{motors}</actuator></mujoco>")
    data = mujoco.MjData(model)
    joints = _healthy_joints(1, limit=10.0)
    joints["torque_scale"] = 2.5
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    start = np.array([0.3, -0.4, 0.5, 0.2, 0.6])
    batch.reset(start)
    data.qpos[:] = start
    worst = 0.0
    commands = np.stack([0.05 * np.sin(np.arange(120) * 0.03 * (1 + k)) for k in range(5)], axis=1).astype(np.float32)
    for command in commands:
        batch.step(command[None])
        for _ in range(10):
            data.ctrl[:] = command * 2.5
            mujoco.mj_step(model, data)
        worst = max(worst, float(np.abs(batch.truth.numpy()[0, :5] - data.qpos).max()))
    mujoco.mj_forward(model, data)
    site = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")]
    assert worst < 2e-3 and np.abs(batch.tip.numpy()[0] - site).max() < 2e-3


def test_numeric_inverse_kinematics_finds_the_pose_from_nearby() -> None:
    from robot_ai.sim.arm3d_env import numeric_inverse

    links, tip = so101.so101_links(64)
    rng = np.random.default_rng(1)
    goal = rng.uniform(0.6 * so101.LIMITS[:, 0], 0.6 * so101.LIMITS[:, 1], (64, 5))
    t = lambda a: torch.as_tensor(np.asarray(a)).double()  # noqa: E731
    point, direction = arm3d_tool(t(goal), t(links["joint_pos"]), t(links["axis"]), t(tip))
    pose = torch.cat((point, direction, t(goal[:, -1:])), dim=1)
    seed = t(goal + rng.normal(0.0, 0.1, goal.shape))
    solved = numeric_inverse(pose, seed, t(links["joint_pos"]), t(links["axis"]), t(tip), iterations=20)
    reached, _ = arm3d_tool(solved, t(links["joint_pos"]), t(links["axis"]), t(tip))
    assert (reached - point).norm(dim=1).max() < 1e-4


def test_servo_so101s_sent_their_inverse_kinematics_angles_end_near_their_goals() -> None:
    from robot_ai.sim.arm3d_env import Arm3DEnv

    env = Arm3DEnv(64, device="cpu", seed=3, severity=0.0, changes=False, servo=True, model="so101")
    env.reset()
    with torch.no_grad():
        for _ in range(env.episode_ticks):
            env.step(torch.zeros(64, 5))
    assert env.summary().position_error.median() < 0.02  # proportional servos sag; a baseline, not a solution
