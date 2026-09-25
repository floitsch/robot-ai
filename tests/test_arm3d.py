# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import mujoco
import numpy as np
import torch

from robot_ai.control.computed_torque import ComputedTorqueTeacher
from robot_ai.sim.arm3d_batch import Arm3DBatch
from robot_ai.sim.arm3d_env import (
    JOINTS,
    LENGTHS,
    MASSES,
    NOMINAL_TORQUES,
    TOOL_SIDE_OFFSET,
    Arm3DEnv,
    _links,
    add_tool_mass,
    arm3d_dynamics,
    arm3d_inverse,
    arm3d_tool,
)
from robot_ai.sim.joint_model import JointParams


def _healthy_joints(worlds: int, limit: float = 2.6) -> np.ndarray:
    joints = np.zeros((worlds, JOINTS), dtype=JointParams.numpy_dtype())
    joints["torque_scale"], joints["motor_alpha"], joints["damping"] = NOMINAL_TORQUES, 1.0, 0.02
    joints["bump0_width"] = joints["bump1_width"] = 0.1
    joints["mesh_stiffness"], joints["rotor_inertia"], joints["cur_gain"] = 100.0, 0.002, 1.0
    joints["q_min"], joints["q_max"] = -limit, limit
    return joints


def _arm(worlds: int, payload: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    links, tip = _links(np.tile(LENGTHS, (worlds, 1)), np.tile(MASSES, (worlds, 1)), np.full(worlds, TOOL_SIDE_OFFSET))
    add_tool_mass(links, tip, np.full(worlds, payload))  # makes the last link's inertia a full tensor
    return links, tip


def _mujoco(links: np.ndarray, tip: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml = ""
    for k in range(JOINTS):
        link = links[0, k]
        i = link["inertia"]
        xml += (f'<body pos="{" ".join(map(str, link["joint_pos"]))}">'
                f'<joint name="j{k}" type="hinge" axis="{" ".join(map(str, link["axis"]))}" limited="false" damping="0.02" armature="0.002"/>'
                f'<inertial pos="{" ".join(map(str, link["com"]))}" mass="{link["mass"]}" '
                f'fullinertia="{i[0, 0]} {i[1, 1]} {i[2, 2]} {i[0, 1]} {i[0, 2]} {i[1, 2]}"/>')
    xml += f'<site name="tip" pos="{" ".join(map(str, tip[0]))}"/>' + "</body>" * JOINTS
    motors = "".join(f'<motor joint="j{k}" gear="1"/>' for k in range(JOINTS))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>'
        f"<worldbody>{xml}</worldbody><actuator>{motors}</actuator></mujoco>")
    return model, mujoco.MjData(model)


def _compare_with_mujoco(start: np.ndarray, commands: np.ndarray) -> tuple[float, float, float]:
    """Worst joint-angle difference, joint-angle range covered, and final tool-point difference."""

    links, tip = _arm(1)
    joints = _healthy_joints(1, limit=10.0)
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(start)
    model, data = _mujoco(links, tip)
    data.qpos[:] = start
    mujoco.mj_forward(model, data)
    worst, seen = 0.0, []
    for command in commands:
        batch.step(command[None].astype(np.float32))
        for _ in range(10):
            data.ctrl[:] = command * NOMINAL_TORQUES
            mujoco.mj_step(model, data)
        q = batch.truth.numpy()[0, :JOINTS].copy()  # a CPU Warp array's numpy() is a view
        worst = max(worst, float(np.abs(q - data.qpos).max()))
        seen.append(q)
    mujoco.mj_forward(model, data)
    site = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")]
    return worst, float(np.ptp(np.array(seen), axis=0).max()), float(np.abs(batch.tip.numpy()[0] - site).max())


def test_arm_swinging_near_its_hanging_pose_matches_mujoco() -> None:
    time = np.arange(170) * 0.01
    commands = np.stack([0.05 * np.sin((1.0 + 0.6 * k) * time + k) for k in range(JOINTS)], axis=1)
    worst, covered, tip_error = _compare_with_mujoco(np.array([0.3, np.pi - 0.4, 0.3, -0.2, 0.5]), commands)
    assert covered > 0.3  # it actually moved
    assert worst < 2e-3 and tip_error < 2e-3


def test_arm_falling_from_upright_with_all_joints_coupled_matches_mujoco() -> None:
    commands = np.tile(np.array([0.3, 0.05, -0.05, 0.1, 0.2]), (30, 1))
    worst, covered, tip_error = _compare_with_mujoco(np.array([0.2, 0.25, 0.4, -0.3, 0.6]), commands)
    assert covered > 0.2
    assert worst < 2e-3 and tip_error < 2e-3


def test_torch_dynamics_match_the_kernel_for_one_substep() -> None:
    rng = np.random.default_rng(3)
    worlds = 6
    links, tip = _arm(worlds, payload=0.07)
    links["mass"] *= rng.uniform(0.8, 1.2, (worlds, JOINTS))
    joints = _healthy_joints(worlds, limit=10.0)
    joints["damping"] = rng.uniform(0.0, 0.1, (worlds, JOINTS))
    batch = Arm3DBatch(links, joints, tip, device="cpu", substeps=1)
    q0, dq0 = rng.uniform(-1.2, 1.2, (worlds, JOINTS)), rng.uniform(-3.0, 3.0, (worlds, JOINTS))
    batch.reset(q0.astype(np.float32))
    states = batch.states.numpy()
    states["dq"] = dq0
    batch.states.assign(states)
    command = rng.uniform(-0.5, 0.5, (worlds, JOINTS)).astype(np.float32)
    batch.step(command)
    kernel_dq = batch.truth.numpy()[:, JOINTS:]

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float64)

    m, bias = arm3d_dynamics(t(q0), t(dq0), t(links["joint_pos"]), t(links["axis"]), t(links["com"]), t(links["mass"]),
                             t(links["inertia"]))
    damping = t(joints["damping"])
    rhs = t(command) * t(NOMINAL_TORQUES) - bias - damping * t(dq0)
    armature = t(joints["rotor_inertia"])
    predicted = t(dq0) + 0.001 * torch.linalg.solve(m + torch.diag_embed(armature + 0.001 * damping), rhs)
    assert np.abs(predicted.numpy() - kernel_dq).max() < 1e-4


def test_undriven_arm_without_damping_stays_bounded() -> None:
    links, tip = _arm(3)
    joints = _healthy_joints(3, limit=10.0)
    joints["damping"] = 0.0
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(np.array([0.5, 1.0, -0.8, 0.6, 0.3]))
    for _ in range(300):
        batch.step(np.zeros((3, JOINTS), dtype=np.float32))
    truth = batch.truth.numpy()
    assert np.isfinite(truth).all() and np.abs(truth[:, JOINTS:]).max() < 40.0


def test_computed_torque_masters_healthy_3d_arms() -> None:
    env = Arm3DEnv(64, device="cpu", seed=9, severity=0.0, changes=False)
    observation = env.reset()
    teacher = ComputedTorqueTeacher(env, omega=12.0, measured=True, ramp=False)
    feeling = teacher.initial(64, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(env.episode_ticks):
            command, feeling = teacher(observation[None], feeling)
            observation, _ = env.step(command[0])
    summary = env.summary()
    assert summary.success.float().mean() > 0.9
    assert summary.final_error.median() < 0.01


def test_an_unbalanced_turntable_is_harder_to_turn_than_a_balanced_one() -> None:
    links, tip = _arm(2)
    links["bearing_loss"][:, 0] = 0.1
    joints = _healthy_joints(2, limit=10.0)
    pitch = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0, 0.0]])  # upright; shoulder out horizontally
    for k in (1, 2, 3, 4):  # hold the pose with the other joints locked
        joints["q_min"][:, k] = joints["q_max"][:, k] = pitch[:, k]
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(pitch)
    command = np.zeros((2, JOINTS), dtype=np.float32)
    command[:, 0] = 0.1  # 0.3 N m, more than the bearing's drag upright, less than with the arm out
    for _ in range(50):
        batch.step(command)
    turned = batch.truth.numpy()[:, 0]
    assert turned[0] > 0.3 and abs(turned[1]) < 1e-3


def test_a_compliant_link_sags_under_the_load_it_carries() -> None:
    compliance = 0.004
    pose = np.array([0.0, 1.5, 0.3, -0.2, 0.4])
    tips = []
    for c in (0.0, compliance):
        links, tip = _arm(1)
        links["compliance"][:, 1] = c
        joints = _healthy_joints(1, limit=10.0)
        joints["q_min"] = joints["q_max"] = pose  # rigidly held: only the bend moves the tool
        batch = Arm3DBatch(links, joints, tip, device="cpu")
        batch.reset(pose)
        for _ in range(3):
            batch.step(np.zeros((1, JOINTS), dtype=np.float32))
        tips.append(batch.tip.numpy()[0].copy())
    links, _ = _arm(1)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float64)

    _, bias = arm3d_dynamics(t(pose[None]), t(np.zeros((1, JOINTS))), t(links["joint_pos"]), t(links["axis"]),
                             t(links["com"]), t(links["mass"]), t(links["inertia"]))
    # The shoulder's load is (almost) all about its own axis, across the link: the shoulder link turns down by
    # compliance * load, and the tool, `reach` beyond the shoulder, drops by that angle times the reach.
    reach = np.hypot(*(tips[0] - np.array([0.0, 0.0, links["joint_pos"][0, 1, 2]]))[[0, 2]])
    drop = tips[0][2] - tips[1][2]
    expected = compliance * abs(float(bias[0, 1])) * reach
    assert expected > 0.003  # millimetres, as on a real printed arm
    assert abs(drop - expected) < 0.1 * expected


def test_a_compliant_link_also_bends_when_the_arm_accelerates() -> None:
    links, tip = _arm(1)
    links["compliance"][:, 1] = 0.004
    joints = _healthy_joints(1, limit=10.0)
    batch = Arm3DBatch(links, joints, tip, device="cpu", gravity=0.0)  # no weight: only motion loads the link
    pose = np.array([0.0, 1.5, 0.0, 0.0, 0.0])  # shoulder link out horizontally along +x
    batch.reset(pose)
    batch.step(np.zeros((1, JOINTS), dtype=np.float32))
    at_rest = batch.tip.numpy()[0].copy()
    command = np.zeros((1, JOINTS), dtype=np.float32)
    command[0, 0] = 1.0  # the base swings the arm round as hard as it can
    batch.step(command)
    yaw = float(batch.truth.numpy()[0, 0])
    moving = batch.tip.numpy()[0]
    lag = yaw - np.arctan2(moving[1], moving[0])  # a rigid arm's tool would point exactly along the base's yaw
    assert abs(np.arctan2(at_rest[1], at_rest[0])) < 1e-6
    assert yaw > 0.0 and 1e-3 < lag < 0.05  # the outstretched link trails behind the turning base


def test_backlash_on_a_light_wrist_roll_stays_stable() -> None:
    links, tip = _arm(8)
    joints = _healthy_joints(8, limit=10.0)
    joints["half_gap"][:, 4], joints["mesh_stiffness"][:, 4], joints["mesh_damping"][:, 4] = 0.01, 400.0, 0.05
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(np.array([0.0, 0.8, 0.5, -0.3, 0.0]))
    rng = np.random.default_rng(1)
    for _ in range(100):
        batch.step(rng.uniform(-0.3, 0.3, (8, JOINTS)).astype(np.float32))
    assert np.abs(batch.truth.numpy()[:, JOINTS:]).max() < 50.0  # explicit mesh coupling reached thousands of rad/s


def test_a_motor_cannot_drive_past_its_no_load_speed() -> None:
    links, tip = _arm(1)
    joints = _healthy_joints(1, limit=10.0)
    joints["no_load_speed"] = 6.0
    joints["q_min"][:, 1:] = joints["q_max"][:, 1:] = 0.0  # the rest of the arm held upright
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(np.zeros(JOINTS))
    command = np.zeros((1, JOINTS), dtype=np.float32)
    command[0, 0] = 1.0  # the base turns about the vertical: nothing but the motor's own curve limits it
    for _ in range(100):
        batch.step(command)
    speed = float(batch.truth.numpy()[0, JOINTS])
    assert 5.0 < speed < 6.0


def test_inverse_kinematics_finds_the_sampled_pose_and_its_alternatives() -> None:
    env = Arm3DEnv(64, device="cpu", seed=4, severity=1.0, changes=False)
    env.reset()
    goals, pose = env._goals[0].double(), env._pose_goals[0].double()
    joint_pos, tip = env._joint_pos.double(), env._tip_offset.double()
    solutions = arm3d_inverse(pose, joint_pos, tip)
    for k in range(4):
        point, direction = arm3d_tool(solutions[:, k], joint_pos, env._axis.double(), tip)
        assert (point - pose[:, :3]).abs().max() < 1e-6 and (direction - pose[:, 3:6]).abs().max() < 1e-6
    wrapped = torch.remainder(goals + torch.pi, 2.0 * torch.pi) - torch.pi
    assert ((solutions - wrapped[:, None]).abs().amax(dim=2).amin(dim=1) < 1e-4).all()  # the sampled one is among them (float32 goals)
    assert ((solutions.float() - env.joint_goal()[:, None]).abs().amax(dim=2).amin(dim=1) < 1e-4).all()  # the teacher picks one


def _servo_joints(worlds: int) -> np.ndarray:
    joints = _healthy_joints(worlds, limit=10.0)
    joints["servo_gain"], joints["servo_damping"] = 10.0, 0.02
    joints["servo_quantum"] = 2.0 * np.pi / 4096
    joints["no_load_speed"] = 6.0
    return joints


def test_position_servos_hold_an_arm_against_gravity() -> None:
    links, tip = _arm(1)
    joints = _servo_joints(1)
    pose = np.array([0.3, 1.0, 0.6, -0.4, 0.2], dtype=np.float32)
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(pose)
    for _ in range(150):
        batch.step(pose[None])  # target: stay where you are
    q, dq = batch.truth.numpy()[0, :JOINTS], batch.truth.numpy()[0, JOINTS:]
    # A proportional servo sags until its error pays for the load: gravity torque / (stall torque * gain).
    assert np.abs(q - pose).max() < 0.05 and np.abs(dq).max() < 0.05


def test_a_position_servo_reaches_a_new_target_despite_host_latency() -> None:
    links, tip = _arm(1)
    joints = _servo_joints(1)
    joints["delay_steps"] = 30  # the host's command arrives 30 ms late; the servo's own loop is not delayed
    joints["q_min"][:, 1:] = joints["q_max"][:, 1:] = 0.0  # the rest held upright: a clean single-joint step
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(np.zeros(JOINTS))
    target = np.zeros((1, JOINTS), dtype=np.float32)
    target[0, 0] = 0.5
    path = []
    for _ in range(100):
        batch.step(target)
        path.append(float(batch.truth.numpy()[0, 0]))
    assert abs(path[1]) < 1e-3  # nothing happens before the command arrives
    assert abs(path[-1] - 0.5) < 0.005 and max(path) < 0.55  # settles on target without much overshoot


def test_position_servos_with_backlash_on_a_light_wrist_stay_stable() -> None:
    links, tip = _arm(4)
    joints = _servo_joints(4)
    joints["half_gap"][:, 4], joints["mesh_stiffness"][:, 4], joints["mesh_damping"][:, 4] = 0.01, 400.0, 0.05
    pose = np.array([0.0, 0.8, 0.5, -0.3, 0.0], dtype=np.float32)
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(pose)
    rng = np.random.default_rng(2)
    for _ in range(200):
        batch.step((pose + rng.uniform(-0.2, 0.2, (4, JOINTS))).astype(np.float32))
    assert np.isfinite(batch.truth.numpy()).all() and np.abs(batch.truth.numpy()[:, JOINTS:]).max() < 20.0


def test_servo_arms_reach_healthy_goals_with_the_inverse_kinematics_targets_alone() -> None:
    env = Arm3DEnv(64, device="cpu", seed=9, severity=0.0, changes=False, servo=True)
    env.reset()
    with torch.no_grad():
        for _ in range(env.episode_ticks):
            env.step(torch.zeros(64, JOINTS))  # the policy's "do nothing": targets at the IK solution
    summary = env.summary()
    # Proportional servos sag under load, so this is a baseline, not a solution: most arms end near the goal.
    assert summary.position_error.median() < 0.02


def test_a_softer_servo_sags_further_under_the_same_load() -> None:
    links, tip = _arm(2)
    joints = _servo_joints(2)
    joints["servo_deadband"] = 0.0
    pose = np.array([0.0, 1.2, 0.3, 0.0, 0.0], dtype=np.float32)
    batch = Arm3DBatch(links, joints, tip, device="cpu")
    batch.reset(pose)
    stiffness = np.array([[1.0] * JOINTS, [0.5] * JOINTS], dtype=np.float32)
    for _ in range(150):
        batch.step(np.tile(pose, (2, 1)), stiffness)
    sag = np.abs(batch.truth.numpy()[:, 1] - pose[1])
    assert 1.6 < sag[1] / sag[0] < 2.4  # a proportional servo's droop is inversely proportional to its gain


def test_a_stiffness_network_steps_with_twice_as_many_actions() -> None:
    env = Arm3DEnv(8, device="cpu", seed=2, severity=0.5, servo=True, stiffness=True)
    observation = env.reset()
    assert env.action_dim == 2 * JOINTS and observation.shape == (8, env.observation_dim)
    for _ in range(5):
        observation, reward = env.step(torch.zeros(8, env.action_dim), reference=torch.zeros(8, env.action_dim))
    assert torch.isfinite(reward).all()


def _obstacles(worlds: int, *items: dict) -> np.ndarray:
    from robot_ai.sim.arm3d_batch import MAX_OBSTACLES, Obstacle

    obstacles = np.zeros((worlds, MAX_OBSTACLES), dtype=Obstacle.numpy_dtype())
    for k, item in enumerate(items):
        for name, value in {"stiffness": 1e4, "damping": 30.0, "friction": 0.3, **item}.items():
            obstacles[name][:, k] = value
    return obstacles


def test_an_arm_resting_on_a_table_is_held_up_by_the_contact_force() -> None:
    from robot_ai.sim.arm3d_batch import HALF_SPACE

    links, tip = _arm(1)
    joints = _servo_joints(1)
    joints["damping"] = 0.2
    joints["servo_gain"][:, 1] = 0.0  # the shoulder is free (a torque drive told to do nothing); servos hold the rest
    pose = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float32)  # shoulder out horizontally, free to fall
    table = {"kind": HALF_SPACE, "a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 1.0)}
    batch = Arm3DBatch(links, joints, tip, device="cpu", obstacles=_obstacles(1, table))
    batch.reset(pose)
    command = pose.copy()
    command[1] = 0.0  # no shoulder torque
    for _ in range(400):
        batch.step(command[None])
    q = batch.truth.numpy()[0, :JOINTS].astype(np.float64)
    assert np.abs(batch.truth.numpy()[0, JOINTS:]).max() < 0.02  # at rest, but for the servos' hold dither

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=torch.float64)

    _, bias = arm3d_dynamics(t(q[None]), t(np.zeros((1, JOINTS))), t(links["joint_pos"]), t(links["axis"]), t(links["com"]),
                             t(links["mass"]), t(links["inertia"]))
    point, _ = arm3d_tool(t(q[None]), t(links["joint_pos"]), t(links["axis"]), t(tip))
    lever = float(point[0, 0])  # the contact is at the tool end, straight below it; the shoulder turns about y at x = 0
    force = float(batch.contact.numpy()[0, 0])
    assert abs(force * lever - abs(float(bias[0, 1]))) < 0.05 * abs(float(bias[0, 1]))
    assert float(point[0, 2]) > links["radius"][0, -1] - 0.003  # resting on the surface, barely in it


def test_a_falling_arm_does_not_tunnel_through_the_table() -> None:
    from robot_ai.sim.arm3d_batch import HALF_SPACE

    links, tip = _arm(1)
    joints = _healthy_joints(1, limit=10.0)
    table = {"kind": HALF_SPACE, "a": (0.0, 0.0, 0.0), "b": (0.0, 0.0, 1.0)}
    batch = Arm3DBatch(links, joints, tip, device="cpu", obstacles=_obstacles(1, table))
    batch.reset(np.array([0.0, 1.2, 0.3, 0.2, 0.0]))
    lowest = 1.0
    command = np.zeros((1, JOINTS), dtype=np.float32)
    command[0, 1] = 1.0  # the shoulder drives the arm down as hard as it can
    for _ in range(150):
        batch.step(command)
        lowest = min(lowest, float(batch.tip.numpy()[0, 2]) - float(links["radius"][0, -1]))
    assert lowest > -0.005 and float(batch.contact.numpy()[0, 2]) > 5.0  # it hit, and stayed on top


def test_a_post_stops_an_arm_swinging_into_it() -> None:
    from robot_ai.sim.arm3d_batch import CAPSULE

    links, tip = _arm(1)
    joints = _healthy_joints(1, limit=10.0)
    pose = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0])
    joints["q_min"][:, 1:] = joints["q_max"][:, 1:] = pose[1:]
    angle = 0.6  # a vertical post at this bearing, half way along the arm
    post = {"kind": CAPSULE, "a": (0.25 * np.cos(angle), 0.25 * np.sin(angle), 0.0),
            "b": (0.25 * np.cos(angle), 0.25 * np.sin(angle), 0.3), "radius": 0.02}
    batch = Arm3DBatch(links, joints, tip, device="cpu", obstacles=_obstacles(1, post))
    batch.reset(pose)
    command = np.zeros((1, JOINTS), dtype=np.float32)
    command[0, 0] = 0.5
    for _ in range(150):
        batch.step(command)
    yaw = float(batch.truth.numpy()[0, 0])
    assert 0.3 < yaw < angle and float(batch.contact.numpy()[0, 0]) > 1.0  # pressed against the post, not through it
