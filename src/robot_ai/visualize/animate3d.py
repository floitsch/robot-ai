"""Animated replays of 3D arms reaching for tool poses: the same imperfect arms under different controllers.

Each panel shows one held-out robot from an oblique camera, with its shadow on the table for depth. The
arm is drawn from the simulator's true joint positions, bends included; the grey marker is the goal
point with the direction the tool should point. The caption is how far the real tool point is from it.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw

from ..sim.arm3d_env import NOMINAL_TORQUES, POSITION_UNIT, Arm3DEnv
from ..sim.reach_env import TOLERANCE
from ..train.reach import EVAL_SEED, load_actor
from .animate import _font

COLORS = {"classical": (42, 120, 214), "network": (27, 175, 122)}
LABELS = {"classical": "Computed torque (perfect knowledge)", "network": "Our network (told only the arm's size)"}
SERVO_LABELS = {"classical": "Servos sent the IK angles", "network": "Our network (told only the arm's size)"}


class IkTargets(torch.nn.Module):
    """Servo arms without a network: every joint's target is the inverse-kinematics solution, as LeRobot users send."""

    def __init__(self, actions: int = 5) -> None:
        super().__init__()
        self.actions = actions

    def initial(self, worlds: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((worlds, 1), device=device)

    def forward(self, observations: torch.Tensor, feeling: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((*observations.shape[:-1], self.actions), device=observations.device), feeling
AZIMUTH, ELEVATION = np.radians(-50.0), np.radians(22.0)


@torch.no_grad()
def record(controller: object, *, worlds: int, device: str, severity: float = 1.0, servo: bool = False,
           model: str = "desktop", contact: bool = False) -> tuple[Arm3DEnv, dict[str, np.ndarray]]:
    """Per tick: joint origins and tool point [ticks, worlds, n + 1, 3], tool direction, goal pose, pose error, contact
    force; and the obstacles."""

    share = 0.5 if contact else 0.0
    env = Arm3DEnv(worlds, device=device, seed=EVAL_SEED, severity=severity, servo=servo, model=model, accidents=share,
                   touches=share)
    observation = env.reset()
    if callable(controller) and not isinstance(controller, torch.nn.Module):
        controller = controller(env)  # type: ignore[operator]
    assert isinstance(controller, torch.nn.Module)
    feeling = controller.initial(worlds, env.torch_device)  # type: ignore[operator]
    origin = wp.to_torch(env.batch.origin)  # type: ignore[attr-defined]  # an Arm3DBatch
    frames: dict[str, list[np.ndarray]] = {"points": [], "direction": [], "goal": [], "error": [], "force": []}
    for _ in range(env.episode_ticks):
        seen = env.privileged(observation) if int(getattr(controller, "oracle", 0)) and not hasattr(controller, "env") else observation
        mean, feeling = controller(seen[None], feeling)
        observation, _ = env.step(mean[0])
        frames["points"].append(torch.cat((origin, env._tip[:, None]), dim=1).cpu().numpy())
        frames["direction"].append(env._tool_axis.cpu().numpy())
        frames["goal"].append(env.goal().cpu().numpy())
        pose_error = env._true_error()
        frames["error"].append(torch.stack((pose_error[:, :3].norm(dim=1) * POSITION_UNIT, pose_error[:, 3:6].norm(dim=1)),
                                           dim=1).cpu().numpy())
        frames["force"].append(env._contact[:, 1].cpu().numpy())
    result = {name: np.stack(values) for name, values in frames.items()}
    obstacles = env.batch.obstacles.numpy()  # type: ignore[attr-defined]
    result["obstacles"] = obstacles
    result["success"] = env.summary().success.cpu().numpy()
    return env, result


def pick_robots(env: Arm3DEnv) -> list[tuple[int, str]]:
    joints, links = env.joints, env.links
    dry = joints["coulomb"] * (1.0 + joints["stribeck"]) + joints["bump0_mag"] + joints["bump1_mag"]
    weakness = 1.0 - joints["torque_scale"] / NOMINAL_TORQUES
    load_drag = links["gear_loss"] + links["bearing_loss"]
    bendy = links["compliance"] * NOMINAL_TORQUES
    score = dry.sum(1) * 2 + joints["half_gap"].sum(1) * 40 + weakness.sum(1) * 2 + load_drag.sum(1) * 4 + bendy.sum(1) * 20
    picks = {"Nearly healthy": int(np.argmin(score)), "Bendy links": int(np.argmax(bendy[:, 1:3].sum(1))),
             "Drags under load": int(np.argmax(load_drag.sum(1))), "Slack gears": int(np.argmax(joints["half_gap"].sum(1))),
             "Weak motors": int(np.argmax(weakness.sum(1))), "Everything at once": int(np.argmax(score))}
    return [(index, title) for title, index in picks.items()]


def _project(points: np.ndarray, centre: tuple[float, float], scale: float) -> np.ndarray:
    """Oblique camera: [..., 3] world points (m) to [..., 2] pixels."""

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    across = x * np.sin(AZIMUTH) + y * np.cos(AZIMUTH)
    depth = x * np.cos(AZIMUTH) - y * np.sin(AZIMUTH)
    up = z * np.cos(ELEVATION) - depth * np.sin(ELEVATION)
    return np.stack((centre[0] + scale * across, centre[1] - scale * up), axis=-1)


def _obstacle(draw: ImageDraw.ImageDraw, centre: tuple[float, float], scale: float, obstacle: np.void) -> None:
    from ..sim.arm3d_batch import BOX, CAPSULE

    grey = (150, 110, 80)
    if int(obstacle["kind"]) == BOX:
        z, x = np.asarray(obstacle["b"]), np.asarray(obstacle["side"])
        y = np.cross(z, x)
        half = np.asarray(obstacle["half"])
        corners = np.array([np.asarray(obstacle["a"]) + sx * half[0] * x + sy * half[1] * y + sz * half[2] * z
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        pixels = _project(corners, centre, scale)
        for i in range(8):
            for j in range(i + 1, 8):
                if (i ^ j).bit_count() == 1:  # corners that differ along one axis share an edge
                    draw.line((*pixels[i], *pixels[j]), fill=grey, width=2)
    elif int(obstacle["kind"]) == CAPSULE:
        ends = _project(np.stack((np.asarray(obstacle["a"]), np.asarray(obstacle["b"]))), centre, scale)
        draw.line((*ends[0], *ends[1]), fill=grey, width=max(3, int(2 * scale * float(obstacle["radius"]))))


def _panel(draw: ImageDraw.ImageDraw, centre: tuple[float, float], scale: float, points: np.ndarray, direction: np.ndarray,
           goal: np.ndarray, color: tuple[int, int, int], obstacles: np.ndarray | None = None) -> None:
    grid = np.linspace(-0.4, 0.4, 5)
    for g in grid:
        for line in (np.array([[g, -0.4, 0.0], [g, 0.4, 0.0]]), np.array([[-0.4, g, 0.0], [0.4, g, 0.0]])):
            a, b = _project(line, centre, scale)
            draw.line((*a, *b), fill=(228, 228, 224), width=1)
    shadow = points.copy()
    shadow[:, 2] = 0.0
    arm = np.concatenate((np.zeros((1, 3)), points))
    flat = np.concatenate((np.zeros((1, 3)), shadow))
    draw.line([tuple(p) for p in _project(flat, centre, scale)], fill=(214, 214, 210), width=7, joint="curve")
    for obstacle in () if obstacles is None else obstacles:
        _obstacle(draw, centre, scale, obstacle)
    target = goal[:3]
    marker = _project(np.stack((target, target + 0.06 * goal[3:6], np.array([target[0], target[1], 0.0]))), centre, scale)
    draw.line((*marker[2], *marker[0]), fill=(205, 205, 205), width=1)
    draw.line((*marker[0], *marker[1]), fill=(150, 150, 150), width=3)
    draw.ellipse((marker[0][0] - 7, marker[0][1] - 7, marker[0][0] + 7, marker[0][1] + 7), outline=(120, 120, 120), width=2)
    pixels = [tuple(p) for p in _project(arm, centre, scale)]
    draw.line(pixels, fill=color, width=9, joint="curve")
    for p in pixels[1:-1]:
        draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(40, 40, 40))
    base = pixels[0]
    draw.rectangle((base[0] - 12, base[1] - 3, base[0] + 12, base[1] + 5), fill=(90, 90, 90))
    tool = _project(np.stack((points[-1], points[-1] + 0.03 * direction)), centre, scale)
    draw.line((*tool[0], *tool[1]), fill=(20, 20, 20), width=3)


def render_gif(runs: dict[str, dict[str, np.ndarray]], picks: list[tuple[int, str]], output: Path, *, stride: int = 3,
               panel: int = 220, contact: bool = False) -> None:
    font, small = _font(14), _font(12)
    names = list(runs)
    label_width = 150
    width, height = label_width + len(picks) * panel, 34 + len(names) * (panel + 22)
    ticks = runs[names[0]]["points"].shape[0]
    frames = []
    for tick in range(0, ticks, stride):
        image = Image.new("RGB", (width, height), (252, 252, 251))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"t = {tick / 100:4.2f} s", fill=(80, 80, 80), font=font)
        for column, (_, title) in enumerate(picks):
            draw.text((label_width + column * panel + 8, 10), title, fill=(11, 11, 11), font=small)
        for row, name in enumerate(names):
            top = 34 + row * (panel + 22)
            words = LABELS[name].split(" (")[0].split(" ")
            for line, word in enumerate(words):
                draw.text((10, top + panel // 2 - 20 + 16 * line), word, fill=COLORS[name], font=font)
            for column, (index, _) in enumerate(picks):
                run = runs[name]
                centre = (label_width + column * panel + panel / 2, top + panel * 0.8)
                _panel(draw, centre, panel * 0.85, run["points"][tick, index], run["direction"][tick, index],
                       run["goal"][tick, index], COLORS[name], run["obstacles"][index])
                position, direction = run["error"][tick, index]
                force = float(run["force"][tick, index])
                if contact:
                    last = tick + stride >= ticks
                    ok = bool(run["success"][index]) if last else force < 10.0
                    text = f"contact {force:5.1f} N {'✓' if last and ok else ''}"
                else:
                    ok = position < TOLERANCE * POSITION_UNIT and direction < TOLERANCE
                    text = f"{1000 * position:5.1f} mm {1000 * direction:4.0f} mrad {'✓' if ok else ''}"
                draw.text((label_width + column * panel + 8, top + panel + 2), text,
                          fill=(20, 130, 60) if ok else (150, 40, 40), font=small)
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=10 * stride, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="network run directory (deployable student or oracle)")
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settle-hold", action="store_true", help="wrap the servo network in the in-position hold")
    args = parser.parse_args()
    from ..control.computed_torque import ComputedTorqueTeacher
    from ..control.settle import SettleHold

    device = torch.device("cuda" if args.device.startswith("cuda") else "cpu")

    def classical(env: Arm3DEnv) -> ComputedTorqueTeacher:
        return ComputedTorqueTeacher(env, omega=15.0, measured=False, ramp=True).to(device)

    network = load_actor(args.run, device) if args.run else None
    task = -1 if network is None else int(network.task)
    servo, model, contact = task in (4, 6, 14, 16), "so101" if task >= 14 else "desktop", task in (6, 16)
    if servo:
        LABELS.update(SERVO_LABELS)
    baseline: object = IkTargets(6 if contact else 5) if servo else classical
    env, first = record(baseline, worlds=args.worlds, device=args.device, severity=args.severity, servo=servo, model=model,
                        contact=contact)
    runs = {"classical": first}
    if network is not None:
        driver: torch.nn.Module = SettleHold(network) if args.settle_hold and servo else network
        runs["network"] = record(driver, worlds=args.worlds, device=args.device, severity=args.severity, servo=servo,
                                 model=model, contact=contact)[1]
    if contact:
        from ..sim.arm3d_env import ACCIDENT, TOUCH

        modes = env.mode
        picks = ([(int(i), "Something in the way") for i in np.flatnonzero(modes == ACCIDENT)[:3]]
                 + [(int(i), "Find the surface") for i in np.flatnonzero(modes == TOUCH)[:3]])
    else:
        picks = pick_robots(env)
    render_gif(runs, picks, args.gif, contact=contact)
    for index, title in picks:
        print(f"robot {index}: {title}; final error " + ", ".join(
            f"{name} {1000 * run['error'][-1, index, 0]:.1f} mm" for name, run in runs.items()))


if __name__ == "__main__":
    main()
