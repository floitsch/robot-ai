"""Animated side-by-side replays: the same imperfect arm driven by different controllers.

Produces a GIF (for a README) and a self-contained HTML page with a scrub bar.
"""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..sim.chain_env import ChainEnv
from ..sim.reach_env import TOLERANCE, ReachEnv
from ..train.reach import EVAL_SEED, PidController, load_actor
from .reach import _pick_robots

COLORS = {"pid": (42, 120, 214), "network": (27, 175, 122)}
LABELS = {"pid": "Tuned PID", "network": "Our network (no configuration)"}
CHAIN_LABELS = {"pid": "Computed torque (perfect knowledge)", "network": "Our network (no configuration)"}


@torch.no_grad()
def record_truth(controller: object, *, worlds: int, device: str, pushes: bool = False, limbs: int = 0,
                 severity: float = 1.0) -> tuple[ReachEnv | ChainEnv, np.ndarray, np.ndarray]:
    """True joint angles [ticks, worlds, n] and goals in encoder units, on the held-out robots.

    `controller` is a network, a PID (`act(env)`), or a factory `env -> network` for teachers that read the environment.
    """

    env: ReachEnv | ChainEnv
    if limbs:
        env = ChainEnv(worlds, limbs, device=device, seed=EVAL_SEED, severity=severity)
    else:
        env = ReachEnv(worlds, device=device, seed=EVAL_SEED, pushes=pushes, severity=severity)
    observation = env.reset()
    if callable(controller) and not isinstance(controller, torch.nn.Module):
        controller = controller(env)  # type: ignore[operator]
    n = getattr(env, "n", 2)
    feeling = controller.initial(worlds, env.torch_device) if isinstance(controller, torch.nn.Module) else None  # type: ignore[operator]
    truth, goals = [], []
    for _ in range(env.episode_ticks):
        if isinstance(controller, torch.nn.Module):
            mean, feeling = controller(observation[None], feeling)
            action = env.integrate(mean[0]) if isinstance(env, ReachEnv) and bool(getattr(controller, "incremental", False)) else mean[0]
        else:
            action = controller.act(env)  # type: ignore[attr-defined]
        goals.append(env.goal().cpu().numpy())
        observation, _ = env.step(action)
        truth.append(env._truth[:, :n].cpu().numpy())
    return env, np.stack(truth), np.stack(goals)


def _points(q: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Joint positions after each link, base excluded, for a planar chain of any length."""

    theta = np.cumsum(q, axis=-1)
    steps = np.stack((lengths * np.sin(theta), -lengths * np.cos(theta)), axis=-1)
    positions = np.cumsum(steps, axis=-2)
    return [positions[..., k, :] for k in range(lengths.shape[-1])]


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_arm(draw: ImageDraw.ImageDraw, origin: tuple[int, int], scale: float, q: np.ndarray, goal_q: np.ndarray,
              lengths: np.ndarray, color: tuple[int, int, int], bias: np.ndarray) -> float:
    """Draw one arm and its goal pose; return the largest joint error in encoder units, rad."""

    def pixel(xy: np.ndarray) -> tuple[float, float]:
        return origin[0] + scale * float(xy[0]), origin[1] - scale * float(xy[1])

    # The goal is given in encoder units; the arm that would satisfy it is at goal - bias in true angles.
    base = pixel(np.zeros(2))
    ghost = [base] + [pixel(point) for point in _points(goal_q - bias, lengths)]
    draw.line(ghost, fill=(200, 200, 200), width=10, joint="curve")
    tip = ghost[-1]
    draw.ellipse((tip[0] - 9, tip[1] - 9, tip[0] + 9, tip[1] + 9), outline=(120, 120, 120), width=2)
    real = [base] + [pixel(point) for point in _points(q, lengths)]
    draw.line(real, fill=color, width=14, joint="curve")
    for point in real[:-1]:
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=(40, 40, 40))
    tip = real[-1]
    draw.ellipse((tip[0] - 5, tip[1] - 5, tip[0] + 5, tip[1] + 5), fill=(20, 20, 20))
    return float(np.abs(goal_q - (q + bias)).max())


def _lengths(env: ReachEnv | ChainEnv, index: int) -> np.ndarray:
    if isinstance(env, ChainEnv):
        return np.asarray(env.links["length"][index], dtype=np.float64)
    return np.array([env.arms["l1"][index], env.arms["l2"][index]])


def _pick_chain_robots(env: ChainEnv) -> list[tuple[int, str]]:
    from ..sim.chain_env import nominal_torques

    joints = env.joints
    dry = joints["coulomb"] * (1.0 + joints["stribeck"]) + joints["bump0_mag"] + joints["bump1_mag"]
    weakness = 1.0 - joints["torque_scale"] / nominal_torques(env.limbs)
    score = dry.sum(1) * 2 + joints["half_gap"].sum(1) * 40 + weakness.sum(1) * 2 + joints["delay_steps"].sum(1) / 30
    changing = env.change_tick < env.episode_ticks

    def fmt(values: np.ndarray, scale: float = 1.0, digits: int = 2) -> str:
        return "/".join(f"{scale * v:.{digits}f}" for v in values)

    def describe(index: int) -> str:
        parts = [f"dry friction {fmt(dry[index])} N m", f"backlash {fmt(joints['half_gap'][index], 2000, 0)} mrad",
                 f"motors at {fmt(100 - 100 * weakness[index], 1, 0)}%", f"command latency {fmt(joints['delay_steps'][index], 1, 0)} ms"]
        if changing[index]:
            parts.append(f"changes at {env.change_tick[index] / 100:.2f} s")
        return "; ".join(parts)

    picks = {"Nearly healthy": int(np.argmin(score)), "Sticky joints": int(np.argmax(dry.sum(1))),
             "Slack gears": int(np.argmax(joints["half_gap"].sum(1))), "Weak motors": int(np.argmax(weakness.sum(1))),
             "Changes mid-move": int(np.argmax(np.where(changing, score, -1.0))), "Everything at once": int(np.argmax(score))}
    return [(index, f"{title} — {describe(index)}") for title, index in picks.items()]


def render_gif(runs: dict[str, tuple[np.ndarray, np.ndarray]], env: ReachEnv, picks: list[tuple[int, str]], output: Path,
               *, stride: int = 3, panel: int = 210) -> None:
    """One column per selected robot, one row per controller, all robots moving at once."""

    font, small = _font(14), _font(12)
    names = list(runs)
    columns, rows = len(picks), len(names)
    label_width = 150
    width, height = label_width + columns * panel, 34 + rows * (panel + 22)
    ticks = next(iter(runs.values()))[0].shape[0]
    frames: list[Image.Image] = []
    for tick in range(0, ticks, stride):
        image = Image.new("RGB", (width, height), (252, 252, 251))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"t = {tick / 100:4.2f} s", fill=(80, 80, 80), font=font)
        for column, (_, title) in enumerate(picks):
            draw.text((label_width + column * panel + 8, 10), title.split(" — ")[0], fill=(11, 11, 11), font=small)
        for row, name in enumerate(names):
            top = 34 + row * (panel + 22)
            for line, word in enumerate(LABELS[name].split(" (")[0].split(" ")):
                draw.text((10, top + panel // 2 - 20 + 16 * line), word, fill=COLORS[name], font=font)
            for column, (index, _) in enumerate(picks):
                lengths = _lengths(env, index)
                bias = env.joints["enc_bias"][index]
                truth, goals = runs[name]
                chain = lengths.size > 2
                origin = (label_width + column * panel + panel // 2, top + panel // 2 - 24 + (30 if chain else 0))
                error = _draw_arm(draw, origin, panel * (0.4 if chain else 0.72), truth[tick, index], goals[tick, index],
                                  lengths, COLORS[name], bias)
                ok = error < TOLERANCE
                draw.text((label_width + column * panel + 8, top + panel + 2), f"{1000 * error:4.0f} mrad {'✓' if ok else ''}",
                          fill=(20, 130, 60) if ok else (150, 40, 40), font=small)
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=10 * stride, loop=0, optimize=True)


def render_html(runs: dict[str, tuple[np.ndarray, np.ndarray]], env: ReachEnv, picks: list[tuple[int, str]], output: Path) -> None:
    payload = {
        "controllers": [{"name": name, "label": LABELS[name], "color": "rgb({},{},{})".format(*COLORS[name])} for name in runs],
        "robots": [{"title": title, "lengths": _lengths(env, index).tolist(),
                    "bias": env.joints["enc_bias"][index].tolist(),
                    "truth": {name: runs[name][0][:, index].round(4).tolist() for name in runs},
                    "goal": {name: runs[name][1][:, index].round(4).tolist() for name in runs}} for index, title in picks],
        "tolerance": TOLERANCE,
    }
    page = f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Arm replays</title>
<style>
:root{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#d9d8d2}}
@media(prefers-color-scheme:dark){{:root{{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#3a3a37}}}}
body{{background:var(--surface);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:960px;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}}p,figcaption{{color:var(--ink2)}}.controls{{display:flex;gap:12px;align-items:center;margin:12px 0;flex-wrap:wrap}}
input[type=range]{{flex:1;min-width:200px}}button{{font:inherit;padding:4px 12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}figure{{margin:0}}canvas{{width:100%;border:1px solid var(--grid);border-radius:6px;background:#fff}}
figcaption{{font-size:13px;margin:4px 0 12px}}.legend{{display:flex;gap:16px;font-size:13px;margin:6px 0}}.legend i{{display:inline-block;width:14px;height:6px;vertical-align:middle;margin-right:6px;border-radius:3px}}
</style>
<h1>Same broken arm, two controllers</h1>
<p>Grey ghost: the pose the goal asks for. Each robot below is one of the held-out arms; hover a caption for its defects.</p>
<div class=legend id=legend></div>
<div class=controls><button id=play>Pause</button><input id=scrub type=range min=0 value=0><span id=time></span></div>
<div class=grid id=grid></div>
<script>
const data={json.dumps(payload)};
const legend=document.getElementById('legend');for(const c of data.controllers){{const s=document.createElement('span');s.innerHTML=`<i style="background:${{c.color}}"></i>${{c.label}}`;legend.appendChild(s);}}
const grid=document.getElementById('grid'),canvases=[];
for(const r of data.robots){{const f=document.createElement('figure');const c=document.createElement('canvas');c.width=600;c.height=380;f.appendChild(c);
const cap=document.createElement('figcaption');cap.textContent=r.title;f.appendChild(cap);grid.appendChild(f);canvases.push(c);}}
const ticks=data.robots[0].truth[data.controllers[0].name].length,scrub=document.getElementById('scrub');scrub.max=ticks-1;
function pose(q,L){{const e=[L[0]*Math.sin(q[0]),-L[0]*Math.cos(q[0])];return [e,[e[0]+L[1]*Math.sin(q[0]+q[1]),e[1]-L[1]*Math.cos(q[0]+q[1])]];}}
function arm(ctx,ox,oy,s,q,L,color,w){{const [e,t]=pose(q,L);ctx.lineWidth=w;ctx.lineCap='round';ctx.lineJoin='round';ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+s*e[0],oy-s*e[1]);ctx.lineTo(ox+s*t[0],oy-s*t[1]);ctx.stroke();return [ox+s*t[0],oy-s*t[1]];}}
function draw(tick){{data.robots.forEach((r,i)=>{{const ctx=canvases[i].getContext('2d');ctx.clearRect(0,0,600,380);
data.controllers.forEach((c,k)=>{{const ox=150+k*300,oy=170,s=200;const goal=r.goal[c.name][tick],q=r.truth[c.name][tick];
arm(ctx,ox,oy,s,[goal[0]-r.bias[0],goal[1]-r.bias[1]],r.lengths,'#cfcfcf',8);const tip=arm(ctx,ox,oy,s,q,r.lengths,c.color,12);
ctx.fillStyle='#222';ctx.beginPath();ctx.arc(tip[0],tip[1],4,0,7);ctx.fill();
const err=Math.max(Math.abs(goal[0]-(q[0]+r.bias[0])),Math.abs(goal[1]-(q[1]+r.bias[1])));ctx.fillStyle=err<data.tolerance?'#148240':'#962828';ctx.font='13px system-ui';
ctx.fillText(`${{c.label.split(' (')[0]}}: ${{(1000*err).toFixed(0)}} mrad${{err<data.tolerance?' ✓':''}}`,ox-135,360);}});}});
document.getElementById('time').textContent=`t = ${{(tick/100).toFixed(2)}} s`;}}
let tick=0,playing=true,last=0;function loop(now){{if(playing&&now-last>20){{tick=(tick+2)%ticks;scrub.value=tick;draw(tick);last=now;}}requestAnimationFrame(loop);}}
scrub.oninput=()=>{{playing=false;document.getElementById('play').textContent='Play';tick=Number(scrub.value);draw(tick);}};
document.getElementById('play').onclick=function(){{playing=!playing;this.textContent=playing?'Pause':'Play';}};draw(0);requestAnimationFrame(loop);
</script></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def build(run: Path, gains: list[float], *, device: str, gif: Path | None, html_output: Path | None, worlds: int = 2048,
          pushes: bool = False, limbs: int = 0, severity: float = 1.0) -> list[tuple[int, str]]:
    torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
    baseline: object
    if limbs:
        from ..control.computed_torque import ComputedTorqueTeacher

        LABELS.update(CHAIN_LABELS)

        def baseline(env: ChainEnv) -> ComputedTorqueTeacher:
            return ComputedTorqueTeacher(env, omega=12.0, measured=True, ramp=False).to(torch_device)
    else:
        baseline = PidController(gains, torch_device)
    runs = {"pid": record_truth(baseline, worlds=worlds, device=device, pushes=pushes, limbs=limbs, severity=severity)[1:],
            "network": None}
    env, truth, goals = record_truth(load_actor(run, torch_device), worlds=worlds, device=device, pushes=pushes, limbs=limbs,
                                     severity=severity)
    runs["network"] = (truth, goals)
    picks = _pick_chain_robots(env) if isinstance(env, ChainEnv) else _pick_robots(env)
    if gif is not None:
        render_gif(runs, env, picks, gif)  # type: ignore[arg-type]
    if html_output is not None:
        render_html(runs, env, picks, html_output)  # type: ignore[arg-type]
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="student run directory")
    parser.add_argument("--report", type=Path, help="report.json holding the tuned PID gains (single arm only)")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--pushes", action="store_true")
    parser.add_argument("--limbs", type=int, default=0, help="animate a stacked chain of this many two-joint limbs")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    gains = json.loads(args.report.read_text())["pid_gains"] if args.report else []
    for index, title in build(args.run, gains, device=args.device, gif=args.gif, html_output=args.html, pushes=args.pushes,
                              limbs=args.limbs, severity=args.severity):
        print(f"robot {index}: {html.unescape(title)}")


if __name__ == "__main__":
    main()
