"""Self-contained HTML comparison of reach controllers on imperfect arms."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from ..sim.population import NOMINAL_TORQUES
from ..sim.reach_env import ReachEnv
from ..train.reach import EVAL_SEED, Actor, PidController, load_actor

# Validated categorical slots (light, dark); identity is also carried by direct labels.
SERIES = {"pid": ("#2a78d6", "#3987e5", "Tuned PID"), "memoryless": ("#eb6834", "#d95926", "Memoryless network"),
          "recurrent": ("#1baf7a", "#199e70", "Recurrent network")}


@torch.no_grad()
def record(controller: PidController | Actor, *, worlds: int, device: str) -> tuple[ReachEnv, np.ndarray, np.ndarray]:
    """Measured joint angles and goals [ticks, worlds, 2] on the held-out robots."""

    env = ReachEnv(worlds, device=device, seed=EVAL_SEED)
    observation = env.reset()
    feeling = controller.initial(worlds, env.torch_device) if isinstance(controller, torch.nn.Module) else None
    measured, goals = [], []
    for _ in range(env.episode_ticks):
        if isinstance(controller, torch.nn.Module):
            mean, feeling = controller(observation[None], feeling)
            action = env.integrate(mean[0]) if controller.incremental else mean[0]
        else:
            action = controller.act(env)
        goals.append(env.goal().cpu().numpy())
        observation, _ = env.step(action)
        measured.append(env.measured_q.cpu().numpy())
    return env, np.stack(measured), np.stack(goals)


def _pick_robots(env: ReachEnv) -> list[tuple[int, str]]:
    joints, arms = env.joints, env.arms
    dry = joints["coulomb"] * (1.0 + joints["stribeck"]) + joints["bump0_mag"] + joints["bump1_mag"]
    weakness = 1.0 - joints["torque_scale"] / NOMINAL_TORQUES
    lag_ms = -1.0 / np.log(np.clip(1.0 - joints["motor_alpha"], 1e-9, 1.0 - 1e-9))  # motor time constant at 1 ms steps
    lag_ms = np.where(joints["motor_alpha"] >= 1.0, 0.0, lag_ms)
    score = (dry.sum(1) * 2 + joints["half_gap"].sum(1) * 40 + weakness.sum(1) * 2 + joints["delay_steps"].sum(1) / 30
             + joints["enc_delay_ticks"].sum(1) / 3 + lag_ms.sum(1) / 40 + joints["enc_quantum"].sum(1) * 80)
    changing = np.where(env.change_tick < env.episode_ticks, env.changed[0]["m2"] - arms["m2"], -1.0)

    def describe(index: int) -> str:
        parts = [f"dry friction {dry[index, 0]:.2f}/{dry[index, 1]:.2f} N m",
                 f"backlash {2000 * joints['half_gap'][index, 0]:.0f}/{2000 * joints['half_gap'][index, 1]:.0f} mrad",
                 f"motors at {100 - 100 * weakness[index, 0]:.0f}/{100 - 100 * weakness[index, 1]:.0f}%",
                 f"command latency {joints['delay_steps'][index, 0]}/{joints['delay_steps'][index, 1]} ms",
                 f"motor lag {lag_ms[index, 0]:.0f}/{lag_ms[index, 1]:.0f} ms",
                 f"encoder delay {10 * joints['enc_delay_ticks'][index, 0]}/{10 * joints['enc_delay_ticks'][index, 1]} ms"]
        if changing[index] > 0:
            parts.append(f"picks up {1000 * changing[index]:.0f} g at {env.change_tick[index] / 100:.2f} s")
        return "; ".join(parts)

    picks = {"Nearly healthy": int(np.argmin(score)), "Sticky joints": int(np.argmax(dry.sum(1))),
             "Slack gears": int(np.argmax(joints["half_gap"].sum(1))), "Weak motors": int(np.argmax(weakness.sum(1))),
             "Payload picked up mid-move": int(np.argmax(changing)), "Everything at once": int(np.argmax(score))}
    return [(index, f"{title} — {describe(index)}") for title, index in picks.items()]


def _polyline(values: np.ndarray, low: float, high: float, width: int, height: int) -> str:
    x = np.linspace(0, width, len(values))
    y = height - (values - low) / (high - low) * height
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y, strict=True))


def _panel(title: str, traces: dict[str, np.ndarray], goal: np.ndarray, seconds: float) -> str:
    width, height = 420, 120
    stacked = np.concatenate([goal, *traces.values()])
    low, high = float(stacked.min()) - 0.1, float(stacked.max()) + 0.1
    lines = [f'<polyline class="goal" points="{_polyline(goal, low, high, width, height)}"/>']
    for name, values in traces.items():
        lines.append(f'<polyline class="s-{name}" points="{_polyline(values, low, high, width, height)}"/>')
    data = html.escape(json.dumps({"goal": goal.round(3).tolist(), **{k: v.round(3).tolist() for k, v in traces.items()}}))
    return (f'<figure class="panel" data-series="{data}" data-seconds="{seconds}"><figcaption>{html.escape(title)}</figcaption>'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" aria-label="{html.escape(title)}">'
            f'{"".join(lines)}<line class="cursor" y1="0" y2="{height}" x1="-9" x2="-9"/></svg>'
            f'<div class="axis"><span>0 s</span><span class="readout"></span><span>{seconds:.0f} s</span></div></figure>')


def _curves(runs: dict[str, Path], pid_success: float) -> str:
    width, height = 860, 220
    series: dict[str, list[tuple[float, float]]] = {}
    for name, run in runs.items():
        rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        series[name] = [(row["robot_ticks"] / 1e6, row["eval"]["success"]) for row in rows if "eval" in row]
    longest = max(points[-1][0] for points in series.values())
    top = max(0.1, np.ceil(10 * 1.15 * max(pid_success, *(y for points in series.values() for _, y in points))) / 10)

    def level(value: float) -> float:
        return height - value / top * height

    body = []
    for tick in np.arange(0.0, top + 1e-9, 0.1):
        body.append(f'<line class="rule" x1="0" x2="{width}" y1="{level(tick):.1f}" y2="{level(tick):.1f}"/>'
                    f'<text class="label" x="2" y="{level(tick) - 3:.1f}">{100 * tick:.0f}%</text>')
    body.append(f'<line class="goal" x1="0" x2="{width}" y1="{level(pid_success):.1f}" y2="{level(pid_success):.1f}"/>')
    ends = [(level(pid_success), f"tuned PID {100 * pid_success:.0f}%")]
    for name, points in series.items():
        kind = "recurrent" if name.startswith("recurrent") else "memoryless"
        path = " ".join(f"{x / longest * width:.1f},{level(y):.1f}" for x, y in points)
        body.append(f'<polyline class="s-{kind}" points="{path}"/>')
        ends.append((level(points[-1][1]), f"{name} {100 * points[-1][1]:.0f}%"))
    placed = -100.0
    for y, text in sorted(ends):  # direct labels, nudged apart so they never collide
        placed = max(y - 6, placed + 15)
        body.append(f'<text class="label strong" x="{width - 4}" y="{placed:.1f}" text-anchor="end">{html.escape(text)}</text>')
    return (f'<svg class="curves" viewBox="0 0 {width} {height}" role="img" aria-label="Held-out success during training">{"".join(body)}</svg>'
            f'<div class="axis"><span>0</span><span>millions of simulated control ticks</span><span>{longest:.0f} M</span></div>')


def _table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    rows = []
    for condition, row in results.items():
        for name, values in row.items():
            rows.append(f"<tr><td>{html.escape(condition)}</td><td>{html.escape(name)}</td><td>{100 * values['success']:.1f}%</td>"
                        f"<td>{values['final_error_mrad']:.1f}</td><td>{values['mean_error_rad']:.3f}</td><td>{values['roughness']:.4f}</td></tr>")
    return ("<table><thead><tr><th>Robots</th><th>Controller</th><th>Success</th><th>Median final error (mrad)</th>"
            "<th>Mean error over episode (rad)</th><th>Command roughness</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


_STYLE = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#d9d8d2;--pid:#2a78d6;--memoryless:#eb6834;--recurrent:#1baf7a}
@media(prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#3a3a37;--pid:#3987e5;--memoryless:#d95926;--recurrent:#199e70}}
body{background:var(--surface);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}p,figcaption,.axis,td,th{color:var(--ink2)}
svg{width:100%;display:block;border-bottom:1px solid var(--grid)}svg.curves{height:220px}.panel svg{height:120px}
polyline{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}.goal{stroke:var(--ink2);stroke-dasharray:4 4;stroke-width:1;fill:none}
.s-pid{stroke:var(--pid)}.s-memoryless{stroke:var(--memoryless)}.s-recurrent{stroke:var(--recurrent)}
.cursor{stroke:var(--ink2);stroke-width:1;vector-effect:non-scaling-stroke}.label{fill:var(--ink2);font-size:11px}.strong{fill:var(--ink);font-size:12px;paint-order:stroke;stroke:var(--surface);stroke-width:4px}.rule{stroke:var(--grid);stroke-width:1;vector-effect:non-scaling-stroke}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}figure{margin:0}figcaption{font-size:13px;margin-bottom:4px}
.axis{display:flex;justify-content:space-between;font-size:12px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:8px 0}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--grid);padding:4px 6px;text-align:right}
td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}.wrap{overflow-x:auto}
"""

_SCRIPT = """
for (const panel of document.querySelectorAll('.panel')) {
  const series = JSON.parse(panel.dataset.series), seconds = +panel.dataset.seconds;
  const svg = panel.querySelector('svg'), cursor = panel.querySelector('.cursor'), readout = panel.querySelector('.readout');
  svg.addEventListener('pointermove', (event) => {
    const box = svg.getBoundingClientRect(), part = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const index = Math.round(part * (series.goal.length - 1)), x = part * svg.viewBox.baseVal.width;
    cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
    readout.textContent = (part * seconds).toFixed(2) + ' s · ' + Object.entries(series).map(([k, v]) => k + ' ' + v[index].toFixed(2)).join(' · ') + ' rad';
  });
  svg.addEventListener('pointerleave', () => { cursor.setAttribute('x1', -9); cursor.setAttribute('x2', -9); readout.textContent = ''; });
}
"""


def build(report: Path, runs: dict[str, Path], output: Path, *, device: str, worlds: int = 4096) -> None:
    results = json.loads(report.read_text())
    torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
    controllers: dict[str, PidController | Actor] = {"pid": PidController(results["pid_gains"], torch_device)}
    for run in runs.values():
        actor = load_actor(run, torch_device)
        controllers.setdefault("recurrent" if actor.recurrent else "memoryless", actor)
    recorded = {name: record(controller, worlds=worlds, device=device) for name, controller in controllers.items()}
    env, _, goals = recorded["pid"]
    panels = []
    for index, title in _pick_robots(env):
        for joint in range(2):
            traces = {name: item[1][:, index, joint] for name, item in recorded.items()}
            panels.append(_panel(f"{title} · joint {joint + 1}", traces, goals[:, index, joint], env.episode_ticks / 100))
    conditions = {key: value for key, value in results.items() if isinstance(value, dict)}
    legend = "".join(f'<span><i style="background:var(--{name})"></i>{SERIES[name][2]}</span>' for name in SERIES if name in recorded)
    page = (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Reach controllers</title><style>{_STYLE}</style><h1>One set of weights, thousands of different broken arms</h1>"
            f"<p>{worlds} held-out robots the networks never trained on. No controller is told anything about the robot it drives.</p>"
            f"<h2>Held-out success while training (defective and changing robots)</h2>{_curves(runs, conditions['defective+changing']['pid']['success'])}"
            f"<h2>Results</h2><div class=wrap>{_table(conditions)}</div>"
            f"<h2>Same robot, same goals, three controllers</h2><div class=legend>{legend}<span><i style='background:var(--ink2)'></i>Goal (dashed)</span></div>"
            f"<div class=grid>{''.join(panels)}</div><script>{_SCRIPT}</script></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    build(args.report, {name: Path(path) for name, path in (item.split("=", 1) for item in args.run)}, args.output, device=args.device)


if __name__ == "__main__":
    main()
