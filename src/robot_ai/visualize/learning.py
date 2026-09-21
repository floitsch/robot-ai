"""Standalone synchronized learning-progress replay export."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import confined_path

PIXELS_PER_METER = 230.0
PANEL_HEIGHT = 310


def _sha256(path: object) -> str | None:
    if not isinstance(path, str):
        return None
    candidate = confined_path(path)
    if not candidate.is_file():
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _panel(report: dict[str, Any]) -> dict[str, Any]:
    trajectory_path = confined_path(report["artifacts"]["trajectory"], must_exist=True)
    trajectory = np.load(trajectory_path, allow_pickle=False)
    return {
        "label": report["checkpoint"], "success_fraction": report["success_fraction"],
        "deadline_error_median_m": report.get("deadline_error_median_m"),
        "deadline_s": 1.5,
        "q": trajectory["q"].tolist(), "endpoint": trajectory["endpoint"].tolist(),
        "time_s": trajectory["time_s"].tolist(), "target": trajectory["target"].tolist(),
        "lengths": [float(trajectory["link_length_1"]), float(trajectory["link_length_2"])],
        "history": report.get("training_history", []),
        "provenance": {
            "trajectory": str(trajectory_path), "trajectory_sha256": _sha256(str(trajectory_path)),
            "checkpoint": report.get("checkpoint_source"),
            "checkpoint_sha256": _sha256(report.get("checkpoint_source")),
            "config": report.get("config_source"),
            "config_sha256": _sha256(report.get("config_source")),
        },
    }


def export_learning_html(reports: list[dict[str, Any]], output: str | Path) -> Path:
    """Embed recorded trajectories in a common-physical-time local Canvas viewer."""

    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panels = [_panel(report) for report in reports]
    rows = math.ceil(len(panels) / 2)
    view_height = rows * PANEL_HEIGHT
    payload = json.dumps(panels, separators=(",", ":"))
    destination.write_text(f"""<!doctype html><meta charset="utf-8"><title>Robot learning comparison</title>
<style>body{{font:14px sans-serif;background:#202124;color:#eee}}canvas{{background:#fff;display:block}}#view{{width:900px;height:{view_height}px}}label{{margin-right:1em}}</style>
<h1>Recorded learning comparison</h1><p>Black trail: actual endpoint. Blue arm: recorded joint state. Red circle: target. Every panel uses the same physical time. Deadline: 1.500 s.</p>
<canvas id="view" width="900" height="{view_height}"></canvas><canvas id="curve" width="900" height="160"></canvas><p><button id="play">Play</button><button id="reset">Reset</button><label>time <input id="scrub" type="range" min="0" step="0.001" value="0"></label><label>speed <select id="speed"><option>0.25</option><option selected>1</option><option>2</option></select></label></p><pre id="status"></pre><details><summary>Embedded provenance</summary><pre id="provenance"></pre></details>
<script>const panels={payload}, canvas=document.querySelector('#view'),ctx=canvas.getContext('2d'),scrub=document.querySelector('#scrub');const end=Math.max(...panels.map(p=>p.time_s.at(-1)));scrub.max=end;let playing=false,last=0;
function frame(p,t){{let i=0;while(i+1<p.time_s.length&&p.time_s[i+1]<=t)i++;return i}}function xy(p,ox,oz){{return[ox+p[0]*{PIXELS_PER_METER},oz-p[1]*{PIXELS_PER_METER}]}}function draw(t){{ctx.clearRect(0,0,900,{view_height});panels.forEach((p,n)=>{{const col=n%2,row=Math.floor(n/2),ox=col*450+225,oz=row*{PANEL_HEIGHT}+175,i=frame(p,t),q=p.q[i],l=p.lengths,p1=[l[0]*Math.sin(q[0]),-l[0]*Math.cos(q[0])],p2=[p1[0]+l[1]*Math.sin(q[0]+q[1]),p1[1]-l[1]*Math.cos(q[0]+q[1])];ctx.strokeStyle='#ddd';ctx.strokeRect(col*450,row*{PANEL_HEIGHT},450,{PANEL_HEIGHT});ctx.strokeStyle='#111';ctx.beginPath();p.endpoint.slice(0,i+1).forEach((v,j)=>{{const z=xy(v,ox,oz);j?ctx.lineTo(...z):ctx.moveTo(...z)}});ctx.stroke();const a=xy([0,0],ox,oz),b=xy(p1,ox,oz),c=xy(p2,ox,oz),target=xy(p.target,ox,oz);ctx.strokeStyle='#2477c9';ctx.lineWidth=12;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.lineTo(...c);ctx.stroke();ctx.strokeStyle='#d33';ctx.lineWidth=2;ctx.beginPath();ctx.arc(...target,6,0,Math.PI*2);ctx.stroke();const e=Math.hypot(p.endpoint[i][0]-p.target[0],p.endpoint[i][1]-p.target[1]);ctx.fillStyle='#111';ctx.font='13px sans-serif';ctx.fillText(`${{p.label}} success=${{(100*p.success_fraction).toFixed(1)}}% err=${{e.toFixed(3)}}m`,col*450+8,row*{PANEL_HEIGHT}+18)}});document.querySelector('#status').textContent=`physical time=${{t.toFixed(3)}} s; deadline=${{panels[0].deadline_s.toFixed(3)}} s`;}}
function tick(now){{if(!last)last=now;if(playing){{const t=Math.min(end,Number(scrub.value)+(now-last)/1000*Number(document.querySelector('#speed').value));scrub.value=t;if(t===end)playing=false;draw(t)}}last=now;requestAnimationFrame(tick)}}document.querySelector('#play').onclick=()=>{{playing=!playing;last=0}};document.querySelector('#reset').onclick=()=>{{playing=false;scrub.value=0;draw(0)}};scrub.oninput=()=>draw(Number(scrub.value));draw(0);document.querySelector('#status').textContent=`physical time=0.000 s; deadline=${{panels[0].deadline_s.toFixed(3)}} s`;requestAnimationFrame(tick);
document.querySelector('#provenance').textContent=JSON.stringify(panels.map(p=>({{label:p.label,...p.provenance}})),null,2);const history=(panels.find(p=>p.history.length)||{{history:[]}}).history,curve=document.querySelector('#curve').getContext('2d');if(history.length){{const losses=history.map(x=>x.loss||0),max=Math.max(...losses,1e-9);curve.fillStyle='#fff';curve.fillRect(0,0,900,160);curve.strokeStyle='#2477c9';curve.beginPath();losses.forEach((v,i)=>{{const x=40+i*820/(losses.length-1),y=145-v/max*120;i?curve.lineTo(x,y):curve.moveTo(x,y)}});curve.stroke();curve.fillStyle='#111';curve.fillText('fitted loss; vertical marks: initial, midpoint, final',40,15);[0,Math.floor((losses.length-1)/2),losses.length-1].forEach(i=>{{curve.strokeStyle='#d33';curve.beginPath();curve.moveTo(40+i*820/(losses.length-1),25);curve.lineTo(40+i*820/(losses.length-1),145);curve.stroke()}})}}
</script>""", encoding="utf-8")
    return destination


def export_learning_gif(reports: list[dict[str, Any]], output: str | Path) -> Path:
    """Export a composite comparison GIF from recorded physical-time samples."""

    panels = [_panel(report) for report in reports]
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = max(panel["time_s"][-1] for panel in panels)
    height = PANEL_HEIGHT * math.ceil(len(panels) / 2)
    # ``arange`` alone stops at 1.68 for the standard 1.70 s trace.  Append the
    # exact recorded hold end, and resolve samples by timestamp so accumulated
    # floating-point frame times still select the right physical sample.
    frame_times = np.arange(0.0, duration, 0.04).tolist()
    if not frame_times or not math.isclose(frame_times[-1], duration, abs_tol=1e-12):
        frame_times.append(duration)
    frames: list[Image.Image] = []
    for time_s in frame_times:
        image = Image.new("RGB", (900, height), "white")
        draw = ImageDraw.Draw(image)
        for number, panel in enumerate(panels):
            col, row = number % 2, number // 2
            ox, oz = col * 450 + 225, row * PANEL_HEIGHT + 165
            index = int(np.searchsorted(panel["time_s"], time_s, side="right") - 1)
            index = max(0, min(index, len(panel["time_s"]) - 1))
            q, lengths = panel["q"][index], panel["lengths"]
            p1 = (lengths[0] * np.sin(q[0]), -lengths[0] * np.cos(q[0]))
            p2 = (p1[0] + lengths[1] * np.sin(q[0] + q[1]), p1[1] - lengths[1] * np.cos(q[0] + q[1]))
            point = lambda p, ox=ox, oz=oz: (int(ox + p[0] * PIXELS_PER_METER), int(oz - p[1] * PIXELS_PER_METER))
            draw.rectangle((col * 450, row * PANEL_HEIGHT, col * 450 + 449,
                            row * PANEL_HEIGHT + PANEL_HEIGHT - 1), outline="#cccccc")
            draw.line([point(value) for value in panel["endpoint"][:index + 1]], fill="#222222", width=2)
            draw.line([point((0.0, 0.0)), point(p1), point(p2)], fill="#2477c9", width=10)
            x, y = point(panel["target"]); draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#cc3333", width=2)
            draw.text((col * 450 + 8, row * PANEL_HEIGHT + 8), str(panel["label"]), fill="#111111")
        frames.append(image)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=40, loop=0, disposal=2)
    return destination
