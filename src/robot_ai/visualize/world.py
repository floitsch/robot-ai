"""Offline visualization of held-out world-model endpoint predictions."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import confined_path


def export_world_trail_html(data: dict[str, np.ndarray | int | str], output: str | Path) -> Path:
    """Write a local, scrub-able actual/predicted endpoint trail view."""

    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in data.items()}, separators=(",", ":"))
    destination.write_text(f"""<!doctype html><meta charset="utf-8"><title>World-model endpoint trail</title>
<style>body{{font:14px sans-serif;background:#202124;color:#eee}}canvas{{background:#fff}}</style>
<h1>Held-out endpoint trail</h1><p id="meta"></p><canvas id="view" width="900" height="600"></canvas>
<p><label>Anchor <input id="scrub" type="range" min="0" value="0"></label></p><pre id="status"></pre>
<script>
const data={payload}, canvas=document.querySelector('#view'), ctx=canvas.getContext('2d'), scrub=document.querySelector('#scrub');
const all=data.actual_endpoint_xz.concat(data.predicted_endpoint_xz), xs=all.map(p=>p[0]), zs=all.map(p=>p[1]);
const lo=Math.min(...xs,...zs)-.05, hi=Math.max(...xs,...zs)+.05, scale=500/(hi-lo), ox=450, oz=300;
const xy=p=>[ox+p[0]*scale,oz-p[1]*scale], path=(points,color)=>{{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)}});ctx.stroke()}};
scrub.max=data.anchor_time_s.length-1; document.querySelector('#meta').textContent=`scenario=${{data.scenario_id}}, horizon=${{data.horizon_control_steps}} control steps`;
function draw(){{const i=Number(scrub.value);ctx.clearRect(0,0,900,600);path(data.actual_endpoint_xz,'#111');path(data.predicted_endpoint_xz,'#d33');for(const [p,c] of [[data.actual_endpoint_xz[i],'#111'],[data.predicted_endpoint_xz[i],'#d33']]){{ctx.fillStyle=c;ctx.beginPath();ctx.arc(...xy(p),5,0,Math.PI*2);ctx.fill()}}const e=Math.hypot(data.actual_endpoint_xz[i][0]-data.predicted_endpoint_xz[i][0],data.actual_endpoint_xz[i][1]-data.predicted_endpoint_xz[i][1]);document.querySelector('#status').textContent=`anchor t=${{data.anchor_time_s[i].toFixed(3)}} s, target t=${{data.target_time_s[i].toFixed(3)}} s, endpoint error=${{e.toFixed(4)}} m`;}}scrub.oninput=draw;draw();
</script>""", encoding="utf-8")
    return destination
