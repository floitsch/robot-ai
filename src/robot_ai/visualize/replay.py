"""Self-contained Canvas and GIF replay export for planar arm trajectories."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import confined_path


def _payload(data: dict[str, np.ndarray | float | str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def _point(q: list[float], lengths: list[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    q1, q2 = q
    p1 = (lengths[0] * np.sin(q1), -lengths[0] * np.cos(q1))
    p2 = (p1[0] + lengths[1] * np.sin(q1 + q2), p1[1] - lengths[1] * np.cos(q1 + q2))
    return p1, p2


def _pixel(point: tuple[float, float], ox: int = 450, oz: int = 430, scale: int = 700) -> tuple[int, int]:
    return int(ox + point[0] * scale), int(oz - point[1] * scale)


def export_html(data: dict[str, np.ndarray | float | str], output: str | Path) -> Path:
    """Export a replay that runs directly from a local file without a server."""

    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_payload(data), separators=(",", ":"))
    html = f"""<!doctype html>
<meta charset=\"utf-8\">
<title>Robot AI arm replay</title>
<style>body{{font:14px sans-serif;background:#202124;color:#eee}} canvas{{background:#fff;display:block;max-width:900px}} label{{margin-right:1em}}</style>
<canvas id=\"view\" width=900 height=600></canvas>
<p><button id=\"play\">Play</button> <button id=\"reset\">Reset</button>
<label>Time <input id=\"scrub\" type=\"range\" min=\"0\" max=\"1\" step=\"1\" value=\"0\"></label>
<label>Speed <select id=\"speed\"><option>0.25</option><option selected>1</option><option>2</option><option>4</option></select></label></p>
<pre id=\"status\"></pre>
<script>
const data={payload}; const canvas=document.getElementById('view'); const ctx=canvas.getContext('2d');
const scrub=document.getElementById('scrub'); const status=document.getElementById('status');
const scale=450, ox=450, oz=300, lengths=[0.30,0.25]; let frame=0, playing=false, last=0;
scrub.max=data.q.length-1;
function xy(p){{return [ox+p[0]*scale,oz-p[1]*scale]}}
function draw(){{const q=data.q[frame], p1=[lengths[0]*Math.sin(q[0]),-lengths[0]*Math.cos(q[0])];
const p2=[p1[0]+lengths[1]*Math.sin(q[0]+q[1]),p1[1]-lengths[1]*Math.cos(q[0]+q[1])];
ctx.clearRect(0,0,900,600); ctx.strokeStyle='#ddd'; ctx.beginPath(); ctx.moveTo(0,oz);ctx.lineTo(900,oz);ctx.moveTo(ox,0);ctx.lineTo(ox,600);ctx.stroke();
if(data.target){{const t=xy(data.target);ctx.strokeStyle='#d33';ctx.beginPath();ctx.arc(t[0],t[1],7,0,Math.PI*2);ctx.stroke();}}
const a=xy([0,0]), b=xy(p1), c=xy(p2);ctx.lineWidth=18;ctx.lineCap='round';ctx.strokeStyle='#3478c5';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.lineTo(...c);ctx.stroke();
ctx.fillStyle='#222';ctx.beginPath();ctx.arc(...c,6,0,Math.PI*2);ctx.fill(); scrub.value=frame;
const t=data.time_s[frame], e=data.target ? Math.hypot(p2[0]-data.target[0],p2[1]-data.target[1]):0;
status.textContent=`t=${{t.toFixed(3)}} s  frame=${{frame}}/${{data.q.length-1}}  endpoint error=${{e.toFixed(4)}} m`;
}}
function tick(now){{if(!last)last=now; if(playing){{const dt=(now-last)/1000*Number(document.getElementById('speed').value); let next=frame;
while(next+1<data.time_s.length && data.time_s[next+1]-data.time_s[frame] <= dt)next++; frame=next; if(frame===data.time_s.length-1)playing=false; draw();}} last=now; requestAnimationFrame(tick)}}
document.getElementById('play').onclick=()=>{{playing=!playing;last=0}}; document.getElementById('reset').onclick=()=>{{frame=0;playing=false;draw()}}; scrub.oninput=()=>{{frame=Number(scrub.value);draw()}}; draw(); requestAnimationFrame(tick);
</script>"""
    destination.write_text(html, encoding="utf-8")
    return destination


def export_gif(data: dict[str, np.ndarray | float | str], output: str | Path, *, stride: int = 2) -> Path:
    """Export a broadly viewable headless GIF using recorded physical samples."""

    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    q = np.asarray(data["q"], dtype=np.float64)
    lengths = [float(data.get("link_length_1", 0.30)), float(data.get("link_length_2", 0.25))]
    target = np.asarray(data.get("target", [0.0, 0.0]), dtype=np.float64)
    frames: list[Image.Image] = []
    for angles in q[::max(1, stride)]:
        p1, p2 = _point(angles.tolist(), lengths)
        image = Image.new("RGB", (900, 600), "white")
        draw = ImageDraw.Draw(image)
        ox, oz, scale = 450, 300, 450
        draw.line((0, oz, 900, oz), fill="#dddddd")
        draw.line((ox, 0, ox, 600), fill="#dddddd")
        tx, tz = _pixel((float(target[0]), float(target[1])), ox, oz, scale)
        draw.ellipse((tx - 7, tz - 7, tx + 7, tz + 7), outline="#cc3333", width=2)
        a, b, c = _pixel((0.0, 0.0), ox, oz, scale), _pixel(p1, ox, oz, scale), _pixel(p2, ox, oz, scale)
        draw.line((a, b, c), fill="#3478c5", width=18, joint="curve")
        draw.ellipse((c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6), fill="#222222")
        frames.append(image)
    if not frames:
        raise ValueError("trajectory contains no frames")
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=20, loop=0, disposal=2)
    return destination
