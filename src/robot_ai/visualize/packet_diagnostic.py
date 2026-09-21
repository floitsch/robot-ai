"""Self-contained, mixed-clock C3a packet-feedback diagnostic playback."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import confined_path

DEADLINE_S = 1.5
HOLD_END_S = 1.7
_STATE_FIELDS = ("q", "dq", "endpoint", "endpoint_velocity")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_array(value: np.ndarray, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"packet diagnostic {name} must be finite with shape {shape}, got {array.shape}")
    return array


def _report(trace_path: Path) -> tuple[Path, dict[str, Any]]:
    path = trace_path.parent / "metrics.json"
    if not path.is_file():
        raise ValueError(f"packet diagnostic trace {trace_path} has no companion metrics.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        physics_dt, control_dt = float(report["physics_dt"]), float(report["control_dt"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"packet diagnostic report {path} lacks clock metadata") from error
    if not all(math.isfinite(value) and value > 0.0 for value in (physics_dt, control_dt)):
        raise ValueError(f"packet diagnostic report {path} has invalid clock metadata")
    return path, report


def _optional_state(loaded: Any, field: str, count: int) -> list[list[float]] | None:
    if field not in loaded.files:
        return None
    value = np.asarray(loaded[field])
    if value.size == 0:
        return None
    return _finite_array(value, field, (count, 2)).tolist()


def _public_state(loaded: Any, count: int) -> tuple[list[list[float | None]] | None,
                                                      list[list[float | None]] | None]:
    """Extract public q/dq samples without replacing a missing packet by truth."""

    present = {"packet_values", "packet_available"}.intersection(loaded.files)
    if not present:
        return None, None
    if len(present) != 2:
        raise ValueError("packet diagnostic requires packet_values and packet_available together")
    values = _finite_array(loaded["packet_values"], "packet_values", (count, 8))
    available = np.asarray(loaded["packet_available"], dtype=bool)
    if available.shape != (count, 8):
        raise ValueError(f"packet diagnostic packet_available must have shape {(count, 8)}")

    def channels(start: int) -> list[list[float | None]]:
        return [[float(value) if is_available else None
                 for value, is_available in zip(row[start:start + 2], mask[start:start + 2], strict=True)]
                for row, mask in zip(values, available, strict=True)]

    return channels(0), channels(2)


def _trace_payload(path: str | Path) -> dict[str, Any]:
    """Load only traces whose N+1 state and N command clocks match their report."""

    trace_path = confined_path(path, must_exist=True)
    report_path, report = _report(trace_path)
    with np.load(trace_path, allow_pickle=False) as loaded:
        required = {"time_s", "target", "accepted_action", "link_length_1", "link_length_2", *_STATE_FIELDS}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"packet diagnostic trace {trace_path} misses {sorted(missing)}")
        time_s = np.asarray(loaded["time_s"], dtype=np.float64)
        physics_dt, control_dt = float(report["physics_dt"]), float(report["control_dt"])
        if time_s.ndim != 1 or len(time_s) < 2 or not np.all(np.isfinite(time_s)) or not np.isclose(time_s[0], 0.0):
            raise ValueError(f"packet diagnostic trace {trace_path} has an invalid physical time vector")
        if not np.allclose(np.diff(time_s), physics_dt, rtol=0.0, atol=physics_dt * 1e-8):
            raise ValueError(f"packet diagnostic trace {trace_path} disagrees with metrics physics_dt")
        substeps = round(control_dt / physics_dt)
        if substeps < 1 or not np.isclose(substeps * physics_dt, control_dt, rtol=0.0, atol=physics_dt * 1e-8):
            raise ValueError(f"packet diagnostic report {report_path} has non-integral clocks")
        if (len(time_s) - 1) % substeps:
            raise ValueError(f"packet diagnostic trace {trace_path} is not an N+1 control trace")
        controls = (len(time_s) - 1) // substeps
        if not np.isclose(time_s[-1], controls * control_dt, rtol=0.0, atol=physics_dt * 1e-8):
            raise ValueError(f"packet diagnostic trace {trace_path} duration disagrees with control_dt")
        physical = {field: _finite_array(loaded[field], field, (len(time_s), 2)).tolist()
                    for field in _STATE_FIELDS}
        accepted = _finite_array(loaded["accepted_action"], "accepted_action", (controls, 2)).tolist()
        teacher = (_finite_array(loaded["truth_teacher_action"], "truth_teacher_action", (controls, 2)).tolist()
                   if "truth_teacher_action" in loaded.files else None)
        measured_q, measured_dq = _public_state(loaded, controls + 1)
        estimated_q, estimated_dq = _optional_state(loaded, "estimated_q", controls + 1), _optional_state(loaded, "estimated_dq", controls + 1)
        if (estimated_q is None) != (estimated_dq is None):
            raise ValueError("packet diagnostic estimated q/dq must be recorded together")
        for field in ("packet_fresh", "packet_age_s"):
            if field in loaded.files:
                _finite_array(loaded[field], field, (controls + 1, 8))
        lengths = [float(loaded["link_length_1"]), float(loaded["link_length_2"])]
        if not all(math.isfinite(value) and value > 0.0 for value in lengths):
            raise ValueError("packet diagnostic link lengths must be positive and finite")
        target = _finite_array(loaded["target"], "target", (2,)).tolist()
    return {
        "path": str(trace_path), "sha256": _sha256(trace_path),
        "metrics_path": str(report_path), "metrics_sha256": _sha256(report_path),
        "physics_dt_s": physics_dt, "control_dt_s": control_dt,
        "physical_time_s": time_s.tolist(),
        "control_time_s": (np.arange(controls + 1, dtype=np.float64) * control_dt).tolist(),
        **physical, "accepted_action": accepted, "truth_teacher_action": teacher,
        "measured_q": measured_q, "measured_dq": measured_dq,
        "estimated_q": estimated_q, "estimated_dq": estimated_dq,
        "target": target, "link_lengths": lengths,
        "deadline_s": DEADLINE_S, "hold_end_s": HOLD_END_S,
    }


def _payload(traces: Mapping[int, Mapping[str, str | Path]], selection: str | Path,
             speed_limit_m_s: float) -> dict[str, Any]:
    if not traces:
        raise ValueError("packet diagnostic viewer requires at least one case")
    if speed_limit_m_s <= 0.0 or not math.isfinite(speed_limit_m_s):
        raise ValueError("speed_limit_m_s must be positive and finite")
    selection_path = confined_path(selection, must_exist=True)
    report = json.loads(selection_path.read_text(encoding="utf-8"))
    reports = report.get("selection", report).get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("selection report has no checkpoint reports")
    curve = [{"checkpoint": Path(str(item["checkpoint"])).name, "success_fraction": float(item["success_fraction"])}
             for item in reports]
    cases = {str(case): {label: _trace_payload(path) for label, path in labelled.items()}
             for case, labelled in sorted(traces.items())}
    if any("selected" not in labelled for labelled in cases.values()):
        raise ValueError("each packet diagnostic case needs a selected trace")
    return {"speed_limit_m_s": speed_limit_m_s, "cases": cases, "selection": curve,
            "selection_provenance": {"path": str(selection_path), "sha256": _sha256(selection_path),
                                      "label": "historical 64-case selection; not renewed C1 evidence"}}


def _html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    document = f"""<!doctype html><meta charset="utf-8"><title>Packet diagnostic replay</title>
<style>body{{font:14px sans-serif;background:#202124;color:#eee;max-width:1240px}}canvas{{display:block;background:#fff;margin:8px 0}}label{{margin-right:1em}}#status{{white-space:pre-wrap}}</style>
<h1>Packet diagnostic replay</h1><p>True q/dq and speed use recorded 1 ms samples. Measured packets, posterior estimates, accepted policy commands, and same-state teacher commands use recorded 10 ms samples. The teacher is diagnostic only. Missing channels remain unavailable. Dashed markers are the 1.500 s deadline and 1.700 s hold end.</p>
<p><label>Case <select id="case"></select></label><button id="play">Play</button><button id="reset">Reset</button><label>time <input id="scrub" type="range" min=0 step=.001 value=0></label><label>speed <select id="rate"><option>.25</option><option selected>1</option><option>2</option></select></label></p>
<canvas id="arms" width="1200" height="560"></canvas><canvas id="signals" width="1200" height="1280"></canvas><canvas id="selection" width="1200" height="220"></canvas><pre id="status"></pre><details><summary>Embedded provenance and original samples</summary><pre id="provenance"></pre></details>
<script>const data={data},pick=document.querySelector('#case'),scrub=document.querySelector('#scrub'),arms=document.querySelector('#arms').getContext('2d'),signals=document.querySelector('#signals').getContext('2d'),selection=document.querySelector('#selection').getContext('2d');Object.keys(data.cases).forEach(k=>pick.add(new Option(k,k)));let playing=false,last=0;
function at(ts,t){{let i=0;while(i+1<ts.length&&ts[i+1]<=t+1e-12)i++;return i}}function point(q,l){{const a=[l[0]*Math.sin(q[0]),-l[0]*Math.cos(q[0])];return[a,[a[0]+l[1]*Math.sin(q[0]+q[1]),a[1]-l[1]*Math.cos(q[0]+q[1])]]}}function xy(p,x,y){{return[x+p[0]*300,y-p[1]*300]}}function domain(lines,extra){{const a=lines.flat().filter(Number.isFinite).concat(extra||[],[0]);let lo=Math.min(...a),hi=Math.max(...a);if(lo===hi){{lo-=1;hi+=1}}const pad=(hi-lo)*.08;return[lo-pad,hi+pad]}}function line(c,ts,v,color,x,y,w,h,end,lo,hi){{if(!v)return;c.strokeStyle=color;c.lineWidth=1.4;c.beginPath();let open=false;v.forEach((z,i)=>{{if(!Number.isFinite(z)){{open=false;return}}const px=x+ts[i]/end*w,py=y+h-(z-lo)/(hi-lo)*h;open?c.lineTo(px,py):c.moveTo(px,py);open=true}});c.stroke()}}function plot(title,unit,series,extra,n,t,p,end){{const x=78,y=30+n*174,w=1080,h=88,lines=series.map(s=>s.values||[]),[lo,hi]=domain(lines,extra);signals.fillStyle='#111';signals.font='12px sans-serif';signals.fillText(`${{title}} (${{unit}})`,x,y-7);signals.strokeStyle='#bbb';signals.strokeRect(x,y,w,h);const zero=y+h-(0-lo)/(hi-lo)*h;signals.strokeStyle='#ddd';signals.beginPath();signals.moveTo(x,zero);signals.lineTo(x+w,zero);signals.stroke();signals.fillStyle='#111';signals.fillText(hi.toFixed(3),x+3,y+11);signals.fillText(lo.toFixed(3),x+3,y+h-3);[p.deadline_s,p.hold_end_s].forEach(mark=>{{if(mark<=end){{signals.setLineDash([4,3]);signals.strokeStyle='#777';signals.beginPath();signals.moveTo(x+mark/end*w,y);signals.lineTo(x+mark/end*w,y+h);signals.stroke();signals.setLineDash([])}}}});series.forEach(s=>line(signals,s.times,s.values,s.color,x,y,w,h,end,lo,hi));if(extra)extra.forEach(v=>{{const py=y+h-(v-lo)/(hi-lo)*h;signals.strokeStyle='#d33';signals.beginPath();signals.moveTo(x,py);signals.lineTo(x+w,py);signals.stroke()}});signals.strokeStyle='#d33';signals.beginPath();signals.moveTo(x+t/end*w,y);signals.lineTo(x+t/end*w,y+h);signals.stroke();if(series.slice(1).some(s=>!s.values)){{signals.fillStyle='#666';signals.fillText('unavailable',x+w-74,y+13)}}}}
function draw(t){{const items=Object.entries(data.cases[pick.value]),end=Math.max(...items.map(([,p])=>p.physical_time_s.at(-1)));scrub.max=end;arms.clearRect(0,0,1200,560);items.forEach(([label,p],n)=>{{const col=n%4,row=Math.floor(n/4),x=col*300,y=row*280,i=at(p.physical_time_s,t),[a,b]=point(p.q[i],p.link_lengths),origin=[x+150,y+155],target=xy(p.target,...origin),pa=xy([0,0],...origin),pb=xy(a,...origin),pc=xy(b,...origin);arms.strokeStyle='#bbb';arms.strokeRect(x,y,299,279);arms.strokeStyle='#111';arms.beginPath();p.endpoint.slice(0,i+1).forEach((v,j)=>{{const z=xy(v,...origin);j?arms.lineTo(...z):arms.moveTo(...z)}});arms.stroke();arms.strokeStyle='#2477c9';arms.lineWidth=9;arms.beginPath();arms.moveTo(...pa);arms.lineTo(...pb);arms.lineTo(...pc);arms.stroke();arms.strokeStyle='#d33';arms.lineWidth=2;arms.beginPath();arms.arc(...target,5,0,Math.PI*2);arms.stroke();const speed=Math.hypot(...p.endpoint_velocity[i]);arms.fillStyle='#111';arms.font='12px sans-serif';arms.fillText(`${{label}} t=${{p.physical_time_s[i].toFixed(3)}}`,x+7,y+16);arms.fillText(`speed=${{speed.toFixed(3)}} m/s`,x+7,y+33)}});const p=items.find(([label])=>label==='selected')[1],pi=at(p.physical_time_s,t),ci=at(p.control_time_s,t),speed=p.endpoint_velocity.map(v=>Math.hypot(...v)),pt=p.physical_time_s,ct=p.control_time_s;signals.clearRect(0,0,1200,1280);const specs=[['endpoint speed','m/s',[{{times:pt,values:speed,color:'#111'}}],[data.speed_limit_m_s]],['q1 true/measured/estimated','rad',[{{times:pt,values:p.q.map(v=>v[0]),color:'#111'}},{{times:ct,values:p.measured_q?.map(v=>v[0]),color:'#188038'}},{{times:ct,values:p.estimated_q?.map(v=>v[0]),color:'#2477c9'}}]],null],['q2 true/measured/estimated','rad',[{{times:pt,values:p.q.map(v=>v[1]),color:'#111'}},{{times:ct,values:p.measured_q?.map(v=>v[1]),color:'#188038'}},{{times:ct,values:p.estimated_q?.map(v=>v[1]),color:'#2477c9'}}]],null],['dq1 true/measured/estimated','rad/s',[{{times:pt,values:p.dq.map(v=>v[0]),color:'#111'}},{{times:ct,values:p.measured_dq?.map(v=>v[0]),color:'#188038'}},{{times:ct,values:p.estimated_dq?.map(v=>v[0]),color:'#2477c9'}}]],null],['dq2 true/measured/estimated','rad/s',[{{times:pt,values:p.dq.map(v=>v[1]),color:'#111'}},{{times:ct,values:p.measured_dq?.map(v=>v[1]),color:'#188038'}},{{times:ct,values:p.estimated_dq?.map(v=>v[1]),color:'#2477c9'}}]],null],['command 1 accepted/teacher','command',[{{times:ct.slice(0,-1),values:p.accepted_action.map(v=>v[0]),color:'#111'}},{{times:ct.slice(0,-1),values:p.truth_teacher_action?.map(v=>v[0]),color:'#d77a00'}}],null],['command 2 accepted/teacher','command',[{{times:ct.slice(0,-1),values:p.accepted_action.map(v=>v[1]),color:'#111'}},{{times:ct.slice(0,-1),values:p.truth_teacher_action?.map(v=>v[1]),color:'#d77a00'}}],null]];specs.forEach((s,n)=>plot(s[0],s[1],s[2],s[3],n,t,p,end));selection.clearRect(0,0,1200,220);selection.fillStyle='#111';selection.font='13px sans-serif';selection.fillText('Historical 64-case packet selection success fraction (not renewed C1 evidence)',78,20);const curve=data.selection,sy=190;selection.strokeStyle='#2477c9';selection.lineWidth=2;selection.beginPath();curve.forEach((v,i)=>{{const px=78+i*1080/Math.max(1,curve.length-1),py=sy-v.success_fraction*150;i?selection.lineTo(px,py):selection.moveTo(px,py);selection.fillStyle='#111';selection.fillText(v.checkpoint.replace('policy-update-','u'),px-19,211)}});selection.stroke();curve.forEach((v,i)=>{{const px=78+i*1080/Math.max(1,curve.length-1),py=sy-v.success_fraction*150;selection.fillStyle='#2477c9';selection.beginPath();selection.arc(px,py,3,0,Math.PI*2);selection.fill()}});const fmt=a=>a?.map(v=>Number.isFinite(v)?v.toFixed(6):'unavailable').join(', ')||'unavailable';document.querySelector('#status').textContent=`case=${{pick.value}}, requested t=${{t.toFixed(3)}} s\nphysics index=${{pi}} @ ${{pt[pi].toFixed(3)}} s; control index=${{ci}} @ ${{ct[ci].toFixed(3)}} s\ntrue q=${{fmt(p.q[pi])}} rad; true dq=${{fmt(p.dq[pi])}} rad/s\nmeasured q=${{fmt(p.measured_q?.[ci])}}; estimated q=${{fmt(p.estimated_q?.[ci])}}`;}}
function tick(now){{if(!last)last=now;if(playing){{const end=Math.max(...Object.values(data.cases[pick.value]).map(p=>p.physical_time_s.at(-1))),t=Math.min(end,Number(scrub.value)+(now-last)/1000*Number(document.querySelector('#rate').value));scrub.value=t;if(t===end)playing=false;draw(t)}}last=now;requestAnimationFrame(tick)}}pick.onchange=()=>{{scrub.value=0;draw(0)}};scrub.oninput=()=>draw(Number(scrub.value));document.querySelector('#play').onclick=()=>{{playing=!playing;last=0}};document.querySelector('#reset').onclick=()=>{{playing=false;scrub.value=0;draw(0)}};document.querySelector('#provenance').textContent=JSON.stringify(data,null,2);draw(0);requestAnimationFrame(tick);</script>\n"""
    return (document.replace("}]],null]", "}],null]")
            .replace("return[x+p[0]*300,y-p[1]*300]", "return[x+p[0]*170,y-p[1]*170]")
            .replace("origin=[x+150,y+155]", "origin=[x+150,y+125]"))


def export_packet_diagnostic_html(traces: Mapping[int, Mapping[str, str | Path]], selection: str | Path,
                                  output: str | Path, *, speed_limit_m_s: float = 0.030) -> Path:
    """Export a local-file packet viewer that retains each recorded sample clock."""

    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_html(_payload(traces, selection, speed_limit_m_s)), encoding="utf-8")
    return destination


def export_packet_diagnostic_report(traces: Mapping[int, Mapping[str, str | Path]], selection: str | Path,
                                    output: str | Path, *, reproduction_command: str,
                                    html: str | Path, gif: str | Path | None = None,
                                    speed_limit_m_s: float = 0.030) -> Path:
    """Record immutable input/output identities for a replay export."""

    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selection_path = confined_path(selection, must_exist=True)
    report = {"purpose": "C3a saved packet diagnostic replay", "reproduction_command": reproduction_command,
              "speed_limit_m_s": speed_limit_m_s,
              "inputs": {"selection": {"path": str(selection_path), "sha256": _sha256(selection_path)},
                         "traces": {str(case): {label: {"path": str(confined_path(path, must_exist=True)),
                                                        "sha256": _sha256(confined_path(path, must_exist=True))}
                                                  for label, path in labelled.items()}
                                    for case, labelled in sorted(traces.items())}},
              "outputs": {"html": {"path": str(confined_path(html, must_exist=True)),
                                   "sha256": _sha256(confined_path(html, must_exist=True))},
                          "gif": None if gif is None else {"path": str(confined_path(gif, must_exist=True)),
                                                              "sha256": _sha256(confined_path(gif, must_exist=True))}}}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def _sample_index(time_s: list[float] | np.ndarray, time: float) -> int:
    """Return the latest recorded sample at or before a possibly accumulated time."""

    samples = np.asarray(time_s, dtype=np.float64)
    return int(np.clip(np.searchsorted(samples, time, side="right") - 1, 0, len(samples) - 1))


def _plot(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], series: list[tuple[list[float], list[float | None] | None, str]],
          end: float, now: float, title: str, unit: str, extra: list[float] | None = None) -> None:
    left, top, right, bottom = bounds
    values = [float(value) for _, line, _ in series if line for value in line if value is not None and math.isfinite(value)] + (extra or [])
    low, high = min(values + [0.0]), max(values + [0.0])
    if low == high:
        low, high = low - 1.0, high + 1.0
    padding = (high - low) * 0.08
    low, high = low - padding, high + padding
    draw.rectangle(bounds, outline="#aaaaaa")
    draw.text((left, top - 14), f"{title} ({unit}) [{low:.3f}, {high:.3f}]", fill="#111111")
    draw.line((left, bottom - (0.0 - low) / (high - low) * (bottom - top), right,
               bottom - (0.0 - low) / (high - low) * (bottom - top)), fill="#dddddd")
    for marker in (DEADLINE_S, HOLD_END_S):
        if marker <= end:
            x = left + marker / end * (right - left)
            draw.line((x, top, x, bottom), fill="#777777")
    for times, line, color in series:
        if not line:
            continue
        previous: tuple[float, float] | None = None
        for sample_time, value in zip(times, line, strict=True):
            if value is None or not math.isfinite(value):
                previous = None
                continue
            point = (left + sample_time / end * (right - left),
                     bottom - (value - low) / (high - low) * (bottom - top))
            if previous:
                draw.line((previous, point), fill=color, width=2)
            previous = point
    for value in extra or []:
        y = bottom - (value - low) / (high - low) * (bottom - top)
        draw.line((left, y, right, y), fill="#cc3333")
    x = left + now / end * (right - left)
    draw.line((x, top, x, bottom), fill="#cc3333")


def export_packet_diagnostic_gif(traces: Mapping[int, Mapping[str, str | Path]], selection: str | Path,
                                 output: str | Path, *, speed_limit_m_s: float = 0.030) -> Path:
    """Export a headless GIF from the same validated payload as the HTML viewer."""

    payload = _payload(traces, selection, speed_limit_m_s)
    destination = confined_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panels = [(case, label, trace) for case, labelled in payload["cases"].items() for label, trace in labelled.items()]
    if len(panels) > 8:
        raise ValueError("packet diagnostic GIF supports at most eight case/checkpoint panels")
    selected_case = "39" if "39" in payload["cases"] else next(iter(payload["cases"]))
    selected = payload["cases"][selected_case]["selected"]
    end = max(trace["physical_time_s"][-1] for _, _, trace in panels)
    frames: list[Image.Image] = []
    frame_times = [*np.arange(0.0, end, 0.04), end]
    for now in frame_times:
        image = Image.new("RGB", (1400, 1450), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 10), f"C3a packet replay: t={now:.3f} s; case {selected_case} selected signals", fill="#111111")
        for number, (case, label, trace) in enumerate(panels):
            column, row = number % 4, number // 4
            left, top = column * 350, 35 + row * 245
            draw.rectangle((left, top, left + 349, top + 235), outline="#bbbbbb")
            index = _sample_index(trace["physical_time_s"], float(now))
            q, lengths = trace["q"][index], trace["link_lengths"]
            elbow = (lengths[0] * math.sin(q[0]), -lengths[0] * math.cos(q[0]))
            endpoint = (elbow[0] + lengths[1] * math.sin(q[0] + q[1]), elbow[1] - lengths[1] * math.cos(q[0] + q[1]))
            origin = (left + 175, top + 125)
            pixel = lambda value, origin=origin: (round(origin[0] + value[0] * 170),
                                                   round(origin[1] - value[1] * 170))
            history = [pixel(value) for value in trace["endpoint"][:index + 1]]
            if len(history) > 1:
                draw.line(history, fill="#222222", width=2)
            draw.line([pixel((0.0, 0.0)), pixel(elbow), pixel(endpoint)], fill="#2477c9", width=8, joint="curve")
            tx, ty = pixel(trace["target"]); draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), outline="#cc3333", width=2)
            speed = math.hypot(*trace["endpoint_velocity"][index])
            draw.text((left + 6, top + 6), f"case {case} {label}", fill="#111111")
            draw.text((left + 6, top + 21), f"speed {speed:.3f} m/s", fill="#cc3333" if speed > speed_limit_m_s else "#111111")
        ptime, ctime = selected["physical_time_s"], selected["control_time_s"]
        speed = [math.hypot(*value) for value in selected["endpoint_velocity"]]
        specs = [
            ("endpoint speed", "m/s", [(ptime, speed, "#111111")], [speed_limit_m_s]),
            ("q1 true/measured/estimated", "rad", [(ptime, [v[0] for v in selected["q"]], "#111111"), (ctime, [v[0] for v in selected["measured_q"]] if selected["measured_q"] else None, "#188038"), (ctime, [v[0] for v in selected["estimated_q"]] if selected["estimated_q"] else None, "#2477c9")], None),
            ("q2 true/measured/estimated", "rad", [(ptime, [v[1] for v in selected["q"]], "#111111"), (ctime, [v[1] for v in selected["measured_q"]] if selected["measured_q"] else None, "#188038"), (ctime, [v[1] for v in selected["estimated_q"]] if selected["estimated_q"] else None, "#2477c9")], None),
            ("dq1 true/measured/estimated", "rad/s", [(ptime, [v[0] for v in selected["dq"]], "#111111"), (ctime, [v[0] for v in selected["measured_dq"]] if selected["measured_dq"] else None, "#188038"), (ctime, [v[0] for v in selected["estimated_dq"]] if selected["estimated_dq"] else None, "#2477c9")], None),
            ("dq2 true/measured/estimated", "rad/s", [(ptime, [v[1] for v in selected["dq"]], "#111111"), (ctime, [v[1] for v in selected["measured_dq"]] if selected["measured_dq"] else None, "#188038"), (ctime, [v[1] for v in selected["estimated_dq"]] if selected["estimated_dq"] else None, "#2477c9")], None),
            ("command 1 accepted/teacher", "command", [(ctime[:-1], [v[0] for v in selected["accepted_action"]], "#111111"), (ctime[:-1], [v[0] for v in selected["truth_teacher_action"]] if selected["truth_teacher_action"] else None, "#d77a00")], None),
            ("command 2 accepted/teacher", "command", [(ctime[:-1], [v[1] for v in selected["accepted_action"]], "#111111"), (ctime[:-1], [v[1] for v in selected["truth_teacher_action"]] if selected["truth_teacher_action"] else None, "#d77a00")], None),
        ]
        for number, (title, unit, series, extra) in enumerate(specs):
            column, row = number % 2, number // 2
            _plot(draw, (60 + column * 690, 550 + row * 210, 650 + column * 690, 700 + row * 210), series,
                  end, float(now), title, unit, extra)
        draw.text((20, 1415), "Black=true/accepted; green=measured packet; blue=posterior; orange=same-state teacher. Dashed=deadline/hold.", fill="#111111")
        frames.append(image)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=40, loop=0, disposal=2)
    return destination
