"""R4 standalone learning-comparison export tests."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import json
from pathlib import Path

import numpy as np
from PIL import Image

from robot_ai.visualize.learning import export_learning_gif, export_learning_html
from robot_ai.visualize.packet_diagnostic import (
    _sample_index,
    _trace_payload,
    export_packet_diagnostic_gif,
    export_packet_diagnostic_html,
    export_packet_diagnostic_report,
)


def test_learning_exports_embed_recorded_time_and_provenance(tmp_path: Path) -> None:
    trajectory = tmp_path / "recorded.npz"
    np.savez(
        trajectory,
        q=np.zeros((3, 2)),
        endpoint=np.array([[0.0, -0.55], [0.01, -0.54], [0.02, -0.53]]),
        time_s=np.array([0.0, 0.04, 0.08]),
        target=np.array([0.02, -0.53]),
        link_length_1=0.30,
        link_length_2=0.25,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"recorded checkpoint")
    report = {
        "checkpoint": "initial",
        "checkpoint_source": str(checkpoint),
        "config_source": "configs/healthy.toml",
        "success_fraction": 0.0,
        "deadline_error_median_m": 0.1,
        "artifacts": {"trajectory": str(trajectory)},
        "training_history": [{"loss": 3.0}, {"loss": 1.0}],
    }
    html = export_learning_html([report] * 5, tmp_path / "comparison.html")
    gif = export_learning_gif([report] * 5, tmp_path / "comparison.gif")
    document = html.read_text(encoding="utf-8")
    assert "Deadline: 1.500 s" in document
    assert "checkpoint_sha256" in document
    assert "config_sha256" in document
    assert "physical time" in document
    with Image.open(gif) as image:
        assert image.n_frames >= 2
        assert image.height == 930
        image.seek(0)
        # GIF uses the recorded target's fixed physical scale: 225, 165 is
        # the base and the target circle's top edge is six pixels above it.
        target_x = int(225 + 0.02 * 230)
        target_y = int(165 - (-0.53) * 230) - 6
        assert image.convert("RGB").getpixel((target_x, target_y)) == (204, 51, 51)


def test_learning_gif_appends_exact_hold_end_after_regular_frame_cadence(tmp_path: Path) -> None:
    """The last GIF frame must show the saved 1.70 s hold-end sample."""

    trajectory = tmp_path / "hold-end.npz"
    times = np.arange(0.0, 1.701, 0.01)
    np.savez(
        trajectory, q=np.zeros((len(times), 2)),
        endpoint=np.column_stack((times, np.zeros_like(times))), time_s=times,
        target=np.array([0.0, 0.0]), link_length_1=0.30, link_length_2=0.25,
    )
    report = {
        "checkpoint": "hold-end", "success_fraction": 1.0,
        "deadline_error_median_m": 0.0, "artifacts": {"trajectory": str(trajectory)},
    }
    gif = export_learning_gif([report], tmp_path / "hold-end.gif")
    with Image.open(gif) as image:
        # 0.00 through 1.68 at 40 ms, plus the exact 1.70 s sample.
        assert image.n_frames == 44


def test_packet_diagnostic_embeds_speed_state_command_and_selection(tmp_path: Path) -> None:
    trace = tmp_path / "packet.npz"
    np.savez(trace, q=np.zeros((3, 2)), dq=np.zeros((3, 2)), endpoint=np.zeros((3, 2)),
             endpoint_velocity=np.zeros((3, 2)), accepted_action=np.zeros((2, 2)),
             time_s=np.array([0.0, 0.01, 0.02]), target=np.array([0.0, -0.55]),
             estimated_q=np.zeros((3, 2)), estimated_dq=np.zeros((3, 2)),
             truth_teacher_action=np.zeros((2, 2)), link_length_1=0.3, link_length_2=0.25)
    (tmp_path / "metrics.json").write_text('{"physics_dt":0.01,"control_dt":0.01}')
    selection = tmp_path / "selection.json"
    selection.write_text('{"selection":{"reports":[{"checkpoint":"initial-policy.pt","success_fraction":0.0}]}}')
    html = export_packet_diagnostic_html(
        {39: {"reference": trace, "selected": trace}}, selection, tmp_path / "packet.html"
    )
    gif = export_packet_diagnostic_gif(
        {39: {"reference": trace, "selected": trace}}, selection, tmp_path / "packet.gif"
    )
    report = export_packet_diagnostic_report(
        {39: {"reference": trace, "selected": trace}}, selection, tmp_path / "packet-report.json",
        reproduction_command="robot-ai export-packet-diagnostic", html=html, gif=gif,
    )
    document = html.read_text(encoding="utf-8")
    assert "endpoint speed" in document
    assert "same-state teacher" in document
    assert "64-case packet selection success fraction" in document
    with Image.open(gif) as image:
        assert image.n_frames == 2
    assert json.loads(report.read_text())["outputs"]["gif"]["sha256"]


def test_packet_diagnostic_preserves_mixed_clock_sample_values_and_missing_estimates(tmp_path: Path) -> None:
    """The sample probe checks the data contract before Canvas/GIF rendering."""

    trace = tmp_path / "mixed.npz"
    physical = np.arange(241, dtype=np.float64)
    control = np.arange(25, dtype=np.float64)
    q = np.column_stack((-physical, physical + 0.5))
    dq = np.column_stack((physical / 10.0, -physical / 10.0))
    packets = np.column_stack((-control, control, control * 2.0, -control * 2.0,
                               np.zeros((25, 4))))
    available = np.ones((25, 8), dtype=bool)
    available[23, 0] = False
    np.savez(trace, q=q, dq=dq, endpoint=np.zeros((241, 2)),
             endpoint_velocity=np.column_stack((np.full(241, 0.04), np.zeros(241))),
             accepted_action=np.column_stack((-control[:-1], control[:-1])), time_s=physical * 0.001,
             target=np.array([0.0, -0.55]), packet_values=packets, packet_available=available,
             link_length_1=0.3, link_length_2=0.25)
    (tmp_path / "metrics.json").write_text('{"physics_dt":0.001,"control_dt":0.01}')

    payload = _trace_payload(trace)

    assert payload["physical_time_s"][230] == 0.23
    assert payload["control_time_s"][23] == 0.23
    assert payload["q"][230] == [-230.0, 230.5]
    assert payload["dq"][230] == [23.0, -23.0]
    assert payload["measured_q"][23] == [None, 23.0]
    assert payload["measured_dq"][23] == [46.0, -46.0]
    assert payload["accepted_action"][22] == [-22.0, 22.0]
    assert _sample_index(payload["physical_time_s"], 0.23000000000000018) == 230
    assert _sample_index(payload["physical_time_s"], 0.23999999999999996) == 239


def test_packet_diagnostic_treats_actual_comparator_empty_estimates_as_unavailable() -> None:
    trace = Path("artifacts/p7/c3-current-packet-comparator-30x4x50-selection/initial/demo.npz")

    with np.load(trace, allow_pickle=False) as source:
        assert source["estimated_q"].shape == (0,)
        assert source["estimated_dq"].shape == (0,)
    payload = _trace_payload(trace)

    assert payload["estimated_q"] is None
    assert payload["estimated_dq"] is None
