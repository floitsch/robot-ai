"""P4 dataset manifest and recurrent-window tests."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

import numpy as np

from robot_ai.data.collect import collect_dataset, inspect_dataset, reconstruct_episode
from robot_ai.data.sequences import SequenceWindow, load_window, windows
from robot_ai.data.storage import DatasetWriter, load_array, load_manifest


def _episode(length: int) -> dict[str, np.ndarray]:
    return {
        "observation_values": np.zeros((length + 1, 8)), "available": np.ones((length + 1, 8), dtype=bool),
        "fresh": np.ones((length + 1, 8), dtype=bool), "age_s": np.zeros((length + 1, 8)),
        "time_s": np.arange(length + 1) * 0.01, "accepted_action": np.zeros((length, 2)),
        "true_q": np.zeros((length + 1, 2)), "true_dq": np.zeros((length + 1, 2)),
        "true_endpoint": np.zeros((length + 1, 2)), "true_endpoint_velocity": np.zeros((length + 1, 2)),
        "terminated": np.zeros(length, dtype=bool), "truncated": np.zeros(length, dtype=bool),
        "substep_summary": np.zeros((length, 3)),
    }


def test_dataset_manifest_and_windows_stay_inside_episodes(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "data", seed=3)
    writer.add_episode(_episode(10), scenario_id="episode-a", robot_id="r", scenario={}, policy_source="test")
    manifest_path = writer.write()
    manifest = load_manifest(tmp_path / "data")
    assert manifest_path.exists()
    assert load_array(tmp_path / "data", manifest, "accepted_action").shape == (10, 2)
    sequence_windows = list(windows(manifest, split=manifest["episodes"][0]["split"], burn_in=2, loss_length=4))
    assert sequence_windows
    assert all(item.end <= 10 for item in sequence_windows)


def test_window_loader_reconstructs_contiguous_episode_offsets(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "data", seed=3)
    first = _episode(6)
    second = _episode(6)
    first["accepted_action"][:] = 1.0
    second["accepted_action"][:] = 2.0
    writer.add_episode(first, scenario_id="episode-000000", robot_id="r", scenario={}, policy_source="test")
    writer.add_episode(second, scenario_id="episode-000001", robot_id="r", scenario={}, policy_source="test")
    writer.write()
    manifest = load_manifest(tmp_path / "data")
    window = SequenceWindow(1, manifest["episodes"][1]["split"], 1, 2, 2)
    loaded = load_window(tmp_path / "data", manifest, window)
    assert loaded["observation_values"].shape[0] == 5
    np.testing.assert_allclose(loaded["accepted_action"], 2.0)
    assert loaded["accepted_action"].shape[0] == 4


def test_r3a_collection_reports_required_feedback_and_fresh_bias_coverage(tmp_path) -> None:
    dataset = tmp_path / "r3a"
    collect_dataset(dataset, seed=23, episodes=80, actions_per_episode=4, profile="r3a", config_hash="test-hash")
    manifest = load_manifest(dataset)
    report = inspect_dataset(dataset)
    assert manifest["collection"]["profile"] == "r3a"
    assert manifest["config_hash"] == "test-hash"
    assert report["r3a_coverage_complete"]
    for split in ("train", "validation", "test"):
        coverage = report["coverage"][split]
        assert coverage["source_counts"]["public_feedback"] > 0
        assert coverage["source_counts"]["bounded_exploration"] > 0
        assert coverage["fresh_biased_zero_age_qdq_channels"] > 0
    reconstruction = reconstruct_episode(dataset, "episode-000004", tmp_path / "reconstructed")
    assert reconstruction["reproduced"]
