"""Split-safe recurrent burn-in and loss-window indexing."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import confined_path
from .storage import load_array


@dataclass(frozen=True)
class SequenceWindow:
    episode_index: int
    split: str
    start: int
    burn_in: int
    loss_length: int

    @property
    def end(self) -> int:
        return self.start + self.burn_in + self.loss_length


def load_window(dataset: str | Path, manifest: dict[str, Any], window: SequenceWindow) -> dict[str, np.ndarray]:
    """Load one contiguous burn-in/loss window without crossing its episode."""

    episodes = manifest.get("episodes", [])
    if window.episode_index < 0 or window.episode_index >= len(episodes):
        raise IndexError("sequence window episode index is outside the manifest")
    episode = episodes[window.episode_index]
    if episode["split"] != window.split:
        raise ValueError("sequence window split does not match its episode")
    action_count = int(episode["lengths"]["accepted_action"])
    if window.start < 0 or window.end > action_count:
        raise ValueError("sequence window crosses an episode boundary")
    root = confined_path(dataset)
    observation_start = int(episode["offsets"]["observation_values"]) + window.start
    action_start = int(episode["offsets"]["accepted_action"]) + window.start
    observations_end = observation_start + window.end - window.start + 1
    actions_end = action_start + window.end - window.start
    observation_fields = {"observation_values", "available", "fresh", "age_s", "time_s",
                          "true_q", "true_dq", "true_endpoint", "true_endpoint_velocity"}
    action_fields = {"accepted_action", "terminated", "truncated", "substep_summary"}
    result: dict[str, np.ndarray] = {}
    for field in observation_fields:
        values = load_array(root, manifest, field)
        result[field] = np.asarray(values[observation_start:observations_end])
    for field in action_fields:
        values = load_array(root, manifest, field)
        result[field] = np.asarray(values[action_start:actions_end])
    return result


def windows(manifest: dict[str, Any], *, split: str, burn_in: int, loss_length: int,
            stride: int = 1) -> Iterator[SequenceWindow]:
    if burn_in < 0 or loss_length < 1 or stride < 1:
        raise ValueError("burn_in must be nonnegative and lengths/stride positive")
    for index, episode in enumerate(manifest["episodes"]):
        if episode["split"] != split:
            continue
        action_count = episode["lengths"]["accepted_action"]
        start = 0
        while start + burn_in + loss_length <= action_count:
            yield SequenceWindow(index, split, start, burn_in, loss_length)
            start += stride
