"""Non-object NumPy dataset shards with immutable JSON provenance."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import confined_path

ARRAY_FIELDS = (
    "observation_values", "available", "fresh", "age_s", "time_s", "accepted_action",
    "true_q", "true_dq", "true_endpoint", "true_endpoint_velocity", "terminated", "truncated",
    "substep_summary",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetWriter:
    """Accumulate episodes and commit one self-describing dataset directory."""

    def __init__(self, output: str | Path, *, seed: int, config_hash: str = "",
                 collection: dict[str, Any] | None = None) -> None:
        self.output = confined_path(output)
        self.seed = seed
        self.config_hash = config_hash
        self.collection = collection or {"backend": "native", "device": "cpu"}
        self.episodes: list[dict[str, Any]] = []

    def add_episode(self, arrays: dict[str, np.ndarray], *, scenario_id: str, robot_id: str,
                    scenario: dict[str, Any], policy_source: str) -> None:
        missing = set(ARRAY_FIELDS) - set(arrays)
        if missing:
            raise ValueError(f"episode missing fields: {sorted(missing)}")
        action_count = len(arrays["accepted_action"])
        if len(arrays["observation_values"]) != action_count + 1:
            raise ValueError("an episode must contain N+1 observations for N actions")
        # Hash the complete seeded scenario identity so policy-source schedules
        # do not become correlated with train/validation/test membership.
        split_key = f"{self.seed}:{scenario_id}".encode()
        split_value = int(hashlib.sha256(split_key).hexdigest()[:8], 16) % 100
        split = "train" if split_value < 70 else "validation" if split_value < 85 else "test"
        self.episodes.append({
            "arrays": {name: np.asarray(arrays[name]) for name in ARRAY_FIELDS},
            "scenario_id": scenario_id, "robot_id": robot_id, "scenario": scenario,
            "policy_source": policy_source, "split": split,
        })

    def write(self) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        arrays_dir = self.output / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        concatenated: dict[str, np.ndarray] = {}
        episode_ranges: list[dict[str, Any]] = []
        offsets = {field: 0 for field in ARRAY_FIELDS}
        for episode in self.episodes:
            lengths = {field: len(episode["arrays"][field]) for field in ARRAY_FIELDS}
            episode_ranges.append({key: value for key, value in episode.items() if key != "arrays"} | {"lengths": lengths, "offsets": offsets.copy()})
            for field in ARRAY_FIELDS:
                concatenated[field] = np.concatenate((concatenated[field], episode["arrays"][field])) if field in concatenated else episode["arrays"][field].copy()
                offsets[field] += lengths[field]
        array_manifest: dict[str, Any] = {}
        for field, values in concatenated.items():
            path = arrays_dir / f"{field}.npy"
            np.save(path, values, allow_pickle=False)
            array_manifest[field] = {"path": str(path.relative_to(self.output)), "shape": list(values.shape),
                                     "dtype": str(values.dtype), "sha256": _hash(path)}
        train_obs = [episode["arrays"]["observation_values"] for episode in self.episodes if episode["split"] == "train"]
        train_mask = [episode["arrays"]["available"] for episode in self.episodes if episode["split"] == "train"]
        if train_obs:
            values = np.concatenate(train_obs)
            masks = np.concatenate(train_mask).astype(bool)
            mean = np.zeros(values.shape[1], dtype=np.float64)
            std = np.ones(values.shape[1], dtype=np.float64)
            for channel in range(values.shape[1]):
                selected = values[masks[:, channel], channel]
                if len(selected):
                    mean[channel] = float(selected.mean())
                    std[channel] = max(float(selected.std()), 1e-6)
            normalization = {"source_split": "train", "mean": mean.tolist(), "std": std.tolist()}
        else:
            normalization = {"source_split": "train", "mean": [], "std": []}
        (self.output / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
        manifest = {"schema_version": 1, "seed": self.seed, "config_hash": self.config_hash,
                    "collection": self.collection,
                    "arrays": array_manifest, "episodes": episode_ranges, "normalization": "normalization.json"}
        manifest_path = self.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def load_manifest(dataset: str | Path) -> dict[str, Any]:
    path = confined_path(dataset) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_array(dataset: str | Path, manifest: dict[str, Any], field: str) -> np.ndarray:
    entry = manifest["arrays"][field]
    return np.load(confined_path(dataset) / entry["path"], mmap_mode="r", allow_pickle=False)
