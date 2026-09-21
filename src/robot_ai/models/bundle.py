"""Versioned world-model bundle serialization and reload checks."""

# Copyright (C) 2026 Florian Loitsch. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..config import confined_path
from .world import WorldModel


class WorldBundle:
    """Frozen model, preprocessing, and schema metadata used by controller training."""

    SCHEMA_VERSION = 1
    SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4)

    def __init__(self, model: WorldModel, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata

    def save(self, output: str | Path) -> Path:
        destination = confined_path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), destination / "weights.pt")
        (destination / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> WorldBundle:
        source = confined_path(path, must_exist=True)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") not in cls.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported world bundle schema")
        model = WorldModel(belief_dim=metadata["model"]["belief_dim"],
                           observation_dim=metadata["model"]["observation_dim"],
                           descriptor_dim=metadata["model"]["descriptor_dim"],
                           endpoint_residual=metadata["model"].get("endpoint_residual", False),
                           physics_prior=metadata["model"].get("physics_prior", False),
                           fresh_measurement_correction=metadata["model"].get(
                               "fresh_measurement_correction", False
                           ))
        normalization = metadata.get("normalization")
        if normalization is not None:
            model.set_observation_normalization(
                torch.as_tensor(normalization["mean"], dtype=torch.float32),
                torch.as_tensor(normalization["std"], dtype=torch.float32),
            )
        state = torch.load(source / "weights.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return cls(model, metadata)
